use std::{borrow::Cow, collections::HashMap, hash::Hash, iter::repeat};

use fragments::{MAIN_EPILOGUE, MAIN_PROLOGE};

use crate::{
    ast::{
        auxiliary::{LValue, Str, Var},
        expr::{Expr, ExprKind},
        Program,
    },
    cli::OptLevel,
    environment::{Environment, GLOBAL_SCOPE_ID},
    typecheck::TypeVal,
};

mod cmd_codegen;
mod expr_codegen;
mod fragments;
mod printer;
mod stmt_codegen;

pub const WORD_SIZE: u64 = 8;

const MAIN_FN_IDX: usize = 0;
const MAIN_FN_NAME: &str = "jpl_main";
pub const STARTING_ALIGNMENT: u64 = WORD_SIZE;
pub const STACK_FRAME_ALIGNMENT: u64 = 16;
// TODO: better names
pub const FN_STARTING_STACK_SIZE: u64 = 2 * WORD_SIZE;
pub const MAIN_FN_STARTING_STACK_SIZE: u64 = 3 * WORD_SIZE;

pub const INT_REGS_FOR_ARGS: [Reg; 6] = [Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9];
pub const FLOAT_REGS_FOR_ARGS: [Reg; 8] = [
    Reg::Xmm0,
    Reg::Xmm1,
    Reg::Xmm2,
    Reg::Xmm3,
    Reg::Xmm4,
    Reg::Xmm5,
    Reg::Xmm6,
    Reg::Xmm7,
];

#[derive(Debug, Clone)]
pub struct AsmEnv<'a> {
    env: &'a Environment<'a>,
    opt_level: OptLevel,
    consts: HashMap<ConstKind<'a>, u64>,
    var_locs: HashMap<&'a str, i64>,
    jmp_ctr: u64,
    fns: Vec<AsmFn<'a>>,
    cur_fn: usize,
    cur_scope: usize,
}

#[derive(Debug, Clone)]
struct AsmFn<'a> {
    text: Vec<Asm<'a>>,
    cur_stack_size: u64,
}

impl<'a> AsmEnv<'a> {
    pub fn new(env: &'a Environment<'a>, typed_ast: Program, opt_level: OptLevel) -> Self {
        let main_fn = AsmFn::new(MAIN_FN_NAME);
        let mut asm_env = Self {
            env,
            opt_level,
            fns: vec![main_fn],
            jmp_ctr: 1,
            consts: HashMap::new(),
            cur_fn: MAIN_FN_IDX,
            cur_scope: GLOBAL_SCOPE_ID,
            var_locs: HashMap::from([("args", 0), ("argnum", 0)]),
        };
        asm_env.add_asm(MAIN_PROLOGE);

        for cmd in typed_ast.commands() {
            asm_env.gen_asm_cmd(cmd);
        }
        assert_eq!(asm_env.cur_fn, MAIN_FN_IDX);
        let local_vars_size = asm_env.fns[MAIN_FN_IDX].cur_stack_size - MAIN_FN_STARTING_STACK_SIZE;
        assert!(local_vars_size % WORD_SIZE == 0);
        if local_vars_size > 0 {
            asm_env.add_instrs([Instr::Add(
                Operand::Reg(Reg::Rsp),
                Operand::Value(local_vars_size),
            )])
        }
        asm_env.add_asm(MAIN_EPILOGUE);

        asm_env.opt_pass();
        asm_env
    }

    /// Returns the id of the generated const this will not generate duplicate items in the data
    /// section for the same const
    fn add_const(&mut self, val: &ConstKind<'a>) -> u64 {
        if let Some(id) = self.consts.get(val) {
            return *id;
        }
        let next_id = self.consts.len() as u64;
        self.consts.insert(val.clone(), next_id);
        next_id
    }

    fn load_const(&mut self, const_id: u64) {
        self.add_instrs([
            Instr::Mov(
                Operand::Reg(Reg::Rax),
                Operand::Mem(MemLoc::Const(const_id)),
            ),
            Instr::Push(Operand::Reg(Reg::Rax)),
        ]);
    }

    fn add_asm(&mut self, instrs: impl IntoIterator<Item = Asm<'a>>) {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.text.extend(instrs.into_iter().inspect(|a| {
            if let Asm::Instr(instr) = a {
                match instr {
                    Instr::Call(name) => {
                        assert!(
                            cur_fn.cur_stack_size % STACK_FRAME_ALIGNMENT == 0,
                            "call to function {} not aligned",
                            name
                        );
                    }
                    Instr::Push(_) => cur_fn.cur_stack_size += 8,
                    Instr::Pop(_) => cur_fn.cur_stack_size -= 8,
                    Instr::Add(Operand::Reg(Reg::Rsp), rhs) => {
                        if let Operand::Value(inc) = rhs {
                            assert_eq!(inc % WORD_SIZE, 0);
                            cur_fn.cur_stack_size -= inc;
                        } else {
                            unreachable!("Should only add constant values to the stack")
                        }
                    }
                    Instr::Sub(Operand::Reg(Reg::Rsp), rhs) => {
                        if let Operand::Value(inc) = rhs {
                            assert_eq!(inc % WORD_SIZE, 0);
                            cur_fn.cur_stack_size += inc;
                        } else {
                            unreachable!("Should only add constant values to the stack")
                        }
                    }
                    Instr::Ret => {
                        assert!(cur_fn.cur_stack_size == STARTING_ALIGNMENT);
                        cur_fn.cur_stack_size -= 8;
                    }
                    _ => (),
                }
            }
        }));
    }

    fn add_instrs(&mut self, instrs: impl IntoIterator<Item = Instr<'a>>) {
        self.add_asm(instrs.into_iter().map(Asm::Instr))
    }

    fn align_stack(&mut self, left_to_add: u64) -> bool {
        assert_eq!(left_to_add % WORD_SIZE, 0);
        let cur_alignment = &self.fns[self.cur_fn].aligned();
        let adjustment_aligned = left_to_add % STACK_FRAME_ALIGNMENT == 0;

        if cur_alignment ^ adjustment_aligned {
            self.add_instrs([Instr::Sub(
                Operand::Reg(Reg::Rsp),
                Operand::Value(WORD_SIZE),
            )]);
            true
        } else {
            false
        }
    }

    fn remove_stack_alignment(&mut self, stack_was_aligned: bool) {
        if stack_was_aligned {
            self.add_instrs([Instr::Add(
                Operand::Reg(Reg::Rsp),
                Operand::Value(WORD_SIZE),
            )]);
        }
    }

    fn next_jump(&mut self) -> u64 {
        let jmp = self.jmp_ctr;
        self.jmp_ctr += 1;
        jmp
    }

    fn call_fn(&mut self, name: Option<&'a str>, args: &[Expr], ret_type: &TypeVal) {
        if let None = name {
            self.add_instrs([Instr::Pop(Reg::Rax)])
        }

        let avaliable_int_args = INT_REGS_FOR_ARGS.len() - {
            match ret_type {
                TypeVal::Int
                | TypeVal::Bool
                | TypeVal::Float
                | TypeVal::Void
                | TypeVal::FnPointer(_, _) => 0,
                TypeVal::Array(_, _) | TypeVal::Struct(_) => {
                    self.add_instrs([Instr::Sub(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(self.env.type_size(ret_type)),
                    )]);
                    1
                }
            }
        };

        if let None = name {
            self.add_instrs([Instr::Push(Operand::Reg(Reg::Rax))]);
        }

        let stack_when_fn_pointer_pushed = self.cur_stack_size();
        let num_int_args = args
            .iter()
            .map(|e| e.type_data())
            .filter(|t| {
                matches!(
                    t,
                    TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _)
                )
            })
            .count();

        let num_float_args = args
            .iter()
            .map(|e| e.type_data())
            .filter(|t| matches!(t, TypeVal::Float))
            .count();

        let mut cur_int_arg = num_int_args;
        let mut cur_float_arg = num_float_args;
        // Rework
        let stack_space_for_args: u64 = args
            .iter()
            .rev()
            .filter(|e| match e.type_data() {
                TypeVal::Array(_, _) | TypeVal::Struct(_) => true,
                TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _) => {
                    cur_int_arg -= 1;
                    cur_int_arg >= avaliable_int_args
                }
                TypeVal::Float => {
                    cur_float_arg -= 1;
                    cur_int_arg >= FLOAT_REGS_FOR_ARGS.len()
                }
            })
            .map(|e| self.env.type_size(e.type_data()))
            .sum();

        let stack_aligned = self.align_stack(stack_space_for_args);

        let mut cur_int_arg = num_int_args;
        let mut cur_float_arg = num_float_args;
        for stack_args in args.iter().rev().filter(|e| match e.type_data() {
            TypeVal::Array(_, _) | TypeVal::Struct(_) => true,
            TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _) => {
                cur_int_arg -= 1;
                cur_int_arg >= avaliable_int_args
            }
            TypeVal::Float => {
                cur_float_arg -= 1;
                cur_int_arg >= FLOAT_REGS_FOR_ARGS.len()
            }
        }) {
            self.gen_asm_expr(stack_args);
        }

        let mut cur_int_arg = num_int_args;
        let mut cur_float_arg = num_float_args;
        for stack_args in args.iter().rev().filter(|e| match e.type_data() {
            TypeVal::Array(_, _) | TypeVal::Struct(_) => false,
            TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _) => {
                cur_int_arg -= 1;
                cur_int_arg < avaliable_int_args
            }
            TypeVal::Float => {
                cur_float_arg -= 1;
                cur_int_arg < FLOAT_REGS_FOR_ARGS.len()
            }
        }) {
            self.gen_asm_expr(stack_args);
        }

        let mut cur_int_arg = INT_REGS_FOR_ARGS.len() - avaliable_int_args;
        let mut cur_float_arg = 0;
        for arg in args {
            match arg.type_data() {
                TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _)
                    if cur_int_arg < avaliable_int_args =>
                {
                    self.add_instrs([Instr::Pop(INT_REGS_FOR_ARGS[cur_int_arg])]);
                    cur_int_arg += 1;
                }
                TypeVal::Float if cur_float_arg < FLOAT_REGS_FOR_ARGS.len() => {
                    self.add_instrs([Instr::Pop(FLOAT_REGS_FOR_ARGS[cur_float_arg])]);
                    cur_float_arg += 1;
                }
                _ => (),
            }
        }

        // Load value for retval if needed
        if matches!(ret_type, TypeVal::Array(_, _) | TypeVal::Struct(_)) {
            let offset = stack_space_for_args
                + if stack_aligned { WORD_SIZE } else { 0 }
                + if let None = name { WORD_SIZE } else { 0 };
            self.add_instrs([Instr::Lea(
                INT_REGS_FOR_ARGS[0],
                MemLoc::RegOffset(Reg::Rsp, offset as i64),
            )]);
        }

        if let None = name {
            let cur_stack_size = self.cur_stack_size();
            self.add_instrs([Instr::Mov(
                Operand::Reg(Reg::Rax),
                Operand::Mem(MemLoc::RegOffset(
                    Reg::Rsp,
                    (cur_stack_size - stack_when_fn_pointer_pushed) as i64,
                )),
            )]);
        }

        self.add_instrs([Instr::Call(
            name.map(|s| Operand::Label(s))
                .unwrap_or(Operand::Reg(Reg::Rax)),
        )]);
        if stack_space_for_args > 0 {
            let mut cur_int_arg = 0;
            let mut cur_float_arg = 0;
            self.add_instrs(
                args.iter()
                    .filter(|e| match e.type_data() {
                        TypeVal::Array(_, _) | TypeVal::Struct(_) => true,
                        TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _) => {
                            cur_int_arg += 1;
                            cur_int_arg >= avaliable_int_args
                        }
                        TypeVal::Float => {
                            cur_float_arg += 1;
                            cur_int_arg >= FLOAT_REGS_FOR_ARGS.len()
                        }
                    })
                    .map(|e| {
                        Instr::Add(
                            Operand::Reg(Reg::Rsp),
                            Operand::Value(self.env.type_size(e.type_data())),
                        )
                    }),
            );
        }

        self.remove_stack_alignment(stack_aligned);

        if let None = name {
            self.add_asm([
                Asm::Comment("Removing function pointer after call"),
                Asm::Instr(Instr::Add(
                    Operand::Reg(Reg::Rsp),
                    Operand::Value(WORD_SIZE),
                )),
            ]);
        }
        match ret_type {
            TypeVal::Int | TypeVal::Void | TypeVal::Bool | TypeVal::FnPointer(_, _) => {
                self.add_instrs([Instr::Push(Operand::Reg(Reg::Rax))])
            }
            TypeVal::Float => self.add_instrs([Instr::Push(Operand::Reg(Reg::Xmm0))]),
            // Return value already placed on top of the stack
            TypeVal::Array(_, _) | TypeVal::Struct(_) => (),
        }
    }

    /// Copies size bytes from the top of the stack to the pointer located in the rax register
    fn copy(&mut self, size: u64, from: Reg, from_offset: i64, to: Reg, to_offset: i64) {
        assert!(size % WORD_SIZE == 0);
        self.add_instrs(
            //	mov r10, [rsp + 8]
            //	mov [rax + 8], r10
            (0..size as usize)
                .step_by(WORD_SIZE as usize)
                .rev()
                .flat_map(|offset| {
                    [
                        Instr::Mov(
                            Operand::Reg(Reg::R10),
                            Operand::Mem(MemLoc::RegOffset(from, from_offset + offset as i64)),
                        ),
                        Instr::Mov(
                            Operand::Mem(MemLoc::RegOffset(to, to_offset + offset as i64)),
                            Operand::Reg(Reg::R10),
                        ),
                    ]
                }),
        );
    }
    fn fail_assertion(&mut self, msg: u64) {
        let stack_was_aligned = self.align_stack(0);
        self.add_instrs([
            Instr::Lea(Reg::Rdi, MemLoc::Const(msg)),
            Instr::Call(Operand::Label("fail_assertion")),
        ]);
        self.remove_stack_alignment(stack_was_aligned);
    }

    fn calculate_array_index<'b>(
        &mut self,
        rank: u64,
        loop_vars: impl Iterator<Item = &'b Expr>,
        element_size: u64,
        staring_offset: u64,
        gap: u64,
    ) {
        if self.opt_level == OptLevel::None {
            self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rax), Operand::Value(0))]);
            for i in 0..rank {
                let index_offset = i * WORD_SIZE + staring_offset;
                let bound_offset = index_offset + (rank * WORD_SIZE) + gap;
                self.add_instrs([
                    Instr::Mul(
                        Operand::Reg(Reg::Rax),
                        Operand::Mem(MemLoc::RegOffset(Reg::Rsp, bound_offset as i64)),
                    ),
                    Instr::Add(
                        Operand::Reg(Reg::Rax),
                        Operand::Mem(MemLoc::RegOffset(Reg::Rsp, index_offset as i64)),
                    ),
                ]);
            }

            self.add_instrs([
                Instr::Mul(Operand::Reg(Reg::Rax), Operand::Value(element_size)),
                Instr::Add(
                    Operand::Reg(Reg::Rax),
                    Operand::Mem(MemLoc::RegOffset(
                        Reg::Rsp,
                        rank as i64 * WORD_SIZE as i64 * 2 + staring_offset as i64 + gap as i64,
                    )),
                ),
            ]);
        } else {
            self.add_instrs([Instr::Mov(
                Operand::Reg(Reg::Rax),
                Operand::Mem(MemLoc::RegOffset(Reg::Rsp, staring_offset as i64)),
            )]);

            for (i, bound_kind) in (1..rank).zip(
                loop_vars
                    .into_iter()
                    .map(|e| Some(e.kind()))
                    .skip(1)
                    .chain(repeat(None)),
            ) {
                let index_offset = i * WORD_SIZE + staring_offset;
                let bound_offset = index_offset + (rank * WORD_SIZE) + gap;
                let loop_bound = match bound_kind {
                    Some(ExprKind::IntLit(v)) if *v <= i32::MAX as u64 || v.is_power_of_two() => {
                        Operand::Value(*v)
                    }
                    _ => Operand::Mem(MemLoc::RegOffset(Reg::Rsp, bound_offset as i64)),
                };
                self.add_instrs([
                    Instr::Mul(Operand::Reg(Reg::Rax), loop_bound),
                    Instr::Add(
                        Operand::Reg(Reg::Rax),
                        Operand::Mem(MemLoc::RegOffset(Reg::Rsp, index_offset as i64)),
                    ),
                ]);
            }

            self.add_instrs([
                Instr::Mul(Operand::Reg(Reg::Rax), Operand::Value(element_size)),
                Instr::Add(
                    Operand::Reg(Reg::Rax),
                    Operand::Mem(MemLoc::RegOffset(
                        Reg::Rsp,
                        rank as i64 * WORD_SIZE as i64 * 2 + staring_offset as i64 + gap as i64,
                    )),
                ),
            ]);
        }
    }

    fn add_lvalue(&mut self, lvalue: &LValue, loc: i64) {
        let var_name = lvalue.variable().loc().as_str(self.env.src());
        self.var_locs.insert(var_name, loc);
        let mut lvalue_stack_loc = loc;
        for Var(lvalue_loc) in lvalue.array_bindings().into_iter().flatten() {
            let lvalue_name = lvalue_loc.as_str(self.env.src());
            self.var_locs.insert(lvalue_name, lvalue_stack_loc);
            lvalue_stack_loc -= WORD_SIZE as i64;
        }
    }

    fn opt_pass(&mut self) {
        if self.opt_level == OptLevel::None {
            return;
        }

        for asm_fn in &mut self.fns {
            let mut old_text = Vec::with_capacity(asm_fn.text.len());
            std::mem::swap(&mut old_text, &mut asm_fn.text);
            for asm in old_text {
                let instr = match asm {
                    Asm::Instr(instr) => instr,
                    asm => {
                        asm_fn.text.push(asm);
                        continue;
                    }
                };

                match instr {
                    Instr::Mul(lhs, Operand::Value(val)) if val.is_power_of_two() => {
                        asm_fn
                            .text
                            .push(Asm::Instr(Instr::Shl(lhs, val.ilog2() as u8)));
                    }
                    asm => {
                        asm_fn.text.push(Asm::Instr(asm));
                        continue;
                    }
                }
            }
        }
    }

    fn codegen_assert(&mut self, cond: &Expr, msg: &Str) {
        self.gen_asm_expr(cond);
        let ok_jmp = self.next_jump();
        let msg = msg.inner_loc().as_str(self.env.src());
        let msg_id = self.add_const(&ConstKind::String(Cow::Borrowed(msg)));

        self.add_instrs([
            Instr::Pop(Reg::Rax),
            Instr::Cmp(Operand::Reg(Reg::Rax), Operand::Value(0)),
            Instr::Jne(ok_jmp),
        ]);
        self.fail_assertion(msg_id);
        self.add_asm([Asm::JumpLabel(ok_jmp)]);
    }

    fn cur_stack_size(&self) -> u64 {
        self.fns[self.cur_fn].cur_stack_size
    }
}

impl<'a> AsmFn<'a> {
    fn new(name: &'a str) -> Self {
        Self {
            text: vec![Asm::FnLabel(name)],
            cur_stack_size: STARTING_ALIGNMENT,
        }
    }

    fn aligned(&self) -> bool {
        self.cur_stack_size % STACK_FRAME_ALIGNMENT == 0
    }
}

#[derive(Clone, Debug)]
enum Asm<'a> {
    Global(&'a str),
    Extern(&'a str),
    Section(Section),
    Const(u64, ConstKind<'a>),
    Instr(Instr<'a>),
    FnLabel(&'a str),
    JumpLabel(u64),
    Comment(&'static str),
}

#[derive(Clone, Debug)]
enum Instr<'a> {
    Mov(Operand<'a>, Operand<'a>),
    Lea(Reg, MemLoc<'a>),
    // does not have the leading _
    Call(Operand<'a>),
    Push(Operand<'a>),
    Pop(Reg),
    Add(Operand<'a>, Operand<'a>),
    Sub(Operand<'a>, Operand<'a>),
    Ret,
    Neg(Reg),
    Xor(Operand<'a>, Operand<'a>),
    And(Operand<'a>, Operand<'a>),
    Mul(Operand<'a>, Operand<'a>),
    Div(Operand<'a>, Operand<'a>),
    // Always set register al
    Setl,
    Setg,
    Setle,
    Setge,
    Sete,
    Setne,

    // Only for fp regs
    Cmplt(Reg, Reg),
    Cmple(Reg, Reg),
    Cmpeq(Reg, Reg),
    Cmpneq(Reg, Reg),

    Cmp(Operand<'a>, Operand<'a>),
    Jmp(u64),
    Jne(u64),
    Je(u64),
    Jl(u64),
    Jge(u64),
    Jno(u64),
    Jg(u64),
    Shl(Operand<'a>, u8),
    Cqo,
}

#[derive(Clone, Debug)]
enum Operand<'a> {
    Reg(Reg),
    Mem(MemLoc<'a>),
    Value(u64),
    Label(&'a str),
}

impl Operand<'_> {
    fn kind(&self) -> Option<RegKind> {
        match self {
            Operand::Reg(reg) => Some(reg.kind()),
            Operand::Mem(_) => None,
            Operand::Label(_) | Operand::Value(_) => Some(RegKind::Int),
        }
    }

    fn args_kind(&self, rhs: &Self) -> RegKind {
        match (self, rhs) {
            (Operand::Reg(lhs), Operand::Reg(rhs)) => {
                let kind = lhs.kind();
                assert_eq!(kind, rhs.kind());
                kind
            }
            (Operand::Reg(reg), _) => reg.kind(),
            (_, Operand::Reg(reg)) => reg.kind(),

            (Operand::Mem(_), Operand::Value(_)) => RegKind::Int,

            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Reg {
    Rax,
    Rbp,
    Rsp,
    R12,
    Rdi,
    Rdx,
    Rsi,
    Rcx,
    R8,
    R9,
    R10,
    Xmm0,
    Xmm1,
    Xmm2,
    Xmm3,
    Xmm4,
    Xmm5,
    Xmm6,
    Xmm7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegKind {
    Int,
    Float,
}

impl Reg {
    fn kind(&self) -> RegKind {
        match self {
            Reg::Rax
            | Reg::Rbp
            | Reg::Rsp
            | Reg::R12
            | Reg::Rdi
            | Reg::Rdx
            | Reg::Rsi
            | Reg::Rcx
            | Reg::R8
            | Reg::R9
            | Reg::R10 => RegKind::Int,

            Reg::Xmm0
            | Reg::Xmm1
            | Reg::Xmm2
            | Reg::Xmm3
            | Reg::Xmm4
            | Reg::Xmm5
            | Reg::Xmm6
            | Reg::Xmm7 => RegKind::Float,
        }
    }
}

#[derive(Clone, Debug)]
enum MemLoc<'a> {
    GlobalOffset(i64, u64),
    LocalOffset(i64, u64),
    Const(u64),
    Reg(Reg),
    RegOffset(Reg, i64),
    FnLabel(&'a str),
}

#[derive(Clone, Debug)]
enum Section {
    Data,
    Text,
}

#[derive(Clone, Debug, PartialEq, Hash, Eq)]
enum ConstKind<'a> {
    Int(u64),
    // Stores the bits of the float
    Float(u64),

    // Stores owned strings for type strings, and refs for string literals
    String(Cow<'a, str>),
}
