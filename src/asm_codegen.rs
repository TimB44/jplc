use std::{borrow::Cow, collections::HashMap, hash::Hash, u64};

use fragments::{MAIN_EPILOGUE, MAIN_PROLOGE};

use crate::{
    ast::{expr::Expr, Program},
    environment::Environment,
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
const STARTING_ALIGNMENT: u8 = 8;
const STACK_FRAME_ALIGNMENT: u8 = 16;
pub const INT_REGS_FOR_ARGS: [Reg; 6] = [Reg::Rdi, Reg::Rdx, Reg::Rsi, Reg::Rcx, Reg::R8, Reg::R9];
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

pub struct AsmEnv<'a, 'b> {
    env: &'b Environment<'a>,
    consts: HashMap<ConstKind<'a>, u64>,
    jmp_ctr: u64,
    fns: Vec<AsmFn<'a>>,
    cur_fn: usize,
}

struct AsmFn<'a> {
    text: Vec<Asm<'a>>,
    cur_stack_size: u64,
}

impl<'a, 'b> AsmEnv<'a, 'b> {
    pub fn new(env: &'b Environment<'a>, typed_ast: Program) -> Self {
        let mut main_fn = AsmFn::new(MAIN_FN_NAME);
        main_fn.text.extend_from_slice(MAIN_PROLOGE);
        let mut asm_env = Self {
            env,
            fns: vec![main_fn],
            jmp_ctr: 1,
            consts: HashMap::new(),
            cur_fn: MAIN_FN_IDX,
        };

        for cmd in typed_ast.commands() {
            asm_env.gen_asm_cmd(cmd);
        }

        assert_eq!(asm_env.cur_fn, MAIN_FN_IDX);
        asm_env.fns[MAIN_FN_IDX]
            .text
            .extend_from_slice(MAIN_EPILOGUE);

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
        return next_id;
    }

    fn add_asm(&mut self, instrs: impl IntoIterator<Item = Asm<'a>>) {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.text.extend(instrs.into_iter().inspect(|a| {
            if let Asm::Instr(instr) = a {
                match instr {
                    Instr::Call(_) => {
                        assert!(cur_fn.cur_stack_size % STACK_FRAME_ALIGNMENT as u64 == 0);
                    }
                    Instr::Push(_) => cur_fn.cur_stack_size += 8,
                    Instr::Pop(_) => cur_fn.cur_stack_size += 8,
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
                            cur_fn.cur_stack_size -= inc;
                        } else {
                            unreachable!("Should only add constant values to the stack")
                        }
                    }
                    Instr::Ret => {
                        assert!(cur_fn.cur_stack_size == 8);
                        cur_fn.cur_stack_size -= 8;
                    }
                    _ => (),
                }
            }
        }));
    }

    fn add_instrs(&mut self, instrs: impl IntoIterator<Item = Instr<'a>>) {
        self.add_asm(instrs.into_iter().map(|i| Asm::Instr(i)))
    }

    // TODO memosize result

    fn align_stack(&mut self, left_to_add: u64) -> bool {
        assert_eq!(left_to_add % WORD_SIZE, 0);
        let left_to_add = (left_to_add % STACK_FRAME_ALIGNMENT as u64) as u8;
        let cur_alignment = self.fns[self.cur_fn].aligned();

        let adjustment_aligned = left_to_add % STACK_FRAME_ALIGNMENT == 0;

        if cur_alignment ^ adjustment_aligned {
            self.add_instrs([Instr::Sub(
                Operand::Reg(Reg::Rsp),
                Operand::Value(WORD_SIZE as u64),
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
                Operand::Value(WORD_SIZE as u64),
            )]);
        }
    }

    fn next_jump(&mut self) -> u64 {
        let jmp = self.jmp_ctr;
        self.jmp_ctr += 1;
        jmp
    }

    fn call_fn(&mut self, name: &'a str, args: &[Expr], ret_type: &TypeVal) {
        // TODO: allocate space for struct retval here
        let num_int_args = args
            .iter()
            .map(|e| e.type_data())
            // TODO maybe add void to the matches
            .filter(|t| matches!(t, TypeVal::Int | TypeVal::Bool))
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
                TypeVal::Int | TypeVal::Bool => {
                    cur_int_arg -= 1;
                    cur_int_arg >= INT_REGS_FOR_ARGS.len()
                }
                TypeVal::Float => {
                    cur_float_arg -= 1;
                    cur_int_arg >= FLOAT_REGS_FOR_ARGS.len()
                }
                TypeVal::Void => todo!(),
            })
            .map(|e| self.env.type_size(e.type_data()))
            .sum();

        let stack_aligned = self.align_stack(stack_space_for_args);

        let mut cur_int_arg = num_int_args;
        let mut cur_float_arg = num_float_args;
        for stack_args in args.iter().rev().filter(|e| match e.type_data() {
            TypeVal::Array(_, _) | TypeVal::Struct(_) => true,
            TypeVal::Int | TypeVal::Bool => {
                cur_int_arg -= 1;
                cur_int_arg >= INT_REGS_FOR_ARGS.len()
            }
            TypeVal::Float => {
                cur_float_arg -= 1;
                cur_int_arg >= FLOAT_REGS_FOR_ARGS.len()
            }
            TypeVal::Void => todo!(),
        }) {
            self.gen_asm_expr(stack_args);
        }

        let mut cur_int_arg = num_int_args;
        let mut cur_float_arg = num_float_args;
        for stack_args in args.iter().rev().filter(|e| match e.type_data() {
            TypeVal::Array(_, _) | TypeVal::Struct(_) => false,
            TypeVal::Int | TypeVal::Bool => {
                cur_int_arg -= 1;
                cur_int_arg < INT_REGS_FOR_ARGS.len()
            }
            TypeVal::Float => {
                cur_float_arg -= 1;
                cur_int_arg < FLOAT_REGS_FOR_ARGS.len()
            }
            TypeVal::Void => todo!(),
        }) {
            self.gen_asm_expr(stack_args);
        }

        let mut cur_int_arg = 0;
        let mut cur_float_arg = 0;
        for arg in args {
            match arg.type_data() {
                TypeVal::Int | TypeVal::Bool if cur_int_arg < INT_REGS_FOR_ARGS.len() => {
                    self.add_instrs([Instr::Pop(INT_REGS_FOR_ARGS[cur_int_arg])]);
                    cur_int_arg += 1;
                }
                TypeVal::Float if cur_float_arg < FLOAT_REGS_FOR_ARGS.len() => {
                    self.add_instrs([Instr::Pop(FLOAT_REGS_FOR_ARGS[cur_float_arg])]);
                    cur_float_arg += 1;
                }
                TypeVal::Void => todo!(),
                _ => (),
            }
        }

        self.add_instrs([Instr::Call(name)]);
        self.remove_stack_alignment(stack_aligned);
        self.add_instrs(match ret_type {
            TypeVal::Int => [Instr::Push(Reg::Rax)],
            TypeVal::Bool => [Instr::Push(Reg::Rax)],
            TypeVal::Float => [Instr::Push(Reg::Xmm0)],
            TypeVal::Array(typed, _) => todo!(),
            TypeVal::Struct(_) => todo!(),
            TypeVal::Void => todo!(),
        });
    }
}

impl<'a> AsmFn<'a> {
    fn new(name: &'a str) -> Self {
        Self {
            text: vec![Asm::FnLabel(name)],
            cur_stack_size: STARTING_ALIGNMENT as u64,
        }
    }

    fn aligned(&self) -> bool {
        self.cur_stack_size % STACK_FRAME_ALIGNMENT as u64 == 0
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
}

#[derive(Clone, Debug)]
enum Instr<'a> {
    Mov(Operand, Operand),
    Lea(Reg, MemLoc),
    // does not have the leading _
    Call(&'a str),
    Push(Reg),
    Pop(Reg),
    Add(Operand, Operand),
    Sub(Operand, Operand),
    Ret,
    Neg(Reg),
    Xor(Operand, Operand),
    And(Operand, Operand),
    Mul(Operand, Operand),
    Div(Operand, Operand),
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

    Cmp(Operand, Operand),
    Jne(u64),
    Cqo,
}

#[derive(Clone, Debug)]
enum Operand {
    Reg(Reg),
    Mem(MemLoc),
    Value(u64),
}

impl Operand {
    fn kind(&self) -> Option<RegKind> {
        match self {
            Operand::Reg(reg) => Some(reg.kind()),
            Operand::Mem(_) => None,
            Operand::Value(_) => Some(RegKind::Int),
        }
    }

    fn args_kind(&self, rhs: &Self) -> RegKind {
        match (self, rhs) {
            (Operand::Mem(_), Operand::Mem(_))
            | (Operand::Value(_), Operand::Reg(_))
            | (Operand::Value(_), Operand::Mem(_))
            | (Operand::Value(_), Operand::Value(_)) => unreachable!(),

            (Operand::Reg(lhs), Operand::Reg(rhs)) => {
                let kind = lhs.kind();
                assert_eq!(kind, rhs.kind());
                kind
            }
            (Operand::Reg(reg), _) => reg.kind(),
            (_, Operand::Reg(reg)) => reg.kind(),

            (Operand::Mem(_), Operand::Value(_)) => RegKind::Int,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Reg {
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
enum MemLoc {
    GlobalOffset(i64),
    LocalOffset(i64, u64),
    Const(u64),
    Reg(Reg),
    RegOffset(Reg, i64),
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
