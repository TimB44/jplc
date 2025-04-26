use std::{borrow::Cow, iter::repeat_n};

use crate::{
    asm_codegen::{MAIN_FN_IDX, WORD_SIZE},
    ast::{
        auxiliary::LoopVar,
        expr::{Expr, ExprKind},
    },
    cli::OptLevel,
    typecheck::TypeVal,
};

mod loop_opt;

const DIVIDE_BY_ZERO_ERR_MSG: &str = "divide by zero";
const MOD_BY_ZERO_ERR_MSG: &str = "mod by zero";
const NEGATIVE_ARRAY_INDEX_ERR_MSG: &str = "negative array index";
const INDEX_TOO_LARGE_ERR_MSG: &str = "index too large";
const NEGATIVE_LOOP_BOUND_ERR_MSG: &str = "non-positive loop bound";
const OVERFLOW_ARRAY_ERR_MSG: &str = "overflow computing array size";
pub const VOID_VALUE: u64 = 1;
pub const TRUE_VALUE: u64 = 1;
pub const FALSE_VALUE: u64 = 0;

use super::{Asm, AsmEnv, ConstKind, Instr, MemLoc, Operand, Reg};

impl<'a> AsmEnv<'a> {
    pub fn gen_asm_expr(&mut self, expr: &Expr) {
        if self.gen_asm_expr_opt(expr) {
            return;
        }
        match expr.kind() {
            ExprKind::IntLit(val) => {
                let const_id = self.add_const(&ConstKind::Int(*val));

                self.load_const(const_id);
            }
            ExprKind::FloatLit(val) => {
                let const_id = self.add_const(&ConstKind::Float(val.to_bits()));

                self.load_const(const_id);
            }
            ExprKind::True => {
                let const_id = self.add_const(&ConstKind::Int(TRUE_VALUE));

                self.load_const(const_id);
            }
            ExprKind::False => {
                let const_id = self.add_const(&ConstKind::Int(FALSE_VALUE));

                self.load_const(const_id);
            }
            ExprKind::Void => {
                let const_id = self.add_const(&ConstKind::Int(VOID_VALUE));
                self.load_const(const_id);
            }
            ExprKind::Var => {
                // Check for functions
                if matches!(self.env.get_function(expr.loc()), Ok(_)) {
                    let fn_name = expr.loc().as_str(self.env.src());
                    self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rax), Operand::Label(fn_name))]);
                    return;
                }

                let var_info = self
                    .env
                    .get_variable_info(expr.loc(), self.cur_scope)
                    .expect("variable should be valid after typechecking");
                let type_size = self.env.type_size(var_info.var_type());
                let is_local = self.env.is_local_var(expr.loc(), self.cur_scope);
                // offset are from ebp which is 16 bytes after the start of the stack frame
                let mem_loc =
                    self.var_locs[expr.loc().as_str(self.env.src())] - WORD_SIZE as i64 * 2;

                let var_type = if self.cur_fn == MAIN_FN_IDX || is_local {
                    MemLoc::LocalOffset
                } else {
                    MemLoc::GlobalOffset
                };

                self.add_instrs(
                    [Instr::Sub(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(type_size),
                    )]
                    .into_iter()
                    .chain(
                        (0..(type_size as usize))
                            .step_by(WORD_SIZE as usize)
                            .rev()
                            .flat_map(|offset| {
                                [
                                    Instr::Mov(
                                        Operand::Reg(Reg::R10),
                                        Operand::Mem(var_type(mem_loc, offset as u64)),
                                    ),
                                    Instr::Mov(
                                        Operand::Mem(MemLoc::RegOffset(Reg::Rsp, offset as i64)),
                                        Operand::Reg(Reg::R10),
                                    ),
                                ]
                            }),
                    ),
                );
            }
            ExprKind::Paren(expr) => self.gen_asm_expr(expr.as_ref()),
            ExprKind::ArrayLit(exprs) => {
                let element_size = self.env.type_size(exprs[0].type_data());
                let arr_size = element_size * exprs.len() as u64;
                for expr in exprs.iter().rev() {
                    self.gen_asm_expr(expr);
                }

                self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rdi), Operand::Value(arr_size))]);

                let stack_aligend = self.align_stack(0);
                self.add_instrs([Instr::Call(Operand::Label("jpl_alloc"))]);
                self.remove_stack_alignment(stack_aligend);
                self.copy(arr_size, Reg::Rsp, 0, Reg::Rax, 0);
                self.add_instrs([
                    Instr::Add(Operand::Reg(Reg::Rsp), Operand::Value(arr_size)),
                    Instr::Push(Operand::Reg(Reg::Rax)),
                    Instr::Mov(Operand::Reg(Reg::Rax), Operand::Value(exprs.len() as u64)),
                    Instr::Push(Operand::Reg(Reg::Rax)),
                ]);
            }
            ExprKind::StructInit(_, fields) => {
                for field in fields.iter().rev() {
                    self.gen_asm_expr(field);
                }
            }
            ExprKind::FunctionCall(name, args) => {
                let fn_name = match &self
                    .env
                    .get_variable_info(*name, self.cur_scope)
                    .map(|fn_info| fn_info.var_type())
                {
                    Ok(var_type @ TypeVal::FnPointer(_, _)) => {
                        self.gen_asm_expr(&Expr::new(*name, ExprKind::Var, (*var_type).clone()));
                        None
                    }
                    _ => Some(name.as_str(self.env.src())),
                };

                self.call_fn(fn_name, args, expr.type_data());
            }
            ExprKind::FieldAccess(struct_expr, field_name) => {
                self.gen_asm_expr(struct_expr);
                let field_str = field_name.as_str(self.env.src());
                let struct_type = struct_expr.type_data().as_struct();
                let struct_info = self.env.get_struct_id(struct_type);

                let field_offset: u64 = struct_info
                    .fields()
                    .iter()
                    .take_while(|(name, _)| *name != field_str)
                    .map(|(_, ty)| self.env.type_size(ty))
                    .sum();

                let field_size = self.env.type_size(
                    &struct_info
                        .fields()
                        .iter()
                        .find(|(name, _)| *name == field_str)
                        .expect("field should exist after typechecking")
                        .1,
                );
                let size_diff = (self.env.type_size(struct_expr.type_data()) - field_size);
                self.copy(
                    field_size,
                    Reg::Rsp,
                    field_offset as i64,
                    Reg::Rsp,
                    size_diff as i64,
                );
                self.add_instrs([Instr::Add(
                    Operand::Reg(Reg::Rsp),
                    Operand::Value(size_diff),
                )]);
            }
            ExprKind::ArrayIndex(array_expr, indices) => {
                let rank = indices.len();
                let element_size = self.env.type_size(expr.type_data());
                self.gen_asm_expr(array_expr);
                self.generate_index_bounds(indices, 0);
                self.calculate_array_index(rank as u64, [].iter(), element_size, 0, 0);

                self.add_instrs(
                    repeat_n(
                        Instr::Add(Operand::Reg(Reg::Rsp), Operand::Value(WORD_SIZE)),
                        rank,
                    )
                    .chain([
                        Instr::Add(
                            Operand::Reg(Reg::Rsp),
                            Operand::Value(rank as u64 * WORD_SIZE + WORD_SIZE),
                        ),
                        Instr::Sub(Operand::Reg(Reg::Rsp), Operand::Value(element_size)),
                    ]),
                );

                self.copy(element_size, Reg::Rax, 0, Reg::Rsp, 0);
            }
            ExprKind::If(if_expr) => {
                let [cond, true_branch, flase_branch] = if_expr.as_ref();
                self.gen_asm_expr(cond);
                let false_jump = self.next_jump();
                let end_jump = self.next_jump();
                let output_size = self.env.type_size(expr.type_data());
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Cmp(Operand::Reg(Reg::Rax), Operand::Value(0)),
                    Instr::Je(false_jump),
                ]);
                self.gen_asm_expr(true_branch);
                self.add_asm([Asm::Instr(Instr::Jmp(end_jump)), Asm::JumpLabel(false_jump)]);

                // In this branch we have not pushed the output item on the stack yet so reset the
                // stack to what is was before the true branch
                self.fns[self.cur_fn].cur_stack_size -= output_size;
                self.gen_asm_expr(flase_branch);
                self.add_asm([Asm::JumpLabel(end_jump)]);
            }
            ExprKind::ArrayComp(looping_vars, body, loop_scope)
            | ExprKind::Sum(looping_vars, body, loop_scope) => {
                let is_array = matches!(expr.kind(), ExprKind::ArrayComp(_, _, _));
                let element_size = self.env.type_size(body.type_data());
                self.add_instrs([Instr::Sub(
                    Operand::Reg(Reg::Rsp),
                    Operand::Value(WORD_SIZE),
                )]);
                let loop_rank = looping_vars.len();
                self.check_loop_bounds(looping_vars);
                if is_array {
                    self.alloc_array(looping_vars, element_size, 0);
                } else {
                    self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rax), Operand::Value(0))]);
                }

                self.store_loop_data(WORD_SIZE as i64 * loop_rank as i64);

                self.init_loop_vars(looping_vars);
                let loop_begining = self.gen_loop_body(body, *loop_scope);

                if is_array {
                    self.calculate_array_index(
                        loop_rank as u64,
                        looping_vars.iter().map(|LoopVar(_, expr)| expr),
                        element_size,
                        element_size,
                        0,
                    );
                    self.copy(element_size, Reg::Rsp, 0, Reg::Rax, 0);
                    self.add_instrs([Instr::Add(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(element_size),
                    )]);
                } else {
                    match body.type_data() {
                        TypeVal::Int => {
                            self.add_instrs([
                                Instr::Pop(Reg::Rax),
                                Instr::Add(
                                    Operand::Mem(MemLoc::RegOffset(
                                        Reg::Rsp,
                                        loop_rank as i64 * WORD_SIZE as i64 * 2,
                                    )),
                                    Operand::Reg(Reg::Rax),
                                ),
                            ]);
                        }
                        TypeVal::Float => {
                            self.add_instrs([
                                Instr::Pop(Reg::Xmm0),
                                Instr::Add(
                                    Operand::Reg(Reg::Xmm0),
                                    Operand::Mem(MemLoc::RegOffset(
                                        Reg::Rsp,
                                        loop_rank as i64 * WORD_SIZE as i64 * 2,
                                    )),
                                ),
                                Instr::Mov(
                                    Operand::Mem(MemLoc::RegOffset(
                                        Reg::Rsp,
                                        loop_rank as i64 * WORD_SIZE as i64 * 2,
                                    )),
                                    Operand::Reg(Reg::Xmm0),
                                ),
                            ]);
                        }
                        _ => unreachable!(),
                    }
                }

                self.gen_loop_increment(
                    (0..loop_rank).rev().map(|i| {
                        let index_offset = i as i64 * WORD_SIZE as i64;
                        (
                            index_offset,
                            loop_rank as i64 * WORD_SIZE as i64 + index_offset,
                        )
                    }),
                    loop_begining,
                );

                self.add_instrs([Instr::Add(
                    Operand::Reg(Reg::Rsp),
                    Operand::Value(loop_rank as u64 * WORD_SIZE),
                )]);

                if !is_array {
                    self.add_instrs([Instr::Add(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(loop_rank as u64 * WORD_SIZE),
                    )]);
                }
            }

            ExprKind::Or(args) | ExprKind::And(args) => {
                let is_or = matches!(expr.kind(), ExprKind::Or(_));
                let [lhs, rhs] = args.as_ref();
                self.gen_asm_expr(lhs);
                let end_jmp = self.next_jump();
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Cmp(Operand::Reg(Reg::Rax), Operand::Value(0)),
                    if is_or {
                        Instr::Jne(end_jmp)
                    } else {
                        Instr::Je(end_jmp)
                    },
                ]);
                self.gen_asm_expr(rhs);
                self.add_instrs([Instr::Pop(Reg::Rax)]);
                self.add_asm([Asm::JumpLabel(end_jmp)]);
                self.add_instrs([Instr::Push(Operand::Reg(Reg::Rax))]);
            }

            ExprKind::LessThan(args)
            | ExprKind::GreaterThan(args)
            | ExprKind::LessThanEq(args)
            | ExprKind::GreaterThanEq(args)
            | ExprKind::Eq(args)
            | ExprKind::NotEq(args) => self.comparison_binop(args, expr.kind()),
            ExprKind::Add(args) => self.arithmetic_binop(args, Instr::Add),
            ExprKind::Minus(args) => self.arithmetic_binop(args, Instr::Sub),
            ExprKind::Mulitply(args) => self.arithmetic_binop(args, Instr::Mul),
            ExprKind::Divide(args) => match expr.type_data() {
                TypeVal::Int => {
                    self.gen_div_mod(args, true);
                }
                TypeVal::Float => self.arithmetic_binop(args, Instr::Div),
                _ => unreachable!(),
            },
            ExprKind::Modulo(args) => match expr.type_data() {
                TypeVal::Int => {
                    self.gen_div_mod(args, false);
                }
                TypeVal::Float => {
                    self.call_fn(Some("fmod"), args.as_ref(), &TypeVal::Float);
                }
                _ => unreachable!(),
            },
            ExprKind::Not(expr) => {
                self.gen_asm_expr(expr);
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Xor(Operand::Reg(Reg::Rax), Operand::Value(1)),
                    Instr::Push(Operand::Reg(Reg::Rax)),
                ]);
            }
            ExprKind::Negation(expr) => {
                self.gen_asm_expr(expr);
                match expr.type_data() {
                    TypeVal::Int => {
                        self.add_instrs([
                            Instr::Pop(Reg::Rax),
                            Instr::Neg(Reg::Rax),
                            Instr::Push(Operand::Reg(Reg::Rax)),
                        ]);
                    }
                    TypeVal::Float => {
                        self.add_instrs([
                            Instr::Pop(Reg::Xmm1),
                            Instr::Xor(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm0)),
                            Instr::Sub(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm1)),
                            Instr::Push(Operand::Reg(Reg::Xmm0)),
                        ]);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn gen_asm_expr_opt(&mut self, expr: &Expr) -> bool {
        if self.opt_level == OptLevel::None {
            return false;
        }

        if self.opt_level == OptLevel::O3 {
            if let Some(tensor_info) = self.is_tensor(expr) {
                self.codegen_tensor(tensor_info);
                return true;
            }
        }

        match expr.kind() {
            ExprKind::True => {
                self.add_instrs([Instr::Push(Operand::Value(TRUE_VALUE))]);
            }
            ExprKind::False => {
                self.add_instrs([Instr::Push(Operand::Value(FALSE_VALUE))]);
            }
            ExprKind::Void => {
                self.add_instrs([Instr::Push(Operand::Value(VOID_VALUE))]);
            }
            ExprKind::IntLit(v) if *v <= i32::MAX as u64 => {
                self.add_instrs([Instr::Push(Operand::Value(*v))]);
            }
            ExprKind::Mulitply(ops) => {
                let [lhs, rhs] = ops.as_ref();
                for [lhs, rhs] in [[lhs, rhs], [rhs, lhs]] {
                    match lhs.kind() {
                        ExprKind::IntLit(1) => {
                            self.gen_asm_expr(rhs);
                            return true;
                        }
                        ExprKind::IntLit(v) if v.is_power_of_two() => {
                            self.gen_asm_expr(rhs);
                            self.add_instrs([
                                Instr::Pop(Reg::Rax),
                                Instr::Shl(Operand::Reg(Reg::Rax), v.ilog2() as u8),
                                Instr::Push(Operand::Reg(Reg::Rax)),
                            ]);
                            return true;
                        }
                        _ => (),
                    };
                }
                return false;
            }

            ExprKind::If(args) => {
                let [cond, true_branch, false_branch] = args.as_ref();

                if !(matches!(cond.type_data(), TypeVal::Bool)
                    && matches!(true_branch.kind(), ExprKind::IntLit(1))
                    && matches!(false_branch.kind(), ExprKind::IntLit(0)))
                {
                    return false;
                }
                self.gen_asm_expr(cond);
            }

            ExprKind::ArrayIndex(lhs, rhs)
                if matches!(lhs.kind(), ExprKind::Var)
                    && (self.env.is_local_var(lhs.loc(), self.cur_scope)
                        || self.cur_fn == MAIN_FN_IDX) =>
            {
                let rank = rhs.len();
                let element_size = self.env.type_size(expr.type_data());
                let offset = self.fns[self.cur_fn].cur_stack_size as i64
                    - self.var_locs[lhs.loc().as_str(self.env.src())];
                self.generate_index_bounds(rhs, offset as u64);
                self.calculate_array_index(
                    rank as u64,
                    [].iter(),
                    self.env.type_size(expr.type_data()),
                    0,
                    offset as u64,
                );
                self.add_instrs([
                    Instr::Add(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(WORD_SIZE * rank as u64),
                    ),
                    Instr::Sub(Operand::Reg(Reg::Rsp), Operand::Value(element_size)),
                ]);
                self.copy(element_size, Reg::Rax, 0, Reg::Rsp, 0);
            }
            _ => return false,
        }

        true
    }

    // User for most binops arithmetic binary operators except division and mod
    fn arithmetic_binop(
        &mut self,
        args: &[Expr; 2],
        instr: fn(Operand<'a>, Operand<'a>) -> Instr<'a>,
    ) {
        let [lhs, rhs] = args;
        self.gen_asm_expr(rhs);
        self.gen_asm_expr(lhs);
        match lhs.type_data() {
            TypeVal::Int => {
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Pop(Reg::R10),
                    instr(Operand::Reg(Reg::Rax), Operand::Reg(Reg::R10)),
                    Instr::Push(Operand::Reg(Reg::Rax)),
                ]);
            }
            TypeVal::Float => {
                self.add_instrs([
                    Instr::Pop(Reg::Xmm0),
                    Instr::Pop(Reg::Xmm1),
                    instr(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm1)),
                    Instr::Push(Operand::Reg(Reg::Xmm0)),
                ]);
            }
            _ => unreachable!(),
        }
    }

    fn comparison_binop(&mut self, args: &[Expr; 2], kind: &ExprKind) {
        let [lhs, rhs] = args;
        self.gen_asm_expr(rhs);
        self.gen_asm_expr(lhs);
        match lhs.type_data() {
            TypeVal::Int | TypeVal::Bool => {
                let set_instr = match kind {
                    ExprKind::LessThan(_) => Instr::Setl,
                    ExprKind::GreaterThan(_) => Instr::Setg,
                    ExprKind::LessThanEq(_) => Instr::Setle,
                    ExprKind::GreaterThanEq(_) => Instr::Setge,
                    ExprKind::Eq(_) => Instr::Sete,
                    ExprKind::NotEq(_) => Instr::Setne,
                    _ => unreachable!(),
                };
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Pop(Reg::R10),
                    Instr::Cmp(Operand::Reg(Reg::Rax), Operand::Reg(Reg::R10)),
                    set_instr,
                ]);
            }
            TypeVal::Float => {
                let (set_instr, output_reg) = match kind {
                    ExprKind::LessThan(_) => (Instr::Cmplt(Reg::Xmm0, Reg::Xmm1), Reg::Xmm0),
                    ExprKind::GreaterThan(_) => (Instr::Cmplt(Reg::Xmm1, Reg::Xmm0), Reg::Xmm1),
                    ExprKind::LessThanEq(_) => (Instr::Cmple(Reg::Xmm0, Reg::Xmm1), Reg::Xmm0),
                    ExprKind::GreaterThanEq(_) => (Instr::Cmple(Reg::Xmm1, Reg::Xmm0), Reg::Xmm1),
                    ExprKind::Eq(_) => (Instr::Cmpeq(Reg::Xmm0, Reg::Xmm1), Reg::Xmm0),
                    ExprKind::NotEq(_) => (Instr::Cmpneq(Reg::Xmm0, Reg::Xmm1), Reg::Xmm0),
                    _ => unreachable!(),
                };
                self.add_instrs([
                    Instr::Pop(Reg::Xmm0),
                    Instr::Pop(Reg::Xmm1),
                    set_instr,
                    Instr::Mov(Operand::Reg(Reg::Rax), Operand::Reg(output_reg)),
                ]);
            }
            _ => unreachable!(),
        }

        self.add_instrs([
            Instr::And(Operand::Reg(Reg::Rax), Operand::Value(1)),
            Instr::Push(Operand::Reg(Reg::Rax)),
        ]);
    }

    pub fn gen_div_mod(&mut self, args: &[Expr; 2], divide: bool) {
        let [lhs, rhs] = args;
        self.gen_asm_expr(rhs);
        self.gen_asm_expr(lhs);
        let ok_jump = self.next_jump();
        self.add_instrs([
            Instr::Pop(Reg::Rax),
            Instr::Pop(Reg::R10),
            Instr::Cmp(Operand::Reg(Reg::R10), Operand::Value(0)),
            Instr::Jne(ok_jump),
        ]);

        let err_msg_id = self.add_const(if divide {
            &ConstKind::String(Cow::Borrowed(DIVIDE_BY_ZERO_ERR_MSG))
        } else {
            &ConstKind::String(Cow::Borrowed(MOD_BY_ZERO_ERR_MSG))
        });

        self.fail_assertion(err_msg_id);
        self.add_asm([Asm::JumpLabel(ok_jump)]);

        self.add_instrs([
            Instr::Cqo,
            Instr::Div(Operand::Reg(Reg::Rax), Operand::Reg(Reg::R10)),
        ]);
        if !divide {
            self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rax), Operand::Reg(Reg::Rdx))]);
        }

        self.add_instrs([Instr::Push(Operand::Reg(Reg::Rax))]);
    }

    fn generate_index_bounds(&mut self, indices: &[Expr], gap: u64) {
        let rank = indices.len();
        for index in indices.iter().rev() {
            self.gen_asm_expr(index);
        }
        let negative_err_str = self.add_const(&ConstKind::String(Cow::Borrowed(
            NEGATIVE_ARRAY_INDEX_ERR_MSG,
        )));
        let overflow_err_str =
            self.add_const(&ConstKind::String(Cow::Borrowed(INDEX_TOO_LARGE_ERR_MSG)));
        for (i, _) in indices.iter().enumerate() {
            let index_offset = i as u64 * WORD_SIZE;
            let bound_offset = index_offset + rank as u64 * WORD_SIZE + gap;
            let ok_negative_jmp = self.next_jump();
            let ok_overflow_jmp = self.next_jump();
            self.add_instrs([
                Instr::Mov(
                    Operand::Reg(Reg::Rax),
                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, index_offset as i64)),
                ),
                Instr::Cmp(Operand::Reg(Reg::Rax), Operand::Value(0)),
                Instr::Jge(ok_negative_jmp),
            ]);
            self.fail_assertion(negative_err_str);
            self.add_asm([Asm::JumpLabel(ok_negative_jmp)]);

            self.add_instrs([
                Instr::Cmp(
                    Operand::Reg(Reg::Rax),
                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, bound_offset as i64)),
                ),
                Instr::Jl(ok_overflow_jmp),
            ]);
            self.fail_assertion(overflow_err_str);
            self.add_asm([Asm::JumpLabel(ok_overflow_jmp)]);
        }
    }

    fn check_loop_bounds(&mut self, vars: &[LoopVar]) {
        for LoopVar(_, bound) in vars.into_iter().rev() {
            self.gen_asm_expr(bound);
            let ok_jmp = self.next_jump();
            self.add_instrs([
                Instr::Mov(Operand::Reg(Reg::Rax), Operand::Mem(MemLoc::Reg(Reg::Rsp))),
                Instr::Cmp(Operand::Reg(Reg::Rax), Operand::Value(0)),
                Instr::Jg(ok_jmp),
            ]);
            let err_msg_id = self.add_const(&ConstKind::String(Cow::Borrowed(
                NEGATIVE_LOOP_BOUND_ERR_MSG,
            )));
            self.fail_assertion(err_msg_id);
            self.add_asm([Asm::JumpLabel(ok_jmp)]);
        }
    }

    fn alloc_array(&mut self, looping_vars: &[LoopVar], element_size: u64, offset: i64) {
        self.add_instrs([Instr::Mov(
            Operand::Reg(Reg::Rdi),
            Operand::Value(element_size),
        )]);
        let err_msg = self.add_const(&ConstKind::String(Cow::Borrowed(OVERFLOW_ARRAY_ERR_MSG)));
        for (i, _) in looping_vars.iter().enumerate() {
            let ok_jmp = self.next_jump();
            self.add_instrs([
                Instr::Mul(
                    Operand::Reg(Reg::Rdi),
                    Operand::Mem(MemLoc::RegOffset(
                        Reg::Rsp,
                        i as i64 * WORD_SIZE as i64 + offset,
                    )),
                ),
                Instr::Jno(ok_jmp),
            ]);
            self.fail_assertion(err_msg);
            self.add_asm([Asm::JumpLabel(ok_jmp)])
        }
        let stack_was_aligned = self.align_stack(0);
        self.add_instrs([Instr::Call(Operand::Label("jpl_alloc"))]);
        self.remove_stack_alignment(stack_was_aligned);
    }

    fn store_loop_data(&mut self, offset: i64) {
        self.add_instrs([Instr::Mov(
            Operand::Mem(MemLoc::RegOffset(Reg::Rsp, offset)),
            Operand::Reg(Reg::Rax),
        )]);
    }

    fn init_loop_vars(&mut self, vars: &[LoopVar]) {
        for LoopVar(name_loc, _) in vars.iter().rev() {
            self.add_instrs([
                Instr::Mov(Operand::Reg(Reg::Rax), Operand::Value(0)),
                Instr::Push(Operand::Reg(Reg::Rax)),
            ]);
            let var_name = name_loc.as_str(self.env.src());
            self.var_locs
                .insert(var_name, self.fns[self.cur_fn].cur_stack_size as i64);
        }
    }
    fn gen_loop_body(&mut self, body: &Expr, loop_scope: usize) -> u64 {
        let loop_begining = self.next_jump();
        self.add_asm([Asm::JumpLabel(loop_begining)]);
        let old_scope = self.cur_scope;
        self.cur_scope = loop_scope;
        self.gen_asm_expr(body);
        self.cur_scope = old_scope;
        loop_begining
    }
    fn gen_loop_increment<I: ExactSizeIterator<Item = (i64, i64)>>(
        &mut self,
        bounds_loc: I,
        loop_begining: u64,
    ) {
        let rank = bounds_loc.len();
        for (i, (index_offset, bounds_offset)) in bounds_loc.enumerate() {
            self.add_instrs([
                Instr::Add(
                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, index_offset)),
                    Operand::Value(1),
                ),
                Instr::Mov(
                    Operand::Reg(Reg::Rax),
                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, index_offset)),
                ),
                Instr::Cmp(
                    Operand::Reg(Reg::Rax),
                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, bounds_offset)),
                ),
                Instr::Jl(loop_begining),
            ]);

            if i < rank - 1 {
                self.add_instrs([Instr::Mov(
                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, index_offset)),
                    Operand::Value(0),
                )]);
            }
        }
    }
}
