use std::borrow::Cow;

use crate::{
    ast::expr::{Expr, ExprKind},
    typecheck::Typed,
};

const DIVIDE_BY_ZERO_ERR_MSG: &str = "divide by zero";
const MOD_BY_ZERO_ERR_MSG: &str = "mod by zero";

use super::{fragments::load_const, Asm, AsmEnv, ConstKind, Instr, MemLoc, Operand, Reg};

impl<'a, 'b> AsmEnv<'a, 'b> {
    pub fn gen_asm_expr(&mut self, expr: &Expr<Typed>) {
        match expr.kind() {
            ExprKind::IntLit(val) => {
                let const_id = self.add_const(&ConstKind::Int(*val));

                self.add_instrs(load_const(const_id));
            }
            ExprKind::FloatLit(val) => {
                let const_id = self.add_const(&ConstKind::Float(val.to_bits()));

                self.add_instrs(load_const(const_id));
            }
            ExprKind::True => {
                let const_id = self.add_const(&ConstKind::Int(1));

                self.add_instrs(load_const(const_id));
            }
            ExprKind::False => {
                let const_id = self.add_const(&ConstKind::Int(0));

                self.add_instrs(load_const(const_id));
            }
            ExprKind::Var => todo!(),
            ExprKind::Void => todo!(),
            ExprKind::Paren(expr) => self.gen_asm_expr(expr.as_ref()),
            ExprKind::ArrayLit(exprs) => {
                let element_size = self.type_size(exprs[0].type_data());
                let arr_size = element_size * exprs.len() as u64;
                for expr in exprs.iter().rev() {
                    self.gen_asm_expr(expr);
                }

                self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rdi), Operand::Value(arr_size))]);

                let stack_aligend = dbg!(self.align_stack(0));
                self.add_instrs([Instr::Call("jpl_alloc")]);
                self.remove_stack_alignment(stack_aligend);

                self.add_instrs(
                    (0..(arr_size / 8))
                        .rev()
                        .flat_map(|i| {
                            let offset = i * 8;
                            [
                                Instr::Mov(
                                    Operand::Reg(Reg::R10),
                                    Operand::Mem(MemLoc::RegOffset(Reg::Rsp, offset as i64)),
                                ),
                                Instr::Mov(
                                    Operand::Mem(MemLoc::RegOffset(Reg::Rax, offset as i64)),
                                    Operand::Reg(Reg::R10),
                                ),
                            ]
                        })
                        .chain([
                            Instr::Add(Operand::Reg(Reg::Rsp), Operand::Value(arr_size)),
                            Instr::Push(Reg::Rax),
                            Instr::Mov(Operand::Reg(Reg::Rax), Operand::Value(exprs.len() as u64)),
                            Instr::Push(Reg::Rax),
                        ]),
                );
            }
            ExprKind::StructInit(span, exprs) => todo!(),
            ExprKind::FunctionCall(span, exprs) => todo!(),
            ExprKind::FieldAccess(expr, span) => todo!(),
            ExprKind::ArrayIndex(expr, exprs) => todo!(),
            ExprKind::If(_) => todo!(),
            ExprKind::ArrayComp(items, expr, _) => todo!(),
            ExprKind::Sum(items, expr, _) => todo!(),
            ExprKind::And(_) => todo!(),
            ExprKind::Or(_) => todo!(),
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
                Typed::Int => {
                    self.gen_div_mod(args, true);
                }
                Typed::Float => self.arithmetic_binop(args, Instr::Div),
                _ => unreachable!(),
            },
            ExprKind::Modulo(args) => match expr.type_data() {
                Typed::Int => {
                    self.gen_div_mod(args, false);
                }
                Typed::Float => {
                    //FIXME: The tests are busted so dont align the stack here
                    let (lhs, rhs) = args.as_ref();
                    self.gen_asm_expr(rhs);
                    self.gen_asm_expr(lhs);
                    self.add_instrs([
                        Instr::Pop(Reg::Xmm0),
                        Instr::Pop(Reg::Xmm1),
                        Instr::Call("fmod"),
                        Instr::Push(Reg::Xmm0),
                    ]);
                }
                _ => unreachable!(),
            },
            ExprKind::Not(expr) => {
                self.gen_asm_expr(expr);
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Xor(Operand::Reg(Reg::Rax), Operand::Value(1)),
                    Instr::Push(Reg::Rax),
                ]);
            }
            ExprKind::Negation(expr) => {
                self.gen_asm_expr(expr);
                match expr.type_data() {
                    Typed::Int => {
                        self.add_instrs([
                            Instr::Pop(Reg::Rax),
                            Instr::Neg(Reg::Rax),
                            Instr::Push(Reg::Rax),
                        ]);
                    }
                    Typed::Float => {
                        self.add_instrs([
                            Instr::Pop(Reg::Xmm1),
                            Instr::Xor(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm0)),
                            Instr::Sub(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm1)),
                            Instr::Push(Reg::Xmm0),
                        ]);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    // User for most binops arithmetic binary operators except division and mod
    fn arithmetic_binop(
        &mut self,
        args: &(Expr<Typed>, Expr<Typed>),
        instr: fn(Operand, Operand) -> Instr<'static>,
    ) {
        let (lhs, rhs) = args;
        self.gen_asm_expr(rhs);
        self.gen_asm_expr(lhs);
        match lhs.type_data() {
            Typed::Int => {
                self.add_instrs([
                    Instr::Pop(Reg::Rax),
                    Instr::Pop(Reg::R10),
                    instr(Operand::Reg(Reg::Rax), Operand::Reg(Reg::R10)),
                    Instr::Push(Reg::Rax),
                ]);
            }
            Typed::Float => {
                self.add_instrs([
                    Instr::Pop(Reg::Xmm0),
                    Instr::Pop(Reg::Xmm1),
                    instr(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm1)),
                    Instr::Push(Reg::Xmm0),
                ]);
            }
            _ => unreachable!(),
        }
    }

    fn comparison_binop(&mut self, args: &(Expr<Typed>, Expr<Typed>), kind: &ExprKind<Typed>) {
        let (lhs, rhs) = args;
        self.gen_asm_expr(rhs);
        self.gen_asm_expr(lhs);
        match lhs.type_data() {
            Typed::Int | Typed::Bool => {
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
            Typed::Float => {
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
            Instr::Push(Reg::Rax),
        ]);
    }

    pub fn gen_div_mod(&mut self, args: &(Expr<Typed>, Expr<Typed>), divide: bool) {
        let (lhs, rhs) = args;
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

        let stack_aligned = self.align_stack(0);
        self.add_instrs([
            Instr::Lea(Reg::Rdi, MemLoc::Const(err_msg_id)),
            Instr::Call("fail_assertion"),
        ]);
        self.remove_stack_alignment(stack_aligned);
        self.add_asm([Asm::JumpLabel(ok_jump)]);

        self.add_instrs([
            Instr::Cqo,
            Instr::Div(Operand::Reg(Reg::Rax), Operand::Reg(Reg::R10)),
        ]);
        if !divide {
            self.add_instrs([Instr::Mov(Operand::Reg(Reg::Rax), Operand::Reg(Reg::Rdx))]);
        }

        self.add_instrs([Instr::Push(Reg::Rax)]);
    }
}
