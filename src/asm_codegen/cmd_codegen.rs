use crate::{
    ast::cmd::{Cmd, CmdKind},
    environment::GLOBAL_SCOPE_ID,
    typecheck::TypeVal,
};

use super::{
    fragments::PROLOGE, AsmEnv, AsmFn, ConstKind, Instr, MemLoc, Operand, Reg, FLOAT_REGS_FOR_ARGS,
    INT_REGS_FOR_ARGS, MAIN_FN_IDX,
};

impl AsmEnv<'_> {
    pub fn gen_asm_cmd(&mut self, cmd: &Cmd) {
        match cmd.kind() {
            CmdKind::ReadImage(_, _) => todo!(),
            CmdKind::WriteImage(_, _) => todo!(),
            CmdKind::Let(lvalue, expr) => {
                self.gen_asm_expr(expr);
                self.add_lvalue(lvalue, self.fns[self.cur_fn].cur_stack_size as i64);
            }

            CmdKind::Assert(_, _) => todo!(),
            CmdKind::Print(_) => todo!(),
            CmdKind::Show(expr) => {
                let type_size = self.env.type_size(expr.type_data());
                let stack_aligned = self.align_stack(type_size);
                self.gen_asm_expr(expr);
                let id = self.add_const(&ConstKind::String(
                    expr.type_data().to_type_string(self.env),
                ));

                self.add_instrs([
                    Instr::Lea(Reg::Rdi, MemLoc::Const(id)),
                    Instr::Lea(Reg::Rsi, MemLoc::Reg(Reg::Rsp)),
                    Instr::Call("show"),
                    Instr::Add(Operand::Reg(Reg::Rsp), Operand::Value(type_size)),
                ]);

                self.remove_stack_alignment(stack_aligned);
            }
            CmdKind::Time(_) => todo!(),
            CmdKind::Function {
                name,
                body,
                scope,
                params,
                ..
            } => {
                let fn_info = self
                    .env
                    .get_function(name.0)
                    .expect("function should exist after type-checking");
                let cur_fn = AsmFn::new(fn_info.name());
                let cur_fn_idx = self.fns.len();
                self.fns.push(cur_fn);
                self.cur_fn = cur_fn_idx;
                self.cur_scope = *scope;

                self.add_asm(PROLOGE);
                let aggregate_ret_val =
                    matches!(fn_info.ret(), TypeVal::Array(_, _) | TypeVal::Struct(_));
                if aggregate_ret_val {
                    self.add_instrs([Instr::Push(Operand::Reg(Reg::Rdi))]);
                }
                let mut int_regs = &INT_REGS_FOR_ARGS[if aggregate_ret_val { 1 } else { 0 }..];
                let mut fp_regs = FLOAT_REGS_FOR_ARGS.as_slice();

                let mut stack_args_loc = 0;
                for (arg, lvalue) in fn_info
                    .args()
                    .iter()
                    .zip(params.into_iter().map(|b| b.lvalue()))
                {
                    match arg {
                        TypeVal::Int | TypeVal::Bool | TypeVal::Void if !int_regs.is_empty() => {
                            let reg = int_regs[0];
                            int_regs = &int_regs[1..];
                            self.add_instrs([Instr::Push(Operand::Reg(reg))]);
                            self.add_lvalue(lvalue, self.fns[self.cur_fn].cur_stack_size as i64);
                        }
                        TypeVal::Float if !fp_regs.is_empty() => {
                            let reg = fp_regs[0];
                            fp_regs = &fp_regs[1..];
                            self.add_instrs([Instr::Push(Operand::Reg(reg))]);
                            self.add_lvalue(lvalue, self.fns[self.cur_fn].cur_stack_size as i64);
                        }
                        _ => {
                            self.add_lvalue(lvalue, stack_args_loc);
                            stack_args_loc -= self.env.type_size(arg) as i64;
                        }
                    }
                }

                for stmt in body {
                    self.gen_asm_stmt(stmt);
                }
                // Epilogue added by return statment
                self.cur_fn = MAIN_FN_IDX;
                self.cur_scope = GLOBAL_SCOPE_ID;
            }
            CmdKind::Struct { .. } => todo!(),
        }
    }
}
