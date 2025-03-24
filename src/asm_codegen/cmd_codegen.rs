use crate::{
    ast::cmd::{Cmd, CmdKind},
    environment::GLOBAL_SCOPE_ID,
    typecheck::TypeVal,
};

use super::{
    fragments::{EPILOGUE, PROLOGE},
    AsmEnv, AsmFn, ConstKind, Instr, MemLoc, Operand, Reg, FLOAT_REGS_FOR_ARGS, INT_REGS_FOR_ARGS,
    MAIN_FN_IDX,
};

impl<'a> AsmEnv<'a> {
    pub fn gen_asm_cmd(&mut self, cmd: &Cmd) {
        match cmd.kind() {
            CmdKind::ReadImage(_, lvalue) => todo!(),
            CmdKind::WriteImage(expr, _) => todo!(),
            CmdKind::Let(_, expr) => self.gen_asm_expr(expr),

            CmdKind::Assert(expr, _) => todo!(),
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
            CmdKind::Time(cmd) => todo!(),
            CmdKind::Function {
                name, body, scope, ..
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
                let aggregate_ret_val = dbg!(matches!(
                    fn_info.ret(),
                    TypeVal::Array(_, _) | TypeVal::Struct(_)
                ));
                let mut int_regs = &INT_REGS_FOR_ARGS[if aggregate_ret_val { 1 } else { 0 }..];
                let mut fp_regs = &FLOAT_REGS_FOR_ARGS[if aggregate_ret_val { 1 } else { 0 }..];
                self.add_instrs(fn_info.args().iter().filter_map(|arg| match arg {
                    TypeVal::Int | TypeVal::Bool | TypeVal::Void if int_regs.len() > 0 => {
                        let reg = int_regs[0];
                        int_regs = &int_regs[1..];
                        Some(Instr::Push(reg))
                    }
                    TypeVal::Float if fp_regs.len() > 0 => {
                        let reg = fp_regs[0];
                        fp_regs = &fp_regs[1..];
                        Some(Instr::Push(reg))
                    }
                    _ => None,
                }));

                if aggregate_ret_val {
                    self.add_instrs([Instr::Push(Reg::Rdi)]);
                }

                for stmt in body {
                    self.gen_asm_stmt(stmt);
                }
                // Epilogue added by return statment
                self.cur_fn = MAIN_FN_IDX;
                self.cur_scope = GLOBAL_SCOPE_ID;
            }
            CmdKind::Struct { name, fields } => todo!(),
        }
    }
}
