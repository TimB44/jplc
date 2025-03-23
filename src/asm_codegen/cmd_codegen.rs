use crate::ast::cmd::{Cmd, CmdKind};

use super::{AsmEnv, ConstKind, Instr, MemLoc, Operand, Reg};

impl<'a, 'b> AsmEnv<'a, 'b> {
    pub fn gen_asm_cmd(&mut self, cmd: &Cmd) {
        match cmd.kind() {
            CmdKind::ReadImage(_, lvalue) => todo!(),
            CmdKind::WriteImage(expr, _) => todo!(),
            CmdKind::Let(lvalue, expr) => todo!(),
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
                name,
                params,
                return_type,
                body,
                scope,
            } => todo!(),
            CmdKind::Struct { name, fields } => todo!(),
        }
    }
}
