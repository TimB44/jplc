use super::Cmd;
use crate::c_codegen::CGen;

impl CGen for Cmd {
    fn to_c(&self, env: &mut crate::environment::Environment, c_env: &mut CGenEnv) {
        match self.kind {
            super::CmdKind::ReadImage(_, lvalue) => todo!(),
            super::CmdKind::WriteImage(expr, _) => todo!(),
            super::CmdKind::Let(lvalue, expr) => todo!(),
            super::CmdKind::Assert(expr, _) => todo!(),
            super::CmdKind::Print(_) => todo!(),
            super::CmdKind::Show(expr) => todo!(),
            super::CmdKind::Time(cmd) => todo!(),
            super::CmdKind::Function {
                name,
                params,
                return_type,
                body,
                scope,
            } => todo!(),
            super::CmdKind::Struct { name, fields } => todo!(),
        }
    }
}
