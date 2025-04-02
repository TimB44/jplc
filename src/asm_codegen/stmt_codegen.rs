use crate::{
    asm_codegen::FN_STARTING_STACK_SIZE,
    ast::stmt::{Stmt, StmtType},
    typecheck::TypeVal,
};

use super::{fragments::EPILOGUE, AsmEnv, Instr, MemLoc, Operand, Reg, WORD_SIZE};

impl AsmEnv<'_> {
    pub fn gen_asm_stmt(&mut self, stmt: &Stmt) {
        match stmt.kind() {
            StmtType::Let(_, expr) => {
                self.gen_asm_expr(expr);
            }
            StmtType::Assert(expr, _) => todo!(),
            StmtType::Return(expr) => {
                self.gen_asm_expr(expr);
                match expr.type_data() {
                    TypeVal::Int | TypeVal::Bool | TypeVal::Void => {
                        self.add_instrs([Instr::Pop(Reg::Rax)])
                    }
                    TypeVal::Float => self.add_instrs([Instr::Pop(Reg::Xmm0)]),

                    TypeVal::Array(_, _) | TypeVal::Struct(_) => {
                        let ret_type_size = self.env.type_size(expr.type_data());
                        self.add_instrs([Instr::Mov(
                            Operand::Reg(Reg::Rax),
                            Operand::Mem(MemLoc::RegOffset(Reg::Rbp, -(WORD_SIZE as i64))),
                        )]);

                        self.copy_from_stack(ret_type_size, Reg::Rsp, Reg::Rax);
                    }
                }

                let local_vars_size = self.fns[self.cur_fn].cur_stack_size - FN_STARTING_STACK_SIZE;
                assert!(local_vars_size % WORD_SIZE == 0);
                //FIXME: this check is valid but the tests want the add 0 instruction
                //if local_vars_size > 0 {
                self.add_instrs([Instr::Add(
                    Operand::Reg(Reg::Rsp),
                    Operand::Value(local_vars_size),
                )]);
                //}
                self.add_asm(EPILOGUE);
                let cur_fn = &mut self.fns[self.cur_fn];
                assert_eq!(cur_fn.cur_stack_size, 0);
                cur_fn.cur_stack_size += FN_STARTING_STACK_SIZE + local_vars_size;
                // Add back the old stack size if case of multiple return values
                //
            }
        }
    }
}
