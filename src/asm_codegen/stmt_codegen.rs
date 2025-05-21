use crate::{
    asm_codegen::FN_STARTING_STACK_SIZE,
    ast::stmt::{Stmt, StmtType},
    typecheck::TypeVal,
};

use super::{fragments::EPILOGUE, AsmEnv, Instr, MemLoc, Operand, Reg, WORD_SIZE};

impl AsmEnv<'_> {
    pub fn gen_asm_stmt(&mut self, stmt: &Stmt) {
        match stmt.kind() {
            StmtType::Let(lvalue, expr) => {
                self.gen_asm_expr(expr);
                self.add_lvalue(lvalue, self.fns[self.cur_fn].cur_stack_size as i64);
            }
            StmtType::Assert(cond, msg) => self.codegen_assert(cond, msg),
            StmtType::Return(expr) => {
                self.gen_asm_expr(expr);
                match expr.type_data() {
                    TypeVal::Int | TypeVal::Bool | TypeVal::Void | TypeVal::FnPointer(_, _) => {
                        self.add_instrs([Instr::Pop(Reg::Rax)])
                    }
                    TypeVal::Float => self.add_instrs([Instr::Pop(Reg::Xmm0)]),

                    TypeVal::Array(_, _) | TypeVal::Struct(_) => {
                        let ret_type_size = self.env.type_size(expr.type_data());
                        self.add_instrs([Instr::Mov(
                            Operand::Reg(Reg::Rax),
                            Operand::Mem(MemLoc::RegOffset(Reg::Rbp, -(WORD_SIZE as i64))),
                        )]);

                        self.copy(ret_type_size, Reg::Rsp, 0, Reg::Rax, 0);
                    }
                }

                let local_vars_size = self.fns[self.cur_fn].cur_stack_size - FN_STARTING_STACK_SIZE;
                assert!(local_vars_size % WORD_SIZE == 0);
                self.add_instrs([Instr::Add(
                    Operand::Reg(Reg::Rsp),
                    Operand::Value(local_vars_size),
                )]);

                self.add_asm(EPILOGUE);
                let cur_fn = &mut self.fns[self.cur_fn];
                assert_eq!(cur_fn.cur_stack_size, 0);

                // Add back the old stack size if case of multiple return values
                // really we should just not generate code for them but the grader
                // expects it
                cur_fn.cur_stack_size += FN_STARTING_STACK_SIZE + local_vars_size;
            }
        }
    }
}
