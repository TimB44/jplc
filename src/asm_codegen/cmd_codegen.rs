use std::borrow::Cow;

use crate::{
    asm_codegen::{
        expr_codegen::VOID_VALUE, fragments::EPILOGUE, FN_STARTING_STACK_SIZE, WORD_SIZE,
    },
    ast::{
        cmd::{Cmd, CmdKind},
        stmt::StmtType,
        types::TypeKind,
    },
    environment::{builtins::IMAGE_TYPE, GLOBAL_SCOPE_ID},
    typecheck::TypeVal,
};

use super::{
    fragments::PROLOGE, AsmEnv, AsmFn, ConstKind, Instr, MemLoc, Operand, Reg, FLOAT_REGS_FOR_ARGS,
    INT_REGS_FOR_ARGS, MAIN_FN_IDX,
};

impl AsmEnv<'_> {
    pub fn gen_asm_cmd(&mut self, cmd: &Cmd) {
        match cmd.kind() {
            CmdKind::ReadImage(filename, lvalue) => {
                let filename_str = filename.inner_loc().as_str(self.env.src());
                let filename_id = self.add_const(&ConstKind::String(Cow::Borrowed(filename_str)));
                self.add_instrs([
                    Instr::Sub(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(self.env.type_size(&IMAGE_TYPE)),
                    ),
                    Instr::Lea(Reg::Rdi, MemLoc::Reg(Reg::Rsp)),
                ]);
                let stack_was_aligned = self.align_stack(0);
                self.add_instrs([
                    Instr::Lea(Reg::Rsi, MemLoc::Const(filename_id)),
                    Instr::Call("read_image"),
                ]);
                self.remove_stack_alignment(stack_was_aligned);
                self.add_lvalue(lvalue, self.fns[self.cur_fn].cur_stack_size as i64);
            }
            CmdKind::WriteImage(expr, filename) => {
                let stack_was_aligned = self.align_stack(self.env.type_size(&IMAGE_TYPE));
                self.gen_asm_expr(expr);
                let filename_str = filename.inner_loc().as_str(self.env.src());
                let filename_id = self.add_const(&ConstKind::String(Cow::Borrowed(filename_str)));
                self.add_instrs([
                    Instr::Lea(Reg::Rdi, MemLoc::Const(filename_id)),
                    Instr::Call("write_image"),
                    Instr::Add(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(self.env.type_size(&IMAGE_TYPE)),
                    ),
                ]);
                self.remove_stack_alignment(stack_was_aligned);
            }
            CmdKind::Let(lvalue, expr) => {
                self.gen_asm_expr(expr);
                self.add_lvalue(lvalue, self.fns[self.cur_fn].cur_stack_size as i64);
            }

            CmdKind::Assert(cond, msg) => self.codegen_assert(cond, msg),
            CmdKind::Print(msg) => {
                let msg_str = msg.inner_loc().as_str(self.env.src());
                let msg_id = self.add_const(&ConstKind::String(Cow::Borrowed(msg_str)));
                self.add_instrs([Instr::Lea(Reg::Rdi, MemLoc::Const(msg_id))]);
                let stack_was_aligned = self.align_stack(0);
                self.add_instrs([Instr::Call("print")]);
                self.remove_stack_alignment(stack_was_aligned);
            }
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
            CmdKind::Time(cmd) => {
                let stack_aligned = self.align_stack(0);
                self.add_instrs([Instr::Call("get_time")]);
                self.remove_stack_alignment(stack_aligned);
                self.add_instrs([Instr::Push(Operand::Reg(Reg::Xmm0))]);

                let stack_before = self.fns[self.cur_fn].cur_stack_size;
                self.gen_asm_cmd(cmd);
                let stack_after = self.fns[self.cur_fn].cur_stack_size;

                let stack_aligned = self.align_stack(0);
                self.add_instrs([Instr::Call("get_time")]);
                self.remove_stack_alignment(stack_aligned);

                self.add_instrs([
                    Instr::Push(Operand::Reg(Reg::Xmm0)),
                    Instr::Pop(Reg::Xmm0),
                    Instr::Mov(
                        Operand::Reg(Reg::Xmm1),
                        Operand::Mem(MemLoc::RegOffset(
                            Reg::Rsp,
                            (stack_after - stack_before) as i64,
                        )),
                    ),
                    Instr::Sub(Operand::Reg(Reg::Xmm0), Operand::Reg(Reg::Xmm1)),
                ]);
                let stack_aligned = self.align_stack(0);
                self.add_instrs([Instr::Call("print_time")]);
                self.remove_stack_alignment(stack_aligned);
            }
            CmdKind::Function {
                name,
                body,
                scope,
                params,
                return_type,
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

                let mut found_return = false;
                for stmt in body {
                    found_return |= matches!(stmt.kind(), StmtType::Return(_));
                    self.gen_asm_stmt(stmt);
                }

                if !found_return {
                    assert!(matches!(return_type.kind(), TypeKind::Void));
                    let local_vars_size =
                        self.fns[self.cur_fn].cur_stack_size - FN_STARTING_STACK_SIZE;
                    assert!(local_vars_size % WORD_SIZE == 0);

                    let const_id = self.add_const(&ConstKind::Int(VOID_VALUE));
                    self.load_const(const_id);
                    self.add_instrs([Instr::Pop(Reg::Rax)]);
                    self.add_instrs([Instr::Add(
                        Operand::Reg(Reg::Rsp),
                        Operand::Value(local_vars_size),
                    )]);

                    self.add_asm(EPILOGUE);
                }
                // Epilogue added by return statment
                self.cur_fn = MAIN_FN_IDX;
                self.cur_scope = GLOBAL_SCOPE_ID;
            }
            CmdKind::Struct { .. } => (),
        }
    }
}
