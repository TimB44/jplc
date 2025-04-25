use std::fmt::{self, Display, Formatter};

use crate::asm_codegen::{AsmFn, MAIN_FN_IDX, WORD_SIZE};

use super::{
    fragments::HEADER, Asm, AsmEnv, ConstKind, Instr, MemLoc, Operand, Reg, RegKind, Section,
};
const INDENTATION: &str = "\t";

impl Display for AsmEnv<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for asm in HEADER {
            write!(f, "{}", asm)?;
        }
        writeln!(f)?;

        write!(f, "{}", Asm::Section(Section::Data))?;

        // Sort is not ideal.
        let mut consts: Vec<_> = self.consts.iter().collect();
        consts.sort_by_key(|(_, b)| **b);

        for (c, id) in consts {
            // Clone not ideal
            write!(f, "{}", Asm::Const(*id, c.clone()))?;
        }
        writeln!(f)?;
        write!(f, "{}", Asm::Section(Section::Text))?;
        for AsmFn { text, .. } in self.fns.iter().skip(1).chain([&self.fns[MAIN_FN_IDX]]) {
            for asm in text {
                write!(f, "{}", asm)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
impl Display for Asm<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Asm::Global(s) => writeln!(f, "global {}", s),
            Asm::Extern(s) => writeln!(f, "extern {}", s),
            Asm::Section(section) => writeln!(f, "section {}", section),
            Asm::Const(id, const_kind) => writeln!(f, "const{}: {}", id, const_kind),
            Asm::Instr(instr) => writeln!(f, "{}{}", INDENTATION, instr),
            Asm::FnLabel(name) => write!(f, "{}:\n_{}:\n", name, name),
            Asm::JumpLabel(id) => writeln!(f, ".jump{}:", id),
            Asm::Comment(msg) => writeln!(f, "; {}:", msg),
        }
    }
}

impl Display for Instr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Instr::Mov(Operand::Mem(mem_loc), Operand::Value(value)) => {
                write!(f, "mov qword {}, {}", mem_loc, value)
            }
            Instr::Mov(lhs, rhs) => match (lhs.kind(), rhs.kind()) {
                (Some(RegKind::Int), Some(RegKind::Float))
                | (Some(RegKind::Float), Some(RegKind::Int)) => {
                    write!(f, "movq {}, {}", lhs, rhs)
                }
                (Some(RegKind::Int), _) | (_, Some(RegKind::Int)) | (None, None) => {
                    write!(f, "mov {}, {}", lhs, rhs)
                }
                (Some(RegKind::Float), _) | (_, Some(RegKind::Float)) => {
                    write!(f, "movsd {}, {}", lhs, rhs)
                }
            },
            Instr::Lea(reg, mem_loc) => write!(f, "lea {}, {}", reg, mem_loc),
            Instr::Call(name) => write!(f, "call _{}", name),
            Instr::Push(Operand::Reg(reg)) => match reg.kind() {
                RegKind::Int => {
                    write!(f, "push {}", reg)
                }
                // use a sub an add instead
                RegKind::Float => {
                    writeln!(
                        f,
                        "{}",
                        Instr::Sub(Operand::Reg(Reg::Rsp), Operand::Value(WORD_SIZE))
                    )?;
                    write!(
                        f,
                        "{}{}",
                        INDENTATION,
                        Instr::Mov(Operand::Mem(MemLoc::Reg(Reg::Rsp)), Operand::Reg(*reg))
                    )
                }
            },
            Instr::Push(op) => {
                assert!(matches!(op, Operand::Value(val) if *val <= i32::MAX as u64));
                write!(f, "push qword {}", op)
            }
            Instr::Pop(reg) => match reg.kind() {
                RegKind::Int => {
                    write!(f, "pop {}", reg)
                }
                // use a sub an add instead
                RegKind::Float => {
                    writeln!(
                        f,
                        "{}",
                        Instr::Mov(Operand::Reg(*reg), Operand::Mem(MemLoc::Reg(Reg::Rsp)))
                    )?;
                    write!(
                        f,
                        "{}{}",
                        INDENTATION,
                        Instr::Add(Operand::Reg(Reg::Rsp), Operand::Value(WORD_SIZE))
                    )
                }
            },
            Instr::Add(Operand::Mem(mem_loc), Operand::Value(value)) => {
                write!(f, "add qword {}, {}", mem_loc, value)
            }
            Instr::Add(lhs, rhs) => match lhs.args_kind(rhs) {
                RegKind::Int => {
                    write!(f, "add {}, {}", lhs, rhs)
                }
                RegKind::Float => {
                    write!(f, "addsd {}, {}", lhs, rhs)
                }
            },
            Instr::Sub(lhs, rhs) => match lhs.args_kind(rhs) {
                RegKind::Int => {
                    write!(f, "sub {}, {}", lhs, rhs)
                }
                RegKind::Float => {
                    write!(f, "subsd {}, {}", lhs, rhs)
                }
            },
            Instr::Ret => write!(f, "ret"),
            Instr::Neg(reg) => write!(f, "neg {}", reg),
            Instr::Xor(lhs, rhs) => match lhs.args_kind(rhs) {
                RegKind::Int => write!(f, "xor {}, {}", lhs, rhs),
                RegKind::Float => write!(f, "pxor {}, {}", lhs, rhs),
            },
            Instr::And(lhs, rhs) => write!(f, "and {}, {}", lhs, rhs),
            Instr::Mul(lhs, rhs) => match lhs.args_kind(rhs) {
                RegKind::Int => write!(f, "imul {}, {}", lhs, rhs),
                RegKind::Float => write!(f, "mulsd {}, {}", lhs, rhs),
            },
            Instr::Div(lhs, rhs) => match lhs.args_kind(rhs) {
                RegKind::Int => {
                    assert!(matches!(lhs, |Operand::Reg(Reg::Rax)));
                    write!(f, "idiv {}", rhs)
                }
                RegKind::Float => write!(f, "divsd {}, {}", lhs, rhs),
            },
            Instr::Setl => write!(f, "setl al"),
            Instr::Setg => write!(f, "setg al"),
            Instr::Setle => write!(f, "setle al"),
            Instr::Setge => write!(f, "setge al",),
            Instr::Sete => write!(f, "sete al"),
            Instr::Setne => write!(f, "setne al"),
            Instr::Cmp(lhs, rhs) => write!(f, "cmp {}, {}", lhs, rhs),
            Instr::Jne(jump_id) => write!(f, "jne .jump{}", jump_id),
            Instr::Je(jump_id) => write!(f, "je .jump{}", jump_id),
            Instr::Jge(jump_id) => write!(f, "jge .jump{}", jump_id),
            Instr::Jl(jump_id) => write!(f, "jl .jump{}", jump_id),
            Instr::Jmp(jump_id) => write!(f, "jmp .jump{}", jump_id),
            Instr::Jno(jump_id) => write!(f, "jno .jump{}", jump_id),
            Instr::Jg(jump_id) => write!(f, "jg .jump{}", jump_id),
            Instr::Cqo => write!(f, "cqo"),
            Instr::Cmplt(lhs, rhs) => write!(f, "cmpltsd {}, {}", lhs, rhs),
            Instr::Cmple(lhs, rhs) => write!(f, "cmplesd {}, {}", lhs, rhs),
            Instr::Cmpeq(lhs, rhs) => write!(f, "cmpeqsd {}, {}", lhs, rhs),
            Instr::Cmpneq(lhs, rhs) => write!(f, "cmpneqsd {}, {}", lhs, rhs),
            Instr::Shl(lhs, rhs) => write!(f, "shl {}, {}", lhs, rhs),
        }
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(reg) => write!(f, "{}", reg),
            Operand::Mem(mem_loc) => write!(f, "{}", mem_loc),
            Operand::Value(v) => write!(f, "{}", v),
        }
    }
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Rax => write!(f, "rax"),
            Reg::Rbp => write!(f, "rbp"),
            Reg::Rsp => write!(f, "rsp"),
            Reg::R12 => write!(f, "r12"),
            Reg::Rdi => write!(f, "rdi"),
            Reg::Rdx => write!(f, "rdx"),
            Reg::Rsi => write!(f, "rsi"),
            Reg::Rcx => write!(f, "rcx"),
            Reg::R8 => write!(f, "r8"),
            Reg::R9 => write!(f, "r9"),
            Reg::R10 => write!(f, "r10"),
            Reg::Xmm0 => write!(f, "xmm0"),
            Reg::Xmm1 => write!(f, "xmm1"),
            Reg::Xmm2 => write!(f, "xmm2"),
            Reg::Xmm3 => write!(f, "xmm3"),
            Reg::Xmm4 => write!(f, "xmm4"),
            Reg::Xmm5 => write!(f, "xmm5"),
            Reg::Xmm6 => write!(f, "xmm6"),
            Reg::Xmm7 => write!(f, "xmm7"),
        }
    }
}

impl Display for MemLoc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MemLoc::GlobalOffset(offset, add) => write!(f, "[{} - {} + {}]", Reg::R12, offset, add),
            MemLoc::LocalOffset(offset, add) => write!(f, "[{} - {} + {}]", Reg::Rbp, offset, add),
            MemLoc::Const(id) => write!(f, "[rel const{}]", id),
            MemLoc::Reg(reg) => write!(f, "[{}]", reg),
            MemLoc::RegOffset(reg, offset) => write!(
                f,
                "[{} {} {}]",
                reg,
                if *offset < 0 { '-' } else { '+' },
                offset.abs()
            ),
        }
    }
}

impl Display for Section {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Section::Data => write!(f, ".data"),
            Section::Text => write!(f, ".text"),
        }
    }
}
impl Display for ConstKind<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ConstKind::Int(val) => write!(f, "dq {}", val),
            ConstKind::Float(val) => write!(f, "dq {:?}", f64::from_bits(*val)),
            ConstKind::String(cow) => {
                write!(f, "db `")?;
                for c in cow.as_ref().chars() {
                    assert_ne!(c, '\n');
                    if c == '\\' {
                        write!(f, "\\")?;
                    }
                    write!(f, "{}", c)?;
                }

                write!(f, "`, 0")
            }
        }
    }
}
