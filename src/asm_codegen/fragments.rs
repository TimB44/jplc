use super::{Asm, Instr, MemLoc, Operand, Reg};

pub const HEADER: &[Asm] = &[
    Asm::Global("jpl_main"),
    Asm::Global("_jpl_main"),
    Asm::Extern("_fail_assertion"),
    Asm::Extern("_jpl_alloc"),
    Asm::Extern("_get_time"),
    Asm::Extern("_show"),
    Asm::Extern("_print"),
    Asm::Extern("_print_time"),
    Asm::Extern("_read_image"),
    Asm::Extern("_write_image"),
    Asm::Extern("_fmod"),
    Asm::Extern("_sqrt"),
    Asm::Extern("_exp"),
    Asm::Extern("_sin"),
    Asm::Extern("_cos"),
    Asm::Extern("_tan"),
    Asm::Extern("_asin"),
    Asm::Extern("_acos"),
    Asm::Extern("_atan"),
    Asm::Extern("_log"),
    Asm::Extern("_pow"),
    Asm::Extern("_atan2"),
    Asm::Extern("_to_int"),
    Asm::Extern("_to_float"),
];
pub const MAIN_PROLOGE: &[Asm] = &[
    Asm::Instr(Instr::Push(Reg::Rbp)),
    Asm::Instr(Instr::Mov(Operand::Reg(Reg::Rbp), Operand::Reg(Reg::Rsp))),
    Asm::Instr(Instr::Push(Reg::R12)),
    Asm::Instr(Instr::Mov(Operand::Reg(Reg::R12), Operand::Reg(Reg::Rbp))),
];

pub const MAIN_EPILOGUE: &[Asm] = &[
    Asm::Instr(Instr::Pop(Reg::R12)),
    Asm::Instr(Instr::Pop(Reg::Rbp)),
    Asm::Instr(Instr::Ret),
];

pub const PROLOGE: &[Asm] = &[
    Asm::Instr(Instr::Push(Reg::Rbp)),
    Asm::Instr(Instr::Mov(Operand::Reg(Reg::Rbp), Operand::Reg(Reg::Rsp))),
];

pub const EPILOGUE: &[Asm] = &[Asm::Instr(Instr::Pop(Reg::Rbp)), Asm::Instr(Instr::Ret)];

pub fn load_const(const_id: u64) -> [Instr<'static>; 2] {
    [
        Instr::Mov(
            Operand::Reg(Reg::Rax),
            Operand::Mem(MemLoc::Const(const_id)),
        ),
        Instr::Push(Reg::Rax),
    ]
}
