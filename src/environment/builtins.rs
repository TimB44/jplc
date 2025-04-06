use std::{collections::HashMap, sync::LazyLock};

use crate::{
    asm_codegen::WORD_SIZE,
    environment::{VarInfo, GLOBAL_SCOPE_ID},
    typecheck::TypeVal,
};

use super::{FunctionInfo, Scope, StructInfo};

pub const RGBA_STRUCT_ID: usize = 0;
pub static IMAGE_TYPE: LazyLock<TypeVal> =
    LazyLock::new(|| TypeVal::Array(Box::new(TypeVal::Struct(RGBA_STRUCT_ID)), 2));

pub fn builtin_structs() -> (HashMap<&'static str, usize>, Vec<StructInfo<'static>>) {
    let struct_ids = HashMap::from([("rgba", 0)]);
    let struct_fields = vec![StructInfo {
        fields: vec![
            ("r", TypeVal::Float),
            ("g", TypeVal::Float),
            ("b", TypeVal::Float),
            ("a", TypeVal::Float),
        ]
        .into_boxed_slice(),
        id: 0,
        name: "rgba",
        size: WORD_SIZE * 4,
    }];
    assert_eq!(RGBA_STRUCT_ID, *struct_ids.get("rgba").unwrap());
    (struct_ids, struct_fields)
}

pub fn builtin_vars() -> Scope<'static> {
    Scope {
        names: HashMap::from([
            (
                "args",
                VarInfo {
                    var_type: TypeVal::Array(Box::new(TypeVal::Int), 1),
                    // stack_loc: -16,
                },
            ),
            (
                "argnum",
                VarInfo {
                    var_type: TypeVal::Int,
                    // stack_loc: -16,
                },
            ),
        ]),
        parent: GLOBAL_SCOPE_ID,
        // Space for r12
    }
}

pub fn builtin_fns() -> HashMap<&'static str, FunctionInfo<'static>> {
    HashMap::from([
        (
            "sqrt",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "sqrt",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "exp",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "exp",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "sin",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "sin",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "cos",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "cos",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "tan",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "tan",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "asin",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "asin",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "acos",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "acos",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "atan",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "atan",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "log",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "log",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "pow",
            FunctionInfo {
                args: vec![TypeVal::Float, TypeVal::Float].into_boxed_slice(),
                name: "pow",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "atan2",
            FunctionInfo {
                args: vec![TypeVal::Float, TypeVal::Float].into_boxed_slice(),
                name: "atan2",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "to_float",
            FunctionInfo {
                args: vec![TypeVal::Int].into_boxed_slice(),
                name: "to_float",
                ret: TypeVal::Float,
                scope: 0,
            },
        ),
        (
            "to_int",
            FunctionInfo {
                args: vec![TypeVal::Float].into_boxed_slice(),
                name: "to_int",
                ret: TypeVal::Int,
                scope: 0,
            },
        ),
    ])
}
