use std::{collections::HashMap, sync::LazyLock};

use crate::{
    environment::{VarInfo, GLOBAL_SCOPE_ID},
    typecheck::Typed,
};

use super::{FunctionInfo, Scope, StructInfo};

pub const RGBA_STRUCT_ID: usize = 0;
pub const IMAGE_TYPE: LazyLock<Typed> =
    LazyLock::new(|| Typed::Array(Box::new(Typed::Struct(RGBA_STRUCT_ID)), 2));

pub fn builtin_structs() -> (HashMap<&'static str, usize>, Vec<StructInfo<'static>>) {
    let struct_ids = HashMap::from([("rgba", 0)]);
    let struct_fields = vec![StructInfo {
        fields: vec![
            ("r", Typed::Float),
            ("g", Typed::Float),
            ("b", Typed::Float),
            ("a", Typed::Float),
        ]
        .into_boxed_slice(),
        id: 0,
        name: "rgba",
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
                    var_type: Typed::Array(Box::new(Typed::Int), 1),
                    bindings: vec![].into_boxed_slice(),
                },
            ),
            (
                "argnum",
                VarInfo {
                    var_type: Typed::Int,
                    bindings: vec![].into_boxed_slice(),
                },
            ),
        ]),
        parent: GLOBAL_SCOPE_ID,
    }
}

pub fn builtin_fns() -> HashMap<&'static str, FunctionInfo<'static>> {
    HashMap::from([
        (
            "sqrt",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "sqrt",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "exp",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "exp",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "sin",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "sin",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "cos",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "cos",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "tan",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "tan",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "asin",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "asin",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "acos",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "acos",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "atan",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "atan",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "log",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "log",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "pow",
            FunctionInfo {
                args: vec![Typed::Float, Typed::Float].into_boxed_slice(),
                name: "pow",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "atan2",
            FunctionInfo {
                args: vec![Typed::Float, Typed::Float].into_boxed_slice(),
                name: "atan2",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "to_float",
            FunctionInfo {
                args: vec![Typed::Int].into_boxed_slice(),
                name: "to_float",
                ret: Typed::Float,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
        (
            "to_int",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "to_int",
                ret: Typed::Int,
                //TODO: IDK if this is good
                scope: 0,
            },
        ),
    ])
}
