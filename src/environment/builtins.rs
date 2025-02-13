use std::collections::HashMap;

use crate::typecheck::Typed;

use super::FunctionInfo;

fn generate_builtin_fns() -> HashMap<&'static str, FunctionInfo<'static>> {
    HashMap::from([
        (
            "sqrt",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "sqrt",
                ret: Typed::Float,
            },
        ),
        (
            "exp",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "exp",
                ret: Typed::Float,
            },
        ),
        (
            "sin",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "sin",
                ret: Typed::Float,
            },
        ),
        (
            "cos",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "cos",
                ret: Typed::Float,
            },
        ),
        (
            "tan",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "tan",
                ret: Typed::Float,
            },
        ),
        (
            "asin",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "asin",
                ret: Typed::Float,
            },
        ),
        (
            "acos",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "acos",
                ret: Typed::Float,
            },
        ),
        (
            "atan",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "acos",
                ret: Typed::Float,
            },
        ),
        (
            "log",
            FunctionInfo {
                args: vec![Typed::Float].into_boxed_slice(),
                name: "log",
                ret: Typed::Float,
            },
        ),
    ])
}
