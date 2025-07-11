use std::fmt;

use crate::ast::types::{self, Type};
use crate::environment::Environment;
use crate::parse::{Displayable, SExpr, SExprOptions};
use std::borrow::Cow;
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeVal {
    Int,
    Bool,
    Float,

    // Type of the array and the rank
    Array(Box<TypeVal>, u8),

    // Id of struct
    Struct(usize),
    Void,

    FnPointer(Box<[TypeVal]>, Box<TypeVal>),
}

impl TypeVal {
    pub fn as_str(&self, env: &Environment) -> String {
        match self {
            TypeVal::Int => "int".to_string(),
            TypeVal::Bool => "bool".to_string(),
            TypeVal::Float => "float".to_string(),
            TypeVal::Array(element_type, rank) => {
                format!(
                    "{}[{}]",
                    element_type.as_str(env),
                    ",".repeat((*rank as usize) - 1)
                )
            }
            TypeVal::Struct(id) => env.struct_info()[*id].name().to_string(),
            TypeVal::Void => "void".to_string(),
            TypeVal::FnPointer(arg_types, ret_type) => {
                format!(
                    "fn({}) -> {}",
                    arg_types
                        .iter()
                        .map(|t| t.as_str(env))
                        .collect::<Vec<_>>()
                        .join(", "),
                    ret_type.as_str(env)
                )
            }
        }
    }

    pub fn as_struct(&self) -> usize {
        match self {
            TypeVal::Struct(id) => *id,
            _ => panic!("not a struct"),
        }
    }
}
/// Writes the type s-expression if opt is `SExprOptions::TypeVal`. If it is `SExprOptions::UnTypeVal` then it
/// will not write anything. Note: unlike other s-expression writers this will print a space around it
/// in order of it to be easily used for typed and untyped s-expressions
impl SExpr for TypeVal {
    fn to_s_expr(
        &self,
        f: &mut fmt::Formatter<'_>,
        env: &Environment<'_>,
        opt: SExprOptions,
    ) -> fmt::Result {
        if let SExprOptions::Untyped = opt {
            return Ok(());
        }

        match self {
            TypeVal::Int => write!(f, " (IntType)"),
            TypeVal::Bool => write!(f, " (BoolType)"),
            TypeVal::Float => write!(f, " (FloatType)"),
            TypeVal::Array(typed, rank) => {
                write!(
                    f,
                    " (ArrayType{} {})",
                    Displayable(typed.as_ref(), env, opt),
                    rank
                )
            }
            TypeVal::Struct(id) => write!(f, " (StructType {})", env.get_struct_id(*id).name()),
            TypeVal::Void => write!(f, " (VoidType)"),
            TypeVal::FnPointer(arg_types, ret_type) => {
                write!(
                    f,
                    "(FunctionPointerType ({}) {})",
                    Displayable(arg_types, env, opt),
                    Displayable(ret_type.as_ref(), env, opt),
                )
            }
        }
    }
}

impl TypeVal {
    pub fn from_ast_type(t: &Type, env: &Environment) -> miette::Result<Self> {
        Ok(match t.kind() {
            types::TypeKind::Int => TypeVal::Int,
            types::TypeKind::Bool => TypeVal::Bool,
            types::TypeKind::Float => TypeVal::Float,
            types::TypeKind::Array(inner, rank) => {
                TypeVal::Array(Box::new(Self::from_ast_type(inner, env)?), *rank)
            }
            types::TypeKind::Struct => {
                let id = env.get_struct(t.loc())?.id();

                TypeVal::Struct(id)
            }
            types::TypeKind::Void => TypeVal::Void,
            types::TypeKind::FnPointer(items, ret_type) => TypeVal::FnPointer(
                items
                    .iter()
                    .map(|arg| TypeVal::from_ast_type(arg, env))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
                Box::new(TypeVal::from_ast_type(ret_type, env)?),
            ),
        })
    }

    pub fn write_type_string(&self, s: &mut String, env: &Environment) {
        match self {
            TypeVal::Array(inner_type, rank) => {
                s.push_str("(ArrayType ");
                inner_type.write_type_string(s, env);
                write!(s, " {})", rank).expect("string should not fail to write");
            }
            TypeVal::Struct(id) => {
                s.push_str("(TupleType");
                let info = env.get_struct_id(*id);
                for (_, ty) in info.fields() {
                    s.push(' ');
                    ty.write_type_string(s, env);
                }

                //TODO remove, possible bug in tests
                if info.fields().is_empty() {
                    s.push(' ');
                }
                s.push(')');
            }
            TypeVal::Int => {
                s.push_str("(IntType)");
            }
            TypeVal::Bool => {
                s.push_str("(BoolType)");
            }
            TypeVal::Float => {
                s.push_str("(FloatType)");
            }
            TypeVal::Void => {
                s.push_str("(VoidType)");
            }
            TypeVal::FnPointer(arg_types, ret_type) => write!(
                s,
                "(FunctionPointerType ({}) {})",
                Displayable(arg_types, env, SExprOptions::Untyped),
                Displayable(ret_type.as_ref(), env, SExprOptions::Untyped),
            )
            .unwrap(),
        }
    }

    pub fn to_type_string<'a>(&self, env: &Environment<'a>) -> Cow<'a, str> {
        match self {
            TypeVal::Array(inner_type, rank) => {
                let mut s = String::new();
                s.push_str("(ArrayType ");
                inner_type.write_type_string(&mut s, env);
                write!(s, " {})", rank).expect("string should not fail to write");
                Cow::Owned(s)
            }
            TypeVal::Struct(id) => {
                let mut s = String::new();
                s.push_str("(TupleType");
                let info = env.get_struct_id(*id);
                for (_, ty) in info.fields() {
                    s.push(' ');
                    ty.write_type_string(&mut s, env);
                }

                //TODO remove, possible bug in tests
                if info.fields().is_empty() {
                    s.push(' ');
                }
                s.push(')');
                Cow::Owned(s)
            }
            TypeVal::Int => Cow::Borrowed("(IntType)"),
            TypeVal::Bool => Cow::Borrowed("(BoolType)"),
            TypeVal::Float => Cow::Borrowed("(FloatType)"),
            TypeVal::Void => Cow::Borrowed("(VoidType)"),
            TypeVal::FnPointer(arg_types, ret_type) => Cow::Owned(format!(
                "(FunctionPointerType ({}) {})",
                Displayable(arg_types, env, SExprOptions::Untyped),
                Displayable(ret_type.as_ref(), env, SExprOptions::Untyped),
            )),
        }
    }
}
