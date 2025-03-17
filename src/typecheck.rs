use std::fmt;

use crate::ast::types::{self, Type};
use crate::environment::Environment;
use crate::parse::{Displayable, SExpr, SExprOptions};

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
}

pub trait TypeState {}

#[derive(Debug, Clone)]
pub struct UnTyped {}
impl TypeState for UnTyped {}
impl TypeState for TypeVal {}

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
        }
    }
}

/// Writes the type s-expression if opt is `SExprOptions::Typed`. If it is `SExprOptions::UnTyped` then it
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
            TypeVal::Int => write!(f, " (IntType) "),
            TypeVal::Bool => write!(f, " (BoolType) "),
            TypeVal::Float => write!(f, " (FloatType) "),
            TypeVal::Array(typed, rank) => {
                write!(
                    f,
                    " (ArrayType ({}) {}) ",
                    Displayable(typed.as_ref(), env, opt),
                    rank
                )
            }
            TypeVal::Struct(id) => write!(f, " (StructType {}) ", env.get_struct_id(*id).name()),
            TypeVal::Void => write!(f, " (VoidType) "),
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
                let id = env.get_struct(t.location())?.id();

                TypeVal::Struct(id)
            }
            types::TypeKind::Void => TypeVal::Void,
        })
    }
}
