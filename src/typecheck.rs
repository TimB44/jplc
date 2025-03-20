use crate::ast::types::{self, Type};
use crate::environment::Environment;
use std::borrow::Cow;
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Typed {
    Int,
    Bool,
    Float,

    // Type of the array and the rank
    Array(Box<Typed>, u8),

    // Id of struct
    Struct(usize),
    Void,
}

pub trait TypeState {}

#[derive(Debug, Clone)]
pub struct UnTyped {}
impl TypeState for UnTyped {}
impl TypeState for Typed {}

impl Typed {
    pub fn as_str(&self, env: &Environment) -> String {
        match self {
            Typed::Int => "int".to_string(),
            Typed::Bool => "bool".to_string(),
            Typed::Float => "float".to_string(),
            Typed::Array(element_type, rank) => {
                format!(
                    "{}[{}]",
                    element_type.as_str(env),
                    ",".repeat((*rank as usize) - 1)
                )
            }
            Typed::Struct(id) => env.struct_info()[*id].name().to_string(),
            Typed::Void => "void".to_string(),
        }
    }

    pub fn to_typed_s_exprsision(&self, env: &Environment) -> String {
        match self {
            Typed::Int => "IntType".to_string(),
            Typed::Bool => "BoolType".to_string(),
            Typed::Float => "FloatType".to_string(),
            Typed::Array(typed, rank) => {
                format!("ArrayType ({}) {}", typed.to_typed_s_exprsision(env), rank)
            }
            Typed::Struct(id) => format!("StructType {}", env.get_struct_id(*id).name()),
            Typed::Void => "VoidType".to_string(),
        }
    }

    pub fn from_ast_type(t: &Type, env: &Environment) -> miette::Result<Self> {
        Ok(match t.kind() {
            types::TypeKind::Int => Typed::Int,
            types::TypeKind::Bool => Typed::Bool,
            types::TypeKind::Float => Typed::Float,
            types::TypeKind::Array(inner, rank) => {
                Typed::Array(Box::new(Self::from_ast_type(inner, env)?), *rank)
            }
            types::TypeKind::Struct => {
                let id = env.get_struct(t.location())?.id();

                Typed::Struct(id)
            }
            types::TypeKind::Void => Typed::Void,
        })
    }

    pub fn write_type_string(&self, s: &mut String, env: &Environment) {
        match self {
            Typed::Array(inner_type, rank) => {
                s.push_str("(ArrayType ");
                inner_type.write_type_string(s, env);
                write!(s, " {})", rank).expect("string should not fail to write");
            }
            Typed::Struct(id) => {
                s.push_str("(TupleType");
                let info = env.get_struct_id(*id);
                for (_, ty) in info.fields() {
                    s.push_str(" ");
                    ty.write_type_string(s, env);
                }

                //TODO remove, possible bug in tests
                if info.fields().is_empty() {
                    s.push(' ');
                }
                s.push(')');
            }
            Typed::Int => {
                s.push_str("(IntType)");
            }
            Typed::Bool => {
                s.push_str("(BoolType)");
            }
            Typed::Float => {
                s.push_str("(FloatType)");
            }
            Typed::Void => {
                s.push_str("(VoidType)");
            }
        }
    }

    pub fn to_type_string<'a>(&self, env: &Environment<'a>) -> Cow<'a, str> {
        match self {
            Typed::Array(inner_type, rank) => {
                let mut s = String::new();
                s.push_str("(ArrayType ");
                inner_type.write_type_string(&mut s, env);
                write!(s, " {})", rank).expect("string should not fail to write");
                Cow::Owned(s)
            }
            Typed::Struct(id) => {
                let mut s = String::new();
                s.push_str("(TupleType");
                let info = env.get_struct_id(*id);
                for (_, ty) in info.fields() {
                    s.push_str(" ");
                    ty.write_type_string(&mut s, env);
                }

                //TODO remove, possible bug in tests
                if info.fields().is_empty() {
                    s.push(' ');
                }
                s.push(')');
                Cow::Owned(s)
            }
            Typed::Int => Cow::Borrowed("(IntType)"),
            Typed::Bool => Cow::Borrowed("(BoolType)"),
            Typed::Float => Cow::Borrowed("(FloatType)"),
            Typed::Void => Cow::Borrowed("(VoidType)"),
        }
    }
}
