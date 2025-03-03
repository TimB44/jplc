use crate::ast::types::{self, Type};
use crate::environment::Environment;

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
}

impl Typed {
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
}
