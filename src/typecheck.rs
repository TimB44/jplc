use std::collections::HashMap;

use miette::miette;

use crate::{
    ast::{auxiliary::Binding, types},
    utils::Span,
};

struct StructInfo<'a> {
    fields: Box<[(&'a str, Type<'a>)]>,
    return_type: Type<'a>,
}

pub struct Environment<'a> {
    src: &'a [u8],
    structs: HashMap<&'a str, StructInfo<'a>>,
}

impl<'a> Environment<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            structs: HashMap::from([(
                "rgba",
                StructInfo {
                    fields: [
                        ("r", Type::Float),
                        ("g", Type::Float),
                        ("b", Type::Float),
                        ("a", Type::Float),
                    ]
                    .to_vec()
                    .into_boxed_slice(),

                    return_type: Type::Struct("rgba"),
                },
            )]),
        }
    }

    pub fn add_struct(
        &self,
        name: &Span,
        params: &[Binding],
        return_type: types::Type,
    ) -> miette::Result<()> {
        todo!()
    }

    pub fn src(&self) -> &[u8] {
        self.src
    }
}

pub trait Typecheck {
    fn check(&self, env: &mut Environment) -> miette::Result<()>;
}

pub trait GetType {
    fn get_type(&self, env: &Environment) -> miette::Result<Type>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type<'a> {
    Int,
    Bool,
    Float,
    // Type of the array and the rank
    Array(Box<Type<'a>>, u8),
    Struct(&'a str),
    Void,
}

impl<'a> Type<'a> {
    pub fn from_ast_type(t: &types::Type, src: &'a [u8]) -> Type<'a> {
        match t.kind() {
            types::TypeKind::Int => Type::Int,
            types::TypeKind::Bool => Type::Bool,
            types::TypeKind::Float => Type::Float,
            types::TypeKind::Array(inner, rank) => {
                Type::Array(Box::new(Self::from_ast_type(inner, src)), *rank)
            }
            types::TypeKind::Struct => Type::Struct(t.location().as_str(src)),
            types::TypeKind::Void => Type::Void,
        }
    }
}

fn expect_type<T: GetType>(node: &T, env: &Environment, expeted: Type) -> miette::Result<()> {
    let actual = node.get_type(env)?;
    if actual != expeted {
        Err(miette!("Type mismatch"))
    } else {
        Ok(())
    }
}
