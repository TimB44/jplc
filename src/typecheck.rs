use std::collections::HashMap;

use miette::miette;

use crate::{parse::Program, utils::Span};

const BUILTIN_STRUCTS: [(&str, &[(&str, Type)]); 1] = [(
    "rgba",
    &[
        ("r", Type::Float),
        ("g", Type::Float),
        ("b", Type::Float),
        ("a", Type::Float),
    ],
)];

struct Environment<'a> {
    src: &'a [u8],
    structs: HashMap<&'a str, &'a [(&'a str, Type)]>,
}

impl<'a> Environment<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            structs: HashMap::from(BUILTIN_STRUCTS),
        }
    }
}

trait Typecheck {
    fn check(&self, env: &mut Environment) -> miette::Result<()>;
}

trait GetType {
    fn get_type(&self, env: &Environment) -> miette::Result<Type>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Bool,
    Float,
    // Type of the array and the rank
    Array(Box<Type>, u8),
    // name of struct infered from location
    Struct(Span),
    Void,
}

fn expect_type<T: GetType>(node: &T, env: &Environment, expeted: Type) -> miette::Result<()> {
    let actual = node.get_type(env)?;
    if actual != expeted {
        Err(miette!("Type mismatch"))
    } else {
        Ok(())
    }
}

impl Typecheck for Program {
    fn check(&self, env: &mut Environment) -> miette::Result<()> {
        todo!()
    }
}
