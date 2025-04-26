//! This module defines the types and functions to used to represent types in
//! JPL and in the ast

use super::{next_match, Parse, TokenStream};
use crate::{
    environment::Environment,
    lex::TokenType,
    parse::{expect_tokens, parse_sequence, Displayable, SExpr},
    utils::Span,
};
use miette::{miette, LabeledSpan, Severity};

/// Represents a type in a JPL program. Currently uses the following grammer
/// type: <simple> <cont>
/// simple : int
///        | bool
///        | float
///        | <type>
///        | <variable>
///        | void
///
/// cont : [ , ... ] cont
///      | <empty>
#[derive(Debug, Clone)]
pub struct Type {
    kind: TypeKind,
    loc: Span,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    Int,
    Bool,
    Float,
    // Type of the array and the rank
    Array(Box<Type>, u8),
    // name of struct infered from location
    Struct,
    Void,

    FnPointer(Box<[Type]>, Box<Type>),
}

impl Parse for Type {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let mut current_type = match ts.peek_type() {
            Some(TokenType::Int) => Self::parse_int(ts)?,
            Some(TokenType::Bool) => Self::parse_bool(ts)?,
            Some(TokenType::Float) => Self::parse_float(ts)?,
            Some(TokenType::Variable) => Self::parse_struct(ts)?,
            Some(TokenType::Void) => Self::parse_void(ts)?,
            Some(TokenType::Fn) => Self::parse_fn(ts, env)?,

            Some(t) => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some(format!("expected Type, found: {}", t)),
                        ts.peek().unwrap().loc().start(),
                        ts.peek().unwrap().bytes().len(),
                    )],
                    "Unexpected token found"
                ))
            }
            None => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::at_offset(
                        ts.lexer().bytes().len() - 1,
                        "expected type"
                    )],
                    "Missing expected token"
                ))
            }
        };

        // Parse array type modifiers
        while next_match!(ts, TokenType::LSquare) {
            _ = expect_tokens(ts, [TokenType::LSquare])?;
            let mut rank = 1;
            while next_match!(ts, TokenType::Comma) {
                rank += 1;
                _ = expect_tokens(ts, [TokenType::Comma])?;
            }
            let [r_square] = expect_tokens(ts, [TokenType::RSquare])?;
            let location = r_square.loc().join(current_type.loc);
            current_type = Self {
                kind: TypeKind::Array(Box::new(current_type), rank),
                loc: location,
            }
        }

        Ok(current_type)
    }
}
impl Type {
    fn parse_int(ts: &mut TokenStream) -> miette::Result<Self> {
        let [int_token] = expect_tokens(ts, [TokenType::Int])?;
        Ok(Self {
            loc: int_token.loc(),
            kind: TypeKind::Int,
        })
    }
    fn parse_bool(ts: &mut TokenStream) -> miette::Result<Self> {
        let [bool_token] = expect_tokens(ts, [TokenType::Bool])?;
        Ok(Self {
            loc: bool_token.loc(),
            kind: TypeKind::Bool,
        })
    }
    fn parse_float(ts: &mut TokenStream) -> miette::Result<Self> {
        let [float_token] = expect_tokens(ts, [TokenType::Float])?;
        Ok(Self {
            loc: float_token.loc(),
            kind: TypeKind::Float,
        })
    }

    fn parse_struct(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        Ok(Self {
            loc: var_token.loc(),
            kind: TypeKind::Struct,
        })
    }
    fn parse_void(ts: &mut TokenStream) -> miette::Result<Self> {
        let [void_token] = expect_tokens(ts, [TokenType::Void])?;
        Ok(Self {
            loc: void_token.loc(),
            kind: TypeKind::Void,
        })
    }

    fn parse_fn(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [fn_token, _] = expect_tokens(ts, [TokenType::Fn, TokenType::LParen])?;
        let args = parse_sequence(ts, env, TokenType::Comma, TokenType::RParen)?;
        let [closing_paren] = expect_tokens(ts, [TokenType::RParen])?;
        let ret_type = if ts.peek_type() == Some(TokenType::Arrow) {
            expect_tokens(ts, [TokenType::Arrow]);
            Type::parse(ts, env)?
        } else {
            Type {
                kind: TypeKind::Void,
                loc: closing_paren.loc(),
            }
        };
        let loc = fn_token.loc().join(ret_type.loc());

        Ok(Type {
            kind: TypeKind::FnPointer(args, Box::new(ret_type)),
            loc,
        })
    }

    pub fn loc(&self) -> Span {
        self.loc
    }

    pub fn kind(&self) -> &TypeKind {
        &self.kind
    }
}

impl SExpr for Type {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &crate::environment::Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        match &self.kind {
            TypeKind::Int => write!(f, "(IntType)"),
            TypeKind::Bool => write!(f, "(BoolType)"),
            TypeKind::Float => write!(f, "(FloatType)"),
            TypeKind::Struct => write!(f, "(StructType {})", self.loc.as_str(env.src())),
            TypeKind::Void => write!(f, "(VoidType)"),
            TypeKind::Array(element_type, rank) => {
                write!(
                    f,
                    "(ArrayType {} {})",
                    Displayable(element_type.as_ref(), env, opt),
                    rank
                )
            }
            TypeKind::FnPointer(items, ret_type) => {
                write!(
                    f,
                    "(FunctionPointerType ({}) {})",
                    Displayable(items, env, opt),
                    Displayable(ret_type.as_ref(), env, opt),
                )
            }
        }
    }
}
