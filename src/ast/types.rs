//! This module defines the types and functions to used to represent types in
//! JPL and in the ast

use super::{next_match, Parse};
use crate::{lex::TokenType, parse::expect_tokens, utils::Span};
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
    location: Span,
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
}

impl Parse<Type> for Type {
    fn parse(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let mut current_type = match ts.peek_type() {
            Some(TokenType::Int) => Self::parse_int(ts)?,
            Some(TokenType::Bool) => Self::parse_bool(ts)?,
            Some(TokenType::Float) => Self::parse_float(ts)?,
            Some(TokenType::Variable) => Self::parse_struct(ts)?,
            Some(TokenType::Void) => Self::parse_void(ts)?,

            Some(t) => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some(format!("expected Expression, found: {}", t)),
                        ts.peek().unwrap().span().start(),
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
                        "expected expression"
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
            let location = r_square.span().join(&current_type.location);
            current_type = Self {
                kind: TypeKind::Array(Box::new(current_type), rank),
                location,
            }
        }

        Ok(current_type)
    }
}
impl Type {
    fn parse_int(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let [int_token] = expect_tokens(ts, [TokenType::Int])?;
        Ok(Self {
            location: int_token.span(),
            kind: TypeKind::Int,
        })
    }
    fn parse_bool(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let [bool_token] = expect_tokens(ts, [TokenType::Bool])?;
        Ok(Self {
            location: bool_token.span(),
            kind: TypeKind::Bool,
        })
    }
    fn parse_float(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let [float_token] = expect_tokens(ts, [TokenType::Float])?;
        Ok(Self {
            location: float_token.span(),
            kind: TypeKind::Float,
        })
    }

    fn parse_struct(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        Ok(Self {
            location: var_token.span(),
            kind: TypeKind::Struct,
        })
    }
    fn parse_void(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let [void_token] = expect_tokens(ts, [TokenType::Void])?;
        Ok(Self {
            location: void_token.span(),
            kind: TypeKind::Void,
        })
    }

    pub fn location(&self) -> Span {
        self.location
    }

    pub fn to_s_expr(&self, src: &[u8]) -> String {
        match &self.kind {
            TypeKind::Int => "(IntType)".to_string(),
            TypeKind::Bool => "(BoolType)".to_string(),
            TypeKind::Float => "(FloatType)".to_string(),
            TypeKind::Struct => format!("(StructType {})", self.location.as_str(src)),
            TypeKind::Void => "(VoidType)".to_string(),
            TypeKind::Array(expr, rank) => {
                format!("(ArrayType {} {})", expr.to_s_expr(src), rank)
            }
        }
    }

    pub fn kind(&self) -> &TypeKind {
        &self.kind
    }
}
