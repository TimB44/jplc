use crate::{lex::TokenType, utils::Span};

use super::{expect_tokens, TokenStream};

/// Reprsents a string literal in the source code.
#[derive(Debug, Clone)]
pub struct Str {
    // Includes the opening and closing quote
    location: Span,
}

impl Str {
    pub fn location(&self) -> Span {
        self.location
    }

    pub fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [str_token] = expect_tokens(ts, [TokenType::StringLit])?;

        Ok(Self {
            location: str_token.span(),
        })
    }
}

/// Represents an left value used in let statements and commands
#[derive(Debug, Clone)]
pub struct LValue {
    location: Span,
}

impl LValue {
    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        format!("(VarLValue {})", self.location.as_str(src))
    }

    pub fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;

        Ok(Self {
            location: var_token.span(),
        })
    }
}

//TODO: may need later, removed from this assignment
//pub struct Argument {
//    location: Span,
//}
