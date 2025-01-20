use crate::{lex::TokenType, utils::Span};

use super::{tokens_match, Parse, TokenStream};

/// Reprsents a string literal in the source code.
pub struct Str {
    // This includes the opening and closing quote
    location: Span,
}

impl Str {
    pub fn location(&self) -> Span {
        self.location
    }
}

impl Parse for Str {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [str_token] = tokens_match(ts, [TokenType::StringLit])?;

        Ok(Self {
            location: str_token.span(),
        })
    }
}

pub struct Argument {
    location: Span,
}

pub struct LValue {
    location: Span,
}
impl Parse for LValue {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = tokens_match(ts, [TokenType::Variable])?;

        Ok(Self {
            location: var_token.span(),
        })
    }
}
