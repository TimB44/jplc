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

//TODO: may need later removed from this assignment
//pub struct Argument {
//    location: Span,
//}

pub struct LValue {
    location: Span,
}

impl LValue {
    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        format!("(VarLValue {})", self.location.as_str(src))
    }
}

impl Parse for LValue {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = tokens_match(ts, [TokenType::Variable])?;

        Ok(Self {
            location: var_token.span(),
        })
    }
}
