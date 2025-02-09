use crate::{
    lex::TokenType,
    parse::{expect_tokens, next_match, parse_sequence, Parse, TokenStream},
    utils::Span,
};

use super::types::Type;

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
}
impl Parse for Str {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [str_token] = expect_tokens(ts, [TokenType::StringLit])?;

        Ok(Self {
            location: str_token.span(),
        })
    }
}

/// Span acts a a varible in this case
impl Parse for Span {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        Ok(var_token.span())
    }
}
/// Represents an left value used in let statements and commands
#[derive(Debug, Clone)]
pub struct LValue {
    location: Span,
    variable: Span,
    array_bindings: Option<Box<[Span]>>,
}

impl LValue {
    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        match &self.array_bindings {
            //(FnCmd x (((ArrayLValue y H) (IntType))) (VoidType))
            Some(array_bindings) => {
                let mut s_expr = format!("(ArrayLValue {}", self.variable.as_str(src));
                for binding in array_bindings {
                    s_expr.push(' ');
                    s_expr.push_str(binding.as_str(src));
                }
                s_expr.push(')');
                s_expr
            }
            None => format!("(VarLValue {})", self.variable.as_str(src)),
        }
    }
}
impl Parse for LValue {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        let mut location = var_token.span();
        let array_bindings = if next_match!(ts, TokenType::LSquare) {
            _ = expect_tokens(ts, [TokenType::LSquare])?;
            let bindings = parse_sequence(ts, TokenType::Comma, TokenType::RSquare)?;
            let [r_square_token] = expect_tokens(ts, [TokenType::RSquare])?;
            location = location.join(&r_square_token.span());
            Some(bindings)
        } else {
            None
        };

        Ok(Self {
            variable: var_token.span(),
            array_bindings,
            location,
        })
    }
}

/// binding : <lvalue> : <type>
#[derive(Debug, Clone)]
pub struct Binding {
    _location: Span,
    l_value: LValue,
    variable_type: Type,
}

impl Parse for Binding {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let l_value = LValue::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::Colon])?;
        let variable_type = Type::parse(ts)?;
        let location = l_value.location.join(&variable_type.location());

        Ok(Self {
            _location: location,
            l_value,
            variable_type,
        })
    }
}
impl Binding {
    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        format!(
            "{} {}",
            self.l_value.to_s_expresion(src),
            self.variable_type.to_s_expresion(src)
        )
    }
}
