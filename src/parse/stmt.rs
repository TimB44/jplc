//! Defines the types to represent statments in JPL, and the functions to parse them
use crate::{lex::TokenType, utils::Span};
use miette::{miette, LabeledSpan, Severity};

use super::{
    auxiliary::{LValue, Str},
    expect_tokens,
    exrp::Expr,
    Parse, TokenStream,
};

#[derive(Debug, Clone)]
//TODO: implement this
pub struct Stmt {
    location: Span,
    kind: StmtType,
}

#[derive(Debug, Clone)]
pub enum StmtType {
    Let(LValue, Expr),
    Assert(Expr, Str),
    Return(Expr),
}

impl Parse for Stmt {
    /// Current grammar
    /// stmt : let <lvalue> = <expr>
    ///      | assert <expr> , <string>
    ///      | return <expr>
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        match ts.peek_type() {
            Some(TokenType::Let) => Self::parse_let(ts),
            Some(TokenType::Assert) => Self::parse_assert(ts),
            Some(TokenType::Return) => Self::parse_return(ts),
            Some(t) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected statment, found: {}", t)),
                    ts.peek().unwrap().span().start(),
                    ts.peek().unwrap().bytes().len(),
                )],
                "Unexpected token found"
            )),
            None => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::at_offset(
                    ts.lexer().bytes().len() - 1,
                    "expected statment"
                )],
                "Missing expected token"
            )),
        }
    }
}

impl Stmt {
    fn parse_let(ts: &mut TokenStream) -> miette::Result<Self> {
        let [let_token] = expect_tokens(ts, [TokenType::Let])?;
        let l_value = LValue::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::Equals])?;
        let expr = Expr::parse(ts)?;
        let location = expr.location().join(&let_token.span());

        Ok(Self {
            location,
            kind: StmtType::Let(l_value, expr),
        })
    }
    fn parse_assert(ts: &mut TokenStream) -> miette::Result<Self> {
        let [assert_token] = expect_tokens(ts, [TokenType::Assert])?;
        let expr = Expr::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::Comma])?;
        let str_lit = Str::parse(ts)?;
        let location = str_lit.location().join(&assert_token.span());

        Ok(Self {
            location,
            kind: StmtType::Assert(expr, str_lit),
        })
    }
    fn parse_return(ts: &mut TokenStream) -> miette::Result<Self> {
        let [return_token] = expect_tokens(ts, [TokenType::Return])?;
        let expr = Expr::parse(ts)?;
        let location = expr.location().join(&return_token.span());

        Ok(Self {
            location,
            kind: StmtType::Return(expr),
        })
    }

    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        match &self.kind {
            StmtType::Let(lvalue, expr) => format!(
                "(LetStmt {} {})",
                lvalue.to_s_expresion(src),
                expr.to_s_expresion(src)
            ),
            StmtType::Assert(expr, str_lit) => format!(
                "(AssertStmt {} {})",
                expr.to_s_expresion(src),
                str_lit.location().as_str(src)
            ),
            StmtType::Return(expr) => format!("(ReturnStmt {})", expr.to_s_expresion(src)),
        }
    }
}
