//! Defines the types to represent statments in JPL, and the functions to parse them
use crate::{
    environment::Environment,
    lex::TokenType,
    parse::{Displayable, SExpr},
    typecheck::TypeVal,
    utils::Span,
};
use miette::{miette, LabeledSpan, Severity};

use super::{
    auxiliary::{LValue, Str},
    expect_tokens,
    expr::Expr,
    Parse, TokenStream,
};

#[derive(Debug, Clone)]
pub struct Stmt {
    loc: Span,
    kind: StmtType,
}

#[derive(Debug, Clone)]
pub enum StmtType {
    Let(LValue, Expr),
    Assert(Expr, Str),
    Return(Expr),
}

/// Current grammar
/// stmt : let <lvalue> = <expr>
///      | assert <expr> , <string>
///      | return <expr>
impl Parse for Stmt {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        match ts.peek_type() {
            Some(TokenType::Let) => Self::parse_let(ts, env),
            Some(TokenType::Assert) => Self::parse_assert(ts, env),
            Some(TokenType::Return) => Self::parse_return(ts, env),
            Some(t) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected statment, found: {}", t)),
                    ts.peek().unwrap().loc().start(),
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
    fn parse_let(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [let_token] = expect_tokens(ts, [TokenType::Let])?;
        let lvalue = LValue::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Equals])?;
        let expr = Expr::parse(ts, env)?;
        let location = expr.loc().join(let_token.loc());

        env.add_lvalue(&lvalue, expr.type_data().clone())?;
        if let Some(bindings) = lvalue.array_bindings() {
            expr.expect_array_of_rank(bindings.len(), env)?;
        }

        Ok(Self {
            loc: location,
            kind: StmtType::Let(lvalue, expr),
        })
    }

    fn parse_assert(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [assert_token] = expect_tokens(ts, [TokenType::Assert])?;
        let expr = Expr::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Comma])?;
        let str_lit = Str::parse(ts, env)?;
        let loc = str_lit.loc().join(assert_token.loc());

        expr.expect_type(&TypeVal::Bool, env)?;

        Ok(Self {
            loc,
            kind: StmtType::Assert(expr, str_lit),
        })
    }
    fn parse_return(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [return_token] = expect_tokens(ts, [TokenType::Return])?;

        // The struct command checks that this is the correct type
        let expr = Expr::parse(ts, env)?;
        let location = expr.loc().join(return_token.loc());

        Ok(Self {
            loc: location,
            kind: StmtType::Return(expr),
        })
    }

    pub fn loc(&self) -> Span {
        self.loc
    }

    pub fn kind(&self) -> &StmtType {
        &self.kind
    }
}

impl SExpr for Stmt {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        match &self.kind {
            StmtType::Let(lvalue, expr) => {
                write!(
                    f,
                    "(LetStmt {} {})",
                    Displayable(lvalue, env, opt),
                    Displayable(expr, env, opt),
                )
            }
            StmtType::Assert(expr, str) => write!(
                f,
                "(AssertStmt {} {})",
                Displayable(expr, env, opt),
                Displayable(str, env, opt)
            ),
            StmtType::Return(expr) => write!(f, "(ReturnStmt {})", Displayable(expr, env, opt)),
        }
    }
}
