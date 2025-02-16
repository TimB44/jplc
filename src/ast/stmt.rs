//! Defines the types to represent statments in JPL, and the functions to parse them
use crate::{
    environment::Environment,
    lex::TokenType,
    typecheck::{TypeState, Typed, UnTyped},
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
pub struct Stmt<T: TypeState = UnTyped> {
    location: Span,
    kind: StmtType<T>,
}

#[derive(Debug, Clone)]
pub enum StmtType<T: TypeState = UnTyped> {
    Let(LValue, Expr<T>),
    Assert(Expr<T>, Str),
    Return(Expr<T>),
}

impl Parse<Stmt> for Stmt {
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

    pub fn to_s_expr(&self, src: &[u8]) -> String {
        match &self.kind {
            StmtType::Let(lvalue, expr) => format!(
                "(LetStmt {} {})",
                lvalue.to_s_expr(src),
                expr.to_s_expr(src)
            ),
            StmtType::Assert(expr, str_lit) => format!(
                "(AssertStmt {} {})",
                expr.to_s_expr(src),
                str_lit.location().as_str(src)
            ),
            StmtType::Return(expr) => format!("(ReturnStmt {})", expr.to_s_expr(src)),
        }
    }

    pub fn typecheck(self, env: &mut Environment, scope_id: usize) -> miette::Result<Stmt<Typed>> {
        match self.kind {
            StmtType::Let(lvalue, expr) => {
                let typed_expr = expr.typecheck(env, scope_id)?;
                env.add_lval(&lvalue, typed_expr.type_data().clone(), scope_id)?;

                Ok(Stmt {
                    kind: StmtType::Let(lvalue, typed_expr),
                    location: self.location,
                })
            }
            StmtType::Assert(expr, msg) => {
                let typed_expr = expr.typecheck(env, scope_id)?;
                typed_expr.expect_type(&Typed::Bool, env)?;

                Ok(Stmt {
                    kind: StmtType::Assert(typed_expr, msg),
                    location: self.location,
                })
            }
            StmtType::Return(expr) => Ok(Stmt {
                location: self.location,
                kind: StmtType::Return(expr.typecheck(env, scope_id)?),
            }),
        }
    }
}

impl<T: TypeState> Stmt<T> {
    fn to_s_expr_general(&self, src: &[u8], expr_printer: impl Fn(&Expr<T>) -> String) -> String {
        match &self.kind {
            StmtType::Let(lvalue, expr) => {
                format!("(LetStmt {} {})", lvalue.to_s_expr(src), expr_printer(expr))
            }
            StmtType::Assert(expr, str_lit) => format!(
                "(AssertStmt {} {})",
                expr_printer(expr),
                str_lit.location().as_str(src)
            ),
            StmtType::Return(expr) => format!("(ReturnStmt {})", expr_printer(expr)),
        }
    }

    pub fn kind(&self) -> &StmtType<T> {
        &self.kind
    }

    pub fn location(&self) -> Span {
        self.location
    }
}

impl Stmt<Typed> {
    pub fn to_typed_s_expr(&self, env: &Environment) -> String {
        self.to_s_expr_general(env.src(), |expr| expr.to_typed_s_exprsision(env))
    }
}
