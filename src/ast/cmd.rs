//! Parses Commands in JPL
use super::super::parse::parse_sequence;
use super::{
    auxiliary::{Binding, LValue, Str},
    expect_tokens,
    expr::Expr,
    parse_sequence_trailing,
    stmt::Stmt,
    types::Type,
    Parse, TokenStream,
};

use crate::environment::Environment;
use crate::typecheck::{TypeState, Typed, UnTyped};
use crate::{lex::TokenType, utils::Span};
use miette::{miette, LabeledSpan, Severity};

/// Represents a Command in JPL.
///
/// These are the top level items in a JPL source file
#[derive(Debug, Clone)]
pub struct Cmd<T: TypeState = UnTyped> {
    kind: CmdKind<T>,
    location: Span,
}

/// Enumerates the different types of Commands
#[derive(Debug, Clone)]
pub enum CmdKind<T: TypeState = UnTyped> {
    ReadImage(Str, LValue),
    WriteImage(Expr<T>, Str),
    Let(LValue, Expr<T>),
    Assert(Expr<T>, Str),
    Print(Str),
    Show(Expr<T>),
    Time(Box<Cmd<T>>),
    Function {
        name: Span,
        params: Box<[Binding]>,
        return_type: Type,
        body: Box<[Stmt<T>]>,
        scope: usize,
    },
    Struct {
        name: Span,
        fields: Box<[(Span, Type)]>,
    },
}

/// Currently parses the following grammar:
///
/// cmd : read image <string> to <lvalue>
///    | write image <expr> to <string>
///    | let <lvalue> = <expr>
///    | assert <expr> , <string>
///    | print <string>
///    | show <expr>
///    | time <cmd>
///    | fn <variable> ( <binding> , ... ) : <type> { ;
///          <stmt> ; ... ;
///      }
///    | struct <variable> { ;
///          <variable>: <type> ; ... ;
///      }
impl Parse<Cmd> for Cmd {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        match ts.peek_type() {
            Some(TokenType::Read) => Self::parse_read_image(ts),
            Some(TokenType::Write) => Self::parse_write_image(ts),
            Some(TokenType::Let) => Self::parse_let(ts),
            Some(TokenType::Assert) => Self::parse_assert(ts),
            Some(TokenType::Print) => Self::parse_print(ts),
            Some(TokenType::Show) => Self::parse_show(ts),
            Some(TokenType::Time) => Self::parse_time(ts),
            Some(TokenType::Fn) => Self::parse_function(ts),
            Some(TokenType::Struct) => Self::parse_struct(ts),

            Some(t) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected command, found: {}", t)),
                    ts.peek().unwrap().span().start(),
                    ts.peek().unwrap().bytes().len(),
                )],
                "Unexpected token found"
            )),
            None => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::at_offset(
                    ts.lexer().bytes().len() - 1,
                    "expected command"
                )],
                "Missing expected token"
            )),
        }
    }
}

impl Parse<(Span, Type)> for (Span, Type) {
    /// Parses <variable>: <type>
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token, _] = expect_tokens(ts, [TokenType::Variable, TokenType::Colon])?;
        let var_type = Type::parse(ts)?;
        Ok((var_token.span(), var_type))
    }
}

impl Cmd {
    fn parse_read_image(ts: &mut TokenStream) -> miette::Result<Cmd> {
        let [read_token, _] = expect_tokens(ts, [TokenType::Read, TokenType::Image])?;
        let str = Str::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::To])?;
        let lvalue = LValue::parse(ts)?;
        let location = read_token.span().join(&str.location());
        Ok(Self {
            kind: CmdKind::ReadImage(str, lvalue),
            location,
        })
    }

    fn parse_write_image(ts: &mut TokenStream) -> miette::Result<Self> {
        let [write_token, _] = expect_tokens(ts, [TokenType::Write, TokenType::Image])?;
        let expr = Expr::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::To])?;
        let str = Str::parse(ts)?;
        let location = write_token.span().join(&str.location());
        Ok(Self {
            kind: CmdKind::WriteImage(expr, str),
            location,
        })
    }

    fn parse_let(ts: &mut TokenStream) -> miette::Result<Self> {
        let [let_token] = expect_tokens(ts, [TokenType::Let])?;
        let lvalue = LValue::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::Equals])?;
        let expr = Expr::parse(ts)?;

        let location = let_token.span().join(&expr.location());
        Ok(Self {
            kind: CmdKind::Let(lvalue, expr),
            location,
        })
    }

    fn parse_assert(ts: &mut TokenStream) -> miette::Result<Self> {
        let [assert_token] = expect_tokens(ts, [TokenType::Assert])?;
        let expr = Expr::parse(ts)?;

        _ = expect_tokens(ts, [TokenType::Comma])?;
        let str = Str::parse(ts)?;
        let location = assert_token.span().join(&str.location());

        Ok(Self {
            kind: CmdKind::Assert(expr, str),
            location,
        })
    }

    fn parse_print(ts: &mut TokenStream) -> miette::Result<Self> {
        let [print_token] = expect_tokens(ts, [TokenType::Print])?;
        debug_assert!(matches!(print_token.kind(), TokenType::Print));
        let str = Str::parse(ts)?;
        let location = print_token.span().join(&str.location());
        Ok(Self {
            kind: CmdKind::Print(str),
            location,
        })
    }

    fn parse_show(ts: &mut TokenStream) -> miette::Result<Self> {
        let [show_token] = expect_tokens(ts, [TokenType::Show])?;
        let expr = Expr::parse(ts)?;
        let location = expr.location().join(&show_token.span());
        Ok(Self {
            kind: CmdKind::Show(expr),
            location,
        })
    }

    fn parse_time(ts: &mut TokenStream) -> miette::Result<Self> {
        let [time_token] = expect_tokens(ts, [TokenType::Time])?;
        let cmd = Self::parse(ts)?;
        let location = cmd.location.join(&time_token.span());
        Ok(Self {
            kind: CmdKind::Time(Box::new(cmd)),
            location,
        })
    }

    fn parse_function(ts: &mut TokenStream) -> miette::Result<Cmd> {
        let [fn_token] = expect_tokens(ts, [TokenType::Fn])?;
        let [name] = expect_tokens(ts, [TokenType::Variable])?;
        _ = expect_tokens(ts, [TokenType::LParen])?;
        let params = parse_sequence(ts, TokenType::Comma, TokenType::RParen)?;
        _ = expect_tokens(ts, [TokenType::RParen])?;
        _ = expect_tokens(ts, [TokenType::Colon])?;
        let return_type = Type::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::LCurly])?;
        _ = expect_tokens(ts, [TokenType::Newline])?;
        let body = parse_sequence_trailing(ts, TokenType::Newline, TokenType::RCurly)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RCurly])?;

        let location = fn_token.span().join(&r_curly_token.span());
        Ok(Self {
            kind: CmdKind::Function {
                name: name.span(),
                params,
                return_type,
                body,
                scope: 0,
            },
            location,
        })
    }

    fn parse_struct(ts: &mut TokenStream) -> miette::Result<Cmd> {
        let [struct_token] = expect_tokens(ts, [TokenType::Struct])?;
        let [name] = expect_tokens(ts, [TokenType::Variable])?;
        _ = expect_tokens(ts, [TokenType::LCurly])?;
        _ = expect_tokens(ts, [TokenType::Newline])?;
        let fields = parse_sequence_trailing(ts, TokenType::Newline, TokenType::RCurly)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RCurly])?;
        let location = struct_token.span().join(&r_curly_token.span());

        Ok(Self {
            kind: CmdKind::Struct {
                name: name.span(),
                fields,
            },
            location,
        })
    }

    pub fn to_s_expr(&self, src: &[u8]) -> String {
        self.to_s_expr_general(src, |expr| expr.to_s_expr(src), |stmt| stmt.to_s_expr(src))
    }

    pub fn typecheck(self, env: &mut Environment) -> miette::Result<Cmd<Typed>> {
        match self.kind {
            CmdKind::ReadImage(_, lvalue) => todo!(),
            CmdKind::WriteImage(expr, _) => todo!(),
            CmdKind::Let(lvalue, expr) => todo!(),
            CmdKind::Assert(expr, _) => todo!(),
            CmdKind::Print(_) => todo!(),
            CmdKind::Show(expr) => {
                let typed_expr = expr.typecheck(env)?;

                Ok(Cmd {
                    kind: CmdKind::Show(typed_expr),
                    location: self.location,
                })
            }
            CmdKind::Time(cmd) => todo!(),
            CmdKind::Function {
                name,
                params,
                return_type,
                body,
                scope,
            } => todo!(),
            CmdKind::Struct { name, fields } => {
                env.add_struct(name, &*fields)?;

                Ok(Cmd {
                    kind: CmdKind::Struct { name, fields },
                    location: self.location,
                })
            }
        }
    }
}

impl Cmd<Typed> {
    pub fn to_typed_s_exprsision(&self, env: &Environment) -> String {
        self.to_s_expr_general(
            env.src(),
            |expr| expr.to_typed_s_exprsision(env),
            |stmt| stmt.to_typed_s_expr(env),
        )
    }
}

impl<T: TypeState> Cmd<T> {
    fn to_s_expr_general(
        &self,
        src: &[u8],
        expr_printer: impl Fn(&Expr<T>) -> String,
        stmt_printer: impl Fn(&Stmt<T>) -> String,
    ) -> String {
        match &self.kind {
            CmdKind::ReadImage(str, lvalue) => {
                format!(
                    "(ReadCmd {} {})",
                    str.location().as_str(src),
                    lvalue.to_s_expr(src)
                )
            }
            CmdKind::WriteImage(expr, str) => {
                format!(
                    "(WriteCmd {} {})",
                    expr_printer(expr),
                    str.location().as_str(src)
                )
            }
            CmdKind::Let(lvalue, expr) => {
                format!("(LetCmd {} {})", lvalue.to_s_expr(src), expr_printer(expr))
            }
            CmdKind::Assert(expr, str) => {
                format!(
                    "(AssertCmd {} {})",
                    expr_printer(&expr),
                    str.location().as_str(src)
                )
            }
            CmdKind::Print(str) => format!("(PrintCmd {})", str.location().as_str(src)),
            CmdKind::Show(expr) => format!("(ShowCmd {})", expr_printer(expr)),
            CmdKind::Time(cmd) => format!(
                "(TimeCmd {})",
                cmd.to_s_expr_general(src, expr_printer, stmt_printer)
            ),
            CmdKind::Function {
                name,
                params,
                return_type,
                body,
                ..
            } => {
                let mut s_expr = format!("(FnCmd {} ((", name.as_str(src));
                let mut params_iter = params.iter();
                let last = params_iter.next_back();
                for param in params_iter {
                    s_expr.push_str(&param.to_s_expr(src));
                    s_expr.push(' ');
                }
                if let Some(last) = last {
                    s_expr.push_str(&last.to_s_expr(src));
                }
                s_expr.push_str(")) ");
                s_expr.push_str(&return_type.to_s_expr(src));
                for stmt in body {
                    s_expr.push(' ');
                    s_expr.push_str(&stmt_printer(stmt));
                }
                s_expr.push(')');

                s_expr
            }

            CmdKind::Struct { name, fields } => {
                let mut s_expr = format!("(StructCmd {}", name.as_str(src));

                for (name, field_type) in fields {
                    s_expr.push(' ');
                    s_expr.push_str(name.as_str(src));
                    s_expr.push(' ');
                    s_expr.push_str(&field_type.to_s_expr(src));
                }
                s_expr.push(')');
                s_expr
            }
        }
    }
}
