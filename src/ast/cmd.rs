//! Parses Commands in JPL
use std::fmt::{self, Formatter, Write};

use super::super::parse::parse_sequence;
use super::auxiliary::{StructField, Var};
use super::stmt::StmtType;
use super::{
    auxiliary::{Binding, LValue, Str},
    expect_tokens,
    expr::Expr,
    parse_sequence_trailing,
    stmt::Stmt,
    types::Type,
    Parse, TokenStream,
};
use crate::environment::builtins::IMAGE_TYPE;
use crate::environment::Environment;
use crate::lex::TokenType;
use crate::parse::{Displayable, SExpr, SExprOptions};
use crate::typecheck::TypeVal;
use crate::utils::Span;
use miette::{miette, LabeledSpan, Severity};

/// Represents a Command in JPL.
///
/// These are the top level items in a JPL source file
#[derive(Debug, Clone)]
pub struct Cmd {
    kind: CmdKind,
    loc: Span,
}

/// Enumerates the different types of Commands
#[derive(Debug, Clone)]
pub enum CmdKind {
    ReadImage(Str, LValue),
    WriteImage(Expr, Str),
    Let(LValue, Expr),
    Assert(Expr, Str),
    Print(Str),
    Show(Expr),
    Time(Box<Cmd>),
    Function {
        name: Var,
        params: Box<[Binding]>,
        return_type: Type,
        body: Box<[Stmt]>,
        scope: usize,
    },
    Struct {
        //TODO: use var here instead
        name: Var,
        fields: Box<[StructField]>,
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
impl Parse for Cmd {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        match ts.peek_type() {
            Some(TokenType::Read) => Self::parse_read_image(ts, env),
            Some(TokenType::Write) => Self::parse_write_image(ts, env),
            Some(TokenType::Let) => Self::parse_let(ts, env),
            Some(TokenType::Assert) => Self::parse_assert(ts, env),
            Some(TokenType::Print) => Self::parse_print(ts, env),
            Some(TokenType::Show) => Self::parse_show(ts, env),
            Some(TokenType::Time) => Self::parse_time(ts, env),
            Some(TokenType::Fn) => Self::parse_function(ts, env),
            Some(TokenType::Struct) => Self::parse_struct(ts, env),
            Some(t) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected command, found: {}", t)),
                    ts.peek().unwrap().loc().start(),
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

impl Cmd {
    fn parse_read_image(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Cmd> {
        let [read_token, _] = expect_tokens(ts, [TokenType::Read, TokenType::Image])?;
        let str = Str::parse(ts, env)?;
        expect_tokens(ts, [TokenType::To])?;
        let lvalue = LValue::parse(ts, env)?;
        let loc = read_token.loc().join(str.loc());
        env.add_lvalue(&lvalue, IMAGE_TYPE.clone())?;
        let number_of_bindings = lvalue.array_bindings().map(|b| b.len()).unwrap_or(2);
        if number_of_bindings != 2 {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected 2 bindings found {}", number_of_bindings)),
                    lvalue.loc().start(),
                    lvalue.loc().len(),
                )],
                "Unexpected token found"
            ));
        }
        Ok(Self {
            kind: CmdKind::ReadImage(str, lvalue),
            loc,
        })
    }

    fn parse_write_image(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [write_token, _] = expect_tokens(ts, [TokenType::Write, TokenType::Image])?;
        let expr = Expr::parse(ts, env)?;
        expect_tokens(ts, [TokenType::To])?;
        let str = Str::parse(ts, env)?;
        let location = write_token.loc().join(str.loc());
        expr.expect_type(&IMAGE_TYPE, env)?;
        Ok(Self {
            kind: CmdKind::WriteImage(expr, str),
            loc: location,
        })
    }

    fn parse_let(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [let_token] = expect_tokens(ts, [TokenType::Let])?;
        let lvalue = LValue::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Equals])?;
        let expr = Expr::parse(ts, env)?;

        env.add_lvalue(&lvalue, expr.type_data().clone())?;
        if let Some(bindings) = lvalue.array_bindings() {
            expr.expect_array_of_rank(bindings.len(), env)?;
        }

        let loc = let_token.loc().join(expr.loc());
        Ok(Self {
            kind: CmdKind::Let(lvalue, expr),
            loc,
        })
    }

    fn parse_assert(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [assert_token] = expect_tokens(ts, [TokenType::Assert])?;
        let expr = Expr::parse(ts, env)?;

        expect_tokens(ts, [TokenType::Comma])?;
        let str = Str::parse(ts, env)?;
        let loc = assert_token.loc().join(str.loc());
        expr.expect_type(&TypeVal::Bool, env)?;

        Ok(Self {
            kind: CmdKind::Assert(expr, str),
            loc,
        })
    }

    fn parse_print(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [print_token] = expect_tokens(ts, [TokenType::Print])?;
        let str = Str::parse(ts, env)?;
        let loc = print_token.loc().join(str.loc());
        Ok(Self {
            kind: CmdKind::Print(str),
            loc,
        })
    }

    fn parse_show(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [show_token] = expect_tokens(ts, [TokenType::Show])?;
        let expr = Expr::parse(ts, env)?;
        let loc = show_token.loc().join(expr.loc());
        Ok(Self {
            kind: CmdKind::Show(expr),
            loc,
        })
    }

    fn parse_time(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [time_token] = expect_tokens(ts, [TokenType::Time])?;
        let cmd = Self::parse(ts, env)?;
        let location = cmd.loc.join(time_token.loc());
        Ok(Self {
            kind: CmdKind::Time(Box::new(cmd)),
            loc: location,
        })
    }

    fn parse_function(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Cmd> {
        let [fn_token] = expect_tokens(ts, [TokenType::Fn])?;
        let name = Var::parse(ts, env)?;
        expect_tokens(ts, [TokenType::LParen])?;
        let params = parse_sequence(ts, env, TokenType::Comma, TokenType::RParen)?;
        expect_tokens(ts, [TokenType::RParen, TokenType::Colon])?;
        let return_type = Type::parse(ts, env)?;
        expect_tokens(ts, [TokenType::LCurly, TokenType::Newline])?;

        let scope = env.add_function(name.loc(), &params, &return_type)?;
        let body: Box<[Stmt]> =
            parse_sequence_trailing(ts, env, TokenType::Newline, TokenType::RCurly)?;
        env.end_scope();

        let [r_curly_token] = expect_tokens(ts, [TokenType::RCurly])?;
        let loc = fn_token.loc().join(r_curly_token.loc());
        let ret_type = TypeVal::from_ast_type(&return_type, env)?;

        let mut found_ret = false;
        for stmt in &body {
            if let StmtType::Return(expr) = stmt.kind() {
                expr.expect_type(&ret_type, env)?;
                found_ret = true;
            }
        }

        if !found_ret && ret_type != TypeVal::Void {
            let last_stmt_span = body.last().map(|s: &Stmt| s.loc()).unwrap_or(loc);
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![
                    LabeledSpan::new(
                        Some(format!(
                            "return type: {} defined here",
                            ret_type.as_str(env)
                        )),
                        return_type.location().start(),
                        return_type.location().len(),
                    ),
                    LabeledSpan::new(
                        Some("return statment expected here".to_string()),
                        last_stmt_span.start(),
                        last_stmt_span.len()
                    ),
                ],
                "Missing return statment"
            ));
        }

        Ok(Self {
            kind: CmdKind::Function {
                name,
                params,
                return_type,
                body,
                scope,
            },
            loc,
        })
    }

    fn parse_struct(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Cmd> {
        let [struct_token] = expect_tokens(ts, [TokenType::Struct])?;
        let name = Var::parse(ts, env)?;
        expect_tokens(ts, [TokenType::LCurly, TokenType::Newline])?;
        let fields = parse_sequence_trailing(ts, env, TokenType::Newline, TokenType::RCurly)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RCurly])?;
        let loc = struct_token.loc().join(r_curly_token.loc());
        env.add_struct(name, &fields)?;
        Ok(Self {
            kind: CmdKind::Struct { name, fields },
            loc,
        })
    }

    pub fn kind(&self) -> &CmdKind {
        &self.kind
    }

    pub fn loc(&self) -> Span {
        self.loc
    }
}

impl SExpr for Cmd {
    fn to_s_expr(
        &self,
        f: &mut Formatter<'_>,
        env: &Environment<'_>,
        opt: SExprOptions,
    ) -> fmt::Result {
        match &self.kind {
            CmdKind::ReadImage(str, lvalue) => {
                write!(
                    f,
                    "(ReadCmd {} {})",
                    Displayable(str, env, opt),
                    Displayable(lvalue, env, opt),
                )
            }
            CmdKind::WriteImage(expr, str) => {
                write!(
                    f,
                    "(WriteCmd {} {})",
                    Displayable(expr, env, opt),
                    Displayable(str, env, opt),
                )
            }
            CmdKind::Let(lvalue, expr) => {
                write!(
                    f,
                    "(LetCmd {} {})",
                    Displayable(lvalue, env, opt),
                    Displayable(expr, env, opt),
                )
            }
            CmdKind::Assert(expr, str) => {
                write!(
                    f,
                    "(AssertCmd {} {})",
                    Displayable(expr, env, opt),
                    Displayable(str, env, opt),
                )
            }
            CmdKind::Print(str) => write!(f, "(PrintCmd {})", Displayable(str, env, opt),),
            CmdKind::Show(expr) => write!(f, "(ShowCmd {})", Displayable(expr, env, opt),),
            CmdKind::Time(cmd) => write!(f, "(TimeCmd {})", Displayable(cmd.as_ref(), env, opt),),
            CmdKind::Function {
                name,
                params,
                return_type,
                body,
                ..
            } => {
                write!(f, "(FnCmd {} ((", Displayable(name, env, opt),)?;
                let mut params_iter = params.iter();
                let last = params_iter.next_back();
                for param in params_iter {
                    write!(f, "{} ", Displayable(param, env, opt))?;
                }
                if let Some(last) = last {
                    write!(f, "{}", Displayable(last, env, opt))?;
                }
                write!(f, ")) {}", Displayable(return_type, env, opt))?;

                for stmt in body {
                    write!(f, " {}", Displayable(stmt, env, opt))?;
                }
                f.write_char(')')
            }

            CmdKind::Struct { name, fields } => {
                write!(f, "(StructCmd {}", Displayable(name, env, opt))?;

                for field in fields {
                    write!(f, " {}", Displayable(field, env, opt))?;
                }
                f.write_char(')')
            }
        }
    }
}
