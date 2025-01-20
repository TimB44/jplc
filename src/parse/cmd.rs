use miette::Result;

use crate::{lex::TokenType, utils::Span};

use super::{
    auxiliary::{LValue, Str},
    exrp::Expr,
    tokens_match, Parse, TokenStream,
};

pub struct Cmd {
    kind: CmdKind,
    location: Span,
}

pub enum CmdKind {
    ReadImage(Str, LValue),
    WriteImage(Expr, Str),
    Let(LValue, Expr),
    Assert(Expr, Str),
    Print(Str),
    Show(Expr),
    Time(Box<Cmd>),
}

impl Cmd {
    fn parse_read_image(ts: &mut TokenStream) -> Result<Self> {
        let [read_token, _] = tokens_match(ts, [TokenType::Read, TokenType::Image])?;
        let str = Str::parse(ts)?;
        _ = tokens_match(ts, [TokenType::To])?;
        let lvalue = LValue::parse(ts)?;
        let location = read_token.span().join(&str.location());
        Ok(Self {
            kind: CmdKind::ReadImage(str, lvalue),
            location,
        })
    }

    fn parse_write_image(ts: &mut TokenStream) -> Result<Self> {
        let [write_token, _] = tokens_match(ts, [TokenType::Write, TokenType::Image])?;
        let expr = Expr::parse(ts)?;
        _ = tokens_match(ts, [TokenType::To])?;
        let str = Str::parse(ts)?;
        let location = write_token.span().join(&str.location());
        Ok(Self {
            kind: CmdKind::WriteImage(expr, str),
            location,
        })
    }

    fn parse_let(ts: &mut TokenStream) -> Result<Self> {
        let [let_token] = tokens_match(ts, [TokenType::Let])?;
        let lvalue = LValue::parse(ts)?;
        _ = tokens_match(ts, [TokenType::Equals])?;
        let expr = Expr::parse(ts)?;

        let location = let_token.span().join(&expr.location());
        Ok(Self {
            kind: CmdKind::Let(lvalue, expr),
            location,
        })
    }

    fn parse_assert(ts: &mut TokenStream) -> Result<Self> {
        let [assert_token] = tokens_match(ts, [TokenType::Assert])?;
        let expr = Expr::parse(ts)?;
        let str = Str::parse(ts)?;
        let location = assert_token.span().join(&str.location());

        Ok(Self {
            kind: CmdKind::Assert(expr, str),
            location,
        })
    }

    fn parse_print(ts: &mut TokenStream) -> Result<Self> {
        let [print_token] = tokens_match(ts, [TokenType::Print])?;
        debug_assert!(matches!(print_token.kind(), TokenType::Print));
        let str = Str::parse(ts)?;
        let location = print_token.span().join(&str.location());
        Ok(Self {
            kind: CmdKind::Print(str),
            location,
        })
    }

    fn parse_show(ts: &mut TokenStream) -> Result<Self> {
        let [show_token] = tokens_match(ts, [TokenType::Show])?;
        let expr = Expr::parse(ts)?;
        let location = expr.location().join(&show_token.span());
        Ok(Self {
            kind: CmdKind::Show(expr),
            location,
        })
    }

    fn parse_time(ts: &mut TokenStream) -> Result<Self> {
        let [time_token] = tokens_match(ts, [TokenType::Time])?;
        let cmd = Self::parse(ts)?;
        let location = cmd.location.join(&time_token.span());
        Ok(Self {
            kind: CmdKind::Time(Box::new(cmd)),
            location,
        })
    }

    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        match &self.kind {
            CmdKind::ReadImage(str, lvalue) => {
                format!(
                    "(ReadCmd {} {})",
                    str.location().as_str(src),
                    lvalue.to_s_expresion(src)
                )
            }
            CmdKind::WriteImage(expr, str) => {
                format!(
                    "(WriteCmd {} {})",
                    expr.to_s_expresion(src),
                    str.location().as_str(src)
                )
            }
            CmdKind::Let(lvalue, expr) => format!(
                "(LetCmd {} {})",
                lvalue.to_s_expresion(src),
                expr.to_s_expresion(src)
            ),
            CmdKind::Assert(expr, str) => {
                format!(
                    "(AssertCmd {} {})",
                    expr.to_s_expresion(src),
                    str.location().as_str(src)
                )
            }
            CmdKind::Print(str) => format!("(PrintCmd {})", str.location().as_str(src)),
            CmdKind::Show(expr) => format!("(ShowCmd {})", expr.to_s_expresion(src)),
            CmdKind::Time(cmd) => format!("(TimeCmd {})", cmd.to_s_expresion(src)),
        }
    }
}

/// Currently parses the following grammer
///
/// cmd  : read image <string> to <lvalue>
///      | write image <expr> to <string>
///      | let <lvalue> = <expr>
///      | assert <expr> , <string>
///      | print <string>
///      | show <expr>
///      | time <cmd>
impl Parse for Cmd {
    fn parse(ts: &mut TokenStream) -> Result<Self> {
        let next_token = ts.peek();
        match next_token.map(|t| t.kind()) {
            Some(TokenType::Read) => Self::parse_read_image(ts),
            Some(TokenType::Write) => Self::parse_write_image(ts),
            Some(TokenType::Let) => Self::parse_let(ts),
            Some(TokenType::Assert) => Self::parse_assert(ts),
            Some(TokenType::Print) => Self::parse_print(ts),
            Some(TokenType::Show) => Self::parse_show(ts),
            Some(TokenType::Time) => Self::parse_time(ts),

            _ => todo!("What do i return None or error"),
        }
    }
}
