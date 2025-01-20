use crate::{lex::TokenType, utils::Span};
use miette::{miette, LabeledSpan, Severity};

use super::{next_matches, tokens_match, Parse, TokenStream};

//TODO: for now we allow numbers one over max for positve int literals as we do not yet know if a
//number is positive or negative. We should either raise the min integer value or we need to do
//some static evaluation of expresions to deterimine overflow
const NEGATIVE_INT_LIT_MAX: u64 = 9223372036854775808;

pub struct Expr {
    location: Span,
    kind: ExprKind,
}

impl Expr {
    fn parse_int_lit(ts: &mut TokenStream) -> miette::Result<Self> {
        let [int_lit_token] = tokens_match(ts, [TokenType::IntLit])?;
        let int_val: u64 = match int_lit_token.bytes().parse() {
            Ok(i) if i <= NEGATIVE_INT_LIT_MAX => i,
            _ => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some("Invalid integer literal".to_string()),
                        int_lit_token.start(),
                        int_lit_token.bytes().len()
                    )],
                    "Integer literal outside signed 64 bit range"
                ))
            }
        };

        Ok(Self {
            location: int_lit_token.span(),
            kind: ExprKind::IntLit(int_val),
        })
    }

    fn parse_float_lit(ts: &mut TokenStream) -> miette::Result<Self> {
        let [float_lit_token] = tokens_match(ts, [TokenType::FloatLit])?;
        let float_val = match float_lit_token.bytes().parse::<f64>() {
            Ok(f) if f.is_finite() => f,
            _ => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some("Invalid float literal".to_string()),
                        float_lit_token.start(),
                        float_lit_token.bytes().len()
                    )],
                    "Float literal outside signed 64 bit range"
                ))
            }
        };

        Ok(Self {
            location: float_lit_token.span(),
            kind: ExprKind::FloatLit(float_val),
        })
    }

    fn parse_true(ts: &mut TokenStream) -> miette::Result<Self> {
        let [true_token] = tokens_match(ts, [TokenType::True])?;
        Ok(Self {
            location: true_token.span(),
            kind: ExprKind::True,
        })
    }

    fn parse_false(ts: &mut TokenStream) -> miette::Result<Self> {
        let [false_token] = tokens_match(ts, [TokenType::False])?;
        Ok(Self {
            location: false_token.span(),
            kind: ExprKind::False,
        })
    }

    fn parse_var(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = tokens_match(ts, [TokenType::Variable])?;
        Ok(Self {
            location: var_token.span(),
            kind: ExprKind::Var,
        })
    }

    fn parse_array_lit(ts: &mut TokenStream) -> miette::Result<Self> {
        let [lb_token] = tokens_match(ts, [TokenType::LSquare])?;
        let mut items = Vec::new();
        if !next_matches!(ts, TokenType::RSquare) {
            loop {
                items.push(Expr::parse(ts)?);
                if next_matches!(ts, TokenType::RSquare) {
                    break;
                }
                _ = tokens_match(ts, [TokenType::Comma])?;
            }
        }

        let items = items.into_boxed_slice();
        let [rb_token] = tokens_match(ts, [TokenType::RSquare])?;
        Ok(Self {
            location: lb_token.span().join(&rb_token.span()),
            kind: ExprKind::ArrayLit(items),
        })
    }

    pub fn to_s_expresion(&self, src: &[u8]) -> String {
        match &self.kind {
            ExprKind::IntLit(val) => format!("(IntExpr {})", val),
            ExprKind::FloatLit(val) => format!("(FloatExpr {:.0})", val.trunc()),
            ExprKind::True => "(TrueExpr)".to_string(),
            ExprKind::False => "(FalseExpr)".to_string(),
            ExprKind::Var => format!("(VarExpr {})", self.location.as_str(src)),
            ExprKind::ArrayLit(items) => {
                let mut s_expr = "(ArrayLiteralExpr".to_string();
                for item in items {
                    s_expr.push(' ');
                    s_expr.push_str(&item.to_s_expresion(src));
                }
                s_expr.push(')');
                s_expr
            }
        }
    }
}

impl Expr {
    pub fn location(&self) -> Span {
        self.location
    }
}

pub enum ExprKind {
    IntLit(u64),
    FloatLit(f64),
    True,
    False,
    Var,
    ArrayLit(Box<[Expr]>),
}

impl Parse for Expr {
    fn parse(ts: &mut super::TokenStream) -> miette::Result<Self> {
        match ts.peek().map(|t| t.kind()) {
            Some(TokenType::IntLit) => Self::parse_int_lit(ts),
            Some(TokenType::FloatLit) => Self::parse_float_lit(ts),
            Some(TokenType::True) => Self::parse_true(ts),
            Some(TokenType::False) => Self::parse_false(ts),
            Some(TokenType::Variable) => Self::parse_var(ts),
            Some(TokenType::LSquare) => Self::parse_array_lit(ts),
            Some(t) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected expresion, found: {}", t)),
                    ts.peek().unwrap().span().start(),
                    ts.peek().unwrap().bytes().len(),
                )],
                "Unexpected token found"
            )),
            None => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::at_offset(
                    ts.lexer().bytes().len() - 1,
                    "expected expression"
                )],
                "Missing expected token"
            )),
        }
    }
}
