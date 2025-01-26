//! Defines the types of functions to parse all kinds of expressions in JPL
use super::{expect_tokens, parse_sequence, Parse, TokenStream};
use crate::{lex::TokenType, utils::Span};
use miette::{miette, LabeledSpan, Severity};

//TODO: allow numbers one bigger due to negative numbers
const POSITIVE_INT_LIT_MAX: u64 = 9223372036854775807;

/// Represents an expression in JPL
#[derive(Debug, Clone)]
pub struct Expr {
    location: Span,
    kind: ExprKind,
}

/// Defines the different types of expressions possible in JPL
///
/// The current grammar is as follows.
///
/// expr : <simple expr><expr cont>
///
/// simple expr : <integer>
///             | <float>
///             | true
///             | false
///             | <variable>
///             | [ <expr> , ... ]
///             | variable> { <expr> , ... }
///             | <variable> ( <expr> , ... )
///             | ( <expr> )
///
/// expr cont : .<variable><expr cont>
///           | [ <expr> , ... ]<expr cont>
///           | <empty>
#[derive(Debug, Clone)]
pub enum ExprKind {
    IntLit(u64),
    FloatLit(f64),
    True,
    False,
    Var,
    ArrayLit(Box<[Expr]>),
    StructInit(Span, Box<[Expr]>),
    FunctionCall(Span, Box<[Expr]>),

    // Left recursive, handled specially
    FieldAccess(Box<Expr>, Span),
    ArrayIndex(Box<Expr>, Box<[Expr]>),
}

impl Parse for Expr {
    fn parse(ts: &mut super::TokenStream) -> miette::Result<Self> {
        let mut expr = match (ts.peek_type(), ts.peek_type_at(2)) {
            (Some(TokenType::LParen), _) => {
                _ = expect_tokens(ts, [TokenType::RParen])?;
                //TODO: Should location include the parenthesis
                let expr = Self::parse(ts);
                _ = expect_tokens(ts, [TokenType::LParen])?;
                expr
            }
            (Some(TokenType::IntLit), _) => Self::parse_int_lit(ts),
            (Some(TokenType::FloatLit), _) => Self::parse_float_lit(ts),
            (Some(TokenType::True), _) => Self::parse_true(ts),
            (Some(TokenType::False), _) => Self::parse_false(ts),
            (Some(TokenType::LSquare), _) => Self::parse_array_lit(ts),
            (Some(TokenType::Variable), Some(TokenType::RCurly)) => Self::parse_struct_init(ts),
            (Some(TokenType::Variable), Some(TokenType::RParen)) => Self::parse_fn_call(ts),
            (Some(TokenType::Variable), _) => Self::parse_var(ts),
            (Some(t), _) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected expresion, found: {}", t)),
                    ts.peek().unwrap().span().start(),
                    ts.peek().unwrap().bytes().len(),
                )],
                "Unexpected token found"
            )),
            (None, _) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::at_offset(
                    ts.lexer().bytes().len() - 1,
                    "expected expression"
                )],
                "Missing expected token"
            )),
        }?;

        Ok(loop {
            match ts.peek_type() {
                Some(TokenType::Dot) => {
                    _ = expect_tokens(ts, [TokenType::Dot])?;
                    let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
                    let location = expr.location.join(&var_token.span());
                    expr = Self {
                        location,
                        kind: ExprKind::FieldAccess(Box::new(expr), var_token.span()),
                    }
                }
                Some(TokenType::LSquare) => {
                    _ = expect_tokens(ts, [TokenType::LSquare])?;
                    let indices = parse_sequence(ts, TokenType::Comma, TokenType::RSquare)?;
                    let [r_square_token] = expect_tokens(ts, [TokenType::RSquare])?;
                    let location = expr.location.join(&r_square_token.span());
                    expr = Self {
                        location,
                        kind: ExprKind::ArrayIndex(Box::new(expr), indices),
                    }
                }
                _ => break expr,
            }
        })
    }
}
impl Expr {
    pub fn location(&self) -> Span {
        self.location
    }

    fn parse_int_lit(ts: &mut TokenStream) -> miette::Result<Self> {
        let [int_lit_token] = expect_tokens(ts, [TokenType::IntLit])?;
        let int_val: u64 = match int_lit_token.bytes().parse() {
            Ok(i) if i <= POSITIVE_INT_LIT_MAX => i,
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
        let [float_lit_token] = expect_tokens(ts, [TokenType::FloatLit])?;
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
        let [true_token] = expect_tokens(ts, [TokenType::True])?;
        Ok(Self {
            location: true_token.span(),
            kind: ExprKind::True,
        })
    }

    fn parse_false(ts: &mut TokenStream) -> miette::Result<Self> {
        let [false_token] = expect_tokens(ts, [TokenType::False])?;
        Ok(Self {
            location: false_token.span(),
            kind: ExprKind::False,
        })
    }

    fn parse_var(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        Ok(Self {
            location: var_token.span(),
            kind: ExprKind::Var,
        })
    }

    fn parse_array_lit(ts: &mut TokenStream) -> miette::Result<Self> {
        let [l_square_token] = expect_tokens(ts, [TokenType::LSquare])?;
        let items = parse_sequence(ts, TokenType::Comma, TokenType::RSquare)?;
        let [rb_token] = expect_tokens(ts, [TokenType::RSquare])?;
        Ok(Self {
            location: l_square_token.span().join(&rb_token.span()),
            kind: ExprKind::ArrayLit(items),
        })
    }

    fn parse_struct_init(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        _ = expect_tokens(ts, [TokenType::LCurly])?;
        let members = parse_sequence(ts, TokenType::Comma, TokenType::RCurly)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RCurly])?;
        let location = var_token.span().join(&r_curly_token.span());
        Ok(Self {
            location,
            kind: ExprKind::StructInit(var_token.span(), members),
        })
    }

    fn parse_fn_call(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        _ = expect_tokens(ts, [TokenType::LParen])?;
        let args = parse_sequence(ts, TokenType::Comma, TokenType::RParen)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RParen])?;
        let location = var_token.span().join(&r_curly_token.span());
        Ok(Self {
            location,
            kind: ExprKind::FunctionCall(var_token.span(), args),
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
            ExprKind::StructInit(_span, fields) => {
                let mut s_expr = format!("(StructLiteralExpr");
                //TODO: Refernce compiler does not add struct name however this may be a bug,
                //update when you get more info

                for expr in fields {
                    s_expr.push(' ');
                    s_expr.push_str(&expr.to_s_expresion(src));
                }

                s_expr.push(')');

                s_expr
            }
            ExprKind::FunctionCall(span, args) => {
                let mut s_expr = format!("(CallExpr {}", span.as_str(src));
                for arg in args {
                    s_expr.push(' ');
                    s_expr.push_str(&arg.to_s_expresion(src));
                }
                s_expr.push(')');
                s_expr
            }
            ExprKind::FieldAccess(expr, span) => format!(
                "(DotExpr {} {})",
                expr.to_s_expresion(src),
                span.as_str(src)
            ),
            ExprKind::ArrayIndex(expr, indices) => {
                let mut s_expr = format!("(ArrayIndexExpr {}", expr.to_s_expresion(src));
                for index in indices {
                    s_expr.push(' ');
                    s_expr.push_str(&index.to_s_expresion(src));
                }
                s_expr.push(')');
                s_expr
            }
        }
    }
}
