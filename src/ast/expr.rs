//! Defines the types of functions to parse all kinds of expressions in JPL
use super::{super::parse::parse_sequence, expect_tokens, Parse, TokenStream};
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
/// The current grammar is as follows. Each class will match as much as possible which ensures
/// proper precedence
///
/// Todo: Fix grammer it associativity is currently flipped (but the code is correct)
///```text
/// expr : <control>
///      | <bool>
///
/// control : array [ <variable> : <expr> , ... ] <expr>
///         | sum [ <variable> : <expr> , ... ] <expr>
///         | if <expr> then <expr> else <expr>
///
/// bool : <cmp> && <bool>
///      | <cmp> && <control>
///      | <cmp> || <bool>
///      | <cmp> || <control>
///      | <cmp>
///
/// cmp : <add> < <cmp>
///     | <add> < <control>
///     | <add> > <cmp>
///     | <add> > <control>
///     | <add> <= <cmp>
///     | <add> <= <control>
///     | <add> >= <cmp>
///     | <add> >= <control>
///     | <add> == <cmp>
///     | <add> == <control>
///     | <add> != <cmp>
///     | <add> != <control>
///     | <add>
///
/// add : <mult> + <add>
///     | <mult> + <control>
///     | <mult> - <add>
///     | <mult> - <control>
///     | <mult>
///
///
/// mult : <unary> * <mult>
///      | <unary> * <control>
///      | <unary> / <mult>
///      | <unary> / <control>
///      | <unary> % <mult>
///      | <unary> % <control>
///      | <unary>
///
/// unary : !<unary>
///       | !<control>
///       | -<unary>
///       | -<control>
///       | <terminal>
///
/// terminal : <simple expr><expr cont>
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
///```
#[derive(Debug, Clone)]
pub enum ExprKind {
    // Simple expresions
    IntLit(u64),
    FloatLit(f64),
    True,
    False,
    Var,
    Void,
    Paren(Box<Expr>),
    ArrayLit(Box<[Expr]>),
    StructInit(Span, Box<[Expr]>),
    FunctionCall(Span, Box<[Expr]>),

    // Left recursive, handled specially
    FieldAccess(Box<Expr>, Span),
    ArrayIndex(Box<Expr>, Box<[Expr]>),

    // Lowest Precedence
    If(Box<(Expr, Expr, Expr)>),
    ArrayComp(Box<[(Span, Expr)]>, Box<Expr>),
    Sum(Box<[(Span, Expr)]>, Box<Expr>),

    // Bool ops
    And(Box<(Expr, Expr)>),
    Or(Box<(Expr, Expr)>),

    // Comparisons: <, >, <=, and >=, ==, !=
    LessThan(Box<(Expr, Expr)>),
    GreaterThan(Box<(Expr, Expr)>),
    LessThanEq(Box<(Expr, Expr)>),
    GreaterThanEq(Box<(Expr, Expr)>),
    Eq(Box<(Expr, Expr)>),
    NotEq(Box<(Expr, Expr)>),

    // Additive operations + and -	}
    Add(Box<(Expr, Expr)>),
    Minus(Box<(Expr, Expr)>),

    // Multiplicative operations *, /, and %
    Mulitply(Box<(Expr, Expr)>),
    Divide(Box<(Expr, Expr)>),
    Modulo(Box<(Expr, Expr)>),

    // Unary inverse ! and negation -
    Not(Box<Expr>),
    Negation(Box<Expr>),
}

type VariantBuilder = fn(Box<(Expr, Expr)>) -> ExprKind;

fn parse_binary_op(
    ts: &mut TokenStream,
    ops: &[(&str, VariantBuilder)],
    mut sub_class: impl FnMut(&mut TokenStream) -> miette::Result<Expr>,
) -> miette::Result<Expr> {
    let mut lhs = sub_class(ts)?;
    'outer: loop {
        for (op_as_str, op_var) in ops {
            match ts.peek() {
                Some(t) if t.kind() == TokenType::Op && t.bytes() == *op_as_str => {
                    _ = expect_tokens(ts, [TokenType::Op])?;
                    // Be able to parse a control on the rhs. While control does have a low
                    // precedence there is no ambiguity when it is on the right
                    // ex. 1 + sum[i: 50] i should parse
                    let rhs = match ts.peek_type() {
                        Some(TokenType::Array) => Expr::parse_array_comp(ts),
                        Some(TokenType::Sum) => Expr::parse_sum(ts),
                        Some(TokenType::If) => Expr::parse_if(ts),
                        _ => sub_class(ts),
                    }?;
                    let location = lhs.location.join(&rhs.location);

                    lhs = Expr {
                        location,
                        kind: op_var(Box::new((lhs, rhs))),
                    };
                    continue 'outer;
                }
                _ => (),
            }
        }

        return Ok(lhs);
    }
}

impl Parse for Expr {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        Self::parse_control(ts)
    }
}

/// Used for sum and array looping constructs
impl Parse for (Span, Expr) {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var, _] = expect_tokens(ts, [TokenType::Variable, TokenType::Colon])?;
        let expr = Expr::parse(ts)?;
        Ok((var.span(), expr))
    }
}

impl Expr {
    pub fn location(&self) -> Span {
        self.location
    }

    fn parse_control(ts: &mut TokenStream) -> miette::Result<Self> {
        match ts.peek_type() {
            Some(TokenType::If) => Self::parse_if(ts),
            Some(TokenType::Array) => Self::parse_array_comp(ts),
            Some(TokenType::Sum) => Self::parse_sum(ts),
            _ => Self::parse_bool(ts),
        }
    }

    fn parse_if(ts: &mut TokenStream) -> miette::Result<Self> {
        let [if_token] = expect_tokens(ts, [TokenType::If])?;
        let cond = Expr::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::Then])?;

        let true_expr = Expr::parse(ts)?;
        _ = expect_tokens(ts, [TokenType::Else])?;
        let false_expr = Expr::parse(ts)?;

        let location = if_token.span().join(&false_expr.location);
        Ok(Self {
            location,
            kind: ExprKind::If(Box::new((cond, true_expr, false_expr))),
        })
    }
    fn parse_array_comp(ts: &mut TokenStream) -> miette::Result<Self> {
        let [arr_token, _] = expect_tokens(ts, [TokenType::Array, TokenType::LSquare])?;
        let params = parse_sequence(ts, TokenType::Comma, TokenType::RSquare)?;
        let _ = expect_tokens(ts, [TokenType::RSquare])?;
        let expr = Self::parse(ts)?;
        let location = arr_token.span().join(&expr.location);

        Ok(Self {
            location,
            kind: ExprKind::ArrayComp(params, Box::new(expr)),
        })
    }
    fn parse_sum(ts: &mut TokenStream) -> miette::Result<Self> {
        let [arr_token, _] = expect_tokens(ts, [TokenType::Sum, TokenType::LSquare])?;
        let params = parse_sequence(ts, TokenType::Comma, TokenType::RSquare)?;
        let _ = expect_tokens(ts, [TokenType::RSquare])?;
        let expr = Self::parse(ts)?;
        let location = arr_token.span().join(&expr.location);

        Ok(Self {
            location,
            kind: ExprKind::Sum(params, Box::new(expr)),
        })
    }
    fn parse_bool(ts: &mut TokenStream) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            &[("||", ExprKind::Or), ("&&", ExprKind::And)],
            Self::parse_cmp,
        )
    }

    fn parse_cmp(ts: &mut TokenStream) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            &[
                (">", ExprKind::GreaterThan),
                ("<", ExprKind::LessThan),
                ("<=", ExprKind::LessThanEq),
                (">=", ExprKind::GreaterThanEq),
                ("==", ExprKind::Eq),
                ("!=", ExprKind::NotEq),
            ],
            Self::parse_add,
        )
    }

    fn parse_add(ts: &mut TokenStream) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            &[("+", ExprKind::Add), ("-", ExprKind::Minus)],
            Self::parse_mult,
        )
    }
    fn parse_mult(ts: &mut TokenStream) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            &[
                ("*", ExprKind::Mulitply),
                ("/", ExprKind::Divide),
                ("%", ExprKind::Modulo),
            ],
            Self::parse_unary,
        )
    }

    fn parse_unary(ts: &mut TokenStream) -> miette::Result<Self> {
        let expr_type = match ts.peek() {
            Some(t) if t.kind() == TokenType::Op && t.bytes() == "!" => ExprKind::Not,
            Some(t) if t.kind() == TokenType::Op && t.bytes() == "-" => ExprKind::Negation,
            Some(t) if t.kind() == TokenType::Array => return Expr::parse_array_comp(ts),
            Some(t) if t.kind() == TokenType::Sum => return Expr::parse_sum(ts),
            Some(t) if t.kind() == TokenType::If => return Expr::parse_if(ts),
            _ => return Self::parse_simple(ts),
        };

        let [unary_op_token] = expect_tokens(ts, [TokenType::Op])?;
        let rhs = Self::parse_unary(ts)?;
        let location = unary_op_token.span().join(&rhs.location);
        Ok(Self {
            location,
            kind: expr_type(Box::new(rhs)),
        })
    }

    fn parse_simple(ts: &mut TokenStream) -> miette::Result<Self> {
        let mut expr = match (ts.peek_type(), ts.peek_type_at(2)) {
            (Some(TokenType::LParen), _) => {
                let [l_paren] = expect_tokens(ts, [TokenType::LParen])?;
                let expr = Self::parse(ts)?;
                let [r_paren] = expect_tokens(ts, [TokenType::RParen])?;

                Ok(Self {
                    location: l_paren.span().join(&r_paren.span()),
                    kind: ExprKind::Paren(Box::new(expr)),
                })
            }
            (Some(TokenType::IntLit), _) => Self::parse_int_lit(ts),
            (Some(TokenType::FloatLit), _) => Self::parse_float_lit(ts),
            (Some(TokenType::True), _) => Self::parse_true(ts),
            (Some(TokenType::False), _) => Self::parse_false(ts),
            (Some(TokenType::Void), _) => Self::parse_void(ts),
            (Some(TokenType::LSquare), _) => Self::parse_array_lit(ts),
            (Some(TokenType::Variable), Some(TokenType::LCurly)) => Self::parse_struct_init(ts),
            (Some(TokenType::Variable), Some(TokenType::LParen)) => Self::parse_fn_call(ts),
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

    fn parse_void(ts: &mut TokenStream) -> miette::Result<Self> {
        let [void_token] = expect_tokens(ts, [TokenType::Void])?;
        Ok(Self {
            location: void_token.span(),
            kind: ExprKind::Void,
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
            ExprKind::Void => "(VoidExpr)".to_string(),
            ExprKind::ArrayLit(items) => {
                let mut s_expr = "(ArrayLiteralExpr".to_string();
                for item in items {
                    s_expr.push(' ');
                    s_expr.push_str(&item.to_s_expresion(src));
                }
                s_expr.push(')');
                s_expr
            }
            ExprKind::StructInit(span, fields) => {
                let mut s_expr = format!("(StructLiteralExpr {}", span.as_str(src));

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
            ExprKind::If(if_stmt) => format!(
                "(IfExpr {} {} {})",
                if_stmt.0.to_s_expresion(src),
                if_stmt.1.to_s_expresion(src),
                if_stmt.2.to_s_expresion(src)
            ),
            ExprKind::ArrayComp(args, expr) => {
                let mut s_expr = "(ArrayLoopExpr ".to_string();
                for (var, expr) in args {
                    s_expr.push_str(var.as_str(src));
                    s_expr.push(' ');

                    s_expr.push_str(&expr.to_s_expresion(src));
                    s_expr.push(' ');
                }
                s_expr.push_str(&expr.to_s_expresion(src));
                s_expr.push(')');
                s_expr
            }
            ExprKind::Sum(args, expr) => {
                let mut s_expr = "(SumLoopExpr ".to_string();
                for (var, expr) in args {
                    s_expr.push_str(var.as_str(src));
                    s_expr.push(' ');

                    s_expr.push_str(&expr.to_s_expresion(src));
                    s_expr.push(' ');
                }
                s_expr.push_str(&expr.to_s_expresion(src));
                s_expr.push(')');
                s_expr
            }
            ExprKind::And(operands) => format!(
                "(BinopExpr {} && {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Or(operands) => format!(
                "(BinopExpr {} || {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::LessThan(operands) => format!(
                "(BinopExpr {} < {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::GreaterThan(operands) => format!(
                "(BinopExpr {} > {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::LessThanEq(operands) => format!(
                "(BinopExpr {} <= {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::GreaterThanEq(operands) => format!(
                "(BinopExpr {} >= {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Eq(operands) => format!(
                "(BinopExpr {} == {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::NotEq(operands) => format!(
                "(BinopExpr {} != {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Add(operands) => format!(
                "(BinopExpr {} + {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Minus(operands) => format!(
                "(BinopExpr {} - {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Mulitply(operands) => format!(
                "(BinopExpr {} * {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Divide(operands) => format!(
                "(BinopExpr {} / {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Modulo(operands) => format!(
                "(BinopExpr {} % {})",
                operands.0.to_s_expresion(src),
                operands.1.to_s_expresion(src)
            ),
            ExprKind::Not(expr) => format!("(UnopExpr ! {})", expr.to_s_expresion(src)),
            ExprKind::Negation(expr) => format!("(UnopExpr - {})", expr.to_s_expresion(src)),
            ExprKind::Paren(expr) => expr.to_s_expresion(src),
        }
    }
}
