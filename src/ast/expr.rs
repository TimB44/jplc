//! Defines the types of functions to parse all kinds of expressions in JPL
use super::{super::parse::parse_sequence, expect_tokens, Parse, TokenStream};
use crate::{
    ast::auxiliary::LValue,
    environment::Environment,
    lex::TokenType,
    typecheck::{TypeState, Typed, UnTyped},
    utils::Span,
};
use miette::{miette, LabeledSpan, Severity};

//TODO: allow numbers one bigger due to negative numbers
const POSITIVE_INT_LIT_MAX: u64 = 9223372036854775807;

/// Represents an expression in JPL
#[derive(Debug, Clone)]
pub struct Expr<T: TypeState = UnTyped> {
    location: Span,
    kind: ExprKind<T>,
    type_data: T,
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
pub enum ExprKind<T: TypeState = UnTyped> {
    // Simple expressions
    IntLit(u64),
    FloatLit(f64),
    True,
    False,
    Var,
    Void,
    Paren(Box<Expr<T>>),
    ArrayLit(Box<[Expr<T>]>),
    StructInit(Span, Box<[Expr<T>]>),
    FunctionCall(Span, Box<[Expr<T>]>),

    // Left recursive, handled specially
    FieldAccess(Box<Expr<T>>, Span),
    ArrayIndex(Box<Expr<T>>, Box<[Expr<T>]>),

    // Lowest Precedence
    If(Box<(Expr<T>, Expr<T>, Expr<T>)>),

    // Number represents their scope
    ArrayComp(Box<[(Span, Expr<T>)]>, Box<Expr<T>>, usize),
    Sum(Box<[(Span, Expr<T>)]>, Box<Expr<T>>, usize),

    // Bool ops
    And(Box<(Expr<T>, Expr<T>)>),
    Or(Box<(Expr<T>, Expr<T>)>),

    // Comparisons: <, >, <=, and >=, ==, !=
    LessThan(Box<(Expr<T>, Expr<T>)>),
    GreaterThan(Box<(Expr<T>, Expr<T>)>),
    LessThanEq(Box<(Expr<T>, Expr<T>)>),
    GreaterThanEq(Box<(Expr<T>, Expr<T>)>),
    Eq(Box<(Expr<T>, Expr<T>)>),
    NotEq(Box<(Expr<T>, Expr<T>)>),

    // Additive operations + and -	}
    Add(Box<(Expr<T>, Expr<T>)>),
    Minus(Box<(Expr<T>, Expr<T>)>),

    // Multiplicative operations *, /, and %
    Mulitply(Box<(Expr<T>, Expr<T>)>),
    Divide(Box<(Expr<T>, Expr<T>)>),
    Modulo(Box<(Expr<T>, Expr<T>)>),

    // Unary inverse ! and negation -
    Not(Box<Expr<T>>),
    Negation(Box<Expr<T>>),
}

impl<T: TypeState> ExprKind<T> {
    pub fn varient_eq(&self, other: &ExprKind<T>) -> bool {
        match self {
            ExprKind::Paren(inner) => inner.kind.varient_eq(other),
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

type VariantBuilder = fn(Box<(Expr, Expr)>) -> ExprKind;

fn parse_binary_op(
    ts: &mut TokenStream,
    ops: &[(&str, VariantBuilder)],
    mut sub_class: impl FnMut(&mut TokenStream) -> miette::Result<Expr>,
) -> miette::Result<Expr> {
    let mut lhs: Expr = sub_class(ts)?;
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
                        type_data: UnTyped {},
                    };
                    continue 'outer;
                }
                _ => (),
            }
        }

        return Ok(lhs);
    }
}

impl Parse<Expr> for Expr {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        Self::parse_control(ts)
    }
}

/// Used for sum and array looping constructs
impl Parse<(Span, Expr)> for (Span, Expr) {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var, _] = expect_tokens(ts, [TokenType::Variable, TokenType::Colon])?;
        let expr = Expr::parse(ts)?;
        Ok((var.span(), expr))
    }
}

impl Expr {
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
            type_data: UnTyped {},
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
            kind: ExprKind::ArrayComp(params, Box::new(expr), 0),
            type_data: UnTyped {},
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
            kind: ExprKind::Sum(params, Box::new(expr), 0),

            type_data: UnTyped {},
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
            type_data: UnTyped {},
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
                    type_data: UnTyped {},
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
                    Some(format!("expected expression, found: {}", t)),
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
                        type_data: UnTyped {},
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
                        type_data: UnTyped {},
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
            type_data: UnTyped {},
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
            type_data: UnTyped {},
        })
    }

    fn parse_true(ts: &mut TokenStream) -> miette::Result<Self> {
        let [true_token] = expect_tokens(ts, [TokenType::True])?;
        Ok(Self {
            location: true_token.span(),
            kind: ExprKind::True,
            type_data: UnTyped {},
        })
    }

    fn parse_false(ts: &mut TokenStream) -> miette::Result<Self> {
        let [false_token] = expect_tokens(ts, [TokenType::False])?;
        Ok(Self {
            location: false_token.span(),
            kind: ExprKind::False,
            type_data: UnTyped {},
        })
    }

    fn parse_void(ts: &mut TokenStream) -> miette::Result<Self> {
        let [void_token] = expect_tokens(ts, [TokenType::Void])?;
        Ok(Self {
            location: void_token.span(),
            kind: ExprKind::Void,
            type_data: UnTyped {},
        })
    }

    fn parse_var(ts: &mut TokenStream) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        Ok(Self {
            location: var_token.span(),
            kind: ExprKind::Var,
            type_data: UnTyped {},
        })
    }

    fn parse_array_lit(ts: &mut TokenStream) -> miette::Result<Self> {
        let [l_square_token] = expect_tokens(ts, [TokenType::LSquare])?;
        let items = parse_sequence(ts, TokenType::Comma, TokenType::RSquare)?;
        let [rb_token] = expect_tokens(ts, [TokenType::RSquare])?;
        Ok(Self {
            location: l_square_token.span().join(&rb_token.span()),
            kind: ExprKind::ArrayLit(items),
            type_data: UnTyped {},
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
            type_data: UnTyped {},
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
            type_data: UnTyped {},
        })
    }
}

impl Expr<Typed> {
    pub fn to_typed_s_exprsision(&self, env: &Environment) -> String {
        let mut normal = self.to_s_expr_general(
            env.src(),
            Some(|expr: &Expr<Typed>| expr.to_typed_s_exprsision(env)),
        );
        let index_for_type = normal.find([' ', ')']).expect("invalid s-expr found");

        normal.insert_str(
            index_for_type,
            &format!(" ({})", self.type_data.to_typed_s_exprsision(env)),
        );

        normal
    }

    pub fn expect_type(&self, expected: &Typed, env: &Environment) -> miette::Result<()> {
        if &self.type_data == expected {
            return Ok(());
        }
        Err(miette!(
            severity = Severity::Error,
            labels = vec![LabeledSpan::new(
                Some(format!(
                    "expected type: {}, found: {}",
                    expected.as_str(env),
                    self.type_data.as_str(env)
                )),
                self.location.start(),
                self.location.len(),
            )],
            "Type mismatch"
        ))
    }

    pub fn expect_one_of_types(&self, expected: &[Typed], env: &Environment) -> miette::Result<()> {
        if expected.contains(&self.type_data) {
            return Ok(());
        }
        Err(miette!(
            severity = Severity::Error,
            labels = vec![LabeledSpan::new(
                Some(format!(
                    "expected one of these types: [{}], found: {}",
                    expected
                        .iter()
                        .map(|t| t.as_str(env))
                        .collect::<Vec<String>>()
                        .join(","),
                    self.type_data.as_str(env),
                )),
                self.location.start(),
                self.location.len(),
            )],
            "Type mismatch"
        ))
    }
    pub fn type_data(&self) -> &Typed {
        &self.type_data
    }

    pub fn expect_array_of_rank(
        &self,
        expected_rank: usize,
        env: &Environment,
    ) -> miette::Result<()> {
        match self.type_data() {
            Typed::Array(_, rank) if *rank as usize == expected_rank => Ok(()),
            Typed::Array(_, rank) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "expected array of rank: {}, found array of rank: {}",
                        expected_rank, rank,
                    )),
                    self.location().start(),
                    self.location().len(),
                )],
                "Unexpected token found"
            )),
            acutal_type => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "expected array of rank: {}, found type: {}",
                        expected_rank,
                        acutal_type.as_str(env)
                    )),
                    self.location().start(),
                    self.location().len(),
                )],
                "Unexpected token found"
            )),
        }
    }
}

impl Expr {
    // Add case for compaing bools
    fn typecheck_cmp_binop(
        operands: (Expr, Expr),
        varient: fn(Box<(Expr<Typed>, Expr<Typed>)>) -> ExprKind<Typed>,
        location: Span,
        env: &mut Environment,
        scope_id: usize,
    ) -> miette::Result<Expr<Typed>> {
        let (lhs, rhs) = operands;
        let typed_lhs = lhs.typecheck(env, scope_id)?;
        let typed_rhs = rhs.typecheck(env, scope_id)?;

        typed_lhs.expect_one_of_types(&[Typed::Int, Typed::Float], env)?;
        typed_rhs.expect_type(&typed_lhs.type_data, env)?;

        Ok(Expr {
            location,
            kind: varient(Box::new((typed_lhs, typed_rhs))),
            type_data: Typed::Bool,
        })
    }
    // Add case for compaing bools
    fn typecheck_eq_binop(
        operands: (Expr, Expr),
        varient: fn(Box<(Expr<Typed>, Expr<Typed>)>) -> ExprKind<Typed>,
        location: Span,
        env: &mut Environment,
        scope_id: usize,
    ) -> miette::Result<Expr<Typed>> {
        let (lhs, rhs) = operands;
        let typed_lhs = lhs.typecheck(env, scope_id)?;
        let typed_rhs = rhs.typecheck(env, scope_id)?;

        typed_lhs.expect_one_of_types(&[Typed::Int, Typed::Float, Typed::Bool], env)?;
        typed_rhs.expect_type(&typed_lhs.type_data, env)?;

        Ok(Expr {
            location,
            kind: varient(Box::new((typed_lhs, typed_rhs))),
            type_data: Typed::Bool,
        })
    }
    fn typecheck_numerical_binop(
        operands: (Expr, Expr),
        varient: fn(Box<(Expr<Typed>, Expr<Typed>)>) -> ExprKind<Typed>,
        location: Span,
        env: &mut Environment,
        scope_id: usize,
    ) -> miette::Result<Expr<Typed>> {
        let (lhs, rhs) = operands;
        let typed_lhs = lhs.typecheck(env, scope_id)?;
        let typed_rhs = rhs.typecheck(env, scope_id)?;

        typed_lhs.expect_one_of_types(&[Typed::Int, Typed::Float], env)?;
        let output_type = typed_lhs.type_data.clone();
        typed_rhs.expect_type(&output_type, env)?;

        Ok(Expr {
            location,
            kind: varient(Box::new((typed_lhs, typed_rhs))),
            type_data: output_type,
        })
    }

    pub fn typecheck(self, env: &mut Environment, scope_id: usize) -> miette::Result<Expr<Typed>> {
        Ok(match self.kind {
            ExprKind::IntLit(val) => Expr {
                type_data: Typed::Int,
                location: self.location,
                kind: ExprKind::IntLit(val),
            },

            ExprKind::FloatLit(val) => Expr {
                type_data: Typed::Float,
                location: self.location,
                kind: ExprKind::FloatLit(val),
            },

            ExprKind::True => Expr {
                type_data: Typed::Bool,
                location: self.location,
                kind: ExprKind::True,
            },
            ExprKind::False => Expr {
                type_data: Typed::Bool,
                location: self.location,
                kind: ExprKind::False,
            },

            ExprKind::Var => {
                let var_type = env.get_variable_type(self.location, scope_id)?;
                Expr {
                    location: self.location,
                    kind: ExprKind::Var,
                    type_data: var_type.clone(),
                }
            }
            ExprKind::Void => Expr {
                type_data: Typed::Void,
                location: self.location,
                kind: ExprKind::Void,
            },
            ExprKind::Paren(expr) => {
                let typed_expr = expr.typecheck(env, scope_id)?;

                Expr {
                    type_data: typed_expr.type_data.clone(),
                    location: self.location,
                    kind: ExprKind::Paren(Box::new(typed_expr)),
                }
            }

            ExprKind::ArrayLit(exprs) => {
                if exprs.is_empty() {
                    return Err(miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::new(
                            Some("array literals can not be empty".to_string()),
                            self.location.start(),
                            self.location.len()
                        )],
                        "Invalid array literal"
                    ));
                }

                // TODO: May be able to remove the to_vec
                let typed_exprs = exprs
                    .to_vec()
                    .into_iter()
                    .map(|e| e.typecheck(env, scope_id))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice();

                let element_type = typed_exprs[0].type_data.clone();
                typed_exprs
                    .iter()
                    .skip(1)
                    .map(|e| e.expect_type(&element_type, env))
                    .collect::<Result<Vec<()>, _>>()?;

                Expr {
                    location: self.location,
                    kind: ExprKind::ArrayLit(typed_exprs),
                    type_data: Typed::Array(Box::new(element_type), 1),
                }
            }
            ExprKind::StructInit(span, exprs) => {
                {
                    let info = env.get_struct(span)?;
                    if exprs.len() != info.fields().len() {
                        return Err(miette!(
                            severity = Severity::Error,
                            labels = vec![LabeledSpan::new(
                                Some(format!(
                                    "Expected {} fields, found {}",
                                    info.fields().len(),
                                    exprs.len()
                                )),
                                self.location.start(),
                                self.location.len(),
                            )],
                            "Incorrect number of fields for struct: {}",
                            span.as_str(env.src())
                        ));
                    }
                }

                // Not ideal but must be done to make the borrow checker happy
                let mut checked_exprs = Vec::with_capacity(exprs.len());
                for (i, expr) in exprs.to_vec().into_iter().enumerate() {
                    let typed = expr.typecheck(env, scope_id)?;
                    typed.expect_type(&env.get_struct(span)?.fields()[i].1, env)?;
                    checked_exprs.push(typed);
                }

                let checked_exprs = checked_exprs.into_boxed_slice();

                let info = env.get_struct(span)?;

                Expr {
                    location: self.location,
                    kind: ExprKind::StructInit(span, checked_exprs),
                    type_data: Typed::Struct(info.id()),
                }
            }
            ExprKind::FunctionCall(fn_name, exprs) => {
                let fn_info = env.get_function(fn_name)?;
                if exprs.len() != fn_info.args().len() {
                    return Err(miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::new(
                            Some(format!(
                                "Expected {} arguments, found {}",
                                fn_info.args().len(),
                                exprs.len(),
                            )),
                            self.location.start(),
                            self.location.len(),
                        )],
                        "Incorrect number of arguments for function: {}",
                        fn_name.as_str(env.src())
                    ));
                }

                let typed_args = exprs
                    .to_vec()
                    .into_iter()
                    .map(|e| e.typecheck(env, scope_id))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice();

                let fn_info = env.get_function(fn_name)?;

                for (arg, expected_type) in typed_args.iter().zip(fn_info.args()) {
                    arg.expect_type(expected_type, env)?;
                }

                Expr {
                    location: self.location,
                    kind: ExprKind::FunctionCall(fn_name, typed_args),
                    type_data: fn_info.ret().clone(),
                }
            }
            ExprKind::FieldAccess(expr, span) => {
                let typed_struct = expr.typecheck(env, scope_id)?;
                let id = match typed_struct.type_data {
                    Typed::Struct(id) => id,
                    _ => 0,
                };

                typed_struct.expect_type(&Typed::Struct(id), env)?;

                let field_name = span.as_str(env.src());
                let info = env.get_struct_id(id);
                let field_type = info
                    .fields()
                    .iter()
                    .find(|(name, _)| *name == field_name)
                    .map(|(_, t)| t)
                    .ok_or_else(|| {
                        miette!(
                            severity = Severity::Error,
                            labels = vec![LabeledSpan::new(
                                Some(format!(
                                    "no field named: {} found for struct: {}",
                                    field_name,
                                    info.name()
                                )),
                                span.start(),
                                span.len(),
                            )],
                            "Struct field name not changed"
                        )
                    })?
                    .clone();

                Expr {
                    location: self.location,
                    kind: ExprKind::FieldAccess(Box::new(typed_struct), span),
                    type_data: field_type,
                }
            }

            ExprKind::ArrayIndex(arr_expr, indices) => {
                let arr_expr_typed = arr_expr.typecheck(env, scope_id)?;
                let (element_type, rank) = match &arr_expr_typed.type_data {
                    Typed::Array(element_type, rank) => (element_type, rank),
                    t => {
                        return Err(miette!(
                            severity = Severity::Error,
                            labels = vec![LabeledSpan::new(
                                Some(format!(
                                    "Expected type array type, found: {}",
                                    t.as_str(env)
                                )),
                                arr_expr_typed.location.start(),
                                arr_expr_typed.location.len()
                            )],
                            "Can only index arrays"
                        ));
                    }
                };

                if *rank as usize != indices.len() {
                    return Err(miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::new(
                            Some(format!("Expected {} indices found {}", rank, indices.len())),
                            self.location.start(),
                            self.location.len()
                        )],
                        "Incorrect amount of indices"
                    ));
                }

                let typed_exprs = indices
                    .to_vec()
                    .into_iter()
                    .map(|e| e.typecheck(env, scope_id))
                    .collect::<Result<Vec<Expr<Typed>>, _>>()?
                    .into_boxed_slice();

                typed_exprs
                    .iter()
                    .map(|e| e.expect_type(&Typed::Int, env))
                    .collect::<Result<Vec<()>, _>>()?;

                let element_type = *element_type.clone();

                Expr {
                    location: self.location,
                    kind: ExprKind::ArrayIndex(Box::new(arr_expr_typed), typed_exprs),
                    type_data: element_type,
                }
            }

            ExprKind::If(if_expr) => {
                let (cond, true_branch, false_branch) = *if_expr;
                let typed_cond = cond.typecheck(env, scope_id)?;
                let typed_true_branch = true_branch.typecheck(env, scope_id)?;
                let typed_false_branch = false_branch.typecheck(env, scope_id)?;
                typed_cond.expect_type(&Typed::Bool, env)?;
                let branch_type = &typed_true_branch.type_data.clone();

                typed_false_branch.expect_type(branch_type, env)?;

                Expr {
                    location: self.location,
                    kind: ExprKind::If(Box::new((
                        typed_cond,
                        typed_true_branch,
                        typed_false_branch,
                    ))),
                    type_data: branch_type.clone(),
                }
            }
            ExprKind::ArrayComp(vars, expr, _) => {
                let inner_scope = env.new_scope(scope_id);
                let mut typed_vars = Vec::with_capacity(vars.len());

                if vars.is_empty() {
                    return Err(miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::new(
                            Some("array comprehension must have at least 1 binding".to_string()),
                            self.location.start(),
                            self.location.len()
                        )],
                        "Can not create a zero rank array"
                    ));
                }

                for (name, val) in vars {
                    let typed = val.typecheck(env, scope_id)?;
                    typed.expect_type(&Typed::Int, env)?;
                    env.add_lval(&LValue::from_span(name), Typed::Int, inner_scope)?;
                    typed_vars.push((name, typed));
                }

                let body = expr.typecheck(env, inner_scope)?;
                let type_data =
                    Typed::Array(Box::new(body.type_data.clone()), typed_vars.len() as u8);

                Expr {
                    location: self.location,
                    kind: ExprKind::ArrayComp(
                        typed_vars.into_boxed_slice(),
                        Box::new(body),
                        inner_scope,
                    ),
                    type_data,
                }
            }
            ExprKind::Sum(vars, expr, _) => {
                let inner_scope = env.new_scope(scope_id);
                let mut typed_vars = Vec::with_capacity(vars.len());

                if vars.is_empty() {
                    return Err(miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::new(
                            Some("sum comprehension must have at least 1 binding".to_string()),
                            self.location.start(),
                            self.location.len()
                        )],
                        "Illegal sum expresion"
                    ));
                }

                for (name, val) in vars {
                    let typed = val.typecheck(env, scope_id)?;
                    typed.expect_type(&Typed::Int, env)?;
                    env.add_lval(&LValue::from_span(name), Typed::Int, inner_scope)?;
                    typed_vars.push((name, typed));
                }

                let body = expr.typecheck(env, inner_scope)?;
                body.expect_one_of_types(&[Typed::Int, Typed::Float], env)?;
                let type_data = body.type_data.clone();

                Expr {
                    location: self.location,
                    kind: ExprKind::Sum(typed_vars.into_boxed_slice(), Box::new(body), inner_scope),
                    type_data,
                }
            }
            ExprKind::And(operands) => {
                let (lhs, rhs) = *operands;
                let typed_lhs = lhs.typecheck(env, scope_id)?;
                let typed_rhs = rhs.typecheck(env, scope_id)?;
                typed_lhs.expect_type(&Typed::Bool, env)?;
                typed_rhs.expect_type(&Typed::Bool, env)?;

                Expr {
                    location: self.location,
                    kind: ExprKind::And(Box::new((typed_lhs, typed_rhs))),
                    type_data: Typed::Bool,
                }
            }
            ExprKind::Or(operands) => {
                let (lhs, rhs) = *operands;
                let typed_lhs = lhs.typecheck(env, scope_id)?;
                let typed_rhs = rhs.typecheck(env, scope_id)?;
                typed_lhs.expect_type(&Typed::Bool, env)?;
                typed_rhs.expect_type(&Typed::Bool, env)?;

                Expr {
                    location: self.location,
                    kind: ExprKind::Or(Box::new((typed_lhs, typed_rhs))),
                    type_data: Typed::Bool,
                }
            }

            ExprKind::LessThan(operands) => Self::typecheck_cmp_binop(
                *operands,
                ExprKind::LessThan,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::GreaterThan(operands) => Self::typecheck_cmp_binop(
                *operands,
                ExprKind::GreaterThan,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::LessThanEq(operands) => Self::typecheck_cmp_binop(
                *operands,
                ExprKind::LessThanEq,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::GreaterThanEq(operands) => Self::typecheck_cmp_binop(
                *operands,
                ExprKind::GreaterThanEq,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::Eq(operands) => {
                Self::typecheck_eq_binop(*operands, ExprKind::Eq, self.location, env, scope_id)?
            }
            ExprKind::NotEq(operands) => {
                Self::typecheck_eq_binop(*operands, ExprKind::NotEq, self.location, env, scope_id)?
            }
            ExprKind::Add(operands) => Self::typecheck_numerical_binop(
                *operands,
                ExprKind::Add,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::Minus(operands) => Self::typecheck_numerical_binop(
                *operands,
                ExprKind::Minus,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::Mulitply(operands) => Self::typecheck_numerical_binop(
                *operands,
                ExprKind::Mulitply,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::Divide(operands) => Self::typecheck_numerical_binop(
                *operands,
                ExprKind::Divide,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::Modulo(operands) => Self::typecheck_numerical_binop(
                *operands,
                ExprKind::Modulo,
                self.location,
                env,
                scope_id,
            )?,
            ExprKind::Not(expr) => {
                let typed_expr = expr.typecheck(env, scope_id)?;
                typed_expr.expect_type(&Typed::Bool, env)?;

                Expr {
                    location: self.location,
                    kind: ExprKind::Not(Box::new(typed_expr)),
                    type_data: Typed::Bool,
                }
            }
            ExprKind::Negation(expr) => {
                let typed_expr = expr.typecheck(env, scope_id)?;
                typed_expr.expect_one_of_types(&[Typed::Int, Typed::Float], env)?;
                let output_type = typed_expr.type_data.clone();

                Expr {
                    location: self.location,
                    kind: ExprKind::Negation(Box::new(typed_expr)),
                    type_data: output_type,
                }
            }
        })
    }

    pub fn to_s_expr(&self, src: &[u8]) -> String {
        self.to_s_expr_general(src, None::<fn(&Expr<UnTyped>) -> String>)
    }
}

impl<T: TypeState> Expr<T> {
    pub fn location(&self) -> Span {
        self.location
    }

    pub fn kind(&self) -> &ExprKind<T> {
        &self.kind
    }

    pub fn to_s_expr_general(
        &self,
        src: &[u8],
        expr_printer: Option<impl Fn(&Expr<T>) -> String>,
    ) -> String {
        let expr_printer: Box<dyn Fn(&Expr<T>) -> String> = match expr_printer {
            Some(expr_printer) => Box::new(expr_printer),
            None => {
                Box::new(|e: &Expr<T>| e.to_s_expr_general(src, None::<fn(&Expr<T>) -> String>))
            }
        };

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
                    s_expr.push_str(&expr_printer(item));
                }
                s_expr.push(')');
                s_expr
            }
            ExprKind::StructInit(span, fields) => {
                let mut s_expr = format!("(StructLiteralExpr {}", span.as_str(src));

                for expr in fields {
                    s_expr.push(' ');
                    s_expr.push_str(&expr_printer(expr));
                }

                s_expr.push(')');

                s_expr
            }
            ExprKind::FunctionCall(span, args) => {
                let mut s_expr = format!("(CallExpr {}", span.as_str(src));
                for arg in args {
                    s_expr.push(' ');
                    s_expr.push_str(&expr_printer(arg));
                }
                s_expr.push(')');
                s_expr
            }
            ExprKind::FieldAccess(expr, span) => {
                format!("(DotExpr {} {})", &expr_printer(expr), span.as_str(src))
            }
            ExprKind::ArrayIndex(expr, indices) => {
                let mut s_expr = format!("(ArrayIndexExpr {}", expr_printer(expr));
                for index in indices {
                    s_expr.push(' ');
                    s_expr.push_str(&expr_printer(index));
                }
                s_expr.push(')');
                s_expr
            }
            ExprKind::If(if_stmt) => format!(
                "(IfExpr {} {} {})",
                expr_printer(&if_stmt.0),
                expr_printer(&if_stmt.1),
                expr_printer(&if_stmt.2)
            ),
            ExprKind::ArrayComp(args, expr, _) => {
                let mut s_expr = "(ArrayLoopExpr ".to_string();
                for (var, expr) in args {
                    s_expr.push_str(var.as_str(src));
                    s_expr.push(' ');

                    s_expr.push_str(&expr_printer(expr));
                    s_expr.push(' ');
                }
                s_expr.push_str(&expr_printer(expr));
                s_expr.push(')');
                s_expr
            }
            ExprKind::Sum(args, expr, _) => {
                let mut s_expr = "(SumLoopExpr ".to_string();
                for (var, expr) in args {
                    s_expr.push_str(var.as_str(src));
                    s_expr.push(' ');

                    s_expr.push_str(&expr_printer(expr));
                    s_expr.push(' ');
                }
                s_expr.push_str(&expr_printer(expr));
                s_expr.push(')');
                s_expr
            }
            ExprKind::And(operands) => format!(
                "(BinopExpr {} && {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Or(operands) => format!(
                "(BinopExpr {} || {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::LessThan(operands) => format!(
                "(BinopExpr {} < {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::GreaterThan(operands) => format!(
                "(BinopExpr {} > {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::LessThanEq(operands) => format!(
                "(BinopExpr {} <= {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::GreaterThanEq(operands) => format!(
                "(BinopExpr {} >= {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Eq(operands) => format!(
                "(BinopExpr {} == {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::NotEq(operands) => format!(
                "(BinopExpr {} != {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Add(operands) => format!(
                "(BinopExpr {} + {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Minus(operands) => format!(
                "(BinopExpr {} - {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Mulitply(operands) => format!(
                "(BinopExpr {} * {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Divide(operands) => format!(
                "(BinopExpr {} / {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Modulo(operands) => format!(
                "(BinopExpr {} % {})",
                expr_printer(&operands.0),
                expr_printer(&operands.1)
            ),
            ExprKind::Not(expr) => format!("(UnopExpr ! {})", expr_printer(expr)),
            ExprKind::Negation(expr) => format!("(UnopExpr - {})", expr_printer(expr)),
            ExprKind::Paren(expr) => expr.to_s_expr_general(src, Some(expr_printer)),
        }
    }
}
