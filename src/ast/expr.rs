// TODO: merge typechecking and parsing

//! Defines the types of functions to parse all kinds of expressions in JPL
use std::fmt::{Formatter, Write};

use super::{super::parse::parse_sequence, auxiliary::LoopVar, expect_tokens, Parse, TokenStream};
use crate::{
    environment::Environment,
    lex::TokenType,
    parse::{Displayable, SExpr, SExprOptions},
    typecheck::TypeVal,
    utils::Span,
};
use miette::{miette, LabeledSpan, Severity};

//TODO: allow numbers one bigger due to negative numbers
const POSITIVE_INT_LIT_MAX: u64 = 9223372036854775807;

/// Represents an expression in JPL
#[derive(Debug, Clone)]
pub struct Expr {
    loc: Span,
    kind: ExprKind,
    type_data: TypeVal,
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
    // Simple expressions
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

    // Number represents their scope
    ArrayComp(Box<[LoopVar]>, Box<Expr>, usize),
    Sum(Box<[LoopVar]>, Box<Expr>, usize),

    // Bool ops
    And(Box<[Expr; 2]>),
    Or(Box<[Expr; 2]>),

    // Comparisons: <, >, <=, and >=, ==, !=
    LessThan(Box<[Expr; 2]>),
    GreaterThan(Box<[Expr; 2]>),
    LessThanEq(Box<[Expr; 2]>),
    GreaterThanEq(Box<[Expr; 2]>),
    Eq(Box<[Expr; 2]>),
    NotEq(Box<[Expr; 2]>),

    // Additive operations + and -	}
    Add(Box<[Expr; 2]>),
    Minus(Box<[Expr; 2]>),

    // Multiplicative operations *, /, and %
    Mulitply(Box<[Expr; 2]>),
    Divide(Box<[Expr; 2]>),
    Modulo(Box<[Expr; 2]>),

    // Unary inverse ! and negation -
    Not(Box<Expr>),
    Negation(Box<Expr>),
}

impl Parse for Expr {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        Self::parse_control(ts, env)
    }
}

type VariantBuilder = fn(Box<[Expr; 2]>) -> ExprKind;

fn parse_binary_op(
    ts: &mut TokenStream,
    env: &mut Environment,
    ops: &[(&str, VariantBuilder)],
    mut type_checker: impl FnMut(&Expr, &Expr, &Environment, &str) -> miette::Result<TypeVal>,
    mut sub_class: impl FnMut(&mut TokenStream, &mut Environment) -> miette::Result<Expr>,
) -> miette::Result<Expr> {
    let mut lhs: Expr = sub_class(ts, env)?;
    'outer: loop {
        for (op_as_str, op_var) in ops {
            match ts.peek() {
                Some(t) if t.kind() == TokenType::Op && t.bytes() == *op_as_str => {
                    _ = expect_tokens(ts, [TokenType::Op])?;
                    // Be able to parse a control on the rhs. While control does have a low
                    // precedence there is no ambiguity when it is on the right
                    // ex. 1 + sum[i: 50] i should parse
                    let rhs = match ts.peek_type() {
                        Some(TokenType::Array) => Expr::parse_array_comp(ts, env),
                        Some(TokenType::Sum) => Expr::parse_sum(ts, env),
                        Some(TokenType::If) => Expr::parse_if(ts, env),
                        _ => sub_class(ts, env),
                    }?;
                    let location = lhs.loc.join(rhs.loc);
                    let type_data = type_checker(&lhs, &rhs, env, op_as_str)?;
                    lhs = Expr {
                        loc: location,
                        kind: op_var(Box::new([lhs, rhs])),
                        type_data,
                    };
                    continue 'outer;
                }
                _ => (),
            }
        }

        return Ok(lhs);
    }
}

//TODO move to auxiliary

impl Expr {
    pub fn loc(&self) -> Span {
        self.loc
    }

    pub fn kind(&self) -> &ExprKind {
        &self.kind
    }

    pub fn type_data(&self) -> &TypeVal {
        &self.type_data
    }

    fn parse_control(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        match ts.peek_type() {
            Some(TokenType::If) => Self::parse_if(ts, env),
            Some(TokenType::Array) => Self::parse_array_comp(ts, env),
            Some(TokenType::Sum) => Self::parse_sum(ts, env),
            _ => Self::parse_bool(ts, env),
        }
    }

    fn parse_if(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [if_token] = expect_tokens(ts, [TokenType::If])?;
        let cond = Expr::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Then])?;

        let true_expr = Expr::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Else])?;
        let false_expr = Expr::parse(ts, env)?;

        let loc = if_token.loc().join(false_expr.loc);

        cond.expect_type(&TypeVal::Bool, env)?;
        let branch_type = &true_expr.type_data.clone();

        false_expr.expect_type(branch_type, env)?;
        Ok(Self {
            loc,
            kind: ExprKind::If(Box::new((cond, true_expr, false_expr))),
            type_data: branch_type.clone(),
        })
    }
    fn parse_array_comp(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [arr_token, _] = expect_tokens(ts, [TokenType::Array, TokenType::LSquare])?;
        let scope = env.new_scope();
        let looping_vars = parse_sequence(ts, env, TokenType::Comma, TokenType::RSquare)?;
        let [r_square_token] = expect_tokens(ts, [TokenType::RSquare])?;
        let bindings_loc = arr_token.loc().join(r_square_token.loc());

        if looping_vars.is_empty() {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some("array comprehension must have at least 1 looping variable".to_string()),
                    bindings_loc.start(),
                    bindings_loc.len()
                )],
                "Can not create a zero rank array"
            ));
        }

        if looping_vars.len() > u8::MAX as usize {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "array comprehension can only have {} looping variables",
                        u8::MAX
                    )),
                    bindings_loc.start(),
                    bindings_loc.len()
                )],
                "Can not create a zero rank array"
            ));
        }

        for LoopVar(name, val) in &looping_vars {
            val.expect_type(&TypeVal::Int, env)?;
        }
        env.add_loop_bounds(&looping_vars)?;

        let body = Self::parse(ts, env)?;
        let loc = arr_token.loc().join(body.loc);
        let type_data = TypeVal::Array(Box::new(body.type_data.clone()), looping_vars.len() as u8);
        env.end_scope();

        Ok(Self {
            loc,
            kind: ExprKind::ArrayComp(looping_vars, Box::new(body), scope),
            type_data,
        })
    }
    fn parse_sum(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [arr_token, _] = expect_tokens(ts, [TokenType::Sum, TokenType::LSquare])?;
        let scope = env.new_scope();
        let looping_vars = parse_sequence(ts, env, TokenType::Comma, TokenType::RSquare)?;
        let [r_square_token] = expect_tokens(ts, [TokenType::RSquare])?;
        let loc = arr_token.loc().join(r_square_token.loc());

        if looping_vars.is_empty() {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some("sum loops must have at least 1 looping variable".to_string()),
                    loc.start(),
                    loc.len()
                )],
                "Illegal sum expresion"
            ));
        }
        if looping_vars.len() > u8::MAX as usize {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "array comprehension can only have {} binding",
                        u8::MAX
                    )),
                    loc.start(),
                    loc.len()
                )],
                "Illegal sum loop"
            ));
        }

        for LoopVar(_, val) in &looping_vars {
            val.expect_type(&TypeVal::Int, env)?;
        }
        env.add_loop_bounds(&looping_vars)?;

        let body = Self::parse(ts, env)?;
        body.expect_one_of_types(&[TypeVal::Int, TypeVal::Float], env)?;
        let loc = arr_token.loc().join(body.loc);
        env.end_scope();

        let type_data = body.type_data.clone();
        Ok(Self {
            loc,
            kind: ExprKind::Sum(looping_vars, Box::new(body), scope),
            type_data,
        })
    }
    fn parse_bool(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            env,
            &[("||", ExprKind::Or), ("&&", ExprKind::And)],
            |lhs, rhs, env, _| {
                lhs.expect_type(&TypeVal::Bool, env)?;
                rhs.expect_type(&TypeVal::Bool, env)?;
                Ok(TypeVal::Bool)
            },
            Self::parse_cmp,
        )
    }

    fn parse_cmp(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            env,
            &[
                (">", ExprKind::GreaterThan),
                ("<", ExprKind::LessThan),
                ("<=", ExprKind::LessThanEq),
                (">=", ExprKind::GreaterThanEq),
                ("==", ExprKind::Eq),
                ("!=", ExprKind::NotEq),
            ],
            |lhs, rhs, env, op| match op {
                "==" | "!=" => {
                    lhs.expect_one_of_types(&[TypeVal::Int, TypeVal::Float, TypeVal::Bool], env)?;
                    rhs.expect_type(&lhs.type_data, env)?;

                    Ok(TypeVal::Bool)
                }
                _ => {
                    lhs.expect_one_of_types(&[TypeVal::Int, TypeVal::Float], env)?;
                    rhs.expect_type(&lhs.type_data, env)?;
                    Ok(TypeVal::Bool)
                }
            },
            Self::parse_add,
        )
    }

    fn parse_add(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            env,
            &[("+", ExprKind::Add), ("-", ExprKind::Minus)],
            |lhs, rhs, env, _| {
                lhs.expect_one_of_types(&[TypeVal::Int, TypeVal::Float], env)?;
                rhs.expect_type(&lhs.type_data, env)?;
                Ok(lhs.type_data.clone())
            },
            Self::parse_mult,
        )
    }
    fn parse_mult(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        parse_binary_op(
            ts,
            env,
            &[
                ("*", ExprKind::Mulitply),
                ("/", ExprKind::Divide),
                ("%", ExprKind::Modulo),
            ],
            |lhs, rhs, env, _| {
                lhs.expect_one_of_types(&[TypeVal::Int, TypeVal::Float], env)?;
                rhs.expect_type(&lhs.type_data, env)?;
                Ok(lhs.type_data.clone())
            },
            Self::parse_unary,
        )
    }

    fn parse_unary(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let (expr_kind, ty): (fn(_) -> _, &[TypeVal]) = match ts.peek() {
            Some(t) if t.kind() == TokenType::Op && t.bytes() == "!" => {
                (ExprKind::Not, &[TypeVal::Bool])
            }
            Some(t) if t.kind() == TokenType::Op && t.bytes() == "-" => {
                (ExprKind::Negation, &[TypeVal::Int, TypeVal::Float])
            }
            Some(t) if t.kind() == TokenType::Array => return Expr::parse_array_comp(ts, env),
            Some(t) if t.kind() == TokenType::Sum => return Expr::parse_sum(ts, env),
            Some(t) if t.kind() == TokenType::If => return Expr::parse_if(ts, env),
            _ => return Self::parse_simple(ts, env),
        };

        let [unary_op_token] = expect_tokens(ts, [TokenType::Op])?;
        let inner = Self::parse_unary(ts, env)?;
        let loc = unary_op_token.loc().join(inner.loc);
        let type_data = inner.type_data.clone();
        inner.expect_one_of_types(ty, env)?;

        Ok(Self {
            loc,
            kind: expr_kind(Box::new(inner)),
            type_data,
        })
    }

    fn parse_simple(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let mut expr = match (ts.peek_type(), ts.peek_type_at(2)) {
            (Some(TokenType::LParen), _) => {
                let [l_paren] = expect_tokens(ts, [TokenType::LParen])?;
                let expr = Self::parse(ts, env)?;
                let [r_paren] = expect_tokens(ts, [TokenType::RParen])?;
                let type_data = expr.type_data.clone();

                Ok(Self {
                    loc: l_paren.loc().join(r_paren.loc()),
                    kind: ExprKind::Paren(Box::new(expr)),
                    type_data,
                })
            }
            (Some(TokenType::IntLit), _) => Self::parse_int_lit(ts),
            (Some(TokenType::FloatLit), _) => Self::parse_float_lit(ts),
            (Some(TokenType::True), _) => Self::parse_true(ts),
            (Some(TokenType::False), _) => Self::parse_false(ts),
            (Some(TokenType::Void), _) => Self::parse_void(ts),
            (Some(TokenType::LSquare), _) => Self::parse_array_lit(ts, env),
            (Some(TokenType::Variable), Some(TokenType::LCurly)) => {
                Self::parse_struct_init(ts, env)
            }
            (Some(TokenType::Variable), Some(TokenType::LParen)) => Self::parse_fn_call(ts, env),
            (Some(TokenType::Variable), _) => Self::parse_var(ts, env),
            (Some(t), _) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected expression, found: {}", t)),
                    ts.peek().unwrap().loc().start(),
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
                    let [field_name] = expect_tokens(ts, [TokenType::Variable])?;
                    let location = expr.loc.join(field_name.loc());
                    let id = match &expr.type_data {
                        TypeVal::Struct(id) => *id,
                        _ => 0,
                    };

                    expr.expect_type(&TypeVal::Struct(id), env)?;

                    let field_name_loc = field_name.loc();
                    let field_name = field_name_loc.as_str(env.src());
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
                                    field_name_loc.start(),
                                    field_name_loc.len(),
                                )],
                                "Struct field name not changed"
                            )
                        })?
                        .clone();

                    expr = Self {
                        loc: location,
                        kind: ExprKind::FieldAccess(Box::new(expr), field_name_loc),
                        type_data: field_type,
                    }
                }
                Some(TokenType::LSquare) => {
                    expect_tokens(ts, [TokenType::LSquare])?;
                    let indices: Box<[Expr]> =
                        parse_sequence(ts, env, TokenType::Comma, TokenType::RSquare)?;
                    let [r_square_token] = expect_tokens(ts, [TokenType::RSquare])?;
                    let loc = expr.loc.join(r_square_token.loc());

                    let (element_type, rank) = match &expr.type_data {
                        TypeVal::Array(element_type, rank) => (*element_type.clone(), *rank),
                        t => {
                            return Err(miette!(
                                severity = Severity::Error,
                                labels = vec![LabeledSpan::new(
                                    Some(format!(
                                        "Expected type array type, found: {}",
                                        t.as_str(env)
                                    )),
                                    expr.loc.start(),
                                    expr.loc.len()
                                )],
                                "Can only index arrays"
                            ));
                        }
                    };

                    if rank as usize != indices.len() {
                        return Err(miette!(
                            severity = Severity::Error,
                            labels = vec![LabeledSpan::new(
                                Some(format!("Expected {} indices found {}", rank, indices.len())),
                                expr.loc.start(),
                                expr.loc.len()
                            )],
                            "Incorrect amount of indices"
                        ));
                    }

                    for index in &indices {
                        index.expect_type(&TypeVal::Int, env)?;
                    }
                    expr = Self {
                        loc,
                        kind: ExprKind::ArrayIndex(Box::new(expr), indices),
                        type_data: element_type.clone(),
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
            loc: int_lit_token.loc(),
            kind: ExprKind::IntLit(int_val),
            type_data: TypeVal::Int,
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
            loc: float_lit_token.loc(),
            kind: ExprKind::FloatLit(float_val),
            type_data: TypeVal::Float,
        })
    }

    fn parse_true(ts: &mut TokenStream) -> miette::Result<Self> {
        let [true_token] = expect_tokens(ts, [TokenType::True])?;
        Ok(Self {
            loc: true_token.loc(),
            kind: ExprKind::True,
            type_data: TypeVal::Bool,
        })
    }

    fn parse_false(ts: &mut TokenStream) -> miette::Result<Self> {
        let [false_token] = expect_tokens(ts, [TokenType::False])?;
        Ok(Self {
            loc: false_token.loc(),
            kind: ExprKind::False,
            type_data: TypeVal::Bool,
        })
    }

    fn parse_void(ts: &mut TokenStream) -> miette::Result<Self> {
        let [void_token] = expect_tokens(ts, [TokenType::Void])?;
        Ok(Self {
            loc: void_token.loc(),
            kind: ExprKind::Void,
            type_data: TypeVal::Void,
        })
    }

    fn parse_var(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        let var_type = env.get_variable_type(var_token.loc())?;
        Ok(Self {
            loc: var_token.loc(),
            kind: ExprKind::Var,
            type_data: var_type.clone(),
        })
    }

    fn parse_array_lit(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [l_square_token] = expect_tokens(ts, [TokenType::LSquare])?;
        let exprs: Box<[Expr]> = parse_sequence(ts, env, TokenType::Comma, TokenType::RSquare)?;
        let [rb_token] = expect_tokens(ts, [TokenType::RSquare])?;
        let loc = l_square_token.loc().join(rb_token.loc());

        if exprs.is_empty() {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some("array literals can not be empty".to_string()),
                    loc.start(),
                    loc.len()
                )],
                "Invalid array literal"
            ));
        }

        let element_type = exprs[0].type_data.clone();
        exprs
            .iter()
            .skip(1)
            .map(|e| e.expect_type(&element_type, env))
            .collect::<Result<Vec<()>, _>>()?;

        Ok(Self {
            loc,
            kind: ExprKind::ArrayLit(exprs),
            type_data: TypeVal::Array(Box::new(element_type.clone()), 1),
        })
    }

    fn parse_struct_init(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [struct_name] = expect_tokens(ts, [TokenType::Variable])?;
        expect_tokens(ts, [TokenType::LCurly])?;
        let fields: Box<[Expr]> = parse_sequence(ts, env, TokenType::Comma, TokenType::RCurly)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RCurly])?;
        let loc = struct_name.loc().join(r_curly_token.loc());
        let info = env.get_struct(struct_name.loc())?;

        if fields.len() != info.fields().len() {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "Expected {} fields, found {}",
                        info.fields().len(),
                        fields.len()
                    )),
                    loc.start(),
                    loc.len(),
                )],
                "Incorrect number of fields for struct: {}",
                struct_name.loc().as_str(env.src())
            ));
        }

        for (expr, (_, ty)) in fields.to_vec().into_iter().zip(info.fields()) {
            expr.expect_type(ty, env)?;
        }

        Ok(Self {
            loc,
            kind: ExprKind::StructInit(struct_name.loc(), fields),
            type_data: TypeVal::Struct(info.id()),
        })
    }

    fn parse_fn_call(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [fn_name] = expect_tokens(ts, [TokenType::Variable])?;
        _ = expect_tokens(ts, [TokenType::LParen])?;
        let args: Box<[Expr]> = parse_sequence(ts, env, TokenType::Comma, TokenType::RParen)?;
        let [r_curly_token] = expect_tokens(ts, [TokenType::RParen])?;
        let loc = fn_name.loc().join(r_curly_token.loc());

        let fn_info = env.get_function(fn_name.loc())?;
        if args.len() != fn_info.args().len() {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "Expected {} arguments, found {}",
                        fn_info.args().len(),
                        args.len(),
                    )),
                    loc.start(),
                    loc.len(),
                )],
                "Incorrect number of arguments for function: {}",
                fn_name.loc().as_str(env.src())
            ));
        }

        let fn_info = env.get_function(fn_name.loc())?;

        for (arg, expected_type) in args.iter().zip(fn_info.args()) {
            arg.expect_type(expected_type, env)?;
        }

        Ok(Expr {
            loc,
            kind: ExprKind::FunctionCall(fn_name.loc(), args),
            type_data: fn_info.ret().clone(),
        })
    }

    pub fn expect_type(&self, expected: &TypeVal, env: &Environment) -> miette::Result<()> {
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
                self.loc.start(),
                self.loc.len(),
            )],
            "Type mismatch"
        ))
    }

    pub fn expect_one_of_types(
        &self,
        expected: &[TypeVal],
        env: &Environment,
    ) -> miette::Result<()> {
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
                self.loc.start(),
                self.loc.len(),
            )],
            "Type mismatch"
        ))
    }

    pub fn expect_array_of_rank(
        &self,
        expected_rank: usize,
        env: &Environment,
    ) -> miette::Result<()> {
        match self.type_data() {
            TypeVal::Array(_, rank) if *rank as usize == expected_rank => Ok(()),
            TypeVal::Array(_, rank) => Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!(
                        "expected array of rank: {}, found array of rank: {}",
                        expected_rank, rank,
                    )),
                    self.loc().start(),
                    self.loc().len(),
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
                    self.loc().start(),
                    self.loc().len(),
                )],
                "Unexpected token found"
            )),
        }
    }
}

impl SExpr for Expr {
    fn to_s_expr(
        &self,
        f: &mut Formatter<'_>,
        env: &Environment<'_>,
        opt: SExprOptions,
    ) -> std::fmt::Result {
        let ty = Displayable(&self.type_data, env, opt);
        match &self.kind {
            ExprKind::IntLit(val) => write!(f, "(IntExpr{} {})", ty, val),
            ExprKind::FloatLit(val) => write!(f, "(FloatExpr{} {})", ty, val.trunc()),
            ExprKind::True => write!(f, "(TrueExpr{})", ty),
            ExprKind::False => write!(f, "(FalseExpr{})", ty),
            ExprKind::Var => write!(f, "(VarExpr{} {})", ty, self.loc.as_str(env.src())),
            ExprKind::Void => write!(f, "(VoidExpr{})", ty),
            ExprKind::ArrayLit(items) => {
                write!(
                    f,
                    "(ArrayLiteralExpr{}{})",
                    ty,
                    Displayable(items, env, opt),
                )
            }
            ExprKind::StructInit(struct_name, fields) => {
                write!(
                    f,
                    "(StructLiteralExpr{} {}{})",
                    ty,
                    struct_name.as_str(env.src()),
                    Displayable(fields, env, opt),
                )
            }
            ExprKind::FunctionCall(span, args) => write!(
                f,
                "(CallExpr{} {}{})",
                ty,
                span.as_str(env.src()),
                Displayable(args, env, opt),
            ),
            ExprKind::FieldAccess(expr, span) => {
                write!(
                    f,
                    "(DotExpr{} {} {})",
                    ty,
                    Displayable(expr.as_ref(), env, opt),
                    span.as_str(env.src())
                )
            }
            ExprKind::ArrayIndex(expr, indices) => {
                write!(
                    f,
                    "(ArrayIndexExpr{} {}{})",
                    ty,
                    Displayable(expr.as_ref(), env, opt),
                    Displayable(indices, env, opt)
                )
            }
            ExprKind::If(if_stmt) => write!(
                f,
                "(IfExpr{} {} {} {})",
                ty,
                Displayable(&if_stmt.0, env, opt),
                Displayable(&if_stmt.1, env, opt),
                Displayable(&if_stmt.2, env, opt)
            ),
            ExprKind::ArrayComp(args, expr, _) => {
                write!(
                    f,
                    "(ArrayLoopExpr{}{} {})",
                    ty,
                    Displayable(args, env, opt),
                    Displayable(expr.as_ref(), env, opt),
                )
            }
            ExprKind::Sum(args, expr, _) => {
                write!(
                    f,
                    "(SumLoopExpr{}{} {})",
                    ty,
                    Displayable(args, env, opt),
                    Displayable(expr.as_ref(), env, opt),
                )
            }
            ExprKind::And(operands) => write!(
                f,
                "(BinopExpr{} {} && {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Or(operands) => write!(
                f,
                "(BinopExpr{} {} || {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::LessThan(operands) => write!(
                f,
                "(BinopExpr{} {} < {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::GreaterThan(operands) => write!(
                f,
                "(BinopExpr{} {} > {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::LessThanEq(operands) => write!(
                f,
                "(BinopExpr{} {} <= {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::GreaterThanEq(operands) => write!(
                f,
                "(BinopExpr{} {} >= {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Eq(operands) => write!(
                f,
                "(BinopExpr{} {} == {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::NotEq(operands) => write!(
                f,
                "(BinopExpr{} {} != {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Add(operands) => write!(
                f,
                "(BinopExpr{} {} + {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Minus(operands) => write!(
                f,
                "(BinopExpr{} {} - {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Mulitply(operands) => write!(
                f,
                "(BinopExpr{} {} * {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Divide(operands) => write!(
                f,
                "(BinopExpr{} {} / {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Modulo(operands) => write!(
                f,
                "(BinopExpr{} {} % {})",
                ty,
                Displayable(&operands[0], env, opt),
                Displayable(&operands[1], env, opt)
            ),
            ExprKind::Not(expr) => write!(
                f,
                "(UnopExpr{} ! {})",
                ty,
                Displayable(expr.as_ref(), env, opt)
            ),
            ExprKind::Negation(expr) => write!(
                f,
                "(UnopExpr{} - {})",
                ty,
                Displayable(expr.as_ref(), env, opt)
            ),
            ExprKind::Paren(expr) => write!(f, "{}", Displayable(expr.as_ref(), env, opt)),
        }
    }
}
