use std::fmt::Write;

use crate::{
    environment::Environment,
    lex::TokenType,
    parse::{expect_tokens, next_match, parse_sequence, Displayable, Parse, SExpr, TokenStream},
    utils::Span,
};

use super::{expr::Expr, types::Type};

/// Reprsents a string literal in the source code.
/// The span includes the opening and closing quote.
#[derive(Debug, Clone, Copy)]
pub struct Str(pub Span);

impl Str {
    pub fn loc(&self) -> Span {
        self.0
    }
    pub fn inner_loc(&self) -> Span {
        Span::new(self.0.start() + 1, self.0.len() - 2)
    }
}
impl Parse for Str {
    fn parse(ts: &mut TokenStream, _: &mut Environment) -> miette::Result<Self> {
        let [str_token] = expect_tokens(ts, [TokenType::StringLit])?;

        Ok(Self(str_token.loc()))
    }
}

impl SExpr for Str {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &crate::environment::Environment<'_>,
        _: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        write!(f, "{}", self.0.as_str(env.src()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Var(pub Span);

impl Var {
    pub fn loc(&self) -> Span {
        self.0
    }
}

/// Span acts a a varible in this case
impl Parse for Var {
    fn parse(ts: &mut TokenStream, _: &mut Environment) -> miette::Result<Self> {
        let [var_token] = expect_tokens(ts, [TokenType::Variable])?;
        Ok(Self(var_token.loc()))
    }
}

impl SExpr for Var {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &crate::environment::Environment<'_>,
        _: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        write!(f, "{}", self.0.as_str(env.src()))
    }
}
/// Represents an left value used in let statements and commands
#[derive(Debug, Clone)]
pub struct LValue {
    loc: Span,
    var: Var,
    arr_bindings: Option<Box<[Var]>>,
}

impl SExpr for LValue {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &crate::environment::Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        match &self.arr_bindings {
            //(FnCmd x (((ArrayLValue y H) (IntType))) (VoidType))
            Some(array_bindings) => {
                write!(f, "(ArrayLValue {}", Displayable(&self.var, env, opt))?;
                for binding in array_bindings {
                    write!(f, " {}", Displayable(binding, env, opt))?;
                }
                f.write_char(')')
            }
            None => write!(f, "(VarLValue {})", Displayable(&self.var, env, opt),),
        }
    }
}

impl LValue {
    pub fn array_bindings(&self) -> Option<&[Var]> {
        self.arr_bindings.as_deref()
    }

    pub fn variable(&self) -> &Var {
        &self.var
    }

    pub fn from_span(loc: Span) -> Self {
        LValue {
            loc,
            var: Var(loc),
            arr_bindings: None,
        }
    }

    pub fn loc(&self) -> Span {
        self.loc
    }
}
impl Parse for LValue {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let var = Var::parse(ts, env)?;
        let mut loc = var.loc();
        let array_bindings = if next_match!(ts, TokenType::LSquare) {
            expect_tokens(ts, [TokenType::LSquare])?;
            let bindings = parse_sequence(ts, env, TokenType::Comma, TokenType::RSquare)?;
            let [r_square_token] = expect_tokens(ts, [TokenType::RSquare])?;
            loc = loc.join(r_square_token.loc());
            Some(bindings)
        } else {
            None
        };

        Ok(Self {
            var,
            arr_bindings: array_bindings,
            loc,
        })
    }
}

/// binding : <lvalue> : <type>
#[derive(Debug, Clone)]
pub struct Binding {
    loc: Span,
    lvalue: LValue,
    var_type: Type,
}

impl Binding {
    pub fn loc(&self) -> Span {
        self.loc
    }

    pub fn var_type(&self) -> &Type {
        &self.var_type
    }

    pub fn lvalue(&self) -> &LValue {
        &self.lvalue
    }
}

impl Parse for Binding {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let lvalue = LValue::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Colon])?;
        let variable_type = Type::parse(ts, env)?;
        let loc = lvalue.loc.join(variable_type.location());

        Ok(Self {
            loc,
            lvalue,
            var_type: variable_type,
        })
    }
}

impl SExpr for Binding {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &crate::environment::Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            Displayable(&self.lvalue, env, opt),
            Displayable(&self.var_type, env, opt)
        )
    }
}

#[derive(Debug, Clone)]
pub struct StructField(pub Var, pub Type);

impl Parse for StructField {
    /// Parses <variable>: <type>
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let var = Var::parse(ts, env)?;
        expect_tokens(ts, [TokenType::Colon])?;
        let var_type = Type::parse(ts, env)?;
        Ok(StructField(var, var_type))
    }
}

impl SExpr for StructField {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &crate::environment::Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            Displayable(&self.0, env, opt),
            Displayable(&self.1, env, opt),
        )
    }
}

#[derive(Debug, Clone)]
pub struct LoopVar(pub Span, pub Expr);
/// Used for sum and array looping constructs
impl Parse for LoopVar {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        let [var, _] = expect_tokens(ts, [TokenType::Variable, TokenType::Colon])?;
        let expr = Expr::parse(ts, env)?;
        Ok(LoopVar(var.loc(), expr))
    }
}

impl SExpr for LoopVar {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            self.0.as_str(env.src()),
            Displayable(&self.1, env, opt),
        )
    }
}
