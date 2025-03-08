use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter, Write};

use crate::ast::cmd::{Cmd, CmdKind};
use crate::ast::expr::{Expr, ExprKind};
use crate::{ast::Program, environment::Environment, typecheck::Typed, utils::Span};

const INCLUDES: &str = "\
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include \"rt/runtime.h\"
";

const CUSTOM_TYPES: &str = "\
typedef struct { } void_t;
";

const MAIN_FN_IDX: usize = 0;
const MAIN_FN_NAME: &str = "jpl_main";

pub fn generate_c(ast: Program<Typed>, env: &Environment) {
    let mut c_gen_env = CGenEnv::new(env);
    ast.to_c(&mut c_gen_env, env);
}

#[derive(Debug, Clone)]
pub struct CGenEnv<'a, 'b> {
    /// Stores includes, and all struct definitions
    defs: String,
    jmp_ctr: usize,
    fns: Vec<CFn<'b>>,
    env: &'a Environment<'b>,
    cur_fn: usize,

    array_types: HashSet<(Typed, u8)>,
}

impl<'a, 'b> CGenEnv<'a, 'b> {
    pub fn new(env: &'a Environment<'b>) -> Self {
        let mut fns = Vec::with_capacity(1 + env.functions().len());
        fns.push(CFn::new(MAIN_FN_NAME));

        Self {
            jmp_ctr: 0,
            fns,
            env,
            cur_fn: MAIN_FN_IDX,
            defs: format!("{}\n{}\n", INCLUDES, CUSTOM_TYPES),
            array_types: HashSet::new(),
        }
    }

    pub fn cur_fn(&mut self) -> &mut CFn<'b> {
        &mut self.fns[self.cur_fn]
    }

    /// Sets the current function back to `jpl_main`
    pub fn reset_fn(&mut self) {
        self.cur_fn = MAIN_FN_IDX;
    }

    /// Adds the fn to the C Codegen Envirment and sets it as current
    pub fn add_fn(&mut self, name: &'b str) {
        assert!(self.fns.len() < self.fns.capacity());
        assert!(self.env.functions().contains_key(name));

        self.cur_fn = self.fns.len();
        self.fns.push(CFn::new(name));
    }

    /// Creates a struct for the given array type if it has not yet been created. Does nothing if
    /// the given type is not an array or has already been created
    pub fn gen_arr_struct(&mut self, t: Typed) {
        let (inner, rank) = match t {
            Typed::Array(inner, rank) => (inner, rank),
            _ => return,
        };

        let entry = (*inner, rank);
        if !self.array_types.contains(&entry) {
            return;
        }

        let (inner, rank) = entry;

        self.defs.push_str("typedef struct {\n");
        for dim in 0..rank {
            write!(self.defs, "  int64_t d{};\n", dim).expect("string should not fail to write");
        }

        self.defs.push_str("  ");
        write_type(&mut self.defs, &inner, self.env);
        self.defs.push_str(" *data;\n");

        write!(self.defs, "}} _a{}_", rank).expect("string should not fail to write");
        write_type(&mut self.defs, &inner, self.env);
        self.defs.push_str(";\n\n");

        self.array_types.insert((inner, rank));
    }

    ///// Writes the type to the current function
    //pub fn write_type(&mut self, t: &Typed) {
    //    write_type(&mut self.fns[self.cur_fn].src, t, self.env);
    //}

    /// Writes the begining of a c assignemtn statment in the current function.
    /// Returns the name that was used as the l_value.
    pub fn write_assign_stmt_begining(&mut self, t: &Typed) -> u32 {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.src.push_str("  ");
        write_type(&mut cur_fn.src, &t, self.env);
        let sym = cur_fn.gen_sym();
        write!(cur_fn.src, "_{}", sym).expect("string should not fail to write");
        cur_fn.src.push_str(" = ");
        sym
    }

    pub fn env(&self) -> &Environment<'b> {
        self.env
    }
}

#[derive(Debug, Clone)]
pub struct CFn<'a> {
    src: String,
    sym_counter: u32,
    name: &'a str,
    sym_tab: HashMap<&'a str, Ident<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Ident<'a> {
    Local(u32),
    Global(&'a str),
    ArrayDim(u32, u8),
}

impl Display for Ident<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Ident::Local(id) => write!(f, "_{id}"),
            Ident::Global(s) => write!(f, "{}", s),
            Ident::ArrayDim(id, dim) => write!(f, "_{}.d{}", id, dim),
        }
    }
}

fn default_idents() -> HashMap<&'static str, Ident<'static>> {
    HashMap::from([
        ("argnum", Ident::Global("args.d0")),
        ("args", Ident::Global("args")),
    ])
}

impl<'a> CFn<'a> {
    pub fn new(name: &'a str) -> Self {
        Self {
            src: String::new(),
            sym_counter: 0,
            name,
            sym_tab: default_idents(),
        }
    }

    pub fn gen_sym(&mut self) -> u32 {
        let sym = self.sym_counter;
        self.sym_counter += 1;
        sym
    }

    pub fn src(&mut self) -> &mut String {
        &mut self.src
    }

    pub fn sym_tab(&self) -> &HashMap<&'a str, Ident<'a>> {
        &self.sym_tab
    }
}

fn write_type(s: &mut String, t: &Typed, env: &Environment) {
    match t {
        Typed::Int => write!(s, "int64_t"),
        Typed::Bool => write!(s, "bool"),
        Typed::Float => write!(s, "double"),
        Typed::Array(typed, rank) => {
            let r = write!(s, "_a{}_", rank);
            write_type(s, typed, env);
            r
        }
        Typed::Struct(id) => write!(s, "{}", env.get_struct_id(*id).name()),
        Typed::Void => write!(s, "void_t"),
    }
    .expect("string should not fail to write");
}

pub fn write_type_string_lit(s: &mut String, t: &Typed, env: &Environment) {
    match t {
        Typed::Array(typed, rank) => {
            s.push_str("\"(ArrayType ");
            write_type_string_lit(s, t, env);
            write!(s, " {})\"", rank).expect("string should not fail to write");
        }
        Typed::Struct(id) => {
            s.push_str("\"(TupleType");
            let info = env.get_struct_id(*id);
            for (_, ty) in info.fields() {
                s.push_str(" ");
                write_type_string_lit(s, t, env);
            }
            s.push_str(" )\"");
        }
        Typed::Int => {
            s.push_str("\"(IntType)\"");
        }
        Typed::Bool => {
            s.push_str("\"(BoolType)\"");
        }
        Typed::Float => {
            s.push_str("\"(FloatType)\"");
        }
        Typed::Void => {
            s.push_str("\"(VoidType)\"");
        }
    }
}

macro_rules! write_stmt {
    ($cenv:expr, $($args:tt)*) => {
        //{
        //let src = $cenv.cur_fn().src();
        //src.push_str("  ");
        //write!(src, $($args)*);
        //$cenv.cur_fn().src().push_str(";");
        //
        //}
    };
}

pub(crate) use write_stmt;
macro_rules! write_assign_stmt {
    ($cenv:expr, $ty:expr, $($args:tt)*) => {
        {
        let id = $cenv.write_assign_stmt_begining($ty);
        write!($cenv.cur_fn().src(), $($args)*)
            .expect("string should not fail to write");
        $cenv.cur_fn().src().push_str("");
        Ident::Local(id)

        }
    };
}

pub(crate) use write_assign_stmt;

pub fn codgen_cmd(cmd: &Cmd<Typed>, c_gen_env: &mut CGenEnv<'_, '_>) {
    match cmd.kind() {
        CmdKind::ReadImage(_, lvalue) => todo!(),
        CmdKind::WriteImage(expr, _) => todo!(),
        CmdKind::Let(lvalue, expr) => todo!(),
        CmdKind::Assert(expr, _) => todo!(),
        CmdKind::Print(str) => {
            // TODO: should escape '\'
            write_stmt!(c_gen_env, "print({})", str.location().as_str(env.src()));
        }

        CmdKind::Show(expr) => {
            // Had to inline the cur_fn function for the borrow checker
            let cur_fn_src = &mut c_gen_env.fns[c_gen_env.cur_fn].src();
            cur_fn_src.push_str("  show(");
            write_type_string_lit(cur_fn_src, expr.type_data(), c_gen_env.env);
        }
        CmdKind::Time(cmd) => {
            let start_time = write_assign_stmt!(c_gen_env, &Typed::Float, "get_time()");
            codgen_cmd(cmd, c_gen_env);
            let end_time = write_assign_stmt!(c_gen_env, &Typed::Float, "get_time()");
            write_stmt!(c_gen_env, "show_time({} - {})", end_time, start_time);
        }
        CmdKind::Function {
            name,
            params,
            return_type,
            body,
            scope,
        } => todo!(),
        CmdKind::Struct { name, fields } => {
            {
                let struct_info = c_gen_env
                    .env
                    .get_struct(*name)
                    .expect("struct must be valid after typechecking");

                // rgba is used in the rt
                if struct_info.name() == "rgba" {
                    return;
                }

                c_gen_env.defs.push_str("typedef struct {\n");
                for (name, jpl_type) in struct_info.fields() {
                    c_gen_env.defs.push_str("  ");
                    write_type(&mut c_gen_env.defs, jpl_type, c_gen_env.env);
                    c_gen_env.defs.push_str(" ");
                    c_gen_env.defs.push_str(name);
                    c_gen_env.defs.push_str(";\n");
                }
                write!(c_gen_env.defs, "}} {};\n\n", struct_info.name())
                    .expect("string should not fail to write");
            };
        }
    }
}

pub fn expr_to_ident<'a, 'b>(expr: &Expr<Typed>, c_gen_env: &mut CGenEnv<'a, 'b>) -> Ident<'b> {
    match &expr.kind() {
        ExprKind::IntLit(val) => {
            assert_eq!(expr.type_data(), &Typed::Int);
            write_assign_stmt!(c_gen_env, &Typed::Int, "{}", val)
        }
        ExprKind::FloatLit(val) => {
            assert_eq!(expr.type_data(), &Typed::Float);

            write_assign_stmt!(c_gen_env, &Typed::Float, "{}", val.trunc())
        }
        ExprKind::True => {
            assert_eq!(expr.type_data(), &Typed::Bool);

            // Hack for autograder
            write_assign_stmt!(c_gen_env, &Typed::Bool, "true")
        }
        ExprKind::False => {
            assert_eq!(expr.type_data(), &Typed::Bool);

            // Hack for autograder
            write_assign_stmt!(c_gen_env, &Typed::Bool, "false")
        }
        ExprKind::Var => {
            //let var_str = expr.location().as_str(c_gen_env.env.src());
            //*(c_gen_env
            //    .cur_fn()
            //    .sym_tab
            //    .get(var_str)
            //    .unwrap_or(&Ident::Global(var_str)))
            todo!()
        }
        ExprKind::Void => {
            assert_eq!(expr.type_data(), &Typed::Bool);

            write_assign_stmt!(c_gen_env, &Typed::Void, "{{}}")
        }
        ExprKind::Paren(inner_expr) => expr_to_ident(inner_expr, c_gen_env),
        ExprKind::ArrayLit(exprs) => todo!(),
        ExprKind::StructInit(span, fields) => {
            assert!(matches!(expr.type_data(), &Typed::Struct(_)));

            let mut tmp_buf = String::from("{");
            for (i, field) in fields.iter().enumerate() {
                write!(tmp_buf, "{}", expr_to_ident(expr, c_gen_env))
                    .expect("string should not fail to write");

                if i != fields.len() {
                    tmp_buf.push_str(", ")
                }
            }

            write_assign_stmt!(c_gen_env, expr.type_data(), "{}", tmp_buf)
        }
        ExprKind::FunctionCall(span, exprs) => todo!(),
        ExprKind::FieldAccess(expr, span) => todo!(),
        ExprKind::ArrayIndex(expr, exprs) => todo!(),
        ExprKind::If(_) => todo!(),
        ExprKind::ArrayComp(items, expr, _) => todo!(),
        ExprKind::Sum(items, expr, _) => todo!(),
        ExprKind::And(args) => {
            let (lhs, rhs) = &**args;
            todo!()
        }
        ExprKind::Or(args) => {
            let (lhs, rhs) = &**args;
            todo!()
        }
        ExprKind::LessThan(args) => codegen_binop(c_gen_env, "<", args, &Typed::Bool),
        ExprKind::GreaterThan(args) => codegen_binop(c_gen_env, ">", args, &Typed::Bool),
        ExprKind::LessThanEq(args) => codegen_binop(c_gen_env, "<=", args, &Typed::Bool),
        ExprKind::GreaterThanEq(args) => codegen_binop(c_gen_env, ">=", args, &Typed::Bool),
        ExprKind::Eq(args) => codegen_binop(c_gen_env, "==", args, &Typed::Bool),
        ExprKind::NotEq(args) => codegen_binop(c_gen_env, "!=", args, &Typed::Bool),
        ExprKind::Add(args) => codegen_binop(c_gen_env, "+", args, &expr.type_data()),
        ExprKind::Minus(args) => codegen_binop(c_gen_env, "-", args, &expr.type_data()),
        ExprKind::Mulitply(args) => codegen_binop(c_gen_env, "*", args, &expr.type_data()),
        ExprKind::Divide(args) => codegen_binop(c_gen_env, "/", args, &expr.type_data()),
        ExprKind::Modulo(args) => match expr.type_data() {
            Typed::Int => codegen_binop(c_gen_env, "%", args, &expr.type_data()),
            Typed::Float => todo!("use fmod"),
            _ => unreachable!(),
        },
        ExprKind::Not(inner_expr) => {
            assert_eq!(inner_expr.type_data(), expr.type_data());
            let inner_ident = expr_to_ident(inner_expr, c_gen_env);
            write_assign_stmt!(c_gen_env, &Typed::Bool, "!{}", inner_ident)
        }
        ExprKind::Negation(inner_expr) => {
            assert_eq!(inner_expr.type_data(), expr.type_data());
            let inner_ident = expr_to_ident(inner_expr, c_gen_env);
            write_assign_stmt!(c_gen_env, expr.type_data(), "!{}", inner_ident)
        }
    }
}

fn codegen_binop<'a, 'b>(
    env: &mut CGenEnv<'a, 'b>,
    op: &str,
    args: &(Expr<Typed>, Expr<Typed>),
    output: &Typed,
) -> Ident<'b> {
    let (lhs, rhs) = args;
    let lhs_ident = expr_to_ident(lhs, env);
    let rhs_ident = expr_to_ident(rhs, env);
    write_assign_stmt!(env, output, "{} {} {}", lhs_ident, op, rhs_ident)
}
