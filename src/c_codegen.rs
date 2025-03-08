use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter, Write};

use crate::ast::auxiliary::{LValue, Str};
use crate::ast::cmd::{Cmd, CmdKind};
use crate::ast::expr::{Expr, ExprKind};
use crate::ast::stmt::{Stmt, StmtType};
use crate::environment::builtins::RGBA_STRUCT_ID;
use crate::{ast::Program, environment::Environment, typecheck::Typed, utils::Span};

const INDENTATION: &str = "    ";
const INCLUDES: &str = "\
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include \"rt/runtime.h\"
";

const CUSTOM_TYPES: &str = "\
typedef struct { } void_t;
";

const MAIN_FN_IDX: usize = 0;
const MAIN_FN_DEFINITION: &str = concat!("void jpl_main(struct args args) {\n");

pub fn generate_c(ast: Program<Typed>, env: &Environment) {
    let mut cenv = CGenEnv::new(env);
    for cmd in ast.commands() {
        codegen_cmd(cmd, &mut cenv);
    }

    cenv.finish_main();

    print!("{}", cenv.defs);
    for function in cenv.fns {
        println!("{}", function.src);
    }
}

#[derive(Debug, Clone)]
pub struct CGenEnv<'a, 'b> {
    /// Stores includes, and all struct definitions
    defs: String,
    jmp_ctr: u32,
    fns: Vec<CFn<'b>>,
    env: &'a Environment<'b>,
    cur_fn: usize,

    array_types: HashSet<(Typed, u8)>,
}

impl<'a, 'b> CGenEnv<'a, 'b> {
    pub fn new(env: &'a Environment<'b>) -> Self {
        let mut fns = Vec::with_capacity(1 + env.functions().len());
        fns.push(CFn::new_main());

        Self {
            jmp_ctr: 0,
            fns,
            env,
            cur_fn: MAIN_FN_IDX,
            defs: format!("{}\n{}\n", INCLUDES, CUSTOM_TYPES),
            array_types: HashSet::from([(Typed::Struct(RGBA_STRUCT_ID), 2)]),
        }
    }

    pub fn cur_fn(&mut self) -> &mut CFn<'b> {
        &mut self.fns[self.cur_fn]
    }

    /// Sets the current function back to `jpl_main`
    pub fn reset_fn(&mut self) {
        self.cur_fn = MAIN_FN_IDX;
    }

    pub fn gen_jump_sym(&mut self) -> u32 {
        self.jmp_ctr += 1;
        self.jmp_ctr
    }

    /// Adds the fn to the C Codegen Envirment and sets it as the current function
    /// TODO: don't really need name in here
    pub fn add_fn(&mut self, name: &Span) {
        let name = name.as_str(self.env.src());

        assert!(self.fns.len() < self.fns.capacity());
        assert!(self.env.functions().contains_key(name));

        self.cur_fn = self.fns.len();
        self.fns.push(CFn::new_main());
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
    pub fn write_assign_stmt_begining(&mut self, t: &Typed) -> Ident<'static> {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.src.push_str(INDENTATION);
        write_type(&mut cur_fn.src, &t, self.env);
        let sym = cur_fn.gen_sym();
        write!(cur_fn.src, " {} = ", sym).expect("string should not fail to write");
        sym
    }

    pub fn write_label(&mut self, label: u32) {
        write!(self.cur_fn().src(), "{}_jump{}:;", INDENTATION, label)
            .expect("string should not fail to write");
    }

    pub fn write_var_declaration(&mut self, ty: &Typed) -> Ident<'static> {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.src().push_str(INDENTATION);
        write_type(cur_fn.src(), ty, self.env);
        cur_fn.src().push_str(" ");
        let sym = cur_fn.gen_sym();
        write!(cur_fn.src(), "{}", sym).expect("string should not fail to write");
        cur_fn.src().push_str(";\n");
        sym
    }

    pub fn write_goto(&mut self, label: u32) {
        write!(self.cur_fn().src(), "{}goto _jump{};\n", INDENTATION, label)
            .expect("string should not fail to write");
    }

    pub fn finish_main(&mut self) {
        self.fns[MAIN_FN_IDX].src().push_str("}\n");
    }
}

#[derive(Debug, Clone)]
pub struct CFn<'a> {
    src: String,
    sym_counter: u32,
    sym_tab: HashMap<&'a str, Ident<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Ident<'a> {
    Local(u32),
    Global(&'a str),
    ArrayDimLocal(u32, u8),
    ArrayDimGlobal(&'a str, u8),
}

impl Display for Ident<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Ident::Local(id) => write!(f, "_{}", id),
            Ident::Global(s) => write!(f, "{}", s),
            Ident::ArrayDimLocal(id, dim) => write!(f, "_{}.d{}", id, dim),
            Ident::ArrayDimGlobal(s, dim) => write!(f, "{}.d{}", s, dim),
        }
    }
}

impl<'a> CFn<'a> {
    //TOOD: add main method function header
    pub fn new_main() -> Self {
        Self {
            src: String::from(MAIN_FN_DEFINITION),
            sym_counter: 0,
            sym_tab: HashMap::from([
                ("argnum", Ident::Global("args.d0")),
                ("args", Ident::Global("args")),
            ]),
        }
    }
    pub fn new() -> Self {
        Self {
            src: String::new(),
            sym_counter: 0,
            sym_tab: HashMap::new(),
        }
    }

    pub fn gen_sym(&mut self) -> Ident<'static> {
        let sym = self.sym_counter;
        self.sym_counter += 1;
        Ident::Local(sym)
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
        {
            let src = $cenv.fns[$cenv.cur_fn].src();
            src.push_str(INDENTATION);
            write!(src, $($args)*).expect("string should not fail to write");
            src.push_str(";\n");
        }
    };
}

pub(self) use write_stmt;
macro_rules! write_assign_stmt {
    ($cenv:expr, $ty:expr, $($args:tt)*) => {
        {
            let id = $cenv.write_assign_stmt_begining($ty);
            let src = $cenv.fns[$cenv.cur_fn].src();
            write!(src, $($args)*)
                .expect("string should not fail to write");
            src.push_str(";\n");
            id
        }
    };
}

pub(self) use write_assign_stmt;

macro_rules! write_if_stmt {
    ($cenv:expr, $($args:tt)*) => {
        {
            let src = $cenv.fns[$cenv.cur_fn].src();
            src.push_str(INDENTATION);
            src.push_str("if (");
            write!(src, $($args)*)
                .expect("string should not fail to write");
            src.push_str(")\n");
            let label = $cenv.gen_jump_sym();
            $cenv.write_goto(label);
            label
        }
    };
}

pub(self) use write_if_stmt;

fn codegen_cmd(cmd: &Cmd<Typed>, cenv: &mut CGenEnv<'_, '_>) {
    match cmd.kind() {
        CmdKind::ReadImage(s, lvalue) => {
            let s = s.location().as_str(cenv.env.src());
            let ident =
                write_assign_stmt!(cenv, &Typed::Struct(RGBA_STRUCT_ID), "read_image({})", s);
            codegen_lvalue(cenv, lvalue, ident);
        }
        CmdKind::WriteImage(expr, s) => {
            //write_image(_0, "hello.png");
            let arr_ident = expr_to_ident(expr, cenv);
            let s = s.location().as_str(cenv.env.src());
            write_stmt!(cenv, "write_image({}, {})", arr_ident, s);
        }
        CmdKind::Let(lvalue, expr) => {
            let ident = expr_to_ident(expr, cenv);
            codegen_lvalue(cenv, lvalue, ident);
        }
        CmdKind::Assert(expr, msg) => {
            codegen_assert(cenv, expr, msg);
        }
        CmdKind::Print(str) => {
            // TODO: should escape '\'
            write_stmt!(cenv, "print({})", str.location().as_str(cenv.env.src()));
        }

        CmdKind::Show(expr) => {
            let expr_ident = expr_to_ident(expr, cenv);
            let cur_fn_src = &mut cenv.fns[cenv.cur_fn].src();
            cur_fn_src.push_str(INDENTATION);
            cur_fn_src.push_str("show(");
            write_type_string_lit(cur_fn_src, expr.type_data(), cenv.env);

            write!(cur_fn_src, ", &{});\n", expr_ident).expect("string should not fail to write");
        }

        CmdKind::Time(cmd) => {
            let start_time = write_assign_stmt!(cenv, &Typed::Float, "get_time()");
            codegen_cmd(cmd, cenv);
            let end_time = write_assign_stmt!(cenv, &Typed::Float, "get_time()");
            write_stmt!(cenv, "print_time({} - {})", end_time, start_time);
        }

        CmdKind::Function {
            name, params, body, ..
        } => {
            cenv.add_fn(name);
            let fn_info = &cenv.env.functions()[name.as_str(cenv.env.src())];
            let src = &mut cenv.fns[cenv.cur_fn].src;
            write_type(src, fn_info.ret(), cenv.env);
            src.push_str(" ");
            src.push_str(name.as_str(cenv.env.src()));
            src.push_str("(");
            for (t, name) in fn_info
                .args()
                .iter()
                .zip(params.iter().map(|b| b.location().as_str(cenv.env.src())))
            {
                write_type(src, t, cenv.env);
                src.push(' ');
                src.push_str(name);
                src.push(',');
            }

            // Remove trailing ,
            src.pop();

            src.push_str(") {\n");
            //stmts
            for stmt in body {
                codegen_stmt(stmt, cenv);
            }

            cenv.cur_fn().src().push_str("}\n");
            cenv.reset_fn();
        }
        CmdKind::Struct { name, fields: _ } => {
            {
                let struct_info = cenv
                    .env
                    .get_struct(*name)
                    .expect("struct must be valid after typechecking");

                // rgba is used in the rt
                if struct_info.name() == "rgba" {
                    return;
                }

                cenv.defs.push_str("typedef struct {\n");
                for (name, jpl_type) in struct_info.fields() {
                    cenv.defs.push_str("  ");
                    write_type(&mut cenv.defs, jpl_type, cenv.env);
                    cenv.defs.push_str(" ");
                    cenv.defs.push_str(name);
                    cenv.defs.push_str(";\n");
                }
                write!(cenv.defs, "}} {};\n\n", struct_info.name())
                    .expect("string should not fail to write");
            };
        }
    }
}

fn codegen_stmt(stmt: &Stmt<Typed>, cenv: &mut CGenEnv<'_, '_>) {
    match stmt.kind() {
        StmtType::Let(lvalue, expr) => {
            let ident = expr_to_ident(expr, cenv);
            codegen_lvalue(cenv, lvalue, ident);
        }
        StmtType::Assert(expr, msg) => {
            codegen_assert(cenv, expr, msg);
        }
        StmtType::Return(expr) => {
            let ret_val = expr_to_ident(expr, cenv);
            write_stmt!(cenv, "{}", ret_val);
        }
    }
}

fn codegen_assert(cenv: &mut CGenEnv<'_, '_>, expr: &Expr<Typed>, msg: &Str) {
    let cond = expr_to_ident(expr, cenv);

    let label = write_if_stmt!(cenv, "0 != {}", cond);
    write_stmt!(
        cenv,
        "fail_assertion({})",
        msg.location().as_str(cenv.env.src())
    );
    cenv.write_label(label);
}

fn codegen_lvalue<'a, 'b>(cenv: &mut CGenEnv<'a, 'b>, lvalue: &LValue, rvalue: Ident<'b>) {
    let lvalue_name = lvalue.location().as_str(cenv.env.src());
    cenv.cur_fn().sym_tab.insert(lvalue_name, rvalue);
    for (dim, dim_span) in lvalue.array_bindings().into_iter().flatten().enumerate() {
        let dim_str = dim_span.as_str(cenv.env.src());
        match rvalue {
            Ident::Local(id) => {
                let dim_ident = Ident::ArrayDimLocal(id, dim as u8);
                cenv.cur_fn().sym_tab.insert(dim_str, dim_ident);
            }
            Ident::Global(s) => {
                let dim_ident = Ident::ArrayDimGlobal(s, dim as u8);
                cenv.cur_fn().sym_tab.insert(dim_str, dim_ident);
            }
            Ident::ArrayDimLocal(_, _) | Ident::ArrayDimGlobal(_, _) => {
                unreachable!("Expresion can not generate to array dim ident")
            }
        }
    }
}

pub fn expr_to_ident<'a, 'b>(expr: &Expr<Typed>, cenv: &mut CGenEnv<'a, 'b>) -> Ident<'b> {
    match &expr.kind() {
        ExprKind::IntLit(val) => {
            assert_eq!(expr.type_data(), &Typed::Int);
            write_assign_stmt!(cenv, &Typed::Int, "{}", val)
        }
        ExprKind::FloatLit(val) => {
            assert_eq!(expr.type_data(), &Typed::Float);

            write_assign_stmt!(cenv, &Typed::Float, "{}", val.trunc())
        }
        ExprKind::True => {
            assert_eq!(expr.type_data(), &Typed::Bool);

            // Hack for autograder
            write_assign_stmt!(cenv, &Typed::Bool, "true")
        }
        ExprKind::False => {
            assert_eq!(expr.type_data(), &Typed::Bool);

            // Hack for autograder
            write_assign_stmt!(cenv, &Typed::Bool, "false")
        }
        ExprKind::Var => {
            let var_str = expr.location().as_str(cenv.env.src());

            // If the variable is global then just return its name as the ident
            *(cenv
                .cur_fn()
                .sym_tab
                .get(var_str)
                .unwrap_or(&Ident::Global(var_str)))
        }
        ExprKind::Void => {
            assert_eq!(expr.type_data(), &Typed::Void);

            write_assign_stmt!(cenv, &Typed::Void, "{{}}")
        }
        ExprKind::Paren(inner_expr) => expr_to_ident(inner_expr, cenv),
        ExprKind::ArrayLit(exprs) => {
            let arr_type = expr.type_data();
            assert!(matches!(arr_type, Typed::Array(_, 1)));
            cenv.gen_arr_struct(arr_type.clone());

            let idents: Vec<_> = exprs
                .into_iter()
                .map(|e| expr_to_ident(expr, cenv))
                .collect();

            let arr_ident = cenv.write_var_declaration(arr_type);
            // Not ideal

            write_stmt!(cenv, "{}.d0 = {}", arr_ident, exprs.len());

            let mut type_str = String::new();
            write_type(&mut type_str, exprs[0].type_data(), cenv.env);
            write_stmt!(
                cenv,
                "{}.data = jpl_alloc(sizeof({}) * {})",
                arr_ident,
                type_str,
                exprs.len()
            );

            for (i, ident) in idents.into_iter().enumerate() {
                write_stmt!(cenv, "{}.data[{}] = {}", arr_ident, i, ident);
            }

            arr_ident
        }
        ExprKind::StructInit(_, fields) => {
            assert!(matches!(expr.type_data(), &Typed::Struct(_)));

            let mut tmp_buf = String::from("{");
            for (i, field) in fields.iter().enumerate() {
                write!(tmp_buf, "{}", expr_to_ident(field, cenv))
                    .expect("string should not fail to write");

                if i != fields.len() {
                    tmp_buf.push_str(", ")
                }
            }

            write_assign_stmt!(cenv, expr.type_data(), "{}", tmp_buf)
        }
        ExprKind::FunctionCall(name, exprs) => {
            let fn_info = cenv
                .env
                .get_function(*name)
                .expect("function should be valid after typechecking");

            let mut fn_args = String::new();
            for expr in exprs {
                let ident = expr_to_ident(expr, cenv);
                write!(&mut fn_args, "{}, ", ident).expect("string should not fail to write");
            }
            // Remove trailing comma and space
            fn_args.pop();
            fn_args.pop();

            write_assign_stmt!(cenv, fn_info.ret(), "{}({})", fn_info.name(), fn_args)
        }
        ExprKind::FieldAccess(struct_expr, span) => {
            let struct_ident = expr_to_ident(struct_expr, cenv);
            write_assign_stmt!(
                cenv,
                expr.type_data(),
                "{}.{}",
                struct_ident,
                span.as_str(cenv.env.src())
            )
        }
        ExprKind::ArrayIndex(arr_expr, indices) => {
            //int64_t _8 = 2;
            //int64_t _9 = 1;
            //if (_8 >= 0)
            //goto _jump4;
            //fail_assertion("negative array index");
            //_jump4:;
            //if (_8 < _0.d0)
            //goto _jump5;
            //fail_assertion("index too large");
            //_jump5:;
            let arr_ident = expr_to_ident(arr_expr, cenv);
            let indices_idents: Vec<_> = indices
                .into_iter()
                .map(|i| expr_to_ident(i, cenv))
                .collect();

            // Check the bounds of the indices
            for (dim, index_ident) in indices_idents.iter().enumerate() {
                let safe_label = write_if_stmt!(cenv, "{} >= 0", index_ident);
                write_stmt!(cenv, "fail_assertion(\"negative array index\");");
                cenv.write_label(safe_label);

                let safe_label = write_if_stmt!(cenv, "{} < {}.d{}", index_ident, arr_ident, dim);
                write_stmt!(cenv, "fail_assertion(\"index too large\");");
                cenv.write_label(safe_label);
            }
            //int64_t _10 = 0;
            //_10 *= _0.d0;
            //_10 += _8;
            //_10 *= _0.d1;
            //_10 += _9;
            //bool _11 = _0.data[_10];
            let combined_index = write_assign_stmt!(cenv, &Typed::Int, "0");
            for (i, index_ident) in indices_idents.iter().enumerate() {
                write_stmt!(cenv, "{} *= {}.d{}", combined_index, arr_ident, i);
                write_stmt!(cenv, "{} += {}", combined_index, index_ident);
            }

            write_assign_stmt!(
                cenv,
                expr.type_data(),
                "{}.data[{}]",
                arr_ident,
                combined_index
            )
        }
        ExprKind::If(if_expr) => {
            let (cond, true_branch, false_branch) = if_expr.as_ref();
            let output_type = true_branch.type_data();

            let output_ident = expr_to_ident(cond, cenv);
            let output_ident = cenv.write_var_declaration(output_type);
            let false_label = write_if_stmt!(cenv, "!{}", output_ident);
            let true_ident = expr_to_ident(true_branch, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, true_ident);
            let end_label = cenv.gen_jump_sym();
            cenv.write_goto(end_label);
            cenv.write_label(false_label);
            let false_ident = expr_to_ident(true_branch, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, false_ident);
            cenv.write_label(end_label);
            output_ident
        }
        ExprKind::ArrayComp(loop_bounds, body, _) => {
            let arr_type = expr.type_data();
            assert!(matches!(arr_type, Typed::Array(_, _)));
            // TODO: clone kinda sucks here
            cenv.gen_arr_struct(arr_type.clone());
            let output_arr_ident = cenv.write_var_declaration(arr_type);

            // int64_t _1 = 1;
            //    _0.d0 = _1;
            //    if (_1 > 0)
            //    goto _jump1;
            //    fail_assertion("non-positive loop bound");
            //    _jump1:;
            let index_idents: Vec<_> = loop_bounds
                .into_iter()
                .enumerate()
                .map(|(i, (_, expr))| {
                    let index_ident = expr_to_ident(expr, cenv);
                    write_stmt!(cenv, "{}.d{} = {}", output_arr_ident, i, index_ident);
                    let ok_label = write_if_stmt!(cenv, "{} > 0", index_ident);
                    write_stmt!(cenv, "fail_assertion(\"non-positive loop bound\");");

                    index_ident
                })
                .collect();

            todo!()
        }
        ExprKind::Sum(items, expr, _) => todo!(),
        ExprKind::And(args) => {
            let (lhs, rhs) = &**args;
            let lhs_ident = expr_to_ident(lhs, cenv);
            let output_ident = write_assign_stmt!(cenv, &Typed::Bool, "{}", lhs_ident);
            let done_label = write_if_stmt!(cenv, "0 == {}", output_ident);
            let rhs_ident = expr_to_ident(rhs, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, rhs_ident);
            cenv.write_label(done_label);
            output_ident
        }
        ExprKind::Or(args) => {
            let (lhs, rhs) = &**args;
            let lhs_ident = expr_to_ident(lhs, cenv);
            let output_ident = write_assign_stmt!(cenv, &Typed::Bool, "{}", lhs_ident);
            let done_label = write_if_stmt!(cenv, "0 != {}", output_ident);
            let rhs_ident = expr_to_ident(rhs, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, rhs_ident);
            cenv.write_label(done_label);
            output_ident
        }
        ExprKind::LessThan(args) => codegen_binop(cenv, "<", args, &Typed::Bool),
        ExprKind::GreaterThan(args) => codegen_binop(cenv, ">", args, &Typed::Bool),
        ExprKind::LessThanEq(args) => codegen_binop(cenv, "<=", args, &Typed::Bool),
        ExprKind::GreaterThanEq(args) => codegen_binop(cenv, ">=", args, &Typed::Bool),
        ExprKind::Eq(args) => codegen_binop(cenv, "==", args, &Typed::Bool),
        ExprKind::NotEq(args) => codegen_binop(cenv, "!=", args, &Typed::Bool),
        ExprKind::Add(args) => codegen_binop(cenv, "+", args, &expr.type_data()),
        ExprKind::Minus(args) => codegen_binop(cenv, "-", args, &expr.type_data()),
        ExprKind::Mulitply(args) => codegen_binop(cenv, "*", args, &expr.type_data()),
        ExprKind::Divide(args) => codegen_binop(cenv, "/", args, &expr.type_data()),
        ExprKind::Modulo(args) => match expr.type_data() {
            Typed::Int => codegen_binop(cenv, "%", args, &expr.type_data()),
            Typed::Float => {
                let (lhs, rhs) = &**args;
                let lhs_ident = expr_to_ident(lhs, cenv);
                let rhs_ident = expr_to_ident(rhs, cenv);
                write_assign_stmt!(cenv, &Typed::Float, "fmod({}, {})", lhs_ident, rhs_ident)
            }
            _ => unreachable!(),
        },
        ExprKind::Not(inner_expr) => {
            assert_eq!(inner_expr.type_data(), expr.type_data());
            let inner_ident = expr_to_ident(inner_expr, cenv);
            write_assign_stmt!(cenv, &Typed::Bool, "!{}", inner_ident)
        }
        ExprKind::Negation(inner_expr) => {
            assert_eq!(inner_expr.type_data(), expr.type_data());
            let inner_ident = expr_to_ident(inner_expr, cenv);
            write_assign_stmt!(cenv, expr.type_data(), "!{}", inner_ident)
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
