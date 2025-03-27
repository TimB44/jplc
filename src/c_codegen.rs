use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter, Write};

use crate::ast::auxiliary::{LValue, LoopVar, Str};
use crate::ast::cmd::{Cmd, CmdKind};
use crate::ast::expr::{Expr, ExprKind};
use crate::ast::stmt::{Stmt, StmtType};
use crate::environment::builtins::{IMAGE_TYPE, RGBA_STRUCT_ID};
use crate::{ast::Program, environment::Environment, typecheck::TypeVal, utils::Span};

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

pub fn generate_c(ast: Program, env: &Environment) {
    let mut cenv = CGenEnv::new(env);
    for cmd in ast.commands() {
        codegen_cmd(cmd, &mut cenv);
    }

    cenv.finish_main();

    print!("{}", cenv.defs);
    for function in &cenv.fns[(MAIN_FN_IDX + 1)..] {
        println!("{}", function.src);
    }
    println!("{}", cenv.fns[MAIN_FN_IDX].src());
}

#[derive(Debug, Clone)]
struct CGenEnv<'a, 'b> {
    /// Stores includes, and all struct definitions
    defs: String,
    jmp_ctr: u32,
    fns: Vec<CFn<'b>>,
    env: &'a Environment<'b>,
    cur_fn: usize,

    array_types: HashSet<(TypeVal, u8)>,
}

impl<'a, 'b> CGenEnv<'a, 'b> {
    fn new(env: &'a Environment<'b>) -> Self {
        let mut fns = Vec::with_capacity(1 + env.functions().len());
        fns.push(CFn::new_main());

        Self {
            jmp_ctr: 0,
            fns,
            env,
            cur_fn: MAIN_FN_IDX,
            defs: format!("{}\n{}\n", INCLUDES, CUSTOM_TYPES),
            array_types: HashSet::from([(TypeVal::Struct(RGBA_STRUCT_ID), 2)]),
        }
    }

    fn cur_fn(&mut self) -> &mut CFn<'b> {
        &mut self.fns[self.cur_fn]
    }

    /// Sets the current function back to `jpl_main`
    fn reset_fn(&mut self) {
        self.cur_fn = MAIN_FN_IDX;
    }

    fn gen_jump_sym(&mut self) -> u32 {
        self.jmp_ctr += 1;
        self.jmp_ctr
    }

    /// Adds the fn to the C Codegen Envirment and sets it as the current function.
    /// TODO: don't really need name in here
    fn add_fn(&mut self, name: Span) {
        let name = name.as_str(self.env.src());

        assert!(self.fns.len() < self.fns.capacity());
        assert!(self.env.functions().contains_key(name));

        self.cur_fn = self.fns.len();
        self.fns.push(CFn::new());
    }

    /// Creates a struct for the given array type if it has not yet been created. Does nothing if
    /// the given type is not an array or has already been created
    fn gen_arr_struct(&mut self, t: &TypeVal) {
        let (inner, rank) = match t {
            TypeVal::Array(inner, rank) => (inner.clone(), *rank),
            _ => return,
        };

        let entry = (*inner, rank);
        if self.array_types.contains(&entry) {
            return;
        }

        let (inner, rank) = entry;
        self.gen_arr_struct(&inner);

        self.defs.push_str("typedef struct {\n");
        for dim in 0..rank {
            writeln!(self.defs, "  int64_t d{};", dim).expect("string should not fail to write");
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
    // fn write_type(&mut self, t: &Typed) {
    //    write_type(&mut self.fns[self.cur_fn].src, t, self.env);
    //}

    /// Writes the begining of a c assignemtn statment in the current function.
    /// Returns the name that was used as the l_value.
    fn write_assign_stmt_begining(&mut self, t: &TypeVal) -> Ident<'static> {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.src.push_str(INDENTATION);
        write_type(&mut cur_fn.src, t, self.env);
        let sym = cur_fn.gen_sym();
        write!(cur_fn.src, " {} = ", sym).expect("string should not fail to write");
        sym
    }

    fn write_label(&mut self, label: u32) {
        writeln!(self.cur_fn().src(), "{}_jump{}:;", INDENTATION, label)
            .expect("string should not fail to write");
    }

    fn write_var_declaration(&mut self, ty: &TypeVal) -> Ident<'static> {
        let cur_fn = &mut self.fns[self.cur_fn];
        cur_fn.src().push_str(INDENTATION);
        write_type(cur_fn.src(), ty, self.env);
        cur_fn.src().push(' ');
        let sym = cur_fn.gen_sym();
        write!(cur_fn.src(), "{}", sym).expect("string should not fail to write");
        cur_fn.src().push_str(";\n");
        sym
    }

    fn write_goto(&mut self, label: u32) {
        writeln!(self.cur_fn().src(), "{}goto _jump{};", INDENTATION, label)
            .expect("string should not fail to write");
    }

    fn finish_main(&mut self) {
        self.fns[MAIN_FN_IDX].src().push_str("}\n");
    }
}

#[derive(Debug, Clone)]
struct CFn<'a> {
    src: String,
    sym_counter: u32,
    sym_tab: HashMap<&'a str, Ident<'a>>,
}

#[derive(Debug, Clone, Copy)]
enum Ident<'a> {
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

impl CFn<'_> {
    //TOOD: add main method function header
    fn new_main() -> Self {
        Self {
            src: String::from(MAIN_FN_DEFINITION),
            sym_counter: 0,
            sym_tab: HashMap::from([
                ("argnum", Ident::Global("args.d0")),
                ("args", Ident::Global("args")),
            ]),
        }
    }
    fn new() -> Self {
        Self {
            src: String::new(),
            sym_counter: 0,
            sym_tab: HashMap::new(),
        }
    }

    fn gen_sym(&mut self) -> Ident<'static> {
        let sym = self.sym_counter;
        self.sym_counter += 1;
        Ident::Local(sym)
    }

    fn src(&mut self) -> &mut String {
        &mut self.src
    }
}

fn write_type(s: &mut String, t: &TypeVal, env: &Environment) {
    match t {
        TypeVal::Int => write!(s, "int64_t"),
        TypeVal::Bool => write!(s, "bool"),
        TypeVal::Float => write!(s, "double"),
        TypeVal::Array(typed, rank) => {
            let r = write!(s, "_a{}_", rank);
            write_type(s, typed, env);
            r
        }
        TypeVal::Struct(id) => write!(s, "{}", env.get_struct_id(*id).name()),
        TypeVal::Void => write!(s, "void_t"),
    }
    .expect("string should not fail to write");
}

fn write_type_string_lit(s: &mut String, t: &TypeVal, env: &Environment) {
    s.push('"');
    t.write_type_string(s, env);
    s.push('"');
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

use write_stmt;
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

use write_assign_stmt;

macro_rules! write_if_stmt {
    ($cenv:expr, $format:literal, $($args:tt)*) => {
        {
            let label = $cenv.gen_jump_sym();
            write_if_stmt!($cenv, label, $format, $($args)*)
        }
    };

    ($cenv:expr, $label:expr, $($args:tt)*) => {
        {
            let src = $cenv.fns[$cenv.cur_fn].src();
            src.push_str(INDENTATION);
            src.push_str("if (");
            write!(src, $($args)*)
                .expect("string should not fail to write");
            src.push_str(")\n");
            $cenv.write_goto($label);
            $label
        }
    };
}

use write_if_stmt;

fn codegen_cmd(cmd: &Cmd, cenv: &mut CGenEnv<'_, '_>) {
    match cmd.kind() {
        CmdKind::ReadImage(s, lvalue) => {
            let s = s.loc().as_str(cenv.env.src());
            let ident = write_assign_stmt!(cenv, &IMAGE_TYPE, "read_image({})", s);
            codegen_lvalue(cenv, lvalue, ident);
            //FIXME: we really should not nake these statements but the autograder wants them for
            //some reason
            for (i, binding) in lvalue.array_bindings().into_iter().flatten().enumerate() {
                write_stmt!(
                    cenv,
                    "int64_t {} = {}.d{}",
                    binding.loc().as_str(cenv.env.src()),
                    ident,
                    i
                );
            }
        }
        CmdKind::WriteImage(expr, s) => {
            //write_image(_0, "hello.png");
            let arr_ident = expr_to_ident(expr, cenv);
            let s = s.loc().as_str(cenv.env.src());
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
            write_stmt!(cenv, "print({})", str.loc().as_str(cenv.env.src()));
        }

        CmdKind::Show(expr) => {
            let expr_ident = expr_to_ident(expr, cenv);
            let cur_fn_src = &mut cenv.fns[cenv.cur_fn].src();
            cur_fn_src.push_str(INDENTATION);
            cur_fn_src.push_str("show(");
            write_type_string_lit(cur_fn_src, expr.type_data(), cenv.env);

            writeln!(cur_fn_src, ", &{});", expr_ident).expect("string should not fail to write");
        }

        CmdKind::Time(cmd) => {
            let start_time = write_assign_stmt!(cenv, &TypeVal::Float, "get_time()");
            codegen_cmd(cmd, cenv);
            let end_time = write_assign_stmt!(cenv, &TypeVal::Float, "get_time()");
            write_stmt!(cenv, "print_time({} - {})", end_time, start_time);
        }

        CmdKind::Function {
            name, params, body, ..
        } => {
            cenv.add_fn(name.loc());
            let fn_info = &cenv.env.functions()[name.loc().as_str(cenv.env.src())];
            cenv.gen_arr_struct(fn_info.ret());
            for arg_type in fn_info.args() {
                cenv.gen_arr_struct(arg_type);
            }

            let src = &mut cenv.fns[cenv.cur_fn].src;
            write_type(src, fn_info.ret(), cenv.env);
            src.push(' ');
            src.push_str(name.loc().as_str(cenv.env.src()));
            src.push('(');
            for (arg_type, binding) in fn_info.args().iter().zip(params) {
                let name = binding.lvalue().variable().loc().as_str(cenv.env.src());
                codegen_lvalue(cenv, binding.lvalue(), Ident::Global(name));
                let src = &mut cenv.fns[cenv.cur_fn].src;
                write_type(src, arg_type, cenv.env);
                src.push(' ');
                src.push_str(name);
                src.push_str(", ");
            }

            let src = &mut cenv.fns[cenv.cur_fn].src;
            // Remove trailing comma and space
            if !fn_info.args().is_empty() {
                src.pop();
                src.pop();
            }

            src.push_str(") {\n");
            //stmts

            // If we have not seen a return statment then add one
            if !body.into_iter().fold(false, |seen_ret, stmt| {
                codegen_stmt(stmt, cenv);
                seen_ret | matches!(stmt.kind(), StmtType::Return(_))
            }) {
                assert!(matches!(fn_info.ret(), TypeVal::Void));
                let void_ident = write_assign_stmt!(cenv, &TypeVal::Void, "{{}}");
                write_stmt!(cenv, "return {}", void_ident);
            }

            cenv.cur_fn().src().push_str("}\n");
            cenv.reset_fn();
        }
        CmdKind::Struct { name, fields: _ } => {
            {
                let struct_info = cenv
                    .env
                    .get_struct(name.loc())
                    .expect("struct must be valid after typechecking");

                // rgba is defined in the runtime
                if struct_info.name() == "rgba" {
                    return;
                }
                for (_, jpl_type) in struct_info.fields() {
                    cenv.gen_arr_struct(jpl_type);
                }

                cenv.defs.push_str("typedef struct {\n");
                for (name, jpl_type) in struct_info.fields() {
                    cenv.defs.push_str(INDENTATION);
                    write_type(&mut cenv.defs, jpl_type, cenv.env);
                    cenv.defs.push(' ');
                    cenv.defs.push_str(name);
                    cenv.defs.push_str(";\n");
                }
                write!(cenv.defs, "}} {};\n\n", struct_info.name())
                    .expect("string should not fail to write");
            };
        }
    }
}

fn codegen_stmt(stmt: &Stmt, cenv: &mut CGenEnv<'_, '_>) {
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
            write_stmt!(cenv, "return {}", ret_val);
        }
    }
}

fn codegen_assert(cenv: &mut CGenEnv<'_, '_>, expr: &Expr, msg: &Str) {
    let cond = expr_to_ident(expr, cenv);

    let label = write_if_stmt!(cenv, "0 != {}", cond);
    write_stmt!(cenv, "fail_assertion({})", msg.loc().as_str(cenv.env.src()));
    cenv.write_label(label);
}

fn codegen_lvalue<'b>(cenv: &mut CGenEnv<'_, 'b>, lvalue: &LValue, rvalue: Ident<'b>) {
    let lvalue_name = lvalue.variable().loc().as_str(cenv.env.src());
    cenv.cur_fn().sym_tab.insert(lvalue_name, rvalue);
    for (dim, dim_span) in lvalue.array_bindings().into_iter().flatten().enumerate() {
        let dim_str = dim_span.loc().as_str(cenv.env.src());
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

fn expr_to_ident<'b>(expr: &Expr, cenv: &mut CGenEnv<'_, 'b>) -> Ident<'b> {
    match &expr.kind() {
        ExprKind::IntLit(val) => {
            assert_eq!(expr.type_data(), &TypeVal::Int);
            write_assign_stmt!(cenv, &TypeVal::Int, "{}", val)
        }
        ExprKind::FloatLit(val) => {
            assert_eq!(expr.type_data(), &TypeVal::Float);

            write_assign_stmt!(cenv, &TypeVal::Float, "{}.0", val.trunc())
        }
        ExprKind::True => {
            assert_eq!(expr.type_data(), &TypeVal::Bool);

            write_assign_stmt!(cenv, &TypeVal::Bool, "true")
        }
        ExprKind::False => {
            assert_eq!(expr.type_data(), &TypeVal::Bool);

            // Hack for autograder
            write_assign_stmt!(cenv, &TypeVal::Bool, "false")
        }
        ExprKind::Var => {
            let var_str = expr.loc().as_str(cenv.env.src());

            // If the variable is global then just return its name as the ident
            *(cenv
                .cur_fn()
                .sym_tab
                .get(var_str)
                .unwrap_or(&Ident::Global(var_str)))
        }
        ExprKind::Void => {
            assert_eq!(expr.type_data(), &TypeVal::Void);

            write_assign_stmt!(cenv, &TypeVal::Void, "{{}}")
        }
        ExprKind::Paren(inner_expr) => expr_to_ident(inner_expr, cenv),
        ExprKind::ArrayLit(exprs) => {
            let arr_type = expr.type_data();
            assert!(matches!(arr_type, TypeVal::Array(_, 1)));

            let idents: Vec<_> = exprs.into_iter().map(|e| expr_to_ident(e, cenv)).collect();

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

            cenv.gen_arr_struct(arr_type);
            arr_ident
        }
        ExprKind::StructInit(_, fields) => {
            assert!(matches!(expr.type_data(), &TypeVal::Struct(_)));

            let mut tmp_buf = String::new();
            for (i, field) in fields.iter().enumerate() {
                write!(tmp_buf, "{}", expr_to_ident(field, cenv))
                    .expect("string should not fail to write");
                if i != fields.len() {
                    tmp_buf.push_str(", ")
                }
            }
            tmp_buf.pop();
            tmp_buf.pop();

            write_assign_stmt!(cenv, expr.type_data(), "{{ {} }}", tmp_buf)
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
            let arr_ident = expr_to_ident(arr_expr, cenv);
            let indices_idents: Vec<_> = indices
                .into_iter()
                .map(|i| expr_to_ident(i, cenv))
                .collect();

            // Check the bounds of the indices
            for (dim, index_ident) in indices_idents.iter().enumerate() {
                let safe_label = write_if_stmt!(cenv, "{} >= 0", index_ident);
                write_stmt!(cenv, "fail_assertion(\"negative array index\")");
                cenv.write_label(safe_label);

                let safe_label = write_if_stmt!(cenv, "{} < {}.d{}", index_ident, arr_ident, dim);
                write_stmt!(cenv, "fail_assertion(\"index too large\")");
                cenv.write_label(safe_label);
            }

            let combined_index = write_assign_stmt!(cenv, &TypeVal::Int, "0");
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
            let condition_ident = expr_to_ident(cond, cenv);
            cenv.gen_arr_struct(expr.type_data());
            let output_ident = cenv.write_var_declaration(output_type);
            let false_label = write_if_stmt!(cenv, "!{}", condition_ident);
            let true_ident = expr_to_ident(true_branch, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, true_ident);
            let end_label = cenv.gen_jump_sym();
            cenv.write_goto(end_label);
            cenv.write_label(false_label);
            let false_ident = expr_to_ident(false_branch, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, false_ident);
            cenv.write_label(end_label);
            output_ident
        }

        kind
        @ (ExprKind::ArrayComp(loop_bounds, body, _) | ExprKind::Sum(loop_bounds, body, _)) => {
            let is_array_comp = matches!(kind, ExprKind::ArrayComp(_, _, _));

            let output_ident = if is_array_comp {
                let arr_type = expr.type_data();
                assert!(matches!(arr_type, TypeVal::Array(_, _)));
                cenv.gen_arr_struct(expr.type_data());
                cenv.write_var_declaration(arr_type)
            } else {
                cenv.write_var_declaration(body.type_data())
            };

            let bounds_idents: Vec<_> = loop_bounds
                .into_iter()
                .enumerate()
                .map(|(i, LoopVar(name, expr))| {
                    let bound_ident = expr_to_ident(expr, cenv);
                    if is_array_comp {
                        write_stmt!(cenv, "{}.d{} = {}", output_ident, i, bound_ident);
                    }
                    let ok_label = write_if_stmt!(cenv, "{} > 0", bound_ident);
                    write_stmt!(cenv, "fail_assertion(\"non-positive loop bound\")");
                    cenv.write_label(ok_label);
                    bound_ident
                })
                .collect();

            if is_array_comp {
                let arr_size_ident = write_assign_stmt!(cenv, &TypeVal::Int, "1");
                for bound_ident in bounds_idents.iter() {
                    write_stmt!(cenv, "{} *= {}", arr_size_ident, bound_ident);
                }
                // Extra allocation not ideal
                let mut arr_element_type_str = String::new();
                write_type(&mut arr_element_type_str, body.type_data(), cenv.env);

                write_stmt!(
                    cenv,
                    "{} *= sizeof({})",
                    arr_size_ident,
                    arr_element_type_str
                );
                //_0.data = jpl_alloc(_4);
                write_stmt!(
                    cenv,
                    "{}.data = jpl_alloc({})",
                    output_ident,
                    arr_size_ident
                );
            } else {
                write_stmt!(cenv, "{} = 0", output_ident);
            }

            // List out the looping variables in reverse order
            //    int64_t _5 = 0; // c
            //int64_t _6 = 0; // b
            //int64_t _7 = 0; // a
            let mut looping_vars: Vec<_> = loop_bounds
                .iter()
                .rev()
                .map(|LoopVar(name, _)| {
                    let loop_var_ident = write_assign_stmt!(cenv, &TypeVal::Int, "0");
                    codegen_lvalue(cenv, &LValue::from_span(*name), loop_var_ident);
                    loop_var_ident
                })
                .collect::<Vec<_>>();
            looping_vars.reverse();
            let looping_vars = looping_vars;
            let loop_begining = cenv.gen_jump_sym();
            cenv.write_label(loop_begining);
            let body_output_ident = expr_to_ident(body, cenv);

            if is_array_comp {
                //TODO: use helper function for this
                let combined_index = write_assign_stmt!(cenv, &TypeVal::Int, "0");
                for (i, index_ident) in looping_vars.iter().enumerate() {
                    write_stmt!(cenv, "{} *= {}.d{}", combined_index, output_ident, i);
                    write_stmt!(cenv, "{} += {}", combined_index, index_ident);
                }
                write_stmt!(
                    cenv,
                    "{}.data[{}] = {}",
                    output_ident,
                    combined_index,
                    body_output_ident
                );
            } else {
                write_stmt!(cenv, "{} += {}", output_ident, body_output_ident,);
            }
            for (i, (looping_var_ident, bound_ident)) in looping_vars
                .iter()
                .zip(bounds_idents.iter())
                .enumerate()
                .rev()
            {
                write_stmt!(cenv, "{}++", looping_var_ident);
                write_if_stmt!(
                    cenv,
                    loop_begining,
                    "{} < {}",
                    looping_var_ident,
                    bound_ident
                );
                if i != 0 {
                    write_stmt!(cenv, "{} = 0", looping_var_ident)
                }
            }

            output_ident
        }
        kind @ (ExprKind::Or(args) | ExprKind::And(args)) => {
            let is_or = matches!(kind, ExprKind::Or(_));
            let [lhs, rhs] = args.as_ref();
            // IDK why the autograder wants it this way
            let output_ident = cenv.cur_fn().gen_sym();
            let lhs_ident = expr_to_ident(lhs, cenv);

            // Had to inline macro because autograder want ouput symbol before lhs but after it
            // in the code
            let cur_fn = &mut cenv.fns[cenv.cur_fn];
            cur_fn.src.push_str(INDENTATION);
            write_type(&mut cur_fn.src, &TypeVal::Bool, cenv.env);
            write!(cur_fn.src, " {} = ", output_ident).expect("string should not fail to write");
            let src = cenv.fns[cenv.cur_fn].src();
            write!(src, "{}", lhs_ident).expect("string should not fail to write");
            src.push_str(";\n");

            let cmp_op = if is_or { "!=" } else { "==" };
            let done_label = write_if_stmt!(cenv, "0 {} {}", cmp_op, lhs_ident);
            let rhs_ident = expr_to_ident(rhs, cenv);
            write_stmt!(cenv, "{} = {}", output_ident, rhs_ident);
            cenv.write_label(done_label);
            output_ident
        }
        ExprKind::LessThan(args) => codegen_binop(cenv, "<", args, &TypeVal::Bool),
        ExprKind::GreaterThan(args) => codegen_binop(cenv, ">", args, &TypeVal::Bool),
        ExprKind::LessThanEq(args) => codegen_binop(cenv, "<=", args, &TypeVal::Bool),
        ExprKind::GreaterThanEq(args) => codegen_binop(cenv, ">=", args, &TypeVal::Bool),
        ExprKind::Eq(args) => codegen_binop(cenv, "==", args, &TypeVal::Bool),
        ExprKind::NotEq(args) => codegen_binop(cenv, "!=", args, &TypeVal::Bool),
        ExprKind::Add(args) => codegen_binop(cenv, "+", args, expr.type_data()),
        ExprKind::Minus(args) => codegen_binop(cenv, "-", args, expr.type_data()),
        ExprKind::Mulitply(args) => codegen_binop(cenv, "*", args, expr.type_data()),
        ExprKind::Divide(args) => codegen_binop(cenv, "/", args, expr.type_data()),
        ExprKind::Modulo(args) => match expr.type_data() {
            TypeVal::Int => codegen_binop(cenv, "%", args, expr.type_data()),
            TypeVal::Float => {
                let [lhs, rhs] = &**args;
                let lhs_ident = expr_to_ident(lhs, cenv);
                let rhs_ident = expr_to_ident(rhs, cenv);
                write_assign_stmt!(cenv, &TypeVal::Float, "fmod({}, {})", lhs_ident, rhs_ident)
            }
            _ => unreachable!(),
        },
        ExprKind::Not(inner_expr) => {
            assert_eq!(inner_expr.type_data(), expr.type_data());
            let inner_ident = expr_to_ident(inner_expr, cenv);
            write_assign_stmt!(cenv, &TypeVal::Bool, "!{}", inner_ident)
        }
        ExprKind::Negation(inner_expr) => {
            assert_eq!(inner_expr.type_data(), expr.type_data());
            let inner_ident = expr_to_ident(inner_expr, cenv);
            write_assign_stmt!(cenv, expr.type_data(), "-{}", inner_ident)
        }
    }
}

fn codegen_binop<'b>(
    env: &mut CGenEnv<'_, 'b>,
    op: &str,
    args: &[Expr; 2],
    output: &TypeVal,
) -> Ident<'b> {
    let [lhs, rhs] = args;
    let lhs_ident = expr_to_ident(lhs, env);
    let rhs_ident = expr_to_ident(rhs, env);
    write_assign_stmt!(env, output, "{} {} {}", lhs_ident, op, rhs_ident)
}
