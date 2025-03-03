use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::{
    ast::{cmd, Program},
    environment::Environment,
    typecheck::Typed,
    utils::Span,
};

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
    let c_gen_env = CGenEnv::new(env);
    ast.to_c(&mut c_gen_env);
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

    pub fn add_struct(&mut self, struct_name: Span) {
        let struct_info = self
            .env
            .get_struct(struct_name)
            .expect("struct must be valid after typechecking");

        // rgba is used in the rt
        if struct_info.name() == "rgba" {
            return;
        }

        self.defs.push_str("typedef struct {\n");
        for (name, jpl_type) in struct_info.fields() {
            self.defs.push_str("  ");
            write_type(&mut self.defs, jpl_type, self.env);
            self.defs.push_str(" ");
            self.defs.push_str(name);
            self.defs.push_str(";\n");
        }
        write!(self.defs, "}} {};\n\n", struct_info.name())
            .expect("string should not fail to write");
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
    fn write_assign_stmt(&mut self, t: &Typed) -> u32 {
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

#[derive(Debug, Clone)]
enum Ident<'a> {
    Local(u32),
    Global(&'a str),
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

pub fn write_type_string(s: &mut String, t: &Typed, env: &Environment) {
    match t {
        Typed::Array(typed, rank) => {
            write!(s, "(ArrayType ").expect("string should not fail to write");
            write_type_string(s, t, env);
            write!(s, " {})", rank).expect("string should not fail to write");
        }
        Typed::Struct(id) => {
            let info = env.get_struct_id(id);
            for (_, ty) in info.fields() {
                s.push_str(" ");
                write_type_string(s, t, env);
            }
        }
        Typed::Int => {
            s.push_str("(IntType)");
        }
        Typed::Bool => {
            s.push_str("(BoolType)");
        }
        Typed::Float => {
            s.push_str("(FloatType)");
        }
        Typed::Void => {
            s.push_str("(VoidType)");
        }
    }
}

macro_rules! write_stmt {
    ($cenv:expr, $($args:tt)*) => {
        let src = $cenv.cur_fn().src();
        src.push_str("  ");
        write!(src, $($args)*);
        $cenv.cur_fn().src().push_str(";");

    };
}

pub(crate) use write_stmt;
macro_rules! write_assign_stmt {
    ($cenv:expr, $ty:expr, $($args:tt)*) => {
        let id = cenv.write_assign_stmt(ty);
        write!(expr.cur_fn().src(), $($args)*);
        cenv.cur_fn.src().push_str("");
        id
    };
}

pub(crate) use write_assign_stmt;
