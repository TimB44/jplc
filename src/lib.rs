use std::time::Instant;

use asm_codegen::AsmEnv;
use ast::Program;
use c_codegen::generate_c;
use cli::{Mode, OptLevel};
use environment::Environment;
use lex::Lexer;
use miette::NamedSource;
use parse::{Displayable, SExprOptions};
use utils::exit_with_error;

pub mod cli;
pub mod utils;

mod asm_codegen;
mod ast;
mod c_codegen;
mod environment;
mod lex;
mod parse;
mod typecheck;

//TODO: Unsure about what to return
pub fn compile(source_name: String, source: String, mode: Mode, opt_level: OptLevel) {
    // TODO: Remove assertion for future assignments
    // todo: make mode an enum
    assert!(
        mode.lex | mode.parse | mode.typecheck | mode.c_ir | mode.assembly,
        "Only lexing, parsing, typechecking, C IR, and parts assembly codegen implemented"
    );
    let start_time = Instant::now();

    let token_stream = Lexer::new(&source_name, &source);
    if mode.lex {
        for token in token_stream {
            println!("{}", token)
        }
        println!(
            "Compilation succeeded: lexical analysis complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    let mut env = Environment::new(source.as_bytes());

    let program = match Program::new(token_stream, &mut env) {
        Ok(program) => program,
        Err(err) => exit_with_error(err.with_source_code(NamedSource::new(source_name, source))),
    };

    if mode.parse {
        println!("{}", Displayable(&program, &env, SExprOptions::Untyped));

        println!(
            "Compilation succeeded: parsing complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    if mode.typecheck {
        println!("{}", Displayable(&program, &env, SExprOptions::Typed));

        println!(
            "Compilation succeeded: type analysis complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    if mode.c_ir {
        generate_c(program, &env);
        println!(
            "Compilation succeeded: c code generation complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    if mode.assembly {
        let asm = AsmEnv::new(&env, program, opt_level);
        print!("{}", asm);
        println!(
            "Compilation succeeded: x86-64 code generation complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    unreachable!()
}
