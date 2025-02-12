use std::time::Instant;

use ast::Program;
use cli::Mode;
use lex::Lexer;
use miette::NamedSource;
use typecheck::Environment;
use utils::exit_with_error;

pub mod cli;
pub mod utils;

mod ast;
mod lex;
mod parse;
mod typecheck;

//TODO: Unsure about what to return
pub fn compile(source_name: String, source: String, mode: Mode) {
    // TODO: Remove assertion for future assignments
    assert!(
        mode.lex | mode.parse | mode.typecheck,
        "Only lexing, parsing and part of typechecking implemented"
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

    let program = match Program::new(token_stream) {
        Ok(program) => program,
        Err(err) => exit_with_error(err.with_source_code(NamedSource::new(source_name, source))),
    };

    if mode.parse {
        for cmd in program.commands() {
            println!("{}", cmd.to_s_expr(source.as_bytes()))
        }

        println!(
            "Compilation succeeded: parsing complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    let mut env = Environment::new(source.as_bytes());
    let typed_program = match program.typecheck(&mut env) {
        Ok(p) => p,
        Err(err) => exit_with_error(err.with_source_code(NamedSource::new(source_name, source))),
    };

    if mode.typecheck {
        for cmd in typed_program.commands() {
            println!("{}", cmd.to_typed_s_exprsision(&env))
        }

        println!(
            "Compilation succeeded: parsing complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        return;
    }

    unreachable!()
}
