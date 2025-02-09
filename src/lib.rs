use std::{process::exit, time::Instant};

use ast::Program;
use cli::Mode;
use lex::Lexer;
use miette::NamedSource;
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
        exit(0);
    }

    let program = match Program::new(token_stream) {
        Ok(program) => program,
        Err(err) => exit_with_error(err.with_source_code(NamedSource::new(source_name, source))),
    };

    if mode.parse {
        for cmd in program.commands() {
            println!("{}", cmd.to_s_expresion(source.as_bytes()))
        }

        println!(
            "Compilation succeeded: parsing complete in {}ms",
            Instant::now().duration_since(start_time).as_millis()
        );
        exit(0);
    }

    unreachable!()
}
