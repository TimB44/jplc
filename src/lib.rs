use std::process::exit;

use cli::Mode;
use lex::Lexer;
use miette::NamedSource;
use parse::Program;
use utils::exit_with_error;

pub mod cli;
pub mod utils;

mod lex;
mod parse;

//TODO: Unsure about what to return
pub fn compile(source_name: String, source: String, mode: Mode) {
    // TODO: Remove assertion for future assignemtns
    assert!(
        mode.lex | mode.parse,
        "Only lexing and (some of) parsing implemented"
    );

    let token_stream = Lexer::new(&source_name, &source);
    if mode.lex {
        for token in token_stream {
            println!("{}", token)
        }
        println!("Compilation succeeded: lexical analysis complete");
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

        println!("Compilation succeeded: parsing complete");
        exit(0);
    }
}
