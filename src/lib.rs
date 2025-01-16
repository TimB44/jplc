use cli::Mode;
use lex::Lexer;

pub mod cli;
mod lex;
pub mod utils;

//TODO: Unsure about what to return
pub fn compile(source_name: String, source: String, mode: Mode) -> () {
    // TODO: Remove assertion for future assignemtns
    assert!(mode.lex, "Only lexing implemented");
    let token_stream = Lexer::new(&source_name, &source);
    if mode.lex {
        for token in token_stream {
            println!("{}", token)
        }
        println!("Compilation succeeded: lexical analysis complete")
    }
}
