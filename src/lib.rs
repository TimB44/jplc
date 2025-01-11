use cli::Mode;
use lex::Lexer;

pub mod cli;
mod lex;

// TOOD: Unsure about what to return
pub fn compile(source: String, mode: Mode) -> () {
    // TODO: Remove assertion for future assignemtns
    assert!(mode.lex, "Only lexing implemented");
    let token_stream = Lexer::new(&source);
    if mode.lex {
        for token in token_stream {
            println!("{}", token)
        }
    }
}
