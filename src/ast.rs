use crate::{
    lex::{Lexer, TokenType},
    parse::{expect_tokens, next_match, parse_sequence_trailing, TokenStream},
};

use super::parse::Parse;
use cmd::Cmd;

pub mod auxiliary;
pub mod cmd;
pub mod expr;
pub mod stmt;
pub mod types;

/// Represents an entire JPL program, defined as a sequence of commands.
///
/// This is the top-level item responsible for parsing all other components.
#[derive(Debug, Clone)]
pub struct Program {
    commands: Box<[Cmd]>,
}

impl Parse for Program {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        // Remove a potential leading newline
        if next_match!(ts, TokenType::Newline) {
            _ = expect_tokens(ts, [TokenType::Newline])?;
        }

        let commands = parse_sequence_trailing(ts, TokenType::Newline, TokenType::Eof)?;

        expect_tokens(ts, [TokenType::Eof])?;
        assert!(ts.next().is_none());

        Ok(Self { commands })
    }
}

impl Program {
    /// Creates a `Program` by parsing tokens from the lexer.
    pub fn new(lexer: Lexer) -> miette::Result<Self> {
        Self::parse(&mut TokenStream::new(lexer))
    }

    /// Returns the commands in the program.
    pub fn commands(&self) -> &[Cmd] {
        &self.commands
    }
}
