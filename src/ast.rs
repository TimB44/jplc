use crate::{
    lex::{Lexer, TokenType},
    parse::{expect_tokens, next_match, parse_sequence_trailing, TokenStream},
    typecheck::{self, Typecheck},
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
pub struct Program<'a> {
    commands: Box<[Cmd<'a>]>,
}

impl<'a> Parse<Program<'a>> for Program<'a> {
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

impl<'a> Program<'a> {
    /// Creates a `Program` by parsing tokens from the lexer.
    pub fn new(lexer: Lexer) -> miette::Result<Self> {
        Self::parse(&mut TokenStream::new(lexer))
    }

    /// Returns the commands in the program.
    pub fn commands(&self) -> &[Cmd] {
        &self.commands
    }
}

impl<'a> Typecheck for Program<'a> {
    fn check(&self, env: &mut typecheck::Environment) -> miette::Result<()> {
        for cmd in &self.commands {
            cmd.check(env)?;
        }
        Ok(())
    }
}
