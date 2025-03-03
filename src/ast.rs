use crate::{
    c_codegen::CGenEnv,
    environment::Environment,
    lex::{Lexer, TokenType},
    parse::{expect_tokens, next_match, parse_sequence_trailing, TokenStream},
    typecheck::{TypeState, Typed, UnTyped},
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
pub struct Program<T: TypeState = UnTyped> {
    commands: Box<[Cmd<T>]>,
}

impl Parse<Program> for Program {
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

    pub fn typecheck(self, env: &mut Environment) -> miette::Result<Program<Typed>> {
        Ok(Program {
            commands: self
                .commands()
                .to_vec()
                .into_iter()
                .map(|e| e.typecheck(env))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        })
    }

    pub fn to_c(&self, c_gen_env: &mut CGenEnv<'_, '_>) {
        for cmd in &self.commands {
            cmd.to_c(c_gen_env);
        }
    }
}

impl<T: TypeState> Program<T> {
    /// Returns the commands in the program.
    pub fn commands(&self) -> &[Cmd<T>] {
        &self.commands
    }
}

//impl<'a> Typecheck for Program<'a> {
//    fn check(&mut self, env: &mut typecheck::Environment) -> miette::Result<()> {
//        for cmd in &self.commands {
//            cmd.check(env)?;
//        }
//        Ok(())
//    }
//}
