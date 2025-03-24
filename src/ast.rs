use crate::parse::{Displayable, SExpr};
use crate::{
    environment::Environment,
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

impl Program {
    /// Creates a `Program` by parsing tokens from the lexer.
    pub fn new<'a>(lexer: Lexer<'a>, env: &mut Environment<'a>) -> miette::Result<Self> {
        Self::parse(&mut TokenStream::new(lexer), env)
    }

    //pub fn typecheck(self, env: &mut Environment) -> miette::Result<Program<Typed>> {
    //    Ok(Program {
    //        commands: self
    //            .commands()
    //            .to_vec()
    //            .into_iter()
    //            .map(|e| e.typecheck(env))
    //            .collect::<Result<Vec<_>, _>>()?
    //            .into_boxed_slice(),
    //    })
    //}
    pub fn commands(&self) -> &[Cmd] {
        &self.commands
    }
}

impl Parse for Program {
    fn parse(ts: &mut TokenStream, env: &mut Environment) -> miette::Result<Self> {
        // Remove a potential leading newline
        if next_match!(ts, TokenType::Newline) {
            expect_tokens(ts, [TokenType::Newline])?;
        }

        let commands = parse_sequence_trailing(ts, env, TokenType::Newline, TokenType::Eof)?;

        expect_tokens(ts, [TokenType::Eof])?;
        assert!(ts.next().is_none());

        Ok(Self { commands })
    }
}

impl SExpr for Program {
    fn to_s_expr(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        env: &Environment<'_>,
        opt: crate::parse::SExprOptions,
    ) -> std::fmt::Result {
        for cmd in &self.commands {
            writeln!(f, "{}", Displayable(cmd, env, opt))?;
        }
        Ok(())
    }
}
