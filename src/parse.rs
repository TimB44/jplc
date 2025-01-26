//! Defines types and methods for parsing a JPL program from a Lexer's token stream.
//!
//! Submodules:
//! - `auxiliary`: Auxiliary types like `Strings`, `LValues`, etc.
//! - `cmd`: Command parsing
//! - `exrp`: Expression parsing
//! - `types`: JPL variable types (WIP)
use std::collections::VecDeque;

use cmd::Cmd;

use crate::lex::{Lexer, Token, TokenType};
use miette::{miette, LabeledSpan, Severity};

pub mod auxiliary;
pub mod cmd;
pub mod exrp;
pub mod stmt;
pub mod types;

/// Something that can be parsed from a token stream
trait Parse: Sized {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self>;
}

/// Parses P delimiter ... P and returns a boxed slice of P
/// Does not consume the terminating token or delimiter
fn parse_sequence<P: Parse>(
    ts: &mut TokenStream,
    delimiter: TokenType,
    terminator: TokenType,
) -> miette::Result<Box<[P]>> {
    let mut items = Vec::new();

    // Check for empty sequences
    if ts.peek().map(|t| t.kind()) == Some(terminator) {
        return Ok(items.into_boxed_slice());
    }

    loop {
        items.push(P::parse(ts)?);
        if ts.peek().map(|t| t.kind()) != Some(delimiter) {
            break;
        }
        _ = expect_tokens(ts, [delimiter])
    }

    Ok(items.into_boxed_slice())
}

/// Parses P delimiter ... and returns a boxed slice of P
/// Does not consume the terminating token
fn parse_sequence_trailing<P: Parse>(
    ts: &mut TokenStream,
    delimiter: TokenType,
    terminator: TokenType,
) -> miette::Result<Box<[P]>> {
    let mut items = Vec::new();

    // Check for empty sequences
    if ts.peek().map(|t| t.kind()) == Some(terminator) {
        return Ok(items.into_boxed_slice());
    }

    while ts.peek().map(|t| t.kind()) != Some(terminator) {
        items.push(P::parse(ts)?);
        _ = expect_tokens(ts, [delimiter])?;
    }

    Ok(items.into_boxed_slice())
}

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
        assert!(matches!(ts.next(), None));

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

/// A stream of tokens produced by a lexer, with support for peeking.
#[derive(Debug, Clone)]
pub struct TokenStream<'a> {
    // The front of the peeked queue contains the next token to be processed.
    peeked: VecDeque<Token<'a>>,
    lexer: Lexer<'a>,
}

impl<'a> Iterator for TokenStream<'a> {
    type Item = Token<'a>;

    /// Returns the next token from the peeked queue or the lexer.
    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.pop_front().or_else(|| self.lexer.next())
    }
}

impl<'a> TokenStream<'a> {
    /// Creates a new `TokenStream` wrapping the given lexer.
    pub fn new(lexer: Lexer<'a>) -> Self {
        Self {
            peeked: VecDeque::new(),
            lexer,
        }
    }

    /// Returns the next token that will be yielded by the iterator.
    pub fn peek(&mut self) -> Option<&Token<'a>> {
        self.peek_at(1)
    }

    /// Peeks forward a specified number of items in the iterator.
    ///
    /// # Panics
    ///
    /// If `forward` is 0.
    fn peek_at(&mut self, forward: usize) -> Option<&Token<'a>> {
        assert!(forward != 0);

        for _ in self.peeked.len()..forward {
            self.peeked.push_back(self.lexer.next()?);
        }
        self.peeked.get(forward - 1)
    }

    /// Returns the lexer contained within this `TokenStream`.
    fn lexer(&self) -> Lexer<'a> {
        self.lexer
    }
}

/// Checks if the next tokens from the `TokenStream` match the given patterns, without modifying
/// the stream.
///
/// Returns `true` if all tokens match, `false` otherwise.
macro_rules! next_match {
    ($token_stream:expr, $($token_type:pat),+ ) => {
        {
            let mut ahead = 0;
            let mut all_match = true;
            $(
                ahead += 1;
                all_match = all_match &&  matches!($token_stream.peek_at(ahead).map(|t| t.kind()), Some($token_type));
            )*
            all_match
        }
    };
}
use next_match;

/// Consumes tokens from the `TokenStream` and checks if they match the expected token types.
///
/// Returns the matched tokens if all match. If any token does not match, returns an error with
/// details about the mismatch or missing token.
fn expect_tokens<'a, const N: usize>(
    ts: &mut TokenStream<'a>,
    expected: [TokenType; N],
) -> miette::Result<[Token<'a>; N]> {
    let mut out = [None; N];
    for (i, et) in expected.iter().enumerate() {
        out[i] = match ts.next() {
            Some(t) if t.kind() == *et => Some(t),
            Some(t) => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some(format!("expected: {}, found: {}", et, t.kind())),
                        t.start(),
                        t.bytes().len()
                    )],
                    "Unexpected token found"
                ))
            }
            None => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::at_offset(
                        ts.lexer().bytes().len() - 1,
                        format!("expected: {}", et)
                    )],
                    "Missing expected token"
                ))
            }
        }
    }

    Ok(out.map(Option::unwrap))
}
