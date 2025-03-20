//! Defines types and methods for parsing a JPL program from a Lexer's token stream.
//!
//! Submodules:
//! - `auxiliary`: Auxiliary types like `Strings`, `LValues`, etc.
//! - `cmd`: Command parsing
//! - `exrp`: Expression parsing
//! - `types`: JPL variable types (WIP)
use std::{
    collections::VecDeque,
    fmt::{self, Display, Formatter, Write},
};

use crate::{
    environment::Environment,
    lex::{Lexer, Token, TokenType},
};
use miette::{miette, LabeledSpan, Severity};

/// Something that can be parsed from a token stream
pub(super) trait Parse: Sized {
    fn parse(ts: &mut TokenStream<'_>, env: &mut Environment<'_>) -> miette::Result<Self>;
}

pub trait SExpr {
    fn to_s_expr(
        &self,
        f: &mut Formatter<'_>,
        env: &Environment<'_>,
        opt: SExprOptions,
    ) -> fmt::Result;
}

pub struct Displayable<'a, 'b, T>(pub &'a T, pub &'a Environment<'b>, pub SExprOptions);

impl<'a, 'b, T: SExpr> Display for Displayable<'a, 'b, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.to_s_expr(f, &self.1, self.2)
    }
}

impl<'a, 'b, T: SExpr> Display for Displayable<'a, 'b, Box<[T]>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, item) in self.0.iter().enumerate() {
            write!(f, " {}", Displayable(item, &self.1, self.2))?;
        }
        Ok(())
    }
}

//impl<'a, 'b, T: SExpr> Display for Displayable<'a, 'b, Box<[T]>> {
//    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//        write!(f, "{}", Displayable(&*self.0, &self.1, self.2))
//    }
//}

#[derive(Debug, Clone, Copy)]
pub enum SExprOptions {
    Untyped,
    Typed,
}

/// Parses P delimiter ... P and returns a boxed slice of P
/// Does not consume the terminating token or delimiter
pub fn parse_sequence<P: Parse>(
    ts: &mut TokenStream,
    env: &mut Environment,
    delimiter: TokenType,
    terminator: TokenType,
) -> miette::Result<Box<[P]>> {
    let mut items = Vec::new();

    // Check for empty sequences
    if ts.peek_type() == Some(terminator) {
        return Ok(items.into_boxed_slice());
    }

    loop {
        items.push(P::parse(ts, env)?);
        if ts.peek_type() != Some(delimiter) {
            break;
        }
        _ = expect_tokens(ts, [delimiter])?;
    }

    Ok(items.into_boxed_slice())
}

/// Parses P delimiter ... and returns a boxed slice of P
/// Does not consume the terminating token
pub fn parse_sequence_trailing<P: Parse>(
    ts: &mut TokenStream,
    env: &mut Environment,
    delimiter: TokenType,
    terminator: TokenType,
) -> miette::Result<Box<[P]>> {
    let mut items = Vec::new();

    // Check for empty sequences
    if ts.peek_type() == Some(terminator) {
        return Ok(items.into_boxed_slice());
    }

    while ts.peek_type() != Some(terminator) {
        items.push(P::parse(ts, env)?);
        _ = expect_tokens(ts, [delimiter])?;
    }

    Ok(items.into_boxed_slice())
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

    /// Returns the next token type that will be yielded by the iterator.
    pub fn peek_type(&mut self) -> Option<TokenType> {
        self.peek().map(|t| t.kind())
    }

    /// Peeks forward a specified number of items in the iterator.
    ///
    /// # Panics
    ///
    /// If `forward` is 0.
    pub fn peek_at(&mut self, forward: usize) -> Option<&Token<'a>> {
        assert!(forward != 0);

        for _ in self.peeked.len()..forward {
            self.peeked.push_back(self.lexer.next()?);
        }
        self.peeked.get(forward - 1)
    }

    /// Peeks forward a specified number of items in the iterator and returns the token type.
    ///
    /// # Panics
    ///
    /// If `forward` is 0.
    pub fn peek_type_at(&mut self, forward: usize) -> Option<TokenType> {
        self.peek_at(forward).map(|t| t.kind())
    }

    /// Returns the lexer contained within this `TokenStream`.
    pub fn lexer(&self) -> Lexer<'a> {
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
                all_match = all_match &&  matches!($token_stream.peek_type_at(ahead), Some($token_type));
            )*
            all_match
        }
    };
}
pub(super) use next_match;

/// Consumes tokens from the `TokenStream` and checks if they match the expected token types.
///
/// Returns the matched tokens if all match. If any token does not match, returns an error with
/// details about the mismatch or missing token.
pub fn expect_tokens<'a, const N: usize>(
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
