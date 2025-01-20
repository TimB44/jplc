use std::collections::VecDeque;

use cmd::Cmd;

use crate::lex::{Lexer, Token, TokenType};
use miette::{miette, LabeledSpan, Severity};

pub mod auxiliary;
pub mod cmd;
pub mod exrp;
pub mod types;

pub struct Program {
    commands: Vec<Cmd>,
}

impl Program {
    pub fn new(lexer: Lexer) -> miette::Result<Self> {
        Self::parse(&mut TokenStream::new(lexer))
    }

    pub fn commands(&self) -> &[Cmd] {
        &self.commands
    }
}

impl Parse for Program {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self> {
        let mut commands = Vec::new();

        // Remove a potential leading newline
        if matches!(ts.peek().map(|t| t.kind()), Some(TokenType::Newline)) {
            _ = ts.next()
        }

        while !next_matches!(ts, TokenType::Eof) {
            let cmd = Cmd::parse(ts)?;
            commands.push(cmd);
            tokens_match(ts, [TokenType::Newline])?;
        }
        Ok(Self { commands })
    }
}

trait Parse: Sized {
    fn parse(ts: &mut TokenStream) -> miette::Result<Self>;
}

#[derive(Debug, Clone)]
struct TokenStream<'a> {
    // Queue contains all of the peeked tokens. The tokens in the front come before the tokens in
    // the back
    peeked: VecDeque<Token<'a>>,
    lexer: Lexer<'a>,
}

impl<'a> Iterator for TokenStream<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.pop_front().or_else(|| self.lexer.next())
    }
}

impl<'a> TokenStream<'a> {
    fn new(lexer: Lexer<'a>) -> Self {
        Self {
            peeked: VecDeque::new(),
            lexer,
        }
    }

    fn peek(&mut self) -> Option<&Token<'a>> {
        self.peek_at(1)
    }

    fn peek_at(&mut self, forward: usize) -> Option<&Token<'a>> {
        assert!(forward != 0);

        for _ in self.peeked.len()..forward {
            self.peeked.push_back(self.lexer.next()?);
        }
        self.peeked.get(forward - 1)
    }

    fn lexer(&self) -> Lexer<'a> {
        self.lexer
    }
}

macro_rules! next_matches {
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
pub(self) use next_matches;

fn tokens_match<'a, const N: usize>(
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
