use std::collections::VecDeque;

use cmd::Cmd;

use crate::lex::{Lexer, Token, TokenType};
use miette::{miette, LabeledSpan, Severity};

mod auxiliary;
mod cmd;
mod exrp;
mod types;

struct Program(Vec<Cmd>);

// TODO: maybe use a trait. IDK if it would be usefull later
impl Program {
    fn parse(lexer: Lexer) -> Program {
        let mut token_stream = TokenStream::new(lexer);
        Cmd::parse(&mut token_stream);

        todo!()
    }
}

pub trait Parse: Sized {
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
        self.peeked.pop_front().or(self.lexer.next())
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

    todo!()
}
