use core::fmt::Display;
use regex::bytes::Regex;
use std::{collections::HashMap, process::exit, str, sync::LazyLock};

static VARIABLE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("^[a-zA-Z][a-zA-Z0-9_]*").expect("regex invalid"));

static STRING_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^"[ !#-~]*""#).expect("regex invalid"));

static FLOAT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?:[0-9]+\.[0-9]*)|(?:[0-9]*\.[0-9]+))").expect("regex invalid")
});

static INT_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new("^[0-9]+").expect("regex invalid"));

// This regex should match all valid white space and comments matching until it finds another
// token. If the comment is invalid (single line comment with no newling or multi line comment
// with no */) then it will not match. It will also capture the last unescaped newline in ln group.
// This is used to determine if the text contained a newline and to find its location in the source
// for the token
static COMMENT_WHITESPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?:/\*[\n -~]*?\*/)|(?://[ -~]*)|(?:\\\n)|(?<ln>\n)|(?: ))+")
        .expect("regex invalid")
});

//TODO: use a pnf map instead https://docs.rs/phf/latest/phf/
static KEYWORDS: LazyLock<HashMap<&[u8], TokenType>> = LazyLock::new(|| {
    HashMap::from([
        (b"array".as_slice(), TokenType::Array),
        (b"assert", TokenType::Assert),
        (b"bool", TokenType::Bool),
        (b"else", TokenType::Else),
        (b"equals", TokenType::Equals),
        (b"false", TokenType::False),
        (b"float", TokenType::Float),
        (b"if", TokenType::If),
        (b"image", TokenType::Image),
        (b"int", TokenType::Int),
        (b"let", TokenType::Let),
        (b"fn", TokenType::Fn),
        (b"print", TokenType::Print),
        (b"read", TokenType::Read),
        (b"return", TokenType::Return),
        (b"show", TokenType::Show),
        (b"struct", TokenType::Struct),
        (b"sum", TokenType::Sum),
        (b"then", TokenType::Then),
        (b"time", TokenType::Time),
        (b"to", TokenType::To),
        (b"true", TokenType::True),
        (b"void", TokenType::Void),
        (b"write", TokenType::Write),
    ])
});

//TODO: Keep track of lines and columns for better error messages

/// Converts JPL source code into tokens.
/// This struct implements the interator trait which outputs Tokens
/// If a lex occurs then an error message will be printed and the process will exit
#[derive(Debug, Clone, Copy)]
pub struct Lexer<'a> {
    bytes: &'a [u8],
    cur: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            bytes: source.as_bytes(),
            cur: 0,
        }
    }

    /// Creates a token at the current location in the lexer and updates the location of the cursor
    fn create_token(&mut self, kind: TokenType, len: usize) -> Token<'a> {
        let token = Token {
            start: self.cur,
            bytes: str::from_utf8(&self.bytes[self.cur..(self.cur + len)])
                .expect("lexer bytes must come from a string"),
            kind,
        };
        self.cur += len;
        return token;
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.bytes.get(self.cur) {
                None => {
                    // If this is the first time we are beyond the bytes then reutrn EOF, otherwise
                    // return None
                    return if self.cur == self.bytes.len() {
                        self.cur += 1;
                        Some(Token {
                            start: self.cur - 1,
                            bytes: "",
                            kind: TokenType::Eof,
                        })
                    } else {
                        None
                    };
                }

                // Simple single character tokens
                Some(b':') => return Some(self.create_token(TokenType::Colon, 1)),
                Some(b',') => return Some(self.create_token(TokenType::Comma, 1)),
                Some(b'{') => return Some(self.create_token(TokenType::LCurly, 1)),
                Some(b'}') => return Some(self.create_token(TokenType::RCurly, 1)),
                Some(b'(') => return Some(self.create_token(TokenType::LParen, 1)),
                Some(b')') => return Some(self.create_token(TokenType::RParen, 1)),
                Some(b'[') => return Some(self.create_token(TokenType::LSquare, 1)),
                Some(b']') => return Some(self.create_token(TokenType::RSquare, 1)),

                // Simple operators
                Some(b'+') | Some(b'-') | Some(b'*') | Some(b'%') => {
                    return Some(self.create_token(TokenType::Op, 1));
                }

                // Variables and keywords
                Some(b'a'..=b'z') | Some(b'A'..=b'Z') => {
                    let var: &[u8] = VARIABLE_REGEX
                        .find(&self.bytes[self.cur..])
                        .expect("regex should match due to match arm")
                        .as_bytes();
                    let kind = *KEYWORDS.get(var).unwrap_or(&TokenType::Variable);

                    return Some(self.create_token(kind, var.len()));
                }

                // Int and Floats Literals

                // TODO: Should we check for constant overflow here?
                Some(b'0'..=b'9') | Some(b'.') => {
                    match (
                        FLOAT_REGEX.find(&self.bytes[self.cur..]),
                        INT_REGEX.find(&self.bytes[self.cur..]),
                    ) {
                        (Some(float_lit), _) => {
                            return Some(self.create_token(TokenType::FloatLit, float_lit.len()));
                        }
                        (None, Some(int_lit)) => {
                            return Some(self.create_token(TokenType::IntLit, int_lit.len()))
                        }
                        (None, None) => return Some(self.create_token(TokenType::Dot, 1)),
                    }
                }

                // String literals
                Some(b'"') => match STRING_REGEX.find(&self.bytes[self.cur..]) {
                    Some(str_lit) => {
                        return Some(self.create_token(TokenType::StringLit, str_lit.len()))
                    }
                    None => {
                        //TODO: Better error message or return an error
                        println!("Compilation Failed: Unmatched \" found");
                        exit(1);
                    }
                },

                invalid_char @ (Some(0..=9) | Some(11..=31) | Some(127..)) => {
                    println!(
                        "Compilation failed: Invalid character {:02X?}",
                        invalid_char.unwrap()
                    );
                    exit(1);
                }

                // These characters are valid in some locations (strings, comments, etc.), however
                // they are never start a token
                unexpected_char @ (Some(b'#') | Some(b'$') | Some(b'\'') | Some(b';')
                | Some(b'?') | Some(b'@') | Some(b'^') | Some(b'_')
                | Some(b'`') | Some(b'~')) => {
                    println!(
                        "Compilation failed: Unexpected character {}",
                        unexpected_char.unwrap(),
                    );

                    exit(1);
                }

                // These will be lexed in the following match statment as they need info about the
                // next char
                Some(b'\n') | Some(b' ') | Some(b'=') | Some(b'>') | Some(b'<') | Some(b'!')
                | Some(b'&') | Some(b'|') | Some(b'/') | Some(b'\\') => (),
            };

            // More complex single and double character tokens
            match (self.bytes[self.cur], self.bytes.get(self.cur + 1)) {
                (b'=', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
                (b'=', _) => return Some(self.create_token(TokenType::Equals, 1)),

                (b'>', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
                (b'>', _) => return Some(self.create_token(TokenType::Op, 1)),

                (b'<', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
                (b'<', _) => return Some(self.create_token(TokenType::Op, 1)),

                (b'!', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
                (b'!', _) => return Some(self.create_token(TokenType::Op, 1)),

                (b'&', Some(b'&')) => return Some(self.create_token(TokenType::Op, 2)),
                (b'|', Some(b'|')) => return Some(self.create_token(TokenType::Op, 2)),

                // If there arent 2 of them they can not start a vlid token
                invalid_op @ ((b'|', _) | (b'&', _)) => {
                    // TODO: better error, Mention that it takes 2 of them
                    println!("Compilation failed: Unexpected character {}", invalid_op.0);
                    exit(0);
                }

                // Skip over escaped newlines
                (b'\\', Some(b'\n')) => {
                    self.cur += 2;
                    continue;
                }
                (b'\\', _) => {
                    println!(
                        "Compilation failed: Invalid \\ found. This can only be used in strings, comments, and to escape newline characters"
                    );
                    exit(1);
                }

                // Single line comments and newlines
                // These must contain a new line to lex
                (b'/', Some(b'/')) | (b'\n', _) => {
                    match COMMENT_WHITESPACE_REGEX.captures(&self.bytes[self.cur..]) {
                        Some(c) => {
                            let entire = c.get(0).unwrap();
                            let last_ln = c
                                .name("ln")
                                .expect("match should contain newline due to regex pattern");

                            let t = Token {
                                start: self.cur + last_ln.start(),
                                bytes: str::from_utf8(last_ln.as_bytes())
                                    .expect("lexer bytes must come from a string"),
                                kind: TokenType::Newline,
                            };

                            self.cur += entire.len();
                            return Some(t);
                        }
                        None => {
                            //TODO: better error messages. also should this be a lex error?
                            println!("Compilation failed: Single line comment missing newline");
                            exit(1);
                        }
                    };
                }

                // Multi-line comments and space character. These may not contain a newline,
                // however a multi line comment must closed in order to lex
                (b'/', Some(b'*')) | (b' ', _) => {
                    match COMMENT_WHITESPACE_REGEX.captures(&self.bytes[self.cur..]) {
                        Some(c) => {
                            let entire = c.get(0).unwrap();
                            if let Some(last_ln) = c.name("ln") {
                                let t = Token {
                                    start: self.cur + last_ln.start(),
                                    bytes: str::from_utf8(last_ln.as_bytes())
                                        .expect("lexer bytes must come from a string"),
                                    kind: TokenType::Newline,
                                };
                                self.cur += entire.len();
                                return Some(t);
                            } else {
                                self.cur += entire.len();
                                continue;
                            }
                        }
                        None => {
                            //TODO: better error messages.
                            println!("Compilation failed: Unterminated multi-line comment");
                            exit(1);
                        }
                    };
                }

                // From here we know the slash can no be a comment so it is ok to assume it as
                // division operator
                (b'/', _) => return Some(self.create_token(TokenType::Op, 1)),

                _ => todo!(),
            };
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    start: usize,
    bytes: &'a str,
    kind: TokenType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Array,
    Assert,
    Bool,
    Colon,
    Comma,
    Dot,
    Else,
    Eof,
    Equals,
    False,
    Float,

    // TODO: Do i need to check for overflow?
    FloatLit,
    Fn,
    If,
    Image,
    Int,

    // TODO: should i check for overflow now or later
    IntLit,
    LCurly,
    Let,
    LParen,
    LSquare,
    Newline,
    Op,
    Print,
    RCurly,
    Read,
    Return,
    RParen,
    RSquare,
    Show,
    StringLit,
    Struct,
    Sum,
    Then,
    Time,
    To,
    True,
    Variable,
    Void,
    Write,
}

impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            TokenType::Array => write!(f, "ARRAY 'array'"),
            TokenType::Assert => write!(f, "ASSERT 'assert'"),
            TokenType::Bool => write!(f, "BOOL 'bool'"),
            TokenType::Colon => write!(f, "COLON ':'"),
            TokenType::Comma => write!(f, "COMMA ','"),
            TokenType::Dot => write!(f, "DOT '.'"),
            TokenType::Else => write!(f, "ELSE 'else'"),
            TokenType::Eof => write!(f, "END_OF_FILE"),
            TokenType::Equals => write!(f, "EQUALS '='"),
            TokenType::False => write!(f, "FALSE 'false'"),
            TokenType::Float => write!(f, "FLOAT 'float'"),
            TokenType::FloatLit => write!(f, "FLOATVAL '{}'", self.bytes),
            TokenType::Fn => write!(f, "FN 'fn'"),
            TokenType::If => write!(f, "IF 'if'"),
            TokenType::Image => write!(f, "IMAGE 'image'"),
            TokenType::Int => write!(f, "INT 'int'"),
            TokenType::IntLit => write!(f, "INTVAL '{}'", self.bytes),
            TokenType::LCurly => write!(f, "LCURLY '{{'"),
            TokenType::Let => write!(f, "LET 'let'"),
            TokenType::LParen => write!(f, "LPAREN '('"),
            TokenType::LSquare => write!(f, "LSQUARE '['"),
            TokenType::Newline => write!(f, "NEWLINE"),
            TokenType::Op => write!(f, "OP '{}'", self.bytes),
            TokenType::Print => write!(f, "PRINT 'print'"),
            TokenType::RCurly => write!(f, "RCURLY '}}'"),
            TokenType::Read => write!(f, "READ 'read'"),
            TokenType::Return => write!(f, "RETURN 'return'"),
            TokenType::RParen => write!(f, "RPAREN ')'"),
            TokenType::RSquare => write!(f, "RSQUARE ']'"),
            TokenType::Show => write!(f, "SHOW 'show'"),
            TokenType::StringLit => write!(f, "STRING '{}'", self.bytes),
            TokenType::Struct => write!(f, "STRUCT 'struct'"),
            TokenType::Sum => write!(f, "SUM 'sum'"),
            TokenType::Then => write!(f, "THEN 'then'"),
            TokenType::Time => write!(f, "TIME 'time'"),
            TokenType::To => write!(f, "TO 'to'"),
            TokenType::True => write!(f, "TRUE 'true'"),
            TokenType::Variable => write!(f, "VARIABLE '{}'", self.bytes),
            TokenType::Void => write!(f, "VOID 'void'"),
            TokenType::Write => write!(f, "WRITE 'write'"),
        }
    }
}

#[cfg(test)]
mod lexer_tests {

    use crate::lex::{Token, TokenType};

    use super::Lexer;

    #[test]
    fn ops_and_chars() {
        let source = "[,:+}";
        let tokens: Vec<TokenType> = Lexer::new(source).map(|t| t.kind).collect();
        assert_eq!(
            vec![
                TokenType::LSquare,
                TokenType::Comma,
                TokenType::Colon,
                TokenType::Op,
                TokenType::RCurly,
                TokenType::Eof,
            ],
            tokens
        );
    }

    #[test]
    fn variables() {
        let source = "[fns] hi notakeyword";
        let tokens: Vec<_> = Lexer::new(source).collect();
        assert_eq!(
            vec![
                Token {
                    start: 0,
                    bytes: "[",
                    kind: TokenType::LSquare
                },
                Token {
                    start: 1,
                    bytes: "fns",
                    kind: TokenType::Variable
                },
                Token {
                    start: 4,
                    bytes: "]",
                    kind: TokenType::RSquare
                },
                Token {
                    start: 6,
                    bytes: "hi",
                    kind: TokenType::Variable
                },
                Token {
                    start: 9,
                    bytes: "notakeyword",
                    kind: TokenType::Variable
                },
                Token {
                    start: source.len(),
                    bytes: "",
                    kind: TokenType::Eof
                },
            ],
            tokens
        );
    }

    #[test]
    fn variables_and_keywords() {
        let source = "fn let hi Let show";
        let tokens: Vec<_> = Lexer::new(source).collect();
        assert_eq!(
            vec![
                Token {
                    start: 0,
                    bytes: "fn",
                    kind: TokenType::Fn
                },
                Token {
                    start: 3,
                    bytes: "let",
                    kind: TokenType::Let
                },
                Token {
                    start: 7,
                    bytes: "hi",
                    kind: TokenType::Variable
                },
                Token {
                    start: 10,
                    bytes: "Let",
                    kind: TokenType::Variable
                },
                Token {
                    start: 14,
                    bytes: "show",
                    kind: TokenType::Show
                },
                Token {
                    start: source.len(),
                    bytes: "",
                    kind: TokenType::Eof
                },
            ],
            tokens
        );
    }
}
