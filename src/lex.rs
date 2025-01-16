use core::{fmt::Display, str};
use miette::{miette, LabeledSpan, NamedSource, Severity};
use regex::bytes::Regex;
use std::sync::LazyLock;

use crate::utils::exit_with_error;

// Matches variable tokens. This could also match keywords
static VARIABLE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("^[a-zA-Z][a-zA-Z0-9_]*").expect("regex invalid"));

//
static STRING_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^"[ !#-~]*""#).expect("regex invalid"));

static FLOAT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?:[0-9]+\.[0-9]*)|(?:[0-9]*\.[0-9]+))").expect("regex invalid")
});

static INT_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new("^[0-9]+").expect("regex invalid"));

// This regex should match all valid white space and comments matching until it finds another
// token. If the comment is invalid ( multi-line comment with no */) then it will not match.
// It will also capture the last unescaped newline in the nl group.
// This is used to determine if the text contained a newline and to find its location in the source
// for the token
static COMMENT_WHITESPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?:/\*[\n -~]*?\*/)|(?://[ -~]*)|(?:\\\n)|(?<nl>\n)|(?: ))+")
        .expect("regex invalid")
});

fn bytes_to_keyword(bytes: &[u8]) -> Option<TokenType> {
    match bytes {
        b"array" => Some(TokenType::Array),
        b"assert" => Some(TokenType::Assert),
        b"bool" => Some(TokenType::Bool),
        b"else" => Some(TokenType::Else),
        b"equals" => Some(TokenType::Equals),
        b"false" => Some(TokenType::False),
        b"float" => Some(TokenType::Float),
        b"if" => Some(TokenType::If),
        b"image" => Some(TokenType::Image),
        b"int" => Some(TokenType::Int),
        b"let" => Some(TokenType::Let),
        b"fn" => Some(TokenType::Fn),
        b"print" => Some(TokenType::Print),
        b"read" => Some(TokenType::Read),
        b"return" => Some(TokenType::Return),
        b"show" => Some(TokenType::Show),
        b"struct" => Some(TokenType::Struct),
        b"sum" => Some(TokenType::Sum),
        b"then" => Some(TokenType::Then),
        b"time" => Some(TokenType::Time),
        b"to" => Some(TokenType::To),
        b"true" => Some(TokenType::True),
        b"void" => Some(TokenType::Void),
        b"write" => Some(TokenType::Write),
        _ => None,
    }
}

/// Converts JPL source code into tokens.
/// This struct implements the interator trait which outputs Tokens
/// If a lex occurs then an error message will be printed and the process will exit
#[derive(Debug, Clone, Copy)]
pub struct Lexer<'a> {
    source_name: &'a str,
    bytes: &'a [u8],
    cur: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source_name: &'a str, source: &'a str) -> Self {
        Lexer {
            source_name,
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
        token
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
                    let kind = bytes_to_keyword(var).unwrap_or(TokenType::Variable);

                    return Some(self.create_token(kind, var.len()));
                }

                // Int and Floats Literals
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
                        let next_newline = self.bytes[self.cur..]
                            .iter()
                            .position(|&b| b == b'\n')
                            .map(|b| b + self.cur)
                            .unwrap_or(self.bytes.len() - 1);

                        exit_with_error(
                            miette!(
                                severity = Severity::Error,
                                labels = vec![
                                    LabeledSpan::at_offset(self.cur, "String literal started here"),
                                    LabeledSpan::at_offset(next_newline, "Add \" here")
                                ],
                                help = "String literals cannot span multiple lines",
                                "String literal is not terminated"
                            )
                            .with_source_code(NamedSource::new(
                                self.source_name,
                                self.bytes.to_vec(),
                            )),
                        )
                    }
                },

                // Invalid characters
                Some(b'\t') => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(self.cur, "Tab character found here")],
                        help = "Tabs are not allowed in jpl. Replace them with spaces.",
                        "Invalid tab character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                Some(b'\r') => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(
                            self.cur,
                            "Carriage return found here"
                        )],
                        help = "Windows-style line endings ('\\r\\n') are not allowed in jpl. Use '\\n' instead.",
                        "Invalid carriage return"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),
                Some(0..=8) | Some(11..=12) | Some(14..=31) | Some(127) => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(self.cur, "Invalid control character found here")],
                        help = "Control characters are not allowed in jpl. Remove them.",
                        "Invalid control character"

                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                Some(128..) => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(self.cur, "Non-ASCII character found here")],
                        help = "Only ASCII characters are allowed in jpl. Remove or replace the invalid character.",
                        "Invalid non-ASCII character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                // These characters are valid in some locations (strings, comments, etc.), however
                // they are never start a token

                // Could be confused with a comment
                Some(b'#') => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(self.cur, "Unexpected '#' found here")],
                        help = "If you're trying to write a comment, use // or /* ... */ instead of '#'.",
                        "Unexpected '#' character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                // Could be confused with double quotes for string literals
                invalid_quotes @ (Some(b'\'') | Some(b'`'))=> {
                    let invalid_quotes = invalid_quotes.unwrap();
                    let matching = self.bytes[self.cur..]
                        .iter()
                        .skip(1)
                        .take_while(|&&b| b != b'\n')
                        .position(|c| c == invalid_quotes)
                        .map(|o| o + 1) ;

                    if let Some(other_quote_offset) = matching {
                        exit_with_error(
                     miette!(
                            severity = Severity::Error,
                            labels = vec![
                                LabeledSpan::at_offset(self.cur, "Opening quote here"),
                                LabeledSpan::at_offset(self.cur + other_quote_offset, "Closing quote here")

                            ],
                            help = "Use \" for string literals instead",
                            "Unexpected {} character", *invalid_quotes as char,
                            )
                            .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                        )
                    }
                     else {
                        exit_with_error(
                     miette!(
                            severity = Severity::Error,
                            labels = vec![LabeledSpan::at_offset(self.cur, "Unexpected '{invalid_quotes}' found here")],
                            help = "Use \" for string literals instead",
                            "Unexpected '{}' character", invalid_quotes,
                            )
                            .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                        )
                    }
                },

                // Jpl does not use semicolons to terminate statements
                Some(b';') => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(
                            self.cur,
                            "Unexpected ';' found here"
                        )],
                        help = "Semicolons are not used to terminate statements.",
                        "Unexpected ';' character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                // Could be trying to use bitwise ops
                Some(b'~') => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(
                            self.cur,
                            "Unexpected '~' found here"
                        )],
                        help = "~ bitwise operator not supported",
                        "Unexpected '~' character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                // Can't think of a better error message
                unexpected_char @ (Some(b'$') | Some(b'?') | Some(b'@') | Some(b'^')
                | Some(b'_')) => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(
                            self.cur,
                            format!(
                                "Unexpected '{}' found here",
                                *unexpected_char.unwrap() as char
                            )
                        )],
                        help = "~ bitwise operator not supported",
                        "Unexpected '{}' character",
                        *unexpected_char.unwrap() as char
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

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

                // If there arent 2 of them together they can not start a valid token
                (b'|', _) => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(
                            self.cur,
                            "Unexpected | operator found here"
                        )],
                        help = "Bitwise or '|' not supported. Use '||' for logical or.",
                        "Unexpected '|' character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                (b'&', _) => exit_with_error(
                    miette!(
                        severity = Severity::Error,
                        labels = vec![LabeledSpan::at_offset(
                            self.cur,
                            "Unexpected & operator found here"
                        )],
                        help = "Bitwise and '&' not supported. Use '&&' for logical and.",
                        "Unexpected '&' character"
                    )
                    .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                ),

                // Skip over escaped newlines
                (b'\\', Some(b'\n')) => {
                    self.cur += 2;
                    continue;
                }
                (b'\\', _) => {
                    exit_with_error(
                        miette!(
                            severity = Severity::Error,
                            labels = vec![LabeledSpan::at_offset(self.cur, "Invalid '\\' found here")],
                            help = "The '\\' character is only valid in strings, comments, or to escape newline characters.",
                            "Invalid '\\' character"
                        )
                        .with_source_code(NamedSource::new(self.source_name, self.bytes.to_vec())),
                    )
                }

                // Single line comments and newlines
                // These must contain a new line to lex
                (b'/', Some(b'/')) | (b'\n', _) => {
                    let c = COMMENT_WHITESPACE_REGEX
                        .captures(&self.bytes[self.cur..])
                        .expect("should match due to match arm");
                    let entire = c.get(0).unwrap();
                    if let Some(last_nl) = c.name("nl") {
                        let t = Token {
                            start: self.cur + last_nl.start(),
                            bytes: str::from_utf8(last_nl.as_bytes())
                                .expect("lexer bytes must come from a string"),
                            kind: TokenType::Newline,
                        };

                        self.cur += entire.len();
                        return Some(t);
                    } else {
                        exit_with_error(
                            miette!(
                                severity = Severity::Error,
                                labels = vec![
                                    LabeledSpan::new(
                                        Some("Comment missing newline".to_string()),
                                        self.cur,
                                        self.bytes.len() - self.cur
                                    ),
                                    LabeledSpan::at_offset(self.bytes.len() - 1, r"Add \n here"),
                                ],
                                "Incomplete single line comment"
                            )
                            .with_source_code(NamedSource::new(
                                self.source_name,
                                self.bytes.to_vec(),
                            )),
                        )
                    };
                }

                // Multi-line comments and space character. These may not contain a newline,
                // however a multi line comment must closed in order to lex
                (b'/', Some(b'*')) | (b' ', _) => {
                    match COMMENT_WHITESPACE_REGEX.captures(&self.bytes[self.cur..]) {
                        Some(c) => {
                            let entire = c.get(0).unwrap();
                            if let Some(last_nl) = c.name("nl") {
                                let t = Token {
                                    start: self.cur + last_nl.start(),
                                    bytes: str::from_utf8(last_nl.as_bytes())
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
                        None => exit_with_error(
                            miette!(
                                severity = Severity::Error,
                                labels = vec![
                                    LabeledSpan::at_offset(self.cur, "Begins here"),
                                    LabeledSpan::at_offset(self.bytes.len() - 1, r"Add */ here"),
                                ],
                                "Untermianted multi-line comment"
                            )
                            .with_source_code(NamedSource::new(
                                self.source_name,
                                self.bytes.to_vec(),
                            )),
                        ),
                    };
                }

                // From here we know the slash can not be a comment so it is ok to assume it is
                // a division operator
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
    FloatLit,
    Fn,
    If,
    Image,
    Int,
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

impl Display for Token<'_> {
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
        let tokens: Vec<TokenType> = Lexer::new("test.jpl", source).map(|t| t.kind).collect();
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
        let tokens: Vec<_> = Lexer::new("test.jpl", source).collect();
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
        let tokens: Vec<_> = Lexer::new("test.jpl", source).collect();
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
