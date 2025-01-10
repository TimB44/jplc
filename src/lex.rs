use core::{fmt::Display, str};

/// Converts JPL source code into tokens.
/// This struct implements the interator trait which outputs Tokens
/// If a lex occurs then an error message will be printed and the process will exit
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
                .expect("Lexer bytes must come from a string"),
            kind,
        };
        self.cur += len;
        return token;
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.bytes.get(self.cur) {
            None => {
                // If this is the first time we are beyond the bytes then reutrn EOF, otherwise
                // return None
                return if self.cur == self.bytes.len() {
                    self.cur += 1;
                    Some(Token {
                        start: self.cur,
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

            _ => (),
        };

        // More complex single and double character tokens
        match (self.bytes[self.cur], self.bytes.get(self.cur)) {
            (b'=', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
            (b'=', _) => return Some(self.create_token(TokenType::Op, 1)),
            (b'>', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
            (b'>', _) => return Some(self.create_token(TokenType::Op, 1)),
            (b'<', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
            (b'<', _) => return Some(self.create_token(TokenType::Op, 1)),
            (b'!', Some(b'=')) => return Some(self.create_token(TokenType::Op, 2)),
            (b'!', _) => return Some(self.create_token(TokenType::Op, 1)),
            (b'&', Some(b'&')) => return Some(self.create_token(TokenType::Op, 2)),
            (b'|', Some(b'|')) => return Some(self.create_token(TokenType::Op, 2)),
            _ => todo!(),
        };
    }
}

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

impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            TokenType::Array => write!(f, "ARRAY 'array'"),
            TokenType::Assert => write!(f, "ASSERT 'assert'"),
            TokenType::Bool => write!(f, "BOOL 'bool'"),
            TokenType::Colon => write!(f, "COLON ':'"),
            TokenType::Comma => write!(f, "COMMA ','"),
            TokenType::Else => write!(f, "ELSE 'else'"),
            TokenType::Eof => write!(f, "END_OF_FILE"),
            TokenType::Equals => write!(f, "EQUALS '='"),
            TokenType::False => write!(f, "FALSE 'false'"),
            TokenType::Float => write!(f, "FLOAT 'float'"),

            // TODO: Do i need to check for overflow?
            TokenType::FloatLit => write!(f, "FLOATVAL '{}'", self.bytes),
            TokenType::Fn => write!(f, "FN 'fn'"),
            TokenType::If => write!(f, "IF 'if'"),
            TokenType::Image => write!(f, "IMAGE 'image'"),
            TokenType::Int => write!(f, "INT 'int'"),

            // TODO: should i check for overflow now or later
            TokenType::IntLit => write!(f, "INTVAL '{}'", self.bytes),

            // Weird excape sequence but is correct
            TokenType::LCurly => write!(f, "LCURLY '{{'"),
            TokenType::Let => write!(f, "LET 'let'"),
            TokenType::LParen => write!(f, "LPAREN '('"),
            TokenType::LSquare => write!(f, "LSQUARE '['"),
            TokenType::Newline => write!(f, "NEWLINE"),
            TokenType::Op => write!(f, "OP '{}'", self.bytes),
            TokenType::Print => write!(f, "PRINT 'print'"),

            // Same here
            TokenType::RCurly => write!(f, "RCLURLY '}}'"),
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
mod lexer_tests {}
