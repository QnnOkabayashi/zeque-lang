use logos::{Lexer, Logos};
use thiserror::Error;
use unicode_ident::{is_xid_continue, is_xid_start};

/// Called when '\n' is encountered.
fn newline(lexer: &mut Lexer<'_, Token>) -> logos::Skip {
    lexer.extras.line += 1;
    lexer.extras.line_offset = lexer.span().end;
    logos::Skip
}

fn ident(lexer: &mut Lexer<'_, Token>) -> Result<(), Error> {
    let s = lexer.slice();
    let mut chars = s.chars();
    let start = chars.next().expect("at least 1 because of `\\w+`");

    if !is_xid_start(start) && start != '_' {
        Err(Error::XIDStart(start))
    } else if let Some(not_continue) = chars.find(|&ch| !is_xid_continue(ch)) {
        Err(Error::XIDContinue(not_continue))
    } else {
        Ok(())
    }
}

/// Lex an escaped string literal.
fn string(lexer: &mut Lexer<'_, Token>) -> Result<(), ParseStringError> {
    let mut char_indices = lexer.remainder().char_indices();
    loop {
        let (offset, c) = char_indices.next().ok_or(ParseStringError::EOF)?;
        match c {
            '"' => {
                // bump just past the endquote character
                lexer.bump(offset + 1);
                return Ok(());
            }
            '\n' => {
                // bump to the '\n' character
                lexer.bump(offset);
                return Err(ParseStringError::NonTerminatedString);
            }
            '\\' => {
                // skip the next character
                let _ = char_indices.next().ok_or(ParseStringError::EOF)?;
            }
            _ => {}
        }
    }
}

fn int<const RADIX: u8>(lexer: &mut Lexer<'_, Token>) -> Result<u128, Error> {
    let prefix_len = if RADIX == 10 { 0 } else { 2 };
    let s = &lexer.slice()[prefix_len..];
    parse_u128_ignore_underscores::<RADIX>(s).map_err(Error::ParseInt)
}

fn parse_u128_ignore_underscores<const RADIX: u8>(s: &str) -> Result<u128, ParseIntError> {
    // algorithm: https://github.com/alexhuszagh/rust-lexical/blob/main/lexical-parse-integer/docs/algorithm.md
    let mut value: u128 = 0;
    let mut digits = 0;
    for byte in s.bytes() {
        if byte == b'_' {
            continue;
        }
        let digit = to_digit::<RADIX>(byte).ok_or(ParseIntError::Radix {
            radix: RADIX,
            digit: byte as char,
        })?;
        value = value.wrapping_mul(RADIX as u128);
        value = value.wrapping_add(digit as u128);
        digits += 1;
    }

    let max_digits = u128::MAX.checked_ilog(u128::from(RADIX)).unwrap() + 1;
    let min_value_with_max_digits = u128::from(RADIX).pow(max_digits - 1);

    // check for either
    // 1. too many digits
    // 2. max digits but overflow
    if digits > max_digits || (digits == max_digits && value < min_value_with_max_digits) {
        return Err(ParseIntError::Overflow);
    }

    Ok(value)
}

/// Inline version of [`char::to_digit`]
fn to_digit<const RADIX: u8>(c: u8) -> Option<u8> {
    assert!(2 <= RADIX && RADIX <= 36, "radix must be >= 2 and <= 36");
    // If not a digit, a number greater than radix will be created.
    let mut digit = c.wrapping_sub(b'0');
    if RADIX > 10 {
        if digit < 10 {
            return Some(digit);
        }
        // Force the 6th bit to be set to ensure ascii is lower case.
        digit = (c | 0b10_0000).wrapping_sub(b'a').saturating_add(10);
    }
    (digit < RADIX).then_some(digit)
}

pub struct Extras {
    pub line: usize,
    pub line_offset: usize,
}

impl Default for Extras {
    fn default() -> Self {
        Extras {
            line: 1, // lines start at 1
            line_offset: 0,
        }
    }
}

#[derive(Copy, Clone, Debug, Logos, PartialEq)]
#[logos(extras = Extras)]
#[logos(error = Error)]
pub enum Token {
    #[regex("[0-9][0-9_]*", int::<10>, priority = 3)]
    #[regex("0b[a-zA-Z0-9][a-zA-Z0-9_]*", int::<2>, priority = 4)]
    #[regex("0o[a-zA-Z0-9][a-zA-Z0-9_]*", int::<8>, priority = 4)]
    #[regex("0x[a-zA-Z0-9][a-zA-Z0-9_]*", int::<16>, priority = 4)]
    Int(u128),

    // todo: find a way to not require a verifier
    #[regex(r"\w+", ident)]
    Ident,
    #[token("@\"", string)]
    EscIdent,
    #[token("\"", string)]
    EscString,
    #[regex(r"\\\\[^\n]*")]
    RawString,
    #[regex("@[a-zA-Z_][a-zA-Z0-9_]*")]
    BuiltinCall,

    // Symbols
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token("+")]
    Plus,
    #[token("-")]
    Dash,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("=")]
    Eq,
    #[token("==")]
    EqEq,

    // Keywords
    #[token("true", |_| true)]
    #[token("false", |_| false)]
    Bool(bool),
    #[token("let")]
    Let,
    #[token("fn")]
    Fn,
    #[token("struct")]
    Struct,
    #[token("comptime")]
    Comptime,
    #[token("if")]
    If,
    #[token("else")]
    Else,

    // Skipped
    #[regex(r"//[^\n]*", logos::skip)]
    Comment,
    #[token("\n", newline)]
    #[regex(r"[ \t\r\f]+", logos::skip)]
    Whitespace,
}

#[derive(Copy, Clone, Debug, Error, PartialEq)]
pub enum ParseIntError {
    #[error("overflow")]
    Overflow,
    #[error("bad digit `{digit}` doesn't fit in radix {radix}")]
    Radix { radix: u8, digit: char },
}

#[derive(Copy, Clone, Debug, Error, PartialEq)]
pub enum ParseStringError {
    #[error("non terminated string")]
    NonTerminatedString,
    #[error("EOF")]
    EOF,
}

#[derive(Copy, Clone, Debug, Default, Error, PartialEq)]
pub enum Error {
    #[error("bad XID start: '{0}'")]
    XIDStart(char),
    #[error("bad XID continue: '{0}'")]
    XIDContinue(char),
    #[error("{0}")]
    ParseInt(#[from] ParseIntError),
    #[error("{0}")]
    ParseString(#[from] ParseStringError),
    #[default]
    #[error("unexpected character")]
    UnexpectedCharacter,
}

/// Returns the line and column of the current token.
/// Lines start at 1.
pub fn line_col(lexer: &Lexer<'_, Token>) -> (usize, usize) {
    let token_offset = lexer.span().start;
    let Extras { line, line_offset } = lexer.extras;
    let col = token_offset - line_offset;
    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TokensAndSlices<'source> {
        inner: logos::Lexer<'source, Token>,
    }

    impl<'source> TokensAndSlices<'source> {
        fn new(source: &'source str) -> Self {
            TokensAndSlices {
                inner: Token::lexer(source),
            }
        }
    }

    impl<'source> Iterator for TokensAndSlices<'source> {
        type Item = (Result<Token, Error>, &'source str);

        fn next(&mut self) -> Option<Self::Item> {
            self.inner
                .next()
                .map(|token_result| (token_result, self.inner.slice()))
        }
    }

    fn run(input: &str, expected: &[(Result<Token, Error>, &str)]) {
        let actual = TokensAndSlices::new(input).collect::<Vec<_>>();
        let mut ok = true;
        for (actual, expected) in actual.iter().zip(expected) {
            if actual != expected {
                eprintln!("Expected: {expected:#?}\nActual: {actual:#?}");
                ok = false;
            }
        }

        if actual.len() != expected.len() {
            eprintln!(
                "Expected len: {}\nActual len: {}",
                expected.len(),
                actual.len()
            );
            ok = false;
        }

        if !ok {
            panic!("errors occurred");
        }
    }

    #[test]
    fn test() {
        run(
            r#"
                123 // this is escaped
                hello \\😂
                \\name: "Quinn"
                \\name: "Quinn" // not escaped!
            "#,
            &[
                (Ok(Token::Int(123)), "123"),
                (Ok(Token::Ident), "hello"),
                (Ok(Token::RawString), r#"\\😂"#),
                (Ok(Token::RawString), r#"\\name: "Quinn""#),
                (Ok(Token::RawString), r#"\\name: "Quinn" // not escaped!"#),
            ],
        );
    }

    #[test]
    fn strings() {
        run(
            r#"
                @"hello, world"
                "string a""string b"
                "not terminated
                "another one"
            "#,
            &[
                (Ok(Token::EscIdent), r#"@"hello, world""#),
                (Ok(Token::EscString), r#""string a""#),
                (Ok(Token::EscString), r#""string b""#),
                (
                    Err(Error::ParseString(ParseStringError::NonTerminatedString)),
                    r#""not terminated"#,
                ),
                (Ok(Token::EscString), r#""another one""#),
            ],
        );
    }

    #[test]
    fn string_with_escaped_newlines() {
        run(
            r#"
                "A\nB\nC"
            "#,
            &[(Ok(Token::EscString), r#""A\nB\nC""#)],
        );
    }

    #[test]
    fn string_in_sig_string_in_rust_string() {
        run(
            r#"
                "\"string in a Sig string in a Rust string\""
            "#,
            &[(
                Ok(Token::EscString),
                r#""\"string in a Sig string in a Rust string\"""#,
            )],
        );
    }

    #[test]
    fn string_escaped_backslash() {
        run(
            r#"
                "\\"
            "#,
            &[(Ok(Token::EscString), r#""\\""#)],
        );
    }

    #[test]
    fn string_unescaped_backslash_skips_endquote_err() {
        run(
            r#"
                "\"
            "#,
            &[(
                Err(Error::ParseString(ParseStringError::NonTerminatedString)),
                r#""\""#,
            )],
        );
    }

    #[test]
    fn string_unknown_escape_sequence() {
        // this succeeds now because it successfully parsed a string.
        // However, it's not a valid escape sequence so will fail later
        // when unescaping.
        run(
            r#"
                "\ "
            "#,
            &[(Ok(Token::EscString), r#""\ ""#)],
        );
    }

    #[test]
    fn binary_literal() {
        run(
            "
                0b1111_0000
            ",
            &[(Ok(Token::Int(0b1111_0000)), "0b1111_0000")],
        )
    }

    #[test]
    fn octal_literal() {
        run(
            "
                0o7654_3210
                0o8
                0o777777777777777777777777777777777777777777777777777777777777
            ",
            &[
                (Ok(Token::Int(0o7654_3210)), "0o7654_3210"),
                (
                    Err(Error::ParseInt(ParseIntError::Radix {
                        radix: 8,
                        digit: '8',
                    })),
                    "0o8",
                ),
                (
                    Err(Error::ParseInt(ParseIntError::Overflow)),
                    "0o777777777777777777777777777777777777777777777777777777777777",
                ),
            ],
        )
    }
}
