// use crate::lexer::Tokens;
// use combine::stream::{self, position::DefaultPositioned};
//
// type IteratorStream<'source> = stream::IteratorStream<Tokens<'source>>;
//
// type PositionStream<'source> = stream::position::Stream<
//     IteratorStream<'source>,
//     <IteratorStream<'source> as DefaultPositioned>::Positioner,
// >;
//
// pub type BufferedStream<'source> = stream::buffered::Stream<PositionStream<'source>>;
//
// pub fn make_stream(s: &str) -> BufferedStream<'_> {
//     let lexer = Tokens::new(s);
//     let iterator_stream = IteratorStream::new(lexer);
//     let position_stream = PositionStream::new(iterator_stream);
//     let buffered_stream = BufferedStream::new(position_stream, 10);
//     buffered_stream
// }

use sig_lexer::{Error as LSPError, Token as LSPToken, Tokens as LSPTokens};
use std::ops::Range;
use thiserror::Error;
use unicode_ident::{is_xid_continue, is_xid_start};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum IdentKind {
    Literal,
    String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Int(u128),
    Ident(IdentKind),
    String,
    RawString,
    Builtin,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Dot,
    Comma,
    Colon,
    Semi,
    Plus,
    Dash,
    Star,
    Slash,
    Eq,
    EqEq,
    Bool(bool),
    Let,
    Fn,
    Struct,
    Comptime,
    If,
    Else,
    Errors { range: Range<usize> },
}

/// Used to collect a range of errors without short circuiting.
/// This way, you can do a bunch of fallible operations and then
/// get the indices to all the errors at once at the end, if any.
struct ErrorGroup<'err> {
    errors: &'err mut Vec<Error>,
    initial_error_count: usize,
}

impl<'err> ErrorGroup<'err> {
    fn new(errors: &'err mut Vec<Error>) -> Self {
        ErrorGroup {
            initial_error_count: errors.len(),
            errors,
        }
    }

    fn push_to_group(&mut self, e: impl Into<Error>) {
        self.errors.push(e.into());
    }

    /// Returns the range of errors that were pushed in this error group,
    /// or `None` if no errors were pushed.
    /// ```
    /// # let mut errors = vec![];
    /// let mut error_group = ErrorGroup::new(&mut errors);
    /// // maybe push some errors
    /// if let Some(range) = error_group.finish() {
    ///     Token::Errors { range }
    /// } else {
    ///     Token::Ident
    /// }
    /// ```
    fn finish(self) -> Option<Range<usize>> {
        let err_count_start = self.initial_error_count;
        let err_count_end = self.errors.len();
        (err_count_start < err_count_end).then_some(err_count_start..err_count_end)
    }
}

#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum ParseIntError {
    #[error("overflow")]
    Overflow,
    #[error("bad digit `{digit}` doesn't fit in radix {radix}")]
    Radix { radix: u8, digit: char },
}

#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum UnicodeIdentError {
    #[error("invalid XID start character: '{0}`")]
    XIDStart(char),
    #[error("invalid XID continue character: '{0}`")]
    XIDContinue(char),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("unrecognized")]
    Unrecognized(Range<usize>),
    #[error("nonterminated string")]
    NonterminatedString,
    #[error("{0}")]
    UnicodeIdent(#[from] UnicodeIdentError),
    #[error("{0}")]
    ParseInt(#[from] ParseIntError),
}

pub struct Tokens<'source> {
    lexer: LSPTokens<'source>,
    errors: Vec<Error>,
}

impl<'source> Tokens<'source> {
    pub fn new(source: &'source str) -> Self {
        Tokens {
            lexer: LSPTokens::new(source),
            errors: vec![],
        }
    }

    pub fn span(&self) -> Range<usize> {
        self.lexer.lexer().span()
    }

    pub fn slice(&self) -> &'source str {
        self.lexer.lexer().slice()
    }

    fn parse_int(&mut self) -> Token {
        let s = self.slice();
        if let Some(bin_digits) = s.strip_prefix("0b") {
            self.parse_u128_ignore_underscores::<2>(bin_digits)
        } else if let Some(oct_digits) = s.strip_prefix("0o") {
            self.parse_u128_ignore_underscores::<8>(oct_digits)
        } else if let Some(hex_digits) = s.strip_prefix("0x") {
            self.parse_u128_ignore_underscores::<16>(hex_digits)
        } else {
            self.parse_u128_ignore_underscores::<10>(s)
        }
    }

    // Slightly modified version of the following algorithm:
    // https://github.com/alexhuszagh/rust-lexical/blob/main/lexical-parse-integer/docs/algorithm.md
    fn parse_u128_ignore_underscores<const RADIX: u8>(&mut self, s: &str) -> Token {
        const { assert!(2 <= RADIX && RADIX <= 36, "radix must be between 2 and 36") };

        let mut error_group = ErrorGroup::new(&mut self.errors);

        let mut value: u128 = 0;
        let mut digits = 0;
        for byte in s.bytes() {
            if byte == b'_' {
                continue;
            }

            fn to_digit<const RADIX: u8>(byte: u8) -> Option<u8> {
                // If not a digit, a number greater than radix will be created.
                let mut digit = byte.wrapping_sub(b'0');
                if RADIX > 10 {
                    if digit < 10 {
                        return Some(digit);
                    }
                    // Force the 6th bit to be set to ensure ascii is lower case.
                    digit = (byte | 0b10_0000).wrapping_sub(b'a').saturating_add(10);
                }
                (digit < RADIX).then_some(digit)
            }

            if let Some(digit) = to_digit::<RADIX>(byte) {
                value = value.wrapping_mul(RADIX as u128);
                value = value.wrapping_add(digit as u128);
                digits += 1;
            } else {
                error_group.push_to_group(ParseIntError::Radix {
                    radix: RADIX,
                    digit: byte as char,
                });
            }
        }

        if let Some(range) = error_group.finish() {
            // The value is gibberish anyways since a nonzero number of digits
            // couldn't be parsed.
            return Token::Errors { range };
        }

        // let max_digits = u128::MAX.checked_ilog(u128::from(RADIX)).unwrap() + 1;
        // let min_value_with_max_digits = (RADIX as u128).pow(max_digits - 1);
        let (max_digits, min_value_with_max_digits) = const {
            let max_digits = {
                let base = RADIX as u128;
                // This takes advantage of an optimization only for 128-bit integers
                let mut n = (u128::BITS - 1) / (RADIX.ilog2() + 1);
                let mut r = base.pow(n);

                while r <= u128::MAX / base {
                    n += 1;
                    r *= base;
                }
                n + 1
            };

            let min_value_with_max_digits = (RADIX as u128).pow(max_digits - 1);

            (max_digits, min_value_with_max_digits)
        };

        // check for either
        // 1. too many digits
        // 2. max digits but overflow
        if digits > max_digits || (digits == max_digits && value < min_value_with_max_digits) {
            return self.make_error(ParseIntError::Overflow);
        }

        Token::Int(value)
    }

    fn parse_ident(&mut self) -> Option<Range<usize>> {
        let mut error_group = ErrorGroup::new(&mut self.errors);

        let mut chars = self.lexer.lexer().slice().chars();
        let first = chars.next().expect("at least one char in ident");

        if !is_xid_start(first) && first != '_' {
            error_group.push_to_group(UnicodeIdentError::XIDStart(first));
        }
        for c in chars {
            if !is_xid_continue(c) {
                error_group.push_to_group(UnicodeIdentError::XIDContinue(c));
            }
        }

        error_group.finish()
    }

    /// Convert an [`LSPError`] into a [`Token`].
    fn parse_error(&mut self, lsp_error: LSPError) -> Token {
        let error = match lsp_error {
            LSPError::NonterminatedString => Error::NonterminatedString,
            LSPError::Unrecognized => Error::Unrecognized(self.lexer.lexer().span()),
        };
        self.make_error(error)
    }

    fn make_error(&mut self, error: impl Into<Error>) -> Token {
        let start = self.errors.len();
        self.errors.push(error.into());
        Token::Errors {
            range: start..self.errors.len(),
        }
    }
}

impl Iterator for Tokens<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let token = match self.lexer.next()? {
                LSPToken::Int => self.parse_int(),
                LSPToken::Ident => self
                    .parse_ident()
                    .map(|range| Token::Errors { range })
                    .unwrap_or(Token::Ident(IdentKind::Literal)),
                LSPToken::String => Token::String,
                LSPToken::RawString => Token::RawString,
                LSPToken::LBrace => Token::LBrace,
                LSPToken::RBrace => Token::RBrace,
                LSPToken::LParen => Token::LParen,
                LSPToken::RParen => Token::RParen,
                LSPToken::Dot => Token::Dot,
                LSPToken::Comma => Token::Comma,
                LSPToken::Colon => Token::Colon,
                LSPToken::Semi => Token::Semi,
                LSPToken::Plus => Token::Plus,
                LSPToken::Dash => Token::Dash,
                LSPToken::Star => Token::Star,
                LSPToken::Slash => Token::Slash,
                LSPToken::Eq => Token::Eq,
                LSPToken::True => Token::Bool(true),
                LSPToken::False => Token::Bool(false),
                LSPToken::Let => Token::Let,
                LSPToken::Fn => Token::Fn,
                LSPToken::Struct => Token::Struct,
                LSPToken::Comptime => Token::Comptime,
                LSPToken::If => Token::If,
                LSPToken::Else => Token::Else,
                LSPToken::Comment | LSPToken::Whitespace => continue,
                LSPToken::Error(e) => self.parse_error(e),
                _ => todo!(),
            };

            return Some(token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Token::*;

    fn run(input: &str, expected: &[(Token, &str)]) {
        let actual: Vec<(Token, &str)> = std::iter::from_fn({
            let mut tokens = Tokens::new(input);
            move || tokens.next().map(|tok| (tok, tokens.slice()))
        })
        .collect();
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
                (Int(123), "123"),
                (Ident(IdentKind::Literal), "hello"),
                (RawString, r#"\\😂"#),
                (RawString, r#"\\name: "Quinn""#),
                (RawString, r#"\\name: "Quinn" // not escaped!"#),
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
                (Ident(IdentKind::String), r#"@"hello, world""#),
                (String, r#""string a""#),
                (String, r#""string b""#),
                (
                    Errors {
                        range: Range { start: 0, end: 1 },
                    },
                    r#""not terminated"#,
                ),
                (String, r#""another one""#),
            ],
        );
    }

    #[test]
    fn string_with_escaped_newlines() {
        run(
            r#"
                "A\nB\nC"
            "#,
            &[(String, r#""A\nB\nC""#)],
        );
    }

    #[test]
    fn string_in_sig_string_in_rust_string() {
        run(
            r#"
                "\"string in a Sig string in a Rust string\""
            "#,
            &[(String, r#""\"string in a Sig string in a Rust string\"""#)],
        );
    }

    #[test]
    fn string_escaped_backslash() {
        run(
            r#"
                "\\"
            "#,
            &[(String, r#""\\""#)],
        );
    }

    #[test]
    fn string_unescaped_backslash_skips_endquote_but_is_ok() {
        run(
            r#"
                "\"
            "#,
            &[(
                Errors {
                    range: Range { start: 0, end: 1 },
                },
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
            &[(String, r#""\ ""#)],
        );
    }

    #[test]
    fn ident_starting_with_digit() {
        run(
            r#"
                123hello
            "#,
            &[(
                Errors {
                    range: Range { start: 0, end: 5 },
                },
                "123hello",
            )],
        )
    }

    #[test]
    fn binary_literal() {
        run(
            "
                0b1111_0000
            ",
            &[(Int(0b1111_0000), "0b1111_0000")],
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
                (Int(0o7654_3210), "0o7654_3210"),
                (
                    Errors {
                        range: Range { start: 0, end: 1 },
                    },
                    "0o8",
                ),
                (
                    Errors {
                        range: Range { start: 1, end: 2 },
                    },
                    "0o777777777777777777777777777777777777777777777777777777777777",
                ),
            ],
        )
    }
}
