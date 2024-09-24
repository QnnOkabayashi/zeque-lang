//! Low-level Zeque lexer.
//!
//! The idea with `zeque_lexer` is to make a reusable library,
//! by separating our pure lexing and Zeque-specific concerns.
//!
//! Tokens produces by this lexer are not yet ready for parsing the Zeque syntax.

use logos::{Lexer, Logos};
use thiserror::Error;

fn newline(lexer: &mut Lexer<'_, Token>) {
    lexer.extras.current_line.line += 1;
    lexer.extras.current_line.line_offset = lexer.span().end;
}

fn string(lexer: &mut Lexer<'_, Token>) -> Result<(), Error> {
    let s = lexer.remainder();

    let mut escaped = false;
    for (offset, byte) in s.bytes().enumerate() {
        if byte == b'\n' {
            lexer.bump(offset);
            return Err(Error::NonterminatedString);
        }

        if escaped {
            escaped = false;
            continue;
        }

        if byte == b'"' {
            lexer.bump(offset + 1);
            return Ok(());
        }

        if byte == b'\\' {
            escaped = true;
        }
    }

    lexer.bump(s.len());

    Err(Error::NonterminatedString)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct CurrentLine {
    line: usize,
    line_offset: usize,
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Extras {
    current_line: CurrentLine,
}

impl Extras {
    fn new() -> Self {
        Extras {
            current_line: CurrentLine {
                line: 1,
                line_offset: 0,
            },
        }
    }
}

/// The most primitive kind of token in Zeque.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Logos)]
#[logos(extras = Extras)]
#[logos(error = Error)]
pub enum Token {
    /// Integers, or anything that might look like one.
    #[regex("[0-9][a-zA-Z0-9_]*", priority = 3)]
    Int,
    /// Identifiers.
    #[regex(r"\w+")]
    Ident,
    /// Strings like `"hello world"`.
    #[token("\"", string)]
    String,
    /// Fully escaped raw string.
    #[regex(r"\\\\[^\n]*")]
    RawString,

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
    #[token("@")]
    Amp,

    // Keywords
    #[token("true")]
    True,
    #[token("false")]
    False,
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

    // Other
    #[regex(r"//[^\n]*")]
    Comment,
    #[token("\n", newline)]
    Newline,
    #[regex(r"[ \t\r\f]+")]
    Whitespace,

    // Automatically constructed by [`Tokens<'_>`].
    Error(Error),
}

/// An error representing a failed lex.
#[derive(Copy, Clone, Debug, Default, Error, PartialEq, Eq, Hash)]
pub enum Error {
    /// String that doesn't have a closing double quote.
    #[error("unterminated string")]
    NonterminatedString,

    /// An unrecognized token.
    #[default]
    #[error("unrecognized token")]
    Unrecognized,
}

/// The lexer for [`Token`]s.
#[derive(Clone, Debug)]
pub struct Tokens<'source> {
    lexer: Lexer<'source, Token>,
}

impl<'source> Tokens<'source> {
    /// Returns a new [`Tokens<'_>`].
    pub fn new(source: &'source str) -> Self {
        Tokens {
            lexer: Token::lexer_with_extras(source, Extras::new()),
        }
    }

    /// Returns the line and column of the start of the current token.
    ///
    /// Lines start at 1.
    pub fn line_col(&self) -> (usize, usize) {
        let token_offset = self.lexer.span().start;
        let CurrentLine { line, line_offset } = self.lexer.extras.current_line;
        let col = token_offset - line_offset;
        (line, col)
    }

    pub fn lexer(&self) -> &Lexer<'source, Token> {
        &self.lexer
    }

    pub fn lexer_mut(&mut self) -> &mut Lexer<'source, Token> {
        &mut self.lexer
    }
}

impl<'source> Iterator for Tokens<'source> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.lexer
            .next()
            .map(|result| result.unwrap_or_else(Token::Error))
    }
}
