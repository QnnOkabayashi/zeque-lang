//! Low-level Zeque lexer.
//!
//! The idea with `zeque_lexer` is to make a reusable library,
//! by separating our pure lexing and Zeque-specific concerns.
//!
//! Tokens produces by this lexer are not yet ready for parsing the Zeque syntax.

use logos::{Lexer, Logos};

fn on_newline(lexer: &mut Lexer<'_, Token>) {
    lexer.extras.current_line.line += 1;
    lexer.extras.current_line.line_offset = lexer.span().end;
}

/// Moves the lexer to the end of the string or the end of the line, whichever comes first.
fn lex_string(lexer: &mut Lexer<'_, Token>) {
    let s = lexer.remainder();

    let mut escaped = false;
    for (offset, byte) in s.bytes().enumerate() {
        if byte == b'\n' {
            lexer.bump(offset);
            return;
        }

        if escaped {
            escaped = false;
            continue;
        }

        if byte == b'"' {
            lexer.bump(offset + 1);
            return;
        }

        if byte == b'\\' {
            escaped = true;
        }
    }

    lexer.bump(s.len());
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
pub enum Token {
    /// Integers, or anything that might look like one.
    #[regex("[0-9][a-zA-Z0-9_]*", priority = 3)]
    Int,
    /// Identifiers.
    #[regex(r"\w+")]
    Ident,
    /// Strings like `"hello world"`.
    #[token("\"", lex_string)]
    Str,
    /// Fully escaped raw string.
    #[regex(r"\\\\[^\n]*")]
    RawStr,

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
    #[token("->")]
    RArrow,
    #[token("@")]
    Amp,

    // Keywords
    #[token("pub")]
    Pub,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("let")]
    Let,
    #[token("const")]
    Const,
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
    #[token("\n", on_newline)]
    Newline,
    #[regex(r"[ \t\r\f]+")]
    Whitespace,

    Error,
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
}

impl<'source> Iterator for Tokens<'source> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.lexer
            .next()
            .map(|result| result.unwrap_or(Token::Error))
    }
}
