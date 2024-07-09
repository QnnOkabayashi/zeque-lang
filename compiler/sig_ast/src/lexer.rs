use logos::{Lexer, Logos};
use thiserror::Error;

/// Callback for newlines
fn newline(lexer: &mut Lexer<'_, LSPToken>) -> logos::Skip {
    lexer.extras.line += 1;
    lexer.extras.line_offset = lexer.span().end;
    logos::Skip
}

/// Callback for string literals.
///
/// String literals are not allowed to span multiple lines,
/// and must be terminated by an unescaped doublequote.
fn string(lexer: &mut Lexer<'_, LSPToken>) -> Result<(), NonterminatedStringError> {
    let s = lexer.remainder();

    let mut escaped = false;
    for (offset, byte) in s.bytes().enumerate() {
        if byte == b'\n' {
            lexer.bump(offset);
            return Err(NonterminatedStringError::Newline { escaped });
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

    if let Some(bump) = s.len().checked_sub(1) {
        lexer.bump(bump);
    }

    Err(NonterminatedStringError::Eof { escaped })
}

/// Internal state for [`Lexer<'_, LSPToken>`] to track line and column
/// information.
///
/// Do not use this struct directly. Instead, use [`LSPTokens::line_col`]
/// to get the current line and column.
pub struct CurrentLine {
    line: usize,
    line_offset: usize,
}

/// The most primitive kind of token in Sig.
/// This token should be used by the LSP for syntax highlighting.
#[derive(Copy, Clone, Debug, Logos)]
#[logos(extras = CurrentLine)]
#[logos(error = LSPError)]
pub enum LSPToken {
    // Numbers
    #[regex("[0-9][a-zA-Z0-9_]*", priority = 3)]
    Int,

    // Names
    #[regex(r"\w+")]
    Ident,
    #[token("@\"", string)]
    StringIdent,
    #[regex(r"@\w+")]
    Builtin,

    // Strings
    #[token("\"", string)]
    String,
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
    #[token("==")]
    EqEq,

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
    #[regex(r"[ \t\r\f]+")]
    Whitespace,

    /// Automatically constructed by [`LSPTokens<'_>`].
    Error(LSPError),
}

/// An error representing a failed lex.
#[derive(Copy, Clone, Debug, Default, Error, PartialEq, Eq)]
pub enum LSPError {
    #[error("{0}")]
    NonterminatedString(#[from] NonterminatedStringError),
    #[default]
    #[error("unrecognized token")]
    Unrecognized,
}

/// An error representing a failed string lex.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum NonterminatedStringError {
    #[error("unterminated string, strings cannot span multiple lines")]
    Newline { escaped: bool },
    #[error("string reached EOF before terminating")]
    Eof { escaped: bool },
}

/// The lexer for [`LSPToken`]s, the most primitive kind of token in Sig.
pub struct LSPTokens<'source> {
    lexer: Lexer<'source, LSPToken>,
}

impl<'source> LSPTokens<'source> {
    /// Returns a new [`LSPTokens<'source>`].
    pub fn new(source: &'source str) -> Self {
        LSPTokens {
            lexer: LSPToken::lexer_with_extras(
                source,
                CurrentLine {
                    line: 1,
                    line_offset: 0,
                },
            ),
        }
    }

    /// Returns the line and column of the start of the current token.
    ///
    /// Lines start at 1.
    pub fn line_col(&self) -> (usize, usize) {
        let token_offset = self.lexer.span().start;
        let CurrentLine { line, line_offset } = self.lexer.extras;
        let col = token_offset - line_offset;
        (line, col)
    }
}

impl<'source> std::ops::Deref for LSPTokens<'source> {
    type Target = Lexer<'source, LSPToken>;

    fn deref(&self) -> &Self::Target {
        &self.lexer
    }
}

impl<'source> std::ops::DerefMut for LSPTokens<'source> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lexer
    }
}

impl<'source> Iterator for LSPTokens<'source> {
    type Item = LSPToken;

    fn next(&mut self) -> Option<Self::Item> {
        self.lexer
            .next()
            .map(|result| result.unwrap_or_else(LSPToken::Error))
    }
}
