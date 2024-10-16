use miette::{Diagnostic, LabeledSpan};
use peg::{Parse, ParseElem, RuleResult};
use smol_str::ToSmolStr;
use thiserror::Error;
use zeque_ast::{ast, util::Range};
use zeque_lexer::{Token, Tokens};

#[derive(Debug, Error)]
#[error("an error occurred during parsing")]
pub struct ParseError {
    location: Range,
    expected: Vec<&'static str>,
}

impl Diagnostic for ParseError {
    fn help(&self) -> Option<Box<dyn std::fmt::Display + '_>> {
        let (last, expected) = self.expected.split_last()?;
        let mut s = "expected ".to_string();
        if let Some((second_last, expected)) = expected.split_last() {
            s.push_str("one of ");
            for part in expected {
                s.push_str(part);
                s.push_str(", ");
            }
            s.push_str(second_last);
            s.push_str(" or ");
        }
        s.push_str(last);
        s.push_str(".");

        Some(Box::new(s))
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(std::iter::once(LabeledSpan::at(
            self.location,
            "here",
        ))))
    }
}

/// Parse a Zeque file.
pub fn parse(input: &str) -> Result<ast::File, ParseError> {
    let table = TokenTable::new(input);
    parser::file(&table).map_err(|err| ParseError {
        location: table.ranges[err.location],
        expected: err.expected.tokens().collect(),
    })
}

struct TokenTable<'input> {
    // `Token` is 1 byte so we want to store them contiguously to avoid wasting a ridiculous amount
    // of space.
    tokens: Vec<Token>,
    ranges: Vec<Range>,
    input: &'input str,
}

impl<'input> TokenTable<'input> {
    fn new(input: &'input str) -> Self {
        let mut iter = Tokens::new(&input);

        let mut tokens = Vec::with_capacity(1024);
        let mut ranges = Vec::with_capacity(1024);

        while let Some(token) = iter.next() {
            if matches!(token, Token::Comment | Token::Newline | Token::Whitespace) {
                continue;
            }

            let range = Range::from_usize_range(iter.lexer().span());
            tokens.push(token);
            ranges.push(range);
        }

        TokenTable {
            tokens,
            ranges,
            input,
        }
    }
}

#[derive(Copy, Clone)]
struct TokenAndRange<'input> {
    token: Token,
    text: &'input str,
    range: Range,
}

impl Parse for TokenTable<'_> {
    type PositionRepr = usize;

    fn start(&self) -> usize {
        0
    }

    fn is_eof(&self, p: usize) -> bool {
        p >= self.ranges.len()
    }

    fn position_repr(&self, p: usize) -> Self::PositionRepr {
        p
    }
}

impl<'input> ParseElem<'input> for TokenTable<'input> {
    type Element = TokenAndRange<'input>;

    fn parse_elem(&'input self, pos: usize) -> RuleResult<Self::Element> {
        match self.tokens.get(pos) {
            Some(&token) => {
                let range = self.ranges[pos];
                let text = &self.input[range.start as usize..range.end as usize];
                RuleResult::Matched(pos + 1, TokenAndRange { token, text, range })
            }
            None => RuleResult::Failed,
        }
    }
}

peg::parser! {
    grammar parser<'a>() for TokenTable<'a> {
        rule m(token: Token) -> TokenAndRange<'input> = quiet!{[t if t.token == token]}

        rule ident() -> TokenAndRange<'input> = m(Token::Ident) / expected!("an identifier")
        rule str() -> TokenAndRange<'input> = m(Token::Str) / expected!("a string")
        rule raw_str() -> TokenAndRange<'input> = m(Token::RawStr) / expected!("a raw string")
        rule lbrace() -> TokenAndRange<'input> = m(Token::LBrace) / expected!("`{`")
        rule rbrace() -> TokenAndRange<'input> = m(Token::RBrace) / expected!("`}`")
        rule lparen() -> TokenAndRange<'input> = m(Token::LParen) / expected!("`(`")
        rule rparen() -> TokenAndRange<'input> = m(Token::RParen) / expected!("`)`")
        rule dot() -> TokenAndRange<'input> = m(Token::Dot) / expected!("`.`")
        rule comma() -> TokenAndRange<'input> = m(Token::Comma) / expected!("`,`")
        rule colon() -> TokenAndRange<'input> = m(Token::Colon) / expected!("`:`")
        rule semi() -> TokenAndRange<'input> = m(Token::Semi) / expected!("`;`")
        rule plus() -> TokenAndRange<'input> = m(Token::Plus) / expected!("`+`")
        rule dash() -> TokenAndRange<'input> = m(Token::Dash) / expected!("`-`")
        rule star() -> TokenAndRange<'input> = m(Token::Star) / expected!("`*`")
        rule slash() -> TokenAndRange<'input> = m(Token::Slash) / expected!("`/`")
        rule eq() -> TokenAndRange<'input> = m(Token::Eq) / expected!("`=`")
        rule eqeq() -> TokenAndRange<'input> = m(Token::EqEq) / expected!("`==`")
        rule rarrow() -> TokenAndRange<'input> = m(Token::RArrow) / expected!("`->`")
        rule amp() -> TokenAndRange<'input> = m(Token::Amp) / expected!("`@`")

        rule commas<T>(x: rule<T>) -> Vec<T>
            = item:x() ** comma() comma()? { item }

        rule name() -> ast::Name
            = token:ident() {
                // todo: validate the name
                ast::Name {
                    text: token.text.to_smolstr(),
                    range: token.range,
                }
            }
            / amp:amp() token:str() {
                // todo: check that the string was ended properly
                ast::Name {
                    text: token.text.to_smolstr(),
                    range: amp.range.to(token.range),
                }
            }

        rule int() -> ast::Int
            = token:m(Token::Int) {
                ast::Int {
                    value: token.text.parse().unwrap(),
                    range: token.range,
                }
            }

        rule boolean() -> ast::Bool
            = token:m(Token::True) {
                ast::Bool {
                    value: true,
                    range: token.range,
                }
            }
            / token:m(Token::False) {
                ast::Bool {
                    value: false,
                    range: token.range,
                }
            }

        rule string() -> ast::Str
            = s:str() {
                // todo: check that it's terminated
                ast::Str::Normal {
                    string: s.text.to_smolstr(),
                    range: s.range,
                }
            }
            / parts:raw_str_part()+ {
                ast::Str::Raw(parts)
            }

        rule raw_str_part() -> ast::RawStrPart
            = raw_str:raw_str() {
                ast::RawStrPart {
                    string: raw_str
                        .text
                        .strip_prefix("\\\\")
                        .expect("raw str always starts with \\\\")
                        .to_smolstr(),
                    range: raw_str.range,
                }
            }

        rule pub_() -> ast::Pub
            = token:m(Token::Pub) {
                ast::Pub { range: token.range }
            }

        rule comptime() -> ast::Comptime
            = token:m(Token::Comptime) {
                ast::Comptime { range: token.range }
            }

        rule constructor_field() -> ast::ConstructorField
            = name:name() colon() expr:expr() {
                ast::ConstructorField {
                    name,
                    expr,
                }
            }

        rule call_args() -> Vec<ast::Expr>
            = lparen() arguments:commas(<expr()>) rparen() {
                arguments
            }


        pub rule file() -> ast::File
            = decls:decl()* {
                ast::File { decls }
            }

        rule decl() -> ast::Decl
            = fn_decl:fn_decl() { ast::Decl::Fn(fn_decl) }
            / field_decl:field_decl() { ast::Decl::Field(field_decl) }
            / const_decl:const_decl() { ast::Decl::Const(const_decl) }


        rule field_decl() -> ast::FieldDecl
            = name:name() colon() ty:expr() comma() {
                ast::FieldDecl { name, ty }
            }

        rule const_decl() -> ast::ConstDecl
            = m(Token::Const) name:name() ty:type_ascription()? eq() value:expr() semi() {
                ast::ConstDecl { name, ty, value }
            }

        rule fn_decl() -> ast::FnDecl
            = pub_:pub_()? m(Token::Fn) name:name() lparen() params:commas(<param()>) rparen() return_ty:return_ty()? body:block() {
                ast::FnDecl {
                    pub_,
                    name,
                    params,
                    return_ty,
                    body,
                }
            }

        rule return_ty() -> ast::Expr
            = rarrow() return_ty:expr() {
                return_ty
            }

        rule param() -> ast::Param
            = comptime:comptime()? name:name() colon() ty:expr() {
                ast::Param { comptime, name, ty }
            }

        rule block() -> ast::Block
            = begin:lbrace() stmts:stmt()* returns:expr()? end:rbrace() {
                ast::Block {
                    stmts,
                    returns: returns.map(Box::new),
                    range: begin.range.to(end.range),
                }
            }

        rule stmt() -> ast::Stmt
            = let_:let_() {
                ast::Stmt::Let(let_)
            }

        rule let_() -> ast::Let
            = m(Token::Let) name:name() ty:type_ascription()? eq() expr:expr() semi() {
                ast::Let { name, ty, expr }
            }

        rule type_ascription() -> ast::Expr
            = colon() ty:expr() {
                ty
            }

        rule struct_() -> ast::Struct
            = begin:m(Token::Struct) lbrace() decls:decl()* end:rbrace() {
                ast::Struct {
                    decls,
                    range: begin.range.to(end.range),
                }
            }

        rule constructor_block() -> Vec<ast::ConstructorField>
            = lbrace() fields:commas(<constructor_field()>) rbrace() {
                fields
            }

        rule expr() -> ast::Expr = precedence! {
            lhs:(@) eqeq() rhs:@ {
                ast::Expr::BinOp(Box::new(ast::BinOp { kind: ast::BinOpKind::Eq, lhs, rhs }))
            }
            --
            lhs:(@) plus() rhs:@ {
                ast::Expr::BinOp(Box::new(ast::BinOp { kind: ast::BinOpKind::Add, lhs, rhs }))
            }

            lhs:(@) dash() rhs:@ {
                ast::Expr::BinOp(Box::new(ast::BinOp { kind: ast::BinOpKind::Sub, lhs, rhs }))
            }
            --
            lhs:(@) star() rhs:@ {
                ast::Expr::BinOp(Box::new(ast::BinOp { kind: ast::BinOpKind::Mul, lhs, rhs }))
            }
            --

            begin:m(Token::If) cond:expr() then:block() m(Token::Else) else_:block() {
                ast::Expr::IfThenElse(Box::new(ast::IfThenElse {
                    range: begin.range.to(else_.range),
                    cond,
                    then,
                    else_,
                }))
            }

            callee:(@) args:call_args() {
                ast::Expr::Call {
                    callee: ast::Callee::Expr(Box::new(callee)),
                    args,
                }
            }

            amp:amp() name:ident() args:call_args() {
                ast::Expr::Call {
                    callee: ast::Callee::Builtin {
                        name: name.text.to_smolstr(),
                        range: amp.range.to(name.range),
                    },
                    args,
                }
            }

            comptime:comptime() expr:(@) {
                ast::Expr::Comptime { comptime, expr: Box::new(expr) }
            }

            int:int() {
                ast::Expr::Int(int)
            }

            boolean:boolean() {
                ast::Expr::Bool(boolean)
            }

            string:string() {
                ast::Expr::Str(string)
            }

            struct_:struct_() {
                ast::Expr::Struct(struct_)
            }

            // "_" _ block:span(<constructor_block()>) { Expr::AnonymousConstructor(block) }

            ty:(@) block:constructor_block() {
                ast::Expr::Constructor {
                    ty: Some(Box::new(ty)),
                    fields: block,
                }
            }

            name:name() {
                ast::Expr::Name(name)
            }

            expr:(@) dot() field_name:name() {
                ast::Expr::FieldAccess {
                    expr: Box::new(expr),
                    field_name,
                }
            }

            block:block() {
                ast::Expr::Block(block)
            }

            m(Token::Fn) lparen() rparen() {
                ast::Expr::FnType
            }
        }


    }
}
