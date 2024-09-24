use crate::ast::{
    BinOp, Block, Callee, Comptime, ConstructorField, Decl, Expr, FieldDecl, File, FnDecl, Let,
    Param, Pub, Stmt, Struct,
};
use smol_str::{SmolStr, ToSmolStr};

pub use zeque_parser::file;

peg::parser! {
  grammar zeque_parser() for str {
    rule comma<T>(x: rule<T>) -> Vec<T>
        = item:x() ** ("," _) _ ","? _ { item }

    // rule span<T>(item: rule<T>) -> Span<T>
    //     = start:position!() inner:item() end:position!() { Span::new(start, end, inner) }

    rule name() -> SmolStr
        = quiet!{
            a:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '_' | '0'..='9']*) _ {? match a {
                "struct" => Err("'struct' isn't a valid identifier"),
                _ => Ok(a.to_smolstr())
            } }
        }
        / expected!("identifier")

    #[cache_left_rec]
    rule _ = quiet!{precedence! {
        // Comments
        (@) "//" [n if n != '\n']* @ {}
        --
        // Other whitespace
        ['\t'|'\n'|'\x0C'|'\r'|' ']* {}
    }}

    rule int() -> i32
        = n:$(['0'..='9']+) _ { n.parse().unwrap() }

    rule boolean() -> bool
        = "true" _ { true }
        / "false" _ { false }

    pub rule file() -> File
        = _ decls:decl()* { File { decls } }

    rule decl() -> Decl
        = fn_decl:fn_decl() { Decl::Fn(fn_decl) }
        / field_decl:field_decl() { Decl::Field(field_decl) }

    rule fn_decl() -> FnDecl
        = pub_:pub_()? "fn" _ name:name() "(" _ params:comma(<param()>) ")" _ return_ty:expr() body:block()
        { FnDecl { is_public: pub_, name, params, return_ty, body } }

    rule param() -> Param
        = is_comptime:comptime()? name:name() ":" _ ty:expr() { Param { is_comptime, name, ty } }

    // this is kinda a hack
    rule comptime() -> Comptime
        = "comptime" !['a'..='z' | 'A'..='Z' | '_' | '0'..='9'] _ { Comptime }

    rule pub_() -> Pub
        = "pub" !['a'..='z' | 'A'..='Z' | '_' | '0'..='9'] _ { Pub }

    rule block() -> Block
        = "{" _ stmts:stmt()* returns:expr() "}" _ { Block { stmts, returns: Box::new(returns) } }

    rule stmt() -> Stmt
        = let_:let_() { Stmt::Let(let_) }

    rule let_() -> Let
        = "let" _ name:name() ty:let_ascription()? "=" _ expr:expr() ";" _
        { Let { name, ty, expr } }

    rule let_ascription() -> Expr
        = ":" _ ty:expr() { ty }

    rule expr() -> Expr = precedence! {
        lhs:(@) "==" _ rhs:@
        { Expr::BinOp { op: BinOp::Eq, lhs: Box::new(lhs), rhs: Box::new(rhs) } }
        --
        lhs:(@) "+" _ rhs:@
        { Expr::BinOp { op: BinOp::Add, lhs: Box::new(lhs), rhs: Box::new(rhs) } }

        lhs:(@) "-" _ rhs:@
        { Expr::BinOp { op: BinOp::Sub, lhs: Box::new(lhs), rhs: Box::new(rhs) } }
        --
        lhs:(@) "*" _ rhs:@
        { Expr::BinOp { op: BinOp::Mul, lhs: Box::new(lhs), rhs: Box::new(rhs) } }
        --
        "if" _ cond:expr() "{" _ then:expr() "}" _ "else" _ "{" _ else_:expr() "}" _
        { Expr::IfThenElse { cond: Box::new(cond), then: Box::new(then), else_: Box::new(else_) } }

        callee:(@) args:call_args()
        { Expr::Call { callee: Callee::Expr(Box::new(callee)), args } }

        "@" name:name() args:call_args()
        { Expr::Call { callee: Callee::Builtin(name), args } }

        comptime() inner:(@)  { Expr::Comptime(Box::new(inner)) }

        int:int() { Expr::Int(int) }

        boolean:boolean() { Expr::Bool(boolean) }

        struct_:struct_() { Expr::Struct(struct_) }

        // "_" _ block:span(<constructor_block()>) { Expr::AnonymousConstructor(block) }

        ty:(@) block:constructor_block()
        { Expr::Constructor { ty: Some(Box::new(ty)), fields: block } }

        name:name() { Expr::Name(name) }

        expr:(@) "." _ field_name:name()
        { Expr::FieldAccess { expr: Box::new(expr), field_name } }

        block:block() _ { Expr::Block(block) }
    }

    rule call_args() -> Vec<Expr>
        = "(" _ arguments:comma(<expr()>) ")" _ { arguments }

    rule struct_() -> Struct
        = "struct" _ "{" _ decls:decl()* "}" _ { Struct { decls } }

    rule field_decl() -> FieldDecl
        = name:name() ":" _ ty:expr() "," _ { FieldDecl { name, ty } }

    rule constructor_block() -> Vec<ConstructorField>
        = "{" _ fields:comma(<constructor_field()>) "}" _ { fields }

    rule constructor_field() -> ConstructorField
        = name:name() ":" _ expr:expr() { ConstructorField { name, expr } }
  }
}

#[derive(PartialEq)]
struct A {
    b: bool,
}
fn x() {
    let a = A { b: true };
    if (a == A { b: true }) {
    } else {
    }
}
