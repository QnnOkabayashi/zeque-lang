use crate::ast::{
    BinOp, Block, Callee, Comptime, Expr, Function, Item, Let, Param, Stmt, Struct, StructField,
    StructItem,
};
use crate::util::Span;
use smol_str::{SmolStr, ToSmolStr};

pub use parse::program;

peg::parser! {
  grammar parse() for str {
    rule comma<T>(item: rule<T>) -> Vec<T>
        = item:item() ** ("," _) _ ","? _ { item }

    rule span<T>(item: rule<T>) -> Span<T>
        = start:position!() inner:item() end:position!() { Span::new(start, end, inner) }

    rule name() -> SmolStr
        = quiet!{ a:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '_' | '0'..='9']*) _ { a.to_smolstr() } }
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

    pub rule program() -> Vec<Item>
        = _ items:item()* { items }

    rule item() -> Item
        = function:function() { Item::Fn(function) }

    rule let_() -> Let
        = "let" _ name:span(<name()>) ty:let_ascription()? "=" _ expr:span(<expr()>) ";" _ { Let { name, ty, expr } }

    rule let_ascription() -> Span<Expr>
        = ":" _ ty:span(<expr()>) { ty }

    rule function() -> Function
        = "fn" _ name:span(<name()>) "(" _ params:comma(<span(<parameter()>)>) ")" _ return_type:expr() body:span(<block()>) { Function { name, params, return_type, body } }

    rule parameter() -> Param
        = is_comptime:span(<comptime()>)? name:span(<name()>) ":" _ ty:span(<expr()>) { Param { is_comptime, name, ty } }

    // this is kinda a hack
    rule comptime() -> Comptime
        = "comptime" !['a'..='z' | 'A'..='Z' | '_' | '0'..='9'] _ { Comptime }

    rule block() -> Block
        = "{" _ stmts:span(<stmt()>)* returns:expr() "}" _ { Block { stmts, returns } }

    rule stmt() -> Stmt
        = let_:let_() { Stmt::Let(let_) }

    rule expr() -> Expr = precedence! {
        "if" _ "(" _ cond:expr() ")" _ then:expr() "else" _ else_:expr() { Expr::IfThenElse(Box::new(cond), Box::new(then), Box::new(else_)) }
        --
        lhs:(@) "==" _ rhs:@ { Expr::BinOp(BinOp::Eq, Box::new(lhs), Box::new(rhs)) }
        --
        lhs:(@) "+" _ rhs:@ { Expr::BinOp(BinOp::Add, Box::new(lhs), Box::new(rhs)) }
        lhs:(@) "-" _ rhs:@ { Expr::BinOp(BinOp::Sub, Box::new(lhs), Box::new(rhs)) }
        --
        lhs:(@) "*" _ rhs:@ { Expr::BinOp(BinOp::Mul, Box::new(lhs), Box::new(rhs)) }
        --
        callee:(@) call_args:span(<call_args()>) { Expr::Call(Callee::Expr(Box::new(callee)), call_args) }
        "@" builtin:span(<name()>) call_args:span(<call_args()>) { Expr::Call(Callee::Builtin(builtin), call_args) }
        comptime() inner:(@)  { Expr::Comptime(Box::new(inner)) }
        int:span(<int()>) { Expr::Int(int) }
        boolean:span(<boolean()>) { Expr::Bool(boolean) }
        struct_:span(<struct_()>) { Expr::Struct(struct_) }
        // "_" _ block:span(<constructor_block()>) { Expr::AnonymousConstructor(block) }
        name:(@) block:span(<constructor_block()>) { Expr::Constructor(Box::new(name), block) }
        name:span(<name()>) { Expr::Name(name) }
        value:(@) "." _ field:span(<name()>) { Expr::Field(Box::new(value), field) }
        block:span(<block()>) { Expr::Block(Box::new(block)) }
    }

    rule call_args() -> Vec<Expr>
        = "(" _ arguments:comma(<expr()>) ")" _ { arguments }

    rule struct_() -> Struct
        = "struct" _ "{" _ fields:comma(<struct_item()>) "}" _ { Struct { fields } }

    rule struct_item() -> StructItem
        = field:struct_field() { StructItem::Field(field) }
        // todo: add more here

    rule struct_field() -> StructField
        = name:span(<name()>) ":" _ value:expr() { StructField { name, value } }

    rule constructor_block() -> Vec<StructField>
        = "{" _ fields:comma(<struct_field()>) "}" _ { fields }
  }
}
