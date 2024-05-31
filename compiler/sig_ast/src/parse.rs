use crate::ast::{
    BinOp, Block, Expr, Function, Item, Let, Parameter, Stmt, Struct, StructField, StructItem,
};
use crate::util::StringInterner;
use std::cell::RefCell;
use string_interner::DefaultSymbol;

pub use parse::program;

type Context = RefCell<StringInterner>;

peg::parser! {
  grammar parse() for str {
    rule comma<T>(item: rule<T>) -> Vec<T>
        = item:item() ** ("," _) _ ","? _ { item }

    rule name(ctx: &Context) -> DefaultSymbol
        = quiet!{ a:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '_' | '0'..='9']*) _ { ctx.borrow_mut().get_or_intern(a) } }
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

    pub rule program(ctx: &Context) -> Vec<Item>
        = _ items:item(ctx)* { items }

    rule item(ctx: &Context) -> Item
        = function:function(ctx) { Item::Fn(function) }

    rule let_(ctx: &Context) -> Let
        = "let" _ name:name(ctx) ty:let_ascription(ctx)? "=" _ expr:expr(ctx) ";" _ { Let { name, ty, expr } }

    rule let_ascription(ctx: &Context) -> Expr
        = ":" _ ty:expr(ctx) { ty }

    rule function(ctx: &Context) -> Function
        = "fn" _ name:name(ctx) "(" _ params:comma(<parameter(ctx)>) ")" _ return_type:expr(ctx) body:block(ctx) { Function { name, params, return_type, body } }

    rule parameter(ctx: &Context) -> Parameter
        = comptime:comptime()? name:name(ctx) ":" _ ty:expr(ctx) { Parameter { is_comptime: comptime.is_some(), name, ty } }

    // this is kinda a hack
    rule comptime() -> ()
        = "comptime" !['a'..='z' | 'A'..='Z' | '_' | '0'..='9'] _ {}

    rule block(ctx: &Context) -> Block
        = "{" _ stmts:stmt(ctx)* returns:expr(ctx) "}" _ { Block { stmts, returns } }

    rule stmt(ctx: &Context) -> Stmt
        = let_:let_(ctx) { Stmt::Let(let_) }

    rule expr(ctx: &Context) -> Expr = precedence! {
        "if" _ "(" _ cond:expr(ctx) ")" _ then:expr(ctx) "else" _ else_:expr(ctx) { Expr::IfThenElse(Box::new(cond), Box::new(then), Box::new(else_)) }
        --
        lhs:(@) "==" _ rhs:@ { Expr::BinOp(BinOp::Eq, Box::new(lhs), Box::new(rhs)) }
        --
        lhs:(@) "+" _ rhs:@ { Expr::BinOp(BinOp::Add, Box::new(lhs), Box::new(rhs)) }
        lhs:(@) "-" _ rhs:@ { Expr::BinOp(BinOp::Sub, Box::new(lhs), Box::new(rhs)) }
        --
        lhs:(@) "*" _ rhs:@ { Expr::BinOp(BinOp::Mul, Box::new(lhs), Box::new(rhs)) }
        --
        callee:(@) "(" _ arguments:comma(<expr(ctx)>) ")" _ { Expr::Call(Box::new(callee), arguments) }
        comptime() inner:(@)  { Expr::Comptime(Box::new(inner)) }
        int:int() { Expr::Int(int) }
        boolean:boolean() { Expr::Bool(boolean) }
        struct_:struct_(ctx) { Expr::Struct(struct_) }
        "_" _ block:constructor_block(ctx) { Expr::Constructor(None, block) }
        name:(@) block:constructor_block(ctx) { Expr::Constructor(Some(Box::new(name)), block) }
        name:name(ctx) { Expr::Name(name) }
        value:(@) "." _ field:name(ctx) { Expr::Field(Box::new(value), field) }
        block:block(ctx) { Expr::Block(Box::new(block)) }
    }

    rule struct_(ctx: &Context) -> Struct
        = "struct" _ "{" _ fields:comma(<struct_item(ctx)>) "}" _ { Struct { fields } }

    rule struct_item(ctx: &Context) -> StructItem
        = field:struct_field(ctx) { StructItem::Field(field) }
        // todo: add more here

    rule struct_field(ctx: &Context) -> StructField
        = name:name(ctx) ":" _ value:expr(ctx) { StructField { name, value } }

    rule constructor_block(ctx: &Context) -> Vec<StructField>
        = "{" _ fields:comma(<struct_field(ctx)>) "}" _ { fields }
  }
}
