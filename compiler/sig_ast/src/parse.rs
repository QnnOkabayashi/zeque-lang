use crate::ast::{
    AssociatedLet, BinOp, Block, Callee, Comptime, Expr, Function, Let, Param, Stmt, StringKind,
    Struct, StructField, StructFieldDef, StructItem, Vis,
};
use crate::util::Span;
use smol_str::{SmolStr, ToSmolStr};

pub use parse::struct_body;

peg::parser! {
  grammar parse() for str {
    rule comma<T>(inner: rule<T>) -> Vec<T>
        = item:inner() ** ("," _) _ ","? _ { item }

    rule span<T>(inner: rule<T>) -> Span<T>
        = start:position!() inner:inner() end:position!() { Span::new(start, end, inner) }

    rule name() -> SmolStr
        = quiet!{ a:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '_' | '0'..='9']*) _ { a.to_smolstr() } }
        / expected!("identifier")

    rule _ = quiet!{precedence! {
        // Comments
        (@) "//" [n if n != '\n']* @ {}
        --
        // Other whitespace
        ['\t'|'\n'|'\x0C'|'\r'|' ']* {}
    }}
    / expected!("whitespace")

    rule int() -> i32
        = n:$(['0'..='9']+) _ { n.parse().unwrap() }

    rule boolean() -> bool
        = "true" _ { true }
        / "false" _ { false }

    rule pub_() -> ()
        = "pub" _ {}

    rule vis() -> Vis
        = public:span(<pub_()>)? {
            match public {
                Some(Span((), range)) => Vis::Public(range),
                None => Vis::Private,
            }
        }

    pub rule struct_body() -> Vec<StructItem>
        = _ items:struct_item()* { items }

    rule struct_def() -> Struct
        = "struct" _ "{" _ struct_body:struct_body() "}" _ { Struct { items: struct_body } }

    rule struct_item() -> StructItem
        = function:function() { StructItem::Fn(function) }
        / field:struct_field_def() "," _ { StructItem::Field(field) }
        / let_:associated_let() { StructItem::Let(let_) }

    rule struct_field_def() -> StructFieldDef
        = vis:vis() struct_field:struct_field() { StructFieldDef { vis, struct_field } }

    // used for both struct field defs and constructors,
    // since types are values
    rule struct_field() -> StructField
        = name:span(<name()>) ":" _ value:expr() { StructField { name, value } }

    rule associated_let() -> AssociatedLet
        = vis:vis() let_:let_() { AssociatedLet { vis, let_ } }

    rule let_() -> Let
        = "let" _ name:span(<name()>) ty:let_ascription()? "=" _ expr:expr() ";" _ { Let { name, ty, expr } }

    rule let_ascription() -> Expr
        = ":" _ ty:expr() { ty }

    rule function() -> Function
        = vis:vis() "fn" _ name:span(<name()>) "(" _ params:comma(<span(<parameter()>)>) ")" _ return_type:expr() body:span(<block()>) { Function { vis, name, params, return_type, body } }

    rule parameter() -> Param
        = comptime:span(<comptime()>)? name:span(<name()>) ":" _ ty:span(<expr()>) { Param { comptime, name, ty } }

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
        "@" builtin:span(<name()>) call_args:span(<call_args()>) { Expr::Call(Callee::BuiltinFunction(builtin), call_args) }
        comptime() inner:(@)  { Expr::Comptime(Box::new(inner)) }
        int:span(<int()>) { Expr::Int(int) }
        boolean:span(<boolean()>) { Expr::Bool(boolean) }
        struct_:span(<struct_def()>) { Expr::Struct(struct_) }
        // "_" _ block:span(<constructor_block()>) { Expr::AnonymousConstructor(block) }
        name:(@) block:span(<constructor_block()>) { Expr::Constructor(Box::new(name), block) }
        name:span(<name()>) { Expr::Name(name) }
        value:(@) "." _ field:span(<name()>) { Expr::Field(Box::new(value), field) }
        block:span(<block()>) { Expr::Block(Box::new(block)) }
    }

    rule call_args() -> Vec<Expr>
        = "(" _ arguments:comma(<expr()>) ")" _ { arguments }

    rule constructor_block() -> Vec<StructField>
        = "{" _ fields:comma(<struct_field()>) "}" _ { fields }
  }
}
