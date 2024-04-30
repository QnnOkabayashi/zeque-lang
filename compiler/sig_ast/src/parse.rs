use crate::ast::{BinOp, Block, Expr, Function, Item, Let, Parameter, Stmt};

pub use parse::program;

peg::parser! {
  grammar parse() for str {
    rule comma<T>(item: rule<T>) -> Vec<T>
        = item:item() ** ("," _) _ ","? _ { item }

    rule name() -> String
        = quiet!{ a:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '_' | '0'..='9']*) _ { a.to_string() } }
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
        = "let" _ name:name() ty:let_ascription()? "=" _ expr:expr() ";" _ { Let { name, ty, expr } }

    rule let_ascription() -> Expr
        = ":" _ ty:expr() { ty }

    rule function() -> Function
        = "fn" _ name:name() "(" _ params:comma(<parameter()>) ")" _ return_type:expr() body:block() { Function { name, params, return_type, body } }

    rule parameter() -> Parameter
        = comptime:comptime()? name:name() ":" _ ty:expr() { Parameter { is_comptime: comptime.is_some(), name, ty } }

    // this is kinda a hack
    rule comptime() -> ()
        = "comptime" _ {}

    rule block() -> Block
        = "{" _ stmts:stmt()* returns:expr() "}" _ { Block { stmts, returns } }

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
        callee:(@) "(" _ arguments:comma(<expr()>) ")" _ { Expr::Call(Box::new(callee), arguments) }
        "comptime" _ inner:(@)  { Expr::Comptime(Box::new(inner)) }
        int:int() { Expr::Int(int) }
        boolean:boolean() { Expr::Bool(boolean) }
        name:name() { Expr::Name(name) }
        block:block() { Expr::Block(Box::new(block)) }
    }
  }
}
