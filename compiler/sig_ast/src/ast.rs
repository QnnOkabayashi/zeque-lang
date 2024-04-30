//! The abstract syntax tree representation.

#[derive(Clone, Debug)]
pub enum Item {
    Fn(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Expr,
    pub body: Block,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub is_comptime: bool,
    pub name: String,
    pub ty: Expr,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: String,
    pub ty: Option<Expr>,
    pub expr: Expr,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub returns: Expr,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(Let),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i32),
    Bool(bool),
    BinOp(BinOp, Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    Name(String),
    Call(Box<Self>, Vec<Self>),
    Block(Box<Block>),
    Comptime(Box<Self>),
}

#[derive(Copy, Clone, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Eq,
}
