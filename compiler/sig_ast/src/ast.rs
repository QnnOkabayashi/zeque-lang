//! The abstract syntax tree representation.

use string_interner::DefaultSymbol;

#[derive(Clone, Debug)]
pub enum Item {
    Fn(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: DefaultSymbol,
    pub params: Vec<Parameter>,
    pub return_type: Expr,
    pub body: Block,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub is_comptime: bool,
    pub name: DefaultSymbol,
    pub ty: Expr,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: DefaultSymbol,
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
    Name(DefaultSymbol),
    Block(Box<Block>),
    Call(Box<Self>, Vec<Self>),
    Comptime(Box<Self>),
    Struct(Struct),
    Constructor(ConstructorType, Vec<StructField>),
    Field(Box<Self>, DefaultSymbol),
}

type ConstructorType = Option<Box<Expr>>;

#[derive(Copy, Clone, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Eq,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub fields: Vec<StructItem>,
}

#[derive(Clone, Debug)]
pub enum StructItem {
    Field(StructField),
}

#[derive(Clone, Debug)]
pub struct StructField {
    pub name: DefaultSymbol,
    pub value: Expr,
}
