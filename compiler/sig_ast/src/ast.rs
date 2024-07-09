//! The abstract syntax tree representation.
//!
//! All span information is stored within the AST. Spans of expressions are inferred from their
//! components, e.g. the span of "1 + 2" is inferred as the start of "1" until the end of "2".

use crate::util::{Range, Span};
use smol_str::SmolStr;

#[derive(Clone, Debug)]
pub enum Item {
    Fn(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: Span<SmolStr>,
    pub params: Vec<Span<Param>>,
    pub return_type: Expr,
    pub body: Span<Block>,
}

#[derive(Copy, Clone, Debug)]
pub struct Comptime;

#[derive(Clone, Debug)]
pub struct Param {
    pub is_comptime: Option<Span<Comptime>>,
    pub name: Span<SmolStr>,
    pub ty: Span<Expr>,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: Span<SmolStr>,
    pub ty: Option<Span<Expr>>,
    pub expr: Span<Expr>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Span<Stmt>>,
    pub returns: Expr,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(Let),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(Span<i32>),
    Bool(Span<bool>),
    BinOp(BinOp, Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    Name(Span<SmolStr>),
    Block(Box<Span<Block>>),
    Call(Callee, Span<Vec<Expr>>),
    Comptime(Box<Self>),
    Struct(Span<Struct>),
    Constructor(Box<Self>, Span<Vec<StructField>>),
    // AnonymousConstructor(Span<Vec<StructField>>),
    Field(Box<Self>, Span<SmolStr>),
    Error(Range),
}

#[derive(Clone, Debug)]
pub enum Callee {
    Expr(Box<Expr>),
    Builtin(Span<SmolStr>),
}

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
    pub name: Span<SmolStr>,
    pub value: Expr,
}
