//! The abstract syntax tree representation.

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

// pub mod printer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct File {
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Struct {
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Decl {
    Fn(FnDecl),
    Field(FieldDecl),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldDecl {
    pub name: SmolStr,
    pub ty: Expr,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Pub;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FnDecl {
    pub is_public: Option<Pub>,
    pub name: SmolStr,
    pub params: Vec<Param>,
    pub return_ty: Option<Expr>,
    pub body: Block,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Comptime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Param {
    pub is_comptime: Option<Comptime>,
    pub name: SmolStr,
    pub ty: Expr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Let {
    pub name: SmolStr,
    pub ty: Option<Expr>,
    pub expr: Expr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub returns: Option<Box<Expr>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Stmt {
    // Extracted because we want to be able to index them from resolved names
    Let(Let),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    Int(i32),
    Bool(bool),
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    IfThenElse {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Box<Expr>,
    },
    Name(SmolStr),
    Block(Block),
    Call {
        callee: Callee,
        args: Vec<Expr>,
    },
    Comptime(Box<Expr>),
    Struct(Struct),
    Constructor {
        ty: Option<Box<Expr>>,
        fields: Vec<ConstructorField>,
    },
    FieldAccess {
        expr: Box<Expr>,
        field_name: SmolStr,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Callee {
    Expr(Box<Expr>),
    Builtin(SmolStr),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Eq,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstructorField {
    pub name: SmolStr,
    pub expr: Expr,
}
