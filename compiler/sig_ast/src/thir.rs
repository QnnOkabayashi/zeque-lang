//! The typed high-level intermediate representation.
//!
//! This IR represents a comptime-evaluated version of a Sig program.
//! This means that all comptime evaluation is complete, and all that's left is
//! a representation of what the program will do at runtime. Types are no longer values,
//! and each expression is now typed. The type of an expression is the type located in
//! the `types` field of [`FunctionContext`] at the same index as the expression, e.g.
//! `function_context.exprs.len() == function_context.types.len()` and they line up.
//!
//! For displaying the THIR as a tree structure, see the [`printer`] module.

pub use crate::ast::BinOp;
use crate::util::Ix;
use std::fmt;

pub mod printer;
pub mod typeck;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    I32,
    Bool,
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Builtin::I32 => fmt::Display::fmt("i32", f),
            Builtin::Bool => fmt::Display::fmt("bool", f),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Builtin(Builtin),
    Function(Ix<Function>),
}

// This is after monomorphization.
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub return_type: Type,
    pub body: Ix<Block>,
    pub context: FunctionContext,
}

#[derive(Clone, Debug, Default)]
pub struct FunctionContext {
    pub params: Vec<Param>,
    pub lets: Vec<Let>,
    pub blocks: Vec<Block>,
    // invariant: exprs.len() == types.len()
    pub exprs: Vec<Expr>,
    pub types: Vec<Type>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: String,
    pub ty: Option<Type>,
    pub expr: Ix<Expr>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub returns: Ix<Expr>,
}

#[derive(Copy, Clone, Debug)]
pub enum Stmt {
    Let(Ix<Let>),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i32),
    Bool(bool),
    BinOp(BinOp, Ix<Self>, Ix<Self>),
    IfThenElse(Ix<Self>, Ix<Self>, Ix<Self>),
    Name(Name),
    Block(Ix<Block>),
    DirectCall(Ix<Function>, Vec<Ix<Expr>>),
    IndirectCall(Ix<Expr>, Vec<Ix<Expr>>),
}

#[derive(Copy, Clone, Debug)]
pub enum Name {
    Let(Ix<Let>),
    Parameter(Ix<Param>),
    Function(Ix<Function>),
}

impl Name {
    pub fn as_str(self, ctx: &FunctionContext) -> &str {
        match self {
            Self::Let(index) => &ctx.lets[..][index].name,
            Self::Parameter(index) => &ctx.params[..][index].name,
            Self::Function(_) => todo!(),
        }
    }
}
