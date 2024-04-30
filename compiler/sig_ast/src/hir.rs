//! The high-level intermediate representation.
//!
//! This IR represents a name-resolved version of a Sig program.
//! This means that we now know what every identifier refers to.
//!
//! It also flattens the tree structure into vectors stored in the [`FunctionContext`].
//! For displaying the HIR as a tree structure, see the [`printer`] module.

pub use crate::ast::BinOp;
use crate::util::Ix;
use std::fmt;

pub mod printer;

#[derive(Clone, Debug)]
pub enum Item {
    Fn(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub return_type: Ix<Expr>,
    pub body: Ix<Block>,
    pub context: FunctionContext,
}

#[derive(Clone, Debug)]
pub struct FunctionContext {
    pub params: Vec<Parameter>,
    pub lets: Vec<Let>,
    pub exprs: Vec<Expr>,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub is_comptime: bool,
    pub name: String,
    pub ty: Ix<Expr>,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: String,
    pub ty: Option<Ix<Expr>>,
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
    Call(Ix<Self>, Vec<Ix<Self>>),
    Comptime(Ix<Self>),
}

#[derive(Copy, Clone, Debug)]
pub enum Name {
    Let(Ix<Let>),
    Parameter(Ix<Parameter>),
    Function(Ix<Function>),
    Builtin(Builtin),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    I32,
    Bool,
    Type,
}

impl Builtin {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::Bool => "bool",
            Self::Type => "type",
        }
    }
}

impl std::str::FromStr for Builtin {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i32" => Ok(Self::I32),
            "bool" => Ok(Self::Bool),
            "type" => Ok(Self::Type),
            _ => Err(()),
        }
    }
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}
