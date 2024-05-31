//! The high-level intermediate representation.
//!
//! This IR represents a name-resolved version of a Sig program.
//! This means that we now know what every identifier refers to.
//!
//! It also flattens the tree structure into vectors stored in the [`FunctionContext`].
//! For displaying the HIR as a tree structure, see the [`printer`] module.

use string_interner::DefaultSymbol;

pub use crate::ast::BinOp;
use crate::util::{Indexes, Ix};
use std::fmt;

pub mod printer;

#[derive(Clone, Debug)]
pub enum Item {
    Fn(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: DefaultSymbol,
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
    pub structs: Vec<Struct>,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub is_comptime: bool,
    pub name: DefaultSymbol,
    pub ty: Ix<Expr>,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: DefaultSymbol,
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
    Name(Binding),
    Block(Ix<Block>),
    Call(Ix<Self>, Vec<Ix<Self>>),
    Comptime(Ix<Self>),
    Struct(Ix<Struct>),
    Constructor(ConstructorType, Vec<StructField>),
    Field(Ix<Self>, DefaultSymbol),
}

/// An index into two possible tables: a table of `Binding`s, or a table of `DefaultSymbol`s.
/// This transforms name resolution into simply creating a `Binding`s table to index into, without
/// having to recreate the tree structure.
pub enum Name {}
impl Indexes<Binding> for Name {}
impl Indexes<DefaultSymbol> for Name {}

/// None is anonymous constructor, Some is a given type
type ConstructorType = Option<Ix<Expr>>;

#[derive(Copy, Clone, Debug)]
pub enum Binding {
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
    pub value: Ix<Expr>,
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
