//! The high-level intermediate representation.
//!
//! This IR represents a name-resolved version of a Sig program.
//! This means that we now know what every identifier refers to.
//!
//! It also flattens the tree structure into vectors stored in the [`FunctionContext`].
//! For displaying the HIR as a tree structure, see the [`printer`] module.
//!
//! All span information is stored in FunctionContext (for now).

pub use crate::ast::BinOp;
use crate::util::{Ix, RangeTable, Span};
use std::{fmt, str::FromStr};
use string_interner::DefaultSymbol;

pub mod printer;

#[derive(Clone, Debug)]
pub enum Item {
    Fn(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: Span<DefaultSymbol>,
    pub return_type: Ix<Expr>,
    pub body: Ix<Block>,
    pub context: FunctionContext,
}

#[derive(Clone, Debug, Default)]
pub struct FunctionContext {
    pub params: RangeTable<Param>,
    pub exprs: RangeTable<Expr>,
    pub lets: Vec<Let>,
    pub blocks: Vec<Block>,
    pub structs: Vec<Struct>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub is_comptime: bool,
    pub name: Span<DefaultSymbol>,
    pub ty: Ix<Expr>,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: Span<DefaultSymbol>,
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
    Call(Callee, Vec<Ix<Self>>),
    Comptime(Ix<Self>),
    Struct(Ix<Struct>),
    Constructor(ConstructorType, Vec<StructField>),
    Field(Ix<Self>, Span<DefaultSymbol>),
}

#[derive(Copy, Clone, Debug)]
pub enum Callee {
    Expr(Ix<Expr>),
    Builtin(Span<BuiltinFunction>),
}

/// None is anonymous constructor, Some is a given type
type ConstructorType = Option<Ix<Expr>>;

#[derive(Copy, Clone, Debug)]
pub enum Name {
    Let(Ix<Let>),
    Param(Ix<Param>),
    Function(Ix<Function>),
    Builtin(BuiltinType),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    I32,
    Bool,
    Type,
}

#[derive(Copy, Clone, Debug)]
pub enum BuiltinFunction {
    InComptime,
    Trap,
    Clz,
    Ctz,
}

impl FromStr for BuiltinFunction {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "in_comptime" => Ok(BuiltinFunction::InComptime),
            "trap" => Ok(BuiltinFunction::Trap),
            "clz" => Ok(BuiltinFunction::Clz),
            "ctz" => Ok(BuiltinFunction::Ctz),
            _ => Err(()),
        }
    }
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
    pub name: Span<DefaultSymbol>,
    pub value: Ix<Expr>,
}

impl BuiltinType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::Bool => "bool",
            Self::Type => "type",
        }
    }
}

impl FromStr for BuiltinType {
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

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}
