//! The high-level intermediate representation.
//!
//! This IR represents a name-resolved version of a Sig program.
//! This means that we now know what every identifier refers to.

pub use crate::ast::{BinOp, Comptime, Vis};
use crate::util::{Ix, RangeTable, Span};
use std::{collections::HashMap, fmt, str::FromStr};
use string_interner::DefaultSymbol;

pub mod printer;

#[derive(Clone, Debug, Default)]
pub struct Hir {
    pub source_files: Vec<SourceFile>,
    pub storages: Vec<Storage>,
}

#[derive(Clone, Debug)]
pub struct SourceFile {
    pub storage: Ix<Storage>,
    /// The struct index of the struct that the source file implicitly represents.
    pub struct_index: Ix<Struct>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(u32);

impl Level {
    pub const SOURCE_FILE: Self = Level(0);

    pub fn enter_function(self) -> Self {
        Level(self.0 + 1)
    }
}

// a storage can be cloned when we want to monomorphize
#[derive(Clone, Debug)]
pub struct Storage {
    /// If this is the storage of a function (as opposed to a source file), this points
    /// to the storage that the function lives in.
    /// During monomorphization, it's important that when we clone a Storage,
    /// we update all the necessary backrefs to point to it.
    // When we do AST -> HIR lowering and are tracking which names are visible,
    // we need to also store at what "level" they're at, and when we use them by name,
    // take the difference between the current level and their level to know how many backrefs to
    // follow. This ensures that as we monomorphize, we get the correctly monomorphized versions of
    // the associated lets and functions we are referring to.
    pub level: Level,

    pub structs: Vec<Struct>,

    pub lets: Vec<Let>,
    pub blocks: Vec<Block>,
    pub exprs: RangeTable<Expr>,
}

#[derive(Clone, Debug)]
pub enum DeclKind {
    AssociatedLet(Ix<AssociatedLet>),
    Function(Ix<Function>),
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub struct_field_defs: Vec<StructFieldDef>,
    pub name_to_decl: HashMap<DefaultSymbol, DeclKind>,
    pub is_comptime_only: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct StructFieldDef {
    pub vis: Vis,
    pub struct_field: StructField,
}

#[derive(Copy, Clone, Debug)]
pub struct StructField {
    pub name: Span<DefaultSymbol>,
    pub value: Ix<Expr>,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub vis: Vis,
    pub name: Span<DefaultSymbol>,

    pub params: RangeTable<Param>,

    pub return_type: Ix<Expr>,
    pub body: Ix<Block>,
    // pub storage_index: Ix<Storage>,
    // backref to the context it was defined in, either the storage of the last param or the
    // storage that the enclosing struct lives in if there are no params.
    // pub backref: Ix<Storage>,
}

#[derive(Copy, Clone, Debug)]
pub struct Param {
    pub comptime: Option<Span<Comptime>>,
    pub name: Span<DefaultSymbol>,
    pub ty: Ix<Expr>,
    // backref to the storage it was defined in,
    // either the storage of the previous param or the storage that the enclosing struct was stored
    // in.
    // pub backref: Ix<Storage>,
    // pub non_monomorphized_storage_index: Ix<Storage>,
}

#[derive(Copy, Clone, Debug)]
pub struct Let {
    pub name: Span<DefaultSymbol>,
    pub ty: Option<Ix<Expr>>,
    pub expr: Ix<Expr>,
}

#[derive(Clone, Debug)]
pub struct AssociatedLet {
    pub vis: Vis,
    pub let_index: Ix<Let>,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Struct(Ix<Struct>),
    Function(Ix<Function>),
    Object(Object),
    BuiltinType(BuiltinType),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Object {
    pub storage_index: Ix<Storage>,
    pub struct_index: Ix<Struct>,
    pub fields: Vec<(DefaultSymbol, Value)>,
}

/// After comptime interpretation, every used expression must directly
/// or transitively depend on some runtime-only value (i.e. a runtime param).
#[derive(Clone, Debug)]
pub enum Expr {
    Value(Value),
    BinOp(BinOp, Ix<Self>, Ix<Self>),
    IfThenElse(Ix<Self>, Ix<Self>, Ix<Self>),
    Name(Name),
    Block(Ix<Block>),
    Call(Callee, Vec<Ix<Self>>),
    Comptime(Ix<Self>),
    // this contains all constructors before comptime interpretation.
    // After comptime interpretation though, this should only contain constructors that depend
    // on runtime values.
    Constructor(ConstructorType, Vec<StructField>),
    // After comptime interpretation, the expr whose field is being accessed should only be
    // runtime known. Otherwise we would have inlined the field.
    Field(Ix<Self>, Span<DefaultSymbol>),
}

impl Expr {
    pub fn is_value(&self) -> bool {
        matches!(self, Expr::Value(_))
    }
}

/// None is anonymous constructor, Some is a given type
type ConstructorType = Option<Ix<Expr>>;

#[derive(Copy, Clone, Debug)]
pub enum Callee {
    Expr(Ix<Expr>),
    BuiltinFunction(Span<BuiltinFunction>),
}

#[derive(Copy, Clone, Debug)]
pub struct Name {
    /// The actual name that was written.
    pub symbol: DefaultSymbol,

    /// The level of resolution the name occurred at, telling use which hir::Storage
    /// to look in.
    pub level: Level,

    /// What the name refers to.
    pub kind: NameKind,
}

#[derive(Copy, Clone, Debug)]
pub enum NameKind {
    AssociatedLet(Ix<AssociatedLet>),
    Function(Ix<Function>),
    Param,
    Let(Ix<Let>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    I32,
    Bool,
    Type,
    NoReturn,
}

impl BuiltinType {
    pub fn is_comptime_only(self) -> bool {
        match self {
            BuiltinType::I32 => false,
            BuiltinType::Bool => false,
            BuiltinType::Type => true,
            BuiltinType::NoReturn => false,
        }
    }
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

impl BuiltinType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::Bool => "bool",
            Self::Type => "type",
            Self::NoReturn => "noreturn",
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
            "noreturn" => Ok(Self::NoReturn),
            _ => Err(()),
        }
    }
}

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}
