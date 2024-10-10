pub use crate::ast::BinOp;
use index_vec::IndexVec;
use smol_str::{SmolStr, ToSmolStr};
use std::{fmt, str::FromStr};

// pub mod printer;

index_vec::define_index_type! {
    #[derive(Default)]
    pub struct ExprIdx = u32;
}

index_vec::define_index_type! {
    pub struct LetIdx = u32;
}

index_vec::define_index_type! {
    pub struct StructIdx = u32;
}

index_vec::define_index_type! {
    pub struct ParamIdx = u32;
}

index_vec::define_index_type! {
    pub struct FnIdx = u32;
}

index_vec::define_index_type! {
    pub struct ParentRefIdx = u32;
}

index_vec::define_index_type! {
    #[derive(Default)]
    pub struct LevelIdx = u32;
}

index_vec::define_index_type! {
    pub struct FileIdx = u32;
}

#[derive(Debug)]
pub struct Hir {
    pub structs: IndexVec<StructIdx, Struct>,
    pub files: IndexVec<FileIdx, File>,
    pub main: FileIdx,
}

#[derive(Clone, Debug)]
pub struct File {
    pub struct_idx: StructIdx,
    pub ctx: Ctx,
}

#[derive(Clone, Debug, Default)]
pub struct Struct {
    pub fns: IndexVec<FnIdx, FnDecl>,
    pub fields: Vec<FieldDecl>,
}

#[derive(Clone, Debug, Default)]
pub struct FnDecl {
    pub is_pub: bool,
    pub name: SmolStr,
    pub params: Vec<ParamIdx>,
    pub return_ty: ExprIdx,
    pub body: Block,
    pub ctx: Ctx,
}

#[derive(Clone, Debug, Default)]
pub struct Ctx {
    // Params live in Ctx so NameRefs can refer to them easily
    pub params: IndexVec<ParamIdx, Param>,
    // Exprs live in Ctx so they can all be stored contiguously
    pub exprs: IndexVec<ExprIdx, Expr>,
    pub lets: IndexVec<LetIdx, Let>,
    /// Names in this context can look up where to find
    /// what they're looking for in their parent context.
    pub parent_captures: IndexVec<ParentRefIdx, ParentRef>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub is_comptime: bool,
    pub name: SmolStr,
    pub ty: ExprIdx,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: SmolStr,
    pub ty: Option<ExprIdx>,
    pub expr: ExprIdx,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub returns: ExprIdx,
}

#[derive(Copy, Clone, Debug)]
pub enum Stmt {
    Let(LetIdx),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i32),
    Bool(bool),
    BinOp {
        op: BinOp,
        lhs: ExprIdx,
        rhs: ExprIdx,
    },
    IfThenElse {
        cond: ExprIdx,
        then: ExprIdx,
        else_: ExprIdx,
    },
    Name(Name),
    BuiltinType(BuiltinType),
    SelfType,
    Block(Block),
    Call {
        callee: Callee,
        args: Vec<ExprIdx>,
    },
    Comptime(ExprIdx),
    Struct(StructIdx),
    Constructor {
        ty: Option<ExprIdx>,
        fields: Vec<ConstructorField>,
    },
    Field {
        expr: ExprIdx,
        field_name: SmolStr,
    },
    Error,
}

#[derive(Clone, Debug)]
pub enum Callee {
    Expr(ExprIdx),
    Builtin(BuiltinFn),
}

#[derive(Copy, Clone, Debug)]
pub enum Name {
    Local(Local),
    /// Reference to a name declaration.
    /// go look in your own context, and it will tell you where
    /// in the parent to look
    ParentRef(ParentRefIdx),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Local {
    Let(LetIdx),
    Param(ParamIdx),
    Fn(StructIdx, FnIdx),
}

#[derive(Copy, Clone, Debug)]
pub enum ParentRef {
    /// Look in my parent's layer
    Local(Local),
    /// Something my parent captured, keep going...
    Capture(ParentRefIdx),
}

#[derive(Clone, Debug)]
pub struct ConstructorField {
    pub name: SmolStr,
    pub value: ExprIdx,
}

#[derive(Clone, Debug)]
pub struct FieldDecl {
    pub name: SmolStr,
    pub ty: ExprIdx,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    I32,
    Bool,
    Type,
}

#[derive(Clone, Debug)]
pub enum BuiltinFn {
    InComptime,
    Trap,
    Clz,
    Ctz,
    Unknown(SmolStr),
}

impl FromStr for BuiltinFn {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "in_comptime" => Ok(BuiltinFn::InComptime),
            "trap" => Ok(BuiltinFn::Trap),
            "clz" => Ok(BuiltinFn::Clz),
            "ctz" => Ok(BuiltinFn::Ctz),
            _ => Ok(BuiltinFn::Unknown(s.to_smolstr())),
        }
    }
}

impl BuiltinType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BuiltinType::I32 => "i32",
            BuiltinType::Bool => "bool",
            BuiltinType::Type => "type",
        }
    }
}

impl FromStr for BuiltinType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i32" => Ok(BuiltinType::I32),
            "bool" => Ok(BuiltinType::Bool),
            "type" => Ok(BuiltinType::Type),
            _ => Err(()),
        }
    }
}

impl fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}
