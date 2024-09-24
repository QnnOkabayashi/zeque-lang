pub use crate::ast::BinOp;
use index_vec::IndexVec;
use smol_str::{SmolStr, ToSmolStr};
use std::{fmt, str::FromStr};

// pub mod printer;

index_vec::define_index_type! {
    pub struct Level = u32;
}

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

#[derive(Clone, Debug)]
pub struct File {
    pub struct_idx: StructIdx,
    pub ctx: Ctx,
}

impl File {
    pub fn struct_(&self) -> &Struct {
        &self.ctx.structs[self.struct_idx]
    }
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
    pub params: IndexVec<ParamIdx, Param>,
    pub exprs: IndexVec<ExprIdx, Expr>,
    pub lets: IndexVec<LetIdx, Let>,
    pub structs: IndexVec<StructIdx, Struct>,
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
    BuiltinType(BuiltinType),
}

#[derive(Copy, Clone, Debug)]
pub struct Local {
    pub level: Level,
    pub kind: LocalKind,
}

#[derive(Copy, Clone, Debug)]
pub enum LocalKind {
    Let(LetIdx),
    Param(ParamIdx),
    Fn(StructIdx, FnIdx),
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
