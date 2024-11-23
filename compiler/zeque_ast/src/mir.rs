use crate::ast::BinOpKind;
use index_vec::IndexVec;
use smol_str::SmolStr;

index_vec::define_index_type! {
    pub struct ParamIdx = u32;
}

index_vec::define_index_type! {
    pub struct LetIdx = u32;
}

index_vec::define_index_type! {
    /// Index of a monomorphized struct.
    pub struct StructIdx = u32;
}

index_vec::define_index_type! {
    /// Index type for [`Expr`].
    pub struct ExprIdx = u32;
}

index_vec::define_index_type! {
    pub struct FnIdx = u32;
}

pub struct Mir {
    pub structs: IndexVec<StructIdx, Struct>,
    pub fns: IndexVec<FnIdx, FnDecl>,
}

#[derive(Default)]
pub struct Struct {
    pub fields: Vec<FieldDecl>,
}

pub struct FieldDecl {
    pub name: SmolStr,
    pub ty: Type,
}

pub struct FnDecl {
    pub is_pub: bool,
    pub name: SmolStr,
    pub params: IndexVec<ParamIdx, Param>,
    pub return_ty: Type,
    pub body: Block,
    pub exprs: IndexVec<ExprIdx, Expr>,
}

pub struct Param {
    pub name: SmolStr,
    pub ty: Type,
}

pub enum Expr {
    Int(i32),
    Bool(bool),
    Linear,
    BinOp {
        op: BinOpKind,
        lhs: ExprIdx,
        rhs: ExprIdx,
    },
    IfThenElse {
        cond: ExprIdx,
        then: Block,
        else_: Block,
    },
    Name(Name),
    Block(Block),
    Call {
        callee: Callee,
        args: Vec<ExprIdx>,
    },
    /// Fn singleton value
    Fn(FnIdx),
    Constructor {
        ty: Option<StructIdx>,
        fields: Vec<ConstructorField>,
    },
    Field {
        expr: ExprIdx,
        field_name: SmolStr,
    },
    Error,
}

pub enum Callee {
    Expr(ExprIdx),
    Builtin(BuiltinFn),
}

pub enum BuiltinFn {
    Trap,
    Clz,
    Ctz,
}

pub enum Name {
    Param(ParamIdx),
    Let(LetIdx),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub returns: Option<ExprIdx>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(Let),
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: SmolStr,
    pub ty: Option<Type>,
    pub expr: ExprIdx,
}

pub struct ConstructorField {
    pub name: SmolStr,
    pub value: ExprIdx,
}

#[derive(Clone, Debug)]
pub enum Type {
    Unit,
    I32,
    Bool,
    Linear,
    Struct(StructIdx),
    /// Fn singleton type
    Fn(FnIdx),
}
