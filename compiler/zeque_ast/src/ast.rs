//! The abstract syntax tree representation.

use crate::util::Range;
use smol_str::SmolStr;

#[derive(Clone, Debug)]
pub struct Name {
    pub text: SmolStr,
    pub range: Range,
}

#[derive(Clone, Debug)]
pub struct File {
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub decls: Vec<Decl>,
    pub range: Range,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Fn(FnDecl),
    Const(ConstDecl),
    Field(FieldDecl),
}

#[derive(Clone, Debug)]
pub struct FieldDecl {
    pub name: Name,
    pub ty: Expr,
}

#[derive(Clone, Debug)]
pub struct ConstDecl {
    pub name: Name,
    pub ty: Option<Expr>,
    pub value: Expr,
}

#[derive(Copy, Clone, Debug)]
pub struct Pub {
    pub range: Range,
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub pub_: Option<Pub>,
    pub name: Name,
    pub params: Vec<Param>,
    pub return_ty: Option<Expr>,
    pub body: Block,
}

#[derive(Copy, Clone, Debug)]
pub struct Comptime {
    pub range: Range,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub comptime: Option<Comptime>,
    pub name: Name,
    pub ty: Expr,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: Name,
    pub ty: Option<Expr>,
    pub expr: Expr,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub returns: Option<Box<Expr>>,
    pub range: Range,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(Let),
}

#[derive(Clone, Debug)]
pub struct Int {
    pub value: i32,
    pub range: Range,
}

#[derive(Clone, Debug)]
pub struct Bool {
    pub value: bool,
    pub range: Range,
}

#[derive(Clone, Debug)]
pub enum Str {
    Normal { string: SmolStr, range: Range },
    Raw(Vec<RawStrPart>),
}

#[derive(Clone, Debug)]
pub struct RawStrPart {
    // doesn't include leading \\
    pub string: SmolStr,
    // includes leading \\
    pub range: Range,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(Int),
    Bool(Bool),
    Str(Str),
    BinOp(Box<BinOp>),
    IfThenElse(Box<IfThenElse>),
    Name(Name),
    Block(Block),
    Call {
        callee: Callee,
        args: Vec<Expr>,
    },
    Comptime {
        comptime: Comptime,
        expr: Box<Expr>,
    },
    Struct(Struct),
    Constructor {
        ty: Option<Box<Expr>>,
        fields: Vec<ConstructorField>,
    },
    FieldAccess {
        expr: Box<Expr>,
        field_name: Name,
    },
    FnType,
}

#[derive(Clone, Debug)]
pub enum Callee {
    Expr(Box<Expr>),
    Builtin {
        // doesn't include the @
        name: SmolStr,
        // includes the @
        range: Range,
    },
}

#[derive(Clone, Debug)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub lhs: Expr,
    pub rhs: Expr,
}

#[derive(Copy, Clone, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Eq,
}

#[derive(Clone, Debug)]
pub struct IfThenElse {
    pub cond: Expr,
    pub then: Block,
    pub else_: Block,
    pub range: Range,
}

#[derive(Clone, Debug)]
pub struct ConstructorField {
    pub name: Name,
    pub expr: Expr,
}
