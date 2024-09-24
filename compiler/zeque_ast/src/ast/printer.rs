//! Pretty printing facilities for the AST.

use crate::{ast::*, nameres::ResolvedName};
use index_vec::IndexSlice;
use std::fmt::{Debug, Formatter, Result};

pub struct Printer<'a, T: ?Sized> {
    exprs: &'a IndexSlice<ExprIdx, [Expr]>,
    names: NameTable<'a>,
    structs: &'a IndexSlice<StructIdx, [Struct]>,
    lets: &'a IndexSlice<LetIdx, [Let]>,
    params: &'a IndexSlice<ParamIdx, [Param]>,
    fn_decls: &'a IndexSlice<FnDeclIdx, [FnDecl]>,
    value: &'a T,
}

#[derive(Copy, Clone)]
pub enum NameTable<'a> {
    SmolStr(&'a IndexSlice<NameIdx, [SmolStr]>),
    ResolvedName(&'a IndexSlice<NameIdx, [ResolvedName]>),
}

impl<'a, T: ?Sized> Printer<'a, T> {
    pub fn new(
        exprs: &'a IndexSlice<ExprIdx, [Expr]>,
        names: NameTable<'a>,
        structs: &'a IndexSlice<StructIdx, [Struct]>,
        lets: &'a IndexSlice<LetIdx, [Let]>,
        params: &'a IndexSlice<ParamIdx, [Param]>,
        fn_decls: &'a IndexSlice<FnDeclIdx, [FnDecl]>,
        value: &'a T,
    ) -> Self {
        Printer {
            exprs,
            names,
            structs,
            lets,
            params,
            fn_decls,
            value,
        }
    }

    fn wrap<U: ?Sized>(&self, value: &'a U) -> Printer<'a, U> {
        Printer {
            exprs: self.exprs,
            names: self.names,
            structs: self.structs,
            lets: self.lets,
            params: self.params,
            fn_decls: self.fn_decls,
            value,
        }
    }
}

impl<'a, T> Debug for Printer<'a, [T]>
where
    Printer<'a, T>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_list()
            .entries(self.value.iter().map(|inner| self.wrap(inner)))
            .finish()
    }
}

impl<'a, T> Debug for Printer<'a, Option<T>>
where
    Printer<'a, T>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.value.as_ref().map(|val| self.wrap(val)), f)
    }
}

impl<'a> Debug for Printer<'a, ExprIdx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.exprs[*self.value]), f)
    }
}

impl<'a> Debug for Printer<'a, NameIdx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.names {
            NameTable::SmolStr(names) => Debug::fmt(&names[*self.value], f),
            NameTable::ResolvedName(names) => Debug::fmt(&names[*self.value], f),
        }
    }
}

impl<'a> Debug for Printer<'a, StructIdx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.structs[*self.value]), f)
    }
}

impl<'a> Debug for Printer<'a, LetIdx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.lets[*self.value]), f)
    }
}

impl<'a> Debug for Printer<'a, ParamIdx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.params[*self.value]), f)
    }
}

impl<'a> Debug for Printer<'a, FnDeclIdx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.fn_decls[*self.value]), f)
    }
}

impl<'a> Debug for Printer<'a, Expr> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.value {
            Expr::Int(int) => f.debug_tuple("Int").field(int).finish(),
            Expr::Bool(boolean) => f.debug_tuple("Bool").field(boolean).finish(),
            Expr::BinOp(op, lhs, rhs) => f
                .debug_tuple("BinOp")
                .field(op)
                .field(&self.wrap(lhs))
                .field(&self.wrap(rhs))
                .finish(),
            Expr::IfThenElse(cond, then, else_) => f
                .debug_tuple("IfThenElse")
                .field(&self.wrap(cond))
                .field(&self.wrap(then))
                .field(&self.wrap(else_))
                .finish(),
            Expr::Name(name) => Debug::fmt(&self.wrap(name), f),
            Expr::Call { callee, args } => f
                .debug_struct("Call")
                .field("callee", &self.wrap(callee))
                .field("args", &self.wrap(args.as_slice()))
                .finish(),
            Expr::Block(block) => f.debug_tuple("Block").field(&self.wrap(block)).finish(),
            Expr::Comptime(inner) => f.debug_tuple("Comptime").field(&self.wrap(inner)).finish(),
            Expr::Struct(struct_index) => f
                .debug_tuple("Struct")
                .field(&self.wrap(struct_index))
                .finish(),
            Expr::Constructor { ty, fields } => f
                .debug_struct("Constructor")
                .field("ty", &self.wrap(ty))
                .field("fields", &self.wrap(fields.as_slice()))
                .finish(),
            Expr::FieldAccess {
                expr,
                field_name: field,
            } => f
                .debug_struct("FieldAccess")
                .field("expr", &self.wrap(expr))
                .field("field", field)
                .finish(),
            Expr::Error(_) => write!(f, "Error"),
        }
    }
}

impl Debug for Printer<'_, Callee> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.value {
            Callee::Expr(expr_idx) => f.debug_tuple("Expr").field(&self.wrap(expr_idx)).finish(),
            Callee::Builtin(builtin) => f.debug_tuple("Builtin").field(builtin).finish(),
        }
    }
}

impl Debug for Printer<'_, Block> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Block")
            .field("stmts", &self.wrap(self.value.stmts.as_slice()))
            .field("returns", &self.wrap(&self.value.returns))
            .finish()
    }
}

impl Debug for Printer<'_, Stmt> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.value {
            Stmt::Let(let_) => Debug::fmt(&self.wrap(let_), f),
        }
    }
}

impl Debug for Printer<'_, Let> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Let")
            .field("name", &self.value.name)
            .field("ty", &self.wrap(&self.value.ty))
            .field("expr", &self.wrap(&self.value.expr))
            .finish()
    }
}

impl Debug for Printer<'_, ConstructorField> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("ConstructorField")
            .field("name", &self.value.name)
            .field("value", &self.wrap(&self.value.expr))
            .finish()
    }
}

impl Debug for Printer<'_, Struct> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Struct")
            .field("decls", &self.wrap(self.value.decls.as_slice()))
            .finish()
    }
}

impl Debug for Printer<'_, Decl> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.value {
            Decl::Fn(fn_decl_idx) => Debug::fmt(&self.wrap(fn_decl_idx), f),
            Decl::Field(field_decl) => Debug::fmt(&self.wrap(field_decl), f),
        }
    }
}

impl Debug for Printer<'_, FnDecl> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("FnDecl")
            .field("is_public", &self.value.is_public)
            .field("name", &self.value.name)
            .field("params", &self.wrap(self.value.params.as_slice()))
            .field("return_type", &self.wrap(&self.value.return_ty))
            .field("body", &self.wrap(&self.value.body))
            .finish()
    }
}

impl Debug for Printer<'_, FieldDecl> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("FieldDecl")
            .field("name", &self.value.name)
            .field("value", &self.wrap(&self.value.ty))
            .finish()
    }
}

impl Debug for Printer<'_, Param> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("FnParam")
            .field("is_comptime", &self.value.is_comptime)
            .field("name", &self.value.name)
            .field("ty", &self.wrap(&self.value.ty))
            .finish()
    }
}
