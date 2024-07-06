//! Display the HIR.

use crate::hir::{
    Block, Callee, Expr, Function, Let, Name, Param, Stmt, Struct, StructField, StructStorage,
};
use crate::util::{Ix, Span, StringInterner};
use std::fmt::{Debug, Formatter, Result};
use string_interner::DefaultSymbol;

#[derive(Copy, Clone)]
pub struct Printer<'a, T: ?Sized> {
    storage: &'a StructStorage,
    params: Option<&'a [Param]>,
    structs: &'a [Struct],
    interner: &'a StringInterner,
    inner: &'a T,
}

impl Printer<'_, Ix<Struct>> {
    pub fn new(inner: &Ix<Struct>, structs: &[Struct], interner: &StringInterner) -> Self {
        Printer {
            storage: &structs[*inner].storage,
            params: None,
            structs,
            interner,
            inner,
        }
    }
}

impl<'a, T: ?Sized> Printer<'a, T> {
    fn wrap<U: ?Sized>(&self, inner: &'a U) -> Printer<'a, U> {
        Printer {
            storage: self.storage,
            params: self.params,
            structs: self.structs,
            interner: self.interner,
            inner,
        }
    }

    fn with_params(&self, params: &'a [Param]) -> Self {
        Self {
            params: Some(params),
            ..*self
        }
    }
}

// Prints through [T]
impl<'a, T> Debug for Printer<'a, [T]>
where
    Printer<'a, T>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_list()
            .entries(self.inner.iter().map(|inner| self.wrap(inner)))
            .finish()
    }
}

// Prints through Option<T>
impl<'a, T> Debug for Printer<'a, Option<T>>
where
    Printer<'a, T>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Some(value) => f.debug_tuple("Some").field(&self.wrap(value)).finish(),
            None => write!(f, "None"),
        }
    }
}

impl Debug for Printer<'_, Ix<Expr>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.storage.exprs[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Let>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.storage.lets[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Block>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.storage.blocks[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Struct>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.structs[*self.inner]), f)
    }
}

impl Debug for Printer<'_, DefaultSymbol> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(
            self.interner
                .resolve(*self.inner)
                .expect("symbol in interner"),
            f,
        )
    }
}

impl<'a, T> Debug for Printer<'a, Span<T>>
where
    Printer<'a, T>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Span")
            .field(&self.wrap(self.inner))
            .field(&self.inner.1)
            .finish()
    }
}

impl Debug for Printer<'_, Function> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Function")
            .field("name", &self.wrap(&self.inner.name))
            .field("params", &self.wrap(&*self.inner.params))
            .field("return_type", &self.wrap(&self.inner.return_type))
            .field(
                "body",
                &self.with_params(&self.inner.params).wrap(&self.inner.body),
            )
            .finish()
    }
}

impl Debug for Printer<'_, Param> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Param")
            .field("name", &self.wrap(&self.inner.name))
            .field("ty", &self.wrap(&self.inner.ty))
            .finish()
    }
}

impl Debug for Printer<'_, Block> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Block")
            .field("stmts", &self.wrap(self.inner.stmts.as_slice()))
            .field("returns", &self.wrap(&self.inner.returns))
            .finish()
    }
}

impl Debug for Printer<'_, Struct> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Struct")
            .field("fields", &self.wrap(self.inner.fields.as_slice()))
            .field("lets", &self.wrap(self.inner.lets.as_slice()))
            .field("functions", &self.wrap(self.inner.functions.as_slice()))
            .finish()
    }
}

impl Debug for Printer<'_, Expr> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
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
            Expr::Name(name) => f.debug_tuple("Name").field(&self.wrap(name)).finish(),
            Expr::Call(callee, arguments) => f
                .debug_tuple("Call")
                .field(&self.wrap(callee))
                .field(&self.wrap(arguments.as_slice()))
                .finish(),
            Expr::Block(block) => f.debug_tuple("Block").field(&self.wrap(block)).finish(),
            Expr::Comptime(inner) => f.debug_tuple("Comptime").field(&self.wrap(inner)).finish(),
            Expr::Struct(struct_index) => f
                .debug_tuple("Struct")
                .field(&self.wrap(struct_index))
                .finish(),
            Expr::Constructor(ctor_type, struct_fields) => f
                .debug_tuple("Constructor")
                .field(&self.wrap(ctor_type))
                .field(&self.wrap(struct_fields.as_slice()))
                .finish(),
            Expr::Field(value, field) => f
                .debug_tuple("Field")
                .field(&self.wrap(value))
                .field(&self.wrap(field))
                .finish(),
        }
    }
}

impl Debug for Printer<'_, Name> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Name::AssociatedLet(struct_index, let_index) => {
                let struct_ = self.structs[*struct_index];
                let let_index = struct_.lets[*let_index];
                let let_ = &struct_.storage.lets[let_index];

                f.debug_tuple("Function").field(&let_.name).finish()
            }
            Name::AssociatedFunction(struct_index, function_index) => f
                .debug_tuple("Function")
                .field(&self.structs[*struct_index].functions[*function_index].name)
                .finish(),
            Name::Let(index) => f
                .debug_tuple("Let")
                .field(&self.storage.lets[*index].name)
                .finish(),
            Name::Param(index) => f
                .debug_tuple("Param")
                .field(&self.params.expect("not in a function")[*index].name)
                .finish(),
            Name::Builtin(builtin) => Debug::fmt(builtin, f),
        }
    }
}

impl Debug for Printer<'_, Stmt> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Stmt::Let(index) => f.debug_tuple("Let").field(&self.wrap(index)).finish(),
        }
    }
}

impl Debug for Printer<'_, Let> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Let")
            .field("name", &self.wrap(&self.inner.name))
            .field("ty", &self.wrap(&self.inner.ty))
            .field("expr", &self.wrap(&self.inner.expr))
            .finish()
    }
}

impl Debug for Printer<'_, StructField> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("StructField")
            .field("name", &self.wrap(&self.inner.name))
            .field("value", &self.wrap(&self.inner.value))
            .finish()
    }
}

impl Debug for Printer<'_, Callee> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Callee::Expr(expr) => f.debug_tuple("Expr").field(&self.wrap(expr)).finish(),
            Callee::Builtin(builtin) => f.debug_tuple("Builtin").field(builtin).finish(),
        }
    }
}
