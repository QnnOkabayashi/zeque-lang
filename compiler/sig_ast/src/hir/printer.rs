//! Display the HIR.

use string_interner::DefaultSymbol;

use crate::hir::{
    Block, Expr, Function, FunctionContext, Let, Binding, Parameter, Stmt, Struct, StructField,
    StructItem,
};
use crate::util::{Ix, StringInterner};
use std::fmt::{Debug, Formatter, Result};

#[derive(Copy, Clone)]
pub struct Printer<'a, T: ?Sized> {
    ctx: &'a FunctionContext,
    program: &'a [Function],
    interner: &'a StringInterner,
    inner: &'a T,
}

impl<'a> Printer<'a, Function> {
    pub fn new(inner: &'a Function, program: &'a [Function], interner: &'a StringInterner) -> Self {
        Printer {
            ctx: &inner.context,
            program,
            interner,
            inner,
        }
    }
}

impl<'a, T: ?Sized> Printer<'a, T> {
    fn wrap<U: ?Sized>(&self, inner: &'a U) -> Printer<'a, U> {
        Printer {
            ctx: self.ctx,
            program: self.program,
            interner: self.interner,
            inner,
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
        Debug::fmt(&self.wrap(&self.ctx.exprs[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Let>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.ctx.lets[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Block>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.ctx.blocks[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Struct>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.ctx.structs[*self.inner]), f)
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

impl Debug for Printer<'_, Function> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Function")
            .field("name", &self.wrap(&self.inner.name))
            .field("params", &self.wrap(self.inner.context.params.as_slice()))
            .field("return_type", &self.wrap(&self.inner.return_type))
            .field("body", &self.wrap(&self.inner.body))
            .finish()
    }
}

impl Debug for Printer<'_, Parameter> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Parameter")
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
        f.debug_map()
            .entries(
                self.inner
                    .fields
                    .iter()
                    .map(|struct_item| match struct_item {
                        StructItem::Field(struct_item) => {
                            let name = self.wrap(&struct_item.name);
                            let ty = self.wrap(&struct_item.value);

                            (name, ty)
                        }
                    }),
            )
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

impl Debug for Printer<'_, Binding> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Binding::Let(index) => f
                .debug_tuple("Let")
                .field(&self.ctx.lets[*index].name)
                .finish(),
            Binding::Parameter(index) => f
                .debug_tuple("Parameter")
                .field(&self.ctx.params[*index].name)
                .finish(),
            Binding::Function(index) => f
                .debug_tuple("Function")
                .field(&self.program[*index].name)
                .finish(),
            Binding::Builtin(builtin) => Debug::fmt(builtin, f),
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
