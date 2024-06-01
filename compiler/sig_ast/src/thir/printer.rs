//! Display the THIR.

use string_interner::DefaultSymbol;

use crate::thir::{
    Block, Constructor, Expr, Function, FunctionContext, Let, Name, Param, Stmt, Struct, Type,
};
use crate::util::{Ix, Span, StringInterner};
use std::fmt::{Debug, Formatter, Result};

#[derive(Copy, Clone)]
pub struct Printer<'a, T: ?Sized> {
    ctx: &'a FunctionContext,
    program: &'a [Function],
    structs: &'a [Struct],
    interner: &'a StringInterner,
    inner: &'a T,
}

impl<'a> Printer<'a, Function> {
    pub fn new(
        inner: &'a Function,
        program: &'a [Function],
        structs: &'a [Struct],
        interner: &'a StringInterner,
    ) -> Self {
        Printer {
            ctx: &inner.context,
            program,
            structs,
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
            structs: self.structs,
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

// note: this is only used for printing a function NAME for direct calls.
impl Debug for Printer<'_, Ix<Function>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut debug = f.debug_tuple(
            self.interner
                .resolve(self.program[*self.inner].name.0)
                .expect("symbol in interner"),
        );
        for filled_arg in &self.program[*self.inner].filled_args {
            debug.field(&filled_arg.as_deref().unwrap_or("_"));
        }
        debug.finish()
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

impl Debug for Printer<'_, Ix<Expr>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.ctx.exprs[*self.inner]), f)
    }
}

impl Debug for Printer<'_, Ix<Type>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.wrap(&self.ctx.expr_types[*self.inner]), f)
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
            .field("name", &self.wrap(&self.inner.name.0))
            .field("params", &self.wrap(self.inner.context.params.as_slice()))
            .field("return_type", &self.wrap(&self.inner.return_type))
            .field("body", &self.wrap(&self.inner.body))
            .finish()
    }
}

impl Debug for Printer<'_, Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Type::Builtin(builtin) => f.debug_tuple("Builtin").field(builtin).finish(),
            Type::Function(function_index) => {
                f.debug_tuple("Function").field(function_index).finish()
            }
            Type::Struct(struct_index) => f.debug_tuple("Struct").field(struct_index).finish(),
        }
    }
}

impl Debug for Printer<'_, Param> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Param")
            .field("name", &self.inner.name)
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
            Expr::Block(block) => f.debug_tuple("Block").field(&self.wrap(block)).finish(),
            Expr::DirectCall(callee, arguments) => f
                .debug_tuple("DirectCall")
                .field(&self.wrap(callee))
                .field(&self.wrap(arguments.as_slice()))
                .finish(),
            Expr::IndirectCall(callee, arguments) => f
                .debug_tuple("IndirectCall")
                .field(&self.wrap(callee))
                .field(&self.wrap(arguments.as_slice()))
                .finish(),
            Expr::Constructor(ctor_type, ref fields) => f
                .debug_tuple("Constructor")
                .field(&self.wrap(ctor_type))
                .field(&self.wrap(fields))
                .finish(),
            Expr::Field(expr, field) => f
                .debug_tuple("Field")
                .field(&self.wrap(expr))
                .field(field)
                .finish(),
        }
    }
}

impl Debug for Printer<'_, Name> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            Name::Let(index) => f
                .debug_tuple("Let")
                .field(&self.ctx.lets[*index].name)
                .finish(),
            Name::Parameter(index) => f
                .debug_tuple("Parameter")
                .field(&self.ctx.params[*index].name)
                .finish(),
            Name::Function(index) => f
                .debug_tuple("Function")
                .field(&self.program[*index].name)
                .finish(),
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
            .field("name", &self.inner.name)
            .field("ty", &self.wrap(&self.inner.ty))
            .field("expr", &self.wrap(&self.inner.expr))
            .finish()
    }
}

impl Debug for Printer<'_, Struct> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_map()
            .entries(self.inner.fields.iter().map(|struct_field| {
                let name = self.wrap(&struct_field.name);
                let ty = self.wrap(&struct_field.ty);

                (name, ty)
            }))
            .finish()
    }
}

impl Debug for Printer<'_, Constructor> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_map()
            .entries(self.inner.fields.iter().map(|ctor_field| {
                let name = self.wrap(&ctor_field.name);
                let value = self.wrap(&ctor_field.expr);

                (name, value)
            }))
            .finish()
    }
}
