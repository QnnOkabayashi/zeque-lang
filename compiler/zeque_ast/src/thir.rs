//! The typed high-level intermediate representation.
//!
//! This IR represents a comptime-evaluated version of a Sig program.
//! This means that all comptime evaluation is complete, and all that's left is
//! a representation of what the program will do at runtime. Types are no longer values,
//! and each expression is now typed. The type of an expression is the type located in
//! the `types` field of [`FunctionContext`] at the same index as the expression, e.g.
//! `function_context.exprs.len() == function_context.types.len()` and they line up.
//!
//! For displaying the THIR as a tree structure, see the [`printer`] module.

pub use crate::ast::BinOpKind;
use crate::util::{Ix, Range, Span};
use std::{collections::HashMap, fmt};

pub mod local_offsets;
pub mod printer;
pub mod sizealign;
pub mod typeck;

use sizealign::StructSizeAlign;
use string_interner::DefaultSymbol;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    I32,
    Bool,
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Builtin::I32 => fmt::Display::fmt("i32", f),
            Builtin::Bool => fmt::Display::fmt("bool", f),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Builtin(Builtin),
    Function(Ix<Function>),
    Struct(Ix<Struct>),
    NoReturn,
}

// This is after monomorphization.
#[derive(Clone, Debug)]
pub struct Function {
    pub name: Span<DefaultSymbol>,
    pub filled_args: Vec<Option<String>>,
    pub return_type: Type,
    pub body: Ix<Block>,
    pub context: FunctionContext,
}

type LocalOffsets = u32;

#[derive(Clone, Debug, Default)]
pub struct FunctionContext {
    pub params: Vec<Param>,
    pub param_local_offsets: Vec<LocalOffsets>,

    pub lets: Vec<Let>,
    pub let_local_offsets: Vec<LocalOffsets>,

    pub blocks: Vec<Block>,

    pub exprs: Vec<Expr>,
    pub expr_types: Vec<Type>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: Span<DefaultSymbol>,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub name: Span<DefaultSymbol>,
    pub ty: Option<Type>,
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

#[derive(Copy, Clone, Debug)]
pub enum UnOp {
    Clz,
    Ctz,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i32),
    Bool(bool),
    UnOp(UnOp, Ix<Self>),
    BinOp(BinOpKind, Ix<Self>, Ix<Self>),
    IfThenElse(Ix<Self>, Ix<Self>, Ix<Self>),
    Name(Name),
    Block(Ix<Block>),
    DirectCall(Ix<Function>, Vec<Ix<Self>>),
    IndirectCall(Ix<Self>, Vec<Ix<Self>>),
    Constructor(Ix<Struct>, Constructor),
    Field(Ix<Self>, Span<DefaultSymbol>),
    /// Unconditional trap. In comptime, this is a compiler error. At runtime, this is a trap.
    Trap(Range),
}

#[derive(Copy, Clone, Debug)]
pub enum Name {
    Let(Ix<Let>),
    Parameter(Ix<Param>),
    Function(Ix<Function>),
}

#[derive(Debug)]
pub struct Struct {
    pub fields: Vec<StructField>,
    pub struct_sizealign: StructSizeAlign,
}

#[derive(Debug)]
pub struct StructField {
    pub name: Span<DefaultSymbol>,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Constructor {
    pub fields: Vec<ConstructorField>,
}

#[derive(Clone, Debug)]
pub struct ConstructorField {
    pub name: Span<DefaultSymbol>,
    pub expr: Ix<Expr>,
}

// impl Name {
//     pub fn as_symbol(self, ctx: &FunctionContext) -> DefaultSymbol {
//         match self {
//             Self::Let(index) => ctx.lets[index].name,
//             Self::Parameter(index) => ctx.params[index].name,
//             Self::Function(_) => todo!(),
//         }
//     }
// }

#[derive(Debug)]
pub struct Context {
    pub structs: Vec<Struct>,
    pub functions: Vec<Function>,
    pub type_metadata: TypeMetadata,
}

// None means we're in the process of computing it lower down the callstack.
type Acyclic<T> = Option<T>;

#[derive(Debug, Default)]
pub struct TypeMetadata {
    pub type_to_register_count: HashMap<Type, Acyclic<u32>>,
}

impl TypeMetadata {
    pub fn register_count(&mut self, ty: Type, structs: &[Struct]) -> u32 {
        if !self.type_to_register_count.contains_key(&ty) {
            self.type_to_register_count.insert(ty, None);

            let register_count = match ty {
                Type::Builtin(builtin) => match builtin {
                    Builtin::I32 => 1,
                    Builtin::Bool => 1,
                },
                Type::Function(_) => 0,
                Type::Struct(struct_index) => structs[struct_index]
                    .fields
                    .iter()
                    .map(|field| self.register_count(field.ty, structs))
                    .sum(),
                Type::NoReturn => 0,
            };

            self.type_to_register_count.insert(ty, Some(register_count));
        }

        self.type_to_register_count[&ty].expect("cyclic type")
    }

    pub fn registers(
        &mut self,
        ty: Type,
        structs: &[Struct],
        // may want to change this to take Ix<DefaultSymbol> instead
        accessed_fields: &[DefaultSymbol],
    ) -> OffsetAndLen {
        let Some((accessed_field, remaining_accessed_fields)) = accessed_fields.split_last() else {
            return OffsetAndLen {
                offset: 0,
                num_registers: self.register_count(ty, structs),
            };
        };

        let Type::Struct(struct_index) = ty else {
            panic!("uncaught type error, field access on non-struct");
        };

        let mut offset = 0;

        let struct_field = structs[struct_index]
            .fields
            .iter()
            .find(|field| {
                if field.name.0 == *accessed_field {
                    true
                } else {
                    offset += self.register_count(field.ty, structs);
                    false
                }
            })
            .expect("field not in struct");

        let mut offset_and_len =
            self.registers(struct_field.ty, structs, remaining_accessed_fields);
        offset_and_len.offset += offset;
        offset_and_len
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OffsetAndLen {
    pub offset: u32,
    pub num_registers: u32,
}
