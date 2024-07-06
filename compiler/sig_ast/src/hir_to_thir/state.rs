use crate::util::{Ix, Range, Span};
use crate::{hir, thir};
use std::{fmt, hash, mem};
use string_interner::DefaultSymbol;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no main")]
    NoMain,
    #[error("main has params")]
    MainHasParams(Vec<Range>),
    #[error("value is not a type")]
    ValueIsNotAType,
    #[error("not callable")]
    NotCallable,
    #[error("wrong number of arguments")]
    WrongNumberOfArguments,
    #[error("value not of type `{0:?}`")]
    ValueNotOfType(Type),
    #[error("binary operation")]
    BinOp,
    #[error("comptime requires runtime parameter")]
    ComptimeExprUsesRuntimeArg(Ix<hir::Param>),
    #[error("lowered expr is not a type")]
    LoweredExprIsNotAType,
    #[error("expr is not a type")]
    ExprIsNotAType,
    #[error("type at runtime: `{0}`")]
    TypeAtRuntime(String),
    #[error("unbound recursion at compile time")]
    UnboundRecursionAtComptime,
    #[error("lowered is not a value")]
    ValueOrIxDoesNotHaveValue,
    #[error("let expr is not comptime known")]
    LetExprIsNotComptimeKnown,
    #[error("param is not comptime known")]
    ParamIsNotComptimeKnown,
    #[error("runtime expr passed into comptime param")]
    RuntimeExprPassedIntoComptimeParam(Range),
    #[error("non bool in conditional")]
    NonBoolInConditional,
    #[error("unbound recursion in comptime function")]
    UnboundRecursionInComptimeFunction(InvocationLocation),
    #[error("{0}")]
    TypeError(#[from] thir::typeck::Error),
    #[error("{0}")]
    SizeAlignError(#[from] thir::sizealign::Error),
    #[error("expected struct, found runtime value")]
    ExpectedStructFoundRuntimeValue,
    #[error("expected struct, found other value: `{0}`")]
    ExpectedStructFoundOtherValue(String),
    #[error("field `{0:?}` not found")]
    FieldNotFound(DefaultSymbol, Range),
    #[error("missing fields in constructor: {0:?}")]
    MissingFieldsInCtor(Vec<DefaultSymbol>),
    #[error("anonymous constructors are unimplemented")]
    UnimplementedAnonymousConstructor(Range),
    #[error("comptime-only structs are unimplemented")]
    UnimplementedComptimeOnlyStructs(Range),
    #[error("cannot convet instance of comptime struct into a runtime value")]
    CannotConvertInstanceOfComptimeStructIntoRuntimeValue,
    #[error("field does not exist")]
    FieldDoesNotExist(Range),
    #[error("reached trap in comptime")]
    ReachedTrapInComptime(Range),
    #[error("non int in clz")]
    NonIntInClz,
    #[error("non int in ctz")]
    NonIntInCtz,
}

#[derive(Copy, Clone, Debug)]
pub enum InvocationLocation {
    Main,
    CallSite(Range),
}

/// Values in the comptime interpreter.
#[derive(Clone, Debug, PartialEq, Eq, Hash, displaydoc::Display)]
pub enum Value {
    #[displaydoc("{0}")]
    Int(i32),

    #[displaydoc("{0}")]
    Bool(bool),

    // #[displaydoc("{0}")]
    // AssociatedFunction(Ix<hir::Struct>, Ix<hir::Function>),
    #[displaydoc("{0}")]
    Type(Type),

    #[displaydoc("{0}")]
    StructValue(StructValue),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct StructValue {
    pub kind: StructKind,
    pub fields: Vec<(Span<DefaultSymbol>, Value)>,
}

impl Value {
    pub fn into_type(&self) -> Result<Type, Error> {
        match self {
            Value::Type(ty) => Ok(*ty),
            _ => Err(Error::ValueIsNotAType),
        }
    }

    pub fn type_check(&self, ty: Type) -> Result<(), Error> {
        match (self, ty) {
            (Value::Int(_), Type::Builtin(hir::BuiltinType::I32)) => Ok(()),
            (Value::Type(_), Type::Builtin(hir::BuiltinType::Type)) => Ok(()),
            _ => Err(Error::ValueNotOfType(ty)),
        }
    }

    pub fn into_bool(self) -> Result<bool, Error> {
        match self {
            Value::Bool(b) => Ok(b),
            _ => Err(Error::NonBoolInConditional),
        }
    }

    pub fn into_thir_value(&self) -> Result<thir::Value, Error> {
        match self {
            Value::Int(int) => Ok(thir::Value::I32(*int)),
            Value::Bool(boolean) => Ok(thir::Value::Bool(*boolean)),
            // Value::AssociatedFunction(struct_index, function_index) => {
            //     todo!()
            //     // Ok(thir::Value::AssociatedFunction(
            //     //     *struct_index,
            //     //     *function_index,
            //     // ))
            // }
            Value::StructValue(struct_value) => {
                match struct_value.kind {
                    StructKind::Comptime(_) => {
                        Err(Error::CannotConvertInstanceOfComptimeStructIntoRuntimeValue)
                    }
                    StructKind::Anytime(struct_index) => {
                        // convert the struct into an expr
                        let fields = struct_value
                            .fields
                            .iter()
                            .map(|(name, value)| {
                                let value = value.into_thir_value()?;
                                Ok((*name, value))
                            })
                            .collect::<Result<_, Error>>()?;

                        Ok(thir::Value::Struct(todo!(), fields))
                        // Ok(thir::Expr::Constructor(
                        //     struct_index,
                        //     thir::Constructor { fields },
                        // ))
                    }
                }
            }
            Value::Type(_) => Err(Error::TypeAtRuntime(format!("{self:?}"))),
        }
    }

    // pub fn into_expr(&self, exprs: &mut Vec<thir::Expr>) -> Result<thir::Expr, Error> {
    //     match *self {
    //         Value::Int(int) => Ok(thir::Expr::Int(int)),
    //         Value::Bool(boolean) => Ok(thir::Expr::Bool(boolean)),
    //         // Value::AssociatedFunction(struct_index, function_index) => {
    //         //     todo!()
    //         //     // Ok(thir::Expr::Name(thir::Name::AssociatedFunction(
    //         //     //     struct_index.map(),
    //         //     //     function_index.map(),
    //         //     // )))
    //         // }
    //         Value::StructValue(ref value_struct) => {
    //             match value_struct.kind {
    //                 StructKind::Comptime(_) => {
    //                     Err(Error::CannotConvertInstanceOfComptimeStructIntoRuntimeValue)
    //                 }
    //                 StructKind::Anytime(struct_index) => {
    //                     // convert the struct into an expr
    //                     let fields = value_struct
    //                         .fields
    //                         .iter()
    //                         .map(|(name, value)| {
    //                             let value = value.into_expr(exprs)?;
    //
    //                             Ok(thir::ConstructorField {
    //                                 name: *name,
    //                                 expr: Ix::push(exprs, value),
    //                             })
    //                         })
    //                         .collect::<Result<_, Error>>()?;
    //
    //                     Ok(thir::Expr::Constructor(
    //                         struct_index,
    //                         thir::Constructor { fields },
    //                     ))
    //                 }
    //             }
    //         }
    //         Value::Type(Type::Builtin(_) | Type::Struct(_)) => {
    //             Err(Error::TypeAtRuntime(format!("{self:?}")))
    //         }
    //     }
    // }
}

impl fmt::Display for StructValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.fields.iter().map(|(name, value)| (name, value)))
            .finish()
    }
}

impl fmt::Debug for StructValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, displaydoc::Display)]
pub enum Type {
    #[displaydoc("{0}")]
    Builtin(hir::BuiltinType),

    #[displaydoc("{0}")]
    Struct(StructKind),
}

impl Type {
    pub fn from_thir_type(ty: thir::Type) -> Self {
        match ty {
            thir::Type::Builtin(builtin) => match builtin {
                thir::BuiltinType::I32 => Type::Builtin(hir::BuiltinType::I32),
                thir::BuiltinType::Bool => Type::Builtin(hir::BuiltinType::Bool),
                thir::BuiltinType::NoReturn => todo!(),
            },
            thir::Type::Function(_) => todo!(),
            thir::Type::Struct(struct_index) => Type::Struct(StructKind::Anytime(struct_index)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, displaydoc::Display)]
pub enum StructKind {
    #[displaydoc("{0}")]
    Comptime(Ix<ComptimeStruct>),

    #[displaydoc("{0}")]
    Anytime(Ix<thir::Struct>),
}

#[derive(Clone, Debug)]
pub struct ComptimeStruct {
    pub fields: Vec<ComptimeStructField>,
}

#[derive(Clone, Debug)]
pub struct ComptimeStructField {
    pub name: Span<DefaultSymbol>,
    pub ty: Type,
}

impl Type {
    /// Returns true if values of this type can only exist at compile time, otherwise false.
    pub fn is_comptime_only(self) -> bool {
        match self {
            Type::Builtin(builtin) => match builtin {
                hir::BuiltinType::Type => true,
                hir::BuiltinType::I32 | hir::BuiltinType::Bool | hir::BuiltinType::NoReturn => {
                    false
                }
            },
            Type::Struct(kind) => match kind {
                StructKind::Comptime(_) => true,
                StructKind::Anytime(_) => false,
            },
        }
    }

    pub fn into_runtime_type(self) -> Option<thir::Type> {
        match self {
            Type::Builtin(builtin) => match builtin {
                hir::BuiltinType::I32 => Some(thir::Type::Builtin(thir::BuiltinType::I32)),
                hir::BuiltinType::Bool => Some(thir::Type::Builtin(thir::BuiltinType::Bool)),
                hir::BuiltinType::Type => None,
                hir::BuiltinType::NoReturn => {
                    Some(thir::Type::Builtin(thir::BuiltinType::NoReturn))
                }
            },
            Type::Struct(kind) => match kind {
                StructKind::Comptime(_) => None,
                StructKind::Anytime(struct_index) => Some(thir::Type::Struct(struct_index)),
            },
        }
    }
}

/// The core type behind partial comptime evaluation.
#[derive(Debug)]
pub enum ValueOrIx<T> {
    /// A comptime-known value.
    Value(Value),
    /// A `T` index, which represents a runtime part of the thir.
    Index(Ix<T>),
}

// T doesn't have to be Clone
impl<T> Clone for ValueOrIx<T> {
    fn clone(&self) -> Self {
        match self {
            ValueOrIx::Value(value) => ValueOrIx::Value(value.clone()),
            ValueOrIx::Index(index) => ValueOrIx::Index(*index),
        }
    }
}

// T doesn't have to be PartialEq
impl<T> PartialEq for ValueOrIx<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Value(lhs), Self::Value(rhs)) => lhs == rhs,
            (Self::Index(lhs), Self::Index(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

// T doesn't have to be Eq
impl<T> Eq for ValueOrIx<T> {}

// T doesn't have to be Hash
impl<T> hash::Hash for ValueOrIx<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            ValueOrIx::Value(value) => value.hash(state),
            ValueOrIx::Index(index) => index.hash(state),
        }
    }
}

impl<T> ValueOrIx<T> {
    pub fn as_value(&self) -> Result<&Value, Error> {
        match self {
            ValueOrIx::Value(value) => Ok(value),
            _ => Err(Error::ValueOrIxDoesNotHaveValue),
        }
    }

    pub fn into_value(self) -> Result<Value, Error> {
        match self {
            ValueOrIx::Value(value) => Ok(value),
            _ => Err(Error::ValueOrIxDoesNotHaveValue),
        }
    }

    pub fn into_struct_type(self) -> Result<StructKind, Error> {
        match self {
            ValueOrIx::Value(value) => match value {
                Value::Type(Type::Struct(struct_kind)) => Ok(struct_kind),
                _ => {
                    //
                    Err(Error::ExpectedStructFoundOtherValue(format!("{value}")))
                }
            },
            _ => Err(Error::ExpectedStructFoundRuntimeValue),
        }
    }
}
