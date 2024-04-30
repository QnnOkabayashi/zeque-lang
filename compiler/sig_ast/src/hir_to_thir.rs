use crate::{hir, thir, util::Ix};
use std::collections::hash_map::Entry;
use std::fmt::Write as _;
use std::{collections::HashMap, fmt, hash, mem};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no main")]
    NoMain,
    #[error("main has params")]
    MainHasParams,
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
    ComptimeExprUsesRuntimeArg(Ix<hir::Parameter>),
    #[error("lowered expr is not a type")]
    LoweredExprIsNotAType,
    #[error("expr is not a type")]
    ExprIsNotAType,
    #[error("type at runtime")]
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
    RuntimeExprPassedIntoComptimeParam,
    #[error("non bool in conditional")]
    NonBoolInConditional,
    #[error("unbound recursion in comptime function")]
    UnboundRecursionInComptimeFunction,
    #[error("{0}")]
    TypeError(#[from] thir::typeck::Error),
}

/// Values that can exist during comptime execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Function(Ix<hir::Function>),
    Type(Type),
}

impl Value {
    fn into_type(self) -> Result<Type, Error> {
        match self {
            Value::Type(ty) => Ok(ty),
            _ => Err(Error::ValueIsNotAType),
        }
    }

    fn check_type(self, ty: Type) -> Result<(), Error> {
        match (self, ty) {
            (Value::Int(_), Type::Builtin(hir::Builtin::I32)) => Ok(()),
            (Value::Type(_), Type::Builtin(hir::Builtin::Type)) => Ok(()),
            (Value::Function(value_index), Type::Function(type_index))
                if value_index.index == type_index.index =>
            {
                Ok(())
            }
            _ => Err(Error::ValueNotOfType(ty)),
        }
    }

    fn into_bool(self) -> Result<bool, Error> {
        match self {
            Value::Bool(b) => Ok(b),
            _ => Err(Error::NonBoolInConditional),
        }
    }

    fn into_expr(self) -> Result<thir::Expr, Error> {
        match self {
            Value::Int(int) => Ok(thir::Expr::Int(int)),
            Value::Bool(boolean) => Ok(thir::Expr::Bool(boolean)),
            Value::Function(function_index) => {
                Ok(thir::Expr::Name(thir::Name::Function(function_index.map())))
            }
            Value::Type(Type::Builtin(_) | Type::Function(_)) => {
                Err(Error::TypeAtRuntime(format!("{self:?}")))
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(int) => fmt::Display::fmt(int, f),
            Value::Bool(boolean) => fmt::Display::fmt(boolean, f),
            Value::Function(_function_index) => todo!("display function value"),
            Value::Type(ty) => fmt::Display::fmt(ty, f),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Builtin(hir::Builtin),
    // Make sure that when we create this, it's only talking about functions that can
    // actually be called at runtime.
    Function(Ix<thir::Function>),
}

impl Type {
    // why do we have 2 functions that do almost the same thing.
    fn is_comptime_only(&self) -> bool {
        match self {
            Type::Builtin(builtin) => match builtin {
                hir::Builtin::I32 => false,
                hir::Builtin::Bool => false,
                hir::Builtin::Type => true,
            },
            Type::Function(_) => todo!(),
        }
    }

    fn into_runtime_type(self) -> Option<thir::Type> {
        match self {
            Type::Builtin(builtin) => match builtin {
                hir::Builtin::I32 => Some(thir::Type::Builtin(thir::Builtin::I32)),
                hir::Builtin::Bool => Some(thir::Type::Builtin(thir::Builtin::Bool)),
                hir::Builtin::Type => None,
            },
            Type::Function(function_index) => Some(thir::Type::Function(function_index)),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Builtin(builtin) => fmt::Display::fmt(builtin, f),
            Type::Function(function_index) => fmt::Display::fmt(function_index, f),
        }
    }
}

/// The entry into the `hir_to_thir` module.
///
/// Returns the lowered functions (lowered recursively based on usage by the main function),
/// as well as the index of the main function (or the value if it can be computed entirely at
/// comptime).
pub fn entry(
    functions: &[hir::Function],
) -> Result<(Vec<thir::Function>, ValueOrIx<thir::Function>), Error> {
    let main_index = Ix::new(
        functions
            .iter()
            .position(|function| function.name == "main")
            .ok_or(Error::NoMain)?,
    );

    let main = &functions[main_index];

    if !main.context.params.is_empty() {
        return Err(Error::MainHasParams);
    }

    let next_argument =
        |_: &mut MonomorphizedFunctions, _: usize, _: bool| unreachable!("main has no arguments");

    let mut monomorphized_functions = MonomorphizedFunctions::new();

    let in_comptime = false;
    let value_or_main = monomorphized_functions.lower_function(
        functions,
        main_index,
        next_argument,
        in_comptime,
    )?;

    let mut functions = monomorphized_functions
        .functions
        .into_iter()
        .collect::<Option<Vec<thir::Function>>>()
        .expect("expected all functions to be filled in");

    thir::typeck::entry(&mut functions)?;

    Ok((functions, value_or_main))
}

/// The core type behind partial comptime evaluation.
#[derive(Debug)]
pub enum ValueOrIx<T> {
    /// A comptime-known value.
    Value(Value),
    /// A T index, which represents a runtime part of the thir.
    Index(Ix<T>),
}

// T doesn't have to be copy
impl<T> Copy for ValueOrIx<T> {}

// T doesn't have to be clone
impl<T> Clone for ValueOrIx<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for ValueOrIx<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Value(lhs), Self::Value(rhs)) => lhs == rhs,
            (Self::Index(lhs), Self::Index(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl<T> Eq for ValueOrIx<T> {}

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
    fn into_value(self) -> Result<Value, Error> {
        match self {
            ValueOrIx::Value(value) => Ok(value),
            _ => Err(Error::ValueOrIxDoesNotHaveValue),
        }
    }
}

/// Represents a call to a function call at comptime.
/// These are stored in a map during comptime evaluation to avoid
/// computing the same comptime function twice (we just memoize the result).
/// For example, this allows recursive fib to be linear at comptime.
#[derive(Clone, PartialEq, Eq, Hash)]
struct FunctionInvocation {
    index: Ix<hir::Function>,
    arguments: Box<[ValueOrIx<thir::Param>]>,
    return_type: Type,
}

#[derive(Copy, Clone)]
enum LoweringValueOrIx {
    LoweringValue,
    Index(Ix<thir::Function>),
}

enum Memoized {
    Lowering(LoweringValueOrIx),
    Lowered(ValueOrIx<thir::Function>),
}

// if the invocation always has the same value (and thus resolves to a ValueOrIx::Value),
// then calling it recursively is a compile error.
// If it doesn't, then point to the index of where it will be
struct MonomorphizedFunctions {
    memoized: HashMap<FunctionInvocation, Memoized>,
    functions: Vec<Option<thir::Function>>,
}

impl MonomorphizedFunctions {
    fn new() -> Self {
        MonomorphizedFunctions {
            memoized: HashMap::new(),
            functions: Vec::new(),
        }
    }

    /// Lower a function call with the comptime arguments applied.
    /// If the result of monomorphization is just returning a single
    /// comptime-know value, then the value is returned.
    fn lower_function<F>(
        &mut self,
        functions: &[hir::Function],
        function_index: Ix<hir::Function>,
        mut lower_nth_argument: F,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Function>, Error>
    where
        F: FnMut(&mut MonomorphizedFunctions, usize, bool) -> Result<ValueOrIx<thir::Param>, Error>,
    {
        let hir_function = &functions[function_index];

        let mut arguments = vec![None; hir_function.context.params.len()];

        let mut state = LoweringState {
            functions,
            hir_context: &hir_function.context,
            thir_context: thir::FunctionContext::default(),
            values_or_lets: Vec::with_capacity(hir_function.context.lets.len()),
            values_or_exprs: Vec::with_capacity(hir_function.context.exprs.len()),
            values_or_params: &mut arguments,
        };

        for (index, param) in hir_function.context.params.iter().enumerate() {
            let eval_type_in_comptime = true;
            let param_type_expr_index = state.lower_expr(param.ty, self, eval_type_in_comptime)?;
            let param_type = state.values_or_exprs[..][param_type_expr_index]
                .into_value()?
                .into_type()?;

            // Pattern that I noticed: all the args for comptime params are
            // lowered first, with in_comptime=true.
            // Then, the args for runtime params are lowered, with a varying in_comptime.
            // After monomorphization, only runtime parameters exist.
            match param_type.into_runtime_type() {
                Some(ty) if !param.is_comptime => {
                    state.thir_context.params.push(thir::Param {
                        name: param.name.clone(),
                        ty,
                    });
                }
                _ => {
                    let argument = lower_nth_argument(self, index, true)?;

                    let ValueOrIx::Value(arg_value) = argument else {
                        return Err(Error::RuntimeExprPassedIntoComptimeParam);
                    };

                    arg_value.check_type(param_type)?;

                    let nothing = state.values_or_params[index].replace(argument);
                    assert!(
                        nothing.is_none(),
                        "evaluated the same argument twice? this is a bug"
                    );
                }
            }
        }

        let return_type_expr_index = state.lower_expr(hir_function.return_type, self, true)?;
        let return_type = state.values_or_exprs[..][return_type_expr_index]
            .into_value()?
            .into_type()?;

        // If we find out that the function returns a comptime-only value,
        // then all evaluation must be done at comptime
        let in_comptime = in_comptime || return_type.is_comptime_only();

        // Evaluate the rest of the non-comptime params
        for index in 0..hir_function.context.params.len() {
            if state.values_or_params[index].is_none() {
                let argument = lower_nth_argument(self, index, in_comptime)?;

                state.values_or_params[index] = Some(argument);
            }
        }

        let arguments = state
            .values_or_params
            .iter()
            .cloned()
            .collect::<Option<Vec<ValueOrIx<thir::Param>>>>()
            .expect("not all arguments were evaluated, this is a bug")
            .into_boxed_slice();

        // if the function index, param types, and return type are all the same, then it's the same
        // function.
        let key = FunctionInvocation {
            index: function_index,
            arguments: arguments.clone(),
            return_type,
        };

        // Where this function is going to end up
        let lowered_function_index = match self.memoized.entry(key.clone()) {
            Entry::Occupied(occupied) => {
                return match *occupied.get() {
                    Memoized::Lowering(LoweringValueOrIx::LoweringValue) => {
                        Err(Error::UnboundRecursionInComptimeFunction)
                    }
                    Memoized::Lowering(LoweringValueOrIx::Index(index)) => {
                        Ok(ValueOrIx::Index(index))
                    }
                    Memoized::Lowered(value_or_index) => Ok(value_or_index),
                }
            }
            Entry::Vacant(vacant) => {
                if in_comptime {
                    let lowering_or_index = LoweringValueOrIx::LoweringValue;
                    vacant.insert(Memoized::Lowering(lowering_or_index));
                    lowering_or_index
                } else {
                    let lowered_function_index = Ix::push(&mut self.functions, None).map();
                    let lowering_or_index = LoweringValueOrIx::Index(lowered_function_index);
                    vacant.insert(Memoized::Lowering(lowering_or_index));
                    lowering_or_index
                }
            }
        };

        let value_or_block_index = state.lower_block(hir_function.body, self, in_comptime)?;

        // For function blocks, if the block was evaluated to a value but the function wasn't asked
        // to be generated at comptime, a runtime block expression is generated that just returns
        // the value.
        let value_or_block_index = match value_or_block_index {
            ValueOrIx::Value(value) => {
                if in_comptime {
                    ValueOrIx::Value(value)
                } else {
                    let expr_index = state.alloc_expr(value.into_expr()?);
                    ValueOrIx::Index(Ix::push(
                        &mut state.thir_context.blocks,
                        thir::Block {
                            stmts: vec![],
                            returns: expr_index,
                        },
                    ))
                }
            }
            ValueOrIx::Index(block_index) => ValueOrIx::Index(block_index),
        };

        let value_or_function_index = match value_or_block_index {
            ValueOrIx::Value(value) => ValueOrIx::Value(value),
            ValueOrIx::Index(block_index) => {
                // this name generation part should be its own function probably.
                let mut name = hir_function.name.clone();
                name.push('(');
                if let Some((head, tail)) = arguments.split_first() {
                    match head {
                        ValueOrIx::Value(value) => write!(&mut name, "{value}").unwrap(),
                        ValueOrIx::Index(_) => name.push('_'),
                    }
                    for param in tail {
                        match param {
                            ValueOrIx::Value(value) => write!(&mut name, ", {value}").unwrap(),
                            ValueOrIx::Index(_) => name.push_str(", _"),
                        }
                    }
                }
                name.push(')');

                let function = thir::Function {
                    name,
                    return_type: return_type.into_runtime_type().expect("apparently return type was comptime only but the return expression is only runtime known? that's a bug"),
                    body: block_index,
                    context: state.thir_context,
                };

                // this is so bad pls rewrite with just option
                let lowered_function_index = match lowered_function_index {
                    LoweringValueOrIx::LoweringValue => unreachable!(
                        "tried to lowering a comptime-only function but got a runtime value"
                    ),
                    LoweringValueOrIx::Index(index) => index,
                };

                self.functions[..][lowered_function_index.map()] = Some(function);

                ValueOrIx::Index(lowered_function_index)
            }
        };

        self.memoized
            .insert(key, Memoized::Lowered(value_or_function_index));

        Ok(value_or_function_index)
    }
}

/// Type for lowering an [`hir::Function`] into a [`thir::Function`]. Comptime expressions are
/// evaluated into values, and runtime expressions and lowered into thir instructions.
struct LoweringState<'hir, 'params> {
    functions: &'hir [hir::Function],
    hir_context: &'hir hir::FunctionContext,
    thir_context: thir::FunctionContext,

    values_or_lets: Vec<ValueOrIx<thir::Let>>,
    values_or_exprs: Vec<ValueOrIx<thir::Expr>>,
    values_or_params: &'params mut [Option<ValueOrIx<thir::Param>>],
}

impl LoweringState<'_, '_> {
    fn lower_let(
        &mut self,
        let_index: Ix<hir::Let>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Let>, Error> {
        let let_ = &self.hir_context.lets[..][let_index];
        let ty = let_
            .ty
            .map(|ty| {
                let in_comptime = true;
                let ty = self.lower_expr(ty, monomorphized_functions, in_comptime)?;
                self.values_or_exprs[..][ty].into_value()?.into_type()
            })
            .transpose()?;

        let value_or_expr_index =
            self.lower_expr(let_.expr, monomorphized_functions, in_comptime)?;

        let value_or_let = match self.values_or_exprs[..][value_or_expr_index] {
            ValueOrIx::Value(value) => ValueOrIx::Value(value),
            ValueOrIx::Index(expr) => ValueOrIx::Index(Ix::push(
                &mut self.thir_context.lets,
                thir::Let {
                    name: let_.name.clone(),
                    ty: ty.map(|t| t.into_runtime_type().expect("let expression has type whose values only exist at comptime, yet the expression is only known at runtime")),
                    expr,
                },
            )),
        };
        self.values_or_lets.push(value_or_let);
        Ok(value_or_let)
    }

    fn lower_block(
        &mut self,
        block_index: Ix<hir::Block>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Block>, Error> {
        let mut stmts = Vec::new();
        let function_body = &self.hir_context.blocks[..][block_index];
        for &stmt in &function_body.stmts {
            match stmt {
                hir::Stmt::Let(hir_let_index) => {
                    if let ValueOrIx::Index(let_index) =
                        self.lower_let(hir_let_index, monomorphized_functions, in_comptime)?
                    {
                        stmts.push(thir::Stmt::Let(let_index));
                    }
                }
            }
        }

        let return_expr =
            self.lower_expr(function_body.returns, monomorphized_functions, in_comptime)?;

        match self.values_or_exprs[..][return_expr] {
            ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
            ValueOrIx::Index(returns) => {
                let block = thir::Block { stmts, returns };
                let block_index = Ix::push(&mut self.thir_context.blocks, block);
                Ok(ValueOrIx::Index(block_index))
            }
        }
    }

    fn alloc_expr(&mut self, expr: thir::Expr) -> Ix<thir::Expr> {
        Ix::push(&mut self.thir_context.exprs, expr)
    }

    fn lower_expr(
        &mut self,
        expr_index: Ix<hir::Expr>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<Ix<ValueOrIx<thir::Expr>>, Error> {
        let expr = self.lower_expr_raw(expr_index, monomorphized_functions, in_comptime)?;

        Ok(Ix::push(&mut self.values_or_exprs, expr))
    }

    fn lower_expr_raw(
        &mut self,
        expr_index: Ix<hir::Expr>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Expr>, Error> {
        match self.hir_context.exprs[..][expr_index] {
            hir::Expr::Int(int) => Ok(ValueOrIx::Value(Value::Int(int))),
            hir::Expr::Bool(boolean) => Ok(ValueOrIx::Value(Value::Bool(boolean))),
            hir::Expr::BinOp(op, lhs, rhs) => {
                let lhs = self.lower_expr(lhs, monomorphized_functions, in_comptime)?;
                let rhs = self.lower_expr(rhs, monomorphized_functions, in_comptime)?;
                match (self.values_or_exprs[..][lhs], self.values_or_exprs[..][rhs]) {
                    (ValueOrIx::Index(lhs), ValueOrIx::Index(rhs)) => Ok(ValueOrIx::Index(
                        self.alloc_expr(thir::Expr::BinOp(op, lhs, rhs)),
                    )),
                    (ValueOrIx::Index(lhs), ValueOrIx::Value(rhs)) => {
                        let rhs = self.alloc_expr(rhs.into_expr()?);

                        Ok(ValueOrIx::Index(
                            self.alloc_expr(thir::Expr::BinOp(op, lhs, rhs)),
                        ))
                    }
                    (ValueOrIx::Value(lhs), ValueOrIx::Index(rhs)) => {
                        // this is the source of the bug.
                        // If the lhs is a value and the rhs is an expr,
                        // the lhs needs to be lowered into a runtime value.
                        // but this will make it happen after.
                        // Solution: recursively lower right again so it comes after?
                        // Use a filled option slice that we fill out during type checking?
                        // populate types as we push?
                        let lhs = self.alloc_expr(lhs.into_expr()?);

                        Ok(ValueOrIx::Index(
                            self.alloc_expr(thir::Expr::BinOp(op, lhs, rhs)),
                        ))
                    }
                    (ValueOrIx::Value(lhs), ValueOrIx::Value(rhs)) => match (op, lhs, rhs) {
                        (hir::BinOp::Add, Value::Int(lhs), Value::Int(rhs)) => {
                            Ok(ValueOrIx::Value(Value::Int(lhs + rhs)))
                        }
                        (hir::BinOp::Sub, Value::Int(lhs), Value::Int(rhs)) => {
                            Ok(ValueOrIx::Value(Value::Int(lhs - rhs)))
                        }
                        (hir::BinOp::Mul, Value::Int(lhs), Value::Int(rhs)) => {
                            Ok(ValueOrIx::Value(Value::Int(lhs * rhs)))
                        }
                        (hir::BinOp::Eq, Value::Int(lhs), Value::Int(rhs)) => {
                            Ok(ValueOrIx::Value(Value::Bool(lhs == rhs)))
                        }
                        (hir::BinOp::Eq, Value::Bool(lhs), Value::Bool(rhs)) => {
                            Ok(ValueOrIx::Value(Value::Bool(lhs == rhs)))
                        }
                        _ => Err(Error::BinOp),
                    },
                }
            }
            hir::Expr::IfThenElse(cond, then, else_) => {
                let cond = self.lower_expr_raw(cond, monomorphized_functions, in_comptime)?;
                match cond {
                    ValueOrIx::Value(value) => {
                        // If the condition is known at comptime, the correct branch is inlined
                        // and the other branch is forgotten.
                        if value.into_bool()? {
                            self.lower_expr_raw(then, monomorphized_functions, in_comptime)
                        } else {
                            self.lower_expr_raw(else_, monomorphized_functions, in_comptime)
                        }
                    }
                    ValueOrIx::Index(cond_index) => {
                        let value_or_then_index =
                            self.lower_expr_raw(then, monomorphized_functions, in_comptime)?;
                        let then_index = match value_or_then_index {
                            ValueOrIx::Value(value) => self.alloc_expr(value.into_expr()?),
                            ValueOrIx::Index(index) => index,
                        };
                        let value_or_else_index =
                            self.lower_expr_raw(else_, monomorphized_functions, in_comptime)?;
                        let else_index = match value_or_else_index {
                            ValueOrIx::Value(value) => self.alloc_expr(value.into_expr()?),
                            ValueOrIx::Index(index) => index,
                        };
                        Ok(ValueOrIx::Index(self.alloc_expr(thir::Expr::IfThenElse(
                            cond_index, then_index, else_index,
                        ))))
                    }
                }
            }
            hir::Expr::Name(name) => {
                match name {
                    hir::Name::Let(hir_let_index) => {
                        // if it needs to be known at comptime, then it better be a value
                        let value_or_let = self.values_or_lets[hir_let_index.index];

                        match value_or_let {
                            ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                            ValueOrIx::Index(runtime_let_index) => {
                                if in_comptime {
                                    Err(Error::LetExprIsNotComptimeKnown)
                                } else {
                                    Ok(ValueOrIx::Index(self.alloc_expr(thir::Expr::Name(
                                        thir::Name::Let(runtime_let_index),
                                    ))))
                                }
                            }
                        }
                    }
                    hir::Name::Parameter(param_index) => {
                        match self.values_or_params[param_index.index]
                            .expect("should have been lowered")
                        {
                            ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                            ValueOrIx::Index(runtime_param_index) => {
                                if in_comptime {
                                    Err(Error::ParamIsNotComptimeKnown)
                                } else {
                                    Ok(ValueOrIx::Index(self.alloc_expr(thir::Expr::Name(
                                        thir::Name::Parameter(runtime_param_index),
                                    ))))
                                }
                            }
                        }
                    }
                    hir::Name::Function(function_index) => {
                        Ok(ValueOrIx::Value(Value::Function(function_index)))
                    }
                    hir::Name::Builtin(builtin) => {
                        Ok(ValueOrIx::Value(Value::Type(Type::Builtin(builtin))))
                    }
                }
            }
            hir::Expr::Block(block_index) => {
                let value_or_block_index =
                    self.lower_block(block_index, monomorphized_functions, in_comptime)?;

                match value_or_block_index {
                    ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                    ValueOrIx::Index(block_index) => Ok(ValueOrIx::Index(
                        self.alloc_expr(thir::Expr::Block(block_index)),
                    )),
                }
            }
            hir::Expr::Call(callee_index, ref arguments) => {
                let callee_index =
                    self.lower_expr(callee_index, monomorphized_functions, in_comptime)?;
                let value_or_callee_index = self.values_or_exprs[..][callee_index];

                match value_or_callee_index {
                    ValueOrIx::Index(_callee_index) => {
                        todo!("indirect calls")
                    }
                    ValueOrIx::Value(callee_value) => {
                        let Value::Function(function_index) = callee_value else {
                            return Err(Error::NotCallable);
                        };

                        let params = self.functions[function_index].context.params.as_slice();

                        if params.len() != arguments.len() {
                            return Err(Error::WrongNumberOfArguments);
                        }

                        let mut runtime_args = vec![];

                        // get this slice without self attached, since self is mutably
                        // borrowed in the closure.
                        let functions = self.functions;

                        let lower_nth_argument =
                            |monomorphized_functions: &mut MonomorphizedFunctions,
                             index: usize,
                             in_comptime: bool|
                             -> Result<ValueOrIx<thir::Param>, Error> {
                                let lowered_expr_index = self.lower_expr(
                                    arguments[index],
                                    monomorphized_functions,
                                    in_comptime,
                                )?;
                                let lowered_expr = self.values_or_exprs[..][lowered_expr_index];

                                if in_comptime {
                                    Ok(ValueOrIx::Value(lowered_expr.into_value()?))
                                } else {
                                    let expr_index = match lowered_expr {
                                        ValueOrIx::Index(expr_index) => expr_index,
                                        ValueOrIx::Value(value) => {
                                            self.alloc_expr(value.into_expr()?)
                                        }
                                    };

                                    Ok(ValueOrIx::Index(
                                        Ix::push(&mut runtime_args, expr_index).map(),
                                    ))
                                }
                            };

                        let value_or_function_index = monomorphized_functions.lower_function(
                            functions,
                            function_index,
                            lower_nth_argument,
                            in_comptime,
                        )?;

                        match value_or_function_index {
                            ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                            ValueOrIx::Index(function_index) => {
                                let call_index = self.alloc_expr(thir::Expr::DirectCall(
                                    function_index,
                                    runtime_args,
                                ));

                                Ok(ValueOrIx::Index(call_index))
                            }
                        }
                    }
                }
            }
            hir::Expr::Comptime(inner) => {
                let in_comptime = true;
                self.lower_expr_raw(inner, monomorphized_functions, in_comptime)
            }
        }
    }
}
