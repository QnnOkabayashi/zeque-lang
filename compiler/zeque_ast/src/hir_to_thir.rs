use crate::util::Span;
use crate::{hir, thir, util::Ix};
use std::collections::hash_map::{Entry, HashMap};

mod state;
pub use state::ValueOrIx;
use state::{
    ComptimeStruct, ComptimeStructField, Error, InvocationLocation, StructKind, StructValue, Type,
    Value,
};
use string_interner::DefaultSymbol;

/// The entry into the `hir_to_thir` module.
///
/// Returns the lowered functions (lowered recursively based on usage by the main function),
/// as well as the index of the main function (or the value if it can be computed entirely at
/// comptime).
pub fn entry(
    functions: &[hir::Function],
    main_symbol: DefaultSymbol,
) -> Result<(thir::Context, ValueOrIx<thir::Function>), Error> {
    let main_index = Ix::new(
        functions
            .iter()
            .position(|function| function.name.0 == main_symbol)
            .ok_or(Error::NoMain)?,
    );

    let main = &functions[main_index];

    if !main.context.params.ranges().is_empty() {
        return Err(Error::MainHasParams(main.context.params.ranges().to_vec()));
    }

    let next_argument =
        |_: &mut MonomorphizedFunctions, _: usize, _: bool| unreachable!("main has no arguments");

    let mut monomorphized_functions = MonomorphizedFunctions::default();

    let in_comptime = false;
    let value_or_main = monomorphized_functions.lower_function(
        functions,
        main_index,
        next_argument,
        InvocationLocation::Main,
        in_comptime,
    )?;

    let functions = monomorphized_functions
        .functions
        .into_iter()
        .collect::<Option<Vec<thir::Function>>>()
        .expect("expected all functions to be filled in");

    let mut context = thir::Context {
        structs: monomorphized_functions.anytime_structs,
        functions,
        type_metadata: thir::TypeMetadata::default(),
    };

    thir::typeck::entry(&mut context)?;
    thir::local_offsets::entry(&mut context);

    Ok((context, value_or_main))
}

/// Represents a call to a function call at comptime.
/// These are stored in a hashmap during comptime evaluation to avoid
/// computing the same comptime function twice (we just memoize the result).
/// For example, this allows recursive fib to be linear at comptime.
#[derive(Clone, PartialEq, Eq, Hash)]
struct FunctionInvocation {
    index: Ix<hir::Function>,
    arguments: Box<[ValueOrIx<thir::Param>]>,
    return_type: Type,
}

#[derive(Copy, Clone)]
enum Lowering {
    Comptime(InvocationLocation),
    Runtime(Ix<thir::Function>),
}

enum MemoizedFunction {
    Lowering(Lowering),
    Lowered(ValueOrIx<thir::Function>),
}

// if the invocation always has the same value (and thus resolves to a ValueOrIx::Value),
// then calling it recursively is a compile error.
// If it doesn't, then point to the index of where it will be
#[derive(Default)]
struct MonomorphizedFunctions {
    memoized: HashMap<FunctionInvocation, MemoizedFunction>,
    functions: Vec<Option<thir::Function>>,
    anytime_structs: Vec<thir::Struct>,
    comptime_structs: Vec<ComptimeStruct>,
}

impl MonomorphizedFunctions {
    /// Lower a function call with the comptime arguments applied.
    /// If the result of monomorphization is just returning a single
    /// comptime-know value, then the value is returned.
    fn lower_function<F>(
        &mut self,
        functions: &[hir::Function],
        function_index: Ix<hir::Function>,
        mut lower_nth_argument: F,
        invocation_location: InvocationLocation,
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
            let param_type = state.values_or_exprs[param_type_expr_index]
                .clone()
                .into_value()?
                .into_type()?;

            // Pattern that I noticed: all the args for comptime params are
            // lowered first, with in_comptime=true.
            // Then, the args for runtime params are lowered, with a varying in_comptime.
            // After monomorphization, only runtime parameters exist.
            match param_type.into_runtime_type() {
                Some(ty) if !param.is_comptime => {
                    state.thir_context.params.push(thir::Param {
                        name: param.name,
                        ty,
                    });
                }
                _ => {
                    let argument = lower_nth_argument(self, index, true)?;

                    let arg_value = match argument.clone() {
                        ValueOrIx::Value(arg_value) => arg_value,
                        ValueOrIx::Index(_index) => {
                            let range = hir_function.context.params.ranges()[index];
                            return Err(Error::RuntimeExprPassedIntoComptimeParam(range));
                        }
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
        let return_type = state.values_or_exprs[return_type_expr_index]
            .as_value()?
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
                return match occupied.get() {
                    MemoizedFunction::Lowering(Lowering::Comptime(invocation_location)) => {
                        // want to space the invocation of the call
                        Err(Error::UnboundRecursionInComptimeFunction(
                            *invocation_location,
                        ))
                    }
                    MemoizedFunction::Lowering(Lowering::Runtime(index)) => {
                        Ok(ValueOrIx::Index(*index))
                    }
                    MemoizedFunction::Lowered(value_or_function) => Ok(value_or_function.clone()),
                };
            }
            Entry::Vacant(vacant) => {
                let lowering_or_index = if in_comptime {
                    Lowering::Comptime(invocation_location)
                } else {
                    Lowering::Runtime(Ix::push(&mut self.functions, None))
                };
                vacant.insert(MemoizedFunction::Lowering(lowering_or_index));
                lowering_or_index
            }
        };

        let value_or_block = state.lower_block(hir_function.body, self, in_comptime)?;

        // For function blocks, if the block was evaluated to a value but the function wasn't asked
        // to be generated at comptime, a runtime block expression is generated that just returns
        // the value.
        let value_or_block = match value_or_block {
            ValueOrIx::Value(value) if !in_comptime => {
                let expr = value.into_expr(&mut state.thir_context.exprs)?;
                let block = thir::Block {
                    stmts: vec![],
                    returns: state.alloc_expr(expr),
                };
                ValueOrIx::Index(Ix::push(&mut state.thir_context.blocks, block))
            }
            other => other,
        };

        // the value that the function will always return,
        // or an index to the runtime-lowered function
        let value_or_function = match value_or_block {
            ValueOrIx::Value(value) => ValueOrIx::Value(value),
            ValueOrIx::Index(block_index) => {
                let filled_args = arguments
                    .iter()
                    .map(|value_or_param| match value_or_param {
                        ValueOrIx::Value(value) => Some(format!("{value}")),
                        ValueOrIx::Index(_) => None,
                    })
                    .collect();

                let return_type = return_type.into_runtime_type().expect("apparently return type was comptime only but the return expression is only runtime known? that's a bug");

                let function = thir::Function {
                    name: hir_function.name,
                    filled_args,
                    return_type,
                    body: block_index,
                    context: state.thir_context,
                };

                let Lowering::Runtime(lowered_function_index) = lowered_function_index else {
                    panic!("tried lowering a comptime-only function but got a runtime value");
                };

                self.functions[lowered_function_index] = Some(function);

                ValueOrIx::Index(lowered_function_index)
            }
        };

        self.memoized
            .insert(key, MemoizedFunction::Lowered(value_or_function.clone()));

        Ok(value_or_function)
    }
}

/// Type for lowering an [`hir::Function`] into a [`thir::Function`]. Comptime expressions are
/// evaluated into values, and runtime expressions and lowered into thir instructions.
struct LoweringState<'hir, 'params> {
    functions: &'hir [hir::Function],
    hir_context: &'hir hir::FunctionContext,
    thir_context: thir::FunctionContext,

    /// The comptime value of a let, or an index into the thir_context of where the Let is.
    values_or_lets: Vec<ValueOrIx<thir::Let>>,
    /// The comptime value of an expr, or an index into the thir_context of where the Expr is.
    values_or_exprs: Vec<ValueOrIx<thir::Expr>>,
    values_or_params: &'params mut [Option<ValueOrIx<thir::Param>>],
}

impl<'hir, 'params> LoweringState<'hir, 'params> {
    fn lower_let(
        &mut self,
        let_index: Ix<hir::Let>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<Ix<ValueOrIx<thir::Let>>, Error> {
        let let_ = &self.hir_context.lets[let_index];
        let ty = let_
            .ty
            .map(|ty| {
                let in_comptime = true;
                let ty = self.lower_expr(ty, monomorphized_functions, in_comptime)?;
                self.values_or_exprs[ty].as_value()?.into_type()
            })
            .transpose()?;

        let value_or_expr = self.lower_expr_raw(let_.expr, monomorphized_functions, in_comptime)?;

        let value_or_let = match value_or_expr {
            ValueOrIx::Value(value) => ValueOrIx::Value(value),
            ValueOrIx::Index(expr) => {
                let let_ = thir::Let {
                    name: let_.name,
                    ty: ty.map(|t| t.into_runtime_type().expect("let expression has type whose values only exist at comptime, yet the expression is only known at runtime")),
                    expr,
                };
                let index = Ix::push(&mut self.thir_context.lets, let_);
                ValueOrIx::Index(index)
            }
        };

        Ok(Ix::push(&mut self.values_or_lets, value_or_let))
    }

    fn lower_block(
        &mut self,
        block_index: Ix<hir::Block>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Block>, Error> {
        let mut stmts = Vec::new();
        let function_body = &self.hir_context.blocks[block_index];
        for &stmt in &function_body.stmts {
            match stmt {
                hir::Stmt::Let(hir_let_index) => {
                    let value_or_let_index =
                        self.lower_let(hir_let_index, monomorphized_functions, in_comptime)?;

                    if let ValueOrIx::Index(let_index) = self.values_or_lets[value_or_let_index] {
                        stmts.push(thir::Stmt::Let(let_index));
                    }
                }
            }
        }

        let return_expr =
            self.lower_expr(function_body.returns, monomorphized_functions, in_comptime)?;

        match self.values_or_exprs[return_expr] {
            ValueOrIx::Value(ref value) => Ok(ValueOrIx::Value(value.clone())),
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
        match self.hir_context.exprs[expr_index] {
            hir::Expr::Int(int) => Ok(ValueOrIx::Value(Value::Int(int))),
            hir::Expr::Bool(boolean) => Ok(ValueOrIx::Value(Value::Bool(boolean))),
            hir::Expr::BinOp(op, lhs, rhs) => {
                self.lower_binop(op, lhs, rhs, monomorphized_functions, in_comptime)
            }
            hir::Expr::IfThenElse(cond, then, else_) => {
                self.lower_if_then_else(cond, then, else_, monomorphized_functions, in_comptime)
            }
            hir::Expr::Name(name) => self.lower_name(name, in_comptime),
            hir::Expr::Block(block_index) => {
                let value_or_block =
                    self.lower_block(block_index, monomorphized_functions, in_comptime)?;

                match value_or_block {
                    ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                    ValueOrIx::Index(block_index) => Ok(ValueOrIx::Index(
                        self.alloc_expr(thir::Expr::Block(block_index)),
                    )),
                }
            }
            hir::Expr::Call(callee, ref arguments) => {
                let invocation_location =
                    InvocationLocation::CallSite(self.hir_context.exprs.ranges()[expr_index.index]);

                match callee {
                    hir::Callee::Expr(callee) => self.lower_call(
                        callee,
                        arguments,
                        monomorphized_functions,
                        invocation_location,
                        in_comptime,
                    ),
                    hir::Callee::Builtin(builtin) => self.lower_builtin_call(
                        builtin,
                        arguments,
                        monomorphized_functions,
                        invocation_location,
                        in_comptime,
                    ),
                }
            }

            hir::Expr::Comptime(expr_index) => {
                let in_comptime = true;
                self.lower_expr_raw(expr_index, monomorphized_functions, in_comptime)
            }
            hir::Expr::Struct(struct_index) => {
                let struct_kind =
                    self.lower_struct(struct_index, monomorphized_functions, in_comptime)?;

                Ok(ValueOrIx::Value(Value::Type(Type::Struct(struct_kind))))
            }
            hir::Expr::Constructor(ctor_type, ref ctor_fields) => {
                // if the ctor type is present, evaluate it and make sure that it's
                // a custom struct.
                let ctor_type = ctor_type.ok_or(Error::UnimplementedAnonymousConstructor(
                    self.hir_context.exprs.ranges()[expr_index.index],
                ))?;

                let ctor_type = self
                    .lower_expr_raw(ctor_type, monomorphized_functions, in_comptime)?
                    .into_struct_type()?;

                // if it's a comptime struct, make sure that all field values are comptime known.
                // This is because if any field has a comptime-only type, then all fields
                // must be knowable at comptime.
                match ctor_type {
                    StructKind::Comptime(_) => {
                        // Instances of this struct can only exist at comptime, since it contains at
                        // least one field with a type whose values can only exist at comptime.
                        Err(Error::UnimplementedComptimeOnlyStructs(
                            self.hir_context.exprs.ranges()[expr_index.index],
                        ))
                        // let struct_ = &monomorphized_functions.comptime_structs[index];
                        // for struct_fields in &struct_.fields {
                        //     todo!()
                        // }
                        // todo!()
                    }
                    StructKind::Anytime(index) => {
                        // Instances of this struct can exist at both comptime and runtime,
                        // since it doesn't contain any fields with a type whose values can only
                        // exist at comptime.
                        let struct_ = &monomorphized_functions.anytime_structs[index];
                        let mut unused_struct_fields: HashMap<DefaultSymbol, thir::Type> = struct_
                            .fields
                            .iter()
                            .map(|struct_field| (struct_field.name.0, struct_field.ty))
                            .collect();

                        let field_types_and_values_or_indices: Vec<(
                            Span<DefaultSymbol>,
                            thir::Type,
                            ValueOrIx<thir::Expr>,
                        )> = ctor_fields
                            .iter()
                            .map(|ctor_field| {
                                let Span(symbol, range) = ctor_field.name;
                                let field_type = unused_struct_fields
                                    .remove(&symbol)
                                    .ok_or_else(|| Error::FieldNotFound(symbol, range))?;

                                let field_value = self.lower_expr_raw(
                                    ctor_field.value,
                                    monomorphized_functions,
                                    in_comptime,
                                )?;

                                Ok((ctor_field.name, field_type, field_value))
                            })
                            .collect::<Result<_, Error>>()?;

                        if !unused_struct_fields.is_empty() {
                            return Err(Error::MissingFieldsInCtor(
                                unused_struct_fields.into_keys().collect(),
                            ));
                        }

                        let maybe_field_names_and_comptime_values: Option<
                            Vec<(Span<DefaultSymbol>, Value)>,
                        > = field_types_and_values_or_indices
                            .iter()
                            .map(|(field_name, field_type, field_value)| match field_value {
                                ValueOrIx::Value(value) => {
                                    value.check_type(Type::from_thir_type(*field_type))?;
                                    Ok(Some((*field_name, value.clone())))
                                }
                                ValueOrIx::Index(_) => Ok(None),
                            })
                            .collect::<Result<_, Error>>()?;

                        if let Some(field_types_and_values) = maybe_field_names_and_comptime_values
                        {
                            // All field values are comptime known, so this constructed value is
                            // comptime known.
                            Ok(ValueOrIx::Value(Value::StructValue(StructValue {
                                ty: StructKind::Anytime(index),
                                fields: field_types_and_values,
                            })))
                        } else {
                            // Not all field values are comptime known, so this constructed value
                            // can only be known at runtime.
                            let fields: Vec<thir::ConstructorField> =
                                field_types_and_values_or_indices
                                    .into_iter()
                                    .map(|(name, _field_type, field_value)| {
                                        let expr = match field_value {
                                            ValueOrIx::Value(value) => {
                                                let expr = value
                                                    .into_expr(&mut self.thir_context.exprs)?;
                                                self.alloc_expr(expr)
                                            }
                                            ValueOrIx::Index(index) => index,
                                        };

                                        Ok(thir::ConstructorField { name, expr })
                                    })
                                    .collect::<Result<_, Error>>()?;

                            let construct =
                                thir::Expr::Constructor(index, thir::Constructor { fields });

                            Ok(ValueOrIx::Index(self.alloc_expr(construct)))
                        }
                    }
                }
            }
            hir::Expr::Field(value, field) => {
                let value_or_expr =
                    self.lower_expr_raw(value, monomorphized_functions, in_comptime)?;
                match value_or_expr {
                    ValueOrIx::Value(value) => {
                        let Value::StructValue(struct_value) = value else {
                            return Err(Error::ExpectedStructFoundOtherValue(format!("{value}")));
                        };

                        let field_value = struct_value
                            .fields
                            .iter()
                            .find_map(|(Span(name, _), value)| (*name == field.0).then_some(value))
                            .ok_or_else(|| Error::FieldDoesNotExist(field.range()))?
                            .clone();

                        Ok(ValueOrIx::Value(field_value))
                    }
                    ValueOrIx::Index(index) => {
                        let expr = thir::Expr::Field(index, field);

                        Ok(ValueOrIx::Index(self.alloc_expr(expr)))
                    }
                }
            }
            hir::Expr::Error => Err(Error::ErrorInHir),
        }
    }

    fn lower_binop(
        &mut self,
        op: hir::BinOpKind,
        lhs: Ix<hir::Expr>,
        rhs: Ix<hir::Expr>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Expr>, Error> {
        let lhs = self.lower_expr(lhs, monomorphized_functions, in_comptime)?;
        let rhs = self.lower_expr(rhs, monomorphized_functions, in_comptime)?;
        match (
            self.values_or_exprs[lhs].clone(),
            self.values_or_exprs[rhs].clone(),
        ) {
            (ValueOrIx::Index(lhs), ValueOrIx::Index(rhs)) => Ok(ValueOrIx::Index(
                self.alloc_expr(thir::Expr::BinOp(op, lhs, rhs)),
            )),
            (ValueOrIx::Index(lhs), ValueOrIx::Value(rhs)) => {
                let expr = rhs.into_expr(&mut self.thir_context.exprs)?;
                let rhs = self.alloc_expr(expr);

                Ok(ValueOrIx::Index(
                    self.alloc_expr(thir::Expr::BinOp(op, lhs, rhs)),
                ))
            }
            (ValueOrIx::Value(lhs), ValueOrIx::Index(rhs)) => {
                let expr = lhs.into_expr(&mut self.thir_context.exprs)?;
                let lhs = self.alloc_expr(expr);

                Ok(ValueOrIx::Index(
                    self.alloc_expr(thir::Expr::BinOp(op, lhs, rhs)),
                ))
            }
            (ValueOrIx::Value(lhs), ValueOrIx::Value(rhs)) => match (op, lhs, rhs) {
                (hir::BinOpKind::Add, Value::Int(lhs), Value::Int(rhs)) => {
                    Ok(ValueOrIx::Value(Value::Int(lhs + rhs)))
                }
                (hir::BinOpKind::Sub, Value::Int(lhs), Value::Int(rhs)) => {
                    Ok(ValueOrIx::Value(Value::Int(lhs - rhs)))
                }
                (hir::BinOpKind::Mul, Value::Int(lhs), Value::Int(rhs)) => {
                    Ok(ValueOrIx::Value(Value::Int(lhs * rhs)))
                }
                (hir::BinOpKind::Eq, Value::Int(lhs), Value::Int(rhs)) => {
                    Ok(ValueOrIx::Value(Value::Bool(lhs == rhs)))
                }
                (hir::BinOpKind::Eq, Value::Bool(lhs), Value::Bool(rhs)) => {
                    Ok(ValueOrIx::Value(Value::Bool(lhs == rhs)))
                }
                _ => Err(Error::BinOp),
            },
        }
    }

    fn lower_if_then_else(
        &mut self,
        cond: Ix<hir::Expr>,
        then: Ix<hir::Expr>,
        else_: Ix<hir::Expr>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Expr>, Error> {
        let cond = self.lower_expr_raw(cond, monomorphized_functions, in_comptime)?;
        match cond {
            ValueOrIx::Value(cond) => {
                // The condition is known at comptime.
                // Let's inline the correct branch.
                if cond.into_bool()? {
                    self.lower_expr_raw(then, monomorphized_functions, in_comptime)
                } else {
                    self.lower_expr_raw(else_, monomorphized_functions, in_comptime)
                }
            }
            ValueOrIx::Index(cond_index) => {
                let value_or_then =
                    self.lower_expr_raw(then, monomorphized_functions, in_comptime)?;
                let then_index = match value_or_then {
                    ValueOrIx::Value(value) => {
                        let expr = value.into_expr(&mut self.thir_context.exprs)?;
                        self.alloc_expr(expr)
                    }
                    ValueOrIx::Index(index) => index,
                };

                let value_or_else =
                    self.lower_expr_raw(else_, monomorphized_functions, in_comptime)?;
                let else_index = match value_or_else {
                    ValueOrIx::Value(value) => {
                        let expr = value.into_expr(&mut self.thir_context.exprs)?;
                        self.alloc_expr(expr)
                    }
                    ValueOrIx::Index(index) => index,
                };

                Ok(ValueOrIx::Index(self.alloc_expr(thir::Expr::IfThenElse(
                    cond_index, then_index, else_index,
                ))))
            }
        }
    }

    fn lower_name(
        &mut self,
        name: hir::Name,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Expr>, Error> {
        match name {
            hir::Name::Let(hir_let_index) => {
                match self.values_or_lets[hir_let_index.index].clone() {
                    ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                    ValueOrIx::Index(runtime_let_index) => {
                        if in_comptime {
                            Err(Error::LetExprIsNotComptimeKnown)
                        } else {
                            let expr = thir::Expr::Name(thir::Name::Let(runtime_let_index));

                            Ok(ValueOrIx::Index(self.alloc_expr(expr)))
                        }
                    }
                }
            }
            hir::Name::Param(param_index) => {
                match self.values_or_params[param_index.index]
                    .clone()
                    .expect("should have been lowered")
                {
                    ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                    ValueOrIx::Index(runtime_param_index) => {
                        if in_comptime {
                            Err(Error::ParamIsNotComptimeKnown)
                        } else {
                            let expr = thir::Expr::Name(thir::Name::Parameter(runtime_param_index));

                            Ok(ValueOrIx::Index(self.alloc_expr(expr)))
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
    fn lower_call(
        &mut self,
        callee_index: Ix<hir::Expr>,
        arguments: &[Ix<hir::Expr>],
        monomorphized_functions: &mut MonomorphizedFunctions,
        invocation_location: InvocationLocation,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Expr>, Error> {
        let callee_index = self.lower_expr(callee_index, monomorphized_functions, in_comptime)?;
        let value_or_callee = self.values_or_exprs[callee_index].clone();

        match value_or_callee {
            ValueOrIx::Index(_callee_index) => {
                todo!("indirect calls")
            }
            ValueOrIx::Value(callee_value) => {
                let Value::Function(function_index) = callee_value else {
                    return Err(Error::NotCallable);
                };

                let params = &*self.functions[function_index].context.params;

                if params.len() != arguments.len() {
                    return Err(Error::WrongNumberOfArguments);
                }

                let mut runtime_args = vec![];

                // get this slice without self attached, since self is mutably
                // borrowed in the closure.
                let functions = self.functions;

                let lower_nth_argument = |monomorphized_functions: &mut MonomorphizedFunctions,
                                          index: usize,
                                          in_comptime: bool|
                 -> Result<ValueOrIx<thir::Param>, Error> {
                    let lowered_expr_index =
                        self.lower_expr(arguments[index], monomorphized_functions, in_comptime)?;
                    let lowered_expr = self.values_or_exprs[lowered_expr_index].clone();

                    if in_comptime {
                        Ok(ValueOrIx::Value(lowered_expr.into_value()?))
                    } else {
                        // not in comptime
                        let expr_index = match lowered_expr {
                            ValueOrIx::Index(expr_index) => expr_index,
                            ValueOrIx::Value(value) => {
                                let expr = value.into_expr(&mut self.thir_context.exprs)?;
                                self.alloc_expr(expr)
                            }
                        };

                        Ok(ValueOrIx::Index(
                            // this is stupid, please fix
                            Ix::<Ix<thir::Expr>>::push(&mut runtime_args, expr_index).map(),
                        ))
                    }
                };

                let value_or_function = monomorphized_functions.lower_function(
                    functions,
                    function_index,
                    lower_nth_argument,
                    invocation_location,
                    in_comptime,
                )?;

                match value_or_function {
                    ValueOrIx::Value(value) => Ok(ValueOrIx::Value(value)),
                    ValueOrIx::Index(function_index) => {
                        let call_index =
                            self.alloc_expr(thir::Expr::DirectCall(function_index, runtime_args));

                        Ok(ValueOrIx::Index(call_index))
                    }
                }
            }
        }
    }

    fn lower_builtin_call(
        &mut self,
        Span(builtin, range): Span<hir::BuiltinFunction>,
        arguments: &[Ix<hir::Expr>],
        monomorphized_functions: &mut MonomorphizedFunctions,
        _invocation_location: InvocationLocation,
        in_comptime: bool,
    ) -> Result<ValueOrIx<thir::Expr>, Error> {
        // Make sure to check the arguments!
        match builtin {
            hir::BuiltinFunction::InComptime => {
                if !arguments.is_empty() {
                    return Err(Error::WrongNumberOfArguments);
                }
                Ok(ValueOrIx::Value(Value::Bool(in_comptime)))
            }
            hir::BuiltinFunction::Trap => {
                if !arguments.is_empty() {
                    return Err(Error::WrongNumberOfArguments);
                }
                if in_comptime {
                    // if we reached a trap in comptime, halt the compilation
                    Err(Error::ReachedTrapInComptime(range))
                } else {
                    Ok(ValueOrIx::Index(self.alloc_expr(thir::Expr::Trap(range))))
                }
            }
            hir::BuiltinFunction::Clz => {
                let [arg] = arguments else {
                    return Err(Error::WrongNumberOfArguments);
                };
                let arg = self.lower_expr_raw(*arg, monomorphized_functions, in_comptime)?;
                match arg {
                    ValueOrIx::Value(value) => match value {
                        Value::Int(int) => {
                            let leading_zeros = int.leading_zeros() as i32;
                            Ok(ValueOrIx::Value(Value::Int(leading_zeros)))
                        }
                        _ => Err(Error::NonIntInClz),
                    },
                    ValueOrIx::Index(expr) => {
                        let expr = thir::Expr::UnOp(thir::UnOp::Clz, expr);
                        Ok(ValueOrIx::Index(self.alloc_expr(expr)))
                    }
                }
            }
            hir::BuiltinFunction::Ctz => {
                // mostly copy pasted from above
                let [arg] = arguments else {
                    return Err(Error::WrongNumberOfArguments);
                };
                let arg = self.lower_expr_raw(*arg, monomorphized_functions, in_comptime)?;
                match arg {
                    ValueOrIx::Value(value) => match value {
                        Value::Int(int) => {
                            let leading_zeros = int.trailing_zeros() as i32;
                            Ok(ValueOrIx::Value(Value::Int(leading_zeros)))
                        }
                        _ => Err(Error::NonIntInCtz),
                    },
                    ValueOrIx::Index(expr) => {
                        let expr = thir::Expr::UnOp(thir::UnOp::Ctz, expr);
                        Ok(ValueOrIx::Index(self.alloc_expr(expr)))
                    }
                }
            }
        }
    }

    /// Lowers a struct expression, not a constructor
    fn lower_struct(
        &mut self,
        struct_index: Ix<hir::Struct>,
        monomorphized_functions: &mut MonomorphizedFunctions,
        in_comptime: bool,
    ) -> Result<StructKind, Error> {
        let struct_ = &self.hir_context.structs[struct_index];

        let mut comptime_struct_fields = Vec::with_capacity(struct_.fields.len());
        for struct_item in &struct_.fields {
            match struct_item {
                hir::StructItem::Field(struct_field) => {
                    let ty = self
                        .lower_expr_raw(struct_field.value, monomorphized_functions, in_comptime)?
                        .into_value()?
                        .into_type()?;

                    comptime_struct_fields.push(ComptimeStructField {
                        name: struct_field.name,
                        ty,
                    });
                }
            }
        }

        let anytime_struct_fields = comptime_struct_fields
            .clone()
            .into_iter()
            .map(|struct_field| {
                Some(thir::StructField {
                    name: struct_field.name,
                    ty: struct_field.ty.into_runtime_type()?,
                })
            })
            .collect::<Option<Vec<thir::StructField>>>();

        let struct_kind = if let Some(fields) = anytime_struct_fields {
            let struct_sizealign = thir::sizealign::StructSizeAlign::from_fields(
                &fields,
                &monomorphized_functions.anytime_structs,
                &mut thir::sizealign::Repr::C,
            )?;
            let index = Ix::push(
                &mut monomorphized_functions.anytime_structs,
                thir::Struct {
                    fields,
                    struct_sizealign,
                },
            );
            StructKind::Anytime(index)
        } else {
            let index = Ix::push(
                &mut monomorphized_functions.comptime_structs,
                ComptimeStruct {
                    fields: comptime_struct_fields,
                },
            );
            StructKind::Comptime(index)
        };

        Ok(struct_kind)
    }
}
