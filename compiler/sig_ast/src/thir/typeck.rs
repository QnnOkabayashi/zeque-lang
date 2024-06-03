use crate::thir::{BinOp, Builtin, Context, Expr, Function, Name, Type, UnOp};
use crate::util::Ix;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("type error")]
    TypeError,
    #[error("if condition must be a boolean")]
    IfConditionMustBeBool,
    #[error("if then else expression has incompatible types")]
    IfThenElseHasIncompatibleTypes,
    #[error("attempting to access a field on a non-struct type")]
    FieldOnNonStructType,
    #[error("nonexistant field")]
    NonexistantField,
}

/// This is the entrance to type checking.
///
/// Right now, it pretty much just goes through each expression in the vec storing
/// all expressions of each function and says "figure out your type". There is no
/// tree structure traversal right now because there's no need.
pub fn entry(context: &mut Context) -> Result<(), Error> {
    for function_index in Ix::iter(&context.functions) {
        for expr_index in Ix::iter(&context.functions[function_index].context.exprs) {
            let ty = type_of_expr(
                function_index,
                expr_index,
                context,
                &context.functions[function_index].context.expr_types,
            )?;
            context.functions[function_index]
                .context
                .expr_types
                .push(ty);
        }
    }
    Ok(())
}

/// Returns the type of an expression.
fn type_of_expr(
    function_index: Ix<Function>,
    expr_index: Ix<Expr>,
    context: &Context,
    types: &[Type],
) -> Result<Type, Error> {
    let function = &context.functions[function_index];
    match function.context.exprs[expr_index] {
        Expr::Int(_) => Ok(Type::Builtin(Builtin::I32)),
        Expr::Bool(_) => Ok(Type::Builtin(Builtin::Bool)),
        Expr::UnOp(op, _arg) => match op {
            UnOp::Clz => Ok(Type::Builtin(Builtin::I32)),
            UnOp::Ctz => Ok(Type::Builtin(Builtin::I32)),
        },
        Expr::BinOp(op, lhs, rhs) => type_of_binop(op, lhs, rhs, &function.context.expr_types),
        Expr::IfThenElse(cond, then, else_) => {
            if !matches!(types[cond.index], Type::Builtin(Builtin::Bool)) {
                return Err(Error::IfConditionMustBeBool);
            }
            let then_type = types[then.index];
            let else_type = types[else_.index];
            if then_type != else_type {
                return Err(Error::IfThenElseHasIncompatibleTypes);
            }

            Ok(then_type)
        }
        Expr::Name(name) => match name {
            Name::Let(let_index) => {
                let let_ = &function.context.lets[let_index];
                let ty = function.context.expr_types[let_.expr.index];

                if let Some(ty_ascription) = let_.ty {
                    if ty != ty_ascription {
                        return Err(Error::TypeError);
                    }
                }

                Ok(ty)
            }
            Name::Parameter(param_index) => Ok(function.context.params[param_index].ty),
            Name::Function(function_index) => Ok(Type::Function(function_index)),
        },
        Expr::Block(block_index) => {
            let returns_index = function.context.blocks[block_index].returns;
            Ok(types[returns_index.index])
        }
        Expr::DirectCall(callee_index, ref arguments) => {
            type_of_call(function_index, callee_index, arguments, context, types)
        }
        Expr::IndirectCall(callee_expr_index, ref arguments) => {
            match function.context.expr_types[callee_expr_index.index] {
                Type::Function(callee_index) => {
                    type_of_call(function_index, callee_index, arguments, context, types)
                }
                Type::NoReturn => Ok(Type::NoReturn),
                Type::Builtin(_) | Type::Struct(_) => Err(Error::TypeError),
            }
        }
        Expr::Constructor(ctor_type, _) => Ok(Type::Struct(ctor_type)),
        Expr::Field(value, ref field_name) => {
            let Type::Struct(index) = types[value.index] else {
                return Err(Error::FieldOnNonStructType);
            };

            context.structs[index]
                .fields
                .iter()
                .find_map(|field| (field.name.0 == field_name.0).then_some(field.ty))
                .ok_or(Error::NonexistantField)
        }
        Expr::Trap(_) => Ok(Type::NoReturn),
    }
}

/// Returns the type of a function call.
fn type_of_call(
    caller_index: Ix<Function>,
    callee_index: Ix<Function>,
    arguments: &[Ix<Expr>],
    context: &Context,
    types: &[Type],
) -> Result<Type, Error> {
    let callee = &context.functions[callee_index];

    // Function is being called with correct number of arguments
    if callee.context.params.len() != arguments.len() {
        return Err(Error::TypeError);
    }

    // Function parameter types match argument types
    for (param, arg) in callee.context.params.iter().zip(arguments) {
        type_of_expr(caller_index, *arg, context, types)?;
        if param.ty != types[arg.index] {
            return Err(Error::TypeError);
        }
    }

    Ok(callee.return_type)
}

/// Returns the type of a binary operation.
fn type_of_binop(op: BinOp, lhs: Ix<Expr>, rhs: Ix<Expr>, types: &[Type]) -> Result<Type, Error> {
    match (op, types[lhs.index], types[rhs.index]) {
        (BinOp::Add, Type::Builtin(Builtin::I32), Type::Builtin(Builtin::I32)) => {
            Ok(Type::Builtin(Builtin::I32))
        }
        (BinOp::Sub, Type::Builtin(Builtin::I32), Type::Builtin(Builtin::I32)) => {
            Ok(Type::Builtin(Builtin::I32))
        }
        (BinOp::Mul, Type::Builtin(Builtin::I32), Type::Builtin(Builtin::I32)) => {
            Ok(Type::Builtin(Builtin::I32))
        }
        (BinOp::Eq, Type::Builtin(Builtin::I32), Type::Builtin(Builtin::I32)) => {
            Ok(Type::Builtin(Builtin::Bool))
        }
        (BinOp::Eq, Type::Builtin(Builtin::Bool), Type::Builtin(Builtin::Bool)) => {
            Ok(Type::Builtin(Builtin::Bool))
        }
        _ => Err(Error::TypeError),
    }
}
