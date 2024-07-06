//! A mapping from HIR -> HIR that performs all comptime computations.
//! This changes the HIR in place
use crate::hir::{
    Arg, AssociatedLet, BinOp, Block, BuiltinFunction, BuiltinType, Callee, DeclKind, Expr,
    Function, Hir, Let, Level, Name, NameKind, Param, Stmt, Storage, StorageKind, Struct, Value,
};
use crate::util::{Ix, Range, Span};
use std::collections::HashMap;
use string_interner::DefaultSymbol;

pub enum Error {
    Cycle,
    Binop,
    NoDecl,
    Call,
    TrapAtComptime(Range),
    NoField,
    NotAType,
    ValueUnknownAtComptime,
    NonBoolInComptimeConditional(Range),
}

#[derive(Copy, Clone, Debug)]
pub enum Type {
    Builtin(BuiltinType),
    Struct(Ix<Storage>, Ix<Struct>),
}

impl Type {
    fn is_comptime_only(self, evaluator: &mut Evaluator<'_>) -> bool {
        match self {
            Type::Builtin(builtin_type) => builtin_type.is_comptime_only(),
            Type::Struct(storage_index, struct_index) => {
                let struct_ =
                    &evaluator.with_storage(storage_index).storage().structs[struct_index];

                // I feel like we're overcomplicating this.
                // We should just be passing in the arguments as they are. If they're
                // not simplified, then clearly they weren't comptime known.
                todo!()
            }
        }
    }
}

pub fn entry(hir: &mut Hir, storage: Ix<Storage>, main_index: Ix<Struct>) -> Result<(), Error> {
    let in_comptime = true;

    ProgramEvaluator::new(hir)
        .make_evaluator(storage, in_comptime)
        .eval_struct(main_index)
}

struct ProgramEvaluator<'hir> {
    hir: &'hir mut Hir,
    level_stack: Vec<LevelFrame>,
    storages_data: Vec<StorageProgress>,
}

impl<'hir> ProgramEvaluator<'hir> {
    fn new(hir: &'hir mut Hir) -> Self {
        let storages_data = hir
            .storages
            .iter()
            .map(|storage| StorageProgress {
                expr_progress: vec![Progress::Unevaluated; storage.exprs.len()],
            })
            .collect();

        ProgramEvaluator { hir, storages_data }
    }

    fn make_evaluator(&mut self, storage_index: Ix<Storage>, in_comptime: bool) -> Evaluator<'_> {
        Evaluator {
            program_evaluator: self,
            storage_index,
            in_comptime,
        }
    }
}

struct LevelStack {
    stack: Vec<LevelFrame>,
}

impl LevelStack {
    fn push_frame(&mut self) {
        self.stack.push(LevelFrame::default());
    }

    fn pop_frame(&mut self) {
        self.stack.pop();
    }

    fn lookup(&self, level: u32, name: DefaultSymbol) -> Ix<Expr> {
        self.stack[level as usize].named_values[&name]
    }
}

#[derive(Clone, Debug, Default)]
struct LevelFrame {
    named_values: HashMap<DefaultSymbol, Ix<Expr>>,
}

#[derive(Copy, Clone)]
pub enum Progress {
    Unevaluated,
    Started,
    Evaluated,
}

// each one mirrors an hir::Storage
pub struct StorageProgress {
    expr_progress: Vec<Progress>,
}

struct Evaluator<'hir> {
    program_evaluator: &'hir mut ProgramEvaluator<'hir>,
    storage_index: Ix<Storage>,
    in_comptime: bool,
}

type EvaluatedExpr = Option<Expr>;

impl<'hir> Evaluator<'hir> {
    fn comptime(&mut self, in_comptime: bool) -> Evaluator<'_> {
        Evaluator {
            program_evaluator: self.program_evaluator,
            storage_index: self.storage_index,
            in_comptime,
        }
    }

    fn storage_progress(&mut self) -> &StorageProgress {
        &self.program_evaluator.storages_data[self.storage_index.index]
    }

    fn storage_progress_mut(&mut self) -> &mut StorageProgress {
        &mut self.program_evaluator.storages_data[self.storage_index.index]
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.program_evaluator.hir.storages[self.storage_index]
    }

    fn eval_struct(&mut self, struct_index: Ix<Struct>) -> Result<(), Error> {
        for struct_field_def in &self.storage().structs[struct_index].struct_field_defs {
            if struct_field_def.vis.is_public() {
                // eval all the types of struct fields
                self.comptime(true)
                    .eval_expr_and_inline(struct_field_def.struct_field.value)?;
            }
        }

        for (name, decl) in &self.storage().structs[struct_index].name_to_decl {
            match decl {
                DeclKind::AssociatedLet(associated_let_index) => todo!(),
                DeclKind::Function(function_index) => todo!(),
            }
        }

        Ok(())
    }

    fn with_storage(&mut self, storage_index: Ix<Storage>) -> Evaluator<'_> {
        Evaluator {
            program_evaluator: self.program_evaluator,
            storage_index,
            in_comptime: self.in_comptime,
        }
    }

    // Only use this when a Name::Decl is encountered, then use it on the decl's level.
    fn storage_at_level(&mut self, level: Level) -> Evaluator<'_> {
        let mut storage_index = self.storage_index;
        while level != self.program_evaluator.hir.storages[storage_index].level {
            let storage = &self.program_evaluator.hir.storages[storage_index];
            storage_index = match storage.kind {
                StorageKind::SourceFile => {
                    panic!("hir structure was incorrect or this function was called incorrectly")
                }
                StorageKind::FunctionBody { backref } => backref,
                StorageKind::FunctionParam { backref, arg } => {
                    assert!(arg.is_some(), "not monomorphized?");
                    backref
                }
            }
        }

        self.with_storage(storage_index)
    }

    /// Used for getting the actual expr, in particular for when we want to see if an expression
    /// evaluates to a type.
    fn eval_or_get_expr(&mut self, index: Ix<Expr>) -> Result<Expr, Error> {
        self.eval_expr_if_not_evaluated(index)
            .map(|update| update.unwrap_or_else(|| self.storage().exprs[index]))
    }

    // /// Returns whether or not an expression is comptime known.
    // /// None is returned if the expression hasn't been evaluated yet.
    // fn is_comptime_known(&self, index: Ix<Expr>) -> Option<ComptimeKnown> {
    //     match self.storage_progress_mut().expr_progress[index.index] {
    //         Progress::Evaluated(comptime_known) => Some(comptime_known),
    //         _ => None,
    //     }
    // }

    fn was_evaluated(&self, expr_index: Ix<Expr>) -> bool {
        matches!(
            self.storage_progress().expr_progress[expr_index.index],
            Progress::Evaluated
        )
    }

    fn eval_expr_and_inline(&mut self, index: Ix<Expr>) -> Result<(), Error> {
        if let Some(update) = self.eval_expr_if_not_evaluated(index)? {
            self.storage().exprs[index] = update;
        }

        if self.in_comptime && !self.storage().exprs[index].is_value() {
            return Err(Error::ValueUnknownAtComptime);
        }

        Ok(())
    }

    fn eval_expr_if_not_evaluated(&mut self, index: Ix<Expr>) -> Result<EvaluatedExpr, Error> {
        if self.was_evaluated(index) {
            return Ok(None);
        }

        self.storage_progress_mut().expr_progress[index.index] = Progress::Started;
        let update = self.eval_expr_without_cycle_detection(index)?;
        self.storage_progress_mut().expr_progress[index.index] = Progress::Evaluated;
        Ok(update)
    }

    fn eval_expr_without_cycle_detection(
        &mut self,
        index: Ix<Expr>,
    ) -> Result<EvaluatedExpr, Error> {
        match self.storage().exprs[index] {
            Expr::Value(value) => {
                if let Value::Struct(struct_index) = value {
                    self.eval_struct(struct_index)?;
                }
                Ok(None)
            }
            Expr::BinOp(op, lhs, rhs) => {
                use {crate::hir::Value::*, BinOp::*, Expr::Value};
                self.eval_expr_and_inline(lhs)?;
                self.eval_expr_and_inline(rhs)?;

                let value = match (&self.storage().exprs[lhs], &self.storage().exprs[rhs]) {
                    (Value(Int(lhs)), Value(Int(rhs))) => match op {
                        Add => Int(lhs + rhs),
                        Sub => Int(lhs - rhs),
                        Mul => Int(lhs * rhs),
                        Eq => Bool(lhs == rhs),
                    },
                    (Value(Bool(lhs)), Value(Bool(rhs))) => match op {
                        Eq => Bool(lhs == rhs),
                        _ => return Err(Error::Binop),
                    },
                    (Value(_), Value(_)) => {
                        // everything is a primitive,
                        // but we can't work with them
                        return Err(Error::Binop);
                    }
                    _ => {
                        return if self.in_comptime {
                            // We're in comptime but we can't add the values
                            Err(Error::ValueUnknownAtComptime)
                        } else {
                            // We can't add the values in comptime,
                            // but we're not in comptime so it's okay.
                            Ok(None)
                        };
                    }
                };
                Ok(Some(Expr::Value(value)))
            }
            Expr::IfThenElse(cond, then_expr, else_expr) => {
                self.eval_expr_and_inline(cond)?;

                let Span(cond_expr, cond_range) = self.storage().exprs.span(cond);

                if let Expr::Value(Value::Bool(yes)) = cond_expr {
                    let inlined = if *yes {
                        self.eval_expr_and_inline(then_expr)?;
                        then_expr
                    } else {
                        self.eval_expr_and_inline(else_expr)?;
                        else_expr
                    };
                    Ok(Some(self.storage().exprs[inlined]))
                } else if self.in_comptime {
                    // We wanted comptime but we didn't get it
                    Err(Error::NonBoolInComptimeConditional(cond_range))
                } else {
                    // Evaluate them both (not in comptime)
                    self.eval_expr_and_inline(then_expr)?;
                    self.eval_expr_and_inline(else_expr)?;
                    Ok(None)
                }
            }
            Expr::Name(name) => {
                let mut name_evaluator = self.storage_at_level(name.level);
                match name.kind {
                    NameKind::Decl(decl) => match decl {
                        DeclKind::AssociatedLet(associated_let_index) => {
                            let expr_index =
                                name_evaluator.eval_associated_let(associated_let_index)?;

                            Ok(Some(name_evaluator.storage().exprs[expr_index]))
                        }
                        DeclKind::Function(_) => Ok(None),
                    },
                    NameKind::Let(let_index) => {
                        name_evaluator.eval_let(let_index)?;
                        Ok(None)
                    }
                    NameKind::Param => {
                        // if it's known at comptime, then inline it
                        let arg = name_evaluator.storage().unwrap_arg();

                        if self.in_comptime {
                            assert!(
                                arg.comptime_known_value.is_some(),
                                "expected that we're not in comptime since a parameter is unknown"
                            );
                        }

                        Ok(arg.comptime_known_value.map(Expr::Value))
                    }
                }
            }
            Expr::Block(block_index) => {
                let maybe_value = self.eval_block(block_index)?;

                Ok(maybe_value.map(Expr::Value))
            }
            Expr::Call(callee, ref args) => self.eval_call(callee, args),
            Expr::Comptime(comptime_index) => {
                self.comptime(true).eval_expr_and_inline(comptime_index)?;

                Ok(Some(self.storage().exprs[comptime_index]))
            }
            Expr::Constructor(ctor, ref fields) => {
                if let Some(ctor) = ctor {
                    self.comptime(true).eval_expr_and_inline(ctor)?;
                }
                let mut all_values = true;
                for struct_field in fields {
                    self.eval_expr_and_inline(struct_field.value)?;
                    // todo: if they're all values, turn the ctor into a value
                    // otherwise, leave it as an expr.
                    all_values &= self.storage().exprs[struct_field.value].is_value();
                }

                if !all_values {
                    return Ok(None);
                }

                // let value = Value::Constructor()

                // Ok(Some(Expr::Value(value)))
                todo!()
            }
            Expr::Field(value_index, Span(field_name, field_range)) => {
                self.eval_expr_and_inline(value_index)?;

                if !self.storage().exprs[value_index].is_value() {
                    return if self.in_comptime {
                        Err(Error::ValueUnknownAtComptime)
                    } else {
                        Ok(None)
                    };
                }

                match self.storage().exprs[value_index] {
                    Expr::Value(Value::Struct(struct_index)) => {
                        let decl_kind = *self.storage().structs[struct_index]
                            .name_to_decl
                            .get(&field_name)
                            .ok_or(Error::NoDecl)?;

                        match decl_kind {
                            DeclKind::AssociatedLet(associated_let_index) => {
                                self.eval_associated_let(associated_let_index)?;

                                let associated_let =
                                    &self.storage().associated_lets[associated_let_index];
                                let let_ = &self.storage().lets[associated_let.let_index];
                                Ok(Some(self.storage().exprs[let_.expr]))
                            }
                            DeclKind::Function(function_index) => Ok(Some(Expr::Name(Name {
                                symbol: field_name,
                                level: self.storage().level,
                                kind: NameKind::Decl(decl_kind),
                            }))),
                        }
                    }
                    Expr::Value(Value::Object(ref object)) => {
                        let value = object
                            .fields
                            .iter()
                            .find_map(|(name, value)| (*name == field_name).then(|| value.clone()))
                            .ok_or(Error::NoField)?;

                        Ok(Some(Expr::Value(value)))
                    }
                    Expr::Constructor(ctor, ref fields) => {
                        let field = fields
                            .iter()
                            .find(|field| field.name.0 == field_name)
                            .ok_or(Error::NoField)?;

                        Ok(Some(self.storage().exprs[field.value]))
                    }
                    _ => Err(Error::NoField),
                }
            }
        }
    }

    fn eval_let(&mut self, let_index: Ix<Let>) -> Result<Ix<Expr>, Error> {
        if let Some(ty) = self.storage().lets[let_index].ty {
            self.comptime(true).eval_expr_and_inline(ty)?;
        }
        let expr_index = self.storage().lets[let_index].expr;
        self.eval_expr_and_inline(expr_index)?;
        Ok(expr_index)
    }

    fn eval_associated_let(
        &mut self,
        associated_let_index: Ix<AssociatedLet>,
    ) -> Result<Ix<Expr>, Error> {
        let let_index = self.storage().associated_lets[associated_let_index].let_index;
        // let let_ = &self.storage().lets[let_index];
        self.comptime(true).eval_let(let_index)
    }

    fn eval_block(&mut self, block_index: Ix<Block>) -> Result<Option<Value>, Error> {
        let block = &self.storage().blocks[block_index];
        for stmt in &block.stmts {
            self.eval_stmt(stmt)?;
        }
        self.eval_expr_and_inline(block.returns)?;

        if let Expr::Value(value) = &self.storage().exprs[block.returns] {
            Ok(Some(value.clone()))
        } else {
            Ok(None)
        }
    }

    fn eval_stmt(&mut self, stmt: &Stmt) -> Result<(), Error> {
        match stmt {
            Stmt::Let(let_index) => {
                self.eval_let(*let_index)?;
                Ok(())
            }
        }
    }

    fn eval_call(&mut self, callee: Callee, args: &[Ix<Expr>]) -> Result<Option<Expr>, Error> {
        match callee {
            Callee::Expr(callee) => self.eval_expr_call(callee, args),
            Callee::BuiltinFunction(builtin) => self.eval_builtin_call(builtin, args),
        }
    }

    fn eval_expr_call(
        &mut self,
        callee: Ix<Expr>,
        args: &[Ix<Expr>],
    ) -> Result<Option<Expr>, Error> {
        // I still need to figure out which monomorphized function I'm calling here.
        // Evaluate the types of the call
        let function = self.called_function_of_expr(callee)?;
        if args.len() != function.params.len() {
            return Err(Error::Call);
        }
        for &arg in args {
            self.eval_expr_and_inline(arg);
        }

        // PROBLEM: this is doing a new analysis on every single function call.
        // we need to deduplicate them.
        // We can't just create a single dummy storage for evaluating the function type,
        // because it could itself need to do monomorphization within the computation
        // of the type.
        // We need storage "arenas" so we don't keep all the storages around
        //
        // Here's an example:
        // fn foo(t: {
        //     let f = struct {
        //         fn a(U: type) type { U }
        //     }.a;
        //
        //     // monomorphize twice inside of the type evaluator!
        //     let int32 = f(i32);
        //     let uint32 = f(u32);
        //     int32
        // }) i32 { t }

        for (param, arg) in function.params.iter().zip(args) {
            // This is where monomorphization actually happens.

            let param_storage = self
                .with_storage(param.non_monomorphized_storage_index)
                .storage();

            let monomorphized_storage_index = Ix::push(
                &mut self.program_evaluator.hir.storages,
                Storage {
                    kind: StorageKind::FunctionParam {
                        backref: param_storage.unwrap_backref(),
                        arg: None,
                    },
                    ..param_storage.clone()
                },
            );
            let mut monomorphized_evaluator = self.with_storage(monomorphized_storage_index);
            let ty = monomorphized_evaluator.eval_expr_into_type(param.ty)?;

            let comptime_value = if ty.is_comptime_only(&mut monomorphized_evaluator) {
                let Expr::Value(value) = &self.storage().exprs[*arg] else {
                    return Err(Error::ValueUnknownAtComptime);
                };

                Some(value.clone())
            } else {
                None
            };

            self.with_storage(monomorphized_storage_index)
                .storage()
                .unwrap_arg_and_set(Arg {
                    comptime_known_value: comptime_value,
                });
        }

        if self.in_comptime {
            // let result = self.perform_call(callee, args)?;
            // let result = todo!();
            //
            // Ok(Some(result))
            todo!("function calls at comptime")
        } else {
            Ok(None)
        }
    }

    // optimization opportunity: eval everything reachable but treat params as opaque BEFORE
    // monomorphization, that way we can deduplicate shared work.

    fn called_function_of_expr(&mut self, callee: Ix<Expr>) -> Result<&Function, Error> {
        self.eval_expr_and_inline(callee)?;

        match &self.storage().exprs[callee] {
            Expr::Value(value) => self.called_function_of_value(value),
            Expr::Name(name) => self.called_function_of_name(name),
            _ => Err(Error::Call),
        }
    }

    fn called_function_of_value(&mut self, value: &Value) -> Result<&Function, Error> {
        match value {
            Value::Function(storage_index, function_index) => {
                let function =
                    &self.with_storage(*storage_index).storage().functions[*function_index];
                Ok(function)
            }
            _ => Err(Error::Call),
        }
    }

    fn called_function_of_name(&mut self, name: &Name) -> Result<&Function, Error> {
        let mut name_evaluator = self.storage_at_level(name.level);
        match name.kind {
            NameKind::Decl(decl) => match decl {
                DeclKind::AssociatedLet(associated_let_index) => {
                    let expr = name_evaluator.eval_associated_let(associated_let_index)?;
                    self.called_function_of_expr(expr)
                }
                DeclKind::Function(function_index) => Ok(&self.storage().functions[function_index]),
            },
            NameKind::Let(let_index) => {
                let expr = name_evaluator.storage().lets[let_index].expr;
                self.called_function_of_expr(expr)
            }
            NameKind::Param => {
                let param = name_evaluator
                    .storage()
                    .unwrap_arg()
                    .comptime_known_value
                    .ok_or(Error::ValueUnknownAtComptime)?;
                self.called_function_of_value(&param)
            }
        }
    }

    // fn perform_call(&mut self, callee: Ix<Expr>, args: &[Ix<Expr>]) -> Result<Expr, Error> {
    //     assert!(
    //         self.in_comptime,
    //         "why are we doing a call outside of comptime"
    //     );
    //
    //     let Expr::Name(name) = &self.storage().exprs[callee] else {
    //         return Err(Error::Call);
    //     };
    //
    //     let mut name_evaluator = self.storage_at_level(name.level);
    //     match name.kind {
    //         NameKind::Decl(decl) => match decl {
    //             DeclKind::AssociatedLet(associated_let_index) => {
    //                 let expr = name_evaluator.eval_associated_let(associated_let_index)?;
    //                 self.perform_call(expr, args)
    //             }
    //             DeclKind::Function(function_index) => {
    //                 // this is where the bulk of the work actually is
    //                 let function = &name_evaluator.storage().functions[function_index];
    //
    //                 if function.params.len() != args.len() {
    //                     return Err(Error::Call);
    //                 }
    //
    //                 for param in function.params.iter() {
    //                     let param_ty = name_evaluator.eval_or_get_expr(param.ty)?;
    //                     let param_ty = match &param_ty {
    //                         Expr::Name(name) => match name.kind {
    //                             NameKind::Decl(decl) => match decl {
    //                                 DeclKind::AssociatedLet(_) => todo!(),
    //                                 DeclKind::Function(_) => todo!(),
    //                             },
    //                             NameKind::Let(_) => todo!(),
    //                             NameKind::Param(_) => todo!(),
    //                         },
    //                         Expr::Struct(_) => todo!(),
    //                         Expr::BuiltinType(builtin) => Type::Builtin(*builtin),
    //                         _ => return Err(Error::Call),
    //                     };
    //                 }
    //
    //                 // go through and evaluate the types one by one
    //
    //                 // get the monomorphized function
    //                 todo!()
    //             }
    //         },
    //         NameKind::Let(let_index) => {
    //             let expr = name_evaluator.storage().lets[let_index].expr;
    //             self.perform_call(expr, args)
    //         }
    //         NameKind::Param(_) => todo!(),
    //     }
    // }

    fn eval_builtin_call(
        &mut self,
        Span(builtin_function, range): Span<BuiltinFunction>,
        args: &[Ix<Expr>],
    ) -> Result<Option<Expr>, Error> {
        match builtin_function {
            BuiltinFunction::InComptime => {
                if !args.is_empty() {
                    return Err(Error::Call);
                }
                Ok(Some(Expr::Value(Value::Bool(self.in_comptime))))
            }
            BuiltinFunction::Trap => {
                if !args.is_empty() {
                    return Err(Error::Call);
                }

                if self.in_comptime {
                    return Err(Error::TrapAtComptime(range));
                }

                Ok(None)
            }
            BuiltinFunction::Clz => {
                let [arg] = args else {
                    return Err(Error::Call);
                };

                self.eval_expr_and_inline(*arg)?;

                if self.in_comptime {
                    if let Expr::Value(Value::Int(n)) = self.storage().exprs[*arg] {
                        let leading_zeros = n.leading_zeros() as i32;
                        Ok(Some(Expr::Value(Value::Int(leading_zeros))))
                    } else {
                        Err(Error::Call)
                    }
                } else {
                    Ok(None)
                }
            }
            BuiltinFunction::Ctz => {
                let [arg] = args else {
                    return Err(Error::Call);
                };

                self.eval_expr_and_inline(*arg)?;

                if self.in_comptime {
                    if let Expr::Value(Value::Int(n)) = self.storage().exprs[*arg] {
                        let trailing_zeros = n.trailing_zeros() as i32;
                        Ok(Some(Expr::Value(Value::Int(trailing_zeros))))
                    } else {
                        Err(Error::Call)
                    }
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn eval_expr_into_type(&mut self, type_index: Ix<Expr>) -> Result<Type, Error> {
        let type_expr = self.comptime(true).eval_or_get_expr(type_index)?;

        match type_expr {
            Expr::Name(name) => {
                let mut type_evaluator = self.storage_at_level(name.level).comptime(true);
                match name.kind {
                    NameKind::Decl(decl) => match decl {
                        DeclKind::AssociatedLet(associated_let_index) => {
                            let expr = type_evaluator.eval_associated_let(associated_let_index)?;
                            type_evaluator.eval_expr_into_type(expr)
                        }
                        DeclKind::Function(_) => Err(Error::NotAType),
                    },
                    NameKind::Param => {
                        let arg = type_evaluator
                            .storage()
                            .unwrap_arg()
                            .comptime_known_value
                            .as_ref()
                            .ok_or(Error::ValueUnknownAtComptime)?;

                        self.eval_value_into_type(arg)
                    }
                    NameKind::Let(let_index) => {
                        let expr = type_evaluator.eval_let(let_index)?;
                        type_evaluator.eval_expr_into_type(expr)
                    }
                }
            }
            Expr::Value(ref value) => self.eval_value_into_type(value),
            _ => Err(Error::NotAType),
        }
    }

    fn eval_value_into_type(&mut self, value: &Value) -> Result<Type, Error> {
        match value {
            Value::Struct(struct_index) => todo!(),
            Value::BuiltinType(builtin) => Ok(Type::Builtin(*builtin)),
            _ => Err(Error::NotAType),
        }
    }
}
