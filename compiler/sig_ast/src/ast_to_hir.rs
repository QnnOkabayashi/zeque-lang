//! The AST -> HIR transformation consists of:
//!   * name resolution
//!   * using an array and indices for the tree structure

use crate::hir::Builtin;
use crate::util::{Ix, Scope, StringInterner};
use crate::{ast, hir};
use std::collections::HashSet;
use string_interner::DefaultSymbol;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("name not found: `{0}`")]
    NameNotFound(String),
    #[error("duplicate function names")]
    DuplicateFunctionNames,
}

pub fn entry(
    functions: &[ast::Function],
    interner: &mut StringInterner,
) -> Result<Vec<hir::Function>, Error> {
    let mut hir_functions = Vec::with_capacity(functions.len());

    let mut scope = Vec::with_capacity(functions.len());
    let mut program_scope = Scope::new(&mut scope);

    let mut function_names = HashSet::with_capacity(functions.len());
    for (index, function) in functions.iter().enumerate() {
        if !function_names.insert(function.name) {
            return Err(Error::DuplicateFunctionNames);
        }
        program_scope.push(hir::Binding::Function(Ix::new(index)));
    }

    for function in functions {
        let mut function_scope = program_scope.enter_scope();
        let mut lowerer = LoweringContext {
            context: hir::FunctionContext {
                params: Vec::with_capacity(function.params.len()),
                lets: vec![],
                exprs: vec![],
                blocks: vec![],
                structs: vec![],
            },
            program: functions,
            builtin_string_symbols: BuiltinStringSymbols::new(interner),
            interner,
        };

        for param in &function.params {
            lowerer.lower_param(param, &mut function_scope)?;
        }
        let returns = lowerer.lower_expr(&function.return_type, &mut function_scope)?;
        let block_index = lowerer.lower_block(&function.body, &mut function_scope)?;

        hir_functions.push(hir::Function {
            name: function.name,
            return_type: returns,
            body: block_index,
            context: lowerer.context,
        })
    }

    Ok(hir_functions)
}

struct BuiltinStringSymbols {
    sym_i32: DefaultSymbol,
    sym_bool: DefaultSymbol,
    sym_type: DefaultSymbol,
}

impl BuiltinStringSymbols {
    fn new(interner: &mut StringInterner) -> Self {
        BuiltinStringSymbols {
            sym_i32: interner.get_or_intern(Builtin::I32.as_str()),
            sym_bool: interner.get_or_intern(Builtin::Bool.as_str()),
            sym_type: interner.get_or_intern(Builtin::Type.as_str()),
        }
    }
}

struct LoweringContext<'ast> {
    context: hir::FunctionContext,
    program: &'ast [ast::Function],
    builtin_string_symbols: BuiltinStringSymbols,
    interner: &'ast mut StringInterner,
}

impl<'ast> LoweringContext<'ast> {
    fn lower_expr_raw(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<hir::Expr, Error> {
        match expr {
            ast::Expr::Int(int) => Ok(hir::Expr::Int(*int)),
            ast::Expr::Bool(boolean) => Ok(hir::Expr::Bool(*boolean)),
            ast::Expr::BinOp(op, lhs, rhs) => {
                let lhs_index = self.lower_expr(lhs, scope)?;
                let rhs_index = self.lower_expr(rhs, scope)?;
                Ok(hir::Expr::BinOp(*op, lhs_index, rhs_index))
            }
            ast::Expr::IfThenElse(ref cond, ref then, ref else_) => {
                let cond_index = self.lower_expr(cond, scope)?;
                let then_index = self.lower_expr(then, scope)?;
                let else_index = self.lower_expr(else_, scope)?;

                Ok(hir::Expr::IfThenElse(cond_index, then_index, else_index))
            }
            ast::Expr::Name(name) => {
                let name = self.lower_name(*name, scope)?;

                Ok(hir::Expr::Name(name))
            }
            ast::Expr::Call(callee, arguments) => {
                let callee_index = self.lower_expr(callee, scope)?;

                let argument_indices = arguments
                    .iter()
                    .map(|argument| self.lower_expr(argument, scope))
                    .collect::<Result<_, _>>()?;

                Ok(hir::Expr::Call(callee_index, argument_indices))
            }
            ast::Expr::Block(block) => {
                let mut scope = scope.enter_scope();
                let block_index = self.lower_block(block, &mut scope)?;

                Ok(hir::Expr::Block(block_index))
            }
            ast::Expr::Comptime(expr) => {
                let expr_index = self.lower_expr(expr, scope)?;

                Ok(hir::Expr::Comptime(expr_index))
            }
            ast::Expr::Struct(struct_) => {
                let struct_index = self.lower_struct(struct_, scope)?;

                Ok(hir::Expr::Struct(struct_index))
            }
            ast::Expr::Constructor(ctor_type, fields) => {
                let ctor_type = ctor_type
                    .as_deref()
                    .map(|ctor| self.lower_expr(ctor, scope))
                    .transpose()?;

                let fields = fields
                    .iter()
                    .map(|field| {
                        let name = field.name.clone();
                        let value = self.lower_expr(&field.value, scope)?;

                        Ok(hir::StructField { name, value })
                    })
                    .collect::<Result<_, Error>>()?;

                Ok(hir::Expr::Constructor(ctor_type, fields))
            }
            ast::Expr::Field(ref value, field) => {
                let value = self.lower_expr(value, scope)?;

                Ok(hir::Expr::Field(value, field.clone()))
            }
        }
    }

    fn lower_expr(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<Ix<hir::Expr>, Error> {
        let expr = self.lower_expr_raw(expr, scope)?;

        Ok(Ix::push(&mut self.context.exprs, expr))
    }

    fn lower_param(
        &mut self,
        parameter: &ast::Parameter,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<(), Error> {
        let ty = self.lower_expr(&parameter.ty, scope)?;

        let index = Ix::push(
            &mut self.context.params,
            hir::Parameter {
                is_comptime: parameter.is_comptime,
                name: parameter.name.clone(),
                ty,
            },
        );

        scope.push(hir::Binding::Parameter(index));
        Ok(())
    }

    fn lower_block(
        &mut self,
        block: &ast::Block,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<Ix<hir::Block>, Error> {
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmts.push(self.lower_stmt(stmt, scope)?);
        }

        let returns = self.lower_expr(&block.returns, scope)?;
        let block = hir::Block { stmts, returns };

        Ok(Ix::push(&mut self.context.blocks, block))
    }

    fn lower_stmt(
        &mut self,
        stmt: &ast::Stmt,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<hir::Stmt, Error> {
        match stmt {
            ast::Stmt::Let(let_) => {
                let expr = self.lower_expr(&let_.expr, scope)?;
                let ty = let_
                    .ty
                    .as_ref()
                    .map(|ty| self.lower_expr(ty, scope))
                    .transpose()?;

                let let_ = hir::Let {
                    name: let_.name.clone(),
                    ty,
                    expr,
                };

                let index = Ix::push(&mut self.context.lets, let_);
                scope.push(hir::Binding::Let(index));

                Ok(hir::Stmt::Let(index))
            }
        }
    }

    fn lower_name(
        &mut self,
        symbol: DefaultSymbol,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<hir::Binding, Error> {
        // Look for name in the current scope.
        // This includes other functions.
        for (_, name) in scope.iter() {
            if symbol == self.name_as_symbol(*name) {
                return Ok(*name);
            }
        }

        if symbol == self.builtin_string_symbols.sym_i32 {
            Ok(hir::Binding::Builtin(hir::Builtin::I32))
        } else if symbol == self.builtin_string_symbols.sym_bool {
            Ok(hir::Binding::Builtin(hir::Builtin::Bool))
        } else if symbol == self.builtin_string_symbols.sym_type {
            Ok(hir::Binding::Builtin(hir::Builtin::Type))
        } else {
            Err(Error::NameNotFound(
                self.interner
                    .resolve(symbol)
                    .expect("symbol is in interner")
                    .to_string(),
            ))
        }
    }

    fn lower_struct(
        &mut self,
        struct_: &ast::Struct,
        scope: &mut Scope<'_, hir::Binding>,
    ) -> Result<Ix<hir::Struct>, Error> {
        let mut scope = scope.enter_scope();
        let fields = struct_
            .fields
            .iter()
            .map(|item| match item {
                ast::StructItem::Field(field) => {
                    let field_type_index = self.lower_expr(&field.value, &mut scope)?;

                    Ok(hir::StructItem::Field(hir::StructField {
                        name: field.name,
                        value: field_type_index,
                    }))
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Ix::push(&mut self.context.structs, hir::Struct { fields }))
    }

    fn name_as_symbol(&self, name: hir::Binding) -> DefaultSymbol {
        match name {
            hir::Binding::Let(index) => self.context.lets[index].name,
            hir::Binding::Parameter(index) => self.context.params[index].name,
            hir::Binding::Function(index) => self.program[index.index].name,
            hir::Binding::Builtin(builtin) => match builtin {
                Builtin::I32 => self.builtin_string_symbols.sym_i32,
                Builtin::Bool => self.builtin_string_symbols.sym_bool,
                Builtin::Type => self.builtin_string_symbols.sym_type,
            },
        }
    }
}
