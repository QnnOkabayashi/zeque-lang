//! The AST -> HIR transformation consists of:
//!   * name resolution
//!   * using an array and indices for the tree structure

use crate::util::{Ix, Scope};
use crate::{ast, hir};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("name not found: `{0}`")]
    NameNotFound(String),
    #[error("duplicate function names")]
    DuplicateFunctionNames,
}

pub fn entry(functions: &[ast::Function]) -> Result<Vec<hir::Function>, Error> {
    let mut hir_functions = Vec::with_capacity(functions.len());

    let mut scope = Vec::with_capacity(functions.len());
    let mut program_scope = Scope::new(&mut scope);

    let mut function_names = HashSet::with_capacity(functions.len());
    for (index, function) in functions.iter().enumerate() {
        if !function_names.insert(function.name.as_str()) {
            return Err(Error::DuplicateFunctionNames);
        }
        program_scope.push(hir::Name::Function(Ix::new(index)));
    }

    for function in functions {
        let mut function_scope = program_scope.enter_scope();
        let mut lowerer = LoweringContext {
            context: hir::FunctionContext {
                params: Vec::with_capacity(function.params.len()),
                lets: vec![],
                exprs: vec![],
                blocks: vec![],
            },
            program: functions,
        };

        for param in &function.params {
            lowerer.lower_param(param, &mut function_scope)?;
        }
        let returns = lowerer.lower_expr(&function.return_type, &mut function_scope)?;
        let block_index = lowerer.lower_block(&function.body, &mut function_scope)?;

        hir_functions.push(hir::Function {
            name: function.name.clone(),
            return_type: returns,
            body: block_index,
            context: lowerer.context,
        })
    }

    Ok(hir_functions)
}

struct LoweringContext<'ast> {
    context: hir::FunctionContext,
    program: &'ast [ast::Function],
}

impl<'ast> LoweringContext<'ast> {
    fn lower_expr_raw(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<hir::Expr, Error> {
        match expr {
            ast::Expr::Int(int) => Ok(hir::Expr::Int(*int)),
            ast::Expr::Bool(boolean) => Ok(hir::Expr::Bool(*boolean)),
            ast::Expr::BinOp(op, lhs, rhs) => {
                let lhs = self.lower_expr(lhs, scope)?;
                let rhs = self.lower_expr(rhs, scope)?;
                Ok(hir::Expr::BinOp(*op, lhs, rhs))
            }
            ast::Expr::IfThenElse(cond, then, else_) => {
                let cond = self.lower_expr(cond, scope)?;
                let then = self.lower_expr(then, scope)?;
                let else_ = self.lower_expr(else_, scope)?;
                Ok(hir::Expr::IfThenElse(cond, then, else_))
            }
            ast::Expr::Name(name) => {
                let name = self.lower_name(name, scope)?;

                Ok(hir::Expr::Name(name))
            }
            ast::Expr::Call(function, arguments) => {
                let function = self.lower_expr(function, scope)?;

                let arguments = arguments
                    .iter()
                    .map(|argument| self.lower_expr(argument, scope))
                    .collect::<Result<_, _>>()?;

                Ok(hir::Expr::Call(function, arguments))
            }
            ast::Expr::Block(block) => {
                let mut scope = scope.enter_scope();
                let block_index = self.lower_block(block, &mut scope)?;

                Ok(hir::Expr::Block(block_index))
            }
            ast::Expr::Comptime(inner) => {
                let inner = self.lower_expr(inner, scope)?;

                Ok(hir::Expr::Comptime(inner))
            }
        }
    }

    fn lower_expr(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Ix<hir::Expr>, Error> {
        let expr = self.lower_expr_raw(expr, scope)?;

        Ok(Ix::push(&mut self.context.exprs, expr))
    }

    fn lower_param(
        &mut self,
        parameter: &ast::Parameter,
        scope: &mut Scope<'_, hir::Name>,
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

        scope.push(hir::Name::Parameter(index));
        Ok(())
    }

    fn lower_block(
        &mut self,
        block: &ast::Block,
        scope: &mut Scope<'_, hir::Name>,
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
        scope: &mut Scope<'_, hir::Name>,
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
                scope.push(hir::Name::Let(index));

                Ok(hir::Stmt::Let(index))
            }
        }
    }

    fn lower_name(
        &mut self,
        s: &str,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<hir::Name, Error> {
        // Look for name in the current scope.
        // This includes other functions.
        for (_, name) in scope.iter() {
            if s == self.name_as_str(*name) {
                return Ok(*name);
            }
        }

        // Look for name in the set of builtins.
        if let Ok(builtin) = s.parse() {
            return Ok(hir::Name::Builtin(builtin));
        }

        // Not found
        Err(Error::NameNotFound(s.to_string()))
    }

    fn name_as_str(&self, name: hir::Name) -> &str {
        match name {
            hir::Name::Let(index) => &self.context.lets[..][index].name,
            hir::Name::Parameter(index) => &self.context.params[..][index].name,
            hir::Name::Function(index) => &self.program[index.index].name,
            hir::Name::Builtin(builtin) => builtin.as_str(),
        }
    }
}
