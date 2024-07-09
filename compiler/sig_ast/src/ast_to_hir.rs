//! The AST -> HIR transformation consists of:
//!   * name resolution
//!   * using an array and indices for the tree structure

use crate::hir::BuiltinType;
use crate::util::{Ix, Range, Scope, Span, StringInterner};
use crate::{ast, hir};
use smol_str::SmolStr;
use std::collections::HashMap;
use string_interner::DefaultSymbol;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("name not found: `{0}`")]
    NameNotFound(SmolStr, Range),
    #[error("duplicate function names")]
    DuplicateFunctionNames(Vec<(DefaultSymbol, Vec<Range>)>),
    #[error("unknown builtin: `{0}`")]
    UnknownBuiltin(SmolStr, Range),
}

pub fn entry(functions: &[ast::Function]) -> Result<(Vec<hir::Function>, StringInterner), Error> {
    let mut interner = StringInterner::new();
    let mut hir_functions = Vec::with_capacity(functions.len());

    let mut env = Vec::with_capacity(functions.len());
    let mut program_scope = Scope::new(&mut env);

    let mut function_names: HashMap<DefaultSymbol, Vec<Range>> =
        HashMap::with_capacity(functions.len());
    for (index, function) in functions.iter().enumerate() {
        let Span(name, range) = &function.name;
        function_names
            .entry(interner.get_or_intern(name))
            .or_default()
            .push(*range);
        program_scope.push(hir::Name::Function(Ix::new(index)));
    }

    let duplicate_function_names: Vec<_> = function_names
        .into_iter()
        .filter(|(_, ranges)| ranges.len() > 1)
        .collect();
    if !duplicate_function_names.is_empty() {
        return Err(Error::DuplicateFunctionNames(duplicate_function_names));
    }

    for function in functions {
        let mut function_scope = program_scope.enter_scope();
        let mut lowerer = LoweringContext {
            context: hir::FunctionContext::default(),
            program: functions,
            builtin_string_symbols: BuiltinStringSymbols::new(&mut interner),
            interner: &mut interner,
        };

        for param in &function.params {
            lowerer.lower_param(param, &mut function_scope)?;
        }
        let Span(returns, _) = lowerer.lower_expr(&function.return_type, &mut function_scope)?;
        let Span(block_index, _) = lowerer.lower_block(&function.body, &mut function_scope)?;

        let symbol_index = lowerer.lower_str(&function.name);

        hir_functions.push(hir::Function {
            name: symbol_index,
            return_type: returns,
            body: block_index,
            context: lowerer.context,
        })
    }

    Ok((hir_functions, interner))
}

struct BuiltinStringSymbols {
    sym_i32: DefaultSymbol,
    sym_bool: DefaultSymbol,
    sym_type: DefaultSymbol,
}

impl BuiltinStringSymbols {
    fn new(interner: &mut StringInterner) -> Self {
        BuiltinStringSymbols {
            sym_i32: interner.get_or_intern(BuiltinType::I32.as_str()),
            sym_bool: interner.get_or_intern(BuiltinType::Bool.as_str()),
            sym_type: interner.get_or_intern(BuiltinType::Type.as_str()),
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
    fn lower_str(&mut self, Span(name, range): &Span<SmolStr>) -> Span<DefaultSymbol> {
        let symbol = self.interner.get_or_intern(name);
        Span(symbol, *range)
    }

    fn lower_expr_raw(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<hir::Expr>, Error> {
        match expr {
            ast::Expr::Int(Span(int, range)) => Ok(Span(hir::Expr::Int(*int), *range)),
            ast::Expr::Bool(Span(boolean, range)) => Ok(Span(hir::Expr::Bool(*boolean), *range)),
            ast::Expr::BinOp(op, lhs, rhs) => {
                let Span(lhs, lhs_range) = self.lower_expr(lhs, scope)?;
                let Span(rhs, rhs_range) = self.lower_expr(rhs, scope)?;

                Ok(Span(
                    hir::Expr::BinOp(*op, lhs, rhs),
                    lhs_range.to(rhs_range),
                ))
            }
            ast::Expr::IfThenElse(ref cond, ref then, ref else_) => {
                let Span(cond, cond_range) = self.lower_expr(cond, scope)?;
                let Span(then, ..) = self.lower_expr(then, scope)?;
                let Span(else_, else_range) = self.lower_expr(else_, scope)?;

                Ok(Span(
                    hir::Expr::IfThenElse(cond, then, else_),
                    cond_range.to(else_range),
                ))
            }
            ast::Expr::Name(name) => {
                let Span(name, range) = self.lower_name(name.clone(), scope)?;

                Ok(Span(hir::Expr::Name(name), range))
            }
            ast::Expr::Call(callee, Span(arguments, args_range)) => {
                let Span(callee, callee_range) = match callee {
                    ast::Callee::Expr(callee) => {
                        let Span(callee_index, callee_range) = self.lower_expr(callee, scope)?;
                        Span(hir::Callee::Expr(callee_index), callee_range)
                    }
                    ast::Callee::Builtin(Span(builtin, range)) => {
                        let builtin = builtin
                            .as_str()
                            .parse()
                            .map_err(|()| Error::UnknownBuiltin(builtin.clone(), *range))?;

                        Span(hir::Callee::Builtin(Span(builtin, *range)), *range)
                    }
                };

                let argument_indices = arguments
                    .iter()
                    .map(|argument| self.lower_expr(argument, scope).map(Span::into_inner))
                    .collect::<Result<_, _>>()?;

                Ok(Span(
                    hir::Expr::Call(callee, argument_indices),
                    callee_range.to(*args_range),
                ))
            }
            ast::Expr::Block(block) => {
                let mut scope = scope.enter_scope();
                let Span(block_index, range) = self.lower_block(block, &mut scope)?;

                Ok(Span(hir::Expr::Block(block_index), range))
            }
            ast::Expr::Comptime(expr) => {
                let Span(expr_index, range) = self.lower_expr(expr, scope)?;

                Ok(Span(hir::Expr::Comptime(expr_index), range))
            }
            ast::Expr::Struct(struct_) => {
                let Span(struct_index, range) = self.lower_struct(struct_, scope)?;

                Ok(Span(hir::Expr::Struct(struct_index), range))
            }
            ast::Expr::Constructor(ctor, Span(fields, fields_range)) => {
                let Span(ctor, ctor_range) = self.lower_expr(ctor, scope)?;

                let fields = fields
                    .iter()
                    .map(|field| {
                        let name = self.lower_str(&field.name);
                        let Span(value, _) = self.lower_expr(&field.value, scope)?;

                        Ok(hir::StructField { name, value })
                    })
                    .collect::<Result<_, Error>>()?;

                Ok(Span(
                    hir::Expr::Constructor(Some(ctor), fields),
                    ctor_range.to(*fields_range),
                ))
            }
            ast::Expr::Field(value, field) => {
                let Span(value, value_range) = self.lower_expr(value, scope)?;
                let field = self.lower_str(field);

                Ok(Span(
                    hir::Expr::Field(value, field),
                    value_range.to(field.range()),
                ))
            }
            ast::Expr::Error(range) => Ok(Span(hir::Expr::Error, *range)),
        }
    }

    fn lower_expr(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<Ix<hir::Expr>>, Error> {
        let Span(expr, range) = self.lower_expr_raw(expr, scope)?;

        let index = self.context.exprs.push(Span(expr, range));
        Ok(Span(index, range))
    }

    fn lower_param(
        &mut self,
        Span(param, range): &Span<ast::Param>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<(), Error> {
        let Span(ty, _) = self.lower_expr(&param.ty.0, scope)?;

        let param = hir::Param {
            is_comptime: param.is_comptime.is_some(),
            name: self.lower_str(&param.name),
            ty,
        };

        let index = self.context.params.push(Span(param, *range));

        scope.push(hir::Name::Param(index));
        Ok(())
    }

    fn lower_block(
        &mut self,
        Span(block, range): &Span<ast::Block>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<Ix<hir::Block>>, Error> {
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmts.push(self.lower_stmt(stmt, scope)?);
        }

        let Span(returns, _) = self.lower_expr(&block.returns, scope)?;
        let block = hir::Block { stmts, returns };

        Ok(Span(Ix::push(&mut self.context.blocks, block), *range))
    }

    fn lower_stmt(
        &mut self,
        Span(stmt, _range): &Span<ast::Stmt>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<hir::Stmt, Error> {
        match stmt {
            ast::Stmt::Let(let_) => {
                let Span(expr, _) = self.lower_expr(&let_.expr.0, scope)?;
                let ty = let_
                    .ty
                    .as_ref()
                    .map(|ty| self.lower_expr(&ty.0, scope).map(Span::into_inner))
                    .transpose()?;

                let let_ = hir::Let {
                    name: self.lower_str(&let_.name),
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
        Span(name, range): Span<SmolStr>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<hir::Name>, Error> {
        let symbol = self.interner.get_or_intern(&name);

        // Look for name in the current scope.
        // This includes other functions.
        for (_, name) in scope.iter() {
            if symbol == self.name_as_symbol(*name) {
                return Ok(Span(*name, range));
            }
        }

        if symbol == self.builtin_string_symbols.sym_i32 {
            Ok(Span(hir::Name::Builtin(hir::BuiltinType::I32), range))
        } else if symbol == self.builtin_string_symbols.sym_bool {
            Ok(Span(hir::Name::Builtin(hir::BuiltinType::Bool), range))
        } else if symbol == self.builtin_string_symbols.sym_type {
            Ok(Span(hir::Name::Builtin(hir::BuiltinType::Type), range))
        } else {
            Err(Error::NameNotFound(name, range))
        }
    }

    fn lower_struct(
        &mut self,
        Span(struct_, range): &Span<ast::Struct>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<Ix<hir::Struct>>, Error> {
        let mut scope = scope.enter_scope();
        let fields = struct_
            .fields
            .iter()
            .map(|item| match item {
                ast::StructItem::Field(field) => {
                    let Span(value, _) = self.lower_expr(&field.value, &mut scope)?;

                    Ok(hir::StructItem::Field(hir::StructField {
                        name: self.lower_str(&field.name),
                        value,
                    }))
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Span(
            Ix::push(&mut self.context.structs, hir::Struct { fields }),
            *range,
        ))
    }

    fn name_as_symbol(&mut self, name: hir::Name) -> DefaultSymbol {
        match name {
            hir::Name::Let(index) => self.context.lets[index].name.0,
            hir::Name::Param(index) => self.context.params[index].name.0,
            hir::Name::Function(index) => self
                .interner
                .get_or_intern(&self.program[index.index].name.0),
            hir::Name::Builtin(builtin) => match builtin {
                BuiltinType::I32 => self.builtin_string_symbols.sym_i32,
                BuiltinType::Bool => self.builtin_string_symbols.sym_bool,
                BuiltinType::Type => self.builtin_string_symbols.sym_type,
            },
        }
    }
}
