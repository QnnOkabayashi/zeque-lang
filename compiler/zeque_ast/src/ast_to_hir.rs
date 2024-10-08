//! The AST -> HIR transformation consists of:
//!   * name resolution
//!   * using an array and indices for the tree structure

use crate::util::Scope;
use crate::{ast, hir};
use index_vec::IndexVec;
use smol_str::SmolStr;
use std::collections::HashMap;
use std::mem;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("name not found: {0}")]
    NameNotFound(String),
}

pub fn entry(file: &ast::File) -> Result<hir::Hir, Error> {
    let mut env = vec![];
    let mut scope = Scope::new(&mut env);

    let mut structs = IndexVec::new();
    let mut ctxs = IndexVec::new();
    let mut lowerer = LoweringContext {
        current_ctx: CtxAndCaptures::default(),
        structs: &mut structs,
        ctxs: &mut ctxs,
    };

    let struct_idx = lowerer.lower_decls(&file.decls, &mut scope)?;

    let mut files = IndexVec::with_capacity(1);
    let main = files.push(hir::File {
        struct_idx,
        ctx: lowerer.current_ctx.ctx,
    });

    Ok(hir::Hir {
        structs,
        files,
        main,
    })
}

struct LoweringContext<'a> {
    current_ctx: CtxAndCaptures,
    ctxs: &'a mut IndexVec<hir::LevelIdx, CtxAndCaptures>,
    structs: &'a mut IndexVec<hir::StructIdx, hir::Struct>,
}

#[derive(Default)]
struct CtxAndCaptures {
    ctx: hir::Ctx,
    /// for child to look into
    /// oh, you want this local at my level?
    /// sure, just access my captures with this ParentRefIdx
    // this exists for deduping purposes
    captures: HashMap<NameInScope, hir::ParentRefIdx>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Ref {
    SelfType,
    NameInScope(NameInScope),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct NameInScope {
    name: SmolStr,
    level: hir::LevelIdx,
    local: hir::Local,
}

impl<'a> LoweringContext<'a> {
    fn level(&self) -> hir::LevelIdx {
        self.ctxs.next_idx()
    }

    fn ctx_at_level_mut(&mut self, level: hir::LevelIdx) -> &mut CtxAndCaptures {
        assert!(level <= self.ctxs.next_idx(), "level too high");
        if level == self.ctxs.next_idx() {
            &mut self.current_ctx
        } else {
            &mut self.ctxs[level]
        }
    }

    fn lower_expr_raw(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::Expr, Error> {
        match expr {
            ast::Expr::Int(int) => Ok(hir::Expr::Int(*int)),
            ast::Expr::Bool(boolean) => Ok(hir::Expr::Bool(*boolean)),
            ast::Expr::BinOp { op, lhs, rhs } => {
                let lhs = self.lower_expr(lhs, scope)?;
                let rhs = self.lower_expr(rhs, scope)?;

                Ok(hir::Expr::BinOp { op: *op, lhs, rhs })
            }
            ast::Expr::IfThenElse { cond, then, else_ } => {
                let cond = self.lower_expr(cond, scope)?;
                let then = self.lower_expr(then, scope)?;
                let else_ = self.lower_expr(else_, scope)?;

                Ok(hir::Expr::IfThenElse { cond, then, else_ })
            }
            ast::Expr::Name(name) => self.lower_name(name.clone(), scope),
            ast::Expr::Call { callee, args } => {
                let callee = match callee {
                    ast::Callee::Expr(callee) => {
                        let callee_index = self.lower_expr(callee, scope)?;
                        hir::Callee::Expr(callee_index)
                    }
                    ast::Callee::Builtin(builtin_name) => {
                        let builtin = builtin_name.parse().unwrap_or_else(|e| match e {});

                        hir::Callee::Builtin(builtin)
                    }
                };

                let args = args
                    .iter()
                    .map(|argument| self.lower_expr(argument, scope))
                    .collect::<Result<_, _>>()?;

                Ok(hir::Expr::Call { callee, args })
            }
            ast::Expr::Block(block) => {
                let block_index = self.lower_block(block, scope)?;

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
            ast::Expr::Constructor { ty, fields } => {
                let ty = ty
                    .as_deref()
                    .map(|ty| self.lower_expr(ty, scope))
                    .transpose()?;

                let fields = fields
                    .iter()
                    .map(|field| {
                        let value = self.lower_expr(&field.expr, scope)?;

                        Ok(hir::ConstructorField {
                            name: field.name.clone(),
                            value,
                        })
                    })
                    .collect::<Result<_, Error>>()?;

                Ok(hir::Expr::Constructor { ty, fields })
            }
            ast::Expr::FieldAccess { expr, field_name } => {
                let expr = self.lower_expr(expr, scope)?;

                Ok(hir::Expr::Field {
                    expr,
                    field_name: field_name.clone(),
                })
            }
            ast::Expr::FnType => Ok(hir::Expr::FnType),
        }
    }

    fn lower_expr(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::ExprIdx, Error> {
        let expr = self.lower_expr_raw(expr, scope)?;
        let expr_idx = self.current_ctx.ctx.exprs.push(expr);
        Ok(expr_idx)
    }

    fn lower_param(
        &mut self,
        param: &ast::Param,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::ParamIdx, Error> {
        let ty = self.lower_expr(&param.ty, scope)?;
        let param_idx = self.current_ctx.ctx.params.push(hir::Param {
            is_comptime: param.is_comptime.is_some(),
            name: param.name.clone(),
            ty,
        });
        scope.push(Ref::NameInScope(NameInScope {
            name: param.name.clone(),
            level: self.level(),
            local: hir::Local::Param(param_idx),
        }));
        Ok(param_idx)
    }

    fn lower_block(
        &mut self,
        block: &ast::Block,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::Block, Error> {
        let mut scope = scope.enter_scope();
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmts.push(self.lower_stmt(stmt, &mut scope)?);
        }

        let returns = block
            .returns
            .as_deref()
            .map(|returns| self.lower_expr(returns, &mut scope))
            .transpose()?;
        Ok(hir::Block { stmts, returns })
    }

    fn lower_stmt(
        &mut self,
        stmt: &ast::Stmt,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::Stmt, Error> {
        match stmt {
            ast::Stmt::Let(let_) => {
                let expr = self.lower_expr(&let_.expr, scope)?;
                let ty = let_
                    .ty
                    .as_ref()
                    .map(|ty| self.lower_expr(&ty, scope))
                    .transpose()?;

                let let_idx = self.current_ctx.ctx.lets.push(hir::Let {
                    name: let_.name.clone(),
                    ty,
                    expr,
                });

                scope.push(Ref::NameInScope(NameInScope {
                    name: let_.name.clone(),
                    level: self.level(),
                    local: hir::Local::Let(let_idx),
                }));

                Ok(hir::Stmt::Let(let_idx))
            }
        }
    }

    fn lower_name(
        &mut self,
        name: SmolStr,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::Expr, Error> {
        for name_in_scope in scope.iter() {
            match name_in_scope {
                Ref::SelfType => {
                    if name.as_str() == "Self" {
                        return Ok(hir::Expr::SelfType);
                    }
                }
                Ref::NameInScope(name_in_scope) => {
                    if name_in_scope.name.as_str() == name.as_str() {
                        assert!(
                            self.level() >= name_in_scope.level,
                            "name in scope should have been truncated"
                        );
                        return if self.level() == name_in_scope.level {
                            // No captures, we're just accessing a local variable :)
                            Ok(hir::Expr::Name(hir::Name::Local(name_in_scope.local)))
                        } else {
                            let parent_ref_idx =
                                self.use_name_at_level(name_in_scope, self.level());
                            Ok(hir::Expr::Name(hir::Name::ParentRef(parent_ref_idx)))
                        };
                    }
                }
            }
        }

        if let Ok(builtin_type) = name.parse() {
            return Ok(hir::Expr::BuiltinType(builtin_type));
        }

        Err(Error::NameNotFound(name.to_string()))
    }

    // recursive approach
    fn use_name_at_level(
        &mut self,
        name_in_scope: &NameInScope,
        level: hir::LevelIdx,
    ) -> hir::ParentRefIdx {
        assert!(
            name_in_scope.level <= level,
            "should have already caught this case"
        );
        if !self
            .ctx_at_level_mut(level)
            .captures
            .contains_key(name_in_scope)
        {
            let parent_level = hir::LevelIdx::new(level.index() - 1);
            let parent_ref = if name_in_scope.level == parent_level {
                hir::ParentRef::Local(name_in_scope.local)
            } else {
                hir::ParentRef::Capture(self.use_name_at_level(name_in_scope, parent_level))
            };

            let ctx_and_captures = self.ctx_at_level_mut(level);
            let parent_ref_idx = ctx_and_captures.ctx.parent_captures.push(parent_ref);
            ctx_and_captures
                .captures
                .insert(name_in_scope.clone(), parent_ref_idx);
        }
        self.ctx_at_level_mut(level).captures[name_in_scope]
    }

    fn lower_struct(
        &mut self,
        struct_: &ast::Struct,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::StructIdx, Error> {
        let mut scope = scope.enter_scope();
        self.lower_decls(&struct_.decls, &mut scope)
    }

    fn lower_decls(
        &mut self,
        decls: &[ast::Decl],
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::StructIdx, Error> {
        scope.push(Ref::SelfType);
        // insert a placeholder value
        let struct_idx = self.structs.push(hir::Struct::default());

        let mut num_fns = 0;
        let mut num_fields = 0;
        // Put the names of all the fns in scope (and count so we can preallocate too!)
        for decl in decls {
            match decl {
                ast::Decl::Fn(fn_decl) => {
                    scope.push(Ref::NameInScope(NameInScope {
                        name: fn_decl.name.clone(),
                        level: self.level(),
                        local: hir::Local::Fn(struct_idx, hir::FnIdx::new(num_fns)),
                    }));
                    num_fns += 1;
                }
                ast::Decl::Field(_) => {
                    num_fields += 1;
                }
            }
        }

        let mut fns = IndexVec::with_capacity(num_fns);
        let mut fields = Vec::with_capacity(num_fields);

        for decl in decls {
            match decl {
                ast::Decl::Fn(fn_decl) => {
                    fns.push(self.lower_fn_decl(fn_decl, scope)?);
                }
                ast::Decl::Field(field_decl) => {
                    fields.push(self.lower_field_decl(field_decl, scope)?);
                }
            }
        }

        self.structs[struct_idx] = hir::Struct { fns, fields };

        Ok(struct_idx)
    }

    fn lower_fn_decl(
        &mut self,
        fn_decl: &ast::FnDecl,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::FnDecl, Error> {
        self.ctxs.push(mem::take(&mut self.current_ctx));

        let mut params = Vec::with_capacity(fn_decl.params.len());
        for param in &fn_decl.params {
            params.push(self.lower_param(param, scope)?);
        }

        let return_ty = fn_decl
            .return_ty
            .as_ref()
            .map(|return_ty| self.lower_expr(return_ty, scope))
            .transpose()?;
        let body = self.lower_block(&fn_decl.body, scope)?;

        let ctx_and_captures = mem::replace(
            &mut self.current_ctx,
            self.ctxs.pop().expect("pushed on a ctx"),
        );

        Ok(hir::FnDecl {
            is_pub: fn_decl.is_public.is_some(),
            name: fn_decl.name.clone(),
            params,
            return_ty,
            body,
            ctx: ctx_and_captures.ctx,
        })
    }

    fn lower_field_decl(
        &mut self,
        field_decl: &ast::FieldDecl,
        scope: &mut Scope<'_, Ref>,
    ) -> Result<hir::FieldDecl, Error> {
        let ty = self.lower_expr(&field_decl.ty, scope)?;
        Ok(hir::FieldDecl {
            name: field_decl.name.clone(),
            ty,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! snapshot {
        ($($tokens:tt)*) => {{
            insta::assert_debug_snapshot!(parse(stringify!($($tokens)*)));
        }};
    }

    fn parse(source: &str) -> hir::Hir {
        let ast = crate::parse::file(source).expect("a valid ast");
        entry(&ast).expect("a valid hir")
    }

    #[test]
    fn one_function() {
        snapshot! {
            fn x() -> i32 {
                1
            }
        }
    }

    #[test]
    fn stmt() {
        snapshot! {
            fn x() -> i32 {
                let y = 4;
                y
            }
        }
    }

    #[test]
    fn field_decl() {
        snapshot! {
            a: i32,
        }
    }

    #[test]
    fn struct_def() {
        snapshot! {
            a: struct {
                b: i32,
            },
        }
    }

    #[test]
    fn nested_ctxs() {
        snapshot! {
            fn foo() -> type {
                struct {
                    fn bar() -> type {
                        i32
                    }
                }
            }
        }
    }

    #[test]
    fn reference_outer_ctxs() {
        snapshot! {
            fn foo(comptime T: type) -> type {
                struct {
                    fn bar() -> type {
                        T
                    }
                }
            }
        }
    }

    #[test]
    fn function_call() {
        snapshot! {
            fn main() -> i32 {
                foo()
            }

            fn foo() -> i32 {
                1
            }
        }
    }

    #[test]
    fn if_then_else() {
        snapshot! {
            fn main() -> i32 {
                if true { 1 } else { 0 }
            }
        }
    }

    #[test]
    fn precedence() {
        snapshot! {
            fn main() -> bool {
                1 + 2 * 3 == 4
            }
        }
    }

    #[test]
    fn no_return() {
        snapshot! {
            fn main() {}
        }
    }
}
