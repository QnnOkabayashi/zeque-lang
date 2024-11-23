//! The AST -> HIR transformation consists of:
//!   * name resolution
//!   * using an array and indices for the tree structure

use crate::{ast, hir, util::Scope};
use index_vec::IndexVec;
use smol_str::SmolStr;
use std::collections::HashMap;
use std::mem;

pub fn entry(file: &ast::File) -> hir::Hir {
    let mut env = vec![];
    let mut scope = Scope::new(&mut env);

    let mut structs = IndexVec::new();
    let mut ctxs = IndexVec::new();
    let mut lowerer = LoweringContext {
        current_ctx: CtxAndCaptures::default(),
        structs: &mut structs,
        ctxs: &mut ctxs,
        errors: hir::error::ErrorVec::new(),
    };

    let struct_idx = lowerer.lower_decls(&file.decls, &mut scope);

    let mut files = IndexVec::with_capacity(1);
    let main = files.push(hir::File {
        struct_idx,
        ctx: lowerer.current_ctx.ctx,
    });

    hir::Hir {
        errors: lowerer.errors,
        structs,
        files,
        main,
    }
}

struct LoweringContext<'a> {
    current_ctx: CtxAndCaptures,
    ctxs: &'a mut IndexVec<hir::LevelIdx, CtxAndCaptures>,
    structs: &'a mut IndexVec<hir::StructIdx, hir::Struct>,
    errors: hir::error::ErrorVec,
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

    fn lower_expr_raw(&mut self, expr: &ast::Expr, scope: &mut Scope<'_, Ref>) -> hir::Expr {
        match expr {
            ast::Expr::Int(int) => hir::Expr::Int(int.value),
            ast::Expr::Bool(boolean) => hir::Expr::Bool(boolean.value),
            ast::Expr::Str(string) => match string {
                ast::Str::Normal { .. } => todo!(),
                ast::Str::Raw(_) => todo!(),
            },
            ast::Expr::BinOp(binop) => hir::Expr::BinOp {
                op: binop.kind,
                lhs: self.lower_expr(&binop.lhs, scope),
                rhs: self.lower_expr(&binop.rhs, scope),
            },
            ast::Expr::IfThenElse(if_then_else) => hir::Expr::IfThenElse {
                cond: self.lower_expr(&if_then_else.cond, scope),
                then: self.lower_block(&if_then_else.then, scope),
                else_: self.lower_block(&if_then_else.else_, scope),
            },
            ast::Expr::Name(name) => self.lower_name(name, scope),
            ast::Expr::Call { callee, args } => {
                let callee = match callee {
                    ast::Callee::Expr(callee) => {
                        let callee_idx = self.lower_expr(callee, scope);
                        hir::Callee::Expr(callee_idx)
                    }
                    ast::Callee::Builtin { name, .. } => {
                        let builtin = name.parse().unwrap_or_else(|e| match e {});

                        hir::Callee::Builtin(builtin)
                    }
                };

                let args = args
                    .iter()
                    .map(|argument| self.lower_expr(argument, scope))
                    .collect();

                hir::Expr::Call { callee, args }
            }
            ast::Expr::Block(block) => hir::Expr::Block(self.lower_block(block, scope)),
            ast::Expr::Comptime { expr, .. } => hir::Expr::Comptime(self.lower_expr(expr, scope)),
            ast::Expr::Struct(struct_) => hir::Expr::Struct(self.lower_struct(struct_, scope)),
            ast::Expr::Constructor { ty, fields } => {
                let ty = ty.as_deref().map(|ty| self.lower_expr(ty, scope));

                let fields = fields
                    .iter()
                    .map(|field| {
                        let value = self.lower_expr(&field.expr, scope);

                        hir::ConstructorField {
                            name: field.name.text.clone(),
                            value,
                        }
                    })
                    .collect();

                hir::Expr::Constructor { ty, fields }
            }
            ast::Expr::FieldAccess { expr, field_name } => hir::Expr::Field {
                expr: self.lower_expr(expr, scope),
                field_name: field_name.text.clone(),
            },
            ast::Expr::FnType => hir::Expr::FnType,
        }
    }

    fn lower_expr(&mut self, expr: &ast::Expr, scope: &mut Scope<'_, Ref>) -> hir::ExprIdx {
        let expr = self.lower_expr_raw(expr, scope);
        self.current_ctx.ctx.exprs.push(expr)
    }

    fn lower_param(&mut self, param: &ast::Param, scope: &mut Scope<'_, Ref>) -> hir::ParamIdx {
        let ty = self.lower_expr(&param.ty, scope);
        let param_idx = self.current_ctx.ctx.params.push(hir::Param {
            is_comptime: param.comptime.is_some(),
            name: param.name.text.clone(),
            ty,
        });
        scope.push(Ref::NameInScope(NameInScope {
            name: param.name.text.clone(),
            level: self.level(),
            local: hir::Local::Param(param_idx),
        }));
        param_idx
    }

    fn lower_block(&mut self, block: &ast::Block, scope: &mut Scope<'_, Ref>) -> hir::Block {
        let mut scope = scope.enter_scope();
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmts.push(self.lower_stmt(stmt, &mut scope));
        }

        let returns = block
            .returns
            .as_deref()
            .map(|returns| self.lower_expr(returns, &mut scope));
        hir::Block { stmts, returns }
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt, scope: &mut Scope<'_, Ref>) -> hir::Stmt {
        match stmt {
            ast::Stmt::Let(let_) => {
                let expr = self.lower_expr(&let_.expr, scope);
                let ty = let_.ty.as_ref().map(|ty| self.lower_expr(ty, scope));

                let let_idx = self.current_ctx.ctx.lets.push(hir::Let {
                    name: let_.name.text.clone(),
                    ty,
                    expr,
                });

                scope.push(Ref::NameInScope(NameInScope {
                    name: let_.name.text.clone(),
                    level: self.level(),
                    local: hir::Local::Let(let_idx),
                }));

                hir::Stmt::Let(let_idx)
            }
        }
    }

    fn lower_name(&mut self, name: &ast::Name, scope: &mut Scope<'_, Ref>) -> hir::Expr {
        for name_in_scope in scope.iter() {
            match name_in_scope {
                Ref::SelfType => {
                    if name.text.as_str() == "Self" {
                        return hir::Expr::SelfType;
                    }
                }
                Ref::NameInScope(name_in_scope) => {
                    if name.text.as_str() == name_in_scope.name.as_str() {
                        assert!(
                            self.level() >= name_in_scope.level,
                            "name in scope should have been truncated"
                        );
                        return if self.level() == name_in_scope.level {
                            // No captures, we're just accessing a local variable :)
                            hir::Expr::Name(hir::Name::Local(name_in_scope.local))
                        } else {
                            let parent_ref_idx =
                                self.use_name_at_level(name_in_scope, self.level());
                            hir::Expr::Name(hir::Name::ParentRef(parent_ref_idx))
                        };
                    }
                }
            }
        }

        if let Ok(builtin_type) = name.text.parse() {
            return hir::Expr::BuiltinType(builtin_type);
        }

        let error_idx = self
            .errors
            .push(hir::error::NameResolutionError::new(name.clone()));

        hir::Expr::Error(error_idx)
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
    ) -> hir::StructIdx {
        let mut scope = scope.enter_scope();
        self.lower_decls(&struct_.decls, &mut scope)
    }

    fn lower_decls(&mut self, decls: &[ast::Decl], scope: &mut Scope<'_, Ref>) -> hir::StructIdx {
        scope.push(Ref::SelfType);
        // insert a placeholder value
        let struct_idx = self.structs.push(hir::Struct::default());

        let mut num_fns = 0;
        let mut num_consts = 0;
        let mut num_fields = 0;
        // Put the names of all the fns and consts in scope
        // (and count so we can preallocate too!)
        for decl in decls {
            match decl {
                ast::Decl::Fn(fn_decl) => {
                    scope.push(Ref::NameInScope(NameInScope {
                        name: fn_decl.name.text.clone(),
                        level: self.level(),
                        local: hir::Local::Fn(struct_idx, hir::FnIdx::new(num_fns)),
                    }));
                    num_fns += 1;
                }
                ast::Decl::Comptime(comptime_decl) => {
                    scope.push(Ref::NameInScope(NameInScope {
                        name: comptime_decl.name.text.clone(),
                        level: self.level(),
                        local: hir::Local::Const(struct_idx, hir::ConstIdx::new(num_consts)),
                    }));
                    num_consts += 1;
                }
                ast::Decl::Field(_) => num_fields += 1,
            }
        }

        let mut struct_ = hir::Struct {
            fns: IndexVec::with_capacity(num_fns),
            consts: IndexVec::with_capacity(num_consts),
            fields: Vec::with_capacity(num_fields),
        };

        for decl in decls {
            match decl {
                ast::Decl::Fn(fn_decl) => {
                    struct_.fns.push(self.lower_fn_decl(fn_decl, scope));
                }
                ast::Decl::Field(field_decl) => {
                    struct_
                        .fields
                        .push(self.lower_field_decl(field_decl, scope));
                }
                ast::Decl::Comptime(comptime_decl) => {
                    struct_
                        .consts
                        .push(self.lower_const_decl(comptime_decl, scope));
                }
            }
        }

        self.structs[struct_idx] = struct_;

        struct_idx
    }

    fn lower_const_decl(
        &mut self,
        comptime_decl: &ast::ComptimeDecl,
        scope: &mut Scope<'_, Ref>,
    ) -> hir::ConstDecl {
        // ConstDecls get their own Ctx.
        self.ctxs.push(mem::take(&mut self.current_ctx));
        let ty = comptime_decl
            .ty
            .as_ref()
            .map(|ty| self.lower_expr(ty, scope));
        let value = self.lower_expr(&comptime_decl.value, scope);
        let ctx_and_captures = mem::replace(
            &mut self.current_ctx,
            self.ctxs.pop().expect("pushed on a ctx"),
        );
        hir::ConstDecl {
            name: comptime_decl.name.text.clone(),
            ty,
            value,
            ctx: ctx_and_captures.ctx,
        }
    }

    fn lower_fn_decl(&mut self, fn_decl: &ast::FnDecl, scope: &mut Scope<'_, Ref>) -> hir::FnDecl {
        self.ctxs.push(mem::take(&mut self.current_ctx));

        let mut params = Vec::with_capacity(fn_decl.params.len());
        for param in &fn_decl.params {
            params.push(self.lower_param(param, scope));
        }

        let return_ty = fn_decl
            .return_ty
            .as_ref()
            .map(|return_ty| self.lower_expr(return_ty, scope));
        let body = self.lower_block(&fn_decl.body, scope);

        let ctx_and_captures = mem::replace(
            &mut self.current_ctx,
            self.ctxs.pop().expect("pushed on a ctx"),
        );

        hir::FnDecl {
            is_pub: fn_decl.pub_.is_some(),
            name: fn_decl.name.text.clone(),
            params,
            return_ty,
            body,
            ctx: ctx_and_captures.ctx,
        }
    }

    fn lower_field_decl(
        &mut self,
        field_decl: &ast::FieldDecl,
        scope: &mut Scope<'_, Ref>,
    ) -> hir::FieldDecl {
        let ty = self.lower_expr(&field_decl.ty, scope);

        hir::FieldDecl {
            name: field_decl.name.text.clone(),
            ty,
        }
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
        let ast = zeque_parse::parse::file(source).expect("a valid ast");
        let hir = entry(&ast);
        assert!(hir.errors.is_empty(), "expected a valid hir");
        hir
    }

    #[test]
    fn one_function() {
        snapshot! {
            fn x() -> i32 {
                2
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
