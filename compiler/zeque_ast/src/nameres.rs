// deprecated
use crate::ast::{
    Block, Callee, Decl, Expr, ExprIdx, FnDecl, FnDeclIdx, LetIdx, NameIdx, ParamIdx, Stmt, Struct,
    StructIdx,
};
use crate::parse::UnanalyzedAst;
use crate::util::Scope;
use index_vec::IndexVec;
use std::str::FromStr;

/// A name that was user defined
#[derive(Copy, Clone, Debug)]
pub enum LocalItem {
    // Parent(ResolvedName),
    Let(LetIdx),
    Param(ParamIdx),
    FnDecl(FnDeclIdx),
    SelfType(StructIdx),
}

#[derive(Copy, Clone, Debug)]
pub enum ResolvedName {
    LocalItem(LocalItem),
    BuiltinType(BuiltinType),
    Unknown,
}

#[derive(Copy, Clone, Debug)]
pub enum BuiltinType {
    I32,
    Bool,
    Type,
}

impl FromStr for BuiltinType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i32" => Ok(BuiltinType::I32),
            "bool" => Ok(BuiltinType::Bool),
            "type" => Ok(BuiltinType::Type),
            _ => Err(()),
        }
    }
}

pub fn entry(ast: &UnanalyzedAst) -> IndexVec<NameIdx, ResolvedName> {
    let mut stack = Vec::with_capacity(64);
    let mut scope = Scope::new(&mut stack);

    let mut resolver = NameResolver::new(ast);
    resolver.resolve_struct(ast.file_struct_idx, &mut scope);
    assert!(
        resolver.resolved_names.len() == ast.names.len(),
        "resolver missed a name"
    );
    resolver.resolved_names
}

/// Traverses the AST and builds up a table for NameIdx's to index into to find what a name refers
/// to.
struct NameResolver<'a> {
    ast: &'a UnanalyzedAst,
    resolved_names: IndexVec<NameIdx, ResolvedName>,
}

impl<'a> NameResolver<'a> {
    fn new(ast: &'a UnanalyzedAst) -> Self {
        NameResolver {
            ast,
            resolved_names: IndexVec::with_capacity(ast.names.len()),
        }
    }

    fn resolve_decls(&mut self, decls: &[Decl], scope: &mut Scope<'_, LocalItem>) {
        for decl in decls {
            match decl {
                Decl::Fn(fn_decl_idx) => scope.push(LocalItem::FnDecl(*fn_decl_idx)),
                Decl::Field(_field_decl) => {
                    // nothing for now
                }
            }
        }

        for decl in decls {
            self.resolve_decl(decl, scope);
        }
    }

    fn resolve_decl(&mut self, decl: &Decl, scope: &mut Scope<'_, LocalItem>) {
        match decl {
            Decl::Fn(fn_decl_idx) => self.resolve_fn_decl(*fn_decl_idx, scope),
            Decl::Field(field_decl) => self.resolve_expr(field_decl.ty, scope),
        }
    }

    fn resolve_fn_decl(&mut self, fn_decl_idx: FnDeclIdx, scope: &mut Scope<'_, LocalItem>) {
        let mut scope = scope.enter_scope();
        let fn_decl = &self.ast.fn_decls[fn_decl_idx];
        // Resolve the params BEFORE resolving the return type
        for param in &fn_decl.params {
            self.resolve_param(*param, &mut scope);
        }
        self.resolve_expr(fn_decl.return_type, &mut scope);
        self.resolve_block(&fn_decl.body, &mut scope);
    }

    fn resolve_param(&mut self, param_idx: ParamIdx, scope: &mut Scope<'_, LocalItem>) {
        let param = &self.ast.params[param_idx];
        // Resolve the param type BEFORE making the binding accessible
        self.resolve_expr(param.ty, scope);
        scope.push(LocalItem::Param(param_idx));
    }

    fn resolve_expr(&mut self, expr_idx: ExprIdx, scope: &mut Scope<'_, LocalItem>) {
        match &self.ast.exprs[expr_idx] {
            Expr::BinOp(_, lhs, rhs) => {
                self.resolve_expr(*lhs, scope);
                self.resolve_expr(*rhs, scope);
            }
            Expr::IfThenElse(cond, then, else_) => {
                self.resolve_expr(*cond, scope);
                self.resolve_expr(*then, scope);
                self.resolve_expr(*else_, scope);
            }
            Expr::Name(name_idx) => self.resolve_name(*name_idx, scope),
            Expr::Block(block) => self.resolve_block(block, scope),
            Expr::Call { callee, args } => {
                if let Callee::Expr(callee) = callee {
                    self.resolve_expr(*callee, scope);
                }
                for arg in args {
                    self.resolve_expr(*arg, scope);
                }
            }
            Expr::Comptime(inner) => self.resolve_expr(*inner, scope),
            Expr::Struct(struct_idx) => self.resolve_struct(*struct_idx, scope),
            Expr::Constructor { ty, fields } => {
                if let Some(expr_idx) = ty {
                    self.resolve_expr(*expr_idx, scope);
                }
                for field in fields {
                    self.resolve_expr(field.expr, scope);
                }
            }
            Expr::FieldAccess {
                expr,
                field_name: _,
            } => self.resolve_expr(*expr, scope),
            Expr::Int(_) | Expr::Bool(_) | Expr::Error(_) => {
                // no further resolution required
            }
        }
    }

    /// Given a `NameIdx`:
    /// 1. get the string associated with that index
    /// 2. look up what ResolvedName is associated with that string
    ///      in the current scope.
    /// 3. push the ResolvedName into the table at the NameIdx slot.
    fn resolve_name(&mut self, name_idx: NameIdx, scope: &mut Scope<'_, LocalItem>) {
        let name = self.ast.names[name_idx].as_str();

        let resolved_name = scope
            .iter()
            .copied()
            .find(|local_item| name == self.local_item_name(*local_item))
            .map(ResolvedName::LocalItem)
            .or_else(|| {
                name.parse::<BuiltinType>()
                    .map(ResolvedName::BuiltinType)
                    .ok()
            })
            .unwrap_or(ResolvedName::Unknown);

        assert_eq!(name_idx, self.resolved_names.push(resolved_name));
    }

    /// Given a [`LocalItem`], what string can it be referenced by?
    /// We compare the output of this function to a name string, and if equal,
    /// we assume that the name was meant to refer to this LocalItem.
    fn local_item_name(&self, local_item: LocalItem) -> &str {
        match local_item {
            LocalItem::Let(let_idx) => self.ast.lets[let_idx].name.as_str(),
            LocalItem::Param(param_idx) => self.ast.params[param_idx].name.as_str(),
            LocalItem::FnDecl(fn_decl_idx) => self.ast.fn_decls[fn_decl_idx].name.as_str(),
            LocalItem::SelfType(_) => "Self",
        }
    }

    fn resolve_block(&mut self, block: &Block, scope: &mut Scope<'_, LocalItem>) {
        let mut scope = scope.enter_scope();
        for stmt in &block.stmts {
            self.resolve_stmt(stmt, &mut scope);
        }
        self.resolve_expr(block.returns, &mut scope);
    }

    fn resolve_stmt(&mut self, stmt: &Stmt, scope: &mut Scope<'_, LocalItem>) {
        match stmt {
            Stmt::Let(let_idx) => {
                let let_ = &self.ast.lets[*let_idx];
                if let Some(ty) = let_.ty {
                    self.resolve_expr(ty, scope);
                }
                self.resolve_expr(let_.expr, scope);

                scope.push(LocalItem::Let(*let_idx));
            }
        }
    }

    fn resolve_struct(&mut self, struct_idx: StructIdx, scope: &mut Scope<'_, LocalItem>) {
        let struct_ = &self.ast.structs[struct_idx];
        // decls in a struct are all in their own scope.
        let mut scope = scope.enter_scope();
        // Anyone in this scope that says "Self" will now refer to this struct.
        scope.push(LocalItem::SelfType(struct_idx));
        self.resolve_decls(&struct_.decls, &mut scope);
    }
}
