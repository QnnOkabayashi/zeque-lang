//! The AST -> HIR transformation consists of:
//!   * name resolution
//!   * using an array and indices for the tree structure

use crate::hir::{NameKind, Storage, Value};
use crate::util::{Ix, Range, RangeTable, Scope, Span, StringInterner};
use crate::{ast, hir};
use smol_str::SmolStr;
use std::collections::HashMap;
use string_interner::DefaultSymbol;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("name not found: `{0}`")]
    NameNotFound(SmolStr, Range),
    #[error("duplicate struct item names")]
    DuplicateStructItemNames(Vec<DefaultSymbol>),
    #[error("duplicate field names")]
    DuplicateFieldNames(Vec<DefaultSymbol>),
    #[error("unknown builtin: `{0}`")]
    UnknownBuiltin(SmolStr, Range),
}

/// Returns all the structs, as well as the index of the entry struct.
pub fn entry(
    struct_items: &[ast::StructItem],
    interner: &mut StringInterner,
) -> Result<(hir::Hir, Ix<hir::SourceFile>), Error> {
    let mut ast_to_hir = AstToHir::new(interner);
    let source_file_index = ast_to_hir.lower_source_file(struct_items)?;

    Ok((ast_to_hir.hir, source_file_index))
}

type NameMap<T> = HashMap<DefaultSymbol, T>;

fn check_for_duplicate_item_names<'ast>(
    struct_items: &'ast [ast::StructItem],
    interner: &mut StringInterner,
) -> Result<(), Error> {
    #[derive(Copy, Clone)]
    enum AssociatedItem<'ast> {
        Fn(&'ast ast::Function),
        Let(&'ast ast::AssociatedLet),
    }

    let mut associated_names: NameMap<Vec<AssociatedItem<'ast>>> = HashMap::new();
    let mut field_names: NameMap<Vec<&'ast ast::StructFieldDef>> = HashMap::new();
    for (index, struct_item) in struct_items.iter().enumerate() {
        match struct_item {
            ast::StructItem::Field(struct_field_def) => {
                field_names
                    .entry(interner.get_or_intern(&struct_field_def.struct_field.name.0))
                    .or_default()
                    .push(struct_field_def);
            }
            ast::StructItem::Fn(function) => {
                associated_names
                    .entry(interner.get_or_intern(&function.name.0))
                    .or_default()
                    .push(AssociatedItem::Fn(function));
            }
            ast::StructItem::Let(associated_let) => {
                associated_names
                    .entry(interner.get_or_intern(&associated_let.let_.name.0))
                    .or_default()
                    .push(AssociatedItem::Let(associated_let));
            }
        }
    }

    let mut duplicate_associated_item_names: Vec<DefaultSymbol> = associated_names
        .into_iter()
        .filter_map(|(symbol, items)| (items.len() > 1).then_some(symbol))
        .collect();

    if !duplicate_associated_item_names.is_empty() {
        return Err(Error::DuplicateStructItemNames(
            duplicate_associated_item_names,
        ));
    }

    let mut duplicate_field_names = vec![];
    let field_names: Vec<DefaultSymbol> = field_names
        .into_iter()
        .filter_map(|(symbol, fields)| (fields.len() > 1).then_some(symbol))
        .collect();

    if !duplicate_field_names.is_empty() {
        return Err(Error::DuplicateFieldNames(duplicate_field_names));
    }

    Ok(())
}

/// Used for claiming an index in a vec without actually putting something there yet.
/// For example, if we're going to lower a function, it should be able to refer to itself.
/// We can push an option to the vec and then remember the index, which will be used
/// to refer to itself. Once we finish lowering, we put the actual value at that index.
/// This is a custom typedef to show that we intend to put something there.
type StableIndex<T> = Option<T>;

struct AstToHir<'ast> {
    hir: hir::Hir,
    interner: &'ast mut StringInterner,
}

impl<'ast> AstToHir<'ast> {
    fn new(interner: &'ast mut StringInterner) -> Self {
        AstToHir {
            hir: hir::Hir::default(),
            interner,
        }
    }

    fn make_lowerer(&mut self, level: hir::Level) -> StorageLowerer<'_> {
        StorageLowerer {
            storage_index: Ix::push(
                &mut self.hir.storages,
                Storage {
                    level,
                    structs: Vec::new(),
                    lets: Vec::new(),
                    blocks: Vec::new(),
                    exprs: RangeTable::new(),
                },
            ),
            ast_to_hir: self,
            structs: Vec::new(),
            functions: Vec::new(),
            associated_lets: Vec::new(),
        }
    }

    fn lower_source_file(
        &mut self,
        struct_items: &'ast [ast::StructItem],
    ) -> Result<Ix<hir::SourceFile>, Error> {
        let mut lowerer = self.make_lowerer(hir::Level::SOURCE_FILE);

        let struct_index = lowerer.lower_struct(struct_items, &mut Scope::new(&mut vec![]))?;

        Ok(Ix::push(
            &mut self.hir.source_files,
            hir::SourceFile {
                storage: lowerer.finish_and_get_storage_index(),
                struct_index,
            },
        ))
    }

    fn lower_str(&mut self, Span(name, range): &Span<SmolStr>) -> Span<DefaultSymbol> {
        let symbol = self.interner.get_or_intern(name);
        Span(symbol, *range)
    }
}

// We get a new lowerer for each hir::Storage, and there's an hir::Storage for each monomorphizable
// unit. This includes source files (which are already monomorphized), and functions, which may
// potentially be monomorphized. For example, a file is a struct, so it will have its own storage.
// But it may contain functions, each of which has their own storage.
// The purpose of splitting up the storages like this is so that when we want to monomorphize,
// we can just clone the storage and do inline changes on it directly.
struct StorageLowerer<'a> {
    // we remember the level so when we hit an associated let or function, we can say what "level"
    // its at. Then when we reference it later, we can just follow backrefs until we reach the
    // proper level.
    ast_to_hir: &'a mut AstToHir<'a>,
    storage_index: Ix<Storage>,

    // these fields will go into the storage.
    structs: Vec<StableIndex<hir::Struct>>,
    functions: Vec<StableIndex<hir::Function>>,
    associated_lets: Vec<StableIndex<hir::AssociatedLet>>,
}

impl<'ast> StorageLowerer<'ast> {
    // write everything to the storage and return it
    fn finish_and_get_storage_index(mut self) -> Ix<Storage> {
        let storage = self.storage();
        storage.structs = self.structs.into_iter().collect::<Option<_>>().unwrap();
        self.storage_index
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.ast_to_hir.hir.storages[self.storage_index]
    }

    fn categorize_struct_items(
        &mut self,
        struct_items: &'ast [ast::StructItem],
    ) -> (
        Vec<&'ast ast::StructFieldDef>,
        Vec<(Ix<hir::AssociatedLet>, &'ast ast::AssociatedLet)>,
        Vec<(Ix<hir::Function>, &'ast ast::Function)>,
    ) {
        let mut struct_field_defs = vec![];
        let mut associated_lets = vec![];
        let mut functions = vec![];

        for (index, struct_item) in struct_items.iter().enumerate() {
            match struct_item {
                ast::StructItem::Field(struct_field_def) => {
                    struct_field_defs.push(struct_field_def);
                }
                ast::StructItem::Fn(function) => {
                    let index = Ix::push(&mut self.functions, StableIndex::None);
                    functions.push((index, function));
                }
                ast::StructItem::Let(associated_let) => {
                    let index = Ix::push(&mut self.associated_lets, StableIndex::None);
                    associated_lets.push((index, associated_let));
                }
            }
        }

        (struct_field_defs, associated_lets, functions)
    }

    fn lower_struct(
        &mut self,
        items: &[ast::StructItem],
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Ix<hir::Struct>, Error> {
        check_for_duplicate_item_names(items, self.ast_to_hir.interner)?;

        let (struct_field_defs, associated_lets, functions) = self.categorize_struct_items(items);

        // We have to claim our index so we can have a stable index to refer to.
        let struct_index = Ix::push(&mut self.structs, StableIndex::None);

        // Add associated lets and functions to the scope
        let mut scope = scope.enter_scope();

        scope.extend(associated_lets.iter().map(|&(index, associated_let)| {
            let symbol = self
                .ast_to_hir
                .interner
                .get_or_intern(&associated_let.let_.name.0);
            hir::Name {
                symbol,
                level: self.storage().level,
                kind: NameKind::AssociatedLet(index),
            }
        }));

        scope.extend(functions.iter().map(|&(index, function)| {
            let symbol = self.ast_to_hir.interner.get_or_intern(&function.name.0);
            hir::Name {
                symbol,
                level: self.storage().level,
                kind: NameKind::Function(index),
            }
        }));

        // Lower associated lets
        for &(index, associated_let) in &associated_lets {
            let let_index = self.lower_let(&associated_let.let_, &mut scope)?;
            self.associated_lets[index] = StableIndex::Some(hir::AssociatedLet {
                vis: associated_let.vis,
                let_index,
            });
        }

        // Lower functions
        for &(index, function) in &functions {
            let function = self.lower_function(function, &mut scope)?;
            self.functions[index] = StableIndex::Some(function);
        }

        // Lower fields
        let struct_field_defs = struct_field_defs
            .iter()
            .map(|struct_field_def| {
                let Span(type_value, _) =
                    self.lower_expr(&struct_field_def.struct_field.value, &mut scope)?;
                Ok(hir::StructFieldDef {
                    vis: struct_field_def.vis,
                    struct_field: hir::StructField {
                        name: self.lower_str(&struct_field_def.struct_field.name),
                        value: type_value,
                    },
                })
            })
            .collect::<Result<_, Error>>()?;

        // This part where we compute the decl mapping sucks and can be improved.
        let num_decls = functions.len() + associated_lets.len();
        let mut name_to_decl = HashMap::with_capacity(num_decls);

        for (index, _) in associated_lets {
            let let_index = self.associated_lets[index]
                .expect("lowered associated let")
                .let_index;
            let symbol = self.storage().lets[let_index].name.0;
            name_to_decl.insert(symbol, hir::DeclKind::AssociatedLet(index));
        }

        for (index, _) in functions {
            let function = self.functions[index].as_ref().expect("lowered function");
            let symbol = function.name.0;
            name_to_decl.insert(symbol, hir::DeclKind::Function(index));
        }

        assert_eq!(name_to_decl.len(), num_decls, "I did the math wrong");

        self.structs[struct_index] = StableIndex::Some(hir::Struct {
            struct_field_defs,
            name_to_decl,
            is_comptime_only: None,
        });

        Ok(struct_index)
    }

    fn lower_str(&mut self, Span(name, range): &Span<SmolStr>) -> Span<DefaultSymbol> {
        let symbol = self.ast_to_hir.interner.get_or_intern(name);
        Span(symbol, *range)
    }

    fn lower_expr_raw(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<hir::Expr>, Error> {
        match expr {
            ast::Expr::Int(Span(int, range)) => {
                let expr = hir::Expr::Value(Value::Int(*int));
                Ok(Span(expr, *range))
            }
            ast::Expr::Bool(Span(boolean, range)) => {
                let expr = hir::Expr::Value(Value::Bool(*boolean));
                Ok(Span(expr, *range))
            }
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
            ast::Expr::Name(name) => self.lower_name(name.clone(), scope),
            ast::Expr::Call(callee, Span(arguments, args_range)) => {
                let Span(callee, callee_range) = match callee {
                    ast::Callee::Expr(callee) => {
                        let Span(callee_index, callee_range) = self.lower_expr(callee, scope)?;
                        Span(hir::Callee::Expr(callee_index), callee_range)
                    }
                    ast::Callee::BuiltinFunction(Span(builtin, range)) => {
                        let builtin = builtin
                            .as_str()
                            .parse()
                            .map_err(|()| Error::UnknownBuiltin(builtin.clone(), *range))?;

                        Span(hir::Callee::BuiltinFunction(Span(builtin, *range)), *range)
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
                let Span(block_index, range) = self.lower_block(block, scope)?;

                Ok(Span(hir::Expr::Block(block_index), range))
            }
            ast::Expr::Comptime(expr) => {
                let Span(expr_index, range) = self.lower_expr(expr, scope)?;

                Ok(Span(hir::Expr::Comptime(expr_index), range))
            }
            ast::Expr::Struct(struct_) => {
                let Span(struct_, range) = struct_;
                let struct_index = self.lower_struct(&struct_.items, scope)?;
                let expr = hir::Expr::Value(Value::Struct(struct_index));

                Ok(Span(expr, *range))
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
        }
    }

    fn lower_expr(
        &mut self,
        expr: &ast::Expr,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<Ix<hir::Expr>>, Error> {
        let Span(expr, range) = self.lower_expr_raw(expr, scope)?;

        let index = self.storage().exprs.push(Span(expr, range));
        Ok(Span(index, range))
    }

    fn lower_block(
        &mut self,
        Span(block, range): &Span<ast::Block>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<Ix<hir::Block>>, Error> {
        let mut scope = scope.enter_scope();
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmts.push(self.lower_stmt(stmt, &mut scope)?);
        }

        let Span(returns, _) = self.lower_expr(&block.returns, &mut scope)?;
        let block = hir::Block { stmts, returns };

        Ok(Span(Ix::push(&mut self.storage().blocks, block), *range))
    }

    fn lower_stmt(
        &mut self,
        Span(stmt, _range): &Span<ast::Stmt>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<hir::Stmt, Error> {
        match stmt {
            ast::Stmt::Let(let_) => {
                let index = self.lower_let(let_, scope)?;
                let level = self.storage().level;
                let symbol = self.storage().lets[index].name.0;

                scope.push(hir::Name {
                    symbol,
                    level,
                    kind: NameKind::Let(index),
                });

                Ok(hir::Stmt::Let(index))
            }
        }
    }

    fn lower_let(
        &mut self,
        let_: &ast::Let,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Ix<hir::Let>, Error> {
        let Span(expr, _) = self.lower_expr(&let_.expr, scope)?;
        let ty = let_
            .ty
            .as_ref()
            .map(|ty| self.lower_expr(ty, scope).map(Span::into_inner))
            .transpose()?;

        let let_ = hir::Let {
            name: self.lower_str(&let_.name),
            ty,
            expr,
        };
        Ok(Ix::push(&mut self.storage().lets, let_))
    }

    fn lower_name(
        &mut self,
        Span(name, range): Span<SmolStr>,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<Span<hir::Expr>, Error> {
        let symbol = self.ast_to_hir.interner.get_or_intern(&name);

        // Look for name in the current scope.
        // This includes other functions.
        for (_, name) in scope.iter() {
            if symbol == name.symbol {
                return Ok(Span(hir::Expr::Name(*name), range));
            }
        }

        let Ok(builtin) = name.parse() else {
            return Err(Error::NameNotFound(name, range));
        };

        Ok(Span(hir::Expr::Value(Value::BuiltinType(builtin)), range))
    }

    fn lower_function(
        &mut self,
        function: &ast::Function,
        scope: &mut Scope<'_, hir::Name>,
    ) -> Result<hir::Function, Error> {
        let mut scope = scope.enter_scope();
        let mut params = RangeTable::new();

        for Span(param, range) in &function.params {
            let Span(ty, _) = self.lower_expr(&param.ty.0, &mut scope)?;

            let param = hir::Param {
                comptime: param.comptime,
                name: self.lower_str(&param.name),
                ty,
            };

            scope.push(hir::Name {
                symbol: param.name.0,
                level: self.storage().level,
                kind: NameKind::Param,
            });

            params.push(Span(param, *range));
        }

        let Span(return_type, _) = self.lower_expr(&function.return_type, &mut scope)?;
        let Span(body, _) = self.lower_block(&function.body, &mut scope)?;

        let name = self.ast_to_hir.lower_str(&function.name);

        Ok(hir::Function {
            vis: function.vis,
            name,
            return_type,
            body,
            params,
        })
    }
}
