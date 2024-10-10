use crate::hir;
use index_vec::{IndexSlice, IndexVec};
use smol_str::SmolStr;
use std::collections::HashMap;
use std::mem;

pub fn entry(hir_context: hir::Hir) -> Value {
    let main_file = &hir_context.files[hir_context.main];
    let mut vm = VirtualMachine {
        stack: Vec::new(),
        mono_structs: IndexVec::new(),
        cached_fn_calls: HashMap::new(),
        structs: &hir_context.structs,
    };

    let mut main_ctx = Context {
        ctx: &main_file.ctx,
        current_stack_frame: StackFrame::new(&main_file.ctx, MonoStructIdx::new(0)),
        vm: &mut vm,
    };
    let mono_main_struct_idx = main_ctx.eval_struct(main_file.struct_idx);
    let mono_main_struct = &main_ctx.vm.mono_structs[mono_main_struct_idx];

    let main_fn_idx = mono_main_struct
        .fns
        .iter_enumerated()
        .find_map(|(fn_idx, mono_fn)| {
            let fn_name = main_ctx.vm.structs[mono_fn.struct_idx].fns[mono_fn.fn_idx]
                .name
                .as_str();
            (fn_name == "main").then_some(fn_idx)
        })
        .expect("no main found");

    main_ctx.eval_fn_call(mono_main_struct_idx, main_fn_idx, vec![])
}

#[derive(Debug)]
struct StackFrame {
    params: IndexVec<hir::ParamIdx, Value>,
    lets: IndexVec<hir::LetIdx, Value>,
    /// mapping from StructIdx in this context -> what it actually is
    mono_structs: HashMap<hir::StructIdx, MonoStructIdx>,
    parent_captures: IndexVec<hir::ParentRefIdx, Value>,
    self_type: MonoStructIdx,
}

impl StackFrame {
    fn new(ctx: &hir::Ctx, self_type: MonoStructIdx) -> Self {
        StackFrame {
            params: IndexVec::from_vec(vec![Value::Uninitialized; ctx.params.len()]),
            lets: IndexVec::from_vec(vec![Value::Uninitialized; ctx.lets.len()]),
            mono_structs: HashMap::new(),
            parent_captures: IndexVec::new(),
            self_type,
        }
    }
}

index_vec::define_index_type! {
    pub struct StackFrameIdx = u32;
}

index_vec::define_index_type! {
    pub struct MonoStructIdx = u32;
}

struct VirtualMachine<'hir> {
    stack: Vec<StackFrame>,
    mono_structs: IndexVec<MonoStructIdx, MonoStruct>,
    cached_fn_calls: HashMap<(MonoStructIdx, hir::FnIdx, Vec<Value>), Value>,
    structs: &'hir IndexSlice<hir::StructIdx, [hir::Struct]>,
}

struct Context<'a, 'hir> {
    ctx: &'hir hir::Ctx,
    current_stack_frame: StackFrame,
    vm: &'a mut VirtualMachine<'hir>,
}

impl<'a, 'hir> Context<'a, 'hir> {
    fn assert_value_is_type(&self, value: &Value, ty: &Type) {
        match (value, ty) {
            (Value::Int(_), Type::I32) => {}
            (Value::Bool(_), Type::Bool) => {}
            (
                Value::Constructor {
                    mono_struct_idx, ..
                },
                Type::Struct(type_struct_id),
            ) => {
                assert_eq!(
                    *mono_struct_idx, *type_struct_id,
                    "constructed value doesn't match struct type"
                )
            }
            (Value::Type(_), Type::Type) => {}
            (
                Value::Fn {
                    mono_struct_idx: _,
                    fn_idx: _,
                },
                Type::Fn {
                    // params: _,
                    // return_ty: _,
                },
            ) => {
                // don't do any assertions right now.
                // Eventually we want to check these better
                // let _mono_fn = &self.vm.mono_structs[*mono_struct_idx].fns[*fn_idx];
                // todo!("function types")
            }
            (Value::Uninitialized, _) => panic!("value is uninit"),
            _ => panic!("value '{value:?}' and type '{ty:?}' don't match"),
        }
    }

    fn eval_ty(&mut self, expr_idx: hir::ExprIdx) -> Type {
        match self.eval_expr(expr_idx) {
            Value::Type(ty) => ty,
            Value::Uninitialized => panic!("uninit"),
            _ => panic!("expected a type"),
        }
    }

    fn eval_block(&mut self, block: &hir::Block) -> Value {
        for stmt in &block.stmts {
            self.eval_stmt(stmt);
        }
        block
            .returns
            .map(|returns| self.eval_expr(returns))
            .unwrap_or(Value::Unit)
    }

    fn eval_expr(&mut self, expr_idx: hir::ExprIdx) -> Value {
        let expr = &self.ctx.exprs[expr_idx];
        match expr {
            hir::Expr::Int(int) => Value::Int(*int),
            hir::Expr::Bool(b) => Value::Bool(*b),
            hir::Expr::BinOp { op, lhs, rhs } => {
                let lhs = self.eval_expr(*lhs);
                let rhs = self.eval_expr(*rhs);
                match op {
                    hir::BinOp::Add => match (lhs, rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs + rhs),
                        _ => panic!("adding requires 2 ints"),
                    },
                    hir::BinOp::Sub => match (lhs, rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs - rhs),
                        _ => panic!("subtraction requires 2 ints"),
                    },
                    hir::BinOp::Mul => match (lhs, rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs * rhs),
                        _ => panic!("multiplication requires 2 ints"),
                    },
                    hir::BinOp::Eq => match (lhs, rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => Value::Bool(lhs == rhs),
                        _ => panic!("equals requires 2 ints (for now)"),
                    },
                }
            }
            hir::Expr::IfThenElse { cond, then, else_ } => {
                let Value::Bool(cond) = self.eval_expr(*cond) else {
                    panic!("cond not a bool");
                };
                if cond {
                    self.eval_expr(*then)
                } else {
                    self.eval_expr(*else_)
                }
            }
            hir::Expr::Name(name) => self.eval_name(*name),
            hir::Expr::BuiltinType(builtin_type) => Value::Type(Type::from(*builtin_type)),
            hir::Expr::SelfType => Value::Type(Type::Struct(self.current_stack_frame.self_type)),
            hir::Expr::Block(block) => self.eval_block(block),
            hir::Expr::Call { callee, args } => self.eval_call(callee, args),
            hir::Expr::Comptime(expr_idx) => self.eval_expr(*expr_idx),
            hir::Expr::Struct(struct_idx) => {
                Value::Type(Type::Struct(self.eval_struct(*struct_idx)))
            }
            hir::Expr::Constructor { ty, fields } => self.eval_constructor(ty, fields),
            hir::Expr::Field { expr, field_name } => self.eval_field(*expr, field_name),
            hir::Expr::FnType => Value::Type(Type::Fn {}),
            hir::Expr::Error => panic!("an error occurred earlier"),
        }
    }

    fn eval_name(&self, name: hir::Name) -> Value {
        match name {
            hir::Name::Local(local) => match local {
                hir::Local::Let(let_idx) => self.current_stack_frame.lets[let_idx].clone(),
                hir::Local::Param(param_idx) => self.current_stack_frame.params[param_idx].clone(),
                hir::Local::Fn(struct_idx, fn_idx) => {
                    let mono_struct_idx = self.current_stack_frame.mono_structs[&struct_idx];
                    Value::Fn {
                        mono_struct_idx,
                        fn_idx,
                    }
                }
            },
            hir::Name::ParentRef(parent_ref_idx) => {
                self.current_stack_frame.parent_captures[parent_ref_idx].clone()
            }
        }
    }

    fn eval_stmt(&mut self, stmt: &hir::Stmt) {
        match stmt {
            hir::Stmt::Let(let_idx) => {
                let expr_idx = self.ctx.lets[*let_idx].expr;
                let expr = self.eval_expr(expr_idx);
                self.current_stack_frame.lets[*let_idx] = expr;
            }
        }
    }

    fn eval_call(&mut self, callee: &hir::Callee, args: &[hir::ExprIdx]) -> Value {
        match callee {
            hir::Callee::Expr(expr_idx) => {
                // Evaluate the function and the args
                let expr = self.eval_expr(*expr_idx);
                let Value::Fn {
                    mono_struct_idx,
                    fn_idx,
                } = expr
                else {
                    panic!("tried to call non-fn expression, was {expr:?}");
                };
                let args = args
                    .iter()
                    .map(|expr_idx| self.eval_expr(*expr_idx))
                    .collect();

                self.eval_fn_call(mono_struct_idx, fn_idx, args)
            }
            hir::Callee::Builtin(builtin_fn) => self.eval_builtin_call(builtin_fn.clone(), args),
        }
    }

    fn eval_fn_call(
        &mut self,
        mono_struct_idx: MonoStructIdx,
        fn_idx: hir::FnIdx,
        args: Vec<Value>,
    ) -> Value {
        let key = (mono_struct_idx, fn_idx, args);
        if self.vm.cached_fn_calls.contains_key(&key) {
            return self.vm.cached_fn_calls[&key].clone();
        }

        let value = self.eval_fn_call_uncached(mono_struct_idx, fn_idx, &key.2);
        self.vm.cached_fn_calls.insert(key, value.clone());

        value
    }

    fn eval_fn_call_uncached(
        &mut self,
        mono_struct_idx: MonoStructIdx,
        fn_idx: hir::FnIdx,
        args: &[Value],
    ) -> Value {
        let mono_fn = &self.vm.mono_structs[mono_struct_idx].fns[fn_idx];
        let fn_decl = &self.vm.structs[mono_fn.struct_idx].fns[mono_fn.fn_idx];

        let mut callee_stack_frame = StackFrame::new(&fn_decl.ctx, mono_struct_idx);
        callee_stack_frame.parent_captures = mono_fn.parent_refs.clone();

        // Make self represent the callee context
        self.vm.stack.push(mem::replace(
            &mut self.current_stack_frame,
            callee_stack_frame,
        ));
        let caller_ctx = self.ctx;
        self.ctx = &fn_decl.ctx;

        // first, we need to figure out the types of everything.
        // make sure this params and args line up
        assert_eq!(
            self.ctx.params.len(),
            args.len(),
            "fn call doesn't have right number of arguments"
        );

        // go through and evaluate types
        for (param_idx, arg) in fn_decl.params.iter().zip(args) {
            let param = &fn_decl.ctx.params[*param_idx];
            // type check
            let ty = self.eval_ty(param.ty);
            self.assert_value_is_type(arg, &ty);
            // looks good, insert into the params array
            self.current_stack_frame.params[*param_idx] = arg.clone();
        }

        assert!(
            self.current_stack_frame
                .params
                .iter()
                .all(|value| !value.is_uninit()),
            "some of the params weren't initialized?"
        );

        // also need to do return type now
        let return_ty = fn_decl
            .return_ty
            .map(|return_ty| self.eval_ty(return_ty))
            .unwrap_or(Type::Unit);

        // Now we execute the function
        let value = self.eval_block(&fn_decl.body);

        // Ensure that it returned a value whose type matches the return type.
        self.assert_value_is_type(&value, &return_ty);

        // Make self represent the caller context again.
        self.ctx = caller_ctx;
        let caller_stack_frame = self
            .vm
            .stack
            .pop()
            .expect("put on a stack frame, should still be there");
        self.current_stack_frame = caller_stack_frame;

        value
    }

    fn eval_builtin_call(&mut self, builtin_fn: hir::BuiltinFn, args: &[hir::ExprIdx]) -> Value {
        match builtin_fn {
            hir::BuiltinFn::InComptime => {
                assert!(args.is_empty(), "@in_comptime() expects no arguments");
                Value::Bool(true)
            }
            hir::BuiltinFn::Trap => {
                assert!(args.is_empty(), "@trap() expects no arguments");
                panic!("reached a trap at comptime")
            }
            hir::BuiltinFn::Clz => {
                assert!(args.len() == 1, "@clz(..) expects 1 argument");
                let expr = self.eval_expr(args[0]);
                let Value::Int(i) = expr else {
                    panic!("@clz(_) expects an integer");
                };

                let leading_zeros = i.leading_zeros();
                Value::Int(leading_zeros as i32)
            }
            hir::BuiltinFn::Ctz => {
                assert!(args.len() == 1, "@ctz(..) expects 1 argument");
                let expr = self.eval_expr(args[0]);
                let Value::Int(i) = expr else {
                    panic!("@ctz(_) expects an integer");
                };

                let trailing_zeros = i.trailing_zeros();
                Value::Int(trailing_zeros as i32)
            }
            hir::BuiltinFn::Unknown(unknown) => panic!("builtin `{unknown}` not recognized"),
        }
    }

    fn eval_struct(&mut self, struct_idx: hir::StructIdx) -> MonoStructIdx {
        let mono_struct_idx_placeholder = self.vm.mono_structs.push(MonoStruct::default());
        let parent_self_type = mem::replace(
            &mut self.current_stack_frame.self_type,
            mono_struct_idx_placeholder,
        );
        let schema = &self.vm.structs[struct_idx];
        self.current_stack_frame
            .mono_structs
            .insert(struct_idx, mono_struct_idx_placeholder);

        let mut fns = IndexVec::with_capacity(schema.fns.len());

        // Go through the functions
        for (fn_idx, fn_decl) in schema.fns.iter_enumerated() {
            let parent_refs = fn_decl
                .ctx
                .parent_captures
                .iter()
                .map(|parent_ref| match parent_ref {
                    hir::ParentRef::Local(local) => match *local {
                        hir::Local::Let(let_idx) => self.current_stack_frame.lets[let_idx].clone(),
                        hir::Local::Param(param_idx) => {
                            self.current_stack_frame.params[param_idx].clone()
                        }
                        hir::Local::Fn(struct_idx, fn_idx) => {
                            let mono_struct_idx =
                                self.current_stack_frame.mono_structs[&struct_idx];
                            Value::Fn {
                                mono_struct_idx,
                                fn_idx,
                            }
                        }
                    },
                    hir::ParentRef::Capture(parent_ref_idx) => {
                        self.current_stack_frame.parent_captures[*parent_ref_idx].clone()
                    }
                })
                .collect();

            fns.push(MonoFn {
                struct_idx,
                fn_idx,
                parent_refs,
            });
        }

        let fields: HashMap<SmolStr, Type> = schema
            .fields
            .iter()
            .map(|field| (field.name.clone(), self.eval_ty(field.ty)))
            .collect();

        self.vm.mono_structs[mono_struct_idx_placeholder] = MonoStruct { fns, fields };
        self.current_stack_frame.self_type = parent_self_type;
        mono_struct_idx_placeholder
    }

    fn eval_constructor(
        &mut self,
        ty: &Option<hir::ExprIdx>,
        fields: &[hir::ConstructorField],
    ) -> Value {
        let ty_expr_idx = ty.expect("anonymous constructors are not supported yet");
        let ty_expr = self.eval_ty(ty_expr_idx);
        let struct_idx = match &ty_expr {
            Type::Struct(struct_idx) => *struct_idx,
            _ => panic!("expected a struct"),
        };

        let schema = &self.vm.mono_structs[struct_idx].fields;
        assert_eq!(fields.len(), schema.len());

        struct Usage {
            // todo: add useful info to this struct,
            // like a line number or something
        }

        let mut schema_field_usages: HashMap<SmolStr, Vec<Usage>> = schema
            .keys()
            .map(|field_name| (field_name.clone(), Vec::with_capacity(1)))
            .collect();

        let fields: Vec<(SmolStr, Value)> = fields
            .iter()
            .map(|field| {
                let usages = schema_field_usages
                    .get_mut(&field.name)
                    .unwrap_or_else(|| panic!("field '{}' not defined in the type", field.name));
                usages.push(Usage {});

                let value = self.eval_expr(field.value);

                // Type validation
                let ty = &self.vm.mono_structs[struct_idx].fields[&field.name];
                self.assert_value_is_type(&value, ty);

                (field.name.clone(), value)
            })
            .collect();

        for (field, usages) in schema_field_usages {
            match usages.len() {
                0 => panic!("no value provided for field '{field}'"),
                1 => {}
                _ => panic!("multiple values provided for field '{field}'"),
            }
        }

        Value::Constructor {
            mono_struct_idx: struct_idx,
            fields,
        }
    }

    /// Evaluate a field access, e.g., `foo.bar`
    fn eval_field(&mut self, expr_idx: hir::ExprIdx, field_name: &str) -> Value {
        let expr = self.eval_expr(expr_idx);

        match expr {
            Value::Constructor {
                mono_struct_idx: struct_idx,
                fields,
            } => {
                // Getting a field from an instance of a value

                // this is kinda just a sanity check.
                // technically it would also be caught be the below
                // unreachable!() call, but by checking the schema first
                // and then doing an unreachable at the end, we assert more invariants.
                let schema_has_field = self.vm.mono_structs[struct_idx]
                    .fields
                    .iter()
                    .any(|(name, _)| name.as_str() == field_name);
                assert!(
                    schema_has_field,
                    "field '{field_name}' not in constructed value"
                );

                for (name, value) in fields {
                    if name.as_str() == field_name {
                        return value;
                    }
                }
                unreachable!("should have failed the schema check");
            }
            Value::Type(ty) => match ty {
                Type::Struct(mono_struct_idx) => {
                    // Getting a field from a type. Right now, there's just fns
                    let schema = &self.vm.mono_structs[mono_struct_idx];
                    for (fn_idx, mono_fn) in schema.fns.iter_enumerated() {
                        let fn_decl_name = self.vm.structs[mono_fn.struct_idx].fns[mono_fn.fn_idx]
                            .name
                            .as_str();
                        if fn_decl_name == field_name {
                            return Value::Fn {
                                mono_struct_idx,
                                fn_idx,
                            };
                        }
                    }
                    panic!("field '{field_name}' doesn't exist for this struct")
                }
                _ => panic!("expected a struct type, found another type: {ty:?}"),
            },
            Value::Uninitialized => panic!("uninit"),
            _ => panic!("expected a struct type, or an instance of a struct"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Constructor {
        mono_struct_idx: MonoStructIdx,
        fields: Vec<(SmolStr, Value)>,
    },
    Fn {
        mono_struct_idx: MonoStructIdx,
        fn_idx: hir::FnIdx,
    },
    Type(Type),
    Unit,
    /// used as a placeholder value. if this is ever used, it's an impl bug.
    Uninitialized,
}

impl Value {
    fn is_uninit(&self) -> bool {
        matches!(self, Value::Uninitialized)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    I32,
    Bool,
    Struct(MonoStructIdx),
    Fn {
        // for now it will just be a generic "callable" type.
        // we won't know if it's okay until we call it.
        // params: Vec<Type>,
        // return_ty: Box<Type>,
    },
    Type,
    Unit,
}

#[derive(Clone, Debug, Default)]
pub struct MonoStruct {
    pub fns: IndexVec<hir::FnIdx, MonoFn>,
    pub fields: HashMap<SmolStr, Type>,
}

/// Monomorphized function that has captured external values that it uses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MonoFn {
    /// The struct that the fn was declared in.
    struct_idx: hir::StructIdx,
    /// The fn within the struct
    fn_idx: hir::FnIdx,
    /// Concrete values for all foreign-defined references
    parent_refs: IndexVec<hir::ParentRefIdx, Value>,
}

impl From<hir::BuiltinType> for Type {
    fn from(value: hir::BuiltinType) -> Self {
        match value {
            hir::BuiltinType::I32 => Type::I32,
            hir::BuiltinType::Bool => Type::Bool,
            hir::BuiltinType::Type => Type::Type,
        }
    }
}
