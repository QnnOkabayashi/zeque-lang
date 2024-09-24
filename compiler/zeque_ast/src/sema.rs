use crate::hir;
use index_vec::IndexVec;
use smol_str::SmolStr;

pub fn entry(file: hir::File) {
    let main_struct = file.struct_();
    let main_idx = find_main_function(main_struct).expect("no main");
    let main = &main_struct.fns[main_idx];
    assert!(main.params.is_empty(), "main shouldn't have any params");

    // assert that the return type evaluates to i32

    todo!()
}

fn find_main_function(struct_: &hir::Struct) -> Option<hir::FnIdx> {
    struct_
        .fns
        .iter_enumerated()
        .find_map(|(fn_idx, fn_decl)| (fn_decl.name == "main").then_some(fn_idx))
}

struct StackFrame {
    params: IndexVec<hir::ParamIdx, Value>,
    exprs: IndexVec<hir::ExprIdx, Value>,
    lets: IndexVec<hir::LetIdx, hir::Let>,
    structs: IndexVec<hir::StructIdx, Struct>,

    level: hir::Level,
    parent_stack_frame: Option<StackFrameIdx>,
}

impl StackFrame {
    fn new(ctx: &hir::Ctx, level: hir::Level, parent: Option<StackFrameIdx>) -> Self {
        StackFrame {
            params: IndexVec::from_vec(vec![Value::Uninit; ctx.params.len()]),
            exprs: IndexVec::from_vec(vec![Value::Uninit; ctx.exprs.len()]),
            lets: IndexVec::with_capacity(ctx.lets.len()),
            structs: IndexVec::with_capacity(ctx.structs.len()),
            level,
            parent_stack_frame: parent,
        }
    }
}

index_vec::define_index_type! {
    pub struct StackFrameIdx = u32;
}

struct Context<'a> {
    // This is essentially a call stack, we can go back because InterpreterCtxs remember who
    // they were called from.
    current_stack_frame_idx: StackFrameIdx,
    // contexts knows all contexts that exist, or have existed.
    // Remembering all contexts allows us to refer to past ones. They should also remember
    // their parents as well.
    stack_frames: &'a mut IndexVec<StackFrameIdx, StackFrame>,
    ctx: &'a hir::Ctx,
}

impl<'a> Context<'a> {
    fn idx_at_level(&mut self, level: hir::Level) -> StackFrameIdx {
        let mut idx = self.current_stack_frame_idx;
        while self.stack_frames[idx].level != level {
            idx = self.stack_frames[idx]
                .parent_stack_frame
                .expect("should not escape context stack");
        }
        idx
    }

    fn stack_frame(&mut self, idx: StackFrameIdx) -> &mut StackFrame {
        &mut self.stack_frames[idx]
    }

    fn current_stack_frame(&mut self) -> &mut StackFrame {
        self.stack_frame(self.current_stack_frame_idx)
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
                    hir::BinOp::Add => todo!(),
                    hir::BinOp::Sub => todo!(),
                    hir::BinOp::Mul => todo!(),
                    hir::BinOp::Eq => todo!(),
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
            hir::Expr::Name(name) => match name {
                hir::Name::Local(local) => {
                    let idx = self.idx_at_level(local.level);
                    let stack_frame = self.stack_frame(idx);

                    match local.kind {
                        hir::LocalKind::Let(let_idx) => {
                            let let_expr_idx = stack_frame.lets[let_idx].expr;
                            stack_frame.exprs[let_expr_idx].clone()
                        }
                        hir::LocalKind::Param(param_idx) => stack_frame.params[param_idx].clone(),
                        hir::LocalKind::Fn(struct_idx, fn_idx) => Value::Fn {
                            idx,
                            struct_idx,
                            fn_idx,
                        },
                    }
                }
                hir::Name::BuiltinType(builtin_type) => Value::Type(Type::from(*builtin_type)),
            },
            hir::Expr::Block(block) => {
                for stmt in &block.stmts {
                    self.eval_stmt(stmt);
                }
                self.eval_expr(block.returns)
            }
            hir::Expr::Call { callee, args } => todo!(),
            hir::Expr::Comptime(expr_idx) => self.eval_expr(*expr_idx),
            hir::Expr::Struct(_) => todo!(),
            hir::Expr::Constructor { ty, fields } => todo!(),
            hir::Expr::Field { expr, field_name } => todo!(),
            hir::Expr::Error => todo!(),
        }
    }

    fn eval_stmt(&mut self, stmt: &hir::Stmt) {
        match stmt {
            hir::Stmt::Let(let_idx) => {
                let stack_frame = self.stack_frame(self.current_stack_frame_idx);
                let let_ = &self.ctx.lets[*let_idx];

                // we need to examine the let in the original ctx
                // to know what to do with it.
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Constructor {
        ty: Type,
        fields: Vec<(SmolStr, Value)>,
    },
    Fn {
        idx: StackFrameIdx,
        struct_idx: hir::StructIdx,
        fn_idx: hir::FnIdx,
    },
    Type(Type),
    // used as a placeholder value. if this is ever used, it's an impl bug.
    Uninit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    I32,
    Bool,
    Struct(Struct),
    Type,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Struct {
    /// The context of all that which came before us
    /// When a function or field says "oh I captured from level n-2",
    /// we go to THIS CtxIdx and rewind from there.
    pub idx: StackFrameIdx,
    pub fns: Vec<hir::FnIdx>,
    pub fields: Vec<(SmolStr, Type)>,
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
