//! Compile a Sig program to WebAssembly.

use crate::{thir, util::Ix};
use wasm_encoder::{
    BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
    Module, TypeSection, ValType,
};

fn ty_to_valtype(ty: thir::Type) -> ValType {
    match ty {
        thir::Type::Builtin(builtin) => match builtin {
            thir::Builtin::I32 => ValType::I32,
            thir::Builtin::Bool => ValType::I32,
        },
        thir::Type::Function(_) => todo!(),
    }
}

pub fn compile_functions(functions: &[thir::Function]) -> Vec<u8> {
    // Encode the type section.
    let mut type_section = TypeSection::new();
    for function in functions {
        let params = function
            .context
            .params
            .iter()
            .map(|param| ty_to_valtype(param.ty));

        let results = [ty_to_valtype(function.return_type)];
        type_section.function(params, results);
    }

    // Encode the function section.
    let mut function_section = FunctionSection::new();
    for function_index in Ix::iter(functions) {
        let index = fn_index_to_func_index(function_index);
        function_section.function(index);
    }

    // Encode the export section.
    let mut export_section = ExportSection::new();

    // Encode the code section.
    let mut code_section = CodeSection::new();
    for function in functions {
        let mut func = Function::new_with_locals_types(
            function
                .context
                .params
                .iter()
                .map(|param| param.ty)
                .chain(
                    function
                        .context
                        .lets
                        .iter()
                        .map(|let_| function.context.types[let_.expr.index]),
                )
                .map(ty_to_valtype),
        );

        compile_block(function.body, function, functions, &mut func);

        if function.name == "main()" {
            let main_id = code_section.len();
            export_section.export("main", ExportKind::Func, main_id);
        }

        func.instruction(&Instruction::End);

        code_section.function(&func);
    }

    let mut module = Module::new();
    module.section(&type_section);
    module.section(&function_section);
    module.section(&export_section);
    module.section(&code_section);
    module.finish()
}

// See https://webassembly.github.io/spec/core/syntax/modules.html#indices
fn let_index_to_local_index(let_index: Ix<thir::Let>, function: &thir::Function) -> u32 {
    function.context.params.len() as u32 + let_index.index as u32
}

fn param_index_to_local_index(param_index: Ix<thir::Param>) -> u32 {
    param_index.index as u32
}

fn fn_index_to_func_index(function_index: Ix<thir::Function>) -> u32 {
    function_index.index as u32
}

fn compile_block(
    block_index: Ix<thir::Block>,
    function: &thir::Function,
    functions: &[thir::Function],
    func: &mut Function,
) {
    let block = &function.context.blocks[..][block_index];
    for &stmt in &block.stmts {
        match stmt {
            thir::Stmt::Let(let_index) => {
                let let_expr = function.context.lets[..][let_index].expr;
                compile_expr(let_expr, function, functions, func);
                func.instruction(&Instruction::LocalSet(let_index_to_local_index(
                    let_index, function,
                )));
            }
        }
    }
    compile_expr(block.returns, function, functions, func);
}

fn compile_expr(
    expr_index: Ix<thir::Expr>,
    function: &thir::Function,
    functions: &[thir::Function],
    func: &mut Function,
) {
    match function.context.exprs[..][expr_index] {
        thir::Expr::Int(int) => {
            func.instruction(&Instruction::I32Const(int));
        }
        thir::Expr::Bool(boolean) => {
            let int_repr = if boolean { 1 } else { 0 };
            func.instruction(&Instruction::I32Const(int_repr));
        }
        thir::Expr::BinOp(op, lhs, rhs) => {
            compile_expr(lhs, function, functions, func);
            compile_expr(rhs, function, functions, func);
            let inst = match op {
                thir::BinOp::Add => Instruction::I32Add,
                thir::BinOp::Sub => Instruction::I32Sub,
                thir::BinOp::Mul => Instruction::I32Mul,
                thir::BinOp::Eq => Instruction::I32Eq,
            };
            func.instruction(&inst);
        }
        thir::Expr::IfThenElse(cond, then, else_) => {
            compile_expr(cond, function, functions, func);
            let block_return_type = ty_to_valtype(function.context.types[expr_index.index]);
            func.instruction(&Instruction::If(BlockType::Result(block_return_type)));

            // then
            compile_expr(then, function, functions, func);
            func.instruction(&Instruction::Else);

            // else
            compile_expr(else_, function, functions, func);
            func.instruction(&Instruction::End);
        }
        thir::Expr::Name(name) => match name {
            thir::Name::Let(let_index) => {
                func.instruction(&Instruction::LocalGet(let_index_to_local_index(
                    let_index, function,
                )));
            }
            thir::Name::Parameter(param_index) => {
                func.instruction(&Instruction::LocalGet(param_index_to_local_index(
                    param_index,
                )));
            }
            thir::Name::Function(_) => todo!("assigning functions to values"),
        },
        thir::Expr::Block(block_index) => compile_block(block_index, function, functions, func),
        thir::Expr::DirectCall(callee_index, ref arguments) => {
            for &arg in arguments {
                // push all arguments to the stack
                compile_expr(arg, function, functions, func);
            }
            let call_index = fn_index_to_func_index(callee_index);
            func.instruction(&Instruction::Call(call_index));
        }
        thir::Expr::IndirectCall(_, _) => todo!(),
    }
}
