//! Compile a Sig program to WebAssembly.

use crate::{thir, util::Ix};
use std::collections::hash_map::{Entry, HashMap};
use string_interner::DefaultSymbol;
use wasm_encoder::{
    BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
    Module, TypeSection, ValType,
};

struct MemoizeBlockType<'a> {
    type_section: &'a mut TypeSection,
    structs: &'a [thir::Struct],
    memo: HashMap<thir::Type, BlockType>,
}

impl<'a> MemoizeBlockType<'a> {
    fn new(type_section: &'a mut TypeSection, structs: &'a [thir::Struct]) -> Self {
        MemoizeBlockType {
            type_section,
            structs,
            memo: HashMap::new(),
        }
    }

    fn block_type(&mut self, ty: thir::Type) -> BlockType {
        match self.memo.entry(ty) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                // this allocates on each call :(
                let result = collect_valtypes(self.structs, Some(ty));
                let block_type = if let [valtype] = result.as_slice() {
                    BlockType::Result(*valtype)
                } else {
                    let index = self.type_section.len();
                    self.type_section.function([], result);
                    BlockType::FunctionType(index)
                };
                vacant.insert(block_type);
                block_type
            }
        }
    }
}

/// Flattens structs into their primitive parts.
fn ty_to_valtype(ty: thir::Type, structs: &[thir::Struct], valtypes: &mut Vec<ValType>) {
    match ty {
        thir::Type::Builtin(builtin) => match builtin {
            thir::Builtin::I32 => valtypes.push(ValType::I32),
            thir::Builtin::Bool => valtypes.push(ValType::I32),
        },
        thir::Type::Function(_) => todo!(),
        thir::Type::Struct(struct_index) => {
            for field in &structs[struct_index].fields {
                ty_to_valtype(field.ty, structs, valtypes);
            }
        }
    }
}

fn collect_valtypes<I>(structs: &[thir::Struct], iter: I) -> Vec<ValType>
where
    I: IntoIterator<Item = thir::Type>,
{
    let mut valtypes = vec![];
    for ty in iter {
        ty_to_valtype(ty, structs, &mut valtypes);
    }
    valtypes
}

pub fn entry(context: &mut thir::Context, main_symbol: DefaultSymbol) -> Vec<u8> {
    // Encode the type section.
    let mut type_section = TypeSection::new();
    for function in &context.functions {
        let params = collect_valtypes(
            &context.structs,
            function.context.params.iter().map(|param| param.ty),
        );
        let results = collect_valtypes(&context.structs, Some(function.return_type));

        type_section.function(params, results);
    }

    let mut memoize_block_type = MemoizeBlockType::new(&mut type_section, &context.structs);

    // Encode the function section.
    let mut function_section = FunctionSection::new();
    for function_index in Ix::iter(&context.functions) {
        let index = fn_index_to_func_index(function_index);
        function_section.function(index);
    }

    // Encode the export section.
    let mut export_section = ExportSection::new();

    // Encode the code section.
    let mut code_section = CodeSection::new();
    for function in &context.functions {
        let mut func = Function::new_with_locals_types(collect_valtypes(
            &context.structs,
            function.context.params.iter().map(|param| param.ty).chain(
                function
                    .context
                    .lets
                    .iter()
                    .map(|let_| function.context.expr_types[let_.expr.index]),
            ),
        ));

        compile_block(
            function.body,
            function,
            &context.functions,
            &mut func,
            &mut memoize_block_type,
            &context.structs,
            &mut context.type_metadata,
            &mut vec![],
        );

        if function.name.0 == main_symbol && function.filled_args.is_empty() {
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
fn set_binding(offset: u32, num_registers: u32, func: &mut Function) {
    for reg in (0..num_registers).rev() {
        func.instruction(&Instruction::LocalSet(offset + reg));
    }
}

fn get_binding(offset: u32, num_registers: u32, func: &mut Function) {
    for reg in 0..num_registers {
        func.instruction(&Instruction::LocalGet(offset + reg));
    }
}

// move values off the stack into the let binding stuff
fn set_let_binding(
    index: Ix<thir::Let>,
    context: &thir::FunctionContext,
    structs: &[thir::Struct],
    func: &mut Function,
    type_metadata: &mut thir::TypeMetadata,
) {
    set_binding(
        context.let_local_offsets[index.index],
        type_metadata.register_count(type_of_let(index, context), structs),
        func,
    );
}

// Load a let binding.
// If the value is a struct, and we're immediately accessing a
// field of that struct, only load that field from memory.
fn get_let_binding(
    accessed_fields: &[DefaultSymbol],
    index: Ix<thir::Let>,
    context: &thir::FunctionContext,
    structs: &[thir::Struct],
    func: &mut Function,
    type_metadata: &mut thir::TypeMetadata,
) {
    let thir::OffsetAndLen {
        offset,
        num_registers,
    } = type_metadata.registers(type_of_let(index, context), structs, accessed_fields);

    get_binding(
        context.let_local_offsets[index.index] + offset,
        num_registers,
        func,
    );
}

fn get_param(
    accessed_fields: &[DefaultSymbol],
    index: Ix<thir::Param>,
    context: &thir::FunctionContext,
    structs: &[thir::Struct],
    func: &mut Function,
    type_metadata: &mut thir::TypeMetadata,
) {
    let thir::OffsetAndLen {
        offset,
        num_registers,
    } = type_metadata.registers(context.params[index].ty, structs, accessed_fields);

    get_binding(
        context.param_local_offsets[index.index] + offset,
        num_registers,
        func,
    );
}

fn type_of_let(index: Ix<thir::Let>, context: &thir::FunctionContext) -> thir::Type {
    let l = &context.lets[index];
    // Either the type that it was specified to be, or the type of the expression
    l.ty.unwrap_or_else(|| context.expr_types[l.expr.index])
}

fn fn_index_to_func_index(function_index: Ix<thir::Function>) -> u32 {
    function_index.index as u32
}

fn compile_block(
    block_index: Ix<thir::Block>,
    function: &thir::Function,
    functions: &[thir::Function],
    func: &mut Function,
    memoize_block_type: &mut MemoizeBlockType,
    structs: &[thir::Struct],
    type_metadata: &mut thir::TypeMetadata,
    accessed_fields: &mut Vec<DefaultSymbol>,
) {
    let block = &function.context.blocks[block_index];
    for &stmt in &block.stmts {
        match stmt {
            thir::Stmt::Let(let_index) => {
                let let_expr = function.context.lets[let_index].expr;
                compile_expr(
                    let_expr,
                    function,
                    functions,
                    func,
                    memoize_block_type,
                    structs,
                    type_metadata,
                    &mut vec![],
                );
                set_let_binding(let_index, &function.context, structs, func, type_metadata);
            }
        }
    }
    compile_expr(
        block.returns,
        function,
        functions,
        func,
        memoize_block_type,
        structs,
        type_metadata,
        accessed_fields,
    );
}

fn compile_expr(
    expr_index: Ix<thir::Expr>,
    function: &thir::Function,
    functions: &[thir::Function],
    func: &mut Function,
    memoize_block_type: &mut MemoizeBlockType,
    structs: &[thir::Struct],
    type_metadata: &mut thir::TypeMetadata,
    // outermost-access is at the front, innermost-access is at the end.
    accessed_fields: &mut Vec<DefaultSymbol>,
) {
    match function.context.exprs[expr_index] {
        thir::Expr::Int(int) => {
            func.instruction(&Instruction::I32Const(int));
        }
        thir::Expr::Bool(boolean) => {
            let int_repr = if boolean { 1 } else { 0 };
            func.instruction(&Instruction::I32Const(int_repr));
        }
        thir::Expr::BinOp(op, lhs, rhs) => {
            assert!(accessed_fields.is_empty());
            compile_expr(
                lhs,
                function,
                functions,
                func,
                memoize_block_type,
                structs,
                type_metadata,
                &mut vec![],
            );
            compile_expr(
                rhs,
                function,
                functions,
                func,
                memoize_block_type,
                structs,
                type_metadata,
                &mut vec![],
            );
            let inst = match op {
                thir::BinOp::Add => Instruction::I32Add,
                thir::BinOp::Sub => Instruction::I32Sub,
                thir::BinOp::Mul => Instruction::I32Mul,
                thir::BinOp::Eq => Instruction::I32Eq,
            };
            func.instruction(&inst);
        }
        thir::Expr::IfThenElse(cond, then, else_) => {
            // values in blocks need to already live in a local.
            compile_expr(
                cond,
                function,
                functions,
                func,
                memoize_block_type,
                structs,
                type_metadata,
                &mut vec![],
            );
            let block_type =
                memoize_block_type.block_type(function.context.expr_types[expr_index.index]);

            func.instruction(&Instruction::If(block_type));
            compile_expr(
                then,
                function,
                functions,
                func,
                memoize_block_type,
                structs,
                type_metadata,
                accessed_fields,
            );

            func.instruction(&Instruction::Else);
            compile_expr(
                else_,
                function,
                functions,
                func,
                memoize_block_type,
                structs,
                type_metadata,
                accessed_fields,
            );

            func.instruction(&Instruction::End);
        }
        thir::Expr::Name(name) => match name {
            thir::Name::Let(let_index) => {
                get_let_binding(
                    accessed_fields,
                    let_index,
                    &function.context,
                    structs,
                    func,
                    type_metadata,
                );
            }
            thir::Name::Parameter(param_index) => {
                get_param(
                    accessed_fields,
                    param_index,
                    &function.context,
                    structs,
                    func,
                    type_metadata,
                );
            }
            thir::Name::Function(_) => todo!("assigning functions to values"),
        },
        thir::Expr::Block(block_index) => compile_block(
            block_index,
            function,
            functions,
            func,
            memoize_block_type,
            structs,
            type_metadata,
            accessed_fields,
        ),
        thir::Expr::DirectCall(callee_index, ref arguments) => {
            assert!(
                accessed_fields.is_empty(),
                "cannot access fields directly on the result of a function call (for now)"
            );
            for &arg in arguments {
                // push all arguments to the stack
                compile_expr(
                    arg,
                    function,
                    functions,
                    func,
                    memoize_block_type,
                    structs,
                    type_metadata,
                    &mut vec![],
                );
            }
            let call_index = fn_index_to_func_index(callee_index);
            func.instruction(&Instruction::Call(call_index));
        }
        thir::Expr::IndirectCall(_, _) => todo!("indirect calls"),
        thir::Expr::Constructor(struct_index, ref ctor) => {
            if let Some(field_name) = accessed_fields.pop() {
                // We're directly accessing a field, so we only evaluate the field that gets
                // accessed.
                let field_expr = ctor
                    .fields
                    .iter()
                    .find(|field| field.name.0 == field_name)
                    .expect("constructor has same field names as struct")
                    .expr;
                compile_expr(
                    field_expr,
                    function,
                    functions,
                    func,
                    memoize_block_type,
                    structs,
                    type_metadata,
                    accessed_fields,
                );
            } else {
                // for now, we evaluate in the order that fields are defined (cause of Repr::C),
                // instead of the order that fields in the constructor are written.
                // This is because otherwise we have to get memory involved.
                for (field_name, _align) in &structs[struct_index].struct_sizealign.field_ordering {
                    let field_expr = ctor
                        .fields
                        .iter()
                        .find(|field| field.name.0 == *field_name)
                        .expect("constructor has same field names as struct")
                        .expr;
                    compile_expr(
                        field_expr,
                        function,
                        functions,
                        func,
                        memoize_block_type,
                        structs,
                        type_metadata,
                        accessed_fields,
                    );
                }
            }
        }
        thir::Expr::Field(expr, ref field_name) => {
            accessed_fields.push(field_name.0);
            compile_expr(
                expr,
                function,
                functions,
                func,
                memoize_block_type,
                structs,
                type_metadata,
                accessed_fields,
            );
        }
    }
}

// fn compile_store(
//     ty: thir::Type,
//     memarg: MemArg,
//     function: &thir::Function,
//     functions: &[thir::Function],
//     func: &mut Function,
//     memoize_block_type: &mut MemoizeBlockType,
//     structs: &[thir::Struct],
// ) {
//     match ty {
//         thir::Type::Builtin(builtin) => match builtin {
//             thir::Builtin::I32 => {
//                 func.instruction(&Instruction::I32Store(memarg));
//             }
//             thir::Builtin::Bool => {
//                 func.instruction(&Instruction::I32Store8(memarg));
//             }
//         },
//         thir::Type::Function(_) => {
//             // functions (but not fn ptrs) do not exist at runtime since they're ZSTs.
//         }
//         thir::Type::Struct(index) => {
//             let struct_sizealign = &structs[index].struct_sizealign;
//             // We need to go backwards because they're on the stack backwards.
//             for (field, align) in struct_sizealign.field_ordering.iter().rev() {
//                 let offset: u64 = struct_sizealign.field_to_byte_offset[field];
//                 let field_memarg = MemArg {
//                     offset: memarg.offset + offset,
//                     align: align.0,
//                     memory_index: DEFAULT_MEMORY_INDEX,
//                 };
//             }
//         }
//     }
// }
