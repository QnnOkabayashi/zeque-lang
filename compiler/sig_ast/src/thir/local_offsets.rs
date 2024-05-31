use crate::thir::{Context, FunctionContext, Struct, TypeMetadata};

pub fn entry(context: &mut Context) {
    for function in &mut context.functions {
        entry_for_function(
            &mut function.context,
            &context.structs,
            &mut context.type_metadata,
        );
    }
}

fn entry_for_function(
    context: &mut FunctionContext,
    structs: &[Struct],
    type_metadata: &mut TypeMetadata,
) {
    let mut local_offset = 0;

    for param in &context.params {
        context.param_local_offsets.push(local_offset);
        local_offset += type_metadata.register_count(param.ty, structs);
    }

    for let_ in &context.lets {
        context.let_local_offsets.push(local_offset);
        let ty = let_
            .ty
            .unwrap_or_else(|| context.expr_types[let_.expr.index]);
        local_offset += type_metadata.register_count(ty, structs);
    }
}
