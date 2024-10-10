use clap::Parser;
// use zeque_ast::{ast, ast_to_hir, hir, hir_to_thir, thir, thir_to_wasm};

/// Zeque compiler
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The compiled input file to compile.
    name: String,

    // /// The output file
    // #[arg(short, long, default_value_t = String::from("out.wasm"))]
    // out: String,

    /// Debug print the AST
    #[arg(long, default_value_t = false)]
    debug_ast: bool,

    /// Debug print the HIR
    #[arg(long, default_value_t = false)]
    debug_hir: bool,

    // /// Debug print the THIR
    // #[arg(long, default_value_t = false)]
    // debug_thir: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("RUST_BACKTRACE", "1");
    let args = Args::parse();
    let program = std::fs::read_to_string(&args.name)?;
    let ast = zeque_ast::parse::file(&program)?;

    if args.debug_ast {
        println!("{ast:#?}");
    }
    let hir = zeque_ast::ast_to_hir::entry(&ast)?;
    if args.debug_hir {
        println!("{hir:#?}");
    }
    let start = std::time::Instant::now();
    let value = zeque_ast::sema::entry(hir);
    let end = start.elapsed();
    println!("{value:#?}");
    println!("Duration: {}ns", end.as_nanos());

    //
    // let ast_functions: Vec<_> = ast
    //     .into_iter()
    //     .map(|ast::Decl::Fn(function)| function)
    //     .collect();
    //
    // if args.debug_ast {
    //     for function in &ast_functions {
    //         println!("{:#?}", function);
    //     }
    // }
    //
    // let (hir_functions, mut interner) = ast_to_hir::entry(&ast_functions)?;
    // let main_symbol = interner.get_or_intern("main");
    //
    // // for (symbol, str) in interner.iter() {
    // //     println!("{symbol:?}: {str}");
    // // }
    //
    // if args.debug_hir {
    //     for function in &hir_functions {
    //         println!(
    //             "{:#?}",
    //             hir::printer::Printer::new(function, &hir_functions, &interner)
    //         );
    //     }
    // }
    //
    // let (mut thir_context, main) = hir_to_thir::entry(&hir_functions, main_symbol)?;
    //
    // let main_function = match main {
    //     hir_to_thir::ValueOrIx::Value(main_value) => {
    //         println!("Program output: {main_value}");
    //         return Ok(());
    //     }
    //     hir_to_thir::ValueOrIx::Index(main_index) => &thir_context.functions[..][main_index],
    // };
    //
    // assert!(
    //     main_function.return_type == thir::Type::Builtin(thir::Builtin::I32),
    //     "main has to return an i32 for now"
    // );
    //
    // if args.debug_thir {
    //     for function in &thir_context.functions {
    //         println!(
    //             "{:#?}",
    //             thir::printer::Printer::new(
    //                 function,
    //                 &thir_context.functions,
    //                 &thir_context.structs,
    //                 &interner
    //             )
    //         );
    //     }
    // }
    //
    // let bytes = thir_to_wasm::entry(&mut thir_context, main_symbol);
    // std::fs::write(&args.out, bytes)?;

    Ok(())
}
