use clap::Parser;
use sig_ast::{ast, ast_to_hir, hir, hir_to_thir, thir, thir_to_wasm};

/// Sig compiler
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The compiled input file to compile.
    name: String,

    /// The output file
    #[arg(short, long, default_value_t = String::from("out.wasm"))]
    out: String,

    /// The AST
    #[arg(long, default_value_t = false)]
    debug_ast: bool,
    /// The HIR
    #[arg(long, default_value_t = false)]
    debug_hir: bool,
    /// Print THIR
    #[arg(long, default_value_t = false)]
    debug_thir: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let program = std::fs::read_to_string(&args.name)?;
    let ast = sig_ast::parse::program(&program)?;

    let ast_functions: Vec<_> = ast
        .into_iter()
        .map(|ast::Item::Fn(function)| function)
        .collect();

    if args.debug_ast {
        for function in &ast_functions {
            println!("{:#?}", function);
        }
    }

    let hir_functions = ast_to_hir::entry(&ast_functions)?;

    if args.debug_hir {
        for function in &hir_functions {
            println!(
                "{:#?}",
                hir::printer::Printer::new(function, &hir_functions)
            );
        }
    }

    let (thir_functions, main) = hir_to_thir::entry(&hir_functions)?;

    let main_function = match main {
        hir_to_thir::ValueOrIx::Value(main_value) => {
            println!("Program output: {main_value}");
            return Ok(());
        }
        hir_to_thir::ValueOrIx::Index(main_index) => &thir_functions[..][main_index],
    };

    assert!(
        main_function.return_type == thir::Type::Builtin(thir::Builtin::I32),
        "main has to return an i32 for now"
    );

    if args.debug_thir {
        for function in &thir_functions {
            println!(
                "{:#?}",
                thir::printer::Printer::new(function, &thir_functions)
            );
        }
    }

    let bytes = thir_to_wasm::compile_functions(&thir_functions);
    std::fs::write(&args.out, &bytes)?;

    Ok(())
}
