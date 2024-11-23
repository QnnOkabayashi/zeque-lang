use clap::Parser;
use miette::{Diagnostic, GraphicalReportHandler, NamedSource};
use thiserror::Error;
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
}

#[derive(Debug, Diagnostic, Error)]
#[error("{reason}")]
pub struct Errors<E: Diagnostic> {
    #[source_code]
    pub code: NamedSource<String>,
    pub reason: &'static str,

    #[related]
    pub errors: Vec<E>,
}

impl<E: Diagnostic> Errors<E> {
    fn print_report(&self) {
        let mut buf = String::with_capacity(1024);
        GraphicalReportHandler::new()
            .render_report(&mut buf, self)
            .unwrap();

        println!("{buf}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
    let args = Args::parse();
    let program = std::fs::read_to_string(&args.name)?;
    let ast = match zeque_parse::parse(&program) {
        Ok(ast) => ast,
        Err(parse_error) => {
            Errors {
                code: NamedSource::new(&args.name, program).with_language("Zeque"),
                reason: "An error occurred during parsing",
                errors: vec![parse_error],
            }
            .print_report();
            return Ok(());
        }
    };

    if args.debug_ast {
        println!("{ast:#?}");
    }
    let hir = zeque_ast::ast_to_hir::entry(&ast);
    if !hir.errors.is_empty() {
        Errors {
            code: NamedSource::new(&args.name, program).with_language("Zeque"),
            reason: "An error occurred during AST -> HIR lowering",
            errors: hir.errors.raw,
        }
        .print_report();
        return Ok(());
    }
    if args.debug_hir {
        println!("{hir:#?}");
    }
    let start = std::time::Instant::now();
    let value = zeque_ast::sema::entry(hir)?;
    let end = start.elapsed();
    println!("{value:#?}");
    println!("Duration: {}μs", end.as_micros());

    Ok(())
}
