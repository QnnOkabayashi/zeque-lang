use clap::Parser;
use wasmtime::*;

/// Sig runtime
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The compiled .wasm file to run
    name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.name)?;

    let engine = Engine::default();

    let module = Module::new(&engine, &bytes)?;

    // Host functionality can be arbitrary Rust functions and is provided
    // to guests through a `Linker`.
    let linker = Linker::new(&engine);
    // linker.func_wrap(
    //     "libsig",
    //     "print_i32",
    //     |_caller: Caller<'_, u32>, param: i32| {
    //         println!("{}", param);
    //     },
    // )?;

    // All wasm objects operate within the context of a "store". Each
    // `Store` has a type parameter to store host-specific data, which in
    // this case we're using `4` for.
    let mut store: Store<u32> = Store::new(&engine, 4);

    // Instantiation of a module requires specifying its imports and then
    // afterwards we can fetch exports by name, as well as asserting the
    // type signature of the function with `get_typed_func`.
    let instance = linker.instantiate(&mut store, &module)?;
    let start = instance.get_typed_func::<(), i32>(&mut store, "main")?;

    // And finally we can call the wasm!
    let return_value = start.call(&mut store, ())?;

    println!("{}", return_value);

    Ok(())
}
