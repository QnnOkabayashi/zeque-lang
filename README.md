# Zeque

## Quick start

To start running programs, use the `zeque` binary. For example:
```
cargo run -- examples/struct.zq
```

Use the `-h` flag to see what other flags are available:
```
cargo run --bin compile -- -h
```

## About

Zeque is a toy programming language that compiles to WebAssembly.
In its current state, it's a hobby project that attempts to steal Zig's ideas.
Eventually it will support both runtime and comptime computation.
For now though, it's just a comptime evaluation (i.e., an interpreter).

## Features

* Types are (comptime) values
* Anonymous structs
* Comptime closures
* Function call caching: calling the same function with the same args just reads from a cache.

For examples, see `examples/`.

## Getting Started

> Disclaimer: this section is broken for now. See the Quick start section above.

You can compile programs to wasm using `compiler/sig_ast/src/bin/compile.rs`.
For example, you can compile `examples/if.zq` into `if.wasm` by doing:
```
cargo run --bin compile -- examples/if.zq -o if.wasm
```

You can then run the file with the following command:
```
cargo run --bin runtime -- if.wasm
```

If you're not changing the runtime, you should probably build the runtime in release mode and use that:
```
cargo build --bin runtime --release
```

And use it:
```
./target/release/runtime if.wasm
```

If you're additionally not changing the compiler much (e.g. writing experimental programs), you should also build the compiler on release:
```
cargo build --bin compile --release
```

And use it:
```
./target/release/compile examples/if.zq -o if.wasm
```

### All together

Compile the compiler and runtime:
```
cargo build --bin compile --release &&
cargo build --bin runtime --release
```

Run a program:
```
./target/release/compile examples/struct.zq -o out.wasm &&
./target/release/runtime out.wasm
```

### Viewing output wasm

If you have `wasm-tools` installed, you can view the generated wasm output in textual format with:
```
wasm-tools print if.wasm
```
