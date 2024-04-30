# Sig

## About

Sig is a toy programming language that compiles to WebAssembly.
In its current state, it's a hobby project that attempts to steal Zig's ideas.

## Demo

### Basic control flow

Computing the 8th value of the Fibonacci sequence at runtime and compile time, and check that they're the same.
```
fn main() i32 {
    let n = 8;
    let a = fib(n);
    let b = comptime fib(n);
    if (a == b) 1 else 0
}

fn fib(n: i32) i32 {
    if (n == 0) 0 else
    if (n == 1) 1 else
    fib(n - 1) + fib(n - 2)
}
```

### Types are values

Like Zig, types are values. Unfortunately, the only types for now are `i32`, `bool`, and `type`.
```
fn main() Foo(i32) {
    let T = i32;
    let F = Foo;

    let x: F(T) = 1 + 1 * 2;

    let U = bar(type, i32);
    let v = comptime {
        let x2 = x;
        bar(U, x2)
    };

    add(add(v, 1), 5)
}

fn add(a: i32, b: i32) i32 {
    a + b
}

fn bar(T: type, value: T) T {
    value
}

fn Foo(T: type) type {
    T
}
```

Because types are values, there aren't really plans for type inference.

In terms of what gets computed at comptime:
* The `main` function is computed at runtime.
* Every expression and function call inside of a `comptime` section.
* Arithmetic operations.
* If expressions are inlined if the condition is comptime known.
* Arguments to function calls where the param is a comptime-only type (for now, just `type`).


## Getting Started

You can compile programs to wasm using `compiler/sig_ast/src/bin/compile.rs`.
For example, you can compile `examples/if.sig` into `if.wasm` by doing:
```
cargo run --bin compile -- examples/if.sig -o if.wasm
```


You can then run the file with the following command:
```
cargo run --bin runtime -- if.wasm
```

If you have `wasm-tools` installed, you can view the generated wasm output in textual format with:
```
wasm-tools print hello.wasm
```

## Known bugs

Don't use variables that start with "comptime", e.g. `comptime_answer`.
It confuses the lexer/parser and I'm not versed with `peg` enough to fix it.
My current idea is to rewrite the parser using a different library, but that's a lot of work to fix a bug I really don't care about.

## Things that need to be refactored

* Error messages
* Crate and module structure
