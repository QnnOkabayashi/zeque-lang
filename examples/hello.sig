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
    let x = 1 + 1;

    T
}
