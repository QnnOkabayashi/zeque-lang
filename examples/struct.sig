fn main() i32 {
    runtime_main()
}

fn runtime_main() i32 {
    let x = vec_new(1, 2);
    let y = vec_new(3, 4);
    let z = vec_add(x, y);
    z.a + z.b
}

fn Vec() type {
    struct {
        a: i32,
        b: i32,
    }
}

fn vec_new(a: i32, b: i32) Vec() {
    Vec() { a: a, b: b }
}

fn vec_add(x: Vec(), y: Vec()) Vec() {
    let a = x.a + y.a;
    let b = x.b + y.b;
    vec_new(a, b)
}
