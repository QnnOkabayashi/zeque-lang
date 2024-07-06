// Enter level 0
// insert (0, "decl")
let decl = "SourceFile";

fn Bar(
    b: struct { // Push level 2
        // insert (2, "decl")
        let decl = "FunctionParam";
    },  // Pop level 2

    // Push level 1
    // insert (1, "b")
    c: { // Push level 2
        let _ = b;
        i32
    }  // Pop level 2
    // Push level 2
    // insert (2, "c")
) {
    // still in lvl two from above, so the top-level decls, as well as `b` and `c` can be accessed.
    let _ = struct {
        // Push level 3
        // insert (3, "decl")
        let decl = "FunctionBody";
        // Pop level 3
    }
}
// Pop level 2
// Pop level 1

fn F(T: type) type {
    struct {
        let Type = T;

        inner: T,

        fn g(A: type) {
            let Inner = Type;
        }
    }
}

F(i32)

struct {
    let Type = i32;

    inner: i32,

    fn g(A: type) type {
        let Inner = i32;
        Inner
    }
}

F(i32).g(u32)

i32
