use egg_smol::*;

// fn vars() -> [Pattern<Math>; 3] {
//     [
//         Pattern::leaf(OpOrVar::Var(Symbol::from("x"))),
//         Pattern::leaf(OpOrVar::Var(Symbol::from("y"))),
//         Pattern::leaf(OpOrVar::Var(Symbol::from("z"))),
//     ]
// }

// macro_rules! p {
//     ($op:expr $(, $($arg:expr),*)?) => { Pattern::<Math>::new(OpOrVar::Op($op.clone()), vec![$($($arg.clone()),*)?] ) };
// }

// macro_rules! e {
//     ($op:expr $(, $($arg:expr),*)?) => { Expr::<Math>::new($op.clone(), vec![$($($arg.clone()),*)?] ) };
// }

macro_rules! e {
    (( $($inner:tt)* )) => { e!($($inner)*) };
    ($sym:ident) => { Expr::Var(stringify!($sym).into()) };
    ($f:ident $($arg:tt)+) => { Expr::new(stringify!($f), [$(e!($arg)),+]) };
    ($e:expr) => { Expr::leaf($e) };
}

macro_rules! schema {
    (@ty Int) => {
        Type::Int
    };
    (@ty $t:ident) => {
        Type::Sort(stringify!($t).into())
    };
    ($($arg:ident),+ -> $out:ident) => { Schema {
        input: vec![ $(schema!(@ty $arg)),* ],
        output: schema!(@ty $out),
    }};
}

#[test]
fn test_simple_rule() {
    // let [x, y, z] = vars();
    // use Math::*;

    let add_assoc = Rule::rewrite(e![add (add x y) z], e![add x (add y z)]);

    let mut egraph = EGraph::default();
    egraph.declare_sort("Expr");
    egraph.declare_function("add", schema!(Expr, Expr -> Expr));
    egraph.declare_function("int", schema!(Int -> Expr));

    // let (i0, i1, i2, i3) = (e!(Int(0)), e!(Int(1)), e!(Int(2)), e!(Int(3)));

    // let start = e!(Add, e!(Add, e!(Add, i0, i1), i2), i3);
    // let end = e!(Add, i0, e!(Add, i1, e!(Add, i2, i3)));

    let start = e!(add (add (add (int 0) (int 1)) (int 2)) (int 3));
    let end = e!(  add (int 0) (add (int 1) (add (int 2) (int 3))));

    egraph.eval_closed_expr(&e!(int 0));
    egraph.eval_closed_expr(&e!(int 1));
    egraph.eval_closed_expr(&e!(int 2));
    egraph.eval_closed_expr(&e!(int 3));

    egraph.eval_closed_expr(&start);
    egraph.run_rules(2, &[add_assoc]);

    let id1 = egraph.eval_closed_expr(&start);
    let id2 = egraph.eval_closed_expr(&end);
    assert_eq!(id1, id2);
}
