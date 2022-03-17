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

// macro_rules! schema {
//     (@ty Int) => {
//         Type::Int
//     };
//     (@ty $t:ident) => {
//         Type::Sort(stringify!($t).into())
//     };
//     ($($arg:ident),+ -> $out:ident) => { Schema {
//         input: vec![ $(schema!(@ty $arg)),* ],
//         output: schema!(@ty $out),
//     }};
// }

#[test]
fn test_simple_rule() {
    // let [x, y, z] = vars();
    // use Math::*;

    let mut egraph = EGraph::default();
    let t_expr = egraph.declare_sort("Expr");
    egraph.declare_constructor("add", vec![t_expr.clone(), t_expr]);
    egraph.declare_constructor("int", vec![Type::Int]);

    egraph.add_named_rule(
        "add-assoc",
        Rule::rewrite(e![add (add x y) z], e![add x (add y z)]),
    );

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
    egraph.run_rules(2);

    let id1 = egraph.eval_closed_expr(&start);
    let id2 = egraph.eval_closed_expr(&end);
    assert_eq!(id1, id2);
}

#[test]
fn test_simple_parsed() {
    let input = "
    (datatype Expr
      (add Expr Expr)
      (int Int))
    
    (define start (add (add (add (int 0) (int 1)) (int 2)) (int 3)))
    (define end   (add (int 0) (add (int 1) (add (int 2) (int 3)))))

    (rewrite (add (add x y) z) (add x (add y z)))

    (run 2)
    (check-eq start end)
    ";

    let mut egraph = EGraph::default();
    egraph.run_program(input);
}
