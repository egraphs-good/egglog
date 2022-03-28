use egg_smol::*;

macro_rules! e {
    (( $($inner:tt)* )) => { e!($($inner)*) };
    ($sym:ident) => { Expr::Var(stringify!($sym).into()) };
    ($f:ident $($arg:tt)+) => { Expr::new(stringify!($f), [$(e!($arg)),+]) };
    ($e:expr) => { Expr::leaf($e) };
}

#[test]
fn test_simple_rule() {
    let mut egraph = EGraph::default();
    let t_expr = egraph.declare_sort("Expr");
    egraph.declare_constructor("add", vec![t_expr.clone(), t_expr]);
    egraph.declare_constructor("int", vec![Type::Int]);

    egraph.add_named_rule(
        "add-assoc",
        Rule::rewrite(e![add (add x y) z], e![add x (add y z)]),
    );

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

#[test]
fn test_path() {
    let input = "
    (relation path (Int Int))
    (relation edge (Int Int))

    (rule ((true (edge x y)))
          ((assert (path x y))))

    (rule ((true (path x y) (edge y z)))
          ((assert (path x z))))
          
    (assert (edge 1 2) (edge 2 3) (edge 3 4))
    (check-eq true (edge 1 2))
    (run 3)
    (check-eq true (path 1 4))
    ";

    let mut egraph = EGraph::default();
    egraph.run_program(input);
}
