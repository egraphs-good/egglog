use egg_smol::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Math {
    Add,
    Int(i32),
}

impl Operator for Math {}

fn vars() -> [Pattern<Math>; 3] {
    [
        Pattern::leaf(OpOrVar::Var(Symbol::from("x"))),
        Pattern::leaf(OpOrVar::Var(Symbol::from("y"))),
        Pattern::leaf(OpOrVar::Var(Symbol::from("z"))),
    ]
}

macro_rules! p {
    ($op:expr $(, $($arg:expr),*)?) => { Pattern::<Math>::new(OpOrVar::Op($op.clone()), vec![$($($arg.clone()),*)?] ) };
}

macro_rules! e {
    ($op:expr $(, $($arg:expr),*)?) => { Expr::<Math>::new($op.clone(), vec![$($($arg.clone()),*)?] ) };
}

#[test]
fn test_simple_rule() {
    let [x, y, z] = vars();
    use Math::*;

    let add_assoc = Rule::rewrite(p!(Add, p!(Add, x, y), z), p!(Add, x, p!(Add, y, z)));

    let mut egraph = EGraph::<Math>::default();

    let (i0, i1, i2, i3) = (e!(Int(0)), e!(Int(1)), e!(Int(2)), e!(Int(3)));

    let start = e!(Add, e!(Add, e!(Add, i0, i1), i2), i3);
    let end = e!(Add, i0, e!(Add, i1, e!(Add, i2, i3)));

    for i in 0..4 {
        egraph.add_node(Int(i), &[]);
    }

    egraph.add_expr(&start);
    egraph.run_rules(2, &[add_assoc]);

    let id1 = egraph.lookup_expr(&start);
    let id2 = egraph.lookup_expr(&end);
    assert_eq!(id1, id2);
}
