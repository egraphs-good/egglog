use std::{
    fmt::Debug,
    hash::Hash,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use core_relations::{make_external_func, Container, ExternalFunctionId, Rebuilder, Value};
use log::debug;
use num_rational::Rational64;
use numeric_id::NumericId;

use crate::{
    add_expressions, define_rule, ColumnTy, DefaultVal, EGraph, FunctionConfig, FunctionId,
    MergeFn, QueryEntry,
};

/// Run a simple associativity/commutativity test. In addition to testing that the rules properly
/// reassociate a nested sum, this test checks a proof of an arbitrary term in the database if
/// `tracing` is true.
///
/// The `can_subsume` argument is only used to enable subsumption on the underlying tables created
/// during this test, and exercise the different column handling caused by enabling subsumption.
/// Subsumption itself is not used.
fn ac_test(tracing: bool, can_subsume: bool) {
    const N: usize = 5;
    let mut egraph = if tracing {
        EGraph::with_tracing()
    } else {
        EGraph::default()
    };
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume,
    });
    let add_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 3],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume,
    });

    let add_comm = define_rule! {
        [egraph] ((-> (add_table x y) id))
              => ((set (add_table y x) id))
    };

    let add_assoc = define_rule! {
        [egraph] ((-> (add_table x (add_table y z)) id))
              => ((set (add_table (add_table x y) z) id))
    };

    // Running these rules on an empty database should change nothing.
    assert!(!egraph.run_rules(&[add_comm, add_assoc]).unwrap());

    // Fill the database.
    let mut ids = Vec::new();
    //  Add 0 .. N to the database.
    for i in 0..N {
        let i = egraph.primitives_mut().get(i as i64);
        ids.push(egraph.add_term(num_table, &[i], "base number"));
    }

    // construct (0 + ... + N), left-associated, and (N + ... + 0),
    // right-associated. With the assoc and comm rules saturated, these two
    // should be equal.
    let (left_root, right_root) = {
        let mut prev = ids[0];
        for num in &ids[1..] {
            let id = egraph.add_term(add_table, &[*num, prev], "add_left");
            prev = id;
        }
        let left_root = prev;
        let mut prev = *ids.last().unwrap();
        for num in ids[0..(N - 1)].iter() {
            let id = egraph.add_term(add_table, &[prev, *num], "add_right");
            prev = id;
        }
        let right_root = prev;
        (left_root, right_root)
    };
    // Saturate
    while egraph.run_rules(&[add_comm, add_assoc]).unwrap() {}
    let canon_left = egraph.get_canon(left_root);
    let canon_right = egraph.get_canon(right_root);
    assert_eq!(canon_left, canon_right, "failed to reassociate!");
    if tracing {
        let mut row = Vec::new();
        egraph.dump_table(add_table, |vals| {
            row.clear();
            row.extend_from_slice(vals);
        });

        let term_id = egraph.lookup_id(add_table, &row[0..row.len() - 1]).unwrap();
        let term_explanation = egraph.explain_term(term_id).unwrap();
        egraph.check_term_proof(term_explanation).unwrap();
        let eq_explanation = egraph.explain_terms_equal(left_root, right_root).unwrap();
        egraph.check_eq_proof(&eq_explanation).unwrap();
    }
}

#[test]
fn ac_tracing_subsume() {
    ac_test(true, true);
}

#[test]
fn ac_tracing() {
    ac_test(true, false);
}

#[test]
fn ac() {
    ac_test(false, false);
}

#[test]
fn ac_subsume() {
    ac_test(false, true);
}

#[test]
fn ac_fail() {
    const N: usize = 5;
    let mut egraph = EGraph::default();
    egraph.primitives_mut().register_type::<i64>();
    let int_prim = egraph.primitives_mut().get_ty::<i64>();
    let one = egraph.primitive_constant(1i64);
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
    });
    let add_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 3],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume: false,
    });

    let add_comm = define_rule! {
        [egraph] ((-> (add_table x y) id) (-> (num_table {one}) x))
              => ((set (add_table y x) id))
    };

    let add_assoc = define_rule! {
        [egraph] ((-> (add_table x (add_table y z)) id))
              => ((set (add_table (add_table x y) z) id))
    };

    // Running these rules on an empty database should change nothing.
    assert!(!egraph.run_rules(&[add_comm, add_assoc]).unwrap());

    // Fill the database.
    let mut ids = Vec::new();
    //  Add 0 .. N to the database.
    let num_rows = (0..N)
        .map(|i| {
            let id = egraph.fresh_id();
            let i = egraph.primitives_mut().get(i as i64);
            ids.push(id);
            (num_table, vec![i, id])
        })
        .collect::<Vec<_>>();
    egraph.add_values(num_rows);

    // construct (0 + ... + N), left-associated, and (N + ... + 0),
    // right-associated. With the assoc and comm rules saturated, these two
    // should be equal.
    let (left_root, right_root) = {
        let mut to_add = Vec::new();
        let mut prev = ids[0];
        for num in &ids[1..] {
            let id = egraph.fresh_id();
            to_add.push((add_table, vec![*num, prev, id]));
            prev = id;
        }
        let left_root = to_add.last().unwrap().1[2];
        prev = *ids.last().unwrap();
        for num in ids[0..(N - 1)].iter() {
            let id = egraph.fresh_id();
            to_add.push((add_table, vec![prev, *num, id]));
            prev = id;
        }
        let right_root = to_add.last().unwrap().1[2];
        egraph.add_values(to_add);
        (left_root, right_root)
    };
    // Saturate
    while egraph.run_rules(&[add_comm, add_assoc]).unwrap() {}
    let canon_left = egraph.get_canon(left_root);
    let canon_right = egraph.get_canon(right_root);
    assert_ne!(canon_left, canon_right);
}

#[test]
fn math() {
    let handles =
        Vec::from_iter((0..2).map(|_| thread::spawn(|| math_test(EGraph::default(), false))));
    handles.into_iter().for_each(|h| h.join().unwrap());
}

#[test]
fn math_subsume() {
    let handles =
        Vec::from_iter((0..2).map(|_| thread::spawn(|| math_test(EGraph::default(), true))));
    handles.into_iter().for_each(|h| h.join().unwrap());
}

#[test]
fn math_tracing() {
    math_test(EGraph::with_tracing(), false)
}
#[test]
fn math_tracing_subsume() {
    math_test(EGraph::with_tracing(), true)
}

/// Run a more complex benchmark from the egg and egglog test suite. The core of this test is to
/// ensure that the test generates a set of tables of exactly the same
/// size that the corresponding rules in egglog do in egglog's initial implementation.
///
/// As in `ac_test` the `can_subsume` argument is only used to enable subsumption on the underlying
/// tables created during this test, and exercise the different column handling caused by enabling
/// subsumption. Subsumption itself is not used.
fn math_test(mut egraph: EGraph, can_subsume: bool) {
    const N: usize = 8;
    let rational_ty = egraph.primitives_mut().register_type::<Rational64>();
    let string_ty = egraph.primitives_mut().register_type::<&'static str>();
    // tables
    let diff = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "diff".into(),
        can_subsume,
    });
    let integral = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "integral".into(),
        can_subsume,
    });
    let add = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume,
    });
    let sub = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "sub".into(),
        can_subsume,
    });
    let mul = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "mul".into(),
        can_subsume,
    });
    let div = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "div".into(),
        can_subsume,
    });
    let pow = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "pow".into(),
        can_subsume,
    });

    let ln = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "ln".into(),
        can_subsume,
    });
    let sqrt = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "sqrt".into(),
        can_subsume,
    });
    let sin = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "sin".into(),
        can_subsume,
    });
    let cos = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "cos".into(),
        can_subsume,
    });
    let rat = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(rational_ty), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rat".into(),
        can_subsume,
    });
    let var = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(string_ty), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "var".into(),
        can_subsume,
    });

    let zero = egraph.primitive_constant(Rational64::new(0, 1));
    let one = egraph.primitive_constant(Rational64::new(1, 1));
    let neg1 = egraph.primitive_constant(Rational64::new(-1, 1));
    let two = egraph.primitive_constant(Rational64::new(2, 1));
    let rules = [
        define_rule! {
            [egraph] ((-> (add x y) id)) => ((set (add y x) id))
        },
        define_rule! {
            [egraph] ((-> (mul x y) id)) => ((set (mul y x) id))
        },
        define_rule! {
            [egraph] ((-> (add x (add y z)) id)) => ((set (add (add x y) z) id))
        },
        define_rule! {
            [egraph] ((-> (mul x (mul y z)) id)) => ((set (mul (mul x y) z) id))
        },
        define_rule! {
            [egraph] ((-> (sub x y) id)) => ((set (add x (mul (rat {neg1.clone()}) y)) id))
        },
        define_rule! {
            [egraph] ((-> (add a (rat {zero.clone()})) id)) => ((union a id))
        },
        define_rule! {
            [egraph] ((-> (rat {zero.clone()}) z_id) (-> (mul a z_id) id))
                    => ((union id z_id))
        },
        define_rule! {
            [egraph] ((-> (mul a (rat {one.clone()})) id)) => ((union a id))
        },
        define_rule! {
            [egraph] ((-> (sub x x) id)) => ((union id (rat {zero})))
        },
        define_rule! {
            [egraph] ((-> (mul x (add b c)) id)) => ((set (add (mul x b) (mul x c)) id))
        },
        define_rule! {
            [egraph] ((-> (add (mul x a) (mul x b)) id)) => ((set (mul x (add a b)) id))
        },
        define_rule! {
            [egraph] ((-> (mul (pow a b) (pow a c)) id)) => ((set (pow a (add b c)) id))
        },
        define_rule! {
            [egraph] ((-> (pow x (rat {one.clone()})) id)) => ((union x id))
        },
        define_rule! {
            [egraph] ((-> (pow x (rat {two})) id)) => ((set (mul x x) id))
        },
        define_rule! {
            [egraph] ((-> (diff x (add a b)) id)) => ((set (add (diff x a) (diff x b)) id))
        },
        define_rule! {
            [egraph] ((-> (diff x (mul a b)) id)) => ((set (add (mul a (diff x b)) (mul b (diff x a))) id))
        },
        define_rule! {
            [egraph] ((-> (diff x (sin x)) id)) => ((set (cos x) id))
        },
        define_rule! {
            [egraph] ((-> (diff x (cos x)) id)) => ((set (mul (rat {neg1.clone()}) (sin x)) id))
        },
        define_rule! {
            [egraph] ((-> (integral (rat {one}) x) id)) => ((union id x))
        },
        define_rule! {
            [egraph] ((-> (integral (cos x) x) id)) => ((set (sin x) id))
        },
        define_rule! {
            [egraph] ((-> (integral (sin x) x) id)) => ((set (mul (rat {neg1}) (cos x)) id))
        },
        define_rule! {
            [egraph] ((-> (integral (add f g) x) id)) => ((set (add (integral f x) (integral g x)) id))
        },
        define_rule! {
            [egraph] ((-> (integral (sub f g) x) id)) => ((set (sub (integral f x) (integral g x)) id))
        },
        define_rule! {
            [egraph] ((-> (integral (mul a b) x) id))
            => ((set (sub (mul a (integral b x))
                          (integral (mul (diff x a) (integral b x)) x)) id))
        },
    ];

    {
        let one = egraph.primitives_mut().get(Rational64::new(1, 1));
        let two = egraph.primitives_mut().get(Rational64::new(2, 1));
        let three = egraph.primitives_mut().get(Rational64::new(3, 1));
        let seven = egraph.primitives_mut().get(Rational64::new(7, 1));
        let x_str = egraph.primitives_mut().get::<&'static str>("x");
        let y_str = egraph.primitives_mut().get::<&'static str>("y");
        let five_str = egraph.primitives_mut().get::<&'static str>("five");
        add_expressions! {
            [egraph]

            (integral (ln (var x_str)) (var x_str))
            (integral (add (var x_str) (cos (var x_str))) (var x_str))
            (integral (mul (cos (var x_str)) (var x_str)) (var x_str))
            (diff (var x_str)
                (add (rat one) (mul (rat two) (var x_str))))
            (diff (var x_str)
                (sub (pow (var x_str) (rat three)) (mul (rat seven) (pow (var x_str) (rat two)))))
            (add
                (mul (var y_str) (add (var x_str) (var y_str)))
                (sub (add (var x_str) (rat two)) (add (var x_str) (var x_str))))
            (div (rat one)
                 (sub (div (add (rat one) (sqrt (var five_str))) (rat two))
                      (div (sub (rat one) (sqrt (var five_str))) (rat two))))
        }
    }

    for _ in 0..N {
        if !egraph.run_rules(&rules).unwrap() {
            break;
        }
    }

    // numbers validated against the egglog implementation.

    // Print out some debugging info. This gets hidden by default for passing tests.
    debug!("diff_size={:?} vs. 338", egraph.table_size(diff));
    debug!("integral_size={:?} vs. 782 ", egraph.table_size(integral));
    debug!("sub_size={:?} vs 438", egraph.table_size(sub));
    debug!("div_size={:?} vs. 3", egraph.table_size(div));
    debug!("pow_size={:?} vs 2", egraph.table_size(pow));
    debug!("ln_size={:?} vs 1", egraph.table_size(ln));
    debug!("sqrt_size={:?} vs 1", egraph.table_size(sqrt));
    debug!("sin_size={:?} vs 1", egraph.table_size(sin));
    debug!("cos_size={:?} vs 1", egraph.table_size(cos));
    debug!("rat_size={:?} vs 5", egraph.table_size(rat));
    debug!("var_size={:?} vs 3", egraph.table_size(var));
    debug!("add_size={:?} vs 2977", egraph.table_size(add));
    debug!("mul_size={:?} vs 3516", egraph.table_size(mul));

    if !egraph.tracing {
        // NB: we still don't understand why these counts don't match when
        // proofs are enabled. We need better debugging to make this viable
        // though.
        assert_eq!(338, egraph.table_size(diff));
        assert_eq!(782, egraph.table_size(integral));
        assert_eq!(483, egraph.table_size(sub));
        assert_eq!(3, egraph.table_size(div));
        assert_eq!(2, egraph.table_size(pow));
        assert_eq!(1, egraph.table_size(ln));
        assert_eq!(1, egraph.table_size(sqrt));
        assert_eq!(1, egraph.table_size(sin));
        assert_eq!(1, egraph.table_size(cos));
        assert_eq!(5, egraph.table_size(rat));
        assert_eq!(3, egraph.table_size(var));
        assert_eq!(2977, egraph.table_size(add));
        assert_eq!(3516, egraph.table_size(mul));
    }

    if egraph.tracing {
        let mut row = Vec::new();
        egraph.dump_table(mul, |vals| {
            row.clear();
            row.extend_from_slice(vals);
        });
        let term_id = egraph.lookup_id(mul, &row[0..row.len() - 1]).unwrap();
        let explain = egraph.explain_term(term_id).unwrap();
        egraph.check_term_proof(explain).unwrap();
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct VecContainer(Vec<Value>);
impl Container for VecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        rebuilder.rebuild_slice(&mut self.0)
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.0.iter().copied()
    }
}

fn register_vec_push(egraph: &mut EGraph) -> ExternalFunctionId {
    egraph.register_container_ty::<VecContainer>();
    let external_func = make_external_func(move |state, vals| -> Option<Value> {
        let [vec_id, val] = vals else {
            panic!("[vec-push] expected 2 values, got {vals:?}")
        };
        let mut vec: VecContainer = state.containers().get_val::<VecContainer>(*vec_id)?.clone();
        vec.0.push(*val);
        // Vectors are immutable. May as well not use O(n) auxiliary space.
        vec.0.shrink_to_fit();
        Some(state.clone().containers().register_val(vec, state))
    });
    egraph.register_external_func(external_func)
}

fn register_vec_last(egraph: &mut EGraph) -> ExternalFunctionId {
    egraph.register_container_ty::<VecContainer>();
    let external_func = make_external_func(move |state, vals| -> Option<Value> {
        let [vec_id] = vals else {
            panic!("[vec-last] expected 1 value, got {vals:?}")
        };
        state
            .containers()
            .get_val::<VecContainer>(*vec_id)?
            .0
            .last()
            .cloned()
    });
    egraph.register_external_func(external_func)
}

fn dump_vecs(egraph: &EGraph) -> Vec<Vec<Value>> {
    let mut res = Vec::new();
    egraph
        .containers()
        .for_each::<VecContainer>(|vec, _| res.push(vec.0.clone()));
    res
}

fn assert_unordered_eq<T: Ord + std::fmt::Debug>(mut a: Vec<T>, mut b: Vec<T>) {
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

fn container_test() {
    // Test for containers:
    // * Basic math setup: (num i64), (add math math), (Vec (vec math))
    // * start with:
    //   - (Vec vec![1])
    //   - (Vec vec![])
    // * have a rule that does, for any vec, push (add 0 last-elt) onto it.
    // * have a rule that does, for any vec, push (add last-elt 0) onto it.
    // * Run this 3 times.
    // * Check that we get some decent number of vectors out.
    // * Saturate the rule that just evaluates add.
    // * should have just have:
    //  - vec![]
    //  - vec![1]
    //  - vec![1, 1]
    //  - vec![1, 1, 1]
    //  - vec![1, 1, 1, 1]
    //
    //  This tests:
    //  * basic get/set for containers.
    //  * running container operations from a rule, including ones that can fail.
    //  * Rebuilding:
    //      * rebuilding of container ids.
    //      * rebuilding inside of a container.
    //      * saturation for container rebuilding.
    //  * Dumping/foreach functionality.
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
    });
    let add_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 3],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume: false,
    });
    let vec_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 2],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "vec".into(),
        can_subsume: false,
    });
    let int_add = egraph.register_external_func(make_external_func(|exec_state, args| {
        let [x, y] = args else { panic!() };
        let x: i64 = exec_state.prims().unwrap(*x);
        let y: i64 = exec_state.prims().unwrap(*y);
        let z: i64 = x + y;
        Some(exec_state.prims().get(z))
    }));
    let vec_last = register_vec_last(&mut egraph);
    let vec_push = register_vec_push(&mut egraph);

    let mut ids = Vec::new();
    //  Add 0 and 1 to the database.
    let num_rows = (0..=1)
        .map(|i| {
            let id = egraph.fresh_id();
            let i = egraph.primitives_mut().get(i as i64);
            ids.push(id);
            (num_table, vec![i, id])
        })
        .collect::<Vec<_>>();
    egraph.add_values(num_rows);

    let empty_vec = egraph.get_container_val(VecContainer(vec![]));
    let vec1 = egraph.get_container_val(VecContainer(vec![ids[1]]));

    let empty_vec_id = egraph.fresh_id();
    let vec1_id = egraph.fresh_id();

    egraph.add_values(vec![
        (vec_table, vec![empty_vec, empty_vec_id]),
        (vec_table, vec![vec1, vec1_id]),
    ]);

    let vec_expand = {
        let mut rb = egraph.new_rule("", true);
        let vec = rb.new_var(ColumnTy::Id);
        let vec_id = rb.new_var(ColumnTy::Id);
        let last = rb.new_var(ColumnTy::Id);
        rb.query_table(vec_table, &[vec.into(), vec_id.into()], Some(false))
            .unwrap();
        rb.query_prim(vec_last, &[vec.into(), last.into()], ColumnTy::Id)
            .unwrap();
        let add_last_0 = rb.lookup(
            add_table,
            &[
                last.into(),
                QueryEntry::Const {
                    val: ids[0],
                    ty: ColumnTy::Primitive(int_prim),
                },
            ],
            || "add_last_0".to_string(),
        );
        let add_0_last = rb.lookup(
            add_table,
            &[
                QueryEntry::Const {
                    val: ids[0],
                    ty: ColumnTy::Primitive(int_prim),
                },
                last.into(),
            ],
            || "add_0_last".to_string(),
        );
        let new_vec_1 =
            rb.call_external_func(vec_push, &[vec.into(), add_last_0.into()], ColumnTy::Id, "");
        let new_vec_2 =
            rb.call_external_func(vec_push, &[vec.into(), add_0_last.into()], ColumnTy::Id, "");
        rb.lookup(vec_table, &[new_vec_1.into()], String::new);
        rb.lookup(vec_table, &[new_vec_2.into()], String::new);
        rb.build()
    };

    let eval_add = {
        let mut rb = egraph.new_rule("", true);
        let lhs_raw = rb.new_var(ColumnTy::Primitive(int_prim));
        let lhs_id = rb.new_var(ColumnTy::Id);
        let rhs_raw = rb.new_var(ColumnTy::Primitive(int_prim));
        let rhs_id = rb.new_var(ColumnTy::Id);
        let add_id = rb.new_var(ColumnTy::Id);
        rb.query_table(num_table, &[lhs_raw.into(), lhs_id.into()], Some(false))
            .unwrap();
        rb.query_table(num_table, &[rhs_raw.into(), rhs_id.into()], Some(false))
            .unwrap();
        rb.query_table(
            add_table,
            &[lhs_id.into(), rhs_id.into(), add_id.into()],
            Some(false),
        )
        .unwrap();
        let evaled = rb.call_external_func(
            int_add,
            &[lhs_raw.into(), rhs_raw.into()],
            ColumnTy::Primitive(int_prim),
            "",
        );
        let boxed = rb.lookup(num_table, &[evaled.into()], String::new);
        rb.union(add_id.into(), boxed.into());
        rb.build()
    };

    assert_unordered_eq(
        dump_vecs(&egraph),
        vec![vec![], vec![egraph.get_canon(ids[1])]],
    );

    assert!(egraph.run_rules(&[vec_expand]).unwrap());
    assert_eq!(dump_vecs(&egraph).len(), 4);
    // We have 2 new vectors with a last element. Each of those should spawn two more, adding 4.
    assert!(egraph.run_rules(&[vec_expand]).unwrap());
    assert_eq!(dump_vecs(&egraph).len(), 8);
    // We have 4 new vectors with a last element. Each of those should spawn two more, adding 8.
    assert!(egraph.run_rules(&[vec_expand]).unwrap());
    assert_eq!(dump_vecs(&egraph).len(), 16);

    // Now we want to saturate `eval_add`. This should collapse a bunch of new vectors.

    let mut saturated = false;
    for _ in 0..20 {
        saturated = !egraph.run_rules(&[eval_add]).unwrap();
        if saturated {
            break;
        }
    }
    assert!(saturated, "failed to saturate after 20 iterations");

    let one_id = egraph.get_canon(ids[1]);
    assert_unordered_eq(
        dump_vecs(&egraph),
        vec![
            vec![],
            vec![one_id],
            vec![one_id; 2],
            vec![one_id; 3],
            vec![one_id; 4],
        ],
    );
}

#[test]
fn basic_container() {
    // Run the test 8 times to get a decent sample of incremental/nonincremental, parallel/serial.
    for _ in 0..8 {
        container_test()
    }
}

#[test]
fn rhs_only_rule() {
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let zero = egraph.primitives_mut().get(0i64);
    let one = egraph.primitives_mut().get(1i64);
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
    });
    let add_data = {
        let zero = egraph.primitive_constant(0i64);
        let one = egraph.primitive_constant(1i64);
        let mut rb = egraph.new_rule("", true);
        let _zero_id = rb.lookup(num_table, &[zero], String::new);
        let _one_id = rb.lookup(num_table, &[one], String::new);
        rb.build()
    };

    let mut contents = Vec::new();

    assert!(contents.is_empty());
    assert!(egraph.run_rules(&[add_data]).unwrap());
    egraph.dump_table(num_table, |vals| {
        contents.push(vals.to_vec());
    });

    contents.sort();
    assert_eq!(
        contents,
        vec![vec![zero, Value::new(0)], vec![one, Value::new(1)]]
    );
}

#[test]
fn rhs_only_rule_only_runs_once() {
    let mut egraph = EGraph::default();
    let counter = Arc::new(AtomicUsize::new(0));
    let inner = counter.clone();
    let inc_counter_func = egraph.register_external_func(make_external_func(move |_, _| {
        inner.fetch_add(1, Ordering::SeqCst);
        Some(Value::new(0))
    }));
    let inc_counter_rule = {
        let mut rb = egraph.new_rule("", true);
        rb.call_external_func(inc_counter_func, &[], ColumnTy::Id, "");
        rb.build()
    };

    assert!(!egraph.run_rules(&[inc_counter_rule]).unwrap());
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert!(!egraph.run_rules(&[inc_counter_rule]).unwrap());
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn mergefn_arithmetic() {
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();

    // Create external functions for multiplication and addition
    let multiply_func = egraph.register_external_func(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a, b] = vals else {
                return None;
            };
            let a_val = *state.prims().unwrap_ref::<i64>(*a);
            let b_val = *state.prims().unwrap_ref::<i64>(*b);
            let res = state.prims().get::<i64>(a_val * b_val);
            Some(res)
        },
    ));

    let add_func = egraph.register_external_func(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a, b] = vals else {
                return None;
            };
            let a_val = *state.prims().unwrap_ref::<i64>(*a);
            let b_val = *state.prims().unwrap_ref::<i64>(*b);
            let res = state.prims().get::<i64>(a_val + b_val);
            Some(res)
        },
    ));

    let value_1 = egraph.primitives_mut().get(1i64);

    // Create a function with merge function (+ 1 (* old new))
    // This uses nested MergeFn::Primitive with external functions to build the complex merge function
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Primitive(int_prim)],
        default: DefaultVal::Fail,
        merge: MergeFn::Primitive(
            add_func,
            vec![
                MergeFn::Const(value_1),
                MergeFn::Primitive(multiply_func, vec![MergeFn::Old, MergeFn::New]),
            ],
        ),
        name: "f".into(),
        can_subsume: false,
    });

    let value_0 = egraph.primitive_constant(0i64);
    let value_1 = egraph.primitive_constant(1i64);
    let value_2 = egraph.primitive_constant(2i64);
    let value_3 = egraph.primitive_constant(3i64);
    let value_4 = egraph.primitive_constant(4i64);
    let value_5 = egraph.primitive_constant(5i64);
    let value_6 = egraph.primitive_constant(6i64);

    // First rule writes (f 1 0) (f 2 1)
    let rule1 = {
        let mut rb = egraph.new_rule("rule1", true);
        rb.set(f_table, &[value_1.clone(), value_0]);
        rb.set(f_table, &[value_2.clone(), value_1.clone()]);
        rb.build()
    };

    // Run the first rule and check state
    assert!(egraph.run_rules(&[rule1]).unwrap());
    let mut contents = Vec::new();
    egraph.dump_table(f_table, |vals| {
        contents.push((
            *egraph.primitives().unwrap_ref::<i64>(vals[0]),
            *egraph.primitives().unwrap_ref::<i64>(vals[1]),
        ));
    });
    contents.sort();
    assert_eq!(contents, vec![(1, 0), (2, 1)]);

    // Second rule writes (f 1 5) (f 2 6)
    let rule2 = {
        let mut rb = egraph.new_rule("rule2", true);
        rb.set(f_table, &[value_1.clone(), value_5]);
        rb.set(f_table, &[value_2.clone(), value_6]);
        rb.build()
    };

    // Run the second rule and check state
    // Expected: (f 1 1) because 1 + (0 * 5) = 1
    // Expected: (f 2 7) because 1 + (1 * 6) = 7
    assert!(egraph.run_rules(&[rule2]).unwrap());
    contents.clear();
    egraph.dump_table(f_table, |vals| {
        contents.push((
            *egraph.primitives().unwrap_ref::<i64>(vals[0]),
            *egraph.primitives().unwrap_ref::<i64>(vals[1]),
        ));
    });
    contents.sort();
    assert_eq!(contents, vec![(1, 1), (2, 7)]);

    // Third rule writes (f 1 3) (f 2 4)
    let rule3 = {
        let mut rb = egraph.new_rule("rule3", true);
        rb.set(f_table, &[value_1, value_3]);
        rb.set(f_table, &[value_2, value_4]);
        rb.build()
    };

    // Run the third rule and check state
    // Expected: (f 1 4) because 1 + (1 * 3) = 4
    // Expected: (f 2 29) because 1 + (7 * 4) = 29
    assert!(egraph.run_rules(&[rule3]).unwrap());
    contents.clear();
    egraph.dump_table(f_table, |vals| {
        contents.push((
            *egraph.primitives().unwrap_ref::<i64>(vals[0]),
            *egraph.primitives().unwrap_ref::<i64>(vals[1]),
        ));
    });
    contents.sort();
    assert_eq!(contents, vec![(1, 4), (2, 29)]);
}

#[test]
fn mergefn_nested_function() {
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();

    // Create a function g that will be used in the merge function for f
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: true,
    });

    // Create a function f whose merge function is (g (g new new) (g old old))
    // This uses nested MergeFn::Function to build the complex merge function
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::Function(
            g_table,
            vec![
                MergeFn::Function(g_table, vec![MergeFn::New, MergeFn::New]),
                MergeFn::Function(g_table, vec![MergeFn::Old, MergeFn::Old]),
            ],
        ),
        name: "f".into(),
        can_subsume: true,
    });

    let value_1 = egraph.primitive_constant(1i64);
    let value_2 = egraph.primitive_constant(2i64);

    // Create an rhs-only rule that writes f values with fresh IDs
    // We'll run this rule multiple times and observe how the merge function works

    let write_rule = {
        let mut rb = egraph.new_rule("write_rule", true);
        rb.lookup(f_table, &[value_1.clone()], String::new);
        rb.lookup(f_table, &[value_2], String::new);
        rb.build()
    };

    // Helper function to get all g-table entries
    let get_g_entries = |egraph: &EGraph| {
        let mut entries = Vec::new();
        egraph.dump_table(g_table, |vals| {
            entries.push((vals[0], vals[1], vals[2]));
        });
        entries.sort();
        entries
    };

    // Helper function to get all f-table entries
    let get_f_entries = |egraph: &EGraph| {
        let mut entries = Vec::new();
        egraph.dump_table(f_table, |vals| {
            entries.push((*egraph.primitives().unwrap_ref::<i64>(vals[0]), vals[1]));
        });
        entries.sort();
        entries
    };

    // First run of the rule
    assert!(egraph.run_rules(&[write_rule]).unwrap());
    let f_entries_1 = get_f_entries(&egraph);
    let g_entries_1 = get_g_entries(&egraph);
    assert_eq!(f_entries_1.len(), 2);
    let base_1 = f_entries_1[0].1;
    let base_2 = f_entries_1[1].1;
    // After first run, there should be no g entries yet because no merging occurred
    assert_eq!(g_entries_1.len(), 0);

    let set_rule = {
        let mut rb = egraph.new_rule("iterate", true);
        rb.set(
            f_table,
            &[
                value_1,
                QueryEntry::Const {
                    val: base_2,
                    ty: ColumnTy::Id,
                },
            ],
        );
        rb.build()
    };

    // Second run of the rule - should trigger merging with previous values
    assert!(egraph.run_rules(&[set_rule]).unwrap());
    let f_entries_2 = get_f_entries(&egraph);
    let g_entries_2 = get_g_entries(&egraph);
    assert_eq!(f_entries_2.len(), 2);
    // After second run, g table should have entries from the merge functions
    assert_eq!(g_entries_2.len(), 3);

    // Get the entry for (f 1)
    let new_base_1 = f_entries_2[0].1;
    // Find the first layer of g:
    let (mid_1, mid_2, _) = *g_entries_2
        .iter()
        .find(|(_, _, a)| *a == new_base_1)
        .unwrap();
    let (base_l1, base_l2, _) = *g_entries_2.iter().find(|(_, _, a)| *a == mid_1).unwrap();
    let (base_r1, base_r2, _) = *g_entries_2.iter().find(|(_, _, a)| *a == mid_2).unwrap();

    // The merge function for f is (g (g new new) (g old old))
    // new here should have been base_2, old should have been base_1
    //
    // That means basel1 == basel2 == base_2, and baser1 == baser2 == base_1
    assert_eq!(base_l1, base_l2);
    assert_eq!(base_l1, base_2);
    assert_eq!(base_r1, base_r2);
    assert_eq!(base_r1, base_1);
}

#[test]
fn constrain_prims_simple() {
    // Take two functions, f and g. Fill f with (f 1) (f 2) (f 3), then filter for even numbers
    // when adding to 'g'. This should only add 2 to g.
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let bool_prim = egraph.primitives_mut().register_type::<bool>();
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "f".into(),
        can_subsume: false,
    });
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: false,
    });

    let is_even = egraph.register_external_func(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = *state.prims().unwrap_ref::<i64>(*a);
            let result: bool = a_val % 2 == 0;
            Some(state.prims().get(result))
        },
    ));

    let value_1 = egraph.primitive_constant(1i64);
    let value_2 = egraph.primitive_constant(2i64);
    let value_3 = egraph.primitive_constant(3i64);
    let value_true = egraph.primitive_constant(true);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.lookup(f_table, &[value_1.clone()], String::new);
        rb.lookup(f_table, &[value_2.clone()], String::new);
        rb.lookup(f_table, &[value_3.clone()], String::new);
        rb.build()
    };

    let copy_to_g = {
        let mut rb = egraph.new_rule("copy_to_g", true);
        let val = rb.new_var(ColumnTy::Primitive(int_prim));
        let id = rb.new_var(ColumnTy::Id);
        rb.query_table(f_table, &[val.into(), id.into()], Some(false))
            .unwrap();
        rb.query_prim(
            is_even,
            &[val.into(), value_true.clone()],
            ColumnTy::Primitive(bool_prim),
        )
        .unwrap();
        rb.set(g_table, &[val.into(), id.into()]);
        rb.build()
    };
    let get_entries = |egraph: &EGraph, table: FunctionId| {
        let mut entries = Vec::new();
        egraph.dump_table(table, |vals| {
            entries.push((*egraph.primitives().unwrap_ref::<i64>(vals[0]), vals[1]));
        });
        entries.sort();
        entries
    };

    assert!(get_entries(&egraph, f_table).is_empty());
    assert!(get_entries(&egraph, g_table).is_empty());
    egraph.run_rules(&[write_f]).unwrap();
    let f = get_entries(&egraph, f_table);
    assert_eq!(f.len(), 3);
    egraph.run_rules(&[copy_to_g]).unwrap();
    let g = get_entries(&egraph, g_table);
    assert_eq!(g.len(), 1);
    assert_eq!(g[0], f[1])
}

#[test]
fn constrain_prims_abstract() {
    // Take two functions, f and g. Fill f with (f -1) (f 0) (f 1), then filter for numbers where
    // (neg x) = (abs x) when adding to 'g'. This adds only -1 and 0 to g
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "f".into(),
        can_subsume: false,
    });
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: false,
    });

    let neg = egraph.register_external_func(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = *state.prims().unwrap_ref::<i64>(*a);
            Some(state.prims().get(-a_val))
        },
    ));
    let abs = egraph.register_external_func(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = *state.prims().unwrap_ref::<i64>(*a);
            Some(state.prims().get(a_val.abs()))
        },
    ));

    let value_n1 = egraph.primitive_constant(-1i64);
    let value_0 = egraph.primitive_constant(0i64);
    let value_1 = egraph.primitive_constant(1i64);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.lookup(f_table, &[value_n1.clone()], String::new);
        rb.lookup(f_table, &[value_0.clone()], String::new);
        rb.lookup(f_table, &[value_1.clone()], String::new);
        rb.build()
    };

    let copy_to_g = {
        let mut rb = egraph.new_rule("copy_to_g", true);
        let val = rb.new_var(ColumnTy::Primitive(int_prim));
        let id = rb.new_var(ColumnTy::Id);
        let negval = rb.new_var(ColumnTy::Primitive(int_prim));
        rb.query_table(f_table, &[val.into(), id.into()], Some(false))
            .unwrap();
        rb.query_prim(
            neg,
            &[val.into(), negval.into()],
            ColumnTy::Primitive(int_prim),
        )
        .unwrap();
        rb.query_prim(
            abs,
            &[val.into(), negval.into()],
            ColumnTy::Primitive(int_prim),
        )
        .unwrap();
        rb.set(g_table, &[val.into(), id.into()]);
        rb.build()
    };
    let get_entries = |egraph: &EGraph, table: FunctionId| {
        let mut entries = Vec::new();
        egraph.dump_table(table, |vals| {
            entries.push((*egraph.primitives().unwrap_ref::<i64>(vals[0]), vals[1]));
        });
        entries.sort();
        entries
    };

    assert!(get_entries(&egraph, f_table).is_empty());
    assert!(get_entries(&egraph, g_table).is_empty());
    egraph.run_rules(&[write_f]).unwrap();
    let f = get_entries(&egraph, f_table);
    assert_eq!(f.len(), 3);
    egraph.run_rules(&[copy_to_g]).unwrap();
    let g = get_entries(&egraph, g_table);
    assert_eq!(g.len(), 2);
    assert_eq!(g, f[0..2])
}

#[test]
fn basic_subsumption() {
    // fill (f 1) (f 2). Subsume (f 3) (f 2). Copy (f to g). Should only see (g 1)

    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "f".into(),
        can_subsume: true,
    });
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: false,
    });

    let value_1 = egraph.primitive_constant(1i64);
    let value_2 = egraph.primitive_constant(2i64);
    let value_3 = egraph.primitive_constant(3i64);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.lookup(f_table, &[value_1.clone()], String::new);
        rb.lookup(f_table, &[value_2.clone()], String::new);
        rb.build()
    };

    let subsume_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.subsume(f_table, &[value_2.clone()]);
        rb.subsume(f_table, &[value_3.clone()]);
        rb.build()
    };

    let copy_to_g = {
        let mut rb = egraph.new_rule("copy_to_g", true);
        let val = rb.new_var(ColumnTy::Primitive(int_prim));
        let id = rb.new_var(ColumnTy::Id);
        rb.query_table(f_table, &[val.into(), id.into()], Some(false))
            .unwrap();
        rb.set(g_table, &[val.into(), id.into()]);
        rb.build()
    };
    let get_entries = |egraph: &EGraph, table: FunctionId| {
        let mut entries = Vec::new();
        egraph.dump_table(table, |vals| {
            entries.push((*egraph.primitives().unwrap_ref::<i64>(vals[0]), vals[1]));
        });
        entries.sort();
        entries
    };

    assert!(get_entries(&egraph, f_table).is_empty());
    assert!(get_entries(&egraph, g_table).is_empty());
    egraph.run_rules(&[write_f]).unwrap();
    let f = get_entries(&egraph, f_table);
    assert_eq!(f.len(), 2);
    assert_eq!(f.iter().map(|(x, _)| *x).collect::<Vec<_>>(), vec![1, 2]);
    egraph.run_rules(&[subsume_f]).unwrap();
    let f = get_entries(&egraph, f_table);
    assert_eq!(f.len(), 3);
    assert_eq!(f.iter().map(|(x, _)| *x).collect::<Vec<_>>(), vec![1, 2, 3]);
    egraph.run_rules(&[copy_to_g]).unwrap();
    let g = get_entries(&egraph, g_table);
    assert_eq!(g.len(), 1);
    assert_eq!(g[0], f[0])
}

#[test]
fn lookup_failure_panics() {
    let mut egraph = EGraph::default();
    let f = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::Fail,
        merge: MergeFn::UnionId,
        name: "test".into(),
        can_subsume: false,
    });

    let to_entry = |val: u32| QueryEntry::Const {
        val: Value::new(val),
        ty: ColumnTy::Id,
    };

    let value_1 = to_entry(1);
    let value_2 = to_entry(2);
    let value_3 = to_entry(3);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.set(f, &[value_1.clone(), value_1.clone()]);
        rb.set(f, &[value_2.clone(), value_2.clone()]);
        rb.build()
    };
    egraph.run_rules(&[write_f]).unwrap();

    let lookup_success = {
        let mut rb = egraph.new_rule("lookup_success", true);
        rb.lookup(f, &[value_1.clone()], String::new);
        rb.build()
    };
    egraph.run_rules(&[lookup_success]).unwrap();

    let lookup_failure = {
        let mut rb = egraph.new_rule("lookup_fail", true);
        rb.lookup(f, &[value_3.clone()], String::new);
        rb.build()
    };
    egraph.run_rules(&[lookup_failure]).err().unwrap();
}

#[test]
fn primitive_failure_panics() {
    let mut egraph = EGraph::default();
    let _int_prim = egraph.primitives_mut().register_type::<i64>();
    let unit_prim = egraph.primitives_mut().register_type::<()>();

    let value_1 = egraph.primitive_constant(1i64);
    let value_2 = egraph.primitive_constant(2i64);

    let assert_odd = egraph.register_external_func(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = *state.prims().unwrap_ref::<i64>(*a);
            if a_val % 2 == 1 {
                Some(state.prims().get(()))
            } else {
                None
            }
        },
    ));

    let assert_odd_rule = {
        let mut rb = egraph.new_rule("assert_odd", true);
        rb.call_external_func(
            assert_odd,
            &[value_1.clone()],
            ColumnTy::Primitive(unit_prim),
            "",
        );
        rb.call_external_func(
            assert_odd,
            &[value_2.clone()],
            ColumnTy::Primitive(unit_prim),
            "",
        );
        rb.build()
    };

    egraph.run_rules(&[assert_odd_rule]).err().unwrap();
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<EGraph>()
};
