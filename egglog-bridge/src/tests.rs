use std::thread;

use core_relations::{make_external_func, Container, ExternalFunctionId, Rebuilder, Value};
use log::debug;
use num_rational::Rational64;

use crate::{add_expressions, define_rule, ColumnTy, DefaultVal, EGraph, Function, MergeFn};

#[test]
fn ac() {
    const N: usize = 5;
    let mut egraph = EGraph::default();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let num_table = egraph.add_table(
        vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "num",
    );
    let add_table = egraph.add_table(
        vec![ColumnTy::Id; 3],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "add",
    );

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
    assert_eq!(canon_left, canon_right);
}

#[test]
fn ac_tracing() {
    const N: usize = 5;
    let mut egraph = EGraph::with_tracing();
    let int_prim = egraph.primitives_mut().register_type::<i64>();
    let num_table = egraph.add_table(
        vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "num",
    );
    let add_table = egraph.add_table(
        vec![ColumnTy::Id; 3],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "add",
    );

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
    assert_eq!(canon_left, canon_right);
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

#[test]
fn ac_fail() {
    const N: usize = 5;
    let mut egraph = EGraph::default();
    egraph.primitives_mut().register_type::<i64>();
    let int_prim = egraph.primitives_mut().get_ty::<i64>();
    let one = egraph.primitives_mut().get(1i64);
    let num_table = egraph.add_table(
        vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "num",
    );
    let add_table = egraph.add_table(
        vec![ColumnTy::Id; 3],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "add",
    );

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
    let handles = Vec::from_iter((0..2).map(|_| thread::spawn(|| math_test(EGraph::default()))));
    handles.into_iter().for_each(|h| h.join().unwrap());
}

#[test]
fn math_tracing() {
    math_test(EGraph::with_tracing())
}

fn math_test(mut egraph: EGraph) {
    const N: usize = 8;
    let rational_ty = egraph.primitives_mut().register_type::<Rational64>();
    let string_ty = egraph.primitives_mut().register_type::<&'static str>();
    // tables
    let diff = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "diff",
    );
    let integral = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "integral",
    );
    let add = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "add",
    );
    let sub = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "sub",
    );
    let mul = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "mul",
    );
    let div = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "div",
    );
    let pow = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "pow",
    );

    let ln = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "ln",
    );
    let sqrt = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "sqrt",
    );
    let sin = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "sin",
    );
    let cos = egraph.add_table(
        vec![ColumnTy::Id, ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "cos",
    );
    let rat = egraph.add_table(
        vec![ColumnTy::Primitive(rational_ty), ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "rat",
    );
    let var = egraph.add_table(
        vec![ColumnTy::Primitive(string_ty), ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "var",
    );

    let zero = egraph.primitives_mut().get(Rational64::new(0, 1));
    let one = egraph.primitives_mut().get(Rational64::new(1, 1));
    let neg1 = egraph.primitives_mut().get(Rational64::new(-1, 1));
    let two = egraph.primitives_mut().get(Rational64::new(2, 1));
    let three = egraph.primitives_mut().get(Rational64::new(3, 1));
    let seven = egraph.primitives_mut().get(Rational64::new(7, 1));
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
            [egraph] ((-> (sub x y) id)) => ((set (add x (mul (rat {neg1}) y)) id))
        },
        define_rule! {
            [egraph] ((-> (add a (rat {zero})) id)) => ((union a id))
        },
        define_rule! {
            [egraph] ((-> (rat {zero}) z_id) (-> (mul a z_id) id))
                    => ((union id z_id))
        },
        define_rule! {
            [egraph] ((-> (mul a (rat {one})) id)) => ((union a id))
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
            [egraph] ((-> (pow x (rat {one})) id)) => ((union x id))
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
            [egraph] ((-> (diff x (cos x)) id)) => ((set (mul (rat {neg1}) (sin x)) id))
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
    let num_table = egraph.add_table(
        vec![ColumnTy::Primitive(int_prim), ColumnTy::Id],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "num",
    );
    let add_table = egraph.add_table(
        vec![ColumnTy::Id; 3],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "add",
    );
    let vec_table = egraph.add_table(
        vec![ColumnTy::Id; 2],
        DefaultVal::FreshId,
        MergeFn::UnionId,
        "vec",
    );
    let int_add = core_relations::lift_function! {
        [egraph.primitives_mut()] fn add(x: i64, y: i64) -> i64 {
            x + y
        }
    };
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
        let mut rb = egraph.new_query();
        let vec = rb.new_var(ColumnTy::Id);
        let vec_id = rb.new_var(ColumnTy::Id);
        rb.add_atom(Function::Table(vec_table), &[vec.into(), vec_id.into()])
            .unwrap();
        let last = rb.call_external_func(vec_last, &[vec.into()], ColumnTy::Id);
        let add_last_0 = rb.lookup(Function::Table(add_table), &[last.into(), ids[0].into()]);
        let add_0_last = rb.lookup(Function::Table(add_table), &[ids[0].into(), last.into()]);
        let new_vec_1 =
            rb.call_external_func(vec_push, &[vec.into(), add_last_0.into()], ColumnTy::Id);
        let new_vec_2 =
            rb.call_external_func(vec_push, &[vec.into(), add_0_last.into()], ColumnTy::Id);
        rb.lookup(Function::Table(vec_table), &[new_vec_1.into()]);
        rb.lookup(Function::Table(vec_table), &[new_vec_2.into()]);
        rb.build()
    };

    let eval_add = {
        let mut rb = egraph.new_query();
        let lhs_raw = rb.new_var(ColumnTy::Primitive(int_prim));
        let lhs_id = rb.new_var(ColumnTy::Id);
        let rhs_raw = rb.new_var(ColumnTy::Primitive(int_prim));
        let rhs_id = rb.new_var(ColumnTy::Id);
        let add_id = rb.new_var(ColumnTy::Id);
        rb.add_atom(Function::Table(num_table), &[lhs_raw.into(), lhs_id.into()])
            .unwrap();
        rb.add_atom(Function::Table(num_table), &[rhs_raw.into(), rhs_id.into()])
            .unwrap();
        rb.add_atom(
            Function::Table(add_table),
            &[lhs_id.into(), rhs_id.into(), add_id.into()],
        )
        .unwrap();
        let evaled = rb.lookup(Function::Prim(int_add), &[lhs_raw.into(), rhs_raw.into()]);
        let boxed = rb.lookup(Function::Table(num_table), &[evaled.into()]);
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
