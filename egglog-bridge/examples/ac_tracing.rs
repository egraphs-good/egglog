use std::mem;

use egglog_bridge::{define_rule, ColumnTy, DefaultVal, EGraph, MergeFn};

fn main() {
    const N: usize = 12;
    env_logger::init();
    #[cfg(feature = "serial_examples")]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    }
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
    let num_rows = (0..N)
        .map(|i| {
            let id = egraph.fresh_id();
            let i = egraph.primitives().get(i as i64);
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

    // Don't drop the egraph since we are about to exit. Egglog does this too.
    mem::forget(egraph);
}
