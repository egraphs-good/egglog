use std::{
    fmt::Debug,
    hash::Hash,
    slice,
    sync::{
        Arc, Once,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use crate::core_relations;
use crate::core_relations::{
    ContainerValue, ExternalFunctionId, Rebuilder, Value, make_external_func,
};
use crate::numeric_id::NumericId;
use crate::partition_refinement::{
    ConstantPartitionHasher, Crc32PartitionHasher, PartitionRefinementHasher,
};
use crate::partition_refinement::crc32_hash::crc32_hash;
use hashbrown::{HashMap, HashSet};
use log::debug;
use num_rational::Rational64;

use crate::{
    ColumnTy, DefaultVal, EGraph, FunctionConfig, FunctionId, MergeFn, ProofStore, QueryEntry,
    RefinementInput, SchemaMath, add_expressions, define_rule,
};

/// Run a simple associativity/commutativity test. In addition to testing that the rules properly
/// reassociate a nested sum, this test checks a proof of an arbitrary term in the database if
/// `tracing` is true.
///
/// The `can_subsume` argument is only used to enable subsumption on the underlying tables created
/// during this test, and exercise the different column handling caused by enabling subsumption.
/// Subsumption itself is not used.
fn ac_test(tracing: bool, can_subsume: bool, row_id: bool) {
    const N: usize = 5;
    let mut egraph = if tracing {
        EGraph::with_tracing()
    } else {
        EGraph::default()
    };
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume,
        row_id,
    });
    let add_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 3],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume,
        row_id,
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
    assert!(!egraph.run_rules(&[add_comm, add_assoc]).unwrap().changed());

    // Fill the database.
    let mut ids = Vec::new();
    //  Add 0 .. N to the database.
    for i in 0..N {
        let i = egraph.base_values_mut().get(i as i64);
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
    while egraph.run_rules(&[add_comm, add_assoc]).unwrap().changed() {}
    let canon_left = egraph.get_canon_in_uf(left_root);
    let canon_right = egraph.get_canon_in_uf(right_root);
    assert_eq!(canon_left, canon_right, "failed to reassociate!");
    if row_id {
        assert_row_ids_unique(&egraph, &[num_table, add_table]);
    }
    if tracing {
        let mut row = Vec::new();
        egraph.for_each(add_table, |func_row| {
            assert!(!func_row.subsumed);
            row.clear();
            row.extend_from_slice(func_row.vals);
        });

        let term_id = egraph.lookup_id(add_table, &row[0..row.len() - 1]).unwrap();
        let mut proof_store = ProofStore::default();
        let _term_explanation = egraph.explain_term(term_id, &mut proof_store).unwrap();
        let _eq_explanation = egraph
            .explain_terms_equal(left_root, right_root, &mut proof_store)
            .unwrap();
        // to print:
        // proof_store
        //     .print_eq_proof(_eq_explanation, &mut std::io::stderr())
        //     .unwrap();
    }
}

#[test]
fn ac() {
    ac_test(false, false, false);
}

#[test]
fn ac_subsume() {
    ac_test(false, true, false);
}

#[test]
fn ac_with_rowid() {
    ac_test(false, false, true);
}

#[test]
fn ac_subsume_with_rowid() {
    ac_test(false, true, true);
}

#[test]
fn ac_fail() {
    const N: usize = 5;
    let mut egraph = EGraph::default();
    egraph.base_values_mut().register_type::<i64>();
    let int_base = egraph.base_values_mut().get_ty::<i64>();
    let one = egraph.base_value_constant(1i64);
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
        row_id: false,
    });
    let add_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 3],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume: false,
        row_id: false,
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
    assert!(!egraph.run_rules(&[add_comm, add_assoc]).unwrap().changed());

    // Fill the database.
    let mut ids = Vec::new();
    //  Add 0 .. N to the database.
    let num_rows = (0..N)
        .map(|i| {
            let id = egraph.fresh_id();
            let i = egraph.base_values_mut().get(i as i64);
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
    while egraph.run_rules(&[add_comm, add_assoc]).unwrap().changed() {}
    let canon_left = egraph.get_canon_in_uf(left_root);
    let canon_right = egraph.get_canon_in_uf(right_root);
    assert_ne!(canon_left, canon_right);
}

#[test]
fn math() {
    let handles = Vec::from_iter(
        (0..2).map(|_| thread::spawn(|| math_test(EGraph::default(), false, false))),
    );
    handles.into_iter().for_each(|h| h.join().unwrap());
}

#[test]
fn math_subsume() {
    let handles =
        Vec::from_iter((0..2).map(|_| thread::spawn(|| math_test(EGraph::default(), true, false))));
    handles.into_iter().for_each(|h| h.join().unwrap());
}

#[test]
fn math_with_rowid() {
    let handles =
        Vec::from_iter((0..2).map(|_| thread::spawn(|| math_test(EGraph::default(), false, true))));
    handles.into_iter().for_each(|h| h.join().unwrap());
}

#[test]
fn math_subsume_with_rowid() {
    let handles =
        Vec::from_iter((0..2).map(|_| thread::spawn(|| math_test(EGraph::default(), true, true))));
    handles.into_iter().for_each(|h| h.join().unwrap());
}

/// Run a more complex benchmark from the egg and egglog test suite. The core of this test is to
/// ensure that the test generates a set of tables of exactly the same
/// size that the corresponding rules in egglog do in egglog's initial implementation.
///
/// As in `ac_test` the `can_subsume` argument is only used to enable subsumption on the underlying
/// tables created during this test, and exercise the different column handling caused by enabling
/// subsumption. Subsumption itself is not used.
fn math_test(mut egraph: EGraph, can_subsume: bool, row_id: bool) {
    const N: usize = 8;
    let rational_ty = egraph.base_values_mut().register_type::<Rational64>();
    let string_ty = egraph.base_values_mut().register_type::<&'static str>();
    // tables
    let diff = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "diff".into(),
        can_subsume,
        row_id,
    });
    let integral = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "integral".into(),
        can_subsume,
        row_id,
    });
    let add = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume,
        row_id,
    });
    let sub = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "sub".into(),
        can_subsume,
        row_id,
    });
    let mul = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "mul".into(),
        can_subsume,
        row_id,
    });
    let div = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "div".into(),
        can_subsume,
        row_id,
    });
    let pow = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "pow".into(),
        can_subsume,
        row_id,
    });

    let ln = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "ln".into(),
        can_subsume,
        row_id,
    });
    let sqrt = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "sqrt".into(),
        can_subsume,
        row_id,
    });
    let sin = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "sin".into(),
        can_subsume,
        row_id,
    });
    let cos = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "cos".into(),
        can_subsume,
        row_id,
    });
    let rat = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(rational_ty), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rat".into(),
        can_subsume,
        row_id,
    });
    let var = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(string_ty), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "var".into(),
        can_subsume,
        row_id,
    });

    let zero = egraph.base_value_constant(Rational64::new(0, 1));
    let one = egraph.base_value_constant(Rational64::new(1, 1));
    let neg1 = egraph.base_value_constant(Rational64::new(-1, 1));
    let two = egraph.base_value_constant(Rational64::new(2, 1));
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
        let one = egraph.base_values_mut().get(Rational64::new(1, 1));
        let two = egraph.base_values_mut().get(Rational64::new(2, 1));
        let three = egraph.base_values_mut().get(Rational64::new(3, 1));
        let seven = egraph.base_values_mut().get(Rational64::new(7, 1));
        let x_str = egraph.base_values_mut().get::<&'static str>("x");
        let y_str = egraph.base_values_mut().get::<&'static str>("y");
        let five_str = egraph.base_values_mut().get::<&'static str>("five");
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
        if !egraph.run_rules(&rules).unwrap().changed() {
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
    if row_id {
        assert_row_ids_unique(
            &egraph,
            &[
                diff, integral, add, sub, mul, div, pow, ln, sqrt, sin, cos, rat, var,
            ],
        );
    }

    if egraph.tracing {
        let mut row = Vec::new();
        egraph.for_each(mul, |func_row| {
            assert!(!func_row.subsumed);
            row.clear();
            row.extend_from_slice(func_row.vals);
        });
        let term_id = egraph.lookup_id(mul, &row[0..row.len() - 1]).unwrap();
        let mut proof_store = ProofStore::default();
        let _explain = egraph.explain_term(term_id, &mut proof_store).unwrap();
    }
}

fn assert_row_ids_unique(egraph: &EGraph, tables: &[FunctionId]) {
    let mut seen = HashSet::new();
    for table in tables {
        let info = &egraph.funcs[*table];
        assert!(info.row_id, "row ids are not enabled for {}", info.name);
        let table = egraph.db.get_table(info.table);
        egraph.scan_table(table, |row| {
            let row_id = *row.last().expect("row id column missing");
            assert!(
                seen.insert(row_id),
                "duplicate row id {row_id:?} in {}",
                info.name
            );
        });
    }
}

#[allow(unused)]
fn init_envlog() {
    static INIT_ENVLOG: Once = Once::new();
    INIT_ENVLOG.call_once(env_logger::init);
}

/// Debug helper for partition refinement tests. Dumps fingerprint and node-hash
/// state, plus expected hashes, when `RUST_LOG=info` is enabled.
#[allow(unused)]
pub(crate) fn dump_partition_refinement_state(egraph: &EGraph, label: &str, ids: &[Value]) {
    init_envlog();
    if !log::log_enabled!(log::Level::Info) {
        return;
    }
    let state = egraph
        .partition_refinement
        .as_ref()
        .expect("partition refinement should be enabled");
    log::info!("-- {label} --");
    let fingerprint = egraph.db.get_table(state.fingerprint_table.table);
    let fp_key_idx = state.fingerprint_table.key_col.index();
    let fp_hash_idx = state.fingerprint_table.hash_col.index();
    let fp_block_idx = state.fingerprint_table.block_col.index();
    let fp_ts_idx = state.fingerprint_table.ts_col.index();
    let node_hash = egraph.db.get_table(state.node_hash_table.table);
    let nh_row_id_idx = state.node_hash_table.row_id_col.index();
    let nh_hash_idx = state.node_hash_table.hash_col.index();
    let nh_eclass_idx = state.node_hash_table.eclass_col.index();
    let mut blocks = HashMap::new();
    let mut all_ids = HashSet::new();
    egraph.scan_table(fingerprint, |row| {
        let eclass = row[fp_key_idx];
        blocks.insert(eclass, row[fp_block_idx]);
        all_ids.insert(eclass);
    });
    egraph.scan_table(node_hash, |row| {
        all_ids.insert(row[nh_eclass_idx]);
    });
    let mut ids_to_dump = if ids.is_empty() {
        all_ids.into_iter().collect::<Vec<_>>()
    } else {
        ids.to_vec()
    };
    ids_to_dump.sort();
    ids_to_dump.dedup();
    let mut expected_by_row = HashMap::new();
    let hash_func = state.hash_func;
    egraph.db.with_execution_state(|exec| {
        for (_, info) in egraph.funcs.iter() {
            if info.ret_ty() != ColumnTy::Id {
                continue;
            }
            let schema_math = SchemaMath {
                tracing: egraph.tracing,
                subsume: info.can_subsume,
                row_id: info.row_id,
                func_cols: info.schema.len(),
            };
            if !info.row_id {
                log::info!("skipping {}: row ids not enabled", info.name);
                continue;
            }
            let row_id_idx = schema_math.row_id_col();
            let ret_idx = info.schema.len() - 1;
            let table = egraph.db.get_table(info.table);
            let mut hash_inputs = Vec::with_capacity(ret_idx + 1);
            egraph.scan_table(table, |row| {
                let row_id = row[row_id_idx];
                let eclass = row[ret_idx];
                hash_inputs.clear();
                hash_inputs.push(Value::from_usize(info.table.index()));
                for (col_idx, ty) in info.schema[..ret_idx].iter().enumerate() {
                    let val = row[col_idx];
                    match ty {
                        ColumnTy::Id => {
                            let Some(block) = blocks.get(&val) else {
                                log::info!(
                                    "missing fingerprint block for child {val:?} in {}",
                                    info.name
                                );
                                return;
                            };
                            hash_inputs.push(*block);
                        }
                        ColumnTy::Base(_) => hash_inputs.push(val),
                    }
                }
                let hash = exec
                    .call_external_func(hash_func, &hash_inputs)
                    .expect("hash function should return a value");
                log::info!(
                    "row_id={row_id:?}, eclass={eclass:?}, hash={hash:?}, ts={:?}",
                    row[schema_math.ts_col()]
                );
                expected_by_row.insert(row_id, (hash, eclass));
            });
        }
    });
    for &id in &ids_to_dump {
        let canon = egraph.get_canon_repr(id, ColumnTy::Id);
        let row = fingerprint
            .get_row(&[canon])
            .expect("missing fingerprint row");
        log::info!(
            "eclass {id:?} canon={canon:?} hash={:?} block={:?} ts={:?}",
            row.vals[fp_hash_idx],
            row.vals[fp_block_idx],
            row.vals[fp_ts_idx]
        );
        let mut entries = Vec::new();
        let mut expected_sum = 0u32;
        egraph.scan_table(node_hash, |row| {
            if row[nh_eclass_idx] == canon {
                let row_id = row[nh_row_id_idx];
                let actual_hash = row[nh_hash_idx];
                let expected_hash = expected_by_row.get(&row_id).copied().map(|(hash, eclass)| {
                    if eclass != canon {
                        log::info!("row_id {row_id:?} expected eclass {eclass:?} actual {canon:?}");
                    }
                    hash
                });
                if let Some(hash) = expected_hash {
                    expected_sum = expected_sum.wrapping_add(hash.rep());
                }
                entries.push((row_id, actual_hash, expected_hash));
            }
        });
        let sum = entries
            .iter()
            .fold(0u32, |acc, (_, hash, _)| acc.wrapping_add(hash.rep()));
        log::info!(
            "node_hash entries={entries:?} sum={:?} expected_sum={:?}",
            Value::new(sum),
            Value::new(expected_sum)
        );
    }
}

fn partition_refinement_egraph<H: PartitionRefinementHasher>() -> EGraph {
    EGraph::with_partition_refinement_with_hasher::<H>()
}

fn assert_partition_refinement_hashes<H: PartitionRefinementHasher>(egraph: &EGraph) {
    let state = egraph
        .partition_refinement
        .as_ref()
        .expect("partition refinement should be enabled");
    let fingerprint = egraph.db.get_table(state.fingerprint_table.table);
    let node_hash = egraph.db.get_table(state.node_hash_table.table);
    let fp_key_idx = state.fingerprint_table.key_col.index();
    let fp_hash_idx = state.fingerprint_table.hash_col.index();
    let fp_block_idx = state.fingerprint_table.block_col.index();
    let mut actual_blocks = HashMap::new();
    let mut actual_hashes = HashMap::new();
    egraph.scan_table(fingerprint, |row| {
        let eclass = row[fp_key_idx];
        actual_blocks.insert(eclass, row[fp_block_idx]);
        actual_hashes.insert(eclass, row[fp_hash_idx]);
    });

    let mut expected_node = HashMap::new();
    let mut expected_eclass = HashMap::new();
    for (_, info) in egraph.funcs.iter() {
        if info.ret_ty() != ColumnTy::Id {
            continue;
        }
        let schema_math = SchemaMath {
            tracing: egraph.tracing,
            subsume: info.can_subsume,
            row_id: info.row_id,
            func_cols: info.schema.len(),
        };
        assert!(info.row_id, "row ids are not enabled for {}", info.name);
        let row_id_idx = schema_math.row_id_col();
        let ret_idx = info.schema.len() - 1;
        let table = egraph.db.get_table(info.table);
        let mut hash_inputs = Vec::with_capacity(ret_idx + 1);
        egraph.scan_table(table, |row| {
            let row_id = row[row_id_idx];
            let eclass = row[ret_idx];
            hash_inputs.clear();
            hash_inputs.push(Value::from_usize(info.table.index()));
            for (col_idx, ty) in info.schema[..ret_idx].iter().enumerate() {
                let val = row[col_idx];
                match ty {
                    ColumnTy::Id => {
                        let block = *actual_blocks
                            .get(&val)
                            .unwrap_or_else(|| panic!("missing fingerprint block for {val:?}"));
                        hash_inputs.push(block);
                    }
                    ColumnTy::Base(_) => hash_inputs.push(val),
                }
            }
            let hash = H::hash(&hash_inputs);
            if let Some((prev_hash, prev_eclass)) = expected_node.insert(row_id, (hash, eclass)) {
                assert_eq!(
                    prev_hash, hash,
                    "node hash mismatch for duplicated row id {row_id:?}"
                );
                assert_eq!(
                    prev_eclass, eclass,
                    "node eclass mismatch for duplicated row id {row_id:?}"
                );
            }
            let entry = expected_eclass.entry(eclass).or_insert(Value::new_const(0));
            let sum = entry.rep().wrapping_add(hash.rep());
            *entry = Value::new(sum);
        });
    }

    let nh_row_id_idx = state.node_hash_table.row_id_col.index();
    let nh_hash_idx = state.node_hash_table.hash_col.index();
    let nh_eclass_idx = state.node_hash_table.eclass_col.index();
    let mut actual_node = HashMap::new();
    egraph.scan_table(node_hash, |row| {
        let row_id = row[nh_row_id_idx];
        let hash = row[nh_hash_idx];
        let eclass = row[nh_eclass_idx];
        if let Some((prev_hash, prev_eclass)) = actual_node.insert(row_id, (hash, eclass)) {
            assert_eq!(
                prev_hash, hash,
                "duplicate node hash row for row id {row_id:?}"
            );
            assert_eq!(
                prev_eclass, eclass,
                "duplicate node hash row for row id {row_id:?}"
            );
        }
    });

    for (row_id, (expected_hash, expected_eclass)) in expected_node.iter() {
        let Some((actual_hash, actual_eclass)) = actual_node.get(row_id) else {
            panic!("missing node hash entry for row id {row_id:?}");
        };
        assert_eq!(
            actual_hash, expected_hash,
            "node hash mismatch for row id {row_id:?}"
        );
        assert_eq!(
            actual_eclass, expected_eclass,
            "node eclass mismatch for row id {row_id:?}"
        );
    }
    for row_id in actual_node.keys() {
        if !expected_node.contains_key(row_id) {
            panic!("unexpected node hash entry for row id {row_id:?}");
        }
    }

    for (eclass, actual_hash) in actual_hashes.iter() {
        let expected = expected_eclass
            .get(eclass)
            .copied()
            .unwrap_or(Value::new_const(0));
        assert_eq!(
            actual_hash, &expected,
            "fingerprint hash mismatch for eclass {eclass:?}"
        );
    }
    for eclass in expected_eclass.keys() {
        if !actual_hashes.contains_key(eclass) {
            panic!("missing fingerprint row for eclass {eclass:?}");
        }
    }
}

fn run_partition_refinement_and_check<H: PartitionRefinementHasher>(egraph: &mut EGraph) {
    egraph
        .run_hash_partition_refinement()
        .expect("partition refinement failed");
    assert_partition_refinement_hashes::<H>(egraph);
}

fn run_partition_refinement_no_collisions_and_check<H: PartitionRefinementHasher>(
    egraph: &mut EGraph,
) {
    egraph
        .run_hash_partition_refinement_no_collisions()
        .expect("partition refinement failed");
    assert_partition_refinement_hashes::<H>(egraph);
}

fn partition_refinement_scaffolds_row_ids_and_rules_impl<H: PartitionRefinementHasher>() {
    let mut egraph = partition_refinement_egraph::<H>();
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
        row_id: false,
    });
    let value = egraph.base_values_mut().get(42_i64);
    let _ = egraph.add_term(num_table, &[value], "num");
    assert_row_ids_unique(&egraph, &[num_table]);
    let state = egraph
        .partition_refinement
        .as_ref()
        .expect("partition refinement should be enabled");
    assert_eq!(state.seed_rules.len(), 1);
    assert_eq!(state.node_hash_rules.len(), 1);
    let _ = egraph.db.get_table(state.node_hash_table.table);
    let _ = egraph.db.get_table(state.fingerprint_table.table);
}

#[test]
fn partition_refinement_scaffolds_row_ids_and_rules_crc() {
    partition_refinement_scaffolds_row_ids_and_rules_impl::<Crc32PartitionHasher>();
}

#[test]
fn partition_refinement_scaffolds_row_ids_and_rules_constant() {
    partition_refinement_scaffolds_row_ids_and_rules_impl::<ConstantPartitionHasher>();
}

fn add_link_table(egraph: &mut EGraph) -> FunctionId {
    egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "link".into(),
        can_subsume: false,
        row_id: false,
    })
}

fn add_lam_app_var_tables(egraph: &mut EGraph) -> (FunctionId, FunctionId, FunctionId) {
    let lam = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "lam".into(),
        can_subsume: false,
        row_id: false,
    });
    let app = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "app".into(),
        can_subsume: false,
        row_id: false,
    });
    let var = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "var".into(),
        can_subsume: false,
        row_id: false,
    });
    (lam, app, var)
}

fn fingerprint_block(egraph: &EGraph, eclass: Value) -> Value {
    let canon = egraph.get_canon_repr(eclass, ColumnTy::Id);
    let state = egraph
        .partition_refinement
        .as_ref()
        .expect("partition refinement should be enabled");
    let table = egraph.db.get_table(state.fingerprint_table.table);
    let row = table.get_row(&[canon]).expect("missing fingerprint row");
    row.vals[state.fingerprint_table.block_col.index()]
}

fn partition_refinement_lambda_splits_lam_blocks_impl<H: PartitionRefinementHasher>() {
    let mut egraph = partition_refinement_egraph::<H>();
    let (lam, app, var) = add_lam_app_var_tables(&mut egraph);

    // Build (lam x (lam y (app y x))) with cyclic variable references.
    let outer_lam = egraph.fresh_id();
    let inner_lam = egraph.fresh_id();
    let var_x = egraph.fresh_id();
    let var_y = egraph.fresh_id();
    let app_yx = egraph.fresh_id();
    egraph.add_values([
        (var, vec![outer_lam, var_x]),
        (var, vec![inner_lam, var_y]),
        (app, vec![var_y, var_x, app_yx]),
        (lam, vec![app_yx, inner_lam]),
        (lam, vec![inner_lam, outer_lam]),
    ]);

    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_outer = fingerprint_block(&egraph, outer_lam);
    let block_inner = fingerprint_block(&egraph, inner_lam);
    assert_ne!(block_outer, block_inner);
}

#[test]
fn partition_refinement_lambda_splits_lam_blocks_crc() {
    partition_refinement_lambda_splits_lam_blocks_impl::<Crc32PartitionHasher>();
}

fn partition_refinement_self_cycles_share_block_impl<H: PartitionRefinementHasher>() {
    let mut egraph = partition_refinement_egraph::<H>();
    let link = add_link_table(&mut egraph);
    let first = egraph.fresh_id();
    let second = egraph.fresh_id();
    let third = egraph.fresh_id();
    let fourth = egraph.fresh_id();
    egraph.add_values([
        (link, vec![first, first]),
        (link, vec![second, second]),
        (link, vec![third, fourth]),
    ]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_first = fingerprint_block(&egraph, first);
    let block_second = fingerprint_block(&egraph, second);
    let block_third = fingerprint_block(&egraph, third);
    let block_fourth = fingerprint_block(&egraph, fourth);
    assert_eq!(block_first, block_second);
    assert_ne!(block_first, block_third);
    assert_ne!(block_first, block_fourth);
    assert_ne!(block_third, block_fourth);
}

#[test]
fn partition_refinement_self_cycles_share_block_crc() {
    partition_refinement_self_cycles_share_block_impl::<Crc32PartitionHasher>();
}

#[test]
fn partition_refinement_self_cycles_share_block_constant() {
    partition_refinement_self_cycles_share_block_impl::<ConstantPartitionHasher>();
}

fn partition_refinement_two_cycles_share_block_impl<H: PartitionRefinementHasher>() {
    let mut egraph = partition_refinement_egraph::<H>();
    let link = add_link_table(&mut egraph);
    let a = egraph.fresh_id();
    let b = egraph.fresh_id();
    let c = egraph.fresh_id();
    let d = egraph.fresh_id();
    egraph.add_values([
        (link, vec![a, b]),
        (link, vec![b, a]),
        (link, vec![c, d]),
        (link, vec![d, c]),
    ]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a = fingerprint_block(&egraph, a);
    let block_b = fingerprint_block(&egraph, b);
    let block_c = fingerprint_block(&egraph, c);
    let block_d = fingerprint_block(&egraph, d);
    assert_eq!(block_a, block_b);
    assert_eq!(block_a, block_c);
    assert_eq!(block_a, block_d);
}

#[test]
fn partition_refinement_two_cycles_share_block_crc() {
    partition_refinement_two_cycles_share_block_impl::<Crc32PartitionHasher>();
}

#[test]
fn partition_refinement_two_cycles_share_block_constant() {
    partition_refinement_two_cycles_share_block_impl::<ConstantPartitionHasher>();
}

fn partition_refinement_collision_splits_blocks_impl<H: PartitionRefinementHasher>() {
    let mut egraph = partition_refinement_egraph::<H>();
    let link = add_link_table(&mut egraph);
    let pair = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "pair".into(),
        can_subsume: false,
        row_id: false,
    });
    let a = egraph.fresh_id();
    let b = egraph.fresh_id();
    let c = egraph.fresh_id();
    egraph.add_values([
        (link, vec![a, a]),
        (link, vec![b, b]),
        (pair, vec![c, c, c]),
    ]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a = fingerprint_block(&egraph, a);
    let block_b = fingerprint_block(&egraph, b);
    let block_c = fingerprint_block(&egraph, c);
    assert_eq!(block_a, block_b);
    assert_ne!(block_a, block_c);
}

#[test]
fn partition_refinement_collision_splits_blocks_crc() {
    partition_refinement_collision_splits_blocks_impl::<Crc32PartitionHasher>();
}

#[test]
fn partition_refinement_collision_splits_blocks_constant() {
    partition_refinement_collision_splits_blocks_impl::<ConstantPartitionHasher>();
}

#[test]
fn partition_refinement_constant_hash_needs_collision_resolution() {
    let mut egraph = partition_refinement_egraph::<ConstantPartitionHasher>();
    let link = add_link_table(&mut egraph);
    let pair = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "pair".into(),
        can_subsume: false,
        row_id: false,
    });
    let a = egraph.fresh_id();
    let b = egraph.fresh_id();
    let c = egraph.fresh_id();
    egraph.add_values([
        (link, vec![a, a]),
        (link, vec![b, b]),
        (pair, vec![c, c, c]),
    ]);

    run_partition_refinement_no_collisions_and_check::<ConstantPartitionHasher>(&mut egraph);
    let block_a = fingerprint_block(&egraph, a);
    let block_b = fingerprint_block(&egraph, b);
    let block_c = fingerprint_block(&egraph, c);
    assert_eq!(block_a, block_b);
    assert_eq!(block_a, block_c);

    run_partition_refinement_and_check::<ConstantPartitionHasher>(&mut egraph);
    let block_a = fingerprint_block(&egraph, a);
    let block_b = fingerprint_block(&egraph, b);
    let block_c = fingerprint_block(&egraph, c);
    assert_eq!(block_a, block_b);
    assert_ne!(block_a, block_c);
}

fn partition_refinement_cycles_broader_embed_impl<H: PartitionRefinementHasher>() {
    // Two structures like:
    //       a1               a2
    //      /  \             /  \
    //     b1  c1           b2  c2
    //     ^                ^
    //     |                |
    //     +---- self-loop  +---- self-loop
    //
    // One child is a self-loop and the other is a leaf. Corresponding nodes
    // should end up in the same block after refinement.
    let mut egraph = partition_refinement_egraph::<H>();
    let link = add_link_table(&mut egraph);
    let pair = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "pair".into(),
        can_subsume: false,
        row_id: false,
    });
    let a1 = egraph.fresh_id();
    let b1 = egraph.fresh_id();
    let c1 = egraph.fresh_id();
    let a2 = egraph.fresh_id();
    let b2 = egraph.fresh_id();
    let c2 = egraph.fresh_id();
    egraph.add_values([
        (link, vec![b1, b1]),
        (link, vec![b2, b2]),
        (pair, vec![b1, c1, a1]),
        (pair, vec![b2, c2, a2]),
    ]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a1 = fingerprint_block(&egraph, a1);
    let block_a2 = fingerprint_block(&egraph, a2);
    let block_b1 = fingerprint_block(&egraph, b1);
    let block_b2 = fingerprint_block(&egraph, b2);
    let block_c1 = fingerprint_block(&egraph, c1);
    let block_c2 = fingerprint_block(&egraph, c2);
    assert_eq!(block_a1, block_a2);
    assert_eq!(block_b1, block_b2);
    assert_eq!(block_c1, block_c2);
}

#[test]
fn partition_refinement_cycles_broader_embed_crc() {
    partition_refinement_cycles_broader_embed_impl::<Crc32PartitionHasher>();
}

#[test]
fn partition_refinement_cycles_broader_embed_constant() {
    partition_refinement_cycles_broader_embed_impl::<ConstantPartitionHasher>();
}

fn partition_refinement_incremental_updates_impl<H: PartitionRefinementHasher>(
    stable_blocks: bool,
) {
    let mut egraph = partition_refinement_egraph::<H>();
    let link = add_link_table(&mut egraph);
    let pair = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "pair".into(),
        can_subsume: false,
        row_id: false,
    });
    let a = egraph.fresh_id();
    let b = egraph.fresh_id();
    egraph.add_values([(link, vec![a, a]), (pair, vec![b, b, b])]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a0 = fingerprint_block(&egraph, a);
    let block_b0 = fingerprint_block(&egraph, b);
    assert_ne!(block_a0, block_b0);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a1 = fingerprint_block(&egraph, a);
    let block_b1 = fingerprint_block(&egraph, b);
    if stable_blocks {
        assert_eq!(block_a1, block_a0);
        assert_eq!(block_b1, block_b0);
    } else {
        assert_ne!(block_a1, block_b1);
    }

    egraph.add_values([(pair, vec![a, a, a])]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a2 = fingerprint_block(&egraph, a);
    let block_b2 = fingerprint_block(&egraph, b);
    assert_ne!(block_a2, block_b2);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let block_a3 = fingerprint_block(&egraph, a);
    let block_b3 = fingerprint_block(&egraph, b);
    if stable_blocks {
        assert_eq!(block_a3, block_a2);
        assert_eq!(block_b3, block_b2);
    } else {
        assert_ne!(block_a3, block_b3);
    }
}

#[test]
fn partition_refinement_incremental_updates_crc() {
    partition_refinement_incremental_updates_impl::<Crc32PartitionHasher>(true);
}

#[test]
fn partition_refinement_incremental_updates_constant() {
    partition_refinement_incremental_updates_impl::<ConstantPartitionHasher>(false);
}

fn partition_refinement_incremental_merges_impl<H: PartitionRefinementHasher>() {
    let mut egraph = partition_refinement_egraph::<H>();
    let link = add_link_table(&mut egraph);
    let a = egraph.fresh_id();
    let b = egraph.fresh_id();
    egraph.add_values([(link, vec![a, a]), (link, vec![b, b])]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let canon_a0 = egraph.get_canon_repr(a, ColumnTy::Id);
    let canon_b0 = egraph.get_canon_repr(b, ColumnTy::Id);
    assert_eq!(canon_a0, canon_b0);
    let block_a0 = fingerprint_block(&egraph, a);
    let block_b0 = fingerprint_block(&egraph, b);
    assert_eq!(block_a0, block_b0);

    let c = egraph.fresh_id();
    let d = egraph.fresh_id();
    egraph.add_values([(link, vec![c, c]), (link, vec![d, d])]);
    run_partition_refinement_and_check::<H>(&mut egraph);
    let canon_c0 = egraph.get_canon_repr(c, ColumnTy::Id);
    let canon_d0 = egraph.get_canon_repr(d, ColumnTy::Id);
    assert_eq!(canon_c0, canon_d0);
    let canon_a1 = egraph.get_canon_repr(a, ColumnTy::Id);
    let canon_b1 = egraph.get_canon_repr(b, ColumnTy::Id);
    assert_eq!(canon_a1, canon_b1);
    assert_eq!(canon_a1, canon_c0);
    let block_c0 = fingerprint_block(&egraph, c);
    let block_d0 = fingerprint_block(&egraph, d);
    assert_eq!(block_c0, block_d0);
    let block_a1 = fingerprint_block(&egraph, a);
    let block_b1 = fingerprint_block(&egraph, b);
    assert_eq!(block_a1, block_b1);
    assert_eq!(block_a1, block_c0);
}

#[test]
fn partition_refinement_incremental_merges_crc() {
    partition_refinement_incremental_merges_impl::<Crc32PartitionHasher>();
}

#[test]
fn partition_refinement_incremental_merges_constant() {
    partition_refinement_incremental_merges_impl::<ConstantPartitionHasher>();
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct VecContainer(Vec<Value>);
impl ContainerValue for VecContainer {
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
        let mut vec: VecContainer = state
            .container_values()
            .get_val::<VecContainer>(*vec_id)?
            .clone();
        vec.0.push(*val);
        // Vectors are immutable. May as well not use O(n) auxiliary space.
        vec.0.shrink_to_fit();
        Some(state.clone().container_values().register_val(vec, state))
    });
    egraph.register_external_func(Box::new(external_func))
}

fn register_vec_last(egraph: &mut EGraph) -> ExternalFunctionId {
    egraph.register_container_ty::<VecContainer>();
    let external_func = make_external_func(move |state, vals| -> Option<Value> {
        let [vec_id] = vals else {
            panic!("[vec-last] expected 1 value, got {vals:?}")
        };
        state
            .container_values()
            .get_val::<VecContainer>(*vec_id)?
            .0
            .last()
            .cloned()
    });
    egraph.register_external_func(Box::new(external_func))
}

fn dump_vecs(egraph: &EGraph) -> Vec<Vec<Value>> {
    let mut res = Vec::new();
    egraph
        .container_values()
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
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
        row_id: false,
    });
    let add_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 3],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume: false,
        row_id: false,
    });
    let vec_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id; 2],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "vec".into(),
        can_subsume: false,
        row_id: false,
    });
    let int_add =
        egraph.register_external_func(Box::new(make_external_func(|exec_state, args| {
            let [x, y] = args else { panic!() };
            let x: i64 = exec_state.base_values().unwrap(*x);
            let y: i64 = exec_state.base_values().unwrap(*y);
            let z: i64 = x + y;
            Some(exec_state.base_values().get(z))
        })));
    let vec_last = register_vec_last(&mut egraph);
    let vec_push = register_vec_push(&mut egraph);

    let mut ids = Vec::new();
    //  Add 0 and 1 to the database.
    let num_rows = (0..=1)
        .map(|i| {
            let id = egraph.fresh_id();
            let i = egraph.base_values_mut().get(i as i64);
            ids.push(id);
            (num_table, vec![i, id])
        })
        .collect::<Vec<_>>();
    egraph.add_values(num_rows);

    let empty_vec = egraph.get_container_value(VecContainer(vec![]));
    let vec1 = egraph.get_container_value(VecContainer(vec![ids[1]]));

    let empty_vec_id = egraph.fresh_id();
    let vec1_id = egraph.fresh_id();

    egraph.add_values(vec![
        (vec_table, vec![empty_vec, empty_vec_id]),
        (vec_table, vec![vec1, vec1_id]),
    ]);

    let vec_expand = {
        let mut rb = egraph.new_rule("", true);
        let vec: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let vec_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let last: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(vec_table, &[vec.clone(), vec_id], Some(false))
            .unwrap();
        rb.query_prim(vec_last, &[vec.clone(), last.clone()], ColumnTy::Id)
            .unwrap();
        let add_last_0 = rb
            .lookup(
                add_table,
                &[
                    last.clone(),
                    QueryEntry::Const {
                        val: ids[0],
                        ty: ColumnTy::Base(int_base),
                    },
                ],
                || "add_last_0".to_string(),
            )
            .into();
        let add_0_last = rb
            .lookup(
                add_table,
                &[
                    QueryEntry::Const {
                        val: ids[0],
                        ty: ColumnTy::Base(int_base),
                    },
                    last,
                ],
                || "add_0_last".to_string(),
            )
            .into();
        let new_vec_1 = rb
            .call_external_func(vec_push, &[vec.clone(), add_last_0], ColumnTy::Id, || {
                "".to_string()
            })
            .into();
        let new_vec_2 = rb
            .call_external_func(vec_push, &[vec, add_0_last], ColumnTy::Id, || {
                "".to_string()
            })
            .into();
        rb.lookup(vec_table, &[new_vec_1], String::new);
        rb.lookup(vec_table, &[new_vec_2], String::new);
        rb.build()
    };

    let eval_add = {
        let mut rb = egraph.new_rule("", true);
        let lhs_raw: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let lhs_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let rhs_raw: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let rhs_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let add_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(num_table, &[lhs_raw.clone(), lhs_id.clone()], Some(false))
            .unwrap();
        rb.query_table(num_table, &[rhs_raw.clone(), rhs_id.clone()], Some(false))
            .unwrap();
        rb.query_table(
            add_table,
            &[lhs_id.clone(), rhs_id.clone(), add_id.clone()],
            Some(false),
        )
        .unwrap();
        let evaled: QueryEntry = rb
            .call_external_func(
                int_add,
                &[lhs_raw.clone(), rhs_raw.clone()],
                ColumnTy::Base(int_base),
                || "".to_string(),
            )
            .into();
        let boxed: QueryEntry = rb.lookup(num_table, &[evaled.clone()], String::new).into();
        rb.union(add_id.clone(), boxed.clone());
        rb.build()
    };

    assert_unordered_eq(
        dump_vecs(&egraph),
        vec![vec![], vec![egraph.get_canon_in_uf(ids[1])]],
    );

    assert!(egraph.run_rules(&[vec_expand]).unwrap().changed());
    assert_eq!(dump_vecs(&egraph).len(), 4);
    // We have 2 new vectors with a last element. Each of those should spawn two more, adding 4.
    assert!(egraph.run_rules(&[vec_expand]).unwrap().changed());
    assert_eq!(dump_vecs(&egraph).len(), 8);
    // We have 4 new vectors with a last element. Each of those should spawn two more, adding 8.
    assert!(egraph.run_rules(&[vec_expand]).unwrap().changed());
    assert_eq!(dump_vecs(&egraph).len(), 16);

    // Now we want to saturate `eval_add`. This should collapse a bunch of new vectors.

    let mut saturated = false;
    for _ in 0..20 {
        saturated = !egraph.run_rules(&[eval_add]).unwrap().changed();
        if saturated {
            break;
        }
    }
    assert!(saturated, "failed to saturate after 20 iterations");

    let one_id = egraph.get_canon_in_uf(ids[1]);
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
fn partition_refinement_hashes_container_raw() {
    let mut egraph = EGraph::with_partition_refinement();
    egraph.register_container_ty::<VecContainer>();

    let vec_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: Some(vec![RefinementInput::Raw, RefinementInput::Block]),
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "vec".into(),
        can_subsume: false,
        row_id: false,
    });

    let vec_val = egraph.get_container_value(VecContainer(vec![]));
    let vec_id = egraph.fresh_id();
    egraph.add_values(vec![(vec_table, vec![vec_val, vec_id])]);

    let _ = egraph
        .run_hash_partition_refinement()
        .expect("partition refinement failed");

    let state = egraph
        .partition_refinement
        .as_ref()
        .expect("partition refinement should be enabled");
    let node_hash = egraph.db.get_table(state.node_hash_table.table);
    let nh_hash_idx = state.node_hash_table.hash_col.index();
    let nh_eclass_idx = state.node_hash_table.eclass_col.index();
    let mut found = None;
    egraph.scan_table(node_hash, |row| {
        if row[nh_eclass_idx] == vec_id {
            found = Some(row[nh_hash_idx]);
        }
    });

    let found = found.expect("missing node-hash entry for vec");
    let table_id_val = Value::from_usize(egraph.funcs[vec_table].table.index());
    let expected = crc32_hash(&[table_id_val, vec_val]);
    assert_eq!(found, expected);
}

#[test]
fn rhs_only_rule() {
    let mut egraph = EGraph::default();
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let zero = egraph.base_values_mut().get(0i64);
    let one = egraph.base_values_mut().get(1i64);
    let num_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".into(),
        can_subsume: false,
        row_id: false,
    });
    let add_data = {
        let zero = egraph.base_value_constant(0i64);
        let one = egraph.base_value_constant(1i64);
        let mut rb = egraph.new_rule("", true);
        let _zero_id = rb.lookup(num_table, &[zero], String::new);
        let _one_id = rb.lookup(num_table, &[one], String::new);
        rb.build()
    };

    let mut contents = Vec::new();

    assert!(contents.is_empty());
    assert!(egraph.run_rules(&[add_data]).unwrap().changed());
    egraph.for_each(num_table, |func_row| {
        assert!(!func_row.subsumed);
        contents.push(func_row.vals.to_vec());
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
    let inc_counter_func =
        egraph.register_external_func(Box::new(make_external_func(move |_, _| {
            inner.fetch_add(1, Ordering::SeqCst);
            Some(Value::new(0))
        })));
    let inc_counter_rule = {
        let mut rb = egraph.new_rule("", true);
        rb.call_external_func(inc_counter_func, &[], ColumnTy::Id, || "".to_string());
        rb.build()
    };

    assert!(!egraph.run_rules(&[inc_counter_rule]).unwrap().changed());
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert!(!egraph.run_rules(&[inc_counter_rule]).unwrap().changed());
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn mergefn_arithmetic() {
    let mut egraph = EGraph::default();
    let int_base = egraph.base_values_mut().register_type::<i64>();

    // Create external functions for multiplication and addition
    let multiply_func = egraph.register_external_func(Box::new(
        core_relations::make_external_func(|state, vals| -> Option<Value> {
            let [a, b] = vals else {
                return None;
            };
            let a_val = state.base_values().unwrap::<i64>(*a);
            let b_val = state.base_values().unwrap::<i64>(*b);
            let res = state.base_values().get::<i64>(a_val * b_val);
            Some(res)
        }),
    ));

    let add_func = egraph.register_external_func(Box::new(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a, b] = vals else {
                return None;
            };
            let a_val = state.base_values().unwrap::<i64>(*a);
            let b_val = state.base_values().unwrap::<i64>(*b);
            let res = state.base_values().get::<i64>(a_val + b_val);
            Some(res)
        },
    )));

    let value_1 = egraph.base_values_mut().get(1i64);

    // Create a function with merge function (+ 1 (* old new))
    // This uses nested MergeFn::Primitive with external functions to build the complex merge function
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Base(int_base)],
        refinement_inputs: None,
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
        row_id: false,
    });

    let value_0 = egraph.base_value_constant(0i64);
    let value_1 = egraph.base_value_constant(1i64);
    let value_2 = egraph.base_value_constant(2i64);
    let value_3 = egraph.base_value_constant(3i64);
    let value_4 = egraph.base_value_constant(4i64);
    let value_5 = egraph.base_value_constant(5i64);
    let value_6 = egraph.base_value_constant(6i64);

    // First rule writes (f 1 0) (f 2 1)
    let rule1 = {
        let mut rb = egraph.new_rule("rule1", true);
        rb.set(f_table, &[value_1.clone(), value_0]);
        rb.set(f_table, &[value_2.clone(), value_1.clone()]);
        rb.build()
    };

    // Run the first rule and check state
    assert!(egraph.run_rules(&[rule1]).unwrap().changed());
    let mut contents = Vec::new();
    egraph.for_each(f_table, |func_row| {
        assert!(!func_row.subsumed);
        contents.push((
            egraph.base_values().unwrap::<i64>(func_row.vals[0]),
            egraph.base_values().unwrap::<i64>(func_row.vals[1]),
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
    assert!(egraph.run_rules(&[rule2]).unwrap().changed());
    contents.clear();
    egraph.for_each(f_table, |func_row| {
        assert!(!func_row.subsumed);
        contents.push((
            egraph.base_values().unwrap::<i64>(func_row.vals[0]),
            egraph.base_values().unwrap::<i64>(func_row.vals[1]),
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
    assert!(egraph.run_rules(&[rule3]).unwrap().changed());
    contents.clear();
    egraph.for_each(f_table, |func_row| {
        assert!(!func_row.subsumed);
        contents.push((
            egraph.base_values().unwrap::<i64>(func_row.vals[0]),
            egraph.base_values().unwrap::<i64>(func_row.vals[1]),
        ));
    });
    contents.sort();
    assert_eq!(contents, vec![(1, 4), (2, 29)]);
}

#[test]
fn mergefn_nested_function() {
    let mut egraph = EGraph::default();
    let int_base = egraph.base_values_mut().register_type::<i64>();

    // Create a function g that will be used in the merge function for f
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: true,
        row_id: false,
    });

    // Create a function f whose merge function is (g (g new new) (g old old))
    // This uses nested MergeFn::Function to build the complex merge function
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
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
        row_id: false,
    });

    let value_1 = egraph.base_value_constant(1i64);
    let value_2 = egraph.base_value_constant(2i64);

    // Create an rhs-only rule that writes f values with fresh IDs
    // We'll run this rule multiple times and observe how the merge function works

    let write_rule = {
        let mut rb = egraph.new_rule("write_rule", true);
        rb.lookup(f_table, slice::from_ref(&value_1), String::new);
        rb.lookup(f_table, &[value_2], String::new);
        rb.build()
    };

    // Helper function to get all g-table entries
    let get_g_entries = |egraph: &EGraph| {
        let mut entries = Vec::new();
        egraph.for_each(g_table, |func_row| {
            assert!(!func_row.subsumed);
            entries.push((func_row.vals[0], func_row.vals[1], func_row.vals[2]));
        });
        entries.sort();
        entries
    };

    // Helper function to get all f-table entries
    let get_f_entries = |egraph: &EGraph| {
        let mut entries = Vec::new();
        egraph.for_each(f_table, |func_row| {
            assert!(!func_row.subsumed);
            entries.push((
                egraph.base_values().unwrap::<i64>(func_row.vals[0]),
                func_row.vals[1],
            ));
        });
        entries.sort();
        entries
    };

    // First run of the rule
    assert!(egraph.run_rules(&[write_rule]).unwrap().changed());
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
    assert!(egraph.run_rules(&[set_rule]).unwrap().changed());
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
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let bool_base = egraph.base_values_mut().register_type::<bool>();
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "f".into(),
        can_subsume: false,
        row_id: false,
    });
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: false,
        row_id: false,
    });

    let is_even = egraph.register_external_func(Box::new(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = state.base_values().unwrap::<i64>(*a);
            let result: bool = a_val % 2 == 0;
            Some(state.base_values().get(result))
        },
    )));

    let value_1 = egraph.base_value_constant(1i64);
    let value_2 = egraph.base_value_constant(2i64);
    let value_3 = egraph.base_value_constant(3i64);
    let value_true = egraph.base_value_constant(true);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.lookup(f_table, &[value_1], String::new);
        rb.lookup(f_table, &[value_2], String::new);
        rb.lookup(f_table, &[value_3], String::new);
        rb.build()
    };

    let copy_to_g = {
        let mut rb = egraph.new_rule("copy_to_g", true);
        let val: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(f_table, &[val.clone(), id.clone()], Some(false))
            .unwrap();
        rb.query_prim(
            is_even,
            &[val.clone(), value_true.clone()],
            ColumnTy::Base(bool_base),
        )
        .unwrap();
        rb.set(g_table, &[val, id]);
        rb.build()
    };
    let get_entries = |egraph: &EGraph, table: FunctionId| {
        let mut entries = Vec::new();
        egraph.for_each(table, |func_row| {
            assert!(!func_row.subsumed);
            entries.push((
                egraph.base_values().unwrap::<i64>(func_row.vals[0]),
                func_row.vals[1],
            ));
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
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "f".into(),
        can_subsume: false,
        row_id: false,
    });
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: false,
        row_id: false,
    });

    let neg = egraph.register_external_func(Box::new(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = state.base_values().unwrap::<i64>(*a);
            Some(state.base_values().get(-a_val))
        },
    )));
    let abs = egraph.register_external_func(Box::new(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = state.base_values().unwrap::<i64>(*a);
            Some(state.base_values().get(a_val.abs()))
        },
    )));

    let value_n1 = egraph.base_value_constant(-1i64);
    let value_0 = egraph.base_value_constant(0i64);
    let value_1 = egraph.base_value_constant(1i64);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.lookup(f_table, &[value_n1], String::new);
        rb.lookup(f_table, &[value_0], String::new);
        rb.lookup(f_table, &[value_1], String::new);
        rb.build()
    };

    let copy_to_g = {
        let mut rb = egraph.new_rule("copy_to_g", true);
        let val: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let negval: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        rb.query_table(f_table, &[val.clone(), id.clone()], Some(false))
            .unwrap();
        rb.query_prim(
            neg,
            &[val.clone(), negval.clone()],
            ColumnTy::Base(int_base),
        )
        .unwrap();
        rb.query_prim(
            abs,
            &[val.clone(), negval.clone()],
            ColumnTy::Base(int_base),
        )
        .unwrap();
        rb.set(g_table, &[val.clone(), id.clone()]);
        rb.build()
    };
    let get_entries = |egraph: &EGraph, table: FunctionId| {
        let mut entries = Vec::new();
        egraph.for_each(table, |func_row| {
            assert!(!func_row.subsumed);
            entries.push((
                egraph.base_values().unwrap::<i64>(func_row.vals[0]),
                func_row.vals[1],
            ));
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
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let f_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "f".into(),
        can_subsume: true,
        row_id: false,
    });
    let g_table = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "g".into(),
        can_subsume: false,
        row_id: false,
    });

    let value_1 = egraph.base_value_constant(1i64);
    let value_2 = egraph.base_value_constant(2i64);
    let value_3 = egraph.base_value_constant(3i64);
    let write_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.lookup(f_table, slice::from_ref(&value_1), String::new);
        rb.lookup(f_table, slice::from_ref(&value_2), String::new);
        rb.build()
    };

    let subsume_f = {
        let mut rb = egraph.new_rule("write_f", true);
        rb.subsume(f_table, slice::from_ref(&value_2));
        rb.subsume(f_table, slice::from_ref(&value_3));
        rb.build()
    };

    let copy_to_g = {
        let mut rb = egraph.new_rule("copy_to_g", true);
        let val: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(f_table, &[val.clone(), id.clone()], Some(false))
            .unwrap();
        rb.set(g_table, &[val, id]);
        rb.build()
    };
    let get_entries = |egraph: &EGraph, table: FunctionId| {
        let mut entries = Vec::new();
        let mut num_subsumed = 0;
        egraph.for_each(table, |func_row| {
            entries.push((
                egraph.base_values().unwrap::<i64>(func_row.vals[0]),
                func_row.vals[1],
            ));
            if func_row.subsumed {
                num_subsumed += 1;
            }
        });
        entries.sort();
        (entries, num_subsumed)
    };

    assert!(get_entries(&egraph, f_table).0.is_empty());
    assert!(get_entries(&egraph, g_table).0.is_empty());
    egraph.run_rules(&[write_f]).unwrap();
    let f = get_entries(&egraph, f_table);
    assert_eq!((f.0.len(), f.1), (2, 0));
    assert_eq!(f.0.iter().map(|(x, _)| *x).collect::<Vec<_>>(), vec![1, 2]);
    egraph.run_rules(&[subsume_f]).unwrap();
    let f = get_entries(&egraph, f_table);
    assert_eq!((f.0.len(), f.1), (3, 2));
    assert_eq!(
        f.0.iter().map(|(x, _)| *x).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    egraph.run_rules(&[copy_to_g]).unwrap();
    let g = get_entries(&egraph, g_table);
    assert_eq!((g.0.len(), g.1), (1, 0));
    assert_eq!(g.0[0], f.0[0])
}

#[test]
fn lookup_failure_panics() {
    let mut egraph = EGraph::default();
    let f = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::Fail,
        merge: MergeFn::UnionId,
        name: "test".into(),
        can_subsume: false,
        row_id: false,
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
        rb.lookup(f, slice::from_ref(&value_1), String::new);
        rb.build()
    };
    egraph.run_rules(&[lookup_success]).unwrap();

    let lookup_failure = {
        let mut rb = egraph.new_rule("lookup_fail", true);
        rb.lookup(f, slice::from_ref(&value_3), String::new);
        rb.build()
    };
    egraph.run_rules(&[lookup_failure]).err().unwrap();
}

#[test]
fn primitive_failure_panics() {
    let mut egraph = EGraph::default();
    let _int_base = egraph.base_values_mut().register_type::<i64>();
    let unit_base = egraph.base_values_mut().register_type::<()>();

    let value_1 = egraph.base_value_constant(1i64);
    let value_2 = egraph.base_value_constant(2i64);

    let assert_odd = egraph.register_external_func(Box::new(core_relations::make_external_func(
        |state, vals| -> Option<Value> {
            let [a] = vals else {
                return None;
            };
            let a_val = state.base_values().unwrap::<i64>(*a);
            if a_val % 2 == 1 {
                Some(state.base_values().get(()))
            } else {
                None
            }
        },
    )));

    let assert_odd_rule = {
        let mut rb = egraph.new_rule("assert_odd", true);
        rb.call_external_func(
            assert_odd,
            slice::from_ref(&value_1),
            ColumnTy::Base(unit_base),
            || "".to_string(),
        );
        rb.call_external_func(
            assert_odd,
            slice::from_ref(&value_2),
            ColumnTy::Base(unit_base),
            || "".to_string(),
        );
        rb.build()
    };

    egraph.run_rules(&[assert_odd_rule]).err().unwrap();
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<EGraph>()
};
