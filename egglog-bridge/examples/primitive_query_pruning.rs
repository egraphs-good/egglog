//! Characterize how delaying query primitives until actions repeats their work.
//!
//! Run with `cargo run -p egglog-bridge --example primitive_query_pruning`.
//! Both versions compute A(x) × B(y), retaining only x = 0. The materialized
//! version filters A in a separate rule before expanding against B. This is a
//! comparison of delayed primitives, the query-side prototype, and an explicit
//! intermediate table. It checks result equality and actual invocation counts.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use egglog_bridge::{
    ColumnTy, DefaultVal, EGraph, FunctionConfig, FunctionId, MergeFn, QueryEntry,
};

use egglog_core_relations::{Value, make_external_func};

const INPUTS: usize = 32;
const EXPANSION: usize = 128;

fn table(egraph: &mut EGraph, name: &str, key_ty: ColumnTy, keys: usize) -> FunctionId {
    let mut schema = vec![key_ty; keys];
    schema.push(ColumnTy::Id);
    egraph.add_table(FunctionConfig {
        schema,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: name.into(),
        can_subsume: false,
    })
}

fn run(
    materialize_filter: bool,
    pushdown: bool,
    no_decomp: bool,
    seminaive: bool,
) -> (Vec<(i64, i64)>, usize) {
    let mut egraph = EGraph::default();
    let int_ty = ColumnTy::Base(egraph.base_values_mut().register_type::<i64>());
    let unit_ty = ColumnTy::Base(egraph.base_values_mut().register_type::<()>());
    let a = table(&mut egraph, "a", int_ty, 1);
    let b = table(&mut egraph, "b", int_ty, 1);
    let filtered_a = table(&mut egraph, "filtered_a", int_ty, 1);
    let result = table(&mut egraph, "result", int_ty, 2);
    let mut rows = Vec::new();
    for (source, len) in [(a, INPUTS), (b, EXPANSION)] {
        for value in 0..len {
            let id = egraph.fresh_id();
            rows.push((source, vec![egraph.base_values().get(value as i64), id]));
        }
    }
    egraph.add_values(rows);

    let invocations = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&invocations);
    let is_zero = egraph.register_external_func(Box::new(make_external_func(
        move |state, args| -> Option<Value> {
            let [x] = args else { unreachable!() };
            calls.fetch_add(1, Ordering::Relaxed);
            (state.base_values().unwrap::<i64>(*x) == 0).then(|| state.base_values().get(()))
        },
    )));
    let unit = egraph.base_value_constant(());

    if materialize_filter {
        let filter = {
            let mut rb = egraph.new_rule("filter a before expansion", seminaive);
            rb.set_primitive_pushdown(pushdown);
            rb.set_no_decomp(no_decomp);
            let x: QueryEntry = rb.new_var(int_ty).into();
            let x_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
            rb.query_table(a, &[x.clone(), x_id.clone()], Some(false))
                .unwrap();
            rb.query_prim(is_zero, &[x.clone(), unit.clone()], unit_ty)
                .unwrap();
            rb.set(filtered_a, &[x, x_id]);
            rb.build()
        };
        egraph.run_rules(&[filter], None).unwrap();
        assert_eq!(egraph.table_size(filtered_a), 1);
    }

    let expand = {
        let mut rb = egraph.new_rule("expand and write results", seminaive);
        rb.set_primitive_pushdown(pushdown);
        rb.set_no_decomp(no_decomp);
        let x: QueryEntry = rb.new_var(int_ty).into();
        let y: QueryEntry = rb.new_var(int_ty).into();
        let x_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let y_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(
            if materialize_filter { filtered_a } else { a },
            &[x.clone(), x_id.clone()],
            Some(false),
        )
        .unwrap();
        if !materialize_filter {
            // With pushdown disabled, builder order does not move the primitive
            // ahead of the join. With pushdown enabled, x alone makes it ready.
            rb.query_prim(is_zero, &[x.clone(), unit], unit_ty).unwrap();
        }
        rb.query_table(b, &[y.clone(), y_id], Some(false)).unwrap();
        rb.set(result, &[x, y, x_id]);
        rb.build()
    };
    egraph.run_rules(&[expand], None).unwrap();
    let calls_after_first = invocations.load(Ordering::Relaxed);
    if seminaive {
        assert!(!egraph.run_rules(&[expand], None).unwrap().changed());
        assert_eq!(invocations.load(Ordering::Relaxed), calls_after_first);
    }

    let mut results = Vec::new();
    egraph.for_each(result, |row| {
        results.push((
            egraph.base_values().unwrap::<i64>(row.vals[0]),
            egraph.base_values().unwrap::<i64>(row.vals[1]),
        ));
    });
    results.sort_unstable();
    (results, calls_after_first)
}

#[allow(clippy::disallowed_macros)]
fn main() {
    for seminaive in [true, false] {
        for no_decomp in [false, true] {
            let (late_results, late_calls) = run(false, false, no_decomp, seminaive);
            let (early_results, early_calls) = run(true, false, no_decomp, seminaive);
            let (pushed_results, pushed_calls) = run(false, true, no_decomp, seminaive);
            assert_eq!(late_results, early_results);
            assert_eq!(late_results, pushed_results);
            assert_eq!(
                late_results,
                (0..EXPANSION as i64).map(|y| (0, y)).collect::<Vec<_>>()
            );
            assert_eq!(late_calls, INPUTS * EXPANSION);
            assert_eq!(early_calls, INPUTS);
            assert_eq!(pushed_calls, INPUTS);
            println!(
                "seminaive={seminaive}, no_decomp={no_decomp}: results={}, delayed primitive calls={late_calls}, prefiltered calls={early_calls}, pushed calls={pushed_calls}, reduction={}x",
                late_results.len(),
                late_calls / early_calls,
            );
        }
    }
}
