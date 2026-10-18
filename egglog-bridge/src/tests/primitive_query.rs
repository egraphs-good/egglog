use super::*;
use core_relations::PlanStrategy;

fn relation(eg: &mut EGraph, name: &str, int_ty: ColumnTy, keys: usize) -> FunctionId {
    let mut schema = vec![int_ty; keys];
    schema.push(ColumnTy::Id);
    eg.add_table(FunctionConfig {
        schema,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: name.into(),
        can_subsume: false,
    })
}

fn rows(eg: &EGraph, table: FunctionId, keys: usize) -> Vec<Vec<i64>> {
    let mut rows = Vec::new();
    eg.for_each(table, |row| {
        rows.push(
            row.vals[..keys]
                .iter()
                .map(|v| eg.base_values().unwrap::<i64>(*v))
                .collect(),
        );
    });
    rows.sort_unstable();
    rows
}

fn pruning(pushdown: bool, strategy: PlanStrategy, threads: usize) -> (Vec<Vec<i64>>, usize) {
    let mut eg = EGraph::default().with_num_threads(threads);
    let ty = ColumnTy::Base(eg.base_values_mut().register_type::<i64>());
    let a = relation(&mut eg, "a", ty, 1);
    let b = relation(&mut eg, "b", ty, 1);
    let out = relation(&mut eg, "out", ty, 2);
    for (table, count) in [(a, 8), (b, 64)] {
        for x in 0..count {
            let value = eg.base_values().get(x as i64);
            eg.add_term(table, &[value]);
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let filter = eg.register_external_func(Box::new(make_external_func(move |state, args| {
        counted.fetch_add(1, Ordering::Relaxed);
        (state.base_values().unwrap::<i64>(args[0]) % 4 == 0).then_some(args[0])
    })));
    let rule = {
        let mut rb = eg.new_rule("primitive pruning", true);
        rb.set_primitive_pushdown(pushdown);
        rb.set_plan_strategy(strategy);
        let x: QueryEntry = rb.new_var(ty).into();
        let y: QueryEntry = rb.new_var(ty).into();
        let a_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let b_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(a, &[x.clone(), a_id.clone()], Some(false))
            .unwrap();
        rb.query_table(b, &[y.clone(), b_id], Some(false)).unwrap();
        // Input/output alias also checks equality on an already bound output.
        rb.query_prim(filter, &[x.clone(), x.clone()], ty).unwrap();
        rb.set(out, &[x, y, a_id]);
        rb.build()
    };
    eg.run_rules(&[rule], None).unwrap();
    let first_calls = calls.load(Ordering::Relaxed);
    assert_eq!(rows(&eg, out, 2).len(), 128);
    assert!(!eg.run_rules(&[rule], None).unwrap().changed());
    assert_eq!(calls.load(Ordering::Relaxed), first_calls);

    // Reuse the cached plan with a delta on the later, independent relation.
    let extra = eg.base_values().get(64i64);
    eg.add_term(b, &[extra]);
    eg.run_rules(&[rule], None).unwrap();
    let result = rows(&eg, out, 2);
    assert_eq!(result.len(), 130);
    (result, first_calls)
}

#[test]
fn primitive_pushdown_prunes_before_expansion() {
    for strategy in [
        PlanStrategy::Gj,
        PlanStrategy::PureSize,
        PlanStrategy::MinCover,
    ] {
        for threads in [1, 4] {
            let (late, late_calls) = pruning(false, strategy, threads);
            let (early, early_calls) = pruning(true, strategy, threads);
            assert_eq!(late, early);
            assert_eq!(late_calls, 8 * 64);
            assert_eq!(early_calls, 8);
        }
    }
}

#[test]
fn primitive_chain_probes_repeated_output_columns() {
    for threads in [1, 4] {
        let mut eg = EGraph::default().with_num_threads(threads);
        let ty = ColumnTy::Base(eg.base_values_mut().register_type::<i64>());
        let a = relation(&mut eg, "a", ty, 1);
        let b = relation(&mut eg, "b", ty, 2);
        let out = relation(&mut eg, "out", ty, 3);
        for x in 0..8i64 {
            eg.add_term(a, &[eg.base_values().get(x)]);
        }
        for z in 0..32i64 {
            // z=12 is deliberately absent from B(z,z).
            eg.add_term(
                b,
                &[
                    eg.base_values().get(z),
                    eg.base_values().get(if z == 12 { 99 } else { z }),
                ],
            );
        }
        let increment = eg.register_external_func(Box::new(make_external_func(|state, args| {
            Some(
                state
                    .base_values()
                    .get(state.base_values().unwrap::<i64>(args[0]) + 1),
            )
        })));
        let add = eg.register_external_func(Box::new(make_external_func(|state, args| {
            Some(state.base_values().get(
                state.base_values().unwrap::<i64>(args[0])
                    + state.base_values().unwrap::<i64>(args[1]),
            ))
        })));
        let rule = {
            let mut rb = eg.new_rule("computed output probes stored columns", true);
            rb.set_primitive_pushdown(true);
            let x: QueryEntry = rb.new_var(ty).into();
            let y: QueryEntry = rb.new_var(ty).into();
            let z: QueryEntry = rb.new_var(ty).into();
            let a_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
            let b_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
            rb.query_table(a, &[x.clone(), a_id.clone()], Some(false))
                .unwrap();
            rb.query_table(b, &[z.clone(), z.clone(), b_id], Some(false))
                .unwrap();
            // Register the consumer first; readiness must determine call order.
            // Repeated inputs must still be passed twice to the callback.
            rb.query_prim(add, &[y.clone(), y.clone(), z.clone()], ty)
                .unwrap();
            rb.query_prim(increment, &[x.clone(), y.clone()], ty)
                .unwrap();
            rb.set(out, &[x, y, z, a_id]);
            rb.build()
        };
        eg.run_rules(&[rule], None).unwrap();
        let expected: Vec<_> = (0..8i64)
            .filter(|x| *x != 5)
            .map(|x| vec![x, x + 1, 2 * (x + 1)])
            .collect();
        assert_eq!(rows(&eg, out, 3), expected);
    }
}

#[test]
fn primitive_only_query_handles_nullary_constants_and_failure() {
    let mut eg = EGraph::default();
    let ty = ColumnTy::Base(eg.base_values_mut().register_type::<i64>());
    let out = relation(&mut eg, "out", ty, 1);
    let constant = eg.register_external_func(Box::new(make_external_func(|state, args| {
        assert!(args.is_empty());
        Some(state.base_values().get(42i64))
    })));
    let identity = eg.register_external_func(Box::new(make_external_func(|_, args| Some(args[0]))));
    let missing = eg.register_external_func(Box::new(make_external_func(|_, _| None)));
    let forty_two = eg.base_value_constant(42i64);
    let wrong = eg.base_value_constant(99i64);
    let output_id = QueryEntry::Const {
        val: eg.fresh_id(),
        ty: ColumnTy::Id,
    };
    let good = {
        let mut rb = eg.new_rule("nullary then constants", true);
        rb.set_primitive_pushdown(true);
        let x: QueryEntry = rb.new_var(ty).into();
        rb.query_prim(constant, std::slice::from_ref(&x), ty)
            .unwrap();
        rb.query_prim(identity, &[x.clone(), forty_two.clone()], ty)
            .unwrap();
        rb.query_prim(identity, &[forty_two.clone(), x.clone()], ty)
            .unwrap();
        rb.set(out, &[x, output_id.clone()]);
        rb.build()
    };
    let failed = {
        let mut rb = eg.new_rule("partial primitive has no row", true);
        rb.set_primitive_pushdown(true);
        rb.query_prim(missing, std::slice::from_ref(&forty_two), ty)
            .unwrap();
        rb.set(out, &[wrong.clone(), output_id.clone()]);
        rb.build()
    };
    let mismatched = {
        let mut rb = eg.new_rule("constant output mismatch", true);
        rb.set_primitive_pushdown(true);
        rb.query_prim(constant, std::slice::from_ref(&wrong), ty)
            .unwrap();
        rb.set(out, &[wrong, output_id]);
        rb.build()
    };
    eg.run_rules(&[good, failed, mismatched], None).unwrap();
    assert_eq!(rows(&eg, out, 1), vec![vec![42]]);
    assert!(
        !eg.run_rules(&[good, failed, mismatched], None)
            .unwrap()
            .changed()
    );
}
