use std::{
    iter,
    ops::Range,
    sync::{Arc, Mutex},
};

use egglog_reports::ReportLevel;

use crate::numeric_id::NumericId;

use crate::{
    PlanStrategy,
    action::WriteVal,
    common::Value,
    free_join::{
        CounterId, Database, TableId,
        plan::{JoinStage, Plan},
    },
    make_external_func,
    query::RuleSetBuilder,
    table::SortedWritesTable,
    table_shortcuts::v,
    table_spec::{ColumnId, Constraint},
    uf::DisplacedTable,
};

/// On MacOs the system allocator is vulenrable to contention, causing tests to execute quite
/// slowly without mimalloc.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Run a test closure both single-threaded and with 4 threads.
fn run_serial_and_parallel(f: impl Fn() + Send + Sync) {
    for num_threads in [1, 32] {
        if num_threads == 1 {
            f();
        } else {
            let pool = egglog_concurrency::ThreadPool::new(num_threads);
            pool.install(&f);
        }
    }
}

#[test]
fn basic_query() {
    run_serial_and_parallel(basic_query_inner);
}

fn basic_query_inner() {
    let MathEgraph {
        num,
        add,
        id_counter,
        mut db,
        ..
    } = basic_math_egraph();

    db.base_values_mut().register_type::<i64>();
    let add_int = db.add_external_function(Box::new(make_external_func(|exec_state, args| {
        let [x, y] = args else { panic!() };
        let x: i64 = exec_state.base_values().unwrap(*x);
        let y: i64 = exec_state.base_values().unwrap(*y);
        let z: i64 = x + y;
        Some(exec_state.base_values().get(z))
    })));

    // Add the numbers 1 through 10 to the num table at timestamp 0.
    let mut ids = Vec::new();
    {
        let mut num_buf = db.new_buffer(num);
        for i in 0..10 {
            let id = db.inc_counter(id_counter);
            let i = db.base_values().get::<i64>(i as i64);
            ids.push(i);
            num_buf.stage_insert(&[i, Value::from_usize(id), Value::new(0)]);
        }
    } // num_buf flushed

    db.merge_all();

    let mut add_ids = Vec::new();
    {
        let mut add_buf = db.new_buffer(add);
        for i in ids.chunks(2) {
            let &[x, y] = i else { unreachable!() };
            // Insert (add x y) into the database with a fresh id at timestamp 0
            let id = Value::from_usize(db.inc_counter(id_counter));
            add_ids.push(id);
            add_buf.stage_insert(&[x, y, id, Value::new(0)]);
        }
    } // add_buf flushed

    db.merge_all();

    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut add_query = rsb.new_rule();
    // Add(x, y, z, t1),
    // Num(a, x, t2),
    // Num(b, y, t3),
    // =>
    // Num(+ a b, z, 1)
    let x = add_query.new_var_named("x");
    let y = add_query.new_var_named("y");
    let z = add_query.new_var_named("z");
    let t1 = add_query.new_var_named("t1");
    let t2 = add_query.new_var_named("t2");
    let t3 = add_query.new_var_named("t3");
    let a = add_query.new_var_named("a");
    let b = add_query.new_var_named("b");

    add_query
        .add_atom(add, &[x.into(), y.into(), z.into(), t1.into()], &[])
        .unwrap();
    add_query
        .add_atom(num, &[a.into(), x.into(), t2.into()], &[])
        .unwrap();
    add_query
        .add_atom(num, &[b.into(), y.into(), t3.into()], &[])
        .unwrap();
    let mut rules = add_query.build();
    let add_a_b = rules.call_external(add_int, &[a.into(), b.into()]).unwrap();
    rules
        .insert(num, &[add_a_b.into(), z.into(), Value::new(1).into()])
        .unwrap();
    rules.build_with_description("add");
    let rule_set = rsb.build();

    let report = db.run_rule_set(&rule_set, ReportLevel::TimeOnly);

    assert!(report.changed, "{report:?}");
    assert_eq!(report.num_matches("add"), 5, "{report:?}");
    let num_table = db.get_table(num);
    let all_num = num_table.all();
    let items = num_table.scan(all_num.as_ref());
    let mut res = Vec::from_iter(
        items
            .iter()
            .map(|(_, row)| db.base_values().unwrap::<i64>(row[0])),
    );
    res.sort();
    assert_eq!(res, Vec::from_iter((0..10).chain([13, 17].into_iter())));
}

#[test]
fn line_graph_1_fj_puresize() {
    run_serial_and_parallel(|| line_graph_1_test(PlanStrategy::PureSize));
}

#[test]
fn line_graph_1_fj_mincover() {
    run_serial_and_parallel(|| line_graph_1_test(PlanStrategy::MinCover));
}

#[test]
fn line_graph_1_gj() {
    run_serial_and_parallel(|| line_graph_1_test(PlanStrategy::Gj));
}

fn line_graph_1_test(strat: PlanStrategy) {
    let mut db = Database::default();
    let edge_impl = SortedWritesTable::new(
        2,
        2,
        None,
        vec![],
        Box::new(move |_, a, b, _| {
            if a != b {
                panic!("merge not supported")
            } else {
                false
            }
        }),
    );
    let edges = db.add_table(edge_impl, iter::empty(), iter::empty());
    let nodes = Vec::from_iter((0..10).map(Value::new));
    {
        let mut edge_buf = db.new_buffer(edges);
        for edge in nodes.windows(2) {
            edge_buf.stage_insert(edge);
        }
    }
    db.merge_all();

    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut query = rsb.new_rule();
    query.set_plan_strategy(strat);
    // edge(x, y), edge(y, z) => edge(x, z)
    let x = query.new_var_named("x");
    let y = query.new_var_named("y");
    let z = query.new_var_named("z");
    query.add_atom(edges, &[x.into(), y.into()], &[]).unwrap();
    query.add_atom(edges, &[y.into(), z.into()], &[]).unwrap();
    let mut rule = query.build();
    rule.insert(edges, &[x.into(), z.into()]).unwrap();
    rule.build();
    let rule_set = rsb.build();

    assert!(db.run_rule_set(&rule_set, ReportLevel::TimeOnly).changed);

    let mut expected = Vec::from_iter(
        nodes
            .windows(2)
            .map(|x| vec![x[0], x[1]])
            .chain(nodes.windows(3).map(|x| vec![x[0], x[2]])),
    );
    expected.sort();

    let edges_table = db.get_table(edges);
    let all = edges_table.all();
    let vals = edges_table.scan(all.as_ref());
    let mut got = Vec::from_iter(vals.iter().map(|(_, row)| row.to_vec()));
    got.sort();
    assert_eq!(expected, got);
}

#[test]
fn prepared_plan_indexes_refresh_across_runs_and_clear() {
    let mut db = Database::default();
    let make_table = |arity| {
        SortedWritesTable::new(
            arity,
            arity,
            None,
            vec![],
            Box::new(|_, old, new, _| {
                assert_eq!(old, new, "test tables have unique keys");
                false
            }),
        )
    };
    let driver = db.add_table(make_table(2), iter::empty(), iter::empty());
    let target = db.add_table(make_table(3), iter::empty(), iter::empty());
    let allowed = db.add_table(make_table(1), iter::empty(), iter::empty());
    let output = db.add_table(make_table(1), iter::empty(), iter::empty());

    const KEYS: usize = 4;
    const VALUES_PER_KEY: usize = 32;
    let populate_inputs = |db: &Database, value_base: usize| {
        {
            let mut buf = db.new_buffer(target);
            for x in 0..KEYS {
                for z in 0..VALUES_PER_KEY {
                    buf.stage_insert(&[v(x), v(10_000 + x), v(value_base + x * 100 + z)]);
                }
            }
        }
        {
            let mut buf = db.new_buffer(allowed);
            for x in 0..KEYS {
                for z in 0..VALUES_PER_KEY {
                    buf.stage_insert(&[v(value_base + x * 100 + z)]);
                }
            }
        }
    };

    {
        let mut buf = db.new_buffer(driver);
        for x in 0..KEYS {
            buf.stage_insert(&[v(x), v(10_000 + x)]);
        }
    }
    populate_inputs(&db, 20_000);
    db.merge_all();

    let mut rsb = db.new_rule_set();
    let mut query = rsb.new_rule();
    query.set_plan_strategy(PlanStrategy::PureSize);
    query.set_no_decomp(true);
    let x = query.new_var_named("x");
    let y = query.new_var_named("y");
    let z = query.new_var_named("z");
    query.add_atom(driver, &[x.into(), y.into()], &[]).unwrap();
    query
        .add_atom(target, &[x.into(), y.into(), z.into()], &[])
        .unwrap();
    query.add_atom(allowed, &[z.into()], &[]).unwrap();
    let mut rule = query.build();
    rule.insert(output, &[z.into()]).unwrap();
    rule.build_with_description("prepared-index-lifecycle");
    let rules = rsb.build();

    let (plan, _, _) = rules.plans.values().next().unwrap();
    let Plan::SinglePlan(plan) = plan else {
        panic!("set_no_decomp must produce a single plan")
    };
    assert!(
        plan.stages.instrs.iter().any(|stage| matches!(
            stage,
            JoinStage::FusedIntersect { to_intersect, .. }
                if to_intersect
                    .iter()
                    .any(|(scan, _)| scan.to_index.vars.len() > 1)
        )),
        "test must exercise a prepared tuple index"
    );
    assert!(
        plan.stages
            .instrs
            .iter()
            .any(|stage| matches!(stage, JoinStage::Intersect { scans, .. } if !scans.is_empty())),
        "test must exercise a prepared column index"
    );

    let planned_tuple_lookups = plan
        .stages
        .instrs
        .iter()
        .map(|stage| match stage {
            JoinStage::Intersect { .. } => 0,
            JoinStage::FusedIntersect { to_intersect, .. }
            | JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect
                .iter()
                .filter(|(scan, _)| {
                    plan.atoms[scan.to_index.atom].table == target && scan.to_index.vars.len() != 1
                })
                .count(),
        })
        .sum::<usize>();
    let planned_allowed_column_lookups = plan
        .stages
        .instrs
        .iter()
        .map(|stage| match stage {
            JoinStage::Intersect { scans, .. } => scans
                .iter()
                .filter(|scan| plan.atoms[scan.atom].table == allowed)
                .count(),
            JoinStage::FusedIntersect { to_intersect, .. }
            | JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect
                .iter()
                .filter(|(scan, _)| {
                    plan.atoms[scan.to_index.atom].table == allowed && scan.to_index.vars.len() == 1
                })
                .count(),
        })
        .sum::<usize>();
    assert!(planned_tuple_lookups > 0);
    assert!(planned_allowed_column_lookups > 0);

    let expected = KEYS * VALUES_PER_KEY;
    let tuple_lookups_before = db.get_table_info(target).indexes.get_or_insert_calls();
    let allowed_column_lookups_before = db
        .get_table_info(allowed)
        .column_indexes
        .get_or_insert_calls();
    let first = db.run_rule_set(&rules, ReportLevel::TimeOnly);
    assert_eq!(first.num_matches("prepared-index-lifecycle"), expected);
    assert_eq!(db.get_table(output).len(), expected);
    assert_eq!(
        db.get_table_info(target).indexes.get_or_insert_calls() - tuple_lookups_before,
        planned_tuple_lookups,
        "tuple-index catalog access must be plan-static, not recursion-static"
    );
    assert_eq!(
        db.get_table_info(allowed)
            .column_indexes
            .get_or_insert_calls()
            - allowed_column_lookups_before,
        planned_allowed_column_lookups,
        "column-index catalog access must be plan-static, not recursion-static"
    );

    // Reusing the same logical plan must not retain catalog Arcs across the
    // merge at the end of the previous run.
    let second = db.run_rule_set(&rules, ReportLevel::TimeOnly);
    assert_eq!(second.num_matches("prepared-index-lifecycle"), expected);
    assert_eq!(db.get_table(output).len(), expected);

    db.clear_table(target);
    db.clear_table(allowed);
    db.clear_table(output);
    populate_inputs(&db, 40_000);
    db.merge_all();

    // Clearing bumps the table generation, so every prepared global index must
    // be refreshed before its borrowed reference is exposed to execution.
    let after_clear = db.run_rule_set(&rules, ReportLevel::TimeOnly);
    assert_eq!(
        after_clear.num_matches("prepared-index-lifecycle"),
        expected
    );
    let output_table = db.get_table(output);
    let output_rows = output_table.scan(output_table.all().as_ref());
    assert_eq!(output_rows.len(), expected);
    assert!(output_rows.iter().all(|(_, row)| row[0].index() >= 40_000));
}

#[test]
fn prepared_plan_indexes_preserve_uncacheable_columns() {
    let mut db = Database::default();
    let displaced = db.add_table(DisplacedTable::default(), iter::empty(), iter::empty());
    let make_table = || {
        SortedWritesTable::new(
            1,
            1,
            None,
            vec![],
            Box::new(|_, old, new, _| {
                assert_eq!(old, new);
                false
            }),
        )
    };
    let representatives = db.add_table(make_table(), iter::empty(), iter::empty());
    let output = db.add_table(make_table(), iter::empty(), iter::empty());
    {
        let mut buf = db.new_buffer(displaced);
        buf.stage_insert(&[v(0), v(10), v(0)]);
    }
    db.merge_all();
    let displaced_table = db.get_table(displaced);
    let displaced_rows = displaced_table.scan(displaced_table.all().as_ref());
    let displaced_row = displaced_rows.iter().next().unwrap().1;
    let displaced_key = displaced_row[0];
    let canonical = displaced_row[1];
    let displaced_timestamp = displaced_row[2];
    {
        let mut buf = db.new_buffer(representatives);
        buf.stage_insert(&[canonical]);
    }
    db.merge_all();

    let mut rsb = db.new_rule_set();
    let mut query = rsb.new_rule();
    query.set_plan_strategy(PlanStrategy::Gj);
    query.set_no_decomp(true);
    let representative = query.new_var_named("representative");
    query
        .add_atom(
            displaced,
            &[
                displaced_key.into(),
                representative.into(),
                displaced_timestamp.into(),
            ],
            &[],
        )
        .unwrap();
    query
        .add_atom(representatives, &[representative.into()], &[])
        .unwrap();
    let mut rule = query.build();
    rule.insert(output, &[representative.into()]).unwrap();
    rule.build_with_description("uncacheable-prepared-index");
    let rules = rsb.build();

    let (plan, _, _) = rules.plans.values().next().unwrap();
    let Plan::SinglePlan(plan) = plan else {
        panic!("set_no_decomp must produce a single plan")
    };
    assert!(plan.stages.instrs.iter().any(|stage| matches!(
        stage,
        JoinStage::Intersect { scans, .. }
            if scans.iter().any(|scan| {
                plan.atoms[scan.atom].table == displaced && scan.column == ColumnId::new(1)
            })
    )));

    let catalog_calls_before = db
        .get_table_info(displaced)
        .column_indexes
        .get_or_insert_calls();
    let report = db.run_rule_set(&rules, ReportLevel::TimeOnly);
    assert_eq!(report.num_matches("uncacheable-prepared-index"), 1);
    assert_eq!(
        db.get_table_info(displaced)
            .column_indexes
            .get_or_insert_calls(),
        catalog_calls_before,
        "preparation must not create a global index for a dynamic column"
    );
    assert_eq!(db.get_table(output).len(), 1);
}

#[test]
fn gj_top_index_shards_preserve_count_and_materialization() {
    assert!(crate::parallel_heuristics::gj_top_index_sharding());
    assert_eq!(crate::parallel_heuristics::gj_local_depth(), 0);
    assert_eq!(
        crate::parallel_heuristics::gj_child_cache(),
        crate::parallel_heuristics::GjChildCache::Single
    );
    assert_eq!(
        crate::parallel_heuristics::gj_top_promotion(),
        crate::parallel_heuristics::GjTopPromotion::Current
    );
    assert_eq!(crate::parallel_heuristics::gj_min_keys_per_worker(), 16);
    gj_top_index_shards_preserve_count_and_materialization_inner();
}

fn gj_top_index_shards_preserve_count_and_materialization_inner() {
    let pool = egglog_concurrency::ThreadPool::new(4);
    pool.install(|| {
        let mut db = Database::default();
        let make_table = || {
            SortedWritesTable::new(
                2,
                2,
                None,
                vec![],
                Box::new(|_, old, new, _| {
                    assert_eq!(old, new, "test tables have unique keys");
                    false
                }),
            )
        };
        let left = db.add_table(make_table(), iter::empty(), iter::empty());
        let right = db.add_table(make_table(), iter::empty(), iter::empty());
        let output = db.add_table(
            SortedWritesTable::new(
                2,
                2,
                None,
                vec![],
                Box::new(|_, old, new, _| {
                    assert_eq!(old, new, "materialized pairs are unique");
                    false
                }),
            ),
            iter::empty(),
            iter::empty(),
        );

        const KEYS: usize = 513;
        const DEGREE: usize = 17;
        {
            let mut buf = db.new_buffer(left);
            for x in 0..KEYS {
                for y in 0..DEGREE {
                    buf.stage_insert(&[v(x), v(100_000 + x * DEGREE + y)]);
                }
            }
        }
        {
            let mut buf = db.new_buffer(right);
            for x in 0..KEYS {
                for z in 0..DEGREE {
                    buf.stage_insert(&[v(x), v(200_000 + x * DEGREE + z)]);
                }
            }
        }
        db.merge_all();

        let mut rsb = RuleSetBuilder::new(&mut db);
        let mut query = rsb.new_rule();
        query.set_plan_strategy(PlanStrategy::Gj);
        let x = query.new_var_named("x");
        let y = query.new_var_named("y");
        let z = query.new_var_named("z");
        query.add_atom(left, &[x.into(), y.into()], &[]).unwrap();
        query.add_atom(right, &[x.into(), z.into()], &[]).unwrap();
        let mut rule = query.build();
        rule.insert(output, &[y.into(), z.into()]).unwrap();
        rule.build_with_description("sharded-gj");
        let rules = rsb.build();

        pool.reset_scheduler_metrics();
        let report = db.run_rule_set(&rules, ReportLevel::TimeOnly);
        let scheduler = pool.scheduler_metrics();
        // The root rule job plus one job for each of the cached index's
        // 2 * worker-count shards.  Merge work may add more global jobs.
        assert!(
            scheduler.global_pushes >= 9,
            "top index sharding did not activate: {scheduler:?}"
        );
        assert_eq!(
            scheduler.local_pushes, 0,
            "the fixed scheduler policy should keep each shard subtree serial"
        );
        let expected = KEYS * DEGREE * DEGREE;
        assert_eq!(report.num_matches("sharded-gj"), expected);

        let table = db.get_table(output);
        let materialized = table.scan(table.all().as_ref());
        assert_eq!(materialized.len(), expected);
        for (_, row) in materialized.iter() {
            let y = row[0].index() - 100_000;
            let z = row[1].index() - 200_000;
            assert_eq!(y / DEGREE, z / DEGREE);
        }
    });
}

#[test]
fn gj_small_top_fallback_uses_local_queue() {
    let pool = egglog_concurrency::ThreadPool::new(4);
    pool.install(|| {
        let mut db = Database::default();
        let make_table = || {
            SortedWritesTable::new(
                3,
                3,
                None,
                vec![],
                Box::new(|_, old, new, _| {
                    assert_eq!(old, new, "test tables have unique keys");
                    false
                }),
            )
        };
        let left = db.add_table(make_table(), iter::empty(), iter::empty());
        let right = db.add_table(make_table(), iter::empty(), iter::empty());
        let output = db.add_table(make_table(), iter::empty(), iter::empty());

        // The sorted top variable has only two keys and is therefore too small
        // for coarse index-shard partitioning, while the database is large
        // enough to enable parallel rule execution.
        const ROWS: usize = 8192;
        for table in [left, right] {
            let mut buf = db.new_buffer(table);
            for i in 0..ROWS {
                buf.stage_insert(&[v(i % 2), v(10_000 + i), v(20_000 + i)]);
            }
        }
        db.merge_all();

        let mut rsb = RuleSetBuilder::new(&mut db);
        let mut query = rsb.new_rule();
        query.set_plan_strategy(PlanStrategy::Gj);
        query.set_no_decomp(true);
        let x = query.new_var_named("x");
        let y = query.new_var_named("y");
        let z = query.new_var_named("z");
        query
            .add_atom(left, &[x.into(), y.into(), z.into()], &[])
            .unwrap();
        query
            .add_atom(right, &[x.into(), y.into(), z.into()], &[])
            .unwrap();
        let mut rule = query.build();
        rule.insert(output, &[x.into(), y.into(), z.into()])
            .unwrap();
        rule.build_with_description("local-fallback-gj");
        let rules = rsb.build();

        pool.reset_scheduler_metrics();
        let report = db.run_rule_set(&rules, ReportLevel::TimeOnly);
        let scheduler = pool.scheduler_metrics();
        assert!(
            scheduler.local_pushes > 0,
            "small-top fallback did not use worker-local tasks: {scheduler:?}"
        );
        assert_eq!(
            scheduler.local_pushes,
            scheduler.local_pops + scheduler.donated_jobs,
            "every local task must run on its owner or be donated"
        );
        assert_eq!(report.num_matches("local-fallback-gj"), ROWS);

        let table = db.get_table(output);
        let materialized = table.scan(table.all().as_ref());
        assert_eq!(materialized.len(), ROWS);
        for (_, row) in materialized.iter() {
            let i = row[2].index() - 20_000;
            assert_eq!(row[0].index(), i % 2);
            assert_eq!(row[1].index() - 10_000, i);
        }
    });
}

#[test]
fn gj_decomposed_small_top_materialization_uses_local_queue() {
    let pool = egglog_concurrency::ThreadPool::new(4);
    pool.install(|| {
        let mut db = Database::default();
        let make_table = |arity| {
            SortedWritesTable::new(
                arity,
                arity,
                None,
                vec![],
                Box::new(|_, old, new, _| {
                    assert_eq!(old, new, "test tables have unique keys");
                    false
                }),
            )
        };
        let left_xa = db.add_table(make_table(2), iter::empty(), iter::empty());
        let left_ab = db.add_table(make_table(2), iter::empty(), iter::empty());
        let left_bx = db.add_table(make_table(2), iter::empty(), iter::empty());
        let right_xc = db.add_table(make_table(2), iter::empty(), iter::empty());
        let right_cd = db.add_table(make_table(2), iter::empty(), iter::empty());
        let right_dx = db.add_table(make_table(2), iter::empty(), iter::empty());
        let output = db.add_table(make_table(5), iter::empty(), iter::empty());

        // Two cyclic bags joined through x force a decomposed plan. The left
        // bag is large, but x has only two keys, so its materialization block
        // must use recursive fallback work instead of top-index sharding.
        const X_KEYS: usize = 2;
        const LEFT_ROWS_PER_KEY: usize = 4096;
        for x in 0..X_KEYS {
            let mut xa = db.new_buffer(left_xa);
            let mut ab = db.new_buffer(left_ab);
            let mut bx = db.new_buffer(left_bx);
            for i in 0..LEFT_ROWS_PER_KEY {
                let a = 10_000 + x * LEFT_ROWS_PER_KEY + i;
                let b = 20_000 + x * LEFT_ROWS_PER_KEY + i;
                xa.stage_insert(&[v(x), v(a)]);
                ab.stage_insert(&[v(a), v(b)]);
                bx.stage_insert(&[v(b), v(x)]);
            }
        }
        {
            let mut xc = db.new_buffer(right_xc);
            let mut cd = db.new_buffer(right_cd);
            let mut dx = db.new_buffer(right_dx);
            for x in 0..X_KEYS {
                let c = 100_000 + x;
                let d = 200_000 + x;
                xc.stage_insert(&[v(x), v(c)]);
                cd.stage_insert(&[v(c), v(d)]);
                dx.stage_insert(&[v(d), v(x)]);
            }
        }
        db.merge_all();

        let mut rsb = RuleSetBuilder::new(&mut db);
        let mut query = rsb.new_rule();
        query.set_plan_strategy(PlanStrategy::Gj);
        let x = query.new_var_named("x");
        let a = query.new_var_named("a");
        let b = query.new_var_named("b");
        let c = query.new_var_named("c");
        let d = query.new_var_named("d");
        query.add_atom(left_xa, &[x.into(), a.into()], &[]).unwrap();
        query.add_atom(left_ab, &[a.into(), b.into()], &[]).unwrap();
        query.add_atom(left_bx, &[b.into(), x.into()], &[]).unwrap();
        query
            .add_atom(right_xc, &[x.into(), c.into()], &[])
            .unwrap();
        query
            .add_atom(right_cd, &[c.into(), d.into()], &[])
            .unwrap();
        query
            .add_atom(right_dx, &[d.into(), x.into()], &[])
            .unwrap();
        let mut rule = query.build();
        rule.insert(output, &[x.into(), a.into(), b.into(), c.into(), d.into()])
            .unwrap();
        rule.build_with_description("local-materialization-fallback-gj");
        let rules = rsb.build();

        let (plan, _, _) = rules.plans.values().next().unwrap();
        let Plan::DecomposedPlan(plan) = plan else {
            panic!("the two cyclic bags must produce a decomposed plan")
        };
        assert!(plan.stages.blocks.len() >= 2);
        let first_materialization = &plan.stages.blocks[0].0;
        assert!(first_materialization.instrs.len() >= 3);
        let Some(JoinStage::Intersect { var, scans }) = first_materialization.instrs.first() else {
            panic!("the first materialization block must start by intersecting x")
        };
        assert_eq!(*var, x);
        assert!(
            scans.iter().any(|scan| {
                let table = plan.atoms[scan.atom].table;
                table == right_xc || table == right_dx
            }),
            "the top x intersection must include a two-row index, making coarse sharding ineligible"
        );

        pool.reset_scheduler_metrics();
        let report = db.run_rule_set(&rules, ReportLevel::TimeOnly);
        let scheduler = pool.scheduler_metrics();
        assert!(
            scheduler.local_pushes > 0,
            "small-top materialization fallback did not use worker-local tasks: {scheduler:?}"
        );
        assert_eq!(
            scheduler.local_pushes,
            scheduler.local_pops + scheduler.donated_jobs,
            "every local task must run on its owner or be donated"
        );

        let expected = X_KEYS * LEFT_ROWS_PER_KEY;
        assert_eq!(
            report.num_matches("local-materialization-fallback-gj"),
            expected
        );
        let table = db.get_table(output);
        let materialized = table.scan(table.all().as_ref());
        assert_eq!(materialized.len(), expected);
        for (_, row) in materialized.iter() {
            let x = row[0].index();
            let i = row[1].index() - 10_000 - x * LEFT_ROWS_PER_KEY;
            assert!(x < X_KEYS);
            assert!(i < LEFT_ROWS_PER_KEY);
            assert_eq!(row[2].index(), 20_000 + x * LEFT_ROWS_PER_KEY + i);
            assert_eq!(row[3].index(), 100_000 + x);
            assert_eq!(row[4].index(), 200_000 + x);
        }
    });
}

#[test]
fn line_graph_2_fj_puresize() {
    run_serial_and_parallel(|| line_graph_2_test(PlanStrategy::PureSize));
}

#[test]
fn line_graph_2_fj_mincover() {
    run_serial_and_parallel(|| line_graph_2_test(PlanStrategy::MinCover));
}

#[test]
fn line_graph_2_gj() {
    run_serial_and_parallel(|| line_graph_2_test(PlanStrategy::Gj));
}

fn line_graph_2_test(strat: PlanStrategy) {
    let mut db = Database::default();
    let edge_impl = SortedWritesTable::new(
        2,
        2,
        None,
        vec![],
        Box::new(move |_, a, b, _| {
            if a != b {
                panic!("merge not supported")
            } else {
                false
            }
        }),
    );
    let edges = db.add_table(edge_impl, iter::empty(), iter::empty());
    let nodes = Vec::from_iter((0..10).map(Value::new));
    {
        let mut edge_buf = db.new_buffer(edges);
        for edge in nodes.windows(2) {
            edge_buf.stage_insert(edge);
        }
    }
    db.merge_all();

    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut query = rsb.new_rule();
    query.set_plan_strategy(strat);
    // edge(x, y), edge(y, z) => edge(x, z) :where y > 1
    let x = query.new_var_named("x");
    let y = query.new_var_named("y");
    let z = query.new_var_named("z");
    query
        .add_atom(
            edges,
            &[x.into(), y.into()],
            &[Constraint::GtConst {
                col: ColumnId::new(1),
                val: Value::new(1),
            }],
        )
        .unwrap();
    query.add_atom(edges, &[y.into(), z.into()], &[]).unwrap();
    let mut rule = query.build();
    rule.insert(edges, &[x.into(), z.into()]).unwrap();
    rule.build();
    let rule_set = rsb.build();

    assert!(db.run_rule_set(&rule_set, ReportLevel::TimeOnly).changed);

    let mut expected = Vec::from_iter(
        nodes.windows(2).map(|x| vec![x[0], x[1]]).chain(
            nodes
                .windows(3)
                .filter(|x| x[1] > Value::new(1))
                .map(|x| vec![x[0], x[2]]),
        ),
    );
    expected.sort();

    let edges_table = db.get_table(edges);
    let all = edges_table.all();
    let vals = edges_table.scan(all.as_ref());
    let mut got = Vec::from_iter(vals.iter().map(|(_, row)| row.to_vec()));
    got.sort();
    assert_eq!(expected, got);
}

fn intersection_test(strat: PlanStrategy) {
    let mut db = Database::default();
    let rst = (0..3).map(|_| {
        SortedWritesTable::new(
            2,
            2,
            None,
            vec![],
            Box::new(move |_, a, b, _| {
                if a != b {
                    panic!("merge not supported")
                } else {
                    false
                }
            }),
        )
    });
    let u = SortedWritesTable::new(
        1,
        1,
        None,
        vec![],
        Box::new(move |_, a, b, _| {
            if a != b {
                panic!("merge not supported")
            } else {
                false
            }
        }),
    );
    let rst_ids = rst
        .map(|r| db.add_table(r, iter::empty(), iter::empty()))
        .collect::<Vec<TableId>>();
    let u_id = db.add_table(u, iter::empty(), iter::empty());

    for rel in rst_ids.iter() {
        let mut rel_buf = db.new_buffer(*rel);
        for x in 0..10 {
            rel_buf.stage_insert(&[Value::new(x), Value::new(x)]);
        }
    }
    db.merge_all();

    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut query = rsb.new_rule();
    query.set_plan_strategy(strat);
    // R(x), S(x), T(x), x > 5 => U(X)
    let x = query.new_var_named("x");
    for rel in rst_ids.iter() {
        query
            .add_atom(
                *rel,
                &[x.into(), x.into()],
                &[Constraint::GtConst {
                    col: ColumnId::new(0),
                    val: Value::new(5),
                }],
            )
            .unwrap();
    }
    let mut rule = query.build();
    rule.insert(u_id, &[x.into()]).unwrap();
    rule.build();
    let rule_set = rsb.build();

    assert!(db.run_rule_set(&rule_set, ReportLevel::TimeOnly).changed);

    let expected = Vec::from_iter((6..10).map(|x| vec![Value::new(x)]));

    let u_table = db.get_table(u_id);
    let all = u_table.all();
    let vals = u_table.scan(all.as_ref());
    let mut got = Vec::from_iter(vals.iter().map(|(_, row)| row.to_vec()));
    got.sort();
    assert_eq!(expected, got);
}

#[test]
fn intersection_test_fj_puresize() {
    run_serial_and_parallel(|| intersection_test(PlanStrategy::PureSize));
}

#[test]
fn intersection_test_fj_mincover() {
    run_serial_and_parallel(|| intersection_test(PlanStrategy::MinCover));
}

#[test]
fn intersection_test_gj() {
    run_serial_and_parallel(|| intersection_test(PlanStrategy::Gj));
}

#[test]
fn minimal_ac() {
    run_serial_and_parallel(minimal_ac_inner);
}

fn minimal_ac_inner() {
    let MathEgraph {
        add,
        id_counter,
        mut db,
        ..
    } = basic_math_egraph();
    {
        {
            let mut add_buf = db.new_buffer(add);
            add_buf.stage_insert(&[v(0), v(0), v(1), v(0)]);
            add_buf.stage_insert(&[v(0), v(1), v(2), v(0)]);
            add_buf.stage_insert(&[v(0), v(2), v(3), v(0)]);
        }
        db.merge_all();
        {
            let mut add_buf = db.new_buffer(add);
            add_buf.stage_insert(&[v(1), v(0), v(2), v(1)]);
            add_buf.stage_insert(&[v(1), v(1), v(3), v(1)]);
        }
        db.merge_all();
    }
    let mut rsb = db.new_rule_set();
    let mut add_assoc = rsb.new_rule();
    // Add(x, Add(y, z)) => Add(Add(x, y), z)
    //
    // Add(y, z, i1, t1)
    // Add(x, i1, i2, t2)
    // =>
    // Add(x, y, <res>, cur)
    // Add(<res>, z, i2, cur)

    let x = add_assoc.new_var_named("x");
    let y = add_assoc.new_var_named("y");
    let z = add_assoc.new_var_named("z");
    let i1 = add_assoc.new_var_named("i1");
    let i2 = add_assoc.new_var_named("i2");
    let t1 = add_assoc.new_var_named("t1");
    let t2 = add_assoc.new_var_named("t2");
    add_assoc
        .add_atom(
            add,
            &[y.into(), z.into(), i1.into(), t1.into()],
            &[
                Constraint::GeConst {
                    col: ColumnId::new(3),
                    val: v(0),
                },
                Constraint::LtConst {
                    col: ColumnId::new(3),
                    val: v(1),
                },
            ],
        )
        .unwrap();
    add_assoc
        .add_atom(
            add,
            &[x.into(), i1.into(), i2.into(), t2.into()],
            &[
                Constraint::GeConst {
                    col: ColumnId::new(3),
                    val: v(1),
                },
                Constraint::LtConst {
                    col: ColumnId::new(3),
                    val: v(2),
                },
            ],
        )
        .unwrap();
    let mut rules = add_assoc.build();
    let res = rules
        .lookup_or_insert(
            add,
            &[x.into(), y.into()],
            &[
                WriteVal::IncCounter(id_counter),
                WriteVal::QueryEntry(v(2).into()),
            ],
            ColumnId::new(2),
        )
        .unwrap();
    rules
        .insert(add, &[res.into(), z.into(), i2.into(), v(2).into()])
        .unwrap();
    rules.build();
    let rule_set = rsb.build();

    db.run_rule_set(&rule_set, ReportLevel::TimeOnly);
    let add_table = db.get_table(add);
    let all_add = add_table.all();
    let items = add_table.scan(all_add.as_ref());
    let mut res = Vec::from_iter(items.iter().map(|(_, row)| row.to_vec()));
    res.sort();
    let expected = vec![
        vec![v(0), v(0), v(1), v(0)],
        vec![v(0), v(1), v(2), v(0)],
        vec![v(0), v(2), v(3), v(0)],
        vec![v(1), v(0), v(2), v(1)],
        vec![v(1), v(1), v(3), v(1)],
        vec![v(2), v(0), v(3), v(2)],
    ];
    assert_eq!(res, expected);
}

#[test]
fn ac_gj() {
    run_serial_and_parallel(|| ac_test_inner(PlanStrategy::Gj));
}

#[test]
fn ac_fj_mincover() {
    run_serial_and_parallel(|| ac_test_inner(PlanStrategy::MinCover));
}

#[test]
fn ac_fj_puresize() {
    run_serial_and_parallel(|| ac_test_inner(PlanStrategy::PureSize));
}

fn ac_test_inner(strat: PlanStrategy) {
    // This test is very involved. It reimplements major egglog features on top
    // of this library:
    // 1. rebuilding, including heuristics for incremental vs. nonincremental.
    // 2. seminaive evaluation, using sorted columns.
    // 3. iteration until saturation.
    // It does this using the classic "Assoc / Comm" workload, which is also a
    // solid benchmark for "shallow" / non-selective egglog queries.
    const N: usize = 5;
    let MathEgraph {
        num,
        add,
        id_counter,
        mut db,
        uf,
    } = basic_math_egraph();

    // Add the numbers 1 through 10 to the num table at timestamp 0.
    let mut ids = Vec::new();
    db.base_values_mut().register_type::<i64>();
    for i in 0..N {
        let id = db.inc_counter(id_counter);
        let i = db.base_values().get::<i64>(i as i64);
        ids.push(i);
        db.new_buffer(num)
            .stage_insert(&[i, Value::from_usize(id), Value::new(0)]);
    }

    db.merge_all();

    // construct (0 + ... + N), left-associated, and (N + ... + 0),
    // right-associated. With the assoc and comm rules saturated, these two
    // should be equal.
    let (left_root, right_root) = {
        let mut add_ids = Vec::new();
        let mut prev = ids[0];
        for num in &ids[1..] {
            let id = Value::from_usize(db.inc_counter(id_counter));
            db.new_buffer(add)
                .stage_insert(&[*num, prev, id, Value::new(0)]);
            prev = id;
            add_ids.push(id);
        }
        let left_root = *add_ids.last().unwrap();
        add_ids.clear();
        prev = *ids.last().unwrap();
        for num in ids[0..(N - 1)].iter().rev() {
            let id = Value::from_usize(db.inc_counter(id_counter));
            db.new_buffer(add)
                .stage_insert(&[prev, *num, id, Value::new(0)]);
            prev = id;
            add_ids.push(id);
        }
        let right_root = *add_ids.last().unwrap();
        (left_root, right_root)
    };

    db.merge_all();

    let run_ac_rule = move |db: &mut Database, recent_range: Range<Value>| {
        let old_range = Value::new(0)..recent_range.start;
        let all_range = Value::new(0)..recent_range.end;
        let next_ts = recent_range.end;
        let mut rsb = RuleSetBuilder::new(db);
        for (l_range, r_range) in [
            // NB: this could be all, recent; recent, old
            (all_range, recent_range.clone()),
            (recent_range.clone(), old_range.clone()),
        ] {
            let mut add_assoc = rsb.new_rule();
            add_assoc.set_plan_strategy(strat);
            // Add(x, Add(y, z)) => Add(Add(x, y), z)
            //
            // Add(y, z, i1, t1)
            // Add(x, i1, i2, t2)
            // =>
            // Add(x, y, <res>, cur)
            // Add(<res>, z, i2, cur)

            let x = add_assoc.new_var_named("x");
            let y = add_assoc.new_var_named("y");
            let z = add_assoc.new_var_named("z");
            let i1 = add_assoc.new_var_named("i1");
            let i2 = add_assoc.new_var_named("i2");
            let t1 = add_assoc.new_var_named("t1");
            let t2 = add_assoc.new_var_named("t2");
            add_assoc
                .add_atom(
                    add,
                    &[y.into(), z.into(), i1.into(), t1.into()],
                    &[
                        Constraint::GeConst {
                            col: ColumnId::new(3),
                            val: l_range.start,
                        },
                        Constraint::LtConst {
                            col: ColumnId::new(3),
                            val: l_range.end,
                        },
                    ],
                )
                .unwrap();
            add_assoc
                .add_atom(
                    add,
                    &[x.into(), i1.into(), i2.into(), t2.into()],
                    &[
                        Constraint::GeConst {
                            col: ColumnId::new(3),
                            val: r_range.start,
                        },
                        Constraint::LtConst {
                            col: ColumnId::new(3),
                            val: r_range.end,
                        },
                    ],
                )
                .unwrap();
            let mut rules = add_assoc.build();
            let res = rules
                .lookup_or_insert(
                    add,
                    &[x.into(), y.into()],
                    &[
                        WriteVal::IncCounter(id_counter),
                        WriteVal::QueryEntry(next_ts.into()),
                    ],
                    ColumnId::new(2),
                )
                .unwrap();
            rules
                .insert(add, &[res.into(), z.into(), i2.into(), next_ts.into()])
                .unwrap();
            rules.build();
        }

        // Add(x, y, z, t1),
        // => Add(y, x, z, cur)

        let mut add_comm = rsb.new_rule();
        add_comm.set_plan_strategy(strat);
        let x = add_comm.new_var_named("x");
        let y = add_comm.new_var_named("y");
        let z = add_comm.new_var_named("z");
        let t1 = add_comm.new_var_named("t1");
        // Just look for the current timestamp
        add_comm
            .add_atom(
                add,
                &[x.into(), y.into(), z.into(), t1.into()],
                &[Constraint::EqConst {
                    col: ColumnId::new(3),
                    val: recent_range.start,
                }],
            )
            .unwrap();

        let mut rules = add_comm.build();
        rules
            .insert(add, &[y.into(), x.into(), z.into(), next_ts.into()])
            .unwrap();
        rules.build();
        let rule_set = rsb.build();
        db.run_rule_set(&rule_set, ReportLevel::TimeOnly)
    };

    let rebuild = |db: &mut Database, cur_ts: Value| -> (Value, bool) {
        let next_ts = Value::new(cur_ts.rep() + 1);
        let mut rsb = db.new_rule_set();
        let num_rebuild = |rsb: &mut RuleSetBuilder, cur_ts: Value, next_ts: Value| {
            // num(x, id, t1), displaced(id, id2, t2)
            // =>
            // insert num(x, id2, cur) // rebuilding always picks the new value.
            // Compare the size of the num table to the displaced ids at the current timestamp:
            let num_size = rsb.estimate_size(num, None);
            let uf_size = rsb.estimate_size(
                uf,
                Some(Constraint::EqConst {
                    col: ColumnId::new(2),
                    val: cur_ts,
                }),
            );
            let mut num_rebuild = rsb.new_rule();
            num_rebuild.set_plan_strategy(strat);
            if incremental_rebuild(uf_size, num_size) {
                // nonincremental:
                //  num(x, id, t1) =>
                //  num(x, id', t1) where id' is canonical
                let x = num_rebuild.new_var_named("x");
                let id = num_rebuild.new_var_named("id");
                let t1 = num_rebuild.new_var_named("t1");
                num_rebuild
                    .add_atom(num, &[x.into(), id.into(), t1.into()], &[])
                    .unwrap();
                let mut rules = num_rebuild.build();
                let id_canon = rules
                    .lookup_with_default(uf, &[id.into()], id.into(), ColumnId::new(1))
                    .unwrap();
                rules.assert_ne(id.into(), id_canon.into()).unwrap();
                rules
                    .insert(num, &[x.into(), id_canon.into(), next_ts.into()])
                    .unwrap();
                rules.build();
            } else {
                let x = num_rebuild.new_var_named("x");
                let id = num_rebuild.new_var_named("id");
                let t1 = num_rebuild.new_var_named("t1");
                let id_new = num_rebuild.new_var_named("id_new");
                let t2 = num_rebuild.new_var_named("t2");
                num_rebuild
                    .add_atom(num, &[x.into(), id.into(), t1.into()], &[])
                    .unwrap();
                num_rebuild
                    .add_atom(
                        uf,
                        &[id.into(), id_new.into(), t2.into()],
                        &[Constraint::EqConst {
                            col: ColumnId::new(2),
                            val: cur_ts,
                        }],
                    )
                    .unwrap();
                let mut rules = num_rebuild.build();
                rules
                    .insert(num, &[x.into(), id_new.into(), next_ts.into()])
                    .unwrap();
                rules.build();
            }
        };
        num_rebuild(&mut rsb, cur_ts, next_ts);
        let mut changed = false;
        let add_size = rsb.estimate_size(add, None);
        let uf_size = rsb.estimate_size(
            uf,
            Some(Constraint::EqConst {
                col: ColumnId::new(2),
                val: cur_ts,
            }),
        );
        if incremental_rebuild(uf_size, add_size) {
            let mut add_rebuild_id = rsb.new_rule();
            add_rebuild_id.set_plan_strategy(strat);
            let x = add_rebuild_id.new_var_named("x");
            let y = add_rebuild_id.new_var_named("y");
            let id = add_rebuild_id.new_var_named("id");
            let t1 = add_rebuild_id.new_var_named("t1");
            let id_new = add_rebuild_id.new_var_named("id_new");
            let t2 = add_rebuild_id.new_var_named("t2");
            add_rebuild_id
                .add_atom(add, &[x.into(), y.into(), id.into(), t1.into()], &[])
                .unwrap();
            add_rebuild_id
                .add_atom(
                    uf,
                    &[id.into(), id_new.into(), t2.into()],
                    &[Constraint::EqConst {
                        col: ColumnId::new(2),
                        val: cur_ts,
                    }],
                )
                .unwrap();
            let mut rules = add_rebuild_id.build();
            let x_new = rules
                .lookup_with_default(uf, &[x.into()], x.into(), ColumnId::new(1))
                .unwrap();
            let y_new = rules
                .lookup_with_default(uf, &[y.into()], y.into(), ColumnId::new(1))
                .unwrap();
            rules.remove(add, &[x.into(), y.into()]).unwrap();
            rules
                .insert(
                    add,
                    &[x_new.into(), y_new.into(), id_new.into(), next_ts.into()],
                )
                .unwrap();
            rules.build();
            let rs = rsb.build();
            changed |= db.run_rule_set(&rs, ReportLevel::TimeOnly).changed;
            let mut rsb = db.new_rule_set();
            num_rebuild(&mut rsb, cur_ts, next_ts);
            let mut add_rebuild_l = rsb.new_rule();
            add_rebuild_l.set_plan_strategy(strat);
            let x = add_rebuild_l.new_var_named("x");
            let y = add_rebuild_l.new_var_named("y");
            let id = add_rebuild_l.new_var_named("id");
            let t1 = add_rebuild_l.new_var_named("t1");
            let x_new = add_rebuild_l.new_var_named("x_new");
            let t2 = add_rebuild_l.new_var_named("t2");
            add_rebuild_l
                .add_atom(add, &[x.into(), y.into(), id.into(), t1.into()], &[])
                .unwrap();
            add_rebuild_l
                .add_atom(
                    uf,
                    &[x.into(), x_new.into(), t2.into()],
                    &[Constraint::EqConst {
                        col: ColumnId::new(2),
                        val: cur_ts,
                    }],
                )
                .unwrap();
            let mut rules = add_rebuild_l.build();
            let y_new = rules
                .lookup_with_default(uf, &[y.into()], y.into(), ColumnId::new(1))
                .unwrap();
            let id_new = rules
                .lookup_with_default(uf, &[id.into()], id.into(), ColumnId::new(1))
                .unwrap();
            rules.remove(add, &[x.into(), y.into()]).unwrap();
            rules
                .insert(
                    add,
                    &[x_new.into(), y_new.into(), id_new.into(), next_ts.into()],
                )
                .unwrap();
            rules.build();

            let rs = rsb.build();
            changed |= db.run_rule_set(&rs, ReportLevel::TimeOnly).changed;
            let mut rsb = db.new_rule_set();
            num_rebuild(&mut rsb, cur_ts, next_ts);
            let mut add_rebuild_r = rsb.new_rule();
            add_rebuild_r.set_plan_strategy(strat);
            let x = add_rebuild_r.new_var_named("x");
            let y = add_rebuild_r.new_var_named("y");
            let id = add_rebuild_r.new_var_named("id");
            let t1 = add_rebuild_r.new_var_named("t1");
            let y_new = add_rebuild_r.new_var_named("y_new");
            let t2 = add_rebuild_r.new_var_named("t2");
            add_rebuild_r
                .add_atom(add, &[x.into(), y.into(), id.into(), t1.into()], &[])
                .unwrap();
            add_rebuild_r
                .add_atom(
                    uf,
                    &[y.into(), y_new.into(), t2.into()],
                    &[Constraint::EqConst {
                        col: ColumnId::new(2),
                        val: cur_ts,
                    }],
                )
                .unwrap();
            let mut rules = add_rebuild_r.build();
            let x_new = rules
                .lookup_with_default(uf, &[x.into()], x.into(), ColumnId::new(1))
                .unwrap();
            let id_new = rules
                .lookup_with_default(uf, &[id.into()], id.into(), ColumnId::new(1))
                .unwrap();
            rules.remove(add, &[x.into(), y.into()]).unwrap();
            rules
                .insert(
                    add,
                    &[x_new.into(), y_new.into(), id_new.into(), next_ts.into()],
                )
                .unwrap();
            rules.build();
            let rs = rsb.build();
            changed |= db.run_rule_set(&rs, ReportLevel::TimeOnly).changed;
        } else {
            // nonincremental. Just run one rule and recanonicalize everything.
            // add(x, y, id, t1) =>
            //   let x' = lookup_with_default(uf, x, x')
            //   let y' = lookup_with_default(uf, y, y')
            //   let id' = lookup_with_default(uf, id, id')
            //   assertanyne([x, y, id], [x', y', id'])
            //   delete add(x, y)
            //   insert add(x', y', id', cur)
            let mut rebuild = rsb.new_rule();
            rebuild.set_plan_strategy(strat);
            let x = rebuild.new_var_named("x");
            let y = rebuild.new_var_named("y");
            let id = rebuild.new_var_named("id");
            let t1 = rebuild.new_var_named("t1");
            rebuild
                .add_atom(add, &[x.into(), y.into(), id.into(), t1.into()], &[])
                .unwrap();
            let mut rules = rebuild.build();
            let x_canon = rules
                .lookup_with_default(uf, &[x.into()], x.into(), ColumnId::new(1))
                .unwrap();
            let y_canon = rules
                .lookup_with_default(uf, &[y.into()], y.into(), ColumnId::new(1))
                .unwrap();
            let id_canon = rules
                .lookup_with_default(uf, &[id.into()], id.into(), ColumnId::new(1))
                .unwrap();
            rules
                .assert_any_ne(
                    &[x.into(), y.into(), id.into()],
                    &[x_canon.into(), y_canon.into(), id_canon.into()],
                )
                .unwrap();
            rules.remove(add, &[x.into(), y.into()]).unwrap();
            rules
                .insert(
                    add,
                    &[
                        x_canon.into(),
                        y_canon.into(),
                        id_canon.into(),
                        next_ts.into(),
                    ],
                )
                .unwrap();
            rules.build();
            let rs = rsb.build();
            changed |= db.run_rule_set(&rs, ReportLevel::TimeOnly).changed;
        }
        (next_ts, changed)
    };
    let mut cur_ts = Value::new(0);
    let mut next_ts = Value::new(1);
    loop {
        if !run_ac_rule(&mut db, cur_ts..next_ts).changed {
            break;
        }
        let start = next_ts;
        let mut new_ids_at = start;
        let mut changed = true;
        while changed {
            let (next_ts, rebuild_changed) = rebuild(&mut db, new_ids_at);
            new_ids_at = next_ts;
            changed = rebuild_changed;
        }
        cur_ts = start;
        next_ts = Value::new(new_ids_at.rep() + 1);
    }
    let uf_table = db.get_table(uf);
    let l_canon = uf_table
        .get_row(&[left_root])
        .map(|row| row.vals[1])
        .unwrap_or(left_root);
    let r_canon = uf_table
        .get_row(&[right_root])
        .map(|row| row.vals[1])
        .unwrap_or(right_root);
    assert_eq!(l_canon, r_canon);
}

struct MathEgraph {
    uf: TableId,
    num: TableId,
    add: TableId,
    id_counter: CounterId,
    db: Database,
}

fn basic_math_egraph() -> MathEgraph {
    let mut db = Database::default();
    let uf = db.add_table(DisplacedTable::default(), iter::empty(), iter::empty());
    let num_impl = SortedWritesTable::new(
        1,
        3,
        Some(ColumnId::new(2)),
        vec![],
        Box::new(move |state, a, b, res| {
            if a[1] != b[1] {
                // Mark the two ids as equal. Picking b[1] as the 'presumed winner'
                state.stage_insert(uf, &[a[1], b[1], b[2]]);
                res.extend_from_slice(b);
                true
            } else {
                false
            }
        }),
    );

    let id_counter = db.add_counter();
    let num = db.add_table(num_impl, iter::once(uf), iter::empty());
    let add_impl = SortedWritesTable::new(
        2,
        4,
        Some(ColumnId::new(3)),
        vec![],
        Box::new(move |state, a, b, res| {
            // Capture a backtrace as a string
            if a[2] != b[2] {
                // Mark the two ids as equal. Picking b[2] as the 'presumed winner'
                state.stage_insert(uf, &[a[2], b[2], b[3]]);
                res.extend_from_slice(b);
                true
            } else {
                false
            }
        }),
    );

    let add = db.add_table(add_impl, iter::once(uf), iter::empty());

    MathEgraph {
        uf,
        num,
        add,
        id_counter,
        db,
    }
}

fn incremental_rebuild(uf_size: usize, table_size: usize) -> bool {
    uf_size / 4 > table_size
}

#[test]
fn lookup_with_fallback_partial_success() {
    run_serial_and_parallel(lookup_with_fallback_partial_success_inner);
}

fn lookup_with_fallback_partial_success_inner() {
    // Insert (f 1) (f 2), (g 1) (g 3) (g 4).
    // Run a query that iterates over g, binding x to 1, 3, 4.
    // Insert (h (lookup f x, with fallback assert-even))
    // Should get h 1, h 4
    let mut db = Database::default();
    let [f, g, h] = (0..3)
        .map(|_| {
            db.add_table(
                SortedWritesTable::new(
                    1,
                    2,
                    None,
                    vec![],
                    Box::new(move |_, a, b, _| {
                        if a[0] != b[0] {
                            panic!("merge not supported")
                        } else {
                            false
                        }
                    }),
                ),
                iter::empty(),
                iter::empty(),
            )
        })
        .collect::<Vec<_>>()[..]
    else {
        unreachable!()
    };

    {
        let mut buf = db.new_buffer(f);
        buf.stage_insert(&[v(1), v(0)]);
        buf.stage_insert(&[v(2), v(0)]);
    }
    {
        let mut buf = db.new_buffer(g);
        buf.stage_insert(&[v(1), v(0)]);
        buf.stage_insert(&[v(3), v(0)]);
        buf.stage_insert(&[v(4), v(0)]);
        buf.stage_insert(&[v(5), v(0)]);
    }

    db.merge_all();
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_vals = {
        let inner = log.clone();
        db.add_external_function(Box::new(make_external_func(move |_, args| {
            let [x] = args else { panic!() };
            inner.lock().unwrap().push(*x);
            Some(*x)
        })))
    };
    let assert_even = db.add_external_function(Box::new(make_external_func(|_, args| {
        let [x] = args else { panic!() };
        if x.rep().is_multiple_of(2) {
            Some(*x)
        } else {
            None
        }
    })));

    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut query = rsb.new_rule();
    let x = query.new_var_named("x");
    let y = query.new_var_named("y");
    query.add_atom(g, &[x.into(), y.into()], &[]).unwrap();
    let mut rb = query.build();
    let res = rb
        .lookup_with_fallback(f, &[x.into()], ColumnId::new(0), assert_even, &[x.into()])
        .unwrap();
    rb.call_external(log_vals, &[x.into()]).unwrap();
    rb.insert(h, &[res.into(), y.into()]).unwrap();
    rb.build();
    let rs = rsb.build();
    assert!(db.run_rule_set(&rs, ReportLevel::TimeOnly).changed);

    let h = db.get_table(h);
    let all = h.all();
    let mut h_contents = h
        .scan(all.as_ref())
        .iter()
        .map(|(_, row)| row.to_vec())
        .collect::<Vec<_>>();
    h_contents.sort();
    assert_eq!(h_contents, vec![vec![v(1), v(0)], vec![v(4), v(0)],]);
    let sorted_log = {
        let mut log = log.lock().unwrap().clone();
        log.sort();
        log
    };
    assert_eq!(sorted_log, vec![v(1), v(4)]);
}

#[test]
fn call_external_with_fallback() {
    run_serial_and_parallel(call_external_with_fallback_inner);
}

fn call_external_with_fallback_inner() {
    // Insert (f 1) (f 2) (f 3) (f 5).
    // Iterate over f, binding x to 1, 2, 3.
    // Have two external functions:
    // 1. assert_even, which returns None for odd numbers.
    // 2. inc, which increments the input value and only fails on the number 5
    // Insert (h (call assert_even x, with fallback inc x))
    // We should get h 2, h 4.
    let mut db = Database::default();
    let [f, h] = (0..2)
        .map(|_| {
            db.add_table(
                SortedWritesTable::new(
                    1,
                    2,
                    None,
                    vec![],
                    Box::new(move |_, a, b, _| {
                        if a[0] != b[0] {
                            panic!("merge not supported")
                        } else {
                            false
                        }
                    }),
                ),
                iter::empty(),
                iter::empty(),
            )
        })
        .collect::<Vec<_>>()[..]
    else {
        unreachable!()
    };

    {
        let mut buf = db.new_buffer(f);
        buf.stage_insert(&[v(1), v(0)]);
        buf.stage_insert(&[v(2), v(0)]);
        buf.stage_insert(&[v(3), v(0)]);
        buf.stage_insert(&[v(5), v(0)]);
    }
    db.merge_all();
    let assert_even = db.add_external_function(Box::new(make_external_func(|_, args| {
        let [x] = args else { panic!() };
        if x.rep().is_multiple_of(2) {
            Some(*x)
        } else {
            None
        }
    })));

    let inc = db.add_external_function(Box::new(make_external_func(|_, args| {
        let [x] = args else { panic!() };
        if x.rep() == 5 { None } else { Some(x.inc()) }
    })));

    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut query = rsb.new_rule();
    let x = query.new_var_named("x");
    let y = query.new_var_named("y");
    query.add_atom(f, &[x.into(), y.into()], &[]).unwrap();
    let mut rb = query.build();
    let res = rb
        .call_external_with_fallback(assert_even, &[x.into()], inc, &[x.into()])
        .unwrap();
    rb.insert(h, &[res.into(), y.into()]).unwrap();
    rb.build();
    let rs = rsb.build();
    assert!(db.run_rule_set(&rs, ReportLevel::TimeOnly).changed);

    let h = db.get_table(h);
    let all = h.all();
    let mut h_contents = h
        .scan(all.as_ref())
        .iter()
        .map(|(_, row)| row.to_vec())
        .collect::<Vec<_>>();
    h_contents.sort();
    assert_eq!(h_contents, vec![vec![v(2), v(0)], vec![v(4), v(0)],]);
}

#[test]
fn early_stop() {
    run_serial_and_parallel(early_stop_inner);
}

fn early_stop_inner() {
    let mut db = Database::default();

    // Create a table with 1M rows.
    let data_table = db.add_table(
        SortedWritesTable::new(1, 2, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );

    {
        // Populate with 0.5M rows.
        let mut buf = db.new_buffer(data_table);
        for i in 0..500_000 {
            buf.stage_insert(&[Value::from_usize(i), Value::from_usize(i)]);
        }
    }
    db.merge_all();

    // External function that triggers early stop after 1000 calls.
    let call_count = Arc::new(Mutex::new(0usize));
    let call_count_clone = call_count.clone();
    let stop_trigger =
        db.add_external_function(Box::new(make_external_func(move |exec_state, args| {
            let mut count = call_count_clone.lock().unwrap();
            *count += 1;

            if *count >= 1000 {
                exec_state.trigger_early_stop();
            }

            let [x] = args else { panic!() };
            Some(*x)
        })));

    // Build a rule that scans the table and calls the external function.
    let mut rsb = RuleSetBuilder::new(&mut db);
    let mut query = rsb.new_rule();
    let x = query.new_var_named("x");
    let y = query.new_var_named("y");
    query
        .add_atom(data_table, &[x.into(), y.into()], &[])
        .unwrap();
    let mut rb = query.build();
    let _ = rb.call_external(stop_trigger, &[x.into()]).unwrap();
    rb.build_with_description("early_stop_test");
    let rs = rsb.build();

    let report = db.run_rule_set(&rs, ReportLevel::TimeOnly);

    let matches = report.num_matches("early_stop_test");

    // NB: 100K is very loose: this test doesn't appear to flake even with 10K as the upper limit.
    // This is mostly just there to avoid truly unlikely race conditions where there are a huge
    // number of matches in flight at once.
    assert!(
        matches < 100_000,
        "Expected much fewer than 10k matches due to early stopping, got {}, (call_count={})",
        matches,
        call_count.lock().unwrap(),
    );
    assert!(
        matches >= 1000,
        "Expected at least 1000 matches before stopping, got {} (call_count={})",
        matches,
        call_count.lock().unwrap(),
    );

    let final_count = *call_count.lock().unwrap();
    assert!(
        final_count >= 1000,
        "External function called {final_count} times, should be at least 1000"
    );
    assert!(
        final_count < 100_000,
        "External function called {final_count} times, should be much less than 10k"
    );
}
