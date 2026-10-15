use super::*;
use crate::{
    offsets::{Offsets, RowId},
    table::SortedWritesTable,
    table_shortcuts::v,
};
use egglog_reports::ReportLevel;
use std::iter;

fn add_table(db: &mut Database, modulo: usize) -> TableId {
    let table = db.add_table(
        SortedWritesTable::new(
            1,
            3,
            Some(ColumnId::new(2)),
            vec![],
            Box::new(|_, _, _, _| false),
        ),
        iter::empty(),
        iter::empty(),
    );
    {
        let mut buffer = db.new_buffer(table);
        for row in 0..12 {
            buffer.stage_insert(&[v(row), v(row % modulo), v(row)]);
        }
    }
    db.merge_all();
    table
}

fn eq(col: u32, value: usize) -> Constraint {
    Constraint::EqConst {
        col: ColumnId::new(col),
        val: v(value),
    }
}

fn sparse(header: &JoinHeader) -> &Arc<crate::Subset> {
    let SharedHeaderSubset::Sparse(rows) = &header.subset else {
        panic!("expected sparse header")
    };
    rows
}

fn rows(header: &JoinHeader) -> Vec<RowId> {
    let mut rows = Vec::new();
    header.subset.as_ref().offsets(|row| rows.push(row));
    rows
}

#[test]
fn header_cache_shares_sparse_rows_across_atoms_and_keeps_exact_keys() {
    let mut db = Database::default();
    let table = add_table(&mut db, 2);
    let mut builder = db.new_rule_set();
    let constraints = [eq(1, 1)];
    let first = builder
        .reprocess_constraints(table, AtomId::new(0), &constraints)
        .unwrap();
    let second = builder
        .reprocess_constraints(table, AtomId::new(7), &constraints)
        .unwrap();
    assert_eq!(first.atom, AtomId::new(0));
    assert_eq!(second.atom, AtomId::new(7));
    assert!(Arc::ptr_eq(sparse(&first), sparse(&second)));
    assert!(Arc::ptr_eq(sparse(&first), sparse(&first.clone())));
    assert_eq!(&*first.constraints, &constraints);
    assert_eq!(
        rows(&first),
        (1..12).step_by(2).map(RowId::new).collect::<Vec<_>>()
    );

    let lower = Constraint::GeConst {
        col: ColumnId::new(2),
        val: v(4),
    };
    let forward = [eq(1, 1), lower.clone()];
    let reverse = [lower, eq(1, 1)];
    let a = builder
        .reprocess_constraints(table, AtomId::new(0), &forward)
        .unwrap();
    let b = builder
        .reprocess_constraints(table, AtomId::new(0), &reverse)
        .unwrap();
    assert_eq!(rows(&a), rows(&b));
    assert!(!Arc::ptr_eq(sparse(&a), sparse(&b)));
    let hit = builder
        .reprocess_constraints(table, AtomId::new(3), &forward)
        .unwrap();
    assert!(Arc::ptr_eq(sparse(&a), sparse(&hit)));
    // Processing pushes the sorted timestamp constraint before the indexed
    // equality. A cache hit must return that processed order, not the key.
    assert_eq!(&*hit.constraints, &reverse);
    assert_eq!(builder.header_cache[table].len(), 3);
}

#[test]
fn header_cache_isolates_tables_and_refreshes_positive_and_empty_results() {
    let mut db = Database::default();
    let left = add_table(&mut db, 2);
    let right = add_table(&mut db, 3);
    let constraints = [eq(1, 1)];
    let absent = [eq(1, 9)];
    let old = {
        let mut builder = db.new_rule_set();
        let left_header = builder
            .reprocess_constraints(left, AtomId::new(0), &constraints)
            .unwrap();
        let right_header = builder
            .reprocess_constraints(right, AtomId::new(0), &constraints)
            .unwrap();
        assert_eq!(left_header.subset.size(), 6);
        assert_eq!(right_header.subset.size(), 4);
        assert!(!Arc::ptr_eq(sparse(&left_header), sparse(&right_header)));
        for _ in 0..2 {
            assert!(
                builder
                    .reprocess_constraints(left, AtomId::new(4), &absent)
                    .is_none()
            );
        }
        assert!(
            builder.header_cache[left]
                .get(absent.as_slice())
                .unwrap()
                .is_none()
        );
        assert_eq!(builder.header_cache[left].len(), 2);
        drop(builder.build());
        left_header
    };
    {
        let mut buffer = db.new_buffer(left);
        buffer.stage_insert(&[v(12), v(1), v(12)]);
        buffer.stage_insert(&[v(13), v(9), v(13)]);
    }
    // Keeping the old header alive must not retain an index and prevent merge.
    db.merge_all();
    {
        let mut builder = db.new_rule_set();
        let fresh = builder
            .reprocess_constraints(left, AtomId::new(0), &constraints)
            .unwrap();
        assert_eq!(fresh.subset.size(), 7);
        assert_eq!(old.subset.size(), 6);
        assert!(!Arc::ptr_eq(sparse(&old), sparse(&fresh)));
        assert_eq!(
            builder
                .reprocess_constraints(left, AtomId::new(0), &absent)
                .unwrap()
                .subset
                .size(),
            1
        );
    }
    db.clear_table(left);
    let mut builder = db.new_rule_set();
    assert!(
        builder
            .reprocess_constraints(left, AtomId::new(0), &constraints)
            .is_none()
    );
    assert!(
        builder
            .reprocess_constraints(left, AtomId::new(0), &absent)
            .is_none()
    );
}

#[test]
#[should_panic(expected = "Cached plans only support constraints with a fast pushdown")]
fn header_cache_still_rejects_slow_constraints_on_empty_tables() {
    let mut db = Database::default();
    let table = add_table(&mut db, 2);
    db.clear_table(table);
    let mut builder = db.new_rule_set();
    builder.reprocess_constraints(
        table,
        AtomId::new(0),
        &[Constraint::GtConst {
            col: ColumnId::new(1),
            val: v(0),
        }],
    );
}

fn output_table(db: &mut Database, arity: usize) -> TableId {
    db.add_table(
        SortedWritesTable::new(
            arity,
            arity,
            None,
            vec![],
            Box::new(|_, old, new, _| {
                assert_eq!(old, new);
                false
            }),
        ),
        iter::empty(),
        iter::empty(),
    )
}

fn table_rows(db: &Database, table: TableId) -> Vec<Vec<crate::Value>> {
    let table = db.get_table(table);
    let mut rows = table
        .scan(table.all().as_ref())
        .iter()
        .map(|(_, row)| row.to_vec())
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[test]
fn cached_seminaive_variants_share_base_headers_and_produce_correct_rows() {
    let mut db = Database::default();
    let table = add_table(&mut db, 2);
    let output = output_table(&mut db, 1);
    let (cached, first_atom, second_atom) = {
        let mut builder = db.new_rule_set();
        let mut query = builder.new_rule();
        query.set_no_decomp(true);
        let x = query.new_var();
        let ts = query.new_var();
        let first = query
            .add_atom(table, &[x.into(), v(1).into(), ts.into()], &[])
            .unwrap();
        let second = query
            .add_atom(table, &[x.into(), v(1).into(), ts.into()], &[])
            .unwrap();
        let mut rule = query.build();
        rule.insert(output, &[x.into()]).unwrap();
        let id = rule.build_with_description("header-cache-seminaive");
        let rules = builder.build();
        let cached = rules.build_cached_plan(id);
        assert!(Arc::ptr_eq(
            sparse(&rules.plans[id].0.header()[0]),
            sparse(&cached.plan.header()[0])
        ));
        (cached, first, second)
    };
    let mut builder = db.new_rule_set();
    let first = builder
        .add_rule_from_cached_plan(
            &cached,
            &[(
                first_atom,
                Constraint::GeConst {
                    col: ColumnId::new(2),
                    val: v(6),
                },
            )],
        )
        .unwrap();
    let second = builder
        .add_rule_from_cached_plan(
            &cached,
            &[(
                second_atom,
                Constraint::LtConst {
                    col: ColumnId::new(2),
                    val: v(6),
                },
            )],
        )
        .unwrap();
    for _ in 0..2 {
        assert!(
            builder
                .add_rule_from_cached_plan(
                    &cached,
                    &[(
                        first_atom,
                        Constraint::GeConst {
                            col: ColumnId::new(2),
                            val: v(12)
                        }
                    )]
                )
                .is_none()
        );
    }
    let rules = builder.build();
    assert_eq!(rules.plans.len(), 2);
    assert_eq!(rules.actions.iter().count(), 2);
    let first_headers = rules.plans[first].0.header();
    let second_headers = rules.plans[second].0.header();
    assert!(matches!(
        first_headers[0].subset,
        SharedHeaderSubset::Dense(_)
    ));
    for header in first_headers[1..].iter().chain(&second_headers[1..]) {
        assert!(Arc::ptr_eq(sparse(&first_headers[1]), sparse(header)));
    }
    let report = db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
    assert_eq!(report.num_matches("header-cache-seminaive"), 6);
    assert_eq!(
        table_rows(&db, output),
        (1..12)
            .step_by(2)
            .map(|row| vec![v(row)])
            .collect::<Vec<_>>()
    );
}

#[test]
fn cached_decomposed_plans_share_headers_without_changing_results() {
    let mut db = Database::default();
    let tables: Vec<_> = (0..6).map(|_| output_table(&mut db, 3)).collect();
    let output = output_table(&mut db, 5);
    for (table, (left, right)) in
        tables
            .iter()
            .zip([(0, 10), (10, 20), (20, 0), (0, 30), (30, 40), (40, 0)])
    {
        // Merge each row separately to guarantee alternating physical groups.
        for x in 0..4 {
            db.new_buffer(*table)
                .stage_insert(&[v(left + x), v(right + x), v(x % 2)]);
            db.merge_all();
        }
    }
    let (cached, first_atom) = {
        let mut builder = db.new_rule_set();
        let mut query = builder.new_rule();
        query.set_plan_strategy(PlanStrategy::Gj);
        let vars: Vec<_> = (0..5).map(|_| query.new_var()).collect();
        let mut first = None;
        for (table, (left, right)) in
            tables
                .iter()
                .zip([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
        {
            let atom = query
                .add_atom(
                    *table,
                    &[vars[left].into(), vars[right].into(), v(1).into()],
                    &[],
                )
                .unwrap();
            first.get_or_insert(atom);
        }
        let mut rule = query.build();
        rule.insert(
            output,
            &vars.iter().map(|var| (*var).into()).collect::<Vec<_>>(),
        )
        .unwrap();
        let id = rule.build_with_description("header-cache-decomposed");
        let rules = builder.build();
        let cached = rules.build_cached_plan(id);
        assert!(matches!(cached.plan, Plan::DecomposedPlan(_)));
        (cached, first.unwrap())
    };
    let mut builder = db.new_rule_set();
    let first = builder
        .add_rule_from_cached_plan(&cached, &[(first_atom, eq(0, 1))])
        .unwrap();
    let second = builder
        .add_rule_from_cached_plan(&cached, &[(first_atom, eq(0, 3))])
        .unwrap();
    let rules = builder.build();
    for (left, right) in rules.plans[first].0.header()[1..]
        .iter()
        .zip(&rules.plans[second].0.header()[1..])
    {
        assert!(Arc::ptr_eq(sparse(left), sparse(right)));
    }
    db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
    assert_eq!(
        table_rows(&db, output),
        [1, 3]
            .into_iter()
            .map(|x| vec![v(x), v(10 + x), v(20 + x), v(30 + x), v(40 + x)])
            .collect::<Vec<_>>()
    );
}

#[test]
fn initial_atom_clones_headers_and_cached_plan_share_processed_rows() {
    let mut db = Database::default();
    let table = add_table(&mut db, 2);
    let mut builder = db.new_rule_set();
    let mut query = builder.new_rule();
    query.set_no_decomp(true);
    let x = query.new_var();
    let ts = query.new_var();
    let atom = query
        .add_atom(table, &[x.into(), v(1).into(), ts.into()], &[])
        .unwrap();
    let SharedHeaderSubset::Sparse(processed) = &query.query.atoms[atom].constraints.subset else {
        panic!("the indexed constant must produce a sparse subset")
    };
    let processed = Arc::clone(processed);
    let cloned_atom = query.query.atoms[atom].clone();
    let SharedHeaderSubset::Sparse(cloned) = &cloned_atom.constraints.subset else {
        panic!("cloning an atom must preserve the sparse representation")
    };
    assert!(Arc::ptr_eq(&processed, cloned));

    let rule = query.build().build();
    let rules = builder.build();
    let plan = &rules.plans[rule].0;
    let header = plan
        .header()
        .iter()
        .find(|header| header.atom == atom)
        .unwrap();
    assert!(Arc::ptr_eq(&processed, sparse(header)));
    let atoms = plan.atoms();
    let SharedHeaderSubset::Sparse(planned) = &atoms[atom].constraints.subset else {
        panic!("the planned atom must preserve the sparse representation")
    };
    assert!(Arc::ptr_eq(&processed, planned));
    let cached = rules.build_cached_plan(rule);
    let cached_header = cached
        .plan
        .header()
        .iter()
        .find(|header| header.atom == atom)
        .unwrap();
    assert!(Arc::ptr_eq(&processed, sparse(cached_header)));
}

// Build a small decomposition explicitly: the planner currently linearizes its
// bags and rarely emits Lookup. Each parent here depends on the child's keys,
// even though that child does not bind any of the parent's private variables.
fn lookup_chain_plan(
    db: &mut Database,
    tables: [TableId; 3],
    output: TableId,
    calls: Arc<std::sync::atomic::AtomicUsize>,
    keep_private: bool,
) -> (CachedPlan, [AtomId; 3]) {
    use crate::free_join::{
        make_external_func,
        plan::{JoinStage, JoinStages, MatId, MatScanMode, MatSpec, ScanSpec},
    };
    use smallvec::smallvec;

    let count = db.add_external_function(Box::new(make_external_func(move |_, args| {
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(args[0])
    })));
    let mut builder = db.new_rule_set();
    let mut query = builder.new_rule();
    query.set_no_decomp(true);
    let key = query.new_var();
    let private = std::array::from_fn::<_, 3, _>(|_| query.new_var());
    let atoms = std::array::from_fn(|i| {
        query
            .add_atom(
                tables[i],
                &[
                    key.into(),
                    private[i].into(),
                    Variable::placeholder().into(),
                ],
                &[],
            )
            .unwrap()
    });
    let mut rule = query.build();
    // All tuples for a key call the function with identical arguments. Sharing
    // materializations must not suppress these independent action invocations.
    rule.call_external(count, &[key.into()]).unwrap();
    let mut output_vars = vec![key.into()];
    if keep_private {
        output_vars.extend(private.iter().copied().map(QueryEntry::from));
    }
    rule.insert(output, &output_vars).unwrap();
    let id = rule.build_with_description("lookup-chain");
    let rules = builder.build();
    let mut cached = rules.build_cached_plan(id);
    let blocks = (0..3)
        .map(|i| {
            let mut instrs = vec![JoinStage::FusedIntersect {
                cover: ScanSpec {
                    to_index: SubAtom {
                        atom: atoms[i],
                        vars: smallvec![ColumnId::new(0), ColumnId::new(1)],
                    },
                    constraints: vec![],
                },
                bind: smallvec![(ColumnId::new(0), key), (ColumnId::new(1), private[i])],
                to_intersect: vec![],
            }];
            if i > 0 {
                instrs.push(JoinStage::FusedIntersectMat {
                    cover: MatId::from_usize(i - 1),
                    mode: MatScanMode::Lookup(smallvec![key]),
                    bind: smallvec![],
                    to_intersect: vec![],
                });
            }
            (
                JoinStages::new(instrs),
                MatSpec {
                    msg_vars: smallvec![key],
                    val_vars: smallvec![private[i]],
                },
            )
        })
        .collect();
    let mut bind = smallvec![(ColumnId::new(0), key)];
    if keep_private {
        bind.push((ColumnId::new(1), private[2]));
    }
    let mut result = vec![JoinStage::FusedIntersectMat {
        cover: MatId::from_usize(2),
        mode: MatScanMode::Full,
        bind,
        to_intersect: vec![],
    }];
    if keep_private {
        for i in (0..2).rev() {
            result.push(JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(i),
                mode: MatScanMode::Value(smallvec![key]),
                bind: smallvec![(ColumnId::new(0), private[i])],
                to_intersect: vec![],
            });
        }
    }
    cached.plan = Plan::DecomposedPlan(DecomposedPlan {
        atoms: cached.plan.atoms(),
        header: cached.plan.header().to_vec(),
        stages: JoinStageBlocks { blocks },
        result_block: JoinStages::new(result),
        actions: cached.plan.actions(),
    });
    (cached, atoms)
}

fn timestamped_input(db: &mut Database, rows: &[[usize; 3]]) -> TableId {
    let table = db.add_table(
        SortedWritesTable::new(
            2,
            3,
            Some(ColumnId::new(2)),
            vec![],
            Box::new(|_, _, _, _| false),
        ),
        iter::empty(),
        iter::empty(),
    );
    // Hash shards may drain in a different order. As in the frontend, merge
    // one timestamp at a time so the sorted-column contract holds in parallel.
    for batch in rows.chunk_by(|left, right| left[2] == right[2]) {
        {
            let mut buffer = db.new_buffer(table);
            for row in batch {
                buffer.stage_insert(&row.map(v));
            }
        }
        db.merge_all();
    }
    table
}

fn timestamp_constraint(atom: AtomId, cutoff: usize, delta: bool) -> (AtomId, Constraint) {
    let col = ColumnId::new(2);
    let val = v(cutoff);
    (
        atom,
        if delta {
            Constraint::GeConst { col, val }
        } else {
            Constraint::LtConst { col, val }
        },
    )
}

#[test]
fn decomposed_seminaive_lookup_chain_preserves_private_rows_and_action_multiplicity() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    for workers in [1, 4] {
        egglog_concurrency::ThreadPool::new(workers).install(|| {
            let mut db = Database::default();
            let mut inputs = [
                vec![[0, 10, 0], [1, 110, 0], [0, 11, 1]],
                vec![[0, 20, 0], [1, 120, 0], [0, 21, 1], [1, 121, 1]],
                vec![[0, 30, 0], [1, 130, 0], [0, 31, 1], [1, 131, 1]],
            ];
            let tables = inputs
                .each_ref()
                .map(|rows| timestamped_input(&mut db, rows));
            let output = output_table(&mut db, 4);
            if workers > 1 {
                // Cross the database parallelism threshold without enlarging
                // the join's expected result or changing its dependency shape.
                let padding = output_table(&mut db, 1);
                for value in 0..10_001 {
                    db.new_buffer(padding).stage_insert(&[v(value)]);
                }
                db.merge_all();
            }
            let calls = Arc::new(AtomicUsize::new(0));
            let (cached, atoms) = lookup_chain_plan(&mut db, tables, output, calls.clone(), true);
            for cutoff in [1, 2] {
                if cutoff == 2 {
                    db.clear_table(output);
                    for i in 0..3 {
                        for key in 0..2 {
                            let row = [key, (i + 1) * 10 + key * 100 + 2, 2];
                            inputs[i].push(row);
                            db.new_buffer(tables[i]).stage_insert(&row.map(v));
                        }
                    }
                    db.merge_all();
                    calls.store(0, Ordering::Relaxed);
                }
                let mut expected = Vec::new();
                for a in &inputs[0] {
                    for b in &inputs[1] {
                        for c in &inputs[2] {
                            if a[0] == b[0]
                                && b[0] == c[0]
                                && [a[2], b[2], c[2]].into_iter().any(|ts| ts >= cutoff)
                            {
                                expected.push(vec![v(a[0]), v(a[1]), v(b[1]), v(c[1])]);
                            }
                        }
                    }
                }
                expected.sort();
                let mut builder = db.new_rule_set();
                for focus in 0..3 {
                    let mut constraints = vec![timestamp_constraint(atoms[focus], cutoff, true)];
                    constraints.extend(
                        atoms[..focus]
                            .iter()
                            .map(|atom| timestamp_constraint(*atom, cutoff, false)),
                    );
                    builder
                        .add_rule_from_cached_plan(&cached, &constraints)
                        .unwrap();
                }
                let rules = builder.build();
                let report = db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
                assert_eq!(report.num_matches("lookup-chain"), expected.len());
                assert_eq!(calls.load(Ordering::Relaxed), expected.len());
                assert_eq!(table_rows(&db, output), expected);
            }
        });
    }
}

#[test]
fn decomposed_lookup_dependencies_distinguish_empty_and_nonempty_children() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut db = Database::default();
    let tables = [
        timestamped_input(&mut db, &[[0, 10, 0], [1, 110, 1]]),
        timestamped_input(&mut db, &[[0, 20, 0]]),
        timestamped_input(&mut db, &[[0, 30, 0]]),
    ];
    let output = output_table(&mut db, 1);
    let calls = Arc::new(AtomicUsize::new(0));
    let (cached, atoms) = lookup_chain_plan(&mut db, tables, output, calls.clone(), false);
    for include_other_key in [false, true] {
        if include_other_key {
            db.new_buffer(tables[1]).stage_insert(&[v(1), v(120), v(2)]);
            db.new_buffer(tables[2]).stage_insert(&[v(1), v(130), v(2)]);
            db.merge_all();
        }
        let expected = if include_other_key {
            vec![vec![v(0)], vec![v(1)]]
        } else {
            vec![vec![v(0)]]
        };
        for order in [[true, false], [false, true]] {
            db.clear_table(output);
            calls.store(0, Ordering::Relaxed);
            let mut builder = db.new_rule_set();
            for delta in order {
                // Both child headers are nonempty. Initially only the old key
                // joins the middle bag; later both join but yield different keys.
                // Both parent atoms have identical headers in the two variants,
                // so the outer bag must include the transitive child dependency.
                builder
                    .add_rule_from_cached_plan(&cached, &[timestamp_constraint(atoms[0], 1, delta)])
                    .unwrap();
            }
            let rules = builder.build();
            let report = db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
            assert_eq!(report.num_matches("lookup-chain"), expected.len());
            assert_eq!(calls.load(Ordering::Relaxed), expected.len());
            assert_eq!(table_rows(&db, output), expected);
        }
    }
}
