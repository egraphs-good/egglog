use super::*;
use crate::{
    offsets::{Offsets, RowId},
    table::SortedWritesTable,
    table_shortcuts::{TimestampedTable, v},
};
use egglog_reports::ReportLevel;
use std::iter;

fn add_table(db: &mut Database, modulo: usize) -> TableId {
    let table = db.add_table(
        SortedWritesTable::new(1, 3, ColumnId::new(2), vec![], Box::new(|_, _, _, _| false)),
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
        TimestampedTable::new(
            arity,
            arity,
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
