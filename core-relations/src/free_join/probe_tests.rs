use super::AtomRowsKind;
use super::*;

#[test]
fn arena_handle_is_created_only_on_first_packed_allocation() {
    let arena = SharedArena::new();
    let handle = LazyArenaHandle::new(&arena);
    assert!(handle.handle.get().is_none());
    handle.get();
    assert!(handle.handle.get().is_some());
}

use crate::numeric_id::NumericId;
use std::cell::Cell;

#[test]
fn sorted_multi_probe_keeps_a_monotone_cursor_across_batches() {
    let target = [1, 2, 4, 8, 9, 20, 21].map(Value::from_usize);
    let first = [0, 2, 3, 8].map(Value::from_usize);
    let second = [9, 10, 20, 22].map(Value::from_usize);
    let mut cursor = 0;
    let mut matches = Vec::new();

    for (input, key) in first.into_iter().enumerate() {
        if seek_sorted_key(key, target.len(), &mut cursor, |index| target[index]) {
            matches.push((input, cursor));
            cursor += 1;
        }
    }
    assert_eq!(matches, vec![(1, 1), (3, 3)]);
    assert_eq!(cursor, 4);

    matches.clear();
    for (input, key) in second.into_iter().enumerate() {
        if seek_sorted_key(key, target.len(), &mut cursor, |index| target[index]) {
            matches.push((input, cursor));
            cursor += 1;
        }
    }
    assert_eq!(matches, vec![(0, 4), (2, 5)]);
    assert_eq!(cursor, target.len());
}

#[test]
fn sorted_multi_probe_gallops_across_skewed_gaps() {
    let target_len = 100_000;
    let keys = [2, 199_998].map(Value::from_usize);
    let calls = Cell::new(0);
    let mut cursor = 0;
    let mut matches = Vec::new();

    for (input, key) in keys.into_iter().enumerate() {
        if seek_sorted_key(key, target_len, &mut cursor, |index| {
            calls.set(calls.get() + 1);
            Value::from_usize(index * 2)
        }) {
            matches.push((input, cursor));
            cursor += 1;
        }
    }

    assert_eq!(matches, vec![(0, 1), (1, 99_999)]);
    assert!(
        calls.get() < 100,
        "galloping should not linearly scan a large key gap"
    );
}

#[test]
fn sorted_seek_reads_an_immediate_hit_once() {
    let target = [2, 4, 6].map(Value::from_usize);
    let calls = Cell::new(0);
    let mut cursor = 0;
    assert!(seek_sorted_key(
        Value::from_usize(2),
        target.len(),
        &mut cursor,
        |index| {
            calls.set(calls.get() + 1);
            target[index]
        }
    ));
    assert_eq!(calls.get(), 1);
}

mod catalog_filter {
    use std::iter;

    use super::*;
    use crate::{
        free_join::{Database, get_column_index_from_tableinfo, get_index_from_tableinfo},
        offsets::OffsetRange,
        table::SortedWritesTable,
        table_shortcuts::v,
        table_spec::{ColumnId, Constraint},
    };

    const GROUPS: usize = 4;
    const ROWS_PER_GROUP: usize = 20;
    /// Every row of this group is removed, leaving only stale candidates.
    const DEAD_GROUP: usize = 3;

    /// A two-key table whose first column takes `GROUPS` values, each with
    /// `ROWS_PER_GROUP` rows `(g, j)`. Row `(g, 0)` of every group and all of
    /// `DEAD_GROUP` are removed after the persistent indexes are built, so the
    /// indexes keep those rows as stale candidates.
    fn stale_db() -> (Database, crate::TableId) {
        let mut db = Database::new();
        let table = db.add_table(
            SortedWritesTable::new(
                2,
                2,
                None,
                vec![],
                Box::new(|_, old, new, _| {
                    assert_eq!(old, new);
                    false
                }),
            ),
            iter::empty(),
            iter::empty(),
        );
        {
            let mut buf = db.new_buffer(table);
            for g in 0..GROUPS {
                for j in 0..ROWS_PER_GROUP {
                    buf.stage_insert(&[v(g), v(j)]);
                }
            }
        }
        db.merge_all();
        drop(get_column_index_from_tableinfo(
            db.get_table_info(table),
            ColumnId::new(0),
        ));
        drop(get_index_from_tableinfo(
            db.get_table_info(table),
            &[ColumnId::new(0), ColumnId::new(1)],
        ));
        {
            let mut buf = db.new_buffer(table);
            for g in 0..GROUPS {
                buf.stage_remove(&[v(g), v(0)]);
            }
            for j in 1..ROWS_PER_GROUP {
                buf.stage_remove(&[v(DEAD_GROUP), v(j)]);
            }
        }
        db.merge_all();
        assert!(db.get_table(table).has_stale_rows());
        assert_eq!(
            db.get_table(table).all().size(),
            GROUPS * ROWS_PER_GROUP,
            "stale rows must stay in place for this test"
        );
        (db, table)
    }

    fn rows_of(rows: &AtomRows<'_, '_>) -> Vec<RowId> {
        let mut out = Vec::new();
        crate::offsets::Offsets::offsets(&rows.subset(), |row| out.push(row));
        out
    }

    fn live_rows_matching(db: &Database, table: crate::TableId, cs: &[Constraint]) -> Vec<RowId> {
        let table = db.get_table(table);
        let subset = table.refine_ref(table.all().as_ref(), cs, true);
        let mut out = Vec::new();
        crate::offsets::Offsets::offsets(&subset, |row| out.push(row));
        out
    }

    fn column_prober<'ctx, 'rows>(
        db: &'ctx Database,
        table: crate::TableId,
        index: &'rows Index<ColumnIndex>,
        continuations: &'rows RootContinuationCache,
        constraints: &'ctx [Constraint],
        keep_rows: bool,
        child_shape: ChildShape,
    ) -> Prober<'ctx, 'rows, 'rows> {
        let wrapped = db.get_table(table);
        Prober {
            source: AtomRows::dense(OffsetRange::new(
                RowId::new(0),
                RowId::from_usize(GROUPS * ROWS_PER_GROUP),
            )),
            ix: ProbeIndex::CachedColumn {
                intersect_outer: None,
                table: index,
                continuations,
                child_shape,
                filter: CatalogFilter {
                    table: wrapped.as_ref(),
                    constraints,
                    check_live: wrapped.has_stale_rows(),
                },
            },
            keep_rows,
        }
    }

    #[test]
    fn existence_matches_need_a_live_row() {
        let (db, table) = stale_db();
        let index = get_column_index_from_tableinfo(db.get_table_info(table), ColumnId::new(0));
        let index = index.get().unwrap();
        assert!(
            index.get_subset(&v(DEAD_GROUP)).is_some(),
            "the index must still hold the dead group as a candidate"
        );
        let continuations = RootContinuationCache::default();
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &[],
            false,
            ChildShape::Leaf,
        );

        assert!(prober.get_subset(&[v(DEAD_GROUP)]).is_none());
        assert!(matches!(
            prober.get_subset(&[v(0)]),
            Some(ProbeMatch::Present)
        ));

        let mut keys = Vec::new();
        prober.for_each(|key, found| {
            assert!(matches!(found, ProbeMatch::Present));
            keys.push(key[0]);
        });
        keys.sort();
        let expected = (0..GROUPS)
            .filter(|g| *g != DEAD_GROUP)
            .map(v)
            .collect::<Vec<_>>();
        assert_eq!(keys, expected);
    }

    #[test]
    fn retained_unconstrained_matches_borrow_the_catalog_group() {
        let (db, table) = stale_db();
        let index = get_column_index_from_tableinfo(db.get_table_info(table), ColumnId::new(0));
        let index = index.get().unwrap();
        let continuations = RootContinuationCache::default();
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &[],
            true,
            ChildShape::Direct,
        );

        // Stale rows are excluded by every consumer of retained rows, so the
        // group is borrowed as is and keeps its continuation slot.
        let Some(ProbeMatch::Rows(rows)) = prober.get_subset(&[v(0)]) else {
            panic!("expected borrowed catalog rows")
        };
        let AtomRowsKind::Catalog {
            subset,
            continuation,
        } = rows.kind()
        else {
            panic!("expected borrowed catalog rows")
        };
        assert_eq!(subset.size(), ROWS_PER_GROUP);
        assert!(continuation.is_some());
    }

    #[test]
    fn constrained_matches_materialize_only_matching_live_rows() {
        let (db, table) = stale_db();
        let index = get_column_index_from_tableinfo(db.get_table_info(table), ColumnId::new(0));
        let index = index.get().unwrap();
        let continuations = RootContinuationCache::default();

        // Row (g, 0) is stale and rows with j <= 5 fail the constraint.
        let wide = [Constraint::GtConst {
            col: ColumnId::new(1),
            val: v(5),
        }];
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &wide,
            true,
            ChildShape::Direct,
        );
        let Some(ProbeMatch::Rows(rows)) = prober.get_subset(&[v(1)]) else {
            panic!("expected filtered rows")
        };
        let AtomRowsKind::Root(root) = rows.kind() else {
            panic!("a large filtered group must become a residual root")
        };
        assert!(!root.is_plan_root());
        let mut expected = live_rows_matching(
            &db,
            table,
            &[
                Constraint::EqConst {
                    col: ColumnId::new(0),
                    val: v(1),
                },
                wide[0].clone(),
            ],
        );
        expected.sort();
        assert_eq!(rows_of(&rows), expected);
        assert_eq!(rows_of(&rows).len(), ROWS_PER_GROUP - 6);
        assert!(prober.get_subset(&[v(DEAD_GROUP)]).is_none());

        // Only the removed row (g, 0) would have matched this constraint.
        let only_stale = [Constraint::EqConst {
            col: ColumnId::new(1),
            val: v(0),
        }];
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &only_stale,
            true,
            ChildShape::Direct,
        );
        assert!(prober.get_subset(&[v(0)]).is_none());
        let mut seen = 0;
        prober.for_each(|_, _| seen += 1);
        assert_eq!(seen, 0);

        // A small survivor set travels inline instead of allocating.
        let narrow = [Constraint::GtConst {
            col: ColumnId::new(1),
            val: v(ROWS_PER_GROUP - 3),
        }];
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &narrow,
            true,
            ChildShape::Direct,
        );
        let Some(ProbeMatch::Rows(rows)) = prober.get_subset(&[v(2)]) else {
            panic!("expected filtered rows")
        };
        assert!(matches!(rows.kind(), AtomRowsKind::Inline(_)));
        assert_eq!(rows_of(&rows).len(), 2);

        // Existence-only probes apply the same constraint check.
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &only_stale,
            false,
            ChildShape::Leaf,
        );
        assert!(prober.get_subset(&[v(0)]).is_none());
        let prober = column_prober(
            &db,
            table,
            index,
            &continuations,
            &narrow,
            false,
            ChildShape::Leaf,
        );
        assert!(matches!(
            prober.get_subset(&[v(0)]),
            Some(ProbeMatch::Present)
        ));
    }

    #[test]
    fn tuple_catalog_matches_apply_the_same_filter() {
        let (db, table) = stale_db();
        let columns = [ColumnId::new(0), ColumnId::new(1)];
        let index = get_index_from_tableinfo(db.get_table_info(table), &columns);
        let index = index.get().unwrap();
        assert!(index.get_subset(&[v(0), v(0)]).is_some());
        let continuations = RootContinuationCache::default();
        let wrapped = db.get_table(table);
        let filter = CatalogFilter {
            table: wrapped.as_ref(),
            constraints: &[],
            check_live: wrapped.has_stale_rows(),
        };
        let prober = Prober {
            source: AtomRows::dense(OffsetRange::new(
                RowId::new(0),
                RowId::from_usize(GROUPS * ROWS_PER_GROUP),
            )),
            ix: ProbeIndex::CachedTuple {
                intersect_outer: None,
                table: index,
                continuations: &continuations,
                child_shape: ChildShape::Leaf,
                filter,
            },
            keep_rows: false,
        };
        assert!(prober.get_subset(&[v(0), v(0)]).is_none());
        assert!(matches!(
            prober.get_subset(&[v(0), v(1)]),
            Some(ProbeMatch::Present)
        ));
        let mut live = 0;
        prober.for_each(|_, _| live += 1);
        assert_eq!(live, (GROUPS - 1) * (ROWS_PER_GROUP - 1));
    }
}
