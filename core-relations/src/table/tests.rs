use crate::numeric_id::NumericId;
use rand::{Rng, rng};

use crate::{
    common::{HashMap, ShardId, Value},
    offsets::{RowId, SubsetRef},
    row_buffer::TaggedRowBuffer,
    table::{TableEntry, hash_code},
    table_shortcuts::{fill_table, v},
    table_spec::{ColumnId, Constraint, Offset, Table, WrappedTable},
};

use super::sharded_hash_table::ShardedHashTable;

#[test]
fn typed_class_inventory_clear_and_live_buffer_clone() {
    empty_execution_state!(state);
    let mut table = super::SortedWritesTable::new(
        1,
        2,
        None,
        vec![ColumnId::new(1)],
        Box::new(|_, _, new, out| {
            out.extend_from_slice(new);
            true
        }),
    )
    .with_eq_sorts(vec![None, Some("A".into())]);
    let mut buffer = table.new_buffer();
    buffer.stage_insert(&[v(0), v(10)]);
    drop(buffer);
    table.clear();
    table.merge(&mut state);
    assert_eq!(table.len(), 0);
    assert_eq!(
        *table.pending_state.eq_classes.0.lock().unwrap(),
        [("A".into(), v(10))].into_iter().collect()
    );

    let mut buffer = table.new_buffer();
    buffer.stage_insert(&[v(1), v(11)]);
    let mut cloned = table.clone();
    buffer.stage_insert(&[v(2), v(12)]);
    // Clear does not cancel local rows in a still-live buffer.
    table.clear();
    drop(buffer);
    table.merge(&mut state);
    cloned.merge(&mut state);
    assert_eq!(table.len(), 2);
    assert_eq!(cloned.len(), 0);
    assert_eq!(
        *table.pending_state.eq_classes.0.lock().unwrap(),
        [
            ("A".into(), v(10)),
            ("A".into(), v(11)),
            ("A".into(), v(12))
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(
        *cloned.pending_state.eq_classes.0.lock().unwrap(),
        [("A".into(), v(10)), ("A".into(), v(11))]
            .into_iter()
            .collect()
    );
    let generation = table.generation;
    for value in 20..40 {
        table.new_buffer().stage_insert(&[v(0), v(value)]);
        table.merge(&mut state);
    }
    assert_ne!(table.generation, generation, "stale rows were compacted");
    assert_eq!(table.len(), 3);
    assert_eq!(
        *table.pending_state.eq_classes.0.lock().unwrap(),
        [10, 11, 12]
            .into_iter()
            .chain(20..40)
            .map(|value| ("A".into(), v(value)))
            .collect()
    );
}

#[test]
fn typed_class_inventory_selected_lookup_without_staging() {
    use crate::action::{Bindings, Instr, QueryEntry};
    for case in 0..3 {
        let mut db = crate::Database::new();
        let table = db.add_table(
            super::SortedWritesTable::new(
                1,
                2,
                None,
                vec![ColumnId::new(1)],
                Box::new(|_, _, _, _| false),
            )
            .with_eq_sorts(vec![None, Some("A".into())]),
            [],
            [],
        );
        let value = v(41);
        let output = crate::Variable::new(0);
        let instruction = if case == 0 {
            Instr::LookupWithDefault {
                table,
                args: vec![QueryEntry::Const(v(0))],
                dst_col: ColumnId::new(1),
                dst_var: output,
                default: QueryEntry::Const(value),
            }
        } else {
            let func =
                db.add_external_function(Box::new(crate::make_external_func(move |_, _| {
                    (case == 1).then_some(value)
                })));
            Instr::LookupWithFallback {
                table,
                table_key: vec![QueryEntry::Const(v(0))],
                func,
                func_args: vec![],
                dst_col: ColumnId::new(1),
                dst_var: output,
            }
        };
        let mut bindings = Bindings::new(1);
        let matches = db.with_execution_state(None, |state| {
            state.run_instrs(&[instruction], &mut bindings)
        });
        assert_eq!(matches, usize::from(case != 2));
        assert_eq!(db.get_table(table).len(), 0);
        let expected = if case == 2 {
            vec![]
        } else {
            vec![("A".into(), value)]
        };
        assert_eq!(db.materialized_eq_classes(), expected.into_iter().collect());
        let saved = db.clone();
        db.with_execution_state(None, |state| state.materialize_eq_value("B".into(), v(42)));
        assert!(
            !saved
                .materialized_eq_classes()
                .contains(&("B".into(), v(42)))
        );
    }

    // A hit, a successful fallback, and a failed fallback share one active mask.
    let mut db = crate::Database::new();
    let table = db.add_table(
        super::SortedWritesTable::new(
            1,
            2,
            None,
            vec![ColumnId::new(1)],
            Box::new(|_, _, _, _| false),
        )
        .with_eq_sorts(vec![None, Some("A".into())]),
        [],
        [],
    );
    db.new_buffer(table).stage_insert(&[v(0), v(40)]);
    db.merge_all();
    let func = db.add_external_function(Box::new(crate::make_external_func(|_, args| {
        (args[0] == v(1)).then_some(v(41))
    })));
    let key = crate::Variable::new(0);
    let output = crate::Variable::new(1);
    let mut bindings = Bindings::new(3);
    bindings.insert(key, &[v(0), v(1), v(2)]);
    let instruction = Instr::LookupWithFallback {
        table,
        table_key: vec![key.into()],
        func,
        func_args: vec![key.into()],
        dst_col: ColumnId::new(1),
        dst_var: output,
    };
    assert_eq!(
        db.with_execution_state(None, |state| state
            .run_instrs(&[instruction], &mut bindings)),
        2
    );
    assert_eq!(bindings[output][..2], [v(40), v(41)]);
    // An eager default on a hit never becomes a selected result.
    let mut bindings = Bindings::new(1);
    let hit = Instr::LookupWithDefault {
        table,
        args: vec![QueryEntry::Const(v(0))],
        dst_col: ColumnId::new(1),
        dst_var: output,
        default: QueryEntry::Const(v(42)),
    };
    assert_eq!(
        db.with_execution_state(None, |state| state.run_instrs(&[hit], &mut bindings)),
        1
    );
    assert_eq!(bindings[output], [v(40)]);
    assert_eq!(
        db.materialized_eq_classes(),
        [("A".into(), v(40)), ("A".into(), v(41))]
            .into_iter()
            .collect()
    );
    assert_eq!(db.get_table(table).len(), 1);
}

fn dump_buf(buf: &TaggedRowBuffer) -> Vec<(RowId, Vec<Value>)> {
    let mut res = Vec::new();
    buf.iter()
        .for_each(|(id, row)| res.push((id, row.to_vec())));
    res
}

fn dump_subset(table: &impl Table, subset: SubsetRef) -> Vec<(RowId, Vec<Value>)> {
    let mut res = Vec::new();
    table.scan_generic(subset, |id, row| {
        res.push((id, row.to_vec()));
    });
    res
}

#[test]
fn empty_key() {
    empty_execution_state!(e);
    let mut table = fill_table(
        vec![vec![v(1), v(2)], vec![v(2), v(3)]],
        0,
        None,
        |_, new| Some(new.to_vec()),
    );
    let row = table.get_row(&[]).expect("empty key should be present");
    assert_eq!(*row.vals, vec![v(2), v(3)]);
    table.new_buffer().stage_remove(&[]);
    table.merge(&mut e);
    assert!(table.get_row(&[]).is_none(), "empty key should be removed");
    table.new_buffer().stage_insert(&[v(1), v(2)]);
    table.merge(&mut e);
    let row = table.get_row(&[]).expect("empty key should be present");
    assert_eq!(*row.vals, vec![v(1), v(2)]);
}

#[test]
fn insert_scan() {
    let table = fill_table(
        vec![
            vec![v(0), v(1), v(2)],
            vec![v(1), v(2), v(3)],
            vec![v(2), v(3), v(4)],
            vec![v(3), v(4), v(5)],
            vec![v(2), v(3), v(6)],
        ],
        2,
        None,
        |_, new| Some(new.to_vec()),
    );

    let all = table.all();
    let smaller = table.refine_one(
        all,
        &Constraint::GtConst {
            col: ColumnId::new(2),
            val: v(4),
        },
    );
    let rows = dump_subset(&table, smaller.as_ref());
    assert_eq!(
        rows,
        vec![
            (RowId::new(3), vec![v(3), v(4), v(5)]),
            (RowId::new(4), vec![v(2), v(3), v(6)])
        ]
    );
    let mut buf = TaggedRowBuffer::new(2);
    let table = WrappedTable::new(table);
    table.scan_project(
        smaller.as_ref(),
        &[ColumnId::new(1), ColumnId::new(0)],
        Offset::new(0),
        usize::MAX,
        &[],
        &mut buf,
    );

    let projection = dump_buf(&buf);
    assert_eq!(
        projection,
        vec![
            (RowId::new(3), vec![v(4), v(3)]),
            (RowId::new(4), vec![v(3), v(2)])
        ]
    );
}

#[test]
fn insert_scan_sorted() {
    let table = fill_table(
        vec![
            vec![v(0), v(1), v(2)],
            vec![v(1), v(2), v(3)],
            vec![v(2), v(3), v(4)],
            vec![v(3), v(4), v(5)],
            vec![v(2), v(3), v(6)],
        ],
        2,
        Some(ColumnId::new(2)),
        |_, new| Some(new.to_vec()),
    );

    let all = table.all();
    let smaller = table.refine_one(
        all,
        &Constraint::LtConst {
            col: ColumnId::new(1),
            val: v(4),
        },
    );
    let rows = dump_subset(&table, smaller.as_ref());
    assert_eq!(
        rows,
        vec![
            (RowId::new(0), vec![v(0), v(1), v(2)]),
            (RowId::new(1), vec![v(1), v(2), v(3)]),
            (RowId::new(4), vec![v(2), v(3), v(6)]),
        ]
    );

    let all = table.all();
    let sorted_smaller = table.refine_one(
        all,
        &Constraint::LtConst {
            col: ColumnId::new(2),
            val: v(5),
        },
    );
    let rows = dump_subset(&table, sorted_smaller.as_ref());
    assert_eq!(
        rows,
        vec![
            (RowId::new(0), vec![v(0), v(1), v(2)]),
            (RowId::new(1), vec![v(1), v(2), v(3)]),
        ]
    );
}

#[test]
fn shard_math() {
    let mut table = ShardedHashTable::<TableEntry>::with_shards(14);
    // Should be rounded up to 16.
    assert_eq!(table.mut_shards().len(), 16);

    // If we generate a hundred thousand random rows, we should see more than 100
    // items in each shard.
    let mut rng = rng();
    let mut hist = HashMap::default();
    (0..100_000)
        .map(|_| {
            hash_code(
                table.shard_data(),
                &[
                    Value::new(rng.random()),
                    Value::new(rng.random()),
                    Value::new(rng.random()),
                ],
                2,
            )
            .0
        })
        .for_each(|id| *hist.entry(id).or_insert(0) += 1);
    assert!(hist.iter().all(|(_, count)| *count > 100), "{hist:?}");

    // Picking low numbers should all get shard 0.
    assert!((0..100_000).all(|x| table.shard_data().shard_id(x as u64) == ShardId::new(0)));
}
