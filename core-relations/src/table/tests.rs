use crate::numeric_id::NumericId;
use rand::{Rng, rng};

use crate::{
    action::ExecutionState,
    common::{HashMap, ShardId, Value},
    free_join::Database,
    offsets::{RowId, SubsetRef},
    row_buffer::TaggedRowBuffer,
    table::{SortedWritesTable, SortedWritesTableOptions, TableEntry, hash_code},
    table_shortcuts::{fill_table, v},
    table_spec::{ColumnId, Constraint, Offset, Table, WrappedTable},
};

use super::sharded_hash_table::ShardedHashTable;

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
fn stable_row_id_is_assigned_and_preserved() {
    let mut db = Database::default();
    let row_id_counter = db.add_counter();
    let mut exec_state = ExecutionState::new(db.read_only_view(), Default::default());
    let mut table = SortedWritesTable::new(
        1,
        3,
        SortedWritesTableOptions {
            sort_by: None,
            row_id: Some((row_id_counter, ColumnId::new(2))),
        },
        vec![],
        Box::new(|_, cur, new, out| {
            assert_eq!(cur[2], new[2], "row id should be stable across merges");
            if cur[1] != new[1] {
                out.extend_from_slice(new);
                true
            } else {
                false
            }
        }),
    );

    table.new_buffer().stage_insert(&[v(1), v(10), v(999)]);
    table.merge(&mut exec_state);
    let row = table.get_row(&[v(1)]).expect("row should exist");
    let row_id = row.vals[2];
    assert_ne!(row_id, v(999));

    table.new_buffer().stage_insert(&[v(1), v(20), v(1000)]);
    table.merge(&mut exec_state);
    let row = table.get_row(&[v(1)]).expect("row should exist");
    assert_eq!(row.vals[2], row_id);

    table.new_buffer().stage_insert(&[v(2), v(30), v(1001)]);
    table.merge(&mut exec_state);
    let row = table.get_row(&[v(2)]).expect("row should exist");
    assert_ne!(row.vals[2], row_id);
}

#[test]
fn stable_row_id_across_compaction() {
    let mut db = Database::default();
    let row_id_counter = db.add_counter();
    let mut exec_state = ExecutionState::new(db.read_only_view(), Default::default());
    let mut table = SortedWritesTable::new(
        1,
        3,
        SortedWritesTableOptions {
            sort_by: None,
            row_id: Some((row_id_counter, ColumnId::new(2))),
        },
        vec![],
        Box::new(|_, cur, new, out| {
            assert_eq!(cur[2], new[2], "row id should be stable across merges");
            if cur[1] != new[1] {
                out.extend_from_slice(new);
                true
            } else {
                false
            }
        }),
    );

    const ROWS: usize = 50;
    {
        let mut buf = table.new_buffer();
        for i in 0..ROWS {
            buf.stage_insert(&[v(i), v(1000 + i), v(123)]);
        }
    }
    table.merge(&mut exec_state);

    let row_id0 = table.get_row(&[v(0)]).expect("row should exist").vals[2];
    let row_id1 = table.get_row(&[v(1)]).expect("row should exist").vals[2];

    let gen0 = table.version().major.index();
    {
        let mut buf = table.new_buffer();
        for i in 2..ROWS {
            buf.stage_remove(&[v(i)]);
        }
    }
    table.merge(&mut exec_state);
    let gen1 = table.version().major.index();
    assert!(gen1 > gen0, "expected compaction after removals");
    assert_eq!(
        table.get_row(&[v(0)]).expect("row should exist").vals[2],
        row_id0
    );
    assert_eq!(
        table.get_row(&[v(1)]).expect("row should exist").vals[2],
        row_id1
    );

    {
        let mut buf = table.new_buffer();
        for i in 2..ROWS {
            buf.stage_insert(&[v(i), v(2000 + i), v(456)]);
        }
    }
    table.merge(&mut exec_state);

    {
        let mut buf = table.new_buffer();
        for i in 2..ROWS {
            buf.stage_remove(&[v(i)]);
        }
    }
    table.merge(&mut exec_state);
    let gen2 = table.version().major.index();
    assert!(gen2 > gen1, "expected second compaction after removals");
    assert_eq!(
        table.get_row(&[v(0)]).expect("row should exist").vals[2],
        row_id0
    );
    assert_eq!(
        table.get_row(&[v(1)]).expect("row should exist").vals[2],
        row_id1
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
