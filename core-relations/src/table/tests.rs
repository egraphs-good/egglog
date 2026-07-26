use crate::numeric_id::NumericId;
use rand::{Rng, rng};

use crate::{
    common::{HashMap, ShardId, Value},
    offsets::{RowId, SubsetRef},
    row_buffer::TaggedRowBuffer,
    table::{SortedWritesTable, TableEntry, hash_code},
    table_shortcuts::{fill_table, v},
    table_spec::{ColumnId, Constraint, Offset, Table, WrappedTable},
};
use egglog_concurrency::ThreadPool;

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
fn known_row_removals_match_the_staged_row_id() {
    empty_execution_state!(e);
    let mut table = fill_table(
        vec![
            vec![v(0), v(1), v(2)],
            vec![v(1), v(2), v(3)],
            vec![v(2), v(3), v(4)],
        ],
        2,
        None,
        |_, new| Some(new.to_vec()),
    );
    let key = [v(1), v(2)];
    let row = table.get_row(&key).unwrap().id;

    let mut wrong = table.new_table_buffer();
    wrong.stage_remove_row(RowId::from_usize(row.index() + 1), &key);
    drop(wrong);
    table.merge(&mut e);
    assert!(table.get_row(&key).is_some());

    let mut exact = table.new_table_buffer();
    exact.stage_remove_row(row, &key);
    drop(exact);
    table.merge(&mut e);
    assert!(table.get_row(&key).is_none());
    assert_eq!(table.len(), 2);
}

#[test]
fn pending_partitioned_mutations_survive_table_clone() {
    empty_execution_state!(e);
    let mut table = fill_table(
        vec![vec![v(1), v(10)], vec![v(2), v(20)]],
        1,
        None,
        |_, new| Some(new.to_vec()),
    );
    let mut pending = table.new_buffer();
    pending.stage_remove(&[v(1)]);
    pending.stage_insert(&[v(3), v(30)]);
    drop(pending);

    let mut cloned = table.clone();
    table.merge(&mut e);
    cloned.merge(&mut e);

    for table in [&table, &cloned] {
        assert!(table.get_row(&[v(1)]).is_none());
        assert_eq!(&*table.get_row(&[v(2)]).unwrap().vals, &[v(2), v(20)]);
        assert_eq!(&*table.get_row(&[v(3)]).unwrap().vals, &[v(3), v(30)]);
    }
}

#[test]
fn cached_hashes_stay_aligned_through_parallel_duplicate_staging() {
    const KEYS: usize = 10_000;
    let pool = ThreadPool::new(4);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(new);
                old != new
            }),
        );

        for base in (0..KEYS).step_by(257) {
            let mut buffer = table.new_buffer();
            for key in base..(base + 257).min(KEYS) {
                buffer.stage_insert(&[v(key), v(key)]);
                buffer.stage_insert(&[v(key), v(key + 1)]);
            }
        }

        assert!(table.parallel_insert(&e, ()));
        assert_eq!(table.len(), KEYS);
        for key in 0..KEYS {
            assert_eq!(
                &*table.get_row(&[v(key)]).unwrap().vals,
                &[v(key), v(key + 1)]
            );
        }
    });
}

#[test]
fn parallel_insert_uses_merge_output_for_existing_row() {
    let pool = ThreadPool::new(4);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(&[old[0], Value::new(old[1].rep() + new[1].rep())]);
                true
            }),
        );

        table.new_buffer().stage_insert(&[v(1), v(10)]);
        table.merge(&mut e);
        table.new_buffer().stage_insert(&[v(1), v(7)]);

        assert!(table.parallel_insert(&e, ()));
        assert_eq!(&*table.get_row(&[v(1)]).unwrap().vals, &[v(1), v(17)]);
        assert_eq!(table.len(), 1);
    });
}

#[test]
fn parallel_insert_merges_duplicate_split_across_flushes() {
    let pool = ThreadPool::new(1);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(&[old[0], Value::new(old[1].rep() + new[1].rep())]);
                true
            }),
        );
        let target = super::PARALLEL_INSERT_BATCH_SIZE + 1;
        let mut buffer = table.new_buffer();
        buffer.stage_insert(&[v(target), v(10)]);
        for key in 0..super::PARALLEL_INSERT_BATCH_SIZE {
            buffer.stage_insert(&[v(key), v(key)]);
        }
        buffer.stage_insert(&[v(target), v(7)]);
        drop(buffer);

        assert!(table.parallel_insert(&e, ()));
        assert_eq!(
            &*table.get_row(&[v(target)]).unwrap().vals,
            &[v(target), v(17)]
        );
        assert_eq!(table.len(), super::PARALLEL_INSERT_BATCH_SIZE + 1);
    });
}

#[test]
fn parallel_insert_reports_no_change_when_merge_rejects_update() {
    let pool = ThreadPool::new(4);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(1, 2, None, vec![], Box::new(|_, _, _, _| false));

        table.new_buffer().stage_insert(&[v(1), v(10)]);
        table.merge(&mut e);
        table.new_buffer().stage_insert(&[v(1), v(7)]);

        assert!(!table.parallel_insert(&e, ()));
        assert_eq!(&*table.get_row(&[v(1)]).unwrap().vals, &[v(1), v(10)]);
        assert_eq!(table.len(), 1);
    });
}

#[test]
fn parallel_insert_preserves_coalesced_merge_parenthesization() {
    let pool = ThreadPool::new(4);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(&[
                    old[0],
                    Value::new(old[1].rep().checked_sub(new[1].rep()).unwrap()),
                ]);
                true
            }),
        );

        table.new_buffer().stage_insert(&[v(1), v(20)]);
        table.merge(&mut e);

        let mut buffer = table.new_buffer();
        buffer.stage_insert(&[v(1), v(7)]);
        buffer.stage_insert(&[v(1), v(3)]);
        drop(buffer);

        assert!(table.parallel_insert(&e, ()));
        // Pending rows are coalesced before consulting the destination:
        // 20 - (7 - 3) = 16. Direct left-folding would produce 10.
        assert_eq!(&*table.get_row(&[v(1)]).unwrap().vals, &[v(1), v(16)]);
        assert_eq!(table.len(), 1);
    });
}

#[test]
fn parallel_unsorted_rehash_compacts_heavy_stale_rows_and_repoints_entries() {
    const KEYS: usize = 70_000;
    const UPDATES: usize = 5;

    let pool = ThreadPool::new(4);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(new);
                old != new
            }),
        );

        // Deliberately bypass `merge`'s compaction step so the final call
        // crosses the normal parallel-compaction threshold with a row store
        // that is overwhelmingly stale.
        for update in 0..=UPDATES {
            let mut buffer = table.new_buffer();
            for key in 0..KEYS {
                buffer.stage_insert(&[v(key), v(update)]);
            }
            drop(buffer);
            assert!(table.do_insert(&mut e));
        }

        assert_eq!(table.data.data.len(), KEYS * (UPDATES + 1));
        assert_eq!(table.data.stale_rows, KEYS * UPDATES);
        let old_generation = table.generation;

        table.maybe_rehash();

        assert_eq!(table.generation.index(), old_generation.index() + 1);
        assert_eq!(table.data.data.len(), KEYS);
        assert_eq!(table.data.stale_rows, 0);
        assert_eq!(table.len(), KEYS);

        let mut seen_rows = vec![false; KEYS];
        let mut seen_keys = vec![false; KEYS];
        let shard_data = table.hash.shard_data();
        for (shard_index, shard) in table.hash.mut_shards().iter().enumerate() {
            for entry in shard.iter() {
                let row = table.data.data.get_row(entry.row);
                assert_eq!(row[1], v(UPDATES));
                assert_eq!(entry.hashcode, hash_code(shard_data, row, 1).1);
                assert_eq!(shard_data.shard_id(entry.hashcode).index(), shard_index);

                assert!(entry.row.index() < KEYS);
                assert!(!seen_rows[entry.row.index()]);
                seen_rows[entry.row.index()] = true;

                let key = row[0].rep() as usize;
                assert!(key < KEYS);
                assert!(!seen_keys[key]);
                seen_keys[key] = true;
            }
        }
        assert!(seen_rows.into_iter().all(|seen| seen));
        assert!(seen_keys.into_iter().all(|seen| seen));

        // Exercise the repointed hash entries after compaction.
        table.new_buffer().stage_insert(&[v(17), v(UPDATES + 1)]);
        assert!(table.do_insert(&mut e));
        assert_eq!(
            &*table.get_row(&[v(17)]).unwrap().vals,
            &[v(17), v(UPDATES + 1)]
        );
    });
}

#[test]
fn parallel_unsorted_rehash_handles_no_live_rows() {
    const KEYS: usize = 1_024;

    let pool = ThreadPool::new(4);
    pool.install(|| {
        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(1, 2, None, vec![], Box::new(|_, _, _, _| false));

        let mut inserts = table.new_buffer();
        for key in 0..KEYS {
            inserts.stage_insert(&[v(key), v(key + 1)]);
        }
        drop(inserts);
        assert!(table.do_insert(&mut e));

        let mut removals = table.new_buffer();
        for key in 0..KEYS {
            removals.stage_remove(&[v(key)]);
        }
        drop(removals);
        assert!(table.do_delete());
        assert_eq!(table.data.stale_rows, KEYS);
        assert!(table.hash.mut_shards().iter().all(|shard| shard.is_empty()));

        let old_generation = table.generation;
        table.parallel_rehash();

        assert_eq!(table.generation.index(), old_generation.index() + 1);
        assert_eq!(table.data.data.len(), 0);
        assert_eq!(table.data.stale_rows, 0);
        assert_eq!(table.len(), 0);
        assert!(table.hash.mut_shards().iter().all(|shard| shard.is_empty()));
    });
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
