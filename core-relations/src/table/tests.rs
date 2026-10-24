use std::mem::{align_of, size_of};

use crate::numeric_id::NumericId;
use rand::{Rng, rng};

use crate::{
    common::{HashMap, ShardId, Value},
    offsets::{RowId, SubsetRef},
    row_buffer::TaggedRowBuffer,
    table::{
        CompactHash, FullHash, HashedRowBuffer, KnownRemoval, ProbeHash, ShardHash,
        SortedWritesTable, TableEntry, shard_hash,
    },
    table_shortcuts::{fill_table, fill_timestamped_table, v},
    table_spec::{ColumnId, Constraint, MutationBuffer, Offset, Table, WrappedTable},
};
use egglog_concurrency::ThreadPool;

use super::sharded_hash_table::ShardedHashTable;

fn parallel_insert(table: &mut SortedWritesTable, state: &crate::ExecutionState<'_>) -> bool {
    table.parallel_insert(
        state,
        super::SortChecker {
            col: table.sort_by,
            baseline: table.offsets.last().map(|(value, _)| *value),
            current: None,
        },
    )
}

fn compaction_test_table() -> SortedWritesTable {
    let mut table = SortedWritesTable::new(
        2,
        4,
        ColumnId::new(3),
        vec![],
        Box::new(|_, old, new, out| {
            out.extend_from_slice(new);
            old != new
        }),
    );
    table.hash = ShardedHashTable::with_shards(16);
    table.pending_state = std::sync::Arc::new(super::PendingState::new(table.hash.shard_data()));
    table
}

fn assert_parallel_compaction_matches_serial(table: SortedWritesTable) {
    let mut expected = dump_subset(&table, table.all().as_ref())
        .into_iter()
        .map(|(_, row)| row)
        .collect::<Vec<_>>();
    expected.sort();

    let mut serial = table.clone();
    serial.rehash();
    let mut parallel = table;
    parallel.parallel_rehash();

    assert_eq!(parallel.offsets, serial.offsets);
    assert_eq!(parallel.version(), serial.version());
    for compacted in [&serial, &parallel] {
        assert_eq!(compacted.data.stale_rows, 0);
        assert_eq!(compacted.data.data.len(), expected.len());
        let mut actual = dump_subset(compacted, compacted.all().as_ref())
            .into_iter()
            .map(|(_, row)| row)
            .collect::<Vec<_>>();
        actual.sort();
        assert_eq!(actual, expected);
        for row in &expected {
            assert_eq!(&*compacted.get_row(&row[..2]).unwrap().vals, row);
        }
        for (timestamp, _) in &compacted.offsets {
            let (start, end) = compacted.binary_search_sort_val(*timestamp).unwrap();
            assert!(start < end);
            for row in start.index()..end.index() {
                assert_eq!(
                    compacted.data.data.get_row(RowId::from_usize(row))[3],
                    *timestamp
                );
            }
        }
    }
}

#[test]
fn parallel_compaction_matches_serial_with_dominant_final_timestamp() {
    ThreadPool::new(4).install(|| {
        empty_execution_state!(e);
        let mut table = compaction_test_table();
        for (timestamp, start, count) in [(3, 0, 5), (79, 3, 17), (1_000_003, 16, 4097)] {
            let mut buffer = table.new_buffer();
            for key in start..start + count {
                buffer.stage_insert(&[v(key), v(11), v(timestamp + key), v(timestamp)]);
            }
            drop(buffer);
            // Keep obsolete rows until the explicit compaction below.
            assert!(table.do_insert(&mut e));
        }
        assert!(table.data.stale_rows > 0);
        assert_parallel_compaction_matches_serial(table);
    });
}

#[test]
fn parallel_compaction_handles_a_single_timestamp() {
    ThreadPool::new(4).install(|| {
        empty_execution_state!(e);
        let mut table = compaction_test_table();
        for (count, value) in [(64, 10), (32, 20)] {
            let mut buffer = table.new_buffer();
            for key in 0..count {
                buffer.stage_insert(&[v(key), v(17), v(value), v(7_000_001)]);
            }
            drop(buffer);
            assert!(table.do_insert(&mut e));
        }
        assert_eq!(table.offsets.len(), 1);
        assert_eq!(table.data.stale_rows, 32);
        assert_parallel_compaction_matches_serial(table);
    });
}

#[test]
fn parallel_compaction_removes_empty_timestamp_ranges_and_handles_empty_result() {
    ThreadPool::new(4).install(|| {
        empty_execution_state!(e);
        let mut table = compaction_test_table();
        for (timestamp, keys) in [(0, vec![1, 2]), (57, vec![3, 4]), (2_000_011, vec![1, 5])] {
            let mut buffer = table.new_buffer();
            for key in keys {
                buffer.stage_insert(&[v(key), v(13), v(key), v(timestamp)]);
            }
            drop(buffer);
            assert!(table.do_insert(&mut e));
        }
        for key in [2, 3, 4] {
            table.new_buffer().stage_remove(&[v(key), v(13)]);
        }
        table.do_delete();
        assert_eq!(table.offsets.len(), 3);
        assert_eq!(table.len(), 2);
        assert_parallel_compaction_matches_serial(table.clone());

        for key in [1, 5] {
            table.new_buffer().stage_remove(&[v(key), v(13)]);
        }
        table.do_delete();
        assert_eq!(table.len(), 0);
        assert!(!table.offsets.is_empty());
        assert_parallel_compaction_matches_serial(table);
    });
}

#[test]
fn parallel_compaction_preserves_collisions_and_sparse_shards() {
    ThreadPool::new(4).install(|| {
        empty_execution_state!(e);
        let mut table = compaction_test_table();
        let left = [Value::new(70_759_468), Value::new(2_418_062_243)];
        let right = [Value::new(3_427_223_041), Value::new(1_840_803_043)];
        assert_eq!(
            shard_hash(table.hash.shard_data(), &left, 2),
            shard_hash(table.hash.shard_data(), &right, 2)
        );
        for (key, value, timestamp) in [(left, 10, 7), (right, 20, 7), (left, 30, 89)] {
            table
                .new_buffer()
                .stage_insert(&[key[0], key[1], v(value), v(timestamp)]);
            assert!(table.do_insert(&mut e));
        }
        assert_eq!(
            table
                .hash
                .mut_shards()
                .iter()
                .filter(|s| !s.is_empty())
                .count(),
            1
        );
        assert_parallel_compaction_matches_serial(table);
    });
}

#[test]
fn compact_hash_preserves_bucket_bits_and_exact_h2_tag() {
    const LOW_MASK: u64 = (1 << 25) - 1;
    const H2_SHIFT: u32 = if usize::BITS < u64::BITS {
        usize::BITS - 7
    } else {
        u64::BITS - 7
    };
    const H2_MASK: u64 = 0x7f << H2_SHIFT;
    let cases = [
        0,
        1,
        LOW_MASK,
        1 << 25,
        1 << (H2_SHIFT - 1),
        1 << H2_SHIFT,
        0x1234_5678_9abc_def0,
        u64::MAX,
    ];

    for raw in cases {
        let compact = CompactHash::from_full(FullHash(raw));
        let probe = compact.probe().raw();

        assert_eq!(compact.0 as u64 & LOW_MASK, raw & LOW_MASK);
        assert_eq!((compact.0 >> 25) as u64, (raw >> H2_SHIFT) & 0x7f);
        assert_eq!(probe & u32::MAX as u64, compact.0 as u64);
        assert_eq!(probe & LOW_MASK, raw & LOW_MASK);
        assert_eq!(probe & H2_MASK, raw & H2_MASK);
        assert_eq!(probe & !(u32::MAX as u64 | H2_MASK), 0);
    }
}

#[test]
fn compact_hash_layouts_remain_small() {
    assert_eq!(size_of::<CompactHash>(), 4);
    assert_eq!(size_of::<ProbeHash>(), 8);
    assert_eq!(size_of::<ShardHash>(), 8);
    assert_eq!(size_of::<TableEntry>(), 8);
    assert_eq!(align_of::<TableEntry>(), 4);
    assert_eq!(size_of::<KnownRemoval>(), 8);
    assert_eq!(align_of::<KnownRemoval>(), 4);
}

#[test]
fn full_hash_routes_before_omitted_shard_bits_are_compacted() {
    let shard_data = crate::common::ShardData::new(16);
    let left = ShardHash::from_full(shard_data, FullHash(0));
    let right = ShardHash::from_full(shard_data, FullHash(1 << 53));

    assert_eq!(left.compact, right.compact);
    assert_ne!(left.shard, right.shard);
}

#[test]
fn hashed_row_buffer_round_trips_max_compact_hash_in_trailing_lane() {
    let row = [v(7), v(11)];
    let mut buffer = HashedRowBuffer::new(row.len());
    buffer.add_row(CompactHash(u32::MAX), &row);
    buffer.add_row(CompactHash(17), &[v(13), v(19)]);

    assert_eq!(buffer.values[row.len()].rep(), u32::MAX);
    let buffered = buffer.rows_hashed().collect::<Vec<_>>();
    assert_eq!(buffered[0], (CompactHash(u32::MAX), row.as_slice()));
    assert_eq!(buffered[1], (CompactHash(17), [v(13), v(19)].as_slice()));
}

#[test]
fn same_shard_compact_collision_survives_table_lifecycle() {
    let shard_data = crate::common::ShardData::new(16);
    let left_key = [Value::new(70_759_468), Value::new(2_418_062_243)];
    let right_key = [Value::new(3_427_223_041), Value::new(1_840_803_043)];
    let left_hash = shard_hash(shard_data, &left_key, left_key.len());
    let right_hash = shard_hash(shard_data, &right_key, right_key.len());
    assert_eq!(left_hash, right_hash);
    assert_eq!(left_hash.shard, ShardId::new(14));
    assert_eq!(left_hash.compact, CompactHash(0x578d_834a));

    empty_execution_state!(e);
    let mut table = SortedWritesTable::new(
        2,
        4,
        ColumnId::new(3),
        vec![],
        Box::new(|_, old, new, out: &mut Vec<Value>| {
            out.extend_from_slice(new);
            old != new
        }),
    );
    table.hash = ShardedHashTable::with_shards(16);
    table.pending_state = std::sync::Arc::new(super::PendingState::new(shard_data));

    table
        .new_buffer()
        .stage_insert(&[left_key[0], left_key[1], v(10), v(0)]);
    table
        .new_buffer()
        .stage_insert(&[right_key[0], right_key[1], v(20), v(0)]);
    table.merge(&mut e);
    assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(10));
    assert_eq!(table.get_row(&right_key).unwrap().vals[2], v(20));

    table
        .new_buffer()
        .stage_insert(&[left_key[0], left_key[1], v(30), v(1)]);
    table.merge(&mut e);
    assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(30));
    assert_eq!(table.get_row(&right_key).unwrap().vals[2], v(20));

    table.rehash();
    assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(30));
    assert_eq!(table.get_row(&right_key).unwrap().vals[2], v(20));

    table.new_buffer().stage_remove(&right_key);
    table.merge(&mut e);
    assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(30));
    assert!(table.get_row(&right_key).is_none());

    table
        .new_buffer()
        .stage_insert(&[right_key[0], right_key[1], v(40), v(3)]);
    table.merge(&mut e);
    assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(30));
    assert_eq!(table.get_row(&right_key).unwrap().vals[2], v(40));
}

#[test]
fn serial_compaction_preserves_rows_and_timestamp_offsets_across_rank_blocks() {
    for pattern in 0..3 {
        let original: Vec<_> = (0..260)
            .map(|n| vec![v(n + 1), v(n * 7), v(n / 32)])
            .collect();
        let keep: Vec<_> = (0..original.len())
            .map(|old| match pattern {
                0 => true,
                1 => false,
                // Leave a full 64-row block and two timestamp groups empty,
                // and retain rows on both sides of rank-block boundaries.
                _ => old / 64 != 1 && (old % 3 == 1 || old % 64 == 0 || old % 64 == 63),
            })
            .collect();
        let mut table = fill_table(original.clone(), 1, ColumnId::new(2), |_, new| {
            Some(new.to_vec())
        });
        // Stage the state immediately before compaction directly, so automatic
        // threshold checks cannot compact the fixture before its block gaps exist.
        for (old, &keep_row) in keep.iter().enumerate() {
            if !keep_row {
                table.data.set_stale(RowId::from_usize(old));
            }
        }
        for shard in table.hash.mut_shards() {
            shard.retain(|entry| keep[entry.row.index()]);
        }
        let expected: Vec<_> = original
            .iter()
            .zip(&keep)
            .filter(|(_, keep_row)| **keep_row)
            .map(|(row, _)| row.clone())
            .enumerate()
            .map(|(new, row)| (RowId::from_usize(new), row))
            .collect();
        let mut expected_offsets = Vec::new();
        for (new, row) in &expected {
            if expected_offsets
                .last()
                .is_none_or(|(val, _)| *val != row[2])
            {
                expected_offsets.push((row[2], *new));
            }
        }
        // A second compaction also checks the no-stale case after remapping.
        for _ in 0..2 {
            table.rehash();
            assert_eq!(table.data.stale_rows, 0);
            assert_eq!(table.data.data.len(), expected.len());
            assert_eq!(dump_subset(&table, table.all().as_ref()), expected);
            assert_eq!(table.offsets, expected_offsets);
            for (old, row) in original.iter().enumerate() {
                if keep[old] {
                    assert_eq!(table.get_row(&row[..1]).unwrap().vals, row.as_slice());
                } else {
                    assert!(table.get_row(&row[..1]).is_none());
                }
            }
        }
    }
}

#[test]
fn same_shard_compact_collision_survives_parallel_insert_and_delete() {
    ThreadPool::new(4).install(|| {
        let shard_data = crate::common::ShardData::new(16);
        let left_key = [Value::new(70_759_468), Value::new(2_418_062_243)];
        let right_key = [Value::new(3_427_223_041), Value::new(1_840_803_043)];
        assert_eq!(
            shard_hash(shard_data, &left_key, left_key.len()),
            shard_hash(shard_data, &right_key, right_key.len())
        );

        empty_execution_state!(e);
        let mut table = SortedWritesTable::new(
            2,
            4,
            ColumnId::new(3),
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(new);
                old != new
            }),
        );
        table.hash = ShardedHashTable::with_shards(16);
        table.pending_state = std::sync::Arc::new(super::PendingState::new(shard_data));

        table
            .new_buffer()
            .stage_insert(&[left_key[0], left_key[1], v(10), v(0)]);
        table
            .new_buffer()
            .stage_insert(&[right_key[0], right_key[1], v(20), v(0)]);
        assert!(parallel_insert(&mut table, &e));
        assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(10));
        assert_eq!(table.get_row(&right_key).unwrap().vals[2], v(20));

        table.new_buffer().stage_remove(&right_key);
        assert!(table.parallel_delete());
        assert_eq!(table.get_row(&left_key).unwrap().vals[2], v(10));
        assert!(table.get_row(&right_key).is_none());
    });
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
    let mut table =
        fill_timestamped_table(vec![vec![v(1), v(2)], vec![v(2), v(3)]], 0, |_, new| {
            Some(new.to_vec())
        });
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
    let table = fill_timestamped_table(
        vec![
            vec![v(0), v(1), v(2)],
            vec![v(1), v(2), v(3)],
            vec![v(2), v(3), v(4)],
            vec![v(3), v(4), v(5)],
            vec![v(2), v(3), v(6)],
        ],
        2,
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
        ColumnId::new(2),
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
        ColumnId::new(2),
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
fn pending_cached_hash_mutations_survive_table_clone() {
    empty_execution_state!(e);
    let mut table = fill_table(
        vec![vec![v(1), v(10)], vec![v(2), v(20)]],
        1,
        ColumnId::new(1),
        |_, new| Some(new.to_vec()),
    );
    let known_row = table.get_row(&[v(2)]).unwrap().id;
    let mut pending = table.new_table_buffer();
    pending.stage_remove(&[v(1)]);
    pending.stage_remove_row(known_row, &[v(2)]);
    pending.stage_insert(&[v(3), v(30)]);
    drop(pending);

    let mut cloned = table.clone();
    table.merge(&mut e);
    cloned.merge(&mut e);

    for table in [&table, &cloned] {
        assert!(table.get_row(&[v(1)]).is_none());
        assert!(table.get_row(&[v(2)]).is_none());
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
            3,
            ColumnId::new(2),
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(new);
                old != new
            }),
        );

        for base in (0..KEYS).step_by(257) {
            let mut buffer = table.new_buffer();
            for key in base..(base + 257).min(KEYS) {
                buffer.stage_insert(&[v(key), v(key), v(0)]);
                buffer.stage_insert(&[v(key), v(key + 1), v(0)]);
            }
        }

        assert!(parallel_insert(&mut table, &e));
        assert_eq!(table.len(), KEYS);
        for key in 0..KEYS {
            assert_eq!(
                &table.get_row(&[v(key)]).unwrap().vals[..2],
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
            3,
            ColumnId::new(2),
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(&[old[0], Value::new(old[1].rep() + new[1].rep()), new[2]]);
                true
            }),
        );

        table.new_buffer().stage_insert(&[v(1), v(10), v(0)]);
        table.merge(&mut e);
        table.new_buffer().stage_insert(&[v(1), v(7), v(1)]);

        assert!(parallel_insert(&mut table, &e));
        assert_eq!(&table.get_row(&[v(1)]).unwrap().vals[..2], &[v(1), v(17)]);
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
            3,
            ColumnId::new(2),
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(&[old[0], Value::new(old[1].rep() + new[1].rep()), new[2]]);
                true
            }),
        );
        let target = super::PARALLEL_INSERT_BATCH_SIZE + 1;
        let mut buffer = table.new_buffer();
        buffer.stage_insert(&[v(target), v(10), v(0)]);
        for key in 0..super::PARALLEL_INSERT_BATCH_SIZE {
            buffer.stage_insert(&[v(key), v(key), v(0)]);
        }
        buffer.stage_insert(&[v(target), v(7), v(0)]);
        drop(buffer);

        assert!(parallel_insert(&mut table, &e));
        assert_eq!(
            &table.get_row(&[v(target)]).unwrap().vals[..2],
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
        let mut table =
            SortedWritesTable::new(1, 3, ColumnId::new(2), vec![], Box::new(|_, _, _, _| false));

        table.new_buffer().stage_insert(&[v(1), v(10), v(0)]);
        table.merge(&mut e);
        table.new_buffer().stage_insert(&[v(1), v(7), v(1)]);

        assert!(!parallel_insert(&mut table, &e));
        assert_eq!(&table.get_row(&[v(1)]).unwrap().vals[..2], &[v(1), v(10)]);
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
            3,
            ColumnId::new(2),
            vec![],
            Box::new(|_, old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(&[
                    old[0],
                    Value::new(old[1].rep().checked_sub(new[1].rep()).unwrap()),
                    new[2],
                ]);
                true
            }),
        );

        table.new_buffer().stage_insert(&[v(1), v(20), v(0)]);
        table.merge(&mut e);

        let mut buffer = table.new_buffer();
        buffer.stage_insert(&[v(1), v(7), v(1)]);
        buffer.stage_insert(&[v(1), v(3), v(1)]);
        drop(buffer);

        assert!(parallel_insert(&mut table, &e));
        // Pending rows are coalesced before consulting the destination:
        // 20 - (7 - 3) = 16. Direct left-folding would produce 10.
        assert_eq!(&table.get_row(&[v(1)]).unwrap().vals[..2], &[v(1), v(16)]);
        assert_eq!(table.len(), 1);
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
            shard_hash(
                table.shard_data(),
                &[
                    Value::new(rng.random()),
                    Value::new(rng.random()),
                    Value::new(rng.random()),
                ],
                2,
            )
            .shard
        })
        .for_each(|id| *hist.entry(id).or_insert(0) += 1);
    assert!(hist.iter().all(|(_, count)| *count > 100), "{hist:?}");

    // Picking low numbers should all get shard 0.
    assert!((0..100_000).all(|x| table.shard_data().shard_id(x as u64) == ShardId::new(0)));
}
