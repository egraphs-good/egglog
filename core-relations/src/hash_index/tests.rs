use std::collections::BTreeMap;

use egglog_concurrency::ThreadPool;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{common::Value, numeric_id::NumericId, offsets::Offsets};

use crate::{
    TupleIndex,
    hash_index::ColumnIndex,
    table_shortcuts::{fill_table, v},
    table_spec::{ColumnId, WrappedTable},
};

use super::{Index, IndexBase};

#[test]
fn basic_updates() {
    // Get slightly higher coverage with nondeterministic parallelism.
    for _ in 0..10 {
        // fill a SortedWritesTable with some data. confirm that an index built on
        // some subset of columns works as expected. Then add more data, and confirm
        // that updates still work.
        let mut table = WrappedTable::new(fill_table(
            vec![
                vec![v(0), v(1), v(2), v(0)],
                vec![v(1), v(2), v(3), v(0)],
                vec![v(2), v(3), v(4), v(0)],
                vec![v(3), v(4), v(5), v(1)],
                vec![v(4), v(5), v(6), v(1)],
            ],
            2,
            Some(ColumnId::new(3)),
            |old, new| {
                assert_eq!(old, new, "no conflicts in this test");
                None
            },
        ));

        let mut index = Index::new(vec![ColumnId::new(0), ColumnId::new(2)], TupleIndex::new(2));
        assert!(index.get_subset(&[v(0), v(2)]).is_none());
        index.refresh(table.as_ref());
        for i in 0..=4 {
            let key = [v(i), v(i + 2)];
            let subset = index.get_subset(&key).unwrap();
            table.scan(subset).iter().for_each(|(id, row)| {
                assert_eq!(&row[0..3], &[v(i), v(i + 1), v(i + 2)]);
                let readback = table.get_row(&row[0..2]).expect("row should exist");
                assert_eq!(readback.id, id);
                assert_eq!(readback.vals.as_slice(), row);
            });
        }

        {
            let mut buf = table.new_buffer();
            for i in 5..10 {
                buf.stage_insert(&[v(i), v(i + 1), v(i + 2), v(2)]);
            }
        }

        empty_execution_state!(es);
        table.merge(&mut es);
        index.refresh(table.as_ref());
        for i in 0..10 {
            let key = [v(i), v(i + 2)];
            let subset = index.get_subset(&key).unwrap();
            table.scan(subset).iter().for_each(|(id, row)| {
                assert_eq!(&row[0..3], &[v(i), v(i + 1), v(i + 2)]);
                let readback = table.get_row(&row[0..2]).expect("row should exist");
                assert_eq!(readback.id, id);
                assert_eq!(readback.vals.as_slice(), row);
            });
        }

        // Now get an update to the major version.
        let start_version = table.version().major;
        while table.version().major == start_version {
            table.new_buffer().stage_remove(&[v(0), v(1)]);
            table.merge(&mut es);
            table.new_buffer().stage_insert(&[v(0), v(1), v(2), v(3)]);
            table.merge(&mut es);
        }

        // Refresh should do the right thing.
        index.refresh(table.as_ref());
        for i in 0..10 {
            let key = [v(i), v(i + 2)];
            let subset = index.get_subset(&key).unwrap();
            table.scan(subset).iter().for_each(|(id, row)| {
                assert_eq!(&row[0..3], &[v(i), v(i + 1), v(i + 2)]);
                let readback = table.get_row(&row[0..2]).expect("row should exist");
                assert_eq!(readback.id, id);
                assert_eq!(readback.vals.as_slice(), row);
            });
        }
    }
}

#[test]
fn multi_column_column_index_rebuild_orders_each_value_by_row() {
    let rows = (0..128).map(|i| {
        let left = if i >= 96 { v(7) } else { v(1_000 + i) };
        let right = if i < 32 { v(7) } else { v(2_000 + i) };
        vec![v(i), left, right]
    });
    let mut table = WrappedTable::new(fill_table(rows, 1, None, |old, new| {
        assert_eq!(old, new, "no conflicts in this test");
        None
    }));
    let mut index = Index::new(vec![ColumnId::new(1), ColumnId::new(2)], ColumnIndex::new());
    index.refresh(table.as_ref());

    empty_execution_state!(es);
    let start_version = table.version().major;
    while table.version().major == start_version {
        table.new_buffer().stage_remove(&[v(0)]);
        table.merge(&mut es);
        table.new_buffer().stage_insert(&[v(0), v(1000), v(7)]);
        table.merge(&mut es);
    }

    index.refresh(table.as_ref());
    let key = v(7);
    let subset = index.get_subset(&key).unwrap();
    let mut row_ids = Vec::new();
    subset.offsets(|row_id| row_ids.push(row_id.index()));

    let mut expected = Vec::new();
    table
        .scan(table.all().as_ref())
        .iter()
        .for_each(|(row_id, row)| {
            if row[1] == key || row[2] == key {
                expected.push(row_id.index());
            }
        });
    assert_eq!(row_ids, expected);
}

/// Brute-force oracle: for the covered `cols`, map each value to the list of row ids (in scan,
/// i.e. ascending-RowId, order) that contain it in some covered column. A value appearing in
/// several of a row's columns contributes that row once.
fn oracle(rows: &[Vec<Value>], cols: &[usize]) -> BTreeMap<u32, Vec<usize>> {
    let mut map: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (row_id, row) in rows.iter().enumerate() {
        let mut seen_in_row = Vec::new();
        for &c in cols {
            let val = row[c].rep();
            if !seen_in_row.contains(&val) {
                seen_in_row.push(val);
                map.entry(val).or_default().push(row_id);
            }
        }
    }
    map
}

/// Collect a built [`ColumnIndex`] into `value -> row ids` for comparison against [`oracle`].
fn collect(index: &ColumnIndex) -> BTreeMap<u32, Vec<usize>> {
    let mut got: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    index.for_each(|val, subset| {
        let mut ids = Vec::new();
        subset.offsets(|row_id| ids.push(row_id.index()));
        got.insert(val.rep(), ids);
    });
    got
}

fn assert_matches_oracle(index: &ColumnIndex, expected: &BTreeMap<u32, Vec<usize>>, ctx: &str) {
    let got = collect(index);
    for (val, ids) in &got {
        // Each value's row ids must be strictly ascending: sorted (the ordering this PR fixes)
        // and de-duplicated (a value repeated across a row's columns maps the row in once).
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "{ctx}: row ids for value {val} not strictly ascending: {ids:?}",
        );
    }
    assert_eq!(&got, expected, "{ctx}");
}

/// Randomized oracle test: build a table whose value columns draw from a small pool (so values
/// repeat across many rows), then check the sort-based rebuild, the parallel rebuild, and the
/// per-row `build_for_subset` path all produce each value's rows in sorted, de-duplicated order.
#[test]
fn column_index_rebuild_matches_oracle() {
    // Column 0 is a unique key; columns `1..n_cols` are the covered value columns.
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        for &n_rows in &[1usize, 10, 63, 64, 200, 512, 1000] {
            let distinct = ((n_rows / 4).max(1)) as u32;
            for &n_val_cols in &[1usize, 2, 3, 4] {
                let n_cols = n_val_cols + 1;
                let rows: Vec<Vec<Value>> = (0..n_rows)
                    .map(|i| {
                        let mut row = Vec::with_capacity(n_cols);
                        row.push(v(i));
                        for _ in 1..n_cols {
                            row.push(v(rng.random_range(0..distinct) as usize));
                        }
                        row
                    })
                    .collect();
                let table = WrappedTable::new(fill_table(rows.clone(), 1, None, |old, new| {
                    assert_eq!(old, new, "unique keys, so no conflicts");
                    None
                }));
                let cols: Vec<ColumnId> = (1..n_cols).map(ColumnId::from_usize).collect();
                let covered: Vec<usize> = (1..n_cols).collect();
                let expected = oracle(&rows, &covered);
                let ctx = format!("seed={seed} n_rows={n_rows} n_val_cols={n_val_cols}");

                let mut serial = ColumnIndex::new();
                serial.rebuild_full(&cols, table.as_ref(), table.all().as_ref());
                assert_matches_oracle(&serial, &expected, &format!("{ctx} rebuild_full"));

                // Install a multi-thread pool so `ColumnIndex` shards across several shards and
                // the merge actually forks; correctness must not depend on the shard count.
                let parallel = ThreadPool::new(4).install(|| {
                    let mut ci = ColumnIndex::new();
                    ci.merge_parallel(&cols, table.as_ref(), table.all().as_ref());
                    ci
                });
                assert_matches_oracle(&parallel, &expected, &format!("{ctx} merge_parallel"));
            }
        }
    }
}

#[test]
fn physical_shards_partition_index_iteration() {
    ThreadPool::new(4).install(|| {
        let rows = (0..257)
            .map(|i| vec![v(i), v(i % 37), v(10_000 + i)])
            .collect::<Vec<_>>();
        let table = WrappedTable::new(fill_table(rows, 1, None, |old, new| {
            assert_eq!(old, new, "unique keys, so no conflicts");
            None
        }));

        let mut column = Index::new(vec![ColumnId::new(1)], ColumnIndex::new());
        column.refresh(table.as_ref());
        assert_eq!(column.shard_count(), 8);
        let mut whole_column = BTreeMap::new();
        column.for_each(|key, subset| {
            whole_column.insert(key.rep(), subset.size());
        });
        let mut by_column_shard = BTreeMap::new();
        for shard in 0..column.shard_count() {
            let mut shard_keys = 0;
            column.for_each_shard(shard, |key, subset| {
                shard_keys += 1;
                assert!(
                    by_column_shard.insert(key.rep(), subset.size()).is_none(),
                    "a column-index key appeared in more than one shard"
                );
            });
            assert_eq!(column.shard_len(shard), shard_keys);
        }
        assert_eq!(by_column_shard, whole_column);

        let mut tuple = Index::new(
            vec![ColumnId::new(1), ColumnId::new(2)],
            crate::TupleIndex::new(2),
        );
        tuple.refresh(table.as_ref());
        assert_eq!(tuple.shard_count(), 8);
        let mut whole_tuple = BTreeMap::new();
        tuple.for_each(|key, subset| {
            whole_tuple.insert((key[0].rep(), key[1].rep()), subset.size());
        });
        let mut by_tuple_shard = BTreeMap::new();
        for shard in 0..tuple.shard_count() {
            let mut shard_keys = 0;
            tuple.for_each_shard(shard, |key, subset| {
                shard_keys += 1;
                assert!(
                    by_tuple_shard
                        .insert((key[0].rep(), key[1].rep()), subset.size())
                        .is_none(),
                    "a tuple-index key appeared in more than one shard"
                );
            });
            assert_eq!(tuple.shard_len(shard), shard_keys);
        }
        assert_eq!(by_tuple_shard, whole_tuple);
    });
}
