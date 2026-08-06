//! Fixed-arity performance comparison for the variable-arity SequenceTable.
//!
//! Merge-only cases stage mutations outside the timer; end-to-end cases include
//! producer staging. Rebuild includes row scanning and canonicalization for
//! both implementations because that scan is part of the compared operation.

use std::sync::Arc;

use divan::{Bencher, counter::ItemsCount};
use egglog_concurrency::{ThreadPool, current_num_threads, scope};
use egglog_core_relations::{
    Database, MutationBuffer, OffsetRange, RowId, SequenceTable, SortedWritesTable, SubsetRef,
    Table, TableChange, Value, ValueRebuilder,
};
use egglog_numeric_id::NumericId;

fn main() {
    divan::main()
}

const ARITY: usize = 5;
const INSERT_ROWS: usize = 1 << 20;
const DELETE_TABLE_ROWS: usize = 1 << 20;
const DELETE_ROWS: usize = 7 << 16;
const COMPACT_ROWS: usize = 3 << 18;
const REBUILD_ROWS: usize = 1 << 20;
const STAGING_BATCH: usize = 8 * 1024;
const REBUILD_BATCH: usize = 2 * 1024;
const INDEX_ROWS: usize = 1 << 18;
const INDEX_WIDTH: usize = 16;
const INDEXED_REBUILD_ROWS: usize = 1 << 20;

trait BenchTable: Sized + Send + Sync + 'static {
    fn new() -> Self;
    fn new_buffer(&self) -> Box<dyn MutationBuffer>;
    fn merge(&mut self) -> TableChange;
    fn rebuild(&mut self, rebuilder: &BenchRebuilder) -> TableChange;
}

struct Sorted {
    db: Database,
    table: SortedWritesTable,
}

impl BenchTable for Sorted {
    fn new() -> Self {
        Self {
            db: Database::default(),
            table: SortedWritesTable::new(ARITY, ARITY, None, vec![], Box::new(|_, _, _, _| false)),
        }
    }

    fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        self.table.new_buffer()
    }

    fn merge(&mut self) -> TableChange {
        self.db
            .with_execution_state(|state| self.table.merge(state))
    }

    fn rebuild(&mut self, rebuilder: &BenchRebuilder) -> TableChange {
        let physical_rows = self.table.version().minor.index();
        let table = &self.table;
        let starts = (0..physical_rows)
            .step_by(REBUILD_BATCH)
            .collect::<Vec<_>>();
        let stage = |starts: &[usize]| {
            for &start in starts {
                let end = (start + REBUILD_BATCH).min(physical_rows);
                let subset = OffsetRange::new(RowId::from_usize(start), RowId::from_usize(end));
                let mut buffer = table.new_buffer();
                let mut rebuilt = Vec::with_capacity(ARITY);
                table.scan_generic(SubsetRef::Dense(subset), |_, row| {
                    rebuilt.extend_from_slice(row);
                    if rebuilder.rebuild_slice(&mut rebuilt) {
                        buffer.stage_remove(row);
                        buffer.stage_insert(&rebuilt);
                    }
                    rebuilt.clear();
                });
            }
        };
        let workers = current_num_threads();
        let tasks = workers.saturating_mul(parallel_tasks_per_thread()).max(1);
        if tasks == 1 {
            stage(&starts);
        } else {
            let chunk_len = starts.len().div_ceil(tasks).max(1);
            scope(|scope| {
                let stage = &stage;
                for starts in starts.chunks(chunk_len) {
                    scope.spawn(move |_| {
                        stage(starts);
                    });
                }
            });
        }
        self.merge()
    }
}

struct Sequence(SequenceTable);

impl BenchTable for Sequence {
    fn new() -> Self {
        Self(SequenceTable::new())
    }

    fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        self.0.new_buffer()
    }

    fn merge(&mut self) -> TableChange {
        self.0.merge()
    }

    fn rebuild(&mut self, rebuilder: &BenchRebuilder) -> TableChange {
        self.0.rebuild_values(rebuilder)
    }
}

trait SequenceIndexMode: Send + Sync + 'static {
    fn new_table() -> SequenceTable;
}

struct WithoutValueIndex;

impl SequenceIndexMode for WithoutValueIndex {
    fn new_table() -> SequenceTable {
        SequenceTable::new()
    }
}

struct WithValueIndex;

impl SequenceIndexMode for WithValueIndex {
    fn new_table() -> SequenceTable {
        SequenceTable::new_indexed(Box::new(|row, visit| row.iter().copied().for_each(visit)))
    }
}

struct BenchRebuilder;

impl ValueRebuilder for BenchRebuilder {
    fn rebuild_val(&self, value: Value) -> Value {
        Value::new(value.rep() ^ 0x8000_0000)
    }
}

#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn insert_merge_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(INSERT_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let table = T::new();
                    stage_inserts(&table, &rows);
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(INSERT_ROWS))
        .bench_refs(move |table| {
            pool.install(|| divan::black_box(table.merge()));
        });
}

#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn insert_end_to_end_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(INSERT_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let pool = pool.clone();
            move || pool.install(T::new)
        })
        .input_counter(|_| ItemsCount::new(INSERT_ROWS))
        .bench_refs(move |table| {
            pool.install(|| {
                stage_inserts(table, &rows);
                divan::black_box(table.merge())
            });
        });
}

#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn delete_merge_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(DELETE_TABLE_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let mut table = T::new();
                    stage_inserts(&table, &rows);
                    divan::black_box(table.merge());
                    stage_removals(&table, &rows[..DELETE_ROWS]);
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(DELETE_ROWS))
        .bench_refs(move |table| {
            pool.install(|| divan::black_box(table.merge()));
        });
}

#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn delete_end_to_end_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(DELETE_TABLE_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let mut table = T::new();
                    stage_inserts(&table, &rows);
                    divan::black_box(table.merge());
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(DELETE_ROWS))
        .bench_refs(move |table| {
            pool.install(|| {
                stage_removals(table, &rows[..DELETE_ROWS]);
                divan::black_box(table.merge())
            });
        });
}

#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn rebuild_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(REBUILD_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let mut table = T::new();
                    stage_inserts(&table, &rows);
                    divan::black_box(table.merge());
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(REBUILD_ROWS))
        .bench_refs(move |table| {
            pool.install(|| divan::black_box(table.rebuild(&BenchRebuilder)));
        });
}

/// Time the second full rebuild, which crosses both tables' stale-row
/// threshold and therefore includes compaction of the backing storage.
#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn rebuild_with_compaction_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(REBUILD_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let mut table = T::new();
                    stage_inserts(&table, &rows);
                    divan::black_box(table.merge());
                    divan::black_box(table.rebuild(&BenchRebuilder));
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(REBUILD_ROWS))
        .bench_refs(move |table| {
            pool.install(|| divan::black_box(table.rebuild(&BenchRebuilder)));
        });
}

/// Delete enough rows to include stale-storage compaction in the timed merge.
#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [Sorted, Sequence],
    sample_count = 5
)]
fn delete_and_compact_fixed_arity<const THREADS: usize, T: BenchTable>(bench: Bencher) {
    let rows = Arc::new(make_rows(DELETE_TABLE_ROWS));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let mut table = T::new();
                    stage_inserts(&table, &rows);
                    divan::black_box(table.merge());
                    stage_removals(&table, &rows[..COMPACT_ROWS]);
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(COMPACT_ROWS))
        .bench_refs(move |table| {
            pool.install(|| divan::black_box(table.merge()));
        });
}

/// Measure the eager occurrence-index maintenance added to a wide sequence
/// merge. The unindexed variant makes the marginal index cost visible, while
/// thread counts show whether collection, radix sorting, and shard population
/// scale together.
#[divan::bench(
    consts = [1, 2, 4, 8, 12],
    types = [WithoutValueIndex, WithValueIndex],
    sample_count = 5
)]
fn insert_merge_wide_indexed_sequences<const THREADS: usize, M: SequenceIndexMode>(bench: Bencher) {
    let rows = Arc::new(make_wide_rows(INDEX_ROWS, INDEX_WIDTH));
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let rows = rows.clone();
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let table = M::new_table();
                    stage_sequence_inserts(&table, &rows);
                    table
                })
            }
        })
        .input_counter(|_| ItemsCount::new(INDEX_ROWS * INDEX_WIDTH))
        .bench_refs(move |table| {
            pool.install(|| divan::black_box(table.merge()));
        });
}

struct IndexedRebuildInput {
    db: Database,
    table: SequenceTable,
    candidates: egglog_core_relations::Subset,
}

/// One changed value selects a genuinely sparse 1/256 of the table.
#[divan::bench(consts = [1, 2, 4, 8, 12], sample_count = 5)]
fn sparse_indexed_sequence_rebuild<const THREADS: usize>(bench: Bencher) {
    indexed_sequence_rebuild::<THREADS, 256>(bench);
}

/// A high-fanout value selects half the table and crosses the parallel rebuild
/// cutoff, exercising the same index path under a large candidate union.
#[divan::bench(consts = [1, 2, 4, 8, 12], sample_count = 5)]
fn high_fanout_indexed_sequence_rebuild<const THREADS: usize>(bench: Bencher) {
    indexed_sequence_rebuild::<THREADS, 2>(bench);
}

fn indexed_sequence_rebuild<const THREADS: usize, const DIVISOR: usize>(bench: Bencher) {
    let pool = Arc::new(ThreadPool::new(THREADS));
    bench
        .with_inputs({
            let pool = pool.clone();
            move || {
                pool.install(|| {
                    let db = Database::new();
                    let mut table = SequenceTable::new_with_values_and_index(
                        1,
                        Box::new(|_, _, _, _| false),
                        Box::new(|key, visit| key.iter().copied().for_each(visit)),
                    );
                    scope(|scope| {
                        for start in (0..INDEXED_REBUILD_ROWS).step_by(STAGING_BATCH) {
                            let mut writes = table.new_buffer();
                            scope.spawn(move |_| {
                                for index in
                                    start..(start + STAGING_BATCH).min(INDEXED_REBUILD_ROWS)
                                {
                                    writes.stage_insert(&[
                                        Value::from_usize(index % DIVISOR),
                                        Value::from_usize(DIVISOR + index),
                                        Value::from_usize(DIVISOR + INDEXED_REBUILD_ROWS + index),
                                    ]);
                                }
                            });
                        }
                    });
                    db.with_execution_state(|state| {
                        divan::black_box(table.merge_with_state(state));
                    });
                    let candidates = table
                        .owned_rows_for_indexed_value(Value::from_usize(0))
                        .expect("the selected value occurs in the benchmark table");
                    IndexedRebuildInput {
                        db,
                        table,
                        candidates,
                    }
                })
            }
        })
        .input_counter(|input| ItemsCount::new(input.candidates.size()))
        .bench_refs(move |input| {
            pool.install(|| {
                input.db.with_execution_state(|state| {
                    divan::black_box(input.table.rebuild_full_rows_subset_with_state(
                        input.candidates.as_ref(),
                        &|row, rebuilt| {
                            rebuilt.extend_from_slice(row);
                            rebuilt[0] = Value::new(rebuilt[0].rep() ^ 0x4000_0000);
                            true
                        },
                        state,
                    ))
                });
            });
        });
}

fn make_rows(count: usize) -> Vec<[Value; ARITY]> {
    let value = |raw: usize| Value::new(raw as u32 & 0x3fff_ffff);
    (0..count)
        .map(|index| {
            [
                value(index),
                value(index.wrapping_mul(0x9e37_79b1)),
                value(index.wrapping_mul(0x85eb_ca6b)),
                value(index.wrapping_mul(0xc2b2_ae35)),
                value(index.wrapping_mul(0x27d4_eb2f)),
            ]
        })
        .collect()
}

fn make_wide_rows(count: usize, width: usize) -> Vec<Vec<Value>> {
    (0..count)
        .map(|row| {
            (0..width)
                .map(|column| {
                    Value::new(
                        row.wrapping_mul(0x9e37_79b1)
                            .wrapping_add(column.wrapping_mul(0x85eb_ca6b))
                            as u32
                            & 0x3fff_ffff,
                    )
                })
                .collect()
        })
        .collect()
}

fn stage_sequence_inserts(table: &SequenceTable, rows: &[Vec<Value>]) {
    scope(|scope| {
        for rows in rows.chunks(STAGING_BATCH) {
            let mut buffer = table.new_buffer();
            scope.spawn(move |_| {
                for row in rows {
                    buffer.stage_insert(row);
                }
            });
        }
    });
}

fn stage_inserts<T: BenchTable>(table: &T, rows: &[[Value; ARITY]]) {
    scope(|scope| {
        for rows in rows.chunks(STAGING_BATCH) {
            let mut buffer = table.new_buffer();
            scope.spawn(move |_| {
                for row in rows {
                    buffer.stage_insert(row);
                }
            });
        }
    });
}

fn stage_removals<T: BenchTable>(table: &T, rows: &[[Value; ARITY]]) {
    scope(|scope| {
        for rows in rows.chunks(STAGING_BATCH) {
            let mut buffer = table.new_buffer();
            scope.spawn(move |_| {
                for row in rows {
                    buffer.stage_remove(row);
                }
            });
        }
    });
}

fn parallel_tasks_per_thread() -> usize {
    std::env::var("EGGLOG_PARALLEL_TASKS_PER_THREAD")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
        .max(1)
}
