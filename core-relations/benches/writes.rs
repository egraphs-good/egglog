use std::sync::Arc;

use divan::{Bencher, counter::ItemsCount};
use egglog_concurrency::{ThreadPool, scope};
use egglog_core_relations::{Database, SortedWritesTable, Table, Value};
use egglog_numeric_id::NumericId;
use rand::{Rng, rng};

fn main() {
    divan::main()
}

enum Operation<const KEYS: usize, const COLS: usize> {
    Insert([Value; COLS]),
    Remove([Value; KEYS]),
}
impl<const KEYS: usize, const COLS: usize> Operation<KEYS, COLS> {
    fn key(&self) -> &[Value] {
        match self {
            Operation::Insert(row) => &row[..KEYS],
            Operation::Remove(key) => key,
        }
    }
}

fn random_value(rng: &mut impl Rng) -> Value {
    // We exclude u32::MAX as it's the special "stale" value.
    Value::new(rng.random_range(0..u32::MAX))
}

fn random_row<const C: usize>(rng: &mut impl Rng) -> [Value; C] {
    let mut row = [Value::new(0); C];
    for v in row.iter_mut() {
        *v = random_value(rng);
    }
    row
}

fn generate_workload<const K: usize, const C: usize>(
    n: usize,
    insert_pct: f64,
    collision_pct: f64,
) -> Vec<Operation<K, C>> {
    let mut rng = rng();
    let mut ops = Vec::<Operation<K, C>>::with_capacity(n);
    for _ in 0..n {
        if !rng.random_bool(insert_pct) && !ops.is_empty() {
            // All removals need to be a collision. We could add a few in here
            // that aren't but it's not realistic because all egglog removals
            // come from a previous read of the table.
            let key = ops[rng.random_range(0..ops.len())].key();
            ops.push(Operation::Remove(key.try_into().unwrap()));
        } else if rng.random_bool(collision_pct) && !ops.is_empty() {
            let key = ops[rng.random_range(0..ops.len())].key();
            let mut row = random_row::<C>(&mut rng);
            for (dst, src) in row.iter_mut().zip(key.iter()) {
                *dst = *src;
            }
            ops.push(Operation::Insert(row));
        } else {
            ops.push(Operation::Insert(random_row::<C>(&mut rng)));
        }
    }
    ops
}

#[divan::bench(consts = [1, 2, 4, 8, 12, 16], sample_count=25)]
fn parallel_insert<const N: usize>(bench: Bencher) {
    const WORKLOAD_SIZE: usize = 4 << 20;
    bench_workload(
        bench,
        generate_workload::<3, 5>(WORKLOAD_SIZE, 1.0, 0.05),
        1,
        N,
    )
}

#[divan::bench(consts = [1, 2, 4, 8, 12, 16], sample_count=25)]
fn parallel_insert_merge2<const N: usize>(bench: Bencher) {
    const WORKLOAD_SIZE: usize = 4 << 20;
    bench_workload(
        bench,
        generate_workload::<3, 5>(WORKLOAD_SIZE, 1.0, 0.05),
        2,
        N,
    )
}

#[divan::bench(consts = [1, 2, 4, 8, 12, 16])]
fn parallel_insert_remove_with_collisions<const N: usize>(bench: Bencher) {
    const WORKLOAD_SIZE: usize = 1 << 20;
    bench_workload(
        bench,
        generate_workload::<3, 5>(WORKLOAD_SIZE, 0.75, 0.15),
        1,
        N,
    )
}

#[divan::bench(consts = [1, 2, 4, 8, 12, 16], sample_count = 10)]
fn parallel_delete_only<const N: usize>(bench: Bencher) {
    const TABLE_SIZE: usize = 1 << 20;
    // Stay over the production parallel cutoff (400K) but under the 50%
    // compaction threshold, so this benchmark isolates do_delete.
    const REMOVALS: usize = 7 << 16;
    bench_delete_only::<N>(bench, TABLE_SIZE, REMOVALS);
}

#[divan::bench(consts = [1, 2, 4, 8, 12, 16], sample_count = 5)]
fn parallel_delete_only_large<const N: usize>(bench: Bencher) {
    const TABLE_SIZE: usize = 4 << 20;
    const REMOVALS: usize = 7 << 18;
    bench_delete_only::<N>(bench, TABLE_SIZE, REMOVALS);
}

fn bench_delete_only<const N: usize>(bench: Bencher, table_size: usize, removals: usize) {
    let rows = Arc::new(
        (0..table_size)
            .map(|i| {
                [
                    Value::new(i as u32),
                    Value::new(i.wrapping_mul(0x9e3779b1) as u32),
                    Value::new(i.wrapping_mul(0x85ebca6b) as u32),
                    Value::new(i.wrapping_mul(0xc2b2ae35) as u32),
                    Value::new(i.wrapping_mul(0x27d4eb2f) as u32),
                ]
            })
            .collect::<Vec<_>>(),
    );
    let pool = Arc::new(ThreadPool::new(N));
    bench
        .with_inputs({
            let pool = pool.clone();
            let rows = rows.clone();
            move || {
                pool.install(|| {
                    let db = Database::default();
                    let mut table = new_table::<3, 5>();
                    stage_inserts(&table, &rows);
                    db.with_execution_state(|es| table.merge(es));
                    stage_removals(&table, &rows[..removals]);
                    (db, table)
                })
            }
        })
        .input_counter(move |_| ItemsCount::new(removals))
        .bench_refs(move |input| {
            let (db, table) = input;
            pool.install(|| db.with_execution_state(|es| table.merge(es)));
        });
}

fn new_table<const K: usize, const C: usize>() -> SortedWritesTable {
    SortedWritesTable::new(
        K,
        C,
        None,
        vec![],
        Box::new(|_, old, new, out: &mut Vec<Value>| {
            out.extend_from_slice(new);
            old != new
        }),
    )
}

fn stage_inserts<const C: usize>(table: &SortedWritesTable, rows: &[[Value; C]]) {
    scope(|scope| {
        for rows in rows.chunks(8 * 1024) {
            let mut buf = table.new_buffer();
            scope.spawn(move |_| {
                for row in rows {
                    buf.stage_insert(row);
                }
            });
        }
    });
}

fn stage_removals<const C: usize>(table: &SortedWritesTable, rows: &[[Value; C]]) {
    scope(|scope| {
        for rows in rows.chunks(8 * 1024) {
            let mut buf = table.new_buffer();
            scope.spawn(move |_| {
                for row in rows {
                    buf.stage_remove(&row[..3]);
                }
            });
        }
    });
}

fn bench_workload<const K: usize, const C: usize>(
    bench: Bencher,
    workload: Vec<Operation<K, C>>,
    n_merges: usize,
    threads: usize,
) {
    const BATCH_SIZE: usize = 1024;
    let epoch_size = workload.len().next_multiple_of(n_merges) / n_merges;
    let pool = Arc::new(ThreadPool::new(threads));
    let workload_size = workload.len();
    bench
        .with_inputs({
            let pool = pool.clone();
            move || pool.install(|| (Database::default(), new_table::<K, C>()))
        })
        .input_counter(move |_| ItemsCount::new(workload_size))
        .bench_refs(move |input| {
            let (db, table) = input;
            pool.install(|| {
                for outer in workload.chunks(epoch_size) {
                    scope(|scope| {
                        for batch in outer.chunks(BATCH_SIZE) {
                            let mut buf = table.new_buffer();
                            scope.spawn(move |_| {
                                for op in batch {
                                    match op {
                                        Operation::Insert(row) => buf.stage_insert(row),
                                        Operation::Remove(key) => buf.stage_remove(key),
                                    }
                                }
                            });
                        }
                    });
                    db.with_execution_state(|es| table.merge(es));
                }
            })
        })
}
