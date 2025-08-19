use core_relations::{Database, SortedWritesTable, Table, Value};
use divan::{counter::ItemsCount, Bencher};
use numeric_id::NumericId;
use rand::{thread_rng, Rng};
use rayon::{
    iter::{ParallelBridge, ParallelIterator},
    ThreadPoolBuilder,
};

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
    Value::new(rng.gen_range(0..u32::MAX))
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
    let mut rng = thread_rng();
    let mut ops = Vec::<Operation<K, C>>::with_capacity(n);
    for _ in 0..n {
        if !rng.gen_bool(insert_pct) && !ops.is_empty() {
            // All removals need to be a collision. We could add a few in here
            // that aren't but it's not realistic because all egglog removals
            // come from a previous read of the table.
            let key = ops[rng.gen_range(0..ops.len())].key();
            ops.push(Operation::Remove(key.try_into().unwrap()));
        } else if rng.gen_bool(collision_pct) && !ops.is_empty() {
            let key = ops[rng.gen_range(0..ops.len())].key();
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

#[divan::bench(consts = [1, 2, 4, 8, 16], sample_count=25)]
fn parallel_insert<const N: usize>(bench: Bencher) {
    const WORKLOAD_SIZE: usize = 4 << 20;
    bench_workload(
        bench,
        generate_workload::<3, 5>(WORKLOAD_SIZE, 1.0, 0.05),
        1,
        N,
    )
}

#[divan::bench(consts = [1, 2, 4, 8, 16], sample_count=25)]
fn parallel_insert_merge2<const N: usize>(bench: Bencher) {
    const WORKLOAD_SIZE: usize = 4 << 20;
    bench_workload(
        bench,
        generate_workload::<3, 5>(WORKLOAD_SIZE, 1.0, 0.05),
        2,
        N,
    )
}

#[divan::bench(consts = [1, 2, 4, 8, 16])]
fn parallel_insert_remove_with_collisions<const N: usize>(bench: Bencher) {
    const WORKLOAD_SIZE: usize = 1 << 20;
    bench_workload(
        bench,
        generate_workload::<3, 5>(WORKLOAD_SIZE, 0.75, 0.15),
        1,
        N,
    )
}

fn bench_workload<const K: usize, const C: usize>(
    bench: Bencher,
    workload: Vec<Operation<K, C>>,
    n_merges: usize,
    threads: usize,
) {
    const BATCH_SIZE: usize = 1024;
    let epoch_size = workload.len().next_multiple_of(n_merges) / n_merges;
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let workload_size = workload.len();
    bench
        .with_inputs(|| {
            (
                Database::default(),
                SortedWritesTable::new(
                    K,
                    C,
                    None,
                    vec![],
                    Box::new(|_, old, new, out: &mut Vec<Value>| {
                        out.extend_from_slice(new);
                        old != new
                    }),
                ),
            )
        })
        .input_counter(move |_| ItemsCount::new(workload_size))
        .bench_values(|(db, mut table)| {
            pool.install(|| {
                for outer in workload.chunks(epoch_size) {
                    outer.chunks(BATCH_SIZE).par_bridge().for_each(|batch| {
                        let mut buf = table.new_buffer();
                        for op in batch {
                            match op {
                                Operation::Insert(row) => buf.stage_insert(row),
                                Operation::Remove(key) => buf.stage_remove(key),
                            }
                        }
                    });
                    db.with_execution_state(|es| table.merge(es));
                }
            })
        })
}
