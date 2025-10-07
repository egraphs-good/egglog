use std::{
    cmp,
    sync::{Arc, Mutex, RwLock},
    thread,
};

use divan::{Bencher, counter::ItemsCount};
use egglog_concurrency::{Notification, ReadOptimizedLock};
use egglog_union_find::{UnionFind, concurrent};
use rand::{Rng, seq::SliceRandom};

fn main() {
    divan::main()
}

#[derive(Copy, Clone)]
enum Operation {
    Union(usize, usize),
    Find(usize),
}

const LENGTHS: [usize; 4] = [1000, 100000, 1000000, 4000000];

fn prepare_operations_random(n_items: usize) -> (Vec<Operation>, Vec<Operation>) {
    let mut rng = rand::rng();
    let mut seed_operations = Vec::new();
    let mut operations = Vec::new();
    for i in 0..(n_items / 4) {
        if rng.random_bool(0.9) {
            seed_operations.push(Operation::Union(i, rng.random_range(0..n_items)));
        } else {
            operations.push(Operation::Union(i, rng.random_range(0..n_items)));
        }
    }
    for _ in 0..(2 * n_items) {
        operations.push(Operation::Find(rng.random_range(0..n_items)));
    }
    operations.shuffle(&mut rng);
    (seed_operations, operations)
}

fn prepare_operations_local(n_items: usize) -> (Vec<Operation>, Vec<Operation>) {
    let mut rng = rand::rng();
    let mut seed_operations = Vec::new();
    let mut operations = Vec::new();
    for i in 0..n_items {
        // Only create unions for 15% of the items.
        if rng.random_bool(0.85) {
            continue;
        }
        let rhs = cmp::min(i + rng.random_range(0..1000), n_items);
        if rng.random_bool(0.9) {
            seed_operations.push(Operation::Union(i, rhs));
        } else {
            operations.push(Operation::Union(i, rhs));
        }
    }
    for _ in 0..(2 * n_items) {
        operations.push(Operation::Find(rng.random_range(0..n_items)));
    }
    operations.sort_by_key(|op| match op {
        Operation::Union(a, _) => *a,
        Operation::Find(a) => *a,
    });
    (seed_operations, operations)
}

fn find_only_random(n_items: usize) -> (Vec<Operation>, Vec<usize>) {
    let mut rng = rand::rng();
    let mut seed_operations = Vec::new();
    let mut operations = Vec::new();
    for i in 0..(n_items / 4) {
        let rhs = rng.random_range(0..n_items);
        seed_operations.push(Operation::Union(i, rhs));
        seed_operations.push(Operation::Find(i));
        seed_operations.push(Operation::Find(rhs));
    }
    for _ in 0..(2 * n_items) {
        operations.push(rng.random_range(0..n_items));
    }
    operations.shuffle(&mut rng);
    (seed_operations, operations)
}

#[divan::bench_group(sample_count = 50)]
mod find_only {
    use super::*;

    const N_ITEMS: usize = 1_000_000;

    #[divan::bench(
        consts = [32, 128, 256, 1024],
        types = [
            Arc<ReadOptimizedLock<UnionFind<usize>>>,
            Arc<RwLock<UnionFind<usize>>>,
            concurrent::UnionFind<usize>
        ])]
    fn parallel_naive<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        naive_find::<UF>(bench, N, find_only_random)
    }

    fn naive_find<UF: ConcurrentUf>(
        bench: Bencher,
        batch_size: usize,
        prepare_operations: fn(usize) -> (Vec<Operation>, Vec<usize>),
    ) {
        bench
            .with_inputs(|| {
                let uf = UF::with_capacity(N_ITEMS);
                let (seed, to_find) = prepare_operations(N_ITEMS);
                uf.apply_ops(&seed);
                let ops_size = to_find.len();
                (uf, to_find, ops_size)
            })
            .input_counter(|(_, _, ops_size)| ItemsCount::new(*ops_size))
            .bench_local_values(|(uf, ops, _)| {
                rayon::scope(|s| {
                    for chunk in ops.chunks(batch_size) {
                        let uf = &uf;
                        s.spawn(move |_| {
                            uf.find_many_naive(chunk);
                        });
                    }
                });
            })
    }

    #[divan::bench]
    fn best_serial(bench: Bencher) {
        bench
            .with_inputs(|| {
                let mut uf = UnionFind::default();
                uf.find(N_ITEMS);
                let (seed, to_find) = find_only_random(N_ITEMS);
                for op in seed {
                    match op {
                        Operation::Union(a, b) => {
                            uf.union(a, b);
                        }
                        Operation::Find(a) => {
                            uf.find(a);
                        }
                    }
                }
                let ops_size = to_find.len();
                (uf, to_find, ops_size)
            })
            .input_counter(|(_, _, ops_size)| ItemsCount::new(*ops_size))
            .bench_local_values(|(mut uf, ops, _)| {
                for op in ops {
                    uf.find(op);
                }
            })
    }
}

#[divan::bench_group(sample_count = 50)]
mod rayon_work_stealing {
    use super::*;

    #[divan::bench(consts = LENGTHS)]
    fn random<const N: usize>(bench: Bencher) {
        run_using_rayon(bench, N, prepare_operations_random);
    }

    #[divan::bench(consts = LENGTHS)]
    fn local<const N: usize>(bench: Bencher) {
        run_using_rayon(bench, N, prepare_operations_local);
    }

    fn run_using_rayon(
        bench: Bencher,
        n_items: usize,
        prepare_operations: fn(usize) -> (Vec<Operation>, Vec<Operation>),
    ) {
        bench
            .with_inputs(|| {
                let uf = concurrent::UnionFind::<usize>::with_capacity(n_items);
                let (seed_operations, operations) = prepare_operations(n_items);
                for op in seed_operations {
                    match op {
                        Operation::Union(a, b) => {
                            uf.union(a, b);
                        }
                        Operation::Find(a) => {
                            uf.find(a);
                        }
                    }
                }
                let chunk_size = 4096;
                let ops_size = operations.len();
                let chunked_ops = operations
                    .chunks(chunk_size)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>();
                (uf, chunked_ops, ops_size)
            })
            .input_counter(|(_, _, ops_size)| ItemsCount::new(*ops_size))
            .bench_local_values(|(uf, ops, _)| {
                rayon::scope(|s| {
                    for chunk in ops {
                        let uf = &uf;
                        s.spawn(move |_| {
                            for op in chunk {
                                match op {
                                    Operation::Union(a, b) => {
                                        uf.union(a, b);
                                    }
                                    Operation::Find(a) => {
                                        uf.find(a);
                                    }
                                }
                            }
                        });
                    }
                });
            });
    }
}

#[divan::bench_group(sample_count = 50)]
mod local {
    use super::*;

    #[divan::bench(consts = LENGTHS)]
    fn serial<const N: usize>(bench: Bencher) {
        uf_serial(bench, N, prepare_operations_local);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_1threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 1, prepare_operations_local);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_2threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 2, prepare_operations_local);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_4threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 4, prepare_operations_local);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_8threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 8, prepare_operations_local);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_16threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 16, prepare_operations_local);
    }
}

#[divan::bench_group(sample_count = 50)]
mod random {
    use super::*;

    #[divan::bench(consts = LENGTHS)]
    fn serial<const N: usize>(bench: Bencher) {
        uf_serial(bench, N, prepare_operations_random);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_1threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 1, prepare_operations_random);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_2threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 2, prepare_operations_random);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_4threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 4, prepare_operations_random);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_8threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 8, prepare_operations_random);
    }

    #[divan::bench(consts = LENGTHS, types = [concurrent::UnionFind<usize>, Arc<Mutex<UnionFind<usize>>>])]
    fn concurrent_16threads<const N: usize, UF: ConcurrentUf>(bench: Bencher) {
        uf_parallel::<UF>(bench, N, 16, prepare_operations_random);
    }
}

fn uf_parallel<UF: ConcurrentUf>(
    bench: Bencher,
    n_items: usize,
    n_threads: usize,
    prepare_operations: fn(usize) -> (Vec<Operation>, Vec<Operation>),
) {
    bench
        .with_inputs(|| {
            let uf = UF::with_capacity(n_items);
            let (seed_operations, mut operations) = prepare_operations(n_items);
            for op in seed_operations {
                match op {
                    Operation::Union(a, b) => {
                        uf.union(a, b);
                    }
                    Operation::Find(a) => {
                        uf.find(a);
                    }
                }
            }
            let n = Arc::new(Notification::new());
            let chunk_size = operations.len() / n_threads;
            operations.truncate(chunk_size * n_threads);
            let threads: Vec<_> = operations
                .chunks(chunk_size)
                .map(|chunk| {
                    let n = n.clone();
                    let uf = uf.clone();
                    let chunk = chunk.to_vec();
                    thread::spawn(move || {
                        n.wait();
                        for op in chunk {
                            match op {
                                Operation::Union(a, b) => {
                                    uf.union(a, b);
                                }
                                Operation::Find(a) => {
                                    uf.find(a);
                                }
                            }
                        }
                    })
                })
                .collect();
            (n, threads, operations.len())
        })
        .input_counter(|(_, _, n_ops)| ItemsCount::new(*n_ops))
        .bench_local_values(|(n, threads, _)| {
            n.notify();
            threads.into_iter().for_each(|t| t.join().unwrap());
        });
}

fn uf_serial(
    bench: Bencher,
    n_items: usize,
    prepare_operations: fn(usize) -> (Vec<Operation>, Vec<Operation>),
) {
    bench
        .with_inputs(|| {
            let mut uf = UnionFind::<usize>::default();
            // Initialize the UF with N elements.
            uf.find(n_items);
            let (seed_operations, operations) = prepare_operations(n_items);
            for op in seed_operations {
                match op {
                    Operation::Union(a, b) => {
                        uf.union(a, b);
                    }
                    Operation::Find(a) => {
                        uf.find(a);
                    }
                }
            }
            (uf, operations)
        })
        .input_counter(|(_, ops)| ItemsCount::new(ops.len()))
        .bench_local_values(|(mut uf, ops)| {
            for op in ops {
                match op {
                    Operation::Union(a, b) => {
                        uf.union(a, b);
                    }
                    Operation::Find(a) => {
                        uf.find(a);
                    }
                }
            }
        });
}

trait ConcurrentUf: Clone + Send + Sync + 'static {
    fn with_capacity(capacity: usize) -> Self;
    fn apply_ops(&self, ops: &[Operation]) {
        for op in ops {
            match op {
                Operation::Union(a, b) => {
                    self.union(*a, *b);
                }
                Operation::Find(a) => {
                    self.find(*a);
                }
            }
        }
    }
    fn find_many_naive(&self, ops: &[usize]) {
        for op in ops {
            divan::black_box(self.find_naive(*op));
        }
    }
    fn union(&self, a: usize, b: usize);
    fn find(&self, a: usize) -> usize;
    fn find_naive(&self, a: usize) -> usize {
        self.find(a)
    }
}

impl ConcurrentUf for concurrent::UnionFind<usize> {
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    fn union(&self, a: usize, b: usize) {
        self.union(a, b);
    }

    fn find(&self, a: usize) -> usize {
        self.find(a)
    }
}

impl ConcurrentUf for Arc<Mutex<UnionFind<usize>>> {
    fn with_capacity(capacity: usize) -> Self {
        let mut uf = UnionFind::<usize>::default();
        uf.find(capacity);
        Arc::new(Mutex::new(uf))
    }

    fn apply_ops(&self, ops: &[Operation]) {
        let mut uf = self.lock().unwrap();
        for op in ops {
            match op {
                Operation::Union(a, b) => {
                    uf.union(*a, *b);
                }
                Operation::Find(a) => {
                    uf.find(*a);
                }
            }
        }
    }

    fn union(&self, a: usize, b: usize) {
        let mut uf = self.lock().unwrap();
        uf.union(a, b);
    }

    fn find(&self, a: usize) -> usize {
        let mut uf = self.lock().unwrap();
        uf.find(a)
    }
}

impl ConcurrentUf for Arc<RwLock<UnionFind<usize>>> {
    fn with_capacity(capacity: usize) -> Self {
        let mut uf = UnionFind::<usize>::default();
        uf.find(capacity);
        Arc::new(RwLock::new(uf))
    }

    fn apply_ops(&self, ops: &[Operation]) {
        let mut uf = self.write().unwrap();
        for op in ops {
            match op {
                Operation::Union(a, b) => {
                    uf.union(*a, *b);
                }
                Operation::Find(a) => {
                    uf.find(*a);
                }
            }
        }
    }

    fn union(&self, a: usize, b: usize) {
        let mut uf = self.write().unwrap();
        uf.union(a, b);
    }

    fn find(&self, a: usize) -> usize {
        let mut uf = self.write().unwrap();
        uf.find(a)
    }

    fn find_naive(&self, a: usize) -> usize {
        let uf = self.read().unwrap();
        uf.find_naive(a)
    }

    fn find_many_naive(&self, ops: &[usize]) {
        let uf = self.read().unwrap();
        for op in ops {
            divan::black_box(uf.find_naive(*op));
        }
    }
}

impl ConcurrentUf for Arc<ReadOptimizedLock<UnionFind<usize>>> {
    fn with_capacity(capacity: usize) -> Self {
        let mut uf = UnionFind::<usize>::default();
        uf.find(capacity);
        Arc::new(ReadOptimizedLock::new(uf))
    }

    fn union(&self, a: usize, b: usize) {
        let mut uf = self.lock();
        uf.union(a, b);
    }

    fn apply_ops(&self, ops: &[Operation]) {
        let mut uf = self.lock();
        for op in ops {
            match op {
                Operation::Union(a, b) => {
                    uf.union(*a, *b);
                }
                Operation::Find(a) => {
                    uf.find(*a);
                }
            }
        }
    }

    fn find_many_naive(&self, ops: &[usize]) {
        let uf = self.read();
        for op in ops {
            divan::black_box(uf.find_naive(*op));
        }
    }

    fn find(&self, a: usize) -> usize {
        let mut uf = self.lock();
        uf.find(a)
    }

    fn find_naive(&self, a: usize) -> usize {
        let uf = self.read();
        uf.find_naive(a)
    }
}
