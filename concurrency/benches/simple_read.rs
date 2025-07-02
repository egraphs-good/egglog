use std::{ops::Deref, sync::RwLock};

use concurrency::{MutexReader, ReadOptimizedLock};
use divan::{counter::ItemsCount, Bencher};

fn main() {
    divan::main()
}
#[divan::bench(threads = [1, 2, 4, 8, 16, 20], types = [ReadOptimizedLock<usize>, RwLock<usize>], sample_size = 100)]
fn read_contention<T: ReadLock<usize>>(bencher: Bencher) {
    let lock = T::new(0);
    bencher.bench(|| {
        divan::black_box(*lock.read());
    });
}

#[divan::bench(types=[ReadOptimizedLock<usize>, RwLock<usize>], consts = [1, 2, 4, 8, 16, 20], sample_count=50)]
fn read_throughput<const N: usize, T: ReadLock<usize>>(bencher: Bencher) {
    const TOTAL_ITEMS: usize = 1_000_000;
    const BATCH_SIZE: usize = 1_000;
    const N_BATCHES: usize = TOTAL_ITEMS / BATCH_SIZE;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(N)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| ())
        .input_counter(|_| ItemsCount::new(TOTAL_ITEMS))
        .bench_values(|_| {
            let lock = T::new(0);
            pool.scope(|scope| {
                for _ in 0..N_BATCHES {
                    scope.spawn(|_| {
                        for _ in 0..BATCH_SIZE {
                            divan::black_box(*lock.read());
                        }
                    });
                }
            })
        });
}

trait ReadLock<T>: Send + Sync {
    type Guard<'a>: Deref<Target = T>
    where
        Self: 'a;
    fn new(data: T) -> Self;
    fn read(&self) -> Self::Guard<'_>;
}

impl<T: Send + Sync> ReadLock<T> for ReadOptimizedLock<T> {
    type Guard<'a>
        = MutexReader<'a, T>
    where
        Self: 'a;
    fn new(data: T) -> Self {
        ReadOptimizedLock::new(data)
    }
    fn read(&self) -> MutexReader<'_, T> {
        self.read()
    }
}

impl<T: Send + Sync> ReadLock<T> for RwLock<T> {
    type Guard<'a>
        = std::sync::RwLockReadGuard<'a, T>
    where
        Self: 'a;
    fn new(data: T) -> Self {
        RwLock::new(data)
    }
    fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.read().unwrap()
    }
}
