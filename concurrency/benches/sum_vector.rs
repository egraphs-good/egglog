use std::{
    env,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use divan::{Bencher, counter::ItemsCount};
use egglog_concurrency::ThreadPool;
use rayon::prelude::*;

const DEFAULT_ITEMS: usize = 256_000_000;
const DEFAULT_SAMPLE_COUNT: u32 = 5;

fn main() {
    divan::main()
}

#[divan::bench(sample_count = DEFAULT_SAMPLE_COUNT)]
fn single_threaded_sum(bencher: Bencher) {
    let data = data();
    let expected = expected_sum();

    bencher
        .with_inputs(|| data)
        .input_counter(|data| ItemsCount::new(data.len()))
        .bench_values(|data| {
            let sum = sum_slice(divan::black_box(data));
            assert_eq!(sum, expected);
            divan::black_box(sum)
        });
}

#[divan::bench(sample_count = DEFAULT_SAMPLE_COUNT)]
fn rayon_sum(bencher: Bencher) {
    let data = data();
    let expected = expected_sum();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count())
        .build()
        .unwrap();

    bencher
        .with_inputs(|| data)
        .input_counter(|data| ItemsCount::new(data.len()))
        .bench_values(|data| {
            let sum = pool.install(|| {
                divan::black_box(data)
                    .par_iter()
                    .map(|&value| value as u64)
                    .sum::<u64>()
            });
            assert_eq!(sum, expected);
            divan::black_box(sum)
        });
}

#[divan::bench(
    consts = [
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
    ],
    sample_count = DEFAULT_SAMPLE_COUNT,
)]
fn threadpool_chunked_sum<const CHUNK: usize>(bencher: Bencher) {
    let data = data();
    let expected = expected_sum();
    let pool = ThreadPool::new(thread_count());

    bencher
        .with_inputs(|| data)
        .input_counter(|data| ItemsCount::new(data.len()))
        .bench_values(|data| {
            let total = AtomicU64::new(0);
            pool.scope(|scope| {
                for chunk in divan::black_box(data).chunks(CHUNK) {
                    let total = &total;
                    scope.spawn(move |_| {
                        total.fetch_add(sum_slice(chunk), Ordering::Relaxed);
                    });
                }
            });

            let sum = total.load(Ordering::Relaxed);
            assert_eq!(sum, expected);
            divan::black_box(sum)
        });
}

fn data() -> &'static [u32] {
    static DATA: OnceLock<Vec<u32>> = OnceLock::new();
    DATA.get_or_init(|| {
        (0..data_len())
            .map(|i| {
                let value = i as u32;
                value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
            })
            .collect()
    })
}

fn expected_sum() -> u64 {
    static EXPECTED: OnceLock<u64> = OnceLock::new();
    *EXPECTED.get_or_init(|| sum_slice(data()))
}

fn sum_slice(data: &[u32]) -> u64 {
    data.iter().map(|&value| value as u64).sum()
}

fn data_len() -> usize {
    env_usize("EGGLOG_SUM_BENCH_LEN").unwrap_or(DEFAULT_ITEMS)
}

fn thread_count() -> usize {
    env_usize("EGGLOG_SUM_BENCH_THREADS")
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, usize::from))
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.replace('_', "").parse().ok())
        .filter(|&value| value > 0)
}
