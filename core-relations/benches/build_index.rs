//! Microbenchmarks for building `ColumnIndex`es.
//!
//! Two questions motivate these:
//!   * `sort_merge_*`: the rebuild path sorts and merges `(Value, RowId)` pairs with hand-written
//!     radix sort + tournament merge instead of the standard library. These compare the two so we
//!     can track the speedup rather than assume it.
//!   * `build_*`: end-to-end index construction, serially vs. in parallel, across subset sizes.

use divan::{Bencher, counter::ItemsCount};
use egglog_core_relations::bench_support::{
    IndexInput, gen_blocks, sort_merge_custom, sort_merge_std, with_threads,
};

fn main() {
    divan::main()
}

/// Values are drawn from `rows / DISTINCT_DIVISOR` distinct keys, giving each value several rows.
const DISTINCT_DIVISOR: u32 = 8;

fn distinct_for(rows: usize) -> u32 {
    ((rows as u32) / DISTINCT_DIVISOR).max(1)
}

// -- Sort + merge primitives: custom vs. standard library ---------------------------------------

/// Single column: only the per-block sort runs (no merge). Custom = radix sort.
#[divan::bench(consts = [4096, 65_536, 1 << 20])]
fn sort_single_custom<const N: usize>(bench: Bencher) {
    let (pairs, bounds) = gen_blocks(N, 1, distinct_for(N), 1);
    bench
        .with_inputs(|| pairs.clone())
        .input_counter(|p| ItemsCount::new(p.len()))
        .bench_values(|p| sort_merge_custom(p, &bounds));
}

/// Single column, standard `sort_unstable`. No dedup: with one pair per row every `RowId` is
/// distinct, so no `(Value, RowId)` pair repeats -- matching what the custom path does here.
#[divan::bench(consts = [4096, 65_536, 1 << 20])]
fn sort_single_std<const N: usize>(bench: Bencher) {
    let (pairs, _bounds) = gen_blocks(N, 1, distinct_for(N), 1);
    bench
        .with_inputs(|| pairs.clone())
        .input_counter(|p| ItemsCount::new(p.len()))
        .bench_values(|p| sort_merge_std(p, false));
}

/// Four columns: per-block radix sort followed by the tournament merge with dedup.
#[divan::bench(consts = [4096, 65_536, 1 << 20])]
fn merge_multi_custom<const N: usize>(bench: Bencher) {
    let (pairs, bounds) = gen_blocks(N, 4, distinct_for(N), 1);
    bench
        .with_inputs(|| pairs.clone())
        .input_counter(|p| ItemsCount::new(p.len()))
        .bench_values(|p| sort_merge_custom(p, &bounds));
}

/// Four columns, standard `sort_unstable` + `dedup` over the whole concatenation (a value can
/// repeat across a row's columns, so duplicate pairs are possible and must be dropped).
#[divan::bench(consts = [4096, 65_536, 1 << 20])]
fn merge_multi_std<const N: usize>(bench: Bencher) {
    let (pairs, _bounds) = gen_blocks(N, 4, distinct_for(N), 1);
    bench
        .with_inputs(|| pairs.clone())
        .input_counter(|p| ItemsCount::new(p.len()))
        .bench_values(|p| sort_merge_std(p, true));
}

// -- End-to-end index construction: serial vs. parallel -----------------------------------------

const N_VAL_COLS: usize = 3;

#[divan::bench(consts = [4096, 65_536, 1 << 20])]
fn build_serial<const N: usize>(bench: Bencher) {
    bench
        .with_inputs(|| IndexInput::random(N, N_VAL_COLS, distinct_for(N), 1))
        .input_counter(|_| ItemsCount::new(N * N_VAL_COLS))
        .bench_refs(|input| input.build_serial());
}

#[divan::bench(consts = [4096, 65_536, 1 << 20])]
fn build_parallel<const N: usize>(bench: Bencher) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    bench
        .with_inputs(|| IndexInput::random(N, N_VAL_COLS, distinct_for(N), 1))
        .input_counter(|_| ItemsCount::new(N * N_VAL_COLS))
        .bench_refs(|input| with_threads(threads, || input.build_parallel()));
}
