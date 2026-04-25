use std::fmt;

use divan::{Bencher, black_box};
use egglog_core_relations::{BitVecSubset, Offsets, RowId, SortedOffsetVector};
use egglog_numeric_id::NumericId;
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn main() {
    divan::main()
}

/// Benchmark parameters: (n_elements, range).
///
/// density = n / range.  The codebase auto-promotes from Sparse to Bitvec once
/// density exceeds 5%, so each size level is tested in both regimes:
///   - ~20% density  → bitvec territory
///   -  ~1% density  → sparse territory
#[derive(Clone, Copy)]
struct Param {
    n: usize,
    range: u32,
}

impl Param {
    const fn new(n: usize, range: u32) -> Self {
        Self { n, range }
    }
}

impl fmt::Debug for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let density_pct = self.n as f64 * 100.0 / self.range as f64;
        write!(f, "n={}_d={:.0}%", self.n, density_pct)
    }
}

const PARAMS: &[Param] = &[
    Param::new(1_000, 2_000),        // 50%
    Param::new(1_000, 5_000),        // 20%
    Param::new(1_000, 100_000),      //  1%
    Param::new(10_000, 20_000),      // 50%
    Param::new(10_000, 50_000),      // 20%
    Param::new(10_000, 1_000_000),   //  1%
    Param::new(100_000, 200_000),    // 50%
    Param::new(100_000, 500_000),    // 20%
    Param::new(100_000, 10_000_000), //  1%
];

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Number of clusters used by `make_rows`.
const NUM_CLUSTERS: usize = 8;

/// Generate `n` sorted unique row IDs clustered into `NUM_CLUSTERS`
/// randomly-placed windows spread across `[0, range)`.
///
/// The range is divided into `NUM_CLUSTERS` equal tiles.  Within each tile a
/// window is placed at a random offset; the window is wide enough to hold
/// `n / NUM_CLUSTERS` unique values but at most `tile / 4`, leaving large
/// gaps between clusters (≈75% of the range is empty).  Elements are drawn
/// uniformly within each window.
fn make_rows(n: usize, range: u32, seed: u64) -> Vec<RowId> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let tile = (range / NUM_CLUSTERS as u32).max(1);
    let n_per = (n + NUM_CLUSTERS - 1) / NUM_CLUSTERS;
    // At minimum wide enough to hold n_per unique values; at most tile/4 so
    // that gaps are always at least 3× the window width.
    let window = (tile / 4).max(n_per as u32).min(tile);

    let mut set = std::collections::BTreeSet::new();

    for c in 0..NUM_CLUSTERS {
        if set.len() >= n {
            break;
        }
        let tile_lo = c as u32 * tile;
        // Place the window at a random position within the tile.
        let max_offset = tile.saturating_sub(window);
        let lo = tile_lo + if max_offset > 0 { rng.random_range(0..max_offset) } else { 0 };
        let hi = (lo + window).min(range);
        if lo >= hi {
            continue;
        }
        let want = n_per.min(n - set.len());
        let prev = set.len();
        for _ in 0..want * 8 {
            if set.len() >= prev + want {
                break;
            }
            set.insert(rng.random_range(lo..hi));
        }
    }
    // Fallback: top up uniformly if any windows were too narrow.
    while set.len() < n {
        set.insert(rng.random_range(0..range));
    }

    set.into_iter().take(n).map(RowId::new).collect()
}

fn build_sparse(rows: &[RowId]) -> SortedOffsetVector {
    let mut sov = SortedOffsetVector::default();
    for &row in rows {
        sov.push(row);
    }
    sov
}

fn build_bitvec(rows: &[RowId]) -> BitVecSubset {
    let mut bv = BitVecSubset::default();
    for &row in rows {
        bv.push_sorted(row);
    }
    bv
}

/// Hash-based keep predicate: retains roughly half the elements.
/// Uses a fast multiplicative hash on the row index.
#[inline]
fn keep_row(row: RowId) -> bool {
    (row.index() as u32).wrapping_mul(2_654_435_761) & 1 == 0
}

// ─── Build ──────────────────────────────────────────────────────────────────

#[divan::bench(args = PARAMS)]
fn build_sparse_bench(bencher: Bencher, param: &Param) {
    let rows = make_rows(param.n, param.range, 0);
    bencher.bench(|| build_sparse(black_box(&rows)))
}

#[divan::bench(args = PARAMS)]
fn build_bitvec_bench(bencher: Bencher, param: &Param) {
    let rows = make_rows(param.n, param.range, 0);
    bencher.bench(|| build_bitvec(black_box(&rows)))
}

// ─── Enumerate ──────────────────────────────────────────────────────────────

#[divan::bench(args = PARAMS)]
fn enumerate_sparse(bencher: Bencher, param: &Param) {
    let rows = make_rows(param.n, param.range, 0);
    let sov = build_sparse(&rows);
    bencher.bench(|| {
        let mut xor: u64 = 0;
        sov.offsets(|row| xor ^= row.index() as u64);
        black_box(xor)
    })
}

#[divan::bench(args = PARAMS)]
fn enumerate_bitvec(bencher: Bencher, param: &Param) {
    let rows = make_rows(param.n, param.range, 0);
    let bv = build_bitvec(&rows);
    bencher.bench(|| {
        let mut xor: u64 = 0;
        bv.offsets(|row| xor ^= row.index() as u64);
        black_box(xor)
    })
}

// ─── Intersection ───────────────────────────────────────────────────────────

#[divan::bench(args = PARAMS)]
fn intersect_sparse_sparse(bencher: Bencher, param: &Param) {
    let rows_a = make_rows(param.n, param.range, 0);
    let rows_b = make_rows(param.n, param.range, 1);
    bencher
        .with_inputs(|| (build_sparse(&rows_a), build_sparse(&rows_b)))
        .bench_values(|(mut a, b)| {
            a.intersect(b.slice());
            black_box(a)
        })
}

#[divan::bench(args = PARAMS)]
fn intersect_sparse_bitvec(bencher: Bencher, param: &Param) {
    let rows_a = make_rows(param.n, param.range, 0);
    let rows_b = make_rows(param.n, param.range, 1);
    bencher
        .with_inputs(|| (build_sparse(&rows_a), build_bitvec(&rows_b)))
        .bench_values(|(mut a, b)| {
            a.retain(|row| b.contains(row));
            black_box(a)
        })
}

#[divan::bench(args = PARAMS)]
fn intersect_bitvec_bitvec(bencher: Bencher, param: &Param) {
    let rows_a = make_rows(param.n, param.range, 0);
    let rows_b = make_rows(param.n, param.range, 1);
    bencher
        .with_inputs(|| (build_bitvec(&rows_a), build_bitvec(&rows_b)))
        .bench_values(|(mut a, b)| {
            a.intersect_with(&b);
            black_box(a)
        })
}

// ─── Retain / Filter ────────────────────────────────────────────────────────

#[divan::bench(args = PARAMS)]
fn retain_sparse(bencher: Bencher, param: &Param) {
    let rows = make_rows(param.n, param.range, 0);
    bencher
        .with_inputs(|| build_sparse(&rows))
        .bench_values(|mut sov| {
            sov.retain(keep_row);
            black_box(sov)
        })
}

#[divan::bench(args = PARAMS)]
fn retain_bitvec(bencher: Bencher, param: &Param) {
    let rows = make_rows(param.n, param.range, 0);
    bencher
        .with_inputs(|| build_bitvec(&rows))
        .bench_values(|mut bv| {
            bv.retain(keep_row);
            black_box(bv)
        })
}
