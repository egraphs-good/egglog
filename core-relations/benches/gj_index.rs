//! Micro-benchmark comparing indexing strategies for Generic Join on a single
//! relation: a sort-based index (built either by comparison or radix sort)
//! against a parameterized lazy trie.
//!
//! - SortedIndex (`build`): row ids sorted lexicographically by
//!   (col_0, col_1, ..., col_{k-1}) using `sort_unstable_by` with a column
//!   comparator. Descent into a value at level i is done by binary search over
//!   the current range; distinct-value iteration is a linear scan that
//!   detects runs.
//! - SortedIndex (`build_radix`): same output as `build`, produced by an LSD
//!   radix sort on a packed composite key. Requires Σ bits_per_col ≤ 64 (one
//!   pre-pass over the table computes the per-column bit-widths).
//!
//! - LazyTrie: a fresh implementation in the spirit of `free_join::execute::TrieNode`,
//!   but standalone. Each node is one of `Lazy(rows)`, `Expanded(HashMap<Value, Node>)`,
//!   or `Leaf`. The trie is fully lazy and deterministic: at construction the
//!   root is just `Lazy(all_rows)`. The first access forces that single node,
//!   grouping its row set by the next column into one Lazy child per distinct
//!   value. Each child is itself Lazy until it is in turn accessed — we never
//!   expand more than one level per access. This matches the lazy GJ trie in
//!   `executor.rs`.
//!
//! Workload:
//!   - `probe`: N random k-tuple point-lookups.
//!
//! Measurement convention: `probe_*` benches include the index build in their
//! timing. The lazy trie defers essentially all of its build to query time, so
//! including build is the only fair way to compare against the sorted index.
//! The `build_*` benches isolate the pure build cost.
//!
//! Run with:
//!     cargo bench -p egglog-core-relations --bench gj_index
//! Or run a single group:
//!     cargo bench -p egglog-core-relations --bench gj_index -- build

use std::hint::black_box;

use divan::{Bencher, counter::ItemsCount};
use hashbrown::HashMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    divan::main()
}

type Value = u64;
type RowId = u32;

// --- Knobs ---------------------------------------------------------------

const ROWS: usize = 100_000;
const COLS: usize = 3;
/// Per-column cardinalities (number of distinct values). Length must be >= COLS.
const CARDINALITIES: [u64; 3] = [5_000, 1_000, 2];
const PROBES: usize = 100_000;
const SEED: u64 = 0xDEAD_BEEF;

// --- Table ---------------------------------------------------------------

struct Table {
    /// Flat row-major storage: `data[row * n_cols + col]`.
    data: Vec<Value>,
    n_rows: usize,
    n_cols: usize,
}

impl Table {
    fn random(n_rows: usize, n_cols: usize, cards: &[u64], seed: u64) -> Self {
        assert!(cards.len() >= n_cols);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for _ in 0..n_rows {
            for c in 0..n_cols {
                data.push(rng.random_range(0..cards[c]));
            }
        }
        Self { data, n_rows, n_cols }
    }

    #[inline]
    fn get(&self, row: RowId, col: usize) -> Value {
        // Safety: row < n_rows, col < n_cols by construction in the benchmark.
        unsafe { *self.data.get_unchecked(row as usize * self.n_cols + col) }
    }
}

// --- Sorted Index --------------------------------------------------------

struct SortedIndex {
    sorted: Vec<RowId>,
}

impl SortedIndex {
    /// Build by comparison sort (`sort_unstable_by` with a lex comparator over columns).
    fn build(table: &Table) -> Self {
        let mut indices: Vec<RowId> = (0..table.n_rows as RowId).collect();
        indices.sort_unstable_by(|&a, &b| {
            for c in 0..table.n_cols {
                match table.get(a, c).cmp(&table.get(b, c)) {
                    std::cmp::Ordering::Equal => continue,
                    o => return o,
                }
            }
            std::cmp::Ordering::Equal
        });
        Self { sorted: indices }
    }

    /// Build by LSD radix sort over a packed composite key.
    ///
    /// First pass scans the table to find the max value per column, packs
    /// each row's columns into a single `u64` (col_0 in the highest bits),
    /// then performs an 8-bit-digit LSD radix sort on `(key, row_id)` pairs.
    /// The pair is moved together so the resulting `sorted` permutation is
    /// lex order by columns — identical to `build`'s output.
    ///
    /// Panics if the column widths exceed 64 bits combined.
    fn build_radix(table: &Table) -> Self {
        let n = table.n_rows;
        if n == 0 {
            return Self { sorted: Vec::new() };
        }

        // Find max per column (one pass over the table). Needed to size each
        // column's slice of the composite key tightly — otherwise we pay for
        // a worst-case 8 passes regardless of data.
        let mut max_per_col = vec![0u64; table.n_cols];
        for r in 0..n as RowId {
            for c in 0..table.n_cols {
                let v = table.get(r, c);
                if v > max_per_col[c] {
                    max_per_col[c] = v;
                }
            }
        }

        // Bits per column, at least 1 (so all-zero columns still occupy a slot).
        let bits_per_col: Vec<u32> = max_per_col
            .iter()
            .map(|&m| if m == 0 { 1 } else { 64 - m.leading_zeros() })
            .collect();
        let total_bits: u32 = bits_per_col.iter().sum();
        assert!(
            total_bits <= 64,
            "composite key too wide ({total_bits} bits); needs > u64"
        );

        // Shifts: column 0 is most significant.
        let mut shifts = vec![0u32; table.n_cols];
        let mut acc = total_bits;
        for c in 0..table.n_cols {
            acc -= bits_per_col[c];
            shifts[c] = acc;
        }

        // Pack (composite_key, row_id) pairs.
        let mut buf: Vec<(u64, RowId)> = (0..n as RowId)
            .map(|r| {
                let mut key = 0u64;
                for c in 0..table.n_cols {
                    key |= table.get(r, c) << shifts[c];
                }
                (key, r)
            })
            .collect();
        let mut tmp: Vec<(u64, RowId)> = vec![(0, 0); n];

        // LSD radix sort, 8-bit digits, stable scatter.
        let n_passes = total_bits.div_ceil(8);
        for pass in 0..n_passes {
            let shift = pass * 8;
            let mut counts = [0u32; 256];
            for &(k, _) in &buf {
                counts[((k >> shift) & 0xFF) as usize] += 1;
            }
            // Prefix sum → start offsets.
            let mut offsets = [0u32; 256];
            let mut running = 0u32;
            for i in 0..256 {
                offsets[i] = running;
                running += counts[i];
            }
            // Stable scatter (within a bucket, original order is preserved).
            for &(k, r) in &buf {
                let bucket = ((k >> shift) & 0xFF) as usize;
                let pos = offsets[bucket] as usize;
                offsets[bucket] += 1;
                tmp[pos] = (k, r);
            }
            std::mem::swap(&mut buf, &mut tmp);
        }

        Self {
            sorted: buf.into_iter().map(|(_, r)| r).collect(),
        }
    }

    /// Probe a full-length prefix. Returns the number of matching rows.
    fn probe(&self, table: &Table, prefix: &[Value]) -> usize {
        let mut lo = 0usize;
        let mut hi = self.sorted.len();
        for (col, &v) in prefix.iter().enumerate() {
            let slice = &self.sorted[lo..hi];
            // partition_point on the slice; offset by lo to get absolute indices.
            let new_lo = lo + slice.partition_point(|&r| table.get(r, col) < v);
            let new_hi = lo + slice.partition_point(|&r| table.get(r, col) <= v);
            lo = new_lo;
            hi = new_hi;
            if lo == hi {
                return 0;
            }
        }
        hi - lo
    }
}

// --- Lazy Trie -----------------------------------------------------------

enum TrieNode {
    /// Not expanded. Children for the next column have not been computed.
    Lazy(Vec<RowId>),
    /// Expanded: all children at this level are present.
    Expanded(HashMap<Value, TrieNode>),
    /// Reached past the last column.
    Leaf,
}

struct LazyTrie {
    root: TrieNode,
}

impl LazyTrie {
    fn build(table: &Table) -> Self {
        let rows: Vec<RowId> = (0..table.n_rows as RowId).collect();
        // Start fully lazy: no children are computed at construction time.
        // The first access forces root expansion; each newly-created child is
        // itself Lazy until that child is accessed in turn (matching the lazy
        // GJ trie in `free_join::execute::TrieNode`).
        let root = if table.n_cols == 0 {
            TrieNode::Leaf
        } else {
            TrieNode::Lazy(rows)
        };
        Self { root }
    }

    /// Build the trie fully eagerly: every level is grouped and materialized
    /// up-front. Used to measure the cost of constructing the complete hash
    /// trie, independent of any query pattern.
    fn build_full(table: &Table) -> Self {
        let rows: Vec<RowId> = (0..table.n_rows as RowId).collect();
        let root = Self::build_full_node(table, rows, 0);
        Self { root }
    }

    fn build_full_node(table: &Table, rows: Vec<RowId>, col: usize) -> TrieNode {
        if col == table.n_cols {
            return TrieNode::Leaf;
        }
        let mut groups: HashMap<Value, Vec<RowId>> = HashMap::new();
        for r in rows {
            groups.entry(table.get(r, col)).or_default().push(r);
        }
        let mut children = HashMap::with_capacity(groups.len());
        for (k, sub) in groups {
            children.insert(k, Self::build_full_node(table, sub, col + 1));
        }
        TrieNode::Expanded(children)
    }

    /// Group `rows` by the value of column `col`, producing one Lazy child
    /// per distinct value. Children are themselves Lazy — we never expand
    /// more than a single level per access.
    fn expand(table: &Table, rows: Vec<RowId>, col: usize) -> TrieNode {
        let next_col = col + 1;
        let mut groups: HashMap<Value, Vec<RowId>> = HashMap::new();
        for r in rows {
            groups.entry(table.get(r, col)).or_default().push(r);
        }
        let mut children = HashMap::with_capacity(groups.len());
        for (k, sub) in groups {
            let child = if next_col == table.n_cols {
                TrieNode::Leaf
            } else {
                TrieNode::Lazy(sub)
            };
            children.insert(k, child);
        }
        TrieNode::Expanded(children)
    }

    /// Force a node to its `Expanded` form (no-op if already expanded).
    fn force(node: &mut TrieNode, table: &Table, col: usize) {
        if matches!(node, TrieNode::Lazy(_)) {
            let rows = match std::mem::replace(node, TrieNode::Leaf) {
                TrieNode::Lazy(r) => r,
                _ => unreachable!(),
            };
            *node = Self::expand(table, rows, col);
        }
    }

    fn probe(&mut self, table: &Table, prefix: &[Value]) -> usize {
        Self::probe_rec(&mut self.root, table, prefix, 0)
    }

    fn probe_rec(node: &mut TrieNode, table: &Table, prefix: &[Value], col: usize) -> usize {
        if col == prefix.len() {
            return Self::count(node);
        }
        Self::force(node, table, col);
        match node {
            TrieNode::Expanded(children) => match children.get_mut(&prefix[col]) {
                Some(child) => Self::probe_rec(child, table, prefix, col + 1),
                None => 0,
            },
            TrieNode::Leaf => 0,
            TrieNode::Lazy(_) => unreachable!(),
        }
    }

    fn count(node: &TrieNode) -> usize {
        match node {
            TrieNode::Leaf => 1,
            TrieNode::Lazy(rows) => rows.len(),
            TrieNode::Expanded(children) => children.values().map(Self::count).sum(),
        }
    }
}

// --- Shared fixtures -----------------------------------------------------

fn fresh_table() -> Table {
    Table::random(ROWS, COLS, &CARDINALITIES, SEED)
}

fn fresh_probes(table: &Table) -> Vec<Vec<Value>> {
    // Sample each probe by picking an existing row uniformly at random. This
    // guarantees every probe hits at least one matching tuple, so the bench
    // exercises full-depth descent rather than early-termination on misses.
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xCAFE);
    (0..PROBES)
        .map(|_| {
            let row = rng.random_range(0..table.n_rows as RowId);
            (0..table.n_cols).map(|c| table.get(row, c)).collect()
        })
        .collect()
}

// --- Benchmarks: build ---------------------------------------------------

#[divan::bench(sample_count = 20)]
fn build_sorted(bencher: Bencher) {
    let table = fresh_table();
    bencher
        .counter(ItemsCount::new(table.n_rows))
        .bench_local(|| black_box(SortedIndex::build(&table)));
}

#[divan::bench(sample_count = 20)]
fn build_sorted_radix(bencher: Bencher) {
    let table = fresh_table();
    bencher
        .counter(ItemsCount::new(table.n_rows))
        .bench_local(|| black_box(SortedIndex::build_radix(&table)));
}

#[divan::bench(sample_count = 20)]
fn build_trie(bencher: Bencher) {
    // Measures building the fully expanded trie. The lazy variant used in
    // probe_trie has a trivially-cheap build (just a row vector), so timing
    // it here would not be informative — the eager build is what's directly
    // comparable to build_sorted / build_sorted_radix.
    let table = fresh_table();
    bencher
        .counter(ItemsCount::new(table.n_rows))
        .bench_local(|| black_box(LazyTrie::build_full(&table)));
}

// --- Benchmarks: random point probes ------------------------------------

#[divan::bench(sample_count = 20)]
fn probe_sorted(bencher: Bencher) {
    let table = fresh_table();
    let probes = fresh_probes(&table);
    bencher
        .counter(ItemsCount::new(probes.len()))
        .bench_local(|| {
            let index = SortedIndex::build(&table);
            let mut hits = 0;
            for p in &probes {
                if index.probe(&table, p) > 0 {
                    hits += 1;
                }
            }
            black_box(hits)
        });
}

#[divan::bench(sample_count = 20)]
fn probe_sorted_radix(bencher: Bencher) {
    let table = fresh_table();
    let probes = fresh_probes(&table);
    bencher
        .counter(ItemsCount::new(probes.len()))
        .bench_local(|| {
            let index = SortedIndex::build_radix(&table);
            let mut hits = 0;
            for p in &probes {
                if index.probe(&table, p) > 0 {
                    hits += 1;
                }
            }
            black_box(hits)
        });
}

#[divan::bench(sample_count = 20)]
fn probe_trie(bencher: Bencher) {
    let table = fresh_table();
    let probes = fresh_probes(&table);
    bencher
        .counter(ItemsCount::new(probes.len()))
        .bench_local(|| {
            // Build is rolled into the query because the lazy trie's "build"
            // is fully deferred to query time — the construction itself is
            // essentially free. This keeps the timing directly comparable
            // against probe_sorted.
            let mut trie = LazyTrie::build(&table);
            let mut hits = 0;
            for p in &probes {
                if trie.probe(&table, p) > 0 {
                    hits += 1;
                }
            }
            black_box(hits)
        });
}

