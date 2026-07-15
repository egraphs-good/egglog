//! Benchmark-only entry points into the internal index-construction code.
//!
//! Exposed via `#[doc(hidden)] pub` solely so `benches/build_index.rs` can measure the custom
//! radix-sort + tournament-merge against the standard library and time serial vs parallel
//! `ColumnIndex` construction. Not a stable API; signatures may change without notice.

use egglog_concurrency::ThreadPool;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    common::Value,
    free_join::Database,
    numeric_id::NumericId,
    offsets::RowId,
    table::SortedWritesTable,
    table_spec::{ColumnId, Table, WrappedTable},
};

use super::{ColumnIndex, IndexBase, merge_sorted_blocks_dedup, radix_sort_slice_by_value};

/// Generate `n_cols` column blocks of `(Value, RowId)` pairs. Each block is RowId-ascending
/// (matching a table scan) with values drawn from `0..distinct`, so equal-value runs occur.
///
/// Returns the concatenated pairs and the block bounds: block `b` is `[bounds[b], bounds[b + 1])`.
pub fn gen_blocks(
    n_rows: usize,
    n_cols: usize,
    distinct: u32,
    seed: u64,
) -> (Vec<(Value, RowId)>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pairs = Vec::with_capacity(n_rows * n_cols);
    let mut bounds = vec![0usize];
    for _ in 0..n_cols {
        for r in 0..n_rows {
            pairs.push((
                Value::new(rng.random_range(0..distinct)),
                RowId::from_usize(r),
            ));
        }
        bounds.push(pairs.len());
    }
    (pairs, bounds)
}

/// The custom sort/merge from `rebuild_full`: radix-sort each block by value, then merge the
/// blocks with dedup (a single block skips the merge).
pub fn sort_merge_custom(mut pairs: Vec<(Value, RowId)>, bounds: &[usize]) -> Vec<(Value, RowId)> {
    let widest = bounds.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    let mut scratch = vec![(Value::new_const(0), RowId::new_const(0)); widest];
    for b in 0..bounds.len() - 1 {
        radix_sort_slice_by_value(&mut pairs[bounds[b]..bounds[b + 1]], &mut scratch);
    }
    if bounds.len() <= 2 {
        pairs
    } else {
        merge_sorted_blocks_dedup(pairs, bounds)
    }
}

/// The standard-library analogue of [`sort_merge_custom`]: sort the whole concatenation by
/// `(Value, RowId)`, then drop duplicates iff `dedup` is set.
///
/// Pass `dedup = false` for a single column and `true` for several, mirroring what
/// `sort_merge_custom` does: a single block has one pair per row, so every `RowId` is distinct
/// and no `(Value, RowId)` pair can repeat -- deduping there is a wasted pass that would unfairly
/// handicap this side of the comparison. Only multiple columns can put the same value (hence the
/// same pair) on one row.
pub fn sort_merge_std(mut pairs: Vec<(Value, RowId)>, dedup: bool) -> Vec<(Value, RowId)> {
    pairs.sort_unstable();
    if dedup {
        pairs.dedup();
    }
    pairs
}

/// Run `f` with an index thread pool of `n` threads installed, so parallel construction actually
/// forks: `ColumnIndex` derives both its shard count and worker-pool size from the ambient pool.
pub fn with_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
    ThreadPool::new(n.max(1)).install(f)
}

/// A random table set up to (re)build a `ColumnIndex` over its value columns (`1..=n_val_cols`;
/// column 0 is a unique key so no rows merge).
pub struct IndexInput {
    table: WrappedTable,
    cols: Vec<ColumnId>,
}

impl IndexInput {
    pub fn random(n_rows: usize, n_val_cols: usize, distinct: u32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_cols = n_val_cols + 1;
        let mut table = SortedWritesTable::new(
            1,
            n_cols,
            None,
            vec![],
            Box::new(|_, _old, new, out: &mut Vec<Value>| {
                out.extend_from_slice(new);
                false
            }),
        );
        {
            let mut buf = table.new_buffer();
            let mut row = vec![Value::new(0); n_cols];
            for i in 0..n_rows {
                row[0] = Value::from_usize(i);
                for cell in row.iter_mut().skip(1) {
                    *cell = Value::new(rng.random_range(0..distinct));
                }
                buf.stage_insert(&row);
            }
        }
        let db = Database::default();
        db.with_execution_state(|es| {
            table.merge(es);
        });
        IndexInput {
            table: WrappedTable::new(table),
            cols: (1..n_cols).map(ColumnId::from_usize).collect(),
        }
    }

    /// Build the index with the serial radix-sort + merge path.
    pub fn build_serial(&self) {
        let mut ci = ColumnIndex::new();
        ci.rebuild_full(&self.cols, self.table.as_ref(), self.table.all().as_ref());
        std::hint::black_box(&ci);
    }

    /// Build the index with the parallel sharded path. Call inside [`with_threads`].
    pub fn build_parallel(&self) {
        let mut ci = ColumnIndex::new();
        ci.merge_parallel(&self.cols, self.table.as_ref(), self.table.all().as_ref());
        std::hint::black_box(&ci);
    }
}
