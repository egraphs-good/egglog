//! Feature-gated statistics on per-subset join indexes: how many rows each
//! index build covered and how many times the index was used before being
//! dropped.
//!
//! With the `index-stats` feature enabled, every dynamically-built join index
//! (see `DynamicIndex` in `execute.rs`) carries a [`BuildStats`] that counts
//! point probes (`get_subset`) and whole-index iterations (`for_each`). When
//! the index is dropped, the totals are folded into global histograms keyed by
//! index kind, usage class, build size, and use count; [`report`] renders
//! them. Without the feature, [`BuildStats`] is a zero-sized no-op.

/// The kind of index a [`BuildStats`] describes.
#[derive(Copy, Clone, Debug)]
pub(crate) enum IndexKind {
    /// `SortedColumnIndex`: single-column sort-based index, cached per trie node.
    SortedColumn = 0,
    /// `TupleIndex` from `group_by_key`: multi-column hash index, never cached.
    Tuple = 1,
    /// `SparseColumnIndex`: stack-allocated index over at most 8 rows.
    SparseColumn = 2,
    /// `ScanColumn` prober: no index at all; each probe scans the subset.
    /// "Rows" counts the subset size once (each probe passes over it).
    Scan = 3,
}

#[cfg(feature = "index-stats")]
pub(crate) use imp::BuildStats;
#[cfg(feature = "index-stats")]
pub use imp::report;

#[cfg(not(feature = "index-stats"))]
pub(crate) use noop::BuildStats;

#[cfg(not(feature = "index-stats"))]
mod noop {
    use super::IndexKind;

    /// No-op stand-in for the instrumented version; see the module docs.
    pub(crate) struct BuildStats;

    impl BuildStats {
        #[inline(always)]
        pub(crate) fn new(_kind: IndexKind, _rows: usize) -> Self {
            BuildStats
        }

        #[inline(always)]
        pub(crate) fn record_probe(&self) {}

        #[inline(always)]
        pub(crate) fn record_iter(&self) {}
    }
}

#[cfg(feature = "index-stats")]
mod imp {
    use std::fmt::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::IndexKind;

    const KINDS: usize = 4;
    const KIND_NAMES: [&str; KINDS] = ["SortedColumn", "Tuple", "SparseColumn", "Scan"];

    /// How a build was used over its lifetime.
    const CLASSES: usize = 4;
    const CLASS_NAMES: [&str; CLASSES] = ["unused", "probe-only", "iter-only", "mixed"];

    /// Build sizes bucketed by floor(log2(rows)): bucket `i` covers
    /// `[2^i, 2^(i+1))` rows, with 0 and 1 sharing bucket 0.
    const ROW_BUCKETS: usize = 24;

    /// Use counts: 0..=3 exactly, then power-of-two ranges.
    const USE_BUCKETS: usize = 16;
    const USE_LABELS: [&str; USE_BUCKETS] = [
        "0", "1", "2", "3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511",
        "512-1023", "1K-2K", "2K-4K", "4K-8K", ">=8K",
    ];

    fn row_bucket(rows: u64) -> usize {
        (63 - rows.max(1).leading_zeros() as usize).min(ROW_BUCKETS - 1)
    }

    fn use_bucket(uses: u64) -> usize {
        if uses < 4 {
            uses as usize
        } else {
            (2 + (63 - uses.leading_zeros() as usize)).min(USE_BUCKETS - 1)
        }
    }

    type Histogram = [[[[AtomicU64; USE_BUCKETS]; ROW_BUCKETS]; CLASSES]; KINDS];

    // Interior mutability and size are both intentional: this is only the
    // zero-initializer for the two static histograms below.
    #[allow(clippy::declare_interior_mutable_const, clippy::large_const_arrays)]
    const EMPTY_HISTOGRAM: Histogram = [const {
        [const { [const { [const { AtomicU64::new(0) }; USE_BUCKETS] }; ROW_BUCKETS] }; CLASSES]
    }; KINDS];

    /// Histogram cell arrays, indexed by `[kind][class][row_bucket][use_bucket]`.
    static BUILDS: Histogram = EMPTY_HISTOGRAM;
    static ROWS: Histogram = EMPTY_HISTOGRAM;

    /// Exact per-(kind, class) totals: builds, rows, probes, iters.
    static TOTALS: [[[AtomicU64; 4]; CLASSES]; KINDS] =
        [const { [const { [const { AtomicU64::new(0) }; 4] }; CLASSES] }; KINDS];

    /// Per-build usage counters attached to a single join index. Dropping the
    /// value folds its counts into the global histograms.
    pub(crate) struct BuildStats {
        kind: IndexKind,
        rows: u64,
        probes: AtomicU64,
        iters: AtomicU64,
    }

    impl BuildStats {
        pub(crate) fn new(kind: IndexKind, rows: usize) -> Self {
            BuildStats {
                kind,
                rows: rows as u64,
                probes: AtomicU64::new(0),
                iters: AtomicU64::new(0),
            }
        }

        #[inline]
        pub(crate) fn record_probe(&self) {
            self.probes.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub(crate) fn record_iter(&self) {
            self.iters.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Drop for BuildStats {
        fn drop(&mut self) {
            let probes = *self.probes.get_mut();
            let iters = *self.iters.get_mut();
            let class = match (probes > 0, iters > 0) {
                (false, false) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (true, true) => 3,
            };
            let kind = self.kind as usize;
            let rb = row_bucket(self.rows);
            let ub = use_bucket(probes + iters);
            BUILDS[kind][class][rb][ub].fetch_add(1, Ordering::Relaxed);
            ROWS[kind][class][rb][ub].fetch_add(self.rows, Ordering::Relaxed);
            let totals = &TOTALS[kind][class];
            totals[0].fetch_add(1, Ordering::Relaxed);
            totals[1].fetch_add(self.rows, Ordering::Relaxed);
            totals[2].fetch_add(probes, Ordering::Relaxed);
            totals[3].fetch_add(iters, Ordering::Relaxed);
        }
    }

    fn fmt_count(n: u64) -> String {
        if n >= 10_000_000_000 {
            format!("{:.1}G", n as f64 / 1e9)
        } else if n >= 10_000_000 {
            format!("{:.1}M", n as f64 / 1e6)
        } else if n >= 10_000 {
            format!("{:.1}K", n as f64 / 1e3)
        } else {
            n.to_string()
        }
    }

    fn row_label(bucket: usize) -> String {
        let lo = if bucket == 0 { 0u64 } else { 1 << bucket };
        if bucket == ROW_BUCKETS - 1 {
            format!(">={}", fmt_count(lo))
        } else {
            format!("{}-{}", fmt_count(lo), fmt_count((1 << (bucket + 1)) - 1))
        }
    }

    fn render_table(
        out: &mut String,
        title: &str,
        cells: &[[AtomicU64; USE_BUCKETS]; ROW_BUCKETS],
    ) {
        writeln!(out, "  {title} (rows \\ uses):").unwrap();
        write!(out, "  {:>12}", "").unwrap();
        for label in USE_LABELS {
            write!(out, " {label:>8}").unwrap();
        }
        writeln!(out).unwrap();
        for (rb, row) in cells.iter().enumerate() {
            if row.iter().all(|c| c.load(Ordering::Relaxed) == 0) {
                continue;
            }
            write!(out, "  {:>12}", row_label(rb)).unwrap();
            for cell in row {
                let v = cell.load(Ordering::Relaxed);
                if v == 0 {
                    write!(out, " {:>8}", ".").unwrap();
                } else {
                    write!(out, " {:>8}", fmt_count(v)).unwrap();
                }
            }
            writeln!(out).unwrap();
        }
    }

    /// Render the histograms collected so far.
    pub fn report() -> String {
        let mut out = String::new();
        writeln!(out, "=== join index build/usage stats ===").unwrap();
        for kind in 0..KINDS {
            for class in 0..CLASSES {
                let totals = &TOTALS[kind][class];
                let builds = totals[0].load(Ordering::Relaxed);
                if builds == 0 {
                    continue;
                }
                writeln!(
                    out,
                    "\n{} / {}: {} builds, {} rows built, {} probes, {} iters",
                    KIND_NAMES[kind],
                    CLASS_NAMES[class],
                    fmt_count(builds),
                    fmt_count(totals[1].load(Ordering::Relaxed)),
                    fmt_count(totals[2].load(Ordering::Relaxed)),
                    fmt_count(totals[3].load(Ordering::Relaxed)),
                )
                .unwrap();
                render_table(&mut out, "builds", &BUILDS[kind][class]);
                render_table(&mut out, "rows built", &ROWS[kind][class]);
            }
        }
        out
    }
}
