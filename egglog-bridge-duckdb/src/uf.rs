//! Native union-find used by the `--duck-native-uf` path.
//!
//! Eager flattening: every union walks the larger root's subtree
//! and points everyone directly at the smaller root. As a result,
//! the tree is always a single flat level — every non-root node
//! has `parent[x]` equal to *the* root of its class. `find_ro(x)`
//! is therefore O(1): one hash lookup. No path compression needed
//! because there's no chain to compress.
//!
//! The tradeoff vs lazy union-by-min: union is O(class size of
//! larger root) instead of O(α(n)). For workloads with many finds
//! per union (rebuild rules scan every view row × K eq-sort
//! columns × M iterations), the per-find savings dwarf the
//! per-union overhead. As a bonus, eager flattening makes
//! computing the *displaced* set free: every node walked during
//! the subtree flatten is, by definition, a node whose canonical
//! mapping just changed. That set is exactly what an incremental-
//! rebuild pass needs.
//!
//! Invariant: at every observable point (between `drain_pending`
//! calls), the tree is flat. `children[root]` lists every non-root
//! in the class; no entry of `children` ever holds a non-root key.
//!
//! IDs are i64, drawn from the global `__egglog_eqsort_seq`
//! DuckDB sequence so cross-sort collisions don't happen — but we
//! still keep a separate `UfTable` per sort for clarity and so the
//! UDF name namespaces by sort.

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct UfTable {
    /// Direct child → root map. By the flat-tree invariant, every
    /// non-root `x` has `parent[x] = find_ro(x)`. A root is either
    /// absent from this map or maps to itself.
    parent: HashMap<i64, i64>,
    /// Root → direct children. Maintained by `drain_pending` so we
    /// can enumerate an eclass in O(class size) for displaced
    /// tracking and (future) incremental rebuild.
    children: HashMap<i64, Vec<i64>>,
    /// Queued unions, applied together by `drain_pending`. Lets
    /// rule actions stage union intents that all take effect at an
    /// iteration boundary instead of mid-`SELECT` — matches legacy
    /// `egglog-bridge`'s staged-action semantics.
    pending: Vec<(i64, i64)>,
    /// IDs whose canonical changed during the most recent
    /// `drain_pending`. Includes the larger root being demoted
    /// *and* every former member of the larger class (they now
    /// point at the smaller root instead of their old root). The
    /// runner reads this via `drain_displaced` to feed an
    /// incremental-rebuild path.
    displaced: Vec<i64>,
    /// Cumulative `enqueue_union` call count. The runner snapshots
    /// this around each rule action so UDF-mediated unions (which
    /// `conn.execute` reports as 0 affected rows because the
    /// surrounding statement is a SELECT) still contribute to the
    /// saturate loop's stop condition.
    union_calls: u64,
}

impl UfTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read-only find. Single hash lookup thanks to the flat-tree
    /// invariant. Returns `x` for ids that have never been
    /// unioned.
    pub fn find_ro(&self, x: i64) -> i64 {
        match self.parent.get(&x) {
            Some(&p) => p,
            None => x,
        }
    }

    /// Backward-compatible alias for `find_ro` — eager UF makes
    /// the read-only path equivalent to the mutating path (there
    /// is no chain to compress).
    pub fn find(&mut self, x: i64) -> i64 {
        self.find_ro(x)
    }

    /// Stage a union for `drain_pending` to apply at the next
    /// iteration boundary. Increments `union_calls` so the runner
    /// can detect that UF state advanced even when the host SQL
    /// statement was a SELECT.
    pub fn enqueue_union(&mut self, a: i64, b: i64) {
        self.pending.push((a, b));
        self.union_calls = self.union_calls.wrapping_add(1);
    }

    /// Apply all pending unions with eager flattening. Returns the
    /// number of staged unions that were processed (not the number
    /// of nodes whose canonical changed — see `displaced_len` for
    /// that).
    pub fn drain_pending(&mut self) -> usize {
        let n = self.pending.len();
        let pending = std::mem::take(&mut self.pending);
        for (a, b) in pending {
            let ra = self.find_ro(a);
            let rb = self.find_ro(b);
            if ra == rb {
                continue;
            }
            let (smaller, larger) = if ra < rb { (ra, rb) } else { (rb, ra) };
            // Larger's subtree by the flat-tree invariant: just
            // `larger` itself plus the (already-flat) list of its
            // direct children.
            let mut subtree: Vec<i64> = Vec::with_capacity(1);
            subtree.push(larger);
            if let Some(cs) = self.children.remove(&larger) {
                subtree.extend(cs);
            }
            for &node in &subtree {
                self.parent.insert(node, smaller);
                self.displaced.push(node);
            }
            self.children
                .entry(smaller)
                .or_default()
                .extend(subtree);
        }
        n
    }

    /// Total `enqueue_union` calls so far.
    pub fn union_calls(&self) -> u64 {
        self.union_calls
    }

    /// Number of IDs displaced (canonical changed) since the last
    /// `drain_displaced` call.
    pub fn displaced_len(&self) -> usize {
        self.displaced.len()
    }

    /// Take the set of IDs displaced since the last call. Caller
    /// owns the returned `Vec`; internal list resets to empty.
    pub fn drain_displaced(&mut self) -> Vec<i64> {
        std::mem::take(&mut self.displaced)
    }

    /// Total IDs currently tracked (roots are absent from
    /// `parent`, so this counts only non-roots).
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_is_own_root() {
        let uf = UfTable::new();
        assert_eq!(uf.find_ro(7), 7);
    }

    #[test]
    fn union_keeps_smaller() {
        let mut uf = UfTable::new();
        uf.enqueue_union(5, 3);
        uf.drain_pending();
        assert_eq!(uf.find_ro(5), 3);
        assert_eq!(uf.find_ro(3), 3);
    }

    #[test]
    fn chained_unions_collapse_to_min() {
        let mut uf = UfTable::new();
        uf.enqueue_union(10, 7);
        uf.drain_pending();
        uf.enqueue_union(7, 4);
        uf.drain_pending();
        uf.enqueue_union(4, 2);
        uf.drain_pending();
        for x in [10, 7, 4, 2] {
            assert_eq!(uf.find_ro(x), 2, "find({x}) should be 2");
        }
    }

    #[test]
    fn order_independence() {
        let mut a = UfTable::new();
        a.enqueue_union(1, 2);
        a.enqueue_union(2, 3);
        a.drain_pending();

        let mut b = UfTable::new();
        b.enqueue_union(2, 3);
        b.enqueue_union(1, 2);
        b.drain_pending();

        for x in [1, 2, 3] {
            assert_eq!(a.find_ro(x), b.find_ro(x));
        }
    }

    #[test]
    fn flat_tree_after_chained_unions() {
        let mut uf = UfTable::new();
        // Build {1, 2, 3, 4} all in one class through several
        // staged unions.
        uf.enqueue_union(2, 1);
        uf.enqueue_union(3, 2);
        uf.enqueue_union(4, 3);
        uf.drain_pending();
        // Every non-root must point directly at the root.
        for x in [2, 3, 4] {
            assert_eq!(
                uf.find_ro(x),
                1,
                "{x} should point directly at 1; got {}",
                uf.find_ro(x)
            );
        }
    }

    #[test]
    fn displaced_set_is_class_being_merged() {
        let mut uf = UfTable::new();
        // Build class {10, 7, 4} with root 4.
        uf.enqueue_union(10, 7);
        uf.drain_pending();
        uf.enqueue_union(7, 4);
        uf.drain_pending();
        let _ = uf.drain_displaced(); // reset

        // Build class {6, 2} with root 2.
        uf.enqueue_union(6, 2);
        uf.drain_pending();
        let _ = uf.drain_displaced();

        // Merge: now {2, 4, 6, 7, 10} with root 2. The displaced
        // set must include every member of the larger class
        // {4, 7, 10}, including its former root 4.
        uf.enqueue_union(4, 2);
        uf.drain_pending();
        let mut d = uf.drain_displaced();
        d.sort();
        assert_eq!(d, vec![4, 7, 10]);
        for x in [4, 7, 10] {
            assert_eq!(uf.find_ro(x), 2);
        }
    }

    #[test]
    fn drain_pending_handles_already_merged() {
        let mut uf = UfTable::new();
        uf.enqueue_union(5, 3);
        uf.drain_pending();
        let _ = uf.drain_displaced();
        // Re-asserting an existing union must not displace
        // anything new.
        uf.enqueue_union(5, 3);
        uf.drain_pending();
        assert_eq!(uf.drain_displaced(), Vec::<i64>::new());
    }
}
