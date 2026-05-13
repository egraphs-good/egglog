//! Native union-find used by the `DUCK_NATIVE_UF` experimental
//! path. Replaces the SQL-table-backed UF that the term encoding's
//! `@_u_f___<sort>f` function-form table maintains: instead of
//! growing a relation + running singleparent/path_compress/
//! uf_function_index rules to canonicalize it, we keep parent
//! pointers in a `HashMap`, expose `find` via a registered DuckDB
//! scalar UDF, and queue unions to apply between SQL statements
//! (rather than mid-stream where DuckDB's order/parallelism story
//! gets murky).
//!
//! `find` uses path compression; `union` keeps the smaller-ID root,
//! matching the semantics the egglog term encoding expects from
//! `:merge (ordering-min old new)`.
//!
//! IDs are i64, drawn from the global `__egglog_eqsort_seq`
//! DuckDB sequence so cross-sort collisions don't happen — but we
//! still keep a separate `UfTable` per sort for clarity and so the
//! UDF name namespaces by sort.

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct UfTable {
    /// Parent map. An ID that's its own parent (or is missing) is
    /// a root.
    parent: HashMap<i64, i64>,
    /// Queued unions, applied between SQL statements.
    pending: Vec<(i64, i64)>,
}

impl UfTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find with path compression. An ID that's never been seen is
    /// its own root.
    pub fn find(&mut self, mut x: i64) -> i64 {
        // First, walk to the root.
        let mut root = x;
        while let Some(&p) = self.parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        // Path-compress: point everything on the walk directly at
        // the root.
        while let Some(&p) = self.parent.get(&x) {
            if p == root {
                break;
            }
            self.parent.insert(x, root);
            x = p;
        }
        root
    }

    /// Read-only variant: doesn't write path-compression entries.
    /// Used by the find UDF when DuckDB might call it concurrently
    /// from many rows; we accept the slower walk to keep find
    /// fully pure.
    pub fn find_ro(&self, mut x: i64) -> i64 {
        loop {
            match self.parent.get(&x) {
                Some(&p) if p != x => x = p,
                _ => return x,
            }
        }
    }

    /// Queue a union. Applied later by `drain_pending`.
    pub fn enqueue_union(&mut self, a: i64, b: i64) {
        self.pending.push((a, b));
    }

    /// Apply all queued unions. Smaller ID becomes the root —
    /// matches `:merge (ordering-min old new)` semantics.
    pub fn drain_pending(&mut self) -> usize {
        let n = self.pending.len();
        let pending = std::mem::take(&mut self.pending);
        for (a, b) in pending {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra == rb {
                continue;
            }
            let (smaller, larger) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent.insert(larger, smaller);
        }
        n
    }

    /// Number of (input, root) pairs currently tracked.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Drain pending and return the new resolved roots for every
    /// known ID. Used by the runner to mirror UF state into a SQL
    /// table when something else needs to read it via JOIN rather
    /// than the UDF.
    pub fn snapshot(&mut self) -> Vec<(i64, i64)> {
        self.drain_pending();
        let keys: Vec<i64> = self.parent.keys().copied().collect();
        keys.into_iter()
            .map(|k| (k, self.find(k)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_is_own_root() {
        let mut uf = UfTable::new();
        assert_eq!(uf.find(7), 7);
    }

    #[test]
    fn union_keeps_smaller() {
        let mut uf = UfTable::new();
        uf.enqueue_union(5, 3);
        uf.drain_pending();
        assert_eq!(uf.find(5), 3);
        assert_eq!(uf.find(3), 3);
    }

    #[test]
    fn chained_unions_collapse_to_min() {
        let mut uf = UfTable::new();
        uf.enqueue_union(10, 7);
        uf.enqueue_union(7, 4);
        uf.enqueue_union(4, 2);
        uf.drain_pending();
        for x in [10, 7, 4, 2] {
            assert_eq!(uf.find(x), 2, "find({x}) should be 2");
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
            assert_eq!(a.find(x), b.find(x));
        }
    }

    #[test]
    fn find_ro_matches_find() {
        let mut uf = UfTable::new();
        uf.enqueue_union(10, 7);
        uf.enqueue_union(7, 4);
        uf.drain_pending();
        for x in [10, 7, 4] {
            assert_eq!(uf.find_ro(x), uf.find(x));
        }
    }
}
