//! A table implementation backed by a union-find.

use std::{
    any::Any,
    mem,
    sync::{Arc, Weak},
};

use crossbeam_queue::SegQueue;
use indexmap::IndexMap;
use numeric_id::{DenseIdMap, NumericId};
use petgraph::{algo::dijkstra, graph::NodeIndex, visit::EdgeRef, Direction, Graph};

use crate::{
    action::ExecutionState,
    common::{HashMap, IndexSet, Value},
    offsets::{OffsetRange, RowId, Subset, SubsetRef},
    pool::with_pool_set,
    row_buffer::RowBuffer,
    table_spec::{
        ColumnId, Constraint, Generation, MutationBuffer, Offset, Rebuilder, Row, Table, TableSpec,
        TableVersion, WrappedTableRef,
    },
    TableChange, TaggedRowBuffer,
};

#[cfg(test)]
mod tests;

type UnionFind = union_find::UnionFind<Value>;

/// A special table backed by a union-find used to efficiently implement
/// egglog-style canonicaliztion.
///
/// To canonicalize columns, we need to efficiently discover values that have
/// ceased to be canonical. To do that we keep a table of _displaced_ values:
///
/// This table has three columns:
/// 1. (the only key): a value that is _no longer canonical_ in the equivalence relation.
/// 2. The canonical value of the equivalence class.
/// 3. The timestamp at which the key stopped being canonical.
///
/// We do not store the second value explicitly: instead, we compute it
/// on-the-fly using a union-find data-structure.
///
/// This is related to the 'Leader' encoding in some versions of egglog:
/// Displaced is a version of Leader that _only_ stores ids when they cease to
/// be canonical. Rows are also "automatically updated" with the current leader,
/// rather than requiring the DB to replay history or canonicalize redundant
/// values in the table.
///
/// To union new ids `l`, and `r`, stage an update `Displaced(l, r, ts)` where
/// `ts` is the current timestamp. Note that all tie-breaks and other encoding
/// decisions are made internally, so there may not literally be a row added
/// with this value.
pub struct DisplacedTable {
    uf: UnionFind,
    displaced: Vec<(Value, Value)>,
    changed: bool,
    lookup_table: HashMap<Value, RowId>,
    buffered_writes: Arc<SegQueue<RowBuffer>>,
}

struct Canonicalizer<'a> {
    cols: Vec<ColumnId>,
    table: &'a DisplacedTable,
}

impl Rebuilder for Canonicalizer<'_> {
    fn hint_col(&self) -> Option<ColumnId> {
        Some(ColumnId::new(0))
    }
    fn rebuild_val(&self, val: Value) -> Value {
        self.table.uf.find_naive(val)
    }
    fn rebuild_buf(
        &self,
        buf: &RowBuffer,
        start: RowId,
        end: RowId,
        out: &mut TaggedRowBuffer,
        _exec_state: &mut ExecutionState,
    ) {
        if start >= end {
            return;
        }
        assert!(end.index() <= buf.len());
        let mut cur = start;
        let mut scratch = with_pool_set(|ps| ps.get::<Vec<Value>>());
        // SAFETY: `cur` is always in-bounds, guaranteed by the above assertion.
        // Special-case small columns: this gives us a modest speedup on rebuilding-heavy
        // workloads.
        match self.cols.as_slice() {
            [c] => {
                while cur < end {
                    let row = unsafe { buf.get_row_unchecked(cur) };
                    let to_canon = row[c.index()];
                    let canon = self.table.uf.find_naive(to_canon);
                    if canon != to_canon {
                        scratch.extend_from_slice(row);
                        scratch[c.index()] = canon;
                        out.add_row(cur, &scratch);
                        scratch.clear();
                    }
                    cur = cur.inc();
                }
            }
            [c1, c2] => {
                while cur < end {
                    let row = unsafe { buf.get_row_unchecked(cur) };
                    let v1 = row[c1.index()];
                    let v2 = row[c2.index()];
                    let ca1 = self.table.uf.find_naive(v1);
                    let ca2 = self.table.uf.find_naive(v2);
                    if ca1 != v1 || ca2 != v2 {
                        scratch.extend_from_slice(row);
                        scratch[c1.index()] = ca1;
                        scratch[c2.index()] = ca2;
                        out.add_row(cur, &scratch);
                        scratch.clear();
                    }
                    cur = cur.inc();
                }
            }
            [c1, c2, c3] => {
                while cur < end {
                    let row = unsafe { buf.get_row_unchecked(cur) };
                    let v1 = row[c1.index()];
                    let v2 = row[c2.index()];
                    let v3 = row[c3.index()];
                    let ca1 = self.table.uf.find_naive(v1);
                    let ca2 = self.table.uf.find_naive(v2);
                    let ca3 = self.table.uf.find_naive(v3);
                    if ca1 != v1 || ca2 != v2 || ca3 != v3 {
                        scratch.extend_from_slice(row);
                        scratch[c1.index()] = ca1;
                        scratch[c2.index()] = ca2;
                        scratch[c3.index()] = ca3;
                        out.add_row(cur, &scratch);
                        scratch.clear();
                    }
                    cur = cur.inc();
                }
            }
            cs => {
                while cur < end {
                    scratch.extend_from_slice(unsafe { buf.get_row_unchecked(cur) });
                    let mut changed = false;
                    for c in cs {
                        let to_canon = scratch[c.index()];
                        let canon = self.table.uf.find_naive(to_canon);
                        scratch[c.index()] = canon;
                        changed |= canon != to_canon;
                    }
                    if changed {
                        out.add_row(cur, &scratch);
                    }
                    scratch.clear();
                    cur = cur.inc();
                }
            }
        }
    }
    fn rebuild_subset(
        &self,
        other: WrappedTableRef,
        subset: SubsetRef,
        out: &mut TaggedRowBuffer,
        _exec_state: &mut ExecutionState,
    ) {
        let _next = other.scan_bounded(subset, Offset::new(0), usize::MAX, out);
        debug_assert!(_next.is_none());
        for i in 0..u32::try_from(out.len()).expect("row buffer sizes should fit in a u32") {
            let i = RowId::new(i);
            let (_id, row) = out.get_row_mut(i);
            let mut changed = false;
            for col in &self.cols {
                let to_canon = row[col.index()];
                let canon = self.table.uf.find_naive(to_canon);
                changed |= canon != to_canon;
                row[col.index()] = canon;
            }
            if !changed {
                out.set_stale(i);
            }
        }
    }
    fn rebuild_slice(&self, vals: &mut [Value]) -> bool {
        let mut changed = false;
        for val in vals {
            let canon = self.table.uf.find_naive(*val);
            changed |= canon != *val;
            *val = canon;
        }
        changed
    }
}

impl Default for DisplacedTable {
    fn default() -> Self {
        Self {
            uf: UnionFind::default(),
            displaced: Vec::new(),
            changed: false,
            lookup_table: HashMap::default(),
            buffered_writes: Arc::new(SegQueue::new()),
        }
    }
}

impl Clone for DisplacedTable {
    fn clone(&self) -> Self {
        DisplacedTable {
            uf: self.uf.clone(),
            displaced: self.displaced.clone(),
            changed: self.changed,
            lookup_table: self.lookup_table.clone(),
            buffered_writes: Default::default(),
        }
    }
}

struct UfBuffer {
    to_insert: RowBuffer,
    buffered_writes: Weak<SegQueue<RowBuffer>>,
}

impl Drop for UfBuffer {
    fn drop(&mut self) {
        let Some(buffered_writes) = self.buffered_writes.upgrade() else {
            return;
        };
        let arity = self.to_insert.arity();
        buffered_writes.push(mem::replace(&mut self.to_insert, RowBuffer::new(arity)));
    }
}

impl MutationBuffer for UfBuffer {
    fn stage_insert(&mut self, row: &[Value]) {
        self.to_insert.add_row(row);
    }
    fn stage_remove(&mut self, _: &[Value]) {
        panic!("attempting to remove data from a DisplacedTable")
    }
    fn fresh_handle(&self) -> Box<dyn MutationBuffer> {
        Box::new(UfBuffer {
            to_insert: RowBuffer::new(self.to_insert.arity()),
            buffered_writes: self.buffered_writes.clone(),
        })
    }
}

impl Table for DisplacedTable {
    fn dyn_clone(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn spec(&self) -> TableSpec {
        let mut uncacheable_columns = DenseIdMap::default();
        // The second column of this table is determined dynamically by the union-find.
        uncacheable_columns.insert(ColumnId::new(1), true);
        TableSpec {
            n_keys: 1,
            n_vals: 2,
            uncacheable_columns,
            allows_delete: false,
        }
    }

    fn rebuilder<'a>(&'a self, cols: &[ColumnId]) -> Option<Box<dyn Rebuilder + 'a>> {
        Some(Box::new(Canonicalizer {
            cols: cols.to_vec(),
            table: self,
        }))
    }

    fn clear(&mut self) {
        self.uf.reset();
        self.displaced.clear();
    }

    fn all(&self) -> Subset {
        Subset::Dense(OffsetRange::new(
            RowId::new(0),
            RowId::from_usize(self.displaced.len()),
        ))
    }

    fn len(&self) -> usize {
        self.displaced.len()
    }

    fn version(&self) -> TableVersion {
        TableVersion {
            major: Generation::new(0),
            minor: Offset::from_usize(self.displaced.len()),
        }
    }

    fn updates_since(&self, gen: Offset) -> Subset {
        Subset::Dense(OffsetRange::new(
            RowId::from_usize(gen.index()),
            RowId::from_usize(self.displaced.len()),
        ))
    }

    fn scan_generic_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        cs: &[Constraint],
        mut f: impl FnMut(RowId, &[Value]),
    ) -> Option<Offset>
    where
        Self: Sized,
    {
        if cs.is_empty() {
            let start = start.index();
            subset
                .iter_bounded(start, start + n, |row| {
                    f(row, self.expand(row).as_slice());
                })
                .map(Offset::from_usize)
        } else {
            let start = start.index();
            subset
                .iter_bounded(start, start + n, |row| {
                    if cs.iter().all(|c| self.eval(c, row)) {
                        f(row, self.expand(row).as_slice());
                    }
                })
                .map(Offset::from_usize)
        }
    }

    fn refine_one(&self, mut subset: Subset, c: &Constraint) -> Subset {
        subset.retain(|row| self.eval(c, row));
        subset
    }

    fn fast_subset(&self, constraint: &Constraint) -> Option<Subset> {
        let ts = ColumnId::new(2);
        match constraint {
            Constraint::Eq { .. } => None,
            Constraint::EqConst { col, val } => {
                if *col == ColumnId::new(1) {
                    return None;
                }
                if *col == ColumnId::new(0) {
                    return Some(match self.lookup_table.get(val) {
                        Some(row) => Subset::Dense(OffsetRange::new(
                            *row,
                            RowId::from_usize(row.index() + 1),
                        )),
                        None => Subset::empty(),
                    });
                }
                match self.timestamp_bounds(*val) {
                    Ok((start, end)) => Some(Subset::Dense(OffsetRange::new(start, end))),
                    Err(_) => None,
                }
            }
            Constraint::LtConst { col, val } => {
                if *col != ts {
                    return None;
                }
                match self.timestamp_bounds(*val) {
                    Err(bound) | Ok((bound, _)) => {
                        Some(Subset::Dense(OffsetRange::new(RowId::new(0), bound)))
                    }
                }
            }
            Constraint::GtConst { col, val } => {
                if *col != ts {
                    return None;
                }

                match self.timestamp_bounds(*val) {
                    Err(bound) | Ok((_, bound)) => Some(Subset::Dense(OffsetRange::new(
                        bound,
                        RowId::from_usize(self.displaced.len()),
                    ))),
                }
            }
            Constraint::LeConst { col, val } => {
                if *col != ts {
                    return None;
                }

                match self.timestamp_bounds(*val) {
                    Err(bound) | Ok((_, bound)) => {
                        Some(Subset::Dense(OffsetRange::new(RowId::new(0), bound)))
                    }
                }
            }
            Constraint::GeConst { col, val } => {
                if *col != ts {
                    return None;
                }

                match self.timestamp_bounds(*val) {
                    Err(bound) | Ok((bound, _)) => Some(Subset::Dense(OffsetRange::new(
                        bound,
                        RowId::from_usize(self.displaced.len()),
                    ))),
                }
            }
        }
    }

    fn get_row(&self, key: &[Value]) -> Option<Row> {
        assert_eq!(key.len(), 1, "attempt to lookup a row with the wrong key");
        let row_id = *self.lookup_table.get(&key[0])?;
        let mut vals = with_pool_set(|ps| ps.get::<Vec<Value>>());
        vals.extend_from_slice(self.expand(row_id).as_slice());
        Some(Row { id: row_id, vals })
    }

    fn get_row_column(&self, key: &[Value], col: ColumnId) -> Option<Value> {
        assert_eq!(key.len(), 1, "attempt to lookup a row with the wrong key");
        if col == ColumnId::new(1) {
            Some(self.uf.find_naive(key[0]))
        } else {
            let row_id = *self.lookup_table.get(&key[0])?;
            Some(self.expand(row_id)[col.index()])
        }
    }

    fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        Box::new(UfBuffer {
            to_insert: RowBuffer::new(3),
            buffered_writes: Arc::downgrade(&self.buffered_writes),
        })
    }

    fn merge(&mut self, _: &mut ExecutionState) -> TableChange {
        while let Some(rowbuf) = self.buffered_writes.pop() {
            for row in rowbuf.iter() {
                self.changed |= self.insert_impl(row).is_some();
            }
        }
        let changed = mem::take(&mut self.changed);
        // UF table rows can be updated "in place", we count both added and removed as changed in
        // this case.
        TableChange {
            added: changed,
            removed: changed,
        }
    }
}

impl DisplacedTable {
    pub fn underlying_uf(&self) -> &UnionFind {
        &self.uf
    }
    fn expand(&self, row: RowId) -> [Value; 3] {
        let (child, ts) = self.displaced[row.index()];
        [child, self.uf.find_naive(child), ts]
    }
    fn timestamp_bounds(&self, val: Value) -> Result<(RowId, RowId), RowId> {
        match self.displaced.binary_search_by_key(&val, |(_, ts)| *ts) {
            Ok(mut off) => {
                let mut next = off;
                while off > 0 && self.displaced[off - 1].1 == val {
                    off -= 1;
                }
                while next < self.displaced.len() && self.displaced[next].1 == val {
                    next += 1;
                }
                Ok((RowId::from_usize(off), RowId::from_usize(next)))
            }
            Err(off) => Err(RowId::from_usize(off)),
        }
    }
    fn eval(&self, constraint: &Constraint, row: RowId) -> bool {
        let vals = self.expand(row);
        eval_constraint(&vals, constraint)
    }
    fn insert_impl(&mut self, row: &[Value]) -> Option<(Value, Value)> {
        assert_eq!(row.len(), 3, "attempt to insert a row with the wrong arity");
        if self.uf.find(row[0]) == self.uf.find(row[1]) {
            return None;
        }
        let (parent, child) = self.uf.union(row[0], row[1]);

        // Compress paths somewhat, given that we perform naive finds everywhere else.
        let _ = self.uf.find(parent);
        let _ = self.uf.find(child);
        let ts = row[2];
        if let Some((_, highest)) = self.displaced.last() {
            assert!(
                *highest <= ts,
                "must insert rows with increasing timestamps"
            );
        }
        let next = RowId::from_usize(self.displaced.len());
        self.displaced.push((child, ts));
        self.lookup_table.insert(child, next);
        Some((parent, child))
    }
}

/// A variant of `DisplacedTable` that also stores "provenance" information that
/// can be used to generate proofs of equality.
///
/// This table expects a fourth "proof" column, though the values it hands back
/// _are not_ the proofs that come in and generally should not be used directly.
/// To generate a proof that two values are equal, this table exports a separate
/// `get_proof` method.
#[derive(Clone, Default)]
pub struct DisplacedTableWithProvenance {
    base: DisplacedTable,
    /// Added context for a given "displaced" row. We use this to store "proofs
    /// that x = y".
    ///
    /// N.B. We currently only use the first proof that we find. The remaining
    /// proofs are used for debugging. With some further refactoring we should
    /// be able to remove this field entirely, as complete proof information is
    /// now available through `proof_graph`.
    context: HashMap<(Value, Value), IndexSet<Value>>,
    proof_graph: Graph<Value, ProofEdge>,
    node_map: HashMap<Value, NodeIndex>,
    /// The value that was displaced, the value _immediately_ displacing it.
    /// NB: this is different from the 'displaced' table in 'base', which holds
    /// a timestamp.
    displaced: Vec<(Value, Value)>,
    buffered_writes: Arc<SegQueue<RowBuffer>>,
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct ProofEdge {
    reason: ProofReason,
    ts: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofStep {
    pub lhs: Value,
    pub rhs: Value,
    pub reason: ProofReason,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ProofReason {
    Forward(Value),
    Backward(Value),
}

impl DisplacedTableWithProvenance {
    fn expand(&self, row: RowId) -> [Value; 4] {
        let [v1, v2, v3] = self.base.expand(row);
        let (child, parent) = self.displaced[row.index()];
        debug_assert_eq!(child, v1);
        let proof = *self.context[&(child, parent)].get_index(0).unwrap();
        [v1, v2, v3, proof]
    }

    fn eval(&self, constraint: &Constraint, row: RowId) -> bool {
        eval_constraint(&self.expand(row), constraint)
    }

    /// Return the timestamp when `l` and `r` became equal.
    ///
    /// This is used to filter possible paths in the proof graph. The algorithm
    /// we use here is a variant of the classic algorithm in "Proof-Producing
    /// Congruence Closure" by Nieuwenhuis and Oliveras for reconstructing a
    /// proof.
    fn timestamp_when_equal(&self, l: Value, r: Value) -> Option<u32> {
        if l == r {
            return Some(0);
        }
        let mut l_proofs = IndexMap::new();
        let mut r_proofs = IndexMap::new();
        if self.base.uf.find_naive(l) != self.base.uf.find_naive(r) {
            // The two values aren't equal.
            return None;
        }
        let canon = self.base.uf.find_naive(l);

        // General case: collect individual equality proofs that point from `l`
        // (sim. `r`) and move towards canon. We stop early and don't always go
        // to `canon`. To see why consider the following sequences of unions.
        // For simplicity, we'll assume that the "leader" (or new canonical id)
        // is always the second argument to `union`.
        // * left:  A: union(0,2), B: union(2,4), C: union(4,6)
        // * right: D: union(1,3), E: union(3,5), F: union(5,4), C: union(4,6)
        // Where `l` `r` are 0 and 1, and their canonical value is `6`.
        // A simple approach here would be to simply glue the proofs that `l=6`
        // and `r=6` together, something like:
        //
        //    [A;B;C;rev(C);rev(F);rev(E);rev(D)]
        //
        // The code below avoids the redundant common suffix (i.e. `C;rev(C)`)
        // and just uses A,B,D,E, and F.
        //
        // In addition to allowing us to generate smaller proofs, this sort of
        // algorithm also ensures that we are returning the first proof of `l =
        // r` that we learned about, which is important for avoiding cycles when
        // reconstructing a proof.

        // General case: create a proof  that l = canon, then compose it with
        // the proof that r = canon, reversed.
        for (mut cur, steps) in [(l, &mut l_proofs), (r, &mut r_proofs)] {
            while cur != canon {
                // Find where cur became non-canonical.
                let row = *self.base.lookup_table.get(&cur).unwrap();
                let (_, ts) = self.base.displaced[row.index()];
                let (child, parent) = self.displaced[row.index()];
                debug_assert_eq!(child, cur);
                steps.insert(parent, ts);
                cur = parent;
            }
        }

        let mut l_end = None;
        let mut r_start = None;

        if let Some(i) = r_proofs.get_index_of(&l) {
            r_start = Some(i);
        } else {
            for (i, (next_id, _)) in l_proofs.iter().enumerate() {
                if *next_id == r {
                    l_end = Some(i);
                    break;
                }
                if let Some(j) = r_proofs.get_index_of(next_id) {
                    l_end = Some(i);
                    r_start = Some(j);
                    break;
                }
            }
        }
        match (l_end, r_start) {
            (None, Some(start)) => r_proofs.as_slice()[..=start]
                .iter()
                .map(|(_, ts)| ts.rep())
                .max(),
            (Some(end), None) => l_proofs.as_slice()[..=end]
                .iter()
                .map(|(_, ts)| ts.rep())
                .max(),
            (Some(end), Some(start)) => l_proofs.as_slice()[..=end]
                .iter()
                .map(|(_, ts)| ts.rep())
                .chain(r_proofs.as_slice()[..=start].iter().map(|(_, ts)| ts.rep()))
                .max(),
            (None, None) => {
                panic!("did not find common id, despite the values being equivalent {l:?} / {r:?}, l_proofs={l_proofs:?}, r_proofs={r_proofs:?}")
            }
        }
    }

    /// A simple proof generation algorithm that searches for the shortest path
    /// in the proof graph between `l` and `r`.
    ///
    /// The path in the graph is restricted to the timestamps at or before `l`
    /// and `r` first became equal. This is to avoid cycles during proof
    /// reconstruction.
    pub fn get_proof(&self, l: Value, r: Value) -> Option<Vec<ProofStep>> {
        let ts = self.timestamp_when_equal(l, r)?;
        let start = self.node_map[&l];
        let goal = self.node_map[&r];
        let costs = dijkstra(&self.proof_graph, self.node_map[&l], Some(goal), |edge| {
            if edge.weight().ts.rep() > ts {
                // avoid edges added after the two became equal.
                f64::INFINITY
            } else {
                1.0f64
            }
        });
        // Reconstruct the proof steps from the cost map returned from petgraph.
        // Start at the end and then work backwards along the shortest path.
        let mut path = Vec::new();
        let mut cur = goal;
        while cur != start {
            let (_, step, next) = self
                .proof_graph
                .edges_directed(cur, Direction::Incoming)
                .filter_map(|edge| {
                    let source = edge.source();
                    let cost = costs.get(&source)?;
                    let step = ProofStep {
                        lhs: *self.proof_graph.node_weight(source).unwrap(),
                        rhs: *self.proof_graph.node_weight(edge.target()).unwrap(),
                        reason: edge.weight().reason,
                    };
                    Some((cost, step, source))
                })
                .fold(None, |acc, cur| {
                    // Manually implement 'min' because we are using f64 for costs.
                    // We should probably switch these edge costs over to NotNan
                    // or a custom type.
                    let Some(acc) = acc else {
                        return Some(cur);
                    };
                    Some(if acc.0 > cur.0 { cur } else { acc })
                })
                .unwrap();
            path.push(step);
            cur = next;
        }
        path.reverse();
        Some(path)
    }
    fn get_or_create_node(&mut self, val: Value) -> NodeIndex {
        *self
            .node_map
            .entry(val)
            .or_insert_with(|| self.proof_graph.add_node(val))
    }

    fn insert_impl(&mut self, row: &[Value]) {
        let [a, b, ts, reason] = row else {
            panic!("attempt to insert a row with the wrong arity ({:?})", row);
        };
        match self.base.insert_impl(&[*a, *b, *ts]) {
            Some((parent, child)) => {
                self.displaced.push((child, parent));
                self.context
                    .entry((child, parent))
                    .or_default()
                    .insert(*reason);
                self.base.changed = true;

                let a_node = self.get_or_create_node(*a);
                let b_node = self.get_or_create_node(*b);
                self.proof_graph.add_edge(
                    a_node,
                    b_node,
                    ProofEdge {
                        reason: ProofReason::Forward(*reason),
                        ts: *ts,
                    },
                );
                self.proof_graph.add_edge(
                    b_node,
                    a_node,
                    ProofEdge {
                        reason: ProofReason::Backward(*reason),
                        ts: *ts,
                    },
                );
            }
            None => {
                self.context.entry((*a, *b)).or_default().insert(*reason);
                // We don't register a change, even if we learned a new proof.
                // We may want to change this behavior in order to search for
                // smaller proofs.
            }
        }
    }
}

impl Table for DisplacedTableWithProvenance {
    fn refine_one(&self, mut subset: Subset, c: &Constraint) -> Subset {
        subset.retain(|row| self.eval(c, row));
        subset
    }
    fn scan_generic_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        cs: &[Constraint],
        mut f: impl FnMut(RowId, &[Value]),
    ) -> Option<Offset>
    where
        Self: Sized,
    {
        if cs.is_empty() {
            let start = start.index();
            subset
                .iter_bounded(start, start + n, |row| {
                    f(row, self.expand(row).as_slice());
                })
                .map(Offset::from_usize)
        } else {
            let start = start.index();
            subset
                .iter_bounded(start, start + n, |row| {
                    if cs.iter().all(|c| self.eval(c, row)) {
                        f(row, self.expand(row).as_slice());
                    }
                })
                .map(Offset::from_usize)
        }
    }

    fn spec(&self) -> TableSpec {
        TableSpec {
            n_vals: 3,
            ..self.base.spec()
        }
    }

    fn merge(&mut self, exec_state: &mut ExecutionState) -> TableChange {
        while let Some(rowbuf) = self.buffered_writes.pop() {
            for row in rowbuf.iter() {
                self.insert_impl(row);
            }
        }

        self.base.merge(exec_state)
    }

    fn get_row(&self, key: &[Value]) -> Option<Row> {
        let mut inner = self.base.get_row(key)?;
        let (child, parent) = self.displaced[inner.id.index()];
        debug_assert_eq!(child, inner.vals[0]);
        let proof = *self.context[&(child, parent)].get_index(0).unwrap();
        inner.vals.push(proof);
        Some(inner)
    }

    fn get_row_column(&self, key: &[Value], col: ColumnId) -> Option<Value> {
        if col == ColumnId::new(3) {
            let row = *self.base.lookup_table.get(&key[0])?;
            Some(self.expand(row)[3])
        } else {
            self.base.get_row_column(key, col)
        }
    }

    fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        Box::new(UfBuffer {
            to_insert: RowBuffer::new(4),
            buffered_writes: Arc::downgrade(&self.buffered_writes),
        })
    }

    // Many of these methods just delgate to `base`:

    fn dyn_clone(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clear(&mut self) {
        self.base.clear()
    }
    fn all(&self) -> Subset {
        self.base.all()
    }
    fn len(&self) -> usize {
        self.base.len()
    }
    fn updates_since(&self, gen: Offset) -> Subset {
        self.base.updates_since(gen)
    }
    fn version(&self) -> TableVersion {
        self.base.version()
    }
    fn fast_subset(&self, c: &Constraint) -> Option<Subset> {
        self.base.fast_subset(c)
    }
}

fn eval_constraint<const N: usize>(vals: &[Value; N], constraint: &Constraint) -> bool {
    match constraint {
        Constraint::Eq { l_col, r_col } => vals[l_col.index()] == vals[r_col.index()],
        Constraint::EqConst { col, val } => vals[col.index()] == *val,
        Constraint::LtConst { col, val } => vals[col.index()] < *val,
        Constraint::GtConst { col, val } => vals[col.index()] > *val,
        Constraint::LeConst { col, val } => vals[col.index()] <= *val,
        Constraint::GeConst { col, val } => vals[col.index()] >= *val,
    }
}
