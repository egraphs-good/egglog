//! A table implementation backed by a union-find.

use std::{
    any::Any,
    mem::{self, ManuallyDrop},
    sync::{Arc, Weak},
};

use crate::numeric_id::{DenseIdMap, NumericId};
use crossbeam_queue::SegQueue;

use crate::{
    TableChange, TaggedRowBuffer,
    action::ExecutionState,
    common::{HashMap, Value},
    offsets::{OffsetRange, RowId, Subset, SubsetRef},
    pool::with_pool_set,
    row_buffer::RowBuffer,
    table_spec::{
        ColumnId, Constraint, Generation, MutationBuffer, Offset, Rebuilder, Row, Table, TableSpec,
        TableVersion, WrappedTableRef,
    },
};

#[cfg(test)]
mod tests;

type UnionFind = crate::union_find::UnionFind<Value>;

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
        let old_len = u32::try_from(out.len()).expect("row buffer sizes should fit in a u32");
        let _next = other.scan_bounded(subset, Offset::new(0), usize::MAX, out);
        debug_assert!(_next.is_none());
        for i in old_len..u32::try_from(out.len()).expect("row buffer sizes should fit in a u32") {
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
    to_insert: ManuallyDrop<RowBuffer>,
    buffered_writes: Weak<SegQueue<RowBuffer>>,
}

impl Drop for UfBuffer {
    fn drop(&mut self) {
        let Some(buffered_writes) = self.buffered_writes.upgrade() else {
            // SAFETY: If we can't write updates, manually drop to_insert
            unsafe {
                ManuallyDrop::drop(&mut self.to_insert);
            }
            return;
        };
        // SAFETY: self.to_insert will not be used again after this point.
        //
        // This avoids creating a fresh row buffer via `mem::take` or `mem::swap` and
        // dropping it immediately.
        let to_insert = unsafe { ManuallyDrop::take(&mut self.to_insert) };
        buffered_writes.push(to_insert);
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
            to_insert: ManuallyDrop::new(RowBuffer::new(self.to_insert.arity())),
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

    fn updates_since(&self, offset: Offset) -> Subset {
        Subset::Dense(OffsetRange::new(
            RowId::from_usize(offset.index()),
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
            to_insert: ManuallyDrop::new(RowBuffer::new(3)),
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
