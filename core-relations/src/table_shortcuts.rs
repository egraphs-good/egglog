//! Utilities helpful in unit tests for manipulating tables.

use crate::numeric_id::NumericId;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use crate::{
    action::ExecutionState,
    common::Value,
    offsets::{RowId, Subset, SubsetRef},
    table::SortedWritesTable,
    table_spec::{
        ColumnId, Constraint, MutationBuffer, Offset, Row, Table, TableChange, TableSpec,
        TableVersion,
    },
};

/// A very basic macro just used to initialize an ExecutionState for tests that
/// do not actually use one.
macro_rules! empty_execution_state {
    ($es:ident) => {
        #[allow(unused_mut)]
        let mut __db = $crate::free_join::Database::default();
        #[allow(unused_mut)]
        let mut $es =
            $crate::action::ExecutionState::new(__db.read_only_view(), Default::default());
    };
}

/// Shorthand for generating a [`Value`].
pub(crate) fn v(n: usize) -> Value {
    Value::from_usize(n)
}

/// Fill a [`SortedWritesTable`] with the given rows, with conflicts resolved
/// with the given merge function.
pub(crate) fn fill_table(
    rows: impl IntoIterator<Item = Vec<Value>>,
    n_keys: usize,
    sort_by: ColumnId,
    merge_fn: impl Fn(&[Value], &[Value]) -> Option<Vec<Value>> + 'static + Send + Sync,
) -> SortedWritesTable {
    empty_execution_state!(e);
    let mut iter = rows.into_iter();
    let init = iter.next().expect("iterator must be nonempty");
    let n_cols = init.len();
    assert!(n_cols >= n_keys, "must have at least {n_keys} columns");
    let merge = Box::new(
        move |_: &mut crate::ExecutionState<'_>,
              old: &[Value],
              new: &[Value],
              out: &mut Vec<Value>| {
            if let Some(res) = merge_fn(old, new) {
                *out = res;
                true
            } else {
                false
            }
        },
    );
    let mut table = SortedWritesTable::new(n_keys, n_cols, sort_by, vec![], merge);

    // We write tests that assume that assume rows are inserted in order.
    // SortedWritesTable does not preserver ordering of writes between two calls
    // to merge, so we create a buffer and merge for each insertion.
    table.new_buffer().stage_insert(&init);
    table.merge(&mut e);
    for row in iter {
        table.new_buffer().stage_insert(&row);
        table.merge(&mut e);
    }
    table
}

/// Test table that supplies a timestamp while exposing only the logical columns.
/// Each merge advances the timestamp for subsequently staged rows.
#[derive(Clone)]
pub(crate) struct TimestampedTable {
    inner: SortedWritesTable,
    columns: usize,
    timestamp: Arc<AtomicU32>,
}

impl TimestampedTable {
    pub(crate) fn new(n_keys: usize, columns: usize, merge: Box<crate::table::MergeFn>) -> Self {
        let wrapped = Box::new(
            move |state: &mut ExecutionState<'_>,
                  old: &[Value],
                  new: &[Value],
                  out: &mut Vec<Value>| {
                let changed = merge(state, &old[..columns], &new[..columns], out);
                if changed {
                    assert_eq!(out.len(), columns);
                    out.push(new[columns]);
                }
                changed
            },
        );
        Self {
            inner: SortedWritesTable::new(
                n_keys,
                columns + 1,
                ColumnId::from_usize(columns),
                vec![],
                wrapped,
            ),
            columns,
            timestamp: Arc::new(AtomicU32::new(0)),
        }
    }
}

struct TimestampBuffer {
    inner: Box<dyn MutationBuffer>,
    columns: usize,
    timestamp: Arc<AtomicU32>,
}

impl MutationBuffer for TimestampBuffer {
    fn stage_insert(&mut self, row: &[Value]) {
        assert_eq!(row.len(), self.columns);
        let mut stamped = row.to_vec();
        stamped.push(Value::new(self.timestamp.load(Ordering::Relaxed)));
        self.inner.stage_insert(&stamped);
    }

    fn stage_remove(&mut self, key: &[Value]) {
        self.inner.stage_remove(key);
    }

    fn fresh_handle(&self) -> Box<dyn MutationBuffer> {
        Box::new(Self {
            inner: self.inner.fresh_handle(),
            columns: self.columns,
            timestamp: self.timestamp.clone(),
        })
    }
}

impl Table for TimestampedTable {
    fn dyn_clone(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn spec(&self) -> TableSpec {
        let mut spec = self.inner.spec();
        spec.n_vals -= 1;
        spec
    }
    fn clear(&mut self) {
        self.inner.clear()
    }
    fn all(&self) -> Subset {
        self.inner.all()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn version(&self) -> TableVersion {
        self.inner.version()
    }
    fn updates_since(&self, offset: Offset) -> Subset {
        self.inner.updates_since(offset)
    }
    fn scan_generic_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        cs: &[Constraint],
        mut f: impl FnMut(RowId, &[Value]),
    ) -> Option<Offset> {
        self.inner
            .scan_generic_bounded(subset, start, n, cs, |id, row| f(id, &row[..self.columns]))
    }
    fn has_stale_rows(&self) -> bool {
        self.inner.has_stale_rows()
    }
    fn refine_one(&self, subset: Subset, c: &Constraint) -> Subset {
        self.inner.refine_one(subset, c)
    }
    fn refine_ref(&self, subset: SubsetRef, cs: &[Constraint], check_live: bool) -> Subset {
        self.inner.refine_ref(subset, cs, check_live)
    }
    fn fast_subset(&self, c: &Constraint) -> Option<Subset> {
        self.inner.fast_subset(c)
    }
    fn get_row(&self, key: &[Value]) -> Option<Row> {
        let mut row = self.inner.get_row(key)?;
        row.vals.truncate(self.columns);
        Some(row)
    }
    fn merge(&mut self, state: &mut ExecutionState) -> TableChange {
        let change = self.inner.merge(state);
        self.timestamp.fetch_add(1, Ordering::Relaxed);
        change
    }
    fn new_buffer(&self) -> Box<dyn MutationBuffer> {
        Box::new(TimestampBuffer {
            inner: self.inner.new_buffer(),
            columns: self.columns,
            timestamp: self.timestamp.clone(),
        })
    }
}

pub(crate) fn fill_timestamped_table(
    rows: impl IntoIterator<Item = Vec<Value>>,
    n_keys: usize,
    merge_fn: impl Fn(&[Value], &[Value]) -> Option<Vec<Value>> + 'static + Send + Sync,
) -> TimestampedTable {
    empty_execution_state!(e);
    let mut iter = rows.into_iter();
    let init = iter.next().expect("iterator must be nonempty");
    let columns = init.len();
    let merge = Box::new(
        move |_: &mut ExecutionState<'_>, old: &[Value], new: &[Value], out: &mut Vec<Value>| {
            if let Some(result) = merge_fn(old, new) {
                *out = result;
                true
            } else {
                false
            }
        },
    );
    let mut table = TimestampedTable::new(n_keys, columns, merge);
    table.new_buffer().stage_insert(&init);
    table.merge(&mut e);
    for row in iter {
        table.new_buffer().stage_insert(&row);
        table.merge(&mut e);
    }
    table
}
