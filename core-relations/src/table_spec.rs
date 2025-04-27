//! High-level types for specifying the behavior and layout of tables.
//!
//! Tables are a mapping from some set of keys to another set of values. Tables
//! can also be "sorted by" a columna dn "partitioned by" another. This can help
//! speed up queries.

use std::{
    any::Any,
    iter,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use numeric_id::{define_id, DenseIdMap, NumericId};
use smallvec::SmallVec;

use crate::{
    action::{
        mask::{Mask, MaskIter, ValueSource},
        Bindings, ExecutionState,
    },
    common::Value,
    hash_index::{ColumnIndex, IndexBase, TupleIndex},
    offsets::{RowId, Subset, SubsetRef},
    pool::{with_pool_set, PoolSet, Pooled},
    row_buffer::{RowBuffer, TaggedRowBuffer},
    QueryEntry, TableId, Variable,
};

define_id!(pub ColumnId, u32, "a particular column in a table");
define_id!(
    pub Generation,
    u64,
    "the current version of a table -- used to invalidate any existing RowIds"
);
define_id!(
    pub Offset,
    u64,
    "an opaque offset token -- used to encode iterations over a table (within a generation). These always start at 0."
);

/// The version of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableVersion {
    /// New major generations invalidate all existing RowIds for a table.
    pub major: Generation,
    /// New minor generations within a major generation do not invalidate
    /// existing RowIds, but they may indicate that `all` can return a larger
    /// subset than before.
    pub minor: Offset,
    // NB: we may want to make `Offset` and `RowId` the same.
}

#[derive(Clone)]
pub struct TableSpec {
    /// The number of key columns for the table.
    pub n_keys: usize,

    /// The number of non-key (i.e. value) columns in the table.
    ///
    /// The total "arity" of the table is `n_keys + n_vals`.
    pub n_vals: usize,

    /// Columns that cannot be cached across generations.
    ///
    /// These columns should (e.g.) never have indexes built for them, as they
    /// will go out of date too quickly.
    pub uncacheable_columns: DenseIdMap<ColumnId, bool>,

    /// Whether or not deletions are supported for this table.
    ///
    /// Tables where this value is false are allowed to panic on calls to
    /// `stage_remove`.
    pub allows_delete: bool,
}

impl TableSpec {
    /// The total number of columns stored by the table.
    pub fn arity(&self) -> usize {
        self.n_keys + self.n_vals
    }
}

/// A summary of the kinds of changes that a table underwent after a merge operation.
#[derive(Eq, PartialEq, Copy, Clone)]
pub struct TableChange {
    /// Whether or not rows were added to the table.
    pub added: bool,
    /// Whether or not rows were removed from the table.
    pub removed: bool,
}

/// A constraint on the values within a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constraint {
    Eq { l_col: ColumnId, r_col: ColumnId },
    EqConst { col: ColumnId, val: Value },
    LtConst { col: ColumnId, val: Value },
    GtConst { col: ColumnId, val: Value },
    LeConst { col: ColumnId, val: Value },
    GeConst { col: ColumnId, val: Value },
}

/// Custom functions used for tables that encode a bulk value-level rebuild of other tables.
///
/// The initial use-case for this trait is to support optimized implementations of rebuilding,
/// where `Rebuilder` is implemented as a Union-find.
///
/// Value-level rebuilds are difficult to implement efficiently using rules as they require
/// searching for changes to any column for a table: while it is possible to do, implementing this
/// custom is more efficient in the case of rebuilding.
pub trait Rebuilder: Send + Sync {
    /// The column that contains values that should be rebuilt. If this is set, callers can use
    /// this functionality to perform rebuilds incrementally.
    fn hint_col(&self) -> Option<ColumnId>;
    fn rebuild_val(&self, val: Value) -> Value;
    /// Rebuild a contiguous slice of rows in the table.
    fn rebuild_buf(
        &self,
        buf: &RowBuffer,
        start: RowId,
        end: RowId,
        out: &mut TaggedRowBuffer,
        exec_state: &mut ExecutionState,
    );
    /// Rebuild an arbitrary subset of the table.
    fn rebuild_subset(
        &self,
        other: WrappedTableRef,
        subset: SubsetRef,
        out: &mut TaggedRowBuffer,
        exec_state: &mut ExecutionState,
    );
    /// Rebuild a slice of values in place, returning true if any values were changed.
    fn rebuild_slice(&self, vals: &mut [Value]) -> bool;
}

/// A row in a table.
pub struct Row {
    /// The id associated with the row.
    pub id: RowId,
    /// The Row itself.
    pub vals: Pooled<Vec<Value>>,
}

/// An interface for a table.
pub trait Table: Any + Send + Sync {
    /// A variant of clone that returns a boxed trait object; this trait object
    /// must contain all of the data associated with the current table.
    fn dyn_clone(&self) -> Box<dyn Table>;

    /// If this table can perform a table-level rebuild, construct a [`Rebuilder`] for it.
    fn rebuilder<'a>(&'a self, _cols: &[ColumnId]) -> Option<Box<dyn Rebuilder + 'a>> {
        None
    }

    /// Rebuild the table according to the given [`Rebuilder`] implemented by `table`, if
    /// there is one. Applying a rebuild can cause more mutations to be buffered, which can in turn
    /// be flushed by a call to [`Table::merge`].
    ///
    /// Note that value-level rebuilds are only relevant for tables that opt into it. As a result,
    /// tables do nothing by default.
    fn apply_rebuild(
        &mut self,
        _table_id: TableId,
        _table: &WrappedTable,
        _next_ts: Value,
        _exec_state: &mut ExecutionState,
    ) {
        // Default implementation does nothing.
    }

    /// A boilerplate method to make it easier to downcast values of `Table`.
    ///
    /// Implementors should be able to implement this method by returning
    /// `self`.
    fn as_any(&self) -> &dyn Any;

    /// The schema of the table.
    ///
    /// These are immutable properties of the table; callers can assume they
    /// will never change.
    fn spec(&self) -> TableSpec;

    /// Clear all table contents. If the table is nonempty, this will change the
    /// generation of the table. This method also clears any pending data.
    fn clear(&mut self);

    // Used in queries:

    /// Get a subset corresponding to all rows in the table.
    fn all(&self) -> Subset;

    /// Get the length of the table.
    ///
    /// This is not in general equal to the length of the `all` subset: the size
    /// of a subset is allowed to be larger than the number of table entries in
    /// range of the subset.
    fn len(&self) -> usize;

    /// Check if the table is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the current version for the table. [`RowId`]s and [`Subset`]s are
    /// only valid for a given major generation.
    fn version(&self) -> TableVersion;

    /// Get the subset of the table that has appeared since the last offset.
    fn updates_since(&self, gen: Offset) -> Subset;

    /// Iterate over the given subset of the table, starting at an opaque
    /// `start` token, ending after up to `n` rows, returning the next start
    /// token if more rows remain. Only invoke `f` on rows that match the given
    /// constraints.
    ///
    /// This method is _not_ object safe, but it is used to define various
    /// "default" implementations of object-safe methods like `scan` and
    /// `pivot`.
    fn scan_generic_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        cs: &[Constraint],
        f: impl FnMut(RowId, &[Value]),
    ) -> Option<Offset>
    where
        Self: Sized;

    /// Iterate over the given subset of the table.
    ///
    /// This is a variant of [`Table::scan_generic_bounded`] that iterates over
    /// the entire table.
    fn scan_generic(&self, subset: SubsetRef, mut f: impl FnMut(RowId, &[Value]))
    where
        Self: Sized,
    {
        let mut cur = Offset::new(0);
        while let Some(next) = self.scan_generic_bounded(subset, cur, usize::MAX, &[], |id, row| {
            f(id, row);
        }) {
            cur = next;
        }
    }

    /// Filter a given subset of the table for the rows matching the single constraint.
    ///
    /// Implementors must provide at least one of `refine_one` or `refine`.`
    fn refine_one(&self, subset: Subset, c: &Constraint) -> Subset {
        self.refine(subset, std::slice::from_ref(c))
    }

    /// Filter a given subset of the table for the rows matching the given constraints.
    ///
    /// Implementors must provide at least one of `refine_one` or `refine`.`
    fn refine(&self, subset: Subset, cs: &[Constraint]) -> Subset {
        cs.iter()
            .fold(subset, |subset, c| self.refine_one(subset, c))
    }

    /// An optional method for quickly generating a subset from a constraint.
    /// The standard use-case here is to apply constraints based on a column
    /// that is known to be sorted.
    ///
    /// These constraints are very helpful for query planning; it is a good idea
    /// to implement them.
    fn fast_subset(&self, _: &Constraint) -> Option<Subset> {
        None
    }

    /// A helper routine that leverages the existing `fast_subset` method to
    /// preprocess a set of constraints into "fast" and "slow" ones, returning
    /// the subet of indexes that match the fast one.
    fn split_fast_slow(
        &self,
        cs: &[Constraint],
    ) -> (
        Subset,                  /* the subset of the table matching all fast constraints */
        Pooled<Vec<Constraint>>, /* the fast constraints */
        Pooled<Vec<Constraint>>, /* the slow constraints */
    ) {
        with_pool_set(|ps| {
            let mut fast = ps.get::<Vec<Constraint>>();
            let mut slow = ps.get::<Vec<Constraint>>();
            let mut subset = self.all();
            for c in cs {
                if let Some(sub) = self.fast_subset(c) {
                    subset.intersect(sub.as_ref(), &ps.get_pool());
                    fast.push(c.clone());
                } else {
                    slow.push(c.clone());
                }
            }
            (subset, fast, slow)
        })
    }

    // Used in actions:

    /// Look up a single row by the given key values, if it is in the table.
    ///
    /// The number of values specified by `keys` should match the number of
    /// primary keys for the table.
    fn get_row(&self, key: &[Value]) -> Option<Row>;

    /// Look up the given column of single row by the given key values, if it is
    /// in the table.
    ///
    /// The number of values specified by `keys` should match the number of
    /// primary keys for the table.
    fn get_row_column(&self, key: &[Value], col: ColumnId) -> Option<Value> {
        self.get_row(key).map(|row| row.vals[col.index()])
    }

    /// Merge any updates to the table, and potentially update the generation for
    /// the table.
    fn merge(&mut self, exec_state: &mut ExecutionState) -> TableChange;

    /// Create a new buffer for staging mutations on this table.
    fn new_buffer(&self) -> Box<dyn MutationBuffer>;
}

/// A trait specifying a buffer of pending mutations for a [`Table`].
///
/// Dropping an object implementing this trait should "flush" the pending
/// mutations to the table. Calling  [`Table::merge`] on that table would then
/// apply those mutations, making them visible for future readers.
pub trait MutationBuffer: Any + Send + Sync {
    /// Stage the keyed entries for insertion. Changes may not be visible until
    /// this buffer is dropped, and after `merge` is called on the underlying
    /// table.
    fn stage_insert(&mut self, row: &[Value]);

    /// Stage the keyed entries for removal. Changes may not be visible until
    /// this buffer is dropped, and after `merge` is called on the underlying
    /// table.
    fn stage_remove(&mut self, key: &[Value]);

    /// Get a fresh handle to the same table.
    fn fresh_handle(&self) -> Box<dyn MutationBuffer>;
}

struct WrapperImpl<T>(PhantomData<T>);

pub(crate) fn wrapper<T: Table>() -> Box<dyn TableWrapper> {
    Box::new(WrapperImpl::<T>(PhantomData))
}

impl<T: Table> TableWrapper for WrapperImpl<T> {
    fn dyn_clone(&self) -> Box<dyn TableWrapper> {
        Box::new(Self(PhantomData))
    }
    fn scan_bounded(
        &self,
        table: &dyn Table,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset> {
        let table = table.as_any().downcast_ref::<T>().unwrap();
        table.scan_generic_bounded(subset, start, n, &[], |row_id, row| {
            out.add_row(row_id, row);
        })
    }
    fn group_by_col(&self, table: &dyn Table, subset: SubsetRef, col: ColumnId) -> ColumnIndex {
        let table = table.as_any().downcast_ref::<T>().unwrap();
        let mut res = ColumnIndex::new();
        table.scan_generic(subset, |row_id, row| {
            res.add_row(&[row[col.index()]], row_id);
        });
        res
    }
    fn group_by_key(&self, table: &dyn Table, subset: SubsetRef, cols: &[ColumnId]) -> TupleIndex {
        let table = table.as_any().downcast_ref::<T>().unwrap();
        let mut res = TupleIndex::new(cols.len());
        match cols {
            [] => {}
            [col] => table.scan_generic(subset, |row_id, row| {
                res.add_row(&[row[col.index()]], row_id);
            }),
            [x, y] => table.scan_generic(subset, |row_id, row| {
                res.add_row(&[row[x.index()], row[y.index()]], row_id);
            }),
            [x, y, z] => table.scan_generic(subset, |row_id, row| {
                res.add_row(&[row[x.index()], row[y.index()], row[z.index()]], row_id);
            }),
            _ => {
                let mut scratch = SmallVec::<[Value; 8]>::new();
                table.scan_generic(subset, |row_id, row| {
                    for col in cols {
                        scratch.push(row[col.index()]);
                    }
                    res.add_row(&scratch, row_id);
                    scratch.clear();
                });
            }
        }
        res
    }
    fn scan_project(
        &self,
        table: &dyn Table,
        subset: SubsetRef,
        cols: &[ColumnId],
        start: Offset,
        n: usize,
        cs: &[Constraint],
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset> {
        let table = table.as_any().downcast_ref::<T>().unwrap();
        match cols {
            [] => None,
            [col] => table.scan_generic_bounded(subset, start, n, cs, |id, row| {
                out.add_row(id, &[row[col.index()]]);
            }),
            [x, y] => table.scan_generic_bounded(subset, start, n, cs, |id, row| {
                out.add_row(id, &[row[x.index()], row[y.index()]]);
            }),
            [x, y, z] => table.scan_generic_bounded(subset, start, n, cs, |id, row| {
                out.add_row(id, &[row[x.index()], row[y.index()], row[z.index()]]);
            }),
            _ => {
                let mut scratch = SmallVec::<[Value; 8]>::with_capacity(cols.len());
                table.scan_generic_bounded(subset, start, n, cs, |id, row| {
                    for col in cols {
                        scratch.push(row[col.index()]);
                    }
                    out.add_row(id, &scratch);
                    scratch.clear();
                })
            }
        }
    }

    fn lookup_row_vectorized(
        &self,
        table: &dyn Table,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        out_var: Variable,
    ) {
        let table = table.as_any().downcast_ref::<T>().unwrap();
        let mut out = with_pool_set(PoolSet::get::<Vec<Value>>);
        match args {
            [QueryEntry::Var(v)] => {
                mask.iter(&bindings[*v])
                    .fill_vec(&mut out, Value::stale, |_, arg| {
                        table.get_row_column(&[*arg], col)
                    });
            }
            [QueryEntry::Var(v1), QueryEntry::Var(v2)] => {
                mask.iter(&bindings[*v1]).zip(&bindings[*v2]).fill_vec(
                    &mut out,
                    Value::stale,
                    |_, (a1, a2)| table.get_row_column(&[*a1, *a2], col),
                );
            }
            [QueryEntry::Var(v1), QueryEntry::Var(v2), QueryEntry::Var(v3)] => {
                mask.iter(&bindings[*v1])
                    .zip(&bindings[*v2])
                    .zip(&bindings[*v3])
                    .fill_vec(&mut out, Value::stale, |_, ((a1, a2), a3)| {
                        table.get_row_column(&[*a1, *a2, *a3], col)
                    });
            }
            args => {
                let pool = with_pool_set(|ps| ps.get_pool().clone());
                mask.iter_dynamic(
                    pool,
                    args.iter().map(|v| match v {
                        QueryEntry::Var(v) => ValueSource::Slice(&bindings[*v]),
                        QueryEntry::Const(c) => ValueSource::Const(*c),
                    }),
                )
                .fill_vec(&mut out, Value::stale, |_, args| {
                    table.get_row_column(&args, col)
                });
            }
        };
        bindings.insert(out_var, out);
    }

    fn lookup_with_default_vectorized(
        &self,
        table: &dyn Table,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        default: QueryEntry,
        out_var: Variable,
    ) {
        let table = table.as_any().downcast_ref::<T>().unwrap();
        let mut out = with_pool_set(|ps| ps.get::<Vec<Value>>());
        match (args, default) {
            ([QueryEntry::Var(v)], QueryEntry::Var(default)) => mask
                .iter(&bindings[*v])
                .zip(&bindings[default])
                .fill_vec(&mut out, Value::stale, |_, (v, default)| {
                    Some(table.get_row_column(&[*v], col).unwrap_or(*default))
                }),
            ([QueryEntry::Var(v1), QueryEntry::Var(v2)], QueryEntry::Var(default)) => mask
                .iter(&bindings[*v1])
                .zip(&bindings[*v2])
                .zip(&bindings[default])
                .fill_vec(&mut out, Value::stale, |_, ((a1, a2), default)| {
                    Some(table.get_row_column(&[*a1, *a2], col).unwrap_or(*default))
                }),
            (
                [QueryEntry::Var(v1), QueryEntry::Var(v2), QueryEntry::Var(v3)],
                QueryEntry::Var(default),
            ) => mask
                .iter(&bindings[*v1])
                .zip(&bindings[*v2])
                .zip(&bindings[*v3])
                .zip(&bindings[default])
                .fill_vec(&mut out, Value::stale, |_, (((a1, a2), a3), default)| {
                    Some(
                        table
                            .get_row_column(&[*a1, *a2, *a3], col)
                            .unwrap_or(*default),
                    )
                }),
            (args, default) => {
                let pool = with_pool_set(|ps| ps.get_pool().clone());
                mask.iter_dynamic(
                    pool,
                    iter::once(&default).chain(args.iter()).map(|v| match v {
                        QueryEntry::Var(v) => ValueSource::Slice(&bindings[*v]),
                        QueryEntry::Const(c) => ValueSource::Const(*c),
                    }),
                )
                .fill_vec(&mut out, Value::stale, |_, vals| {
                    let default = vals[0];
                    let key = &vals[1..];
                    Some(table.get_row_column(key, col).unwrap_or(default))
                })
            }
        };
        bindings.insert(out_var, out);
    }
}

/// A WrappedTable takes a Table and extends it with a number of helpful,
/// object-safe methods for accessing a table.
///
/// It essentially acts like an extension trait: it is a separate type to allow
/// object-safe extension methods to call methods that require `Self: Sized`.
/// The implementations here downcast manually to the type used when
/// constructing the WrappedTable.
pub struct WrappedTable {
    inner: Box<dyn Table>,
    wrapper: Box<dyn TableWrapper>,
}

impl WrappedTable {
    pub(crate) fn new<T: Table>(inner: T) -> Self {
        let wrapper = wrapper::<T>();
        let inner = Box::new(inner);
        Self { inner, wrapper }
    }

    /// Clone the contents of the table.
    pub fn dyn_clone(&self) -> Self {
        WrappedTable {
            inner: self.inner.dyn_clone(),
            wrapper: self.wrapper.dyn_clone(),
        }
    }

    pub(crate) fn as_ref(&self) -> WrappedTableRef {
        WrappedTableRef {
            inner: &*self.inner,
            wrapper: &*self.wrapper,
        }
    }

    /// Starting at the given [`Offset`] into `subset`, scan up to `n` rows and
    /// write them to `out`. Return the next starting offset. If no offset is
    /// returned then the subset has been scanned completely.
    pub fn scan_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset> {
        self.as_ref().scan_bounded(subset, start, n, out)
    }

    /// Group the contents of the given subset by the given column.
    pub(crate) fn group_by_col(&self, subset: SubsetRef, col: ColumnId) -> ColumnIndex {
        self.as_ref().group_by_col(subset, col)
    }

    /// A multi-column vairant of [`WrappedTable::group_by_col`].
    pub(crate) fn group_by_key(&self, subset: SubsetRef, cols: &[ColumnId]) -> TupleIndex {
        self.as_ref().group_by_key(subset, cols)
    }

    /// A variant fo [`WrappedTable::scan_bounded`] that projects a subset of
    /// columns and only appends rows that match the given constraints.
    pub fn scan_project(
        &self,
        subset: SubsetRef,
        cols: &[ColumnId],
        start: Offset,
        n: usize,
        cs: &[Constraint],
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset> {
        self.as_ref().scan_project(subset, cols, start, n, cs, out)
    }

    /// Return the contents of the subset as a [`TaggedRowBuffer`].
    pub fn scan(&self, subset: SubsetRef) -> TaggedRowBuffer {
        self.as_ref().scan(subset)
    }

    pub(crate) fn lookup_row_vectorized(
        &self,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        out_var: Variable,
    ) {
        self.as_ref()
            .lookup_row_vectorized(mask, bindings, args, col, out_var)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lookup_with_default_vectorized(
        &self,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        default: QueryEntry,
        out_var: Variable,
    ) {
        self.as_ref()
            .lookup_with_default_vectorized(mask, bindings, args, col, default, out_var)
    }
}

impl Deref for WrappedTable {
    type Target = dyn Table;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl DerefMut for WrappedTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

pub(crate) trait TableWrapper: Send + Sync {
    fn dyn_clone(&self) -> Box<dyn TableWrapper>;
    fn scan_bounded(
        &self,
        table: &dyn Table,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset>;
    fn group_by_col(&self, table: &dyn Table, subset: SubsetRef, col: ColumnId) -> ColumnIndex;
    fn group_by_key(&self, table: &dyn Table, subset: SubsetRef, cols: &[ColumnId]) -> TupleIndex;

    #[allow(clippy::too_many_arguments)]
    fn scan_project(
        &self,
        table: &dyn Table,
        subset: SubsetRef,
        cols: &[ColumnId],
        start: Offset,
        n: usize,
        cs: &[Constraint],
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset>;

    fn scan(&self, table: &dyn Table, subset: SubsetRef) -> TaggedRowBuffer {
        let arity = table.spec().arity();
        let mut buf = TaggedRowBuffer::new(arity);
        assert!(self
            .scan_bounded(table, subset, Offset::new(0), usize::MAX, &mut buf)
            .is_none());
        buf
    }

    #[allow(clippy::too_many_arguments)]
    fn lookup_row_vectorized(
        &self,
        table: &dyn Table,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        out_var: Variable,
    );

    #[allow(clippy::too_many_arguments)]
    fn lookup_with_default_vectorized(
        &self,
        table: &dyn Table,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        default: QueryEntry,
        out_var: Variable,
    );
}

/// An extra layer of indirection over a [`WrappedTable`] that does not require that the caller
/// actually own the table. This is useful when a table implementation needs to construct a
/// WrappedTable on its own.
#[derive(Clone, Copy)]
pub struct WrappedTableRef<'a> {
    inner: &'a dyn Table,
    wrapper: &'a dyn TableWrapper,
}

impl WrappedTableRef<'_> {
    pub(crate) fn with_wrapper<T: Table, R>(
        inner: &T,
        f: impl for<'a> FnOnce(WrappedTableRef<'a>) -> R,
    ) -> R {
        let wrapper = WrapperImpl::<T>(PhantomData);
        f(WrappedTableRef {
            inner,
            wrapper: &wrapper,
        })
    }

    /// Starting at the given [`Offset`] into `subset`, scan up to `n` rows and
    /// write them to `out`. Return the next starting offset. If no offset is
    /// returned then the subset has been scanned completely.
    pub fn scan_bounded(
        &self,
        subset: SubsetRef,
        start: Offset,
        n: usize,
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset> {
        self.wrapper.scan_bounded(self.inner, subset, start, n, out)
    }

    /// Group the contents of the given subset by the given column.
    pub(crate) fn group_by_col(&self, subset: SubsetRef, col: ColumnId) -> ColumnIndex {
        self.wrapper.group_by_col(self.inner, subset, col)
    }

    /// A multi-column vairant of [`WrappedTable::group_by_col`].
    pub(crate) fn group_by_key(&self, subset: SubsetRef, cols: &[ColumnId]) -> TupleIndex {
        self.wrapper.group_by_key(self.inner, subset, cols)
    }

    /// A variant fo [`WrappedTable::scan_bounded`] that projects a subset of
    /// columns and only appends rows that match the given constraints.
    pub fn scan_project(
        &self,
        subset: SubsetRef,
        cols: &[ColumnId],
        start: Offset,
        n: usize,
        cs: &[Constraint],
        out: &mut TaggedRowBuffer,
    ) -> Option<Offset> {
        self.wrapper
            .scan_project(self.inner, subset, cols, start, n, cs, out)
    }

    /// Return the contents of the subset as a [`TaggedRowBuffer`].
    pub fn scan(&self, subset: SubsetRef) -> TaggedRowBuffer {
        self.wrapper.scan(self.inner, subset)
    }

    pub(crate) fn lookup_row_vectorized(
        &self,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        out_var: Variable,
    ) {
        self.wrapper
            .lookup_row_vectorized(self.inner, mask, bindings, args, col, out_var);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lookup_with_default_vectorized(
        &self,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        col: ColumnId,
        default: QueryEntry,
        out_var: Variable,
    ) {
        self.wrapper.lookup_with_default_vectorized(
            self.inner, mask, bindings, args, col, default, out_var,
        );
    }
}

impl Deref for WrappedTableRef<'_> {
    type Target = dyn Table;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}
