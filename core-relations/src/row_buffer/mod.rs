//! A basic data-structure encapsulating a batch of rows.

use core::slice;
use std::{cell::Cell, mem, ops::Deref};

use concurrency::ParallelVecWriter;
use numeric_id::NumericId;
use rayon::iter::ParallelIterator;
use smallvec::SmallVec;

use crate::{
    common::Value,
    offsets::RowId,
    pool::{with_pool_set, Pooled},
};

#[cfg(test)]
mod tests;

/// A batch of rows. This is a common enough pattern that it makes sense to make
/// it its own data-structure. The advantage of this abstraction is that it
/// allows us to store multiple rows in a single allocation.
///
/// RowBuffer stores data in row-major order.
pub struct RowBuffer {
    n_columns: usize,
    total_rows: usize,
    data: Pooled<Vec<Cell<Value>>>,
}

// Safety constraints for RowBuffer.
//
// All of the unsafe code in RowBuffer is due to the use of `Cell<Value>` for
// the backing `data`. We do not want to expose raw `Cell`s to users (they
// complicate the API), but every use-case for RowBuffer uses entries in data
// like normal values _but one_: that is the `set_stale_shared` method. See the
// documentation for that method for more context.
//
// This method enabled multiple threads to write to exclusive rows in the table
// without performing any additional synchronization, or slowing down future
// readers by requiring atomic operations for every read.
unsafe impl Send for RowBuffer {}
unsafe impl Sync for RowBuffer {}

impl Clone for RowBuffer {
    fn clone(&self) -> Self {
        RowBuffer {
            n_columns: self.n_columns,
            total_rows: self.total_rows,
            data: Pooled::cloned(&self.data),
        }
    }
}

impl RowBuffer {
    /// Create a new RowBuffer with the given arity.
    pub(crate) fn new(n_columns: usize) -> RowBuffer {
        assert_ne!(
            n_columns, 0,
            "attempting to create a row batch with no columns"
        );
        RowBuffer {
            n_columns,
            total_rows: 0,
            data: with_pool_set(|ps| ps.get()),
        }
    }

    pub(crate) fn parallel_writer(&mut self) -> ParallelRowBufWriter {
        let data = mem::take(&mut self.data);
        ParallelRowBufWriter {
            buf: RowBuffer {
                n_columns: self.n_columns,
                total_rows: self.total_rows,
                data: Default::default(),
            },
            vec: Some(ParallelVecWriter::new(Pooled::into_inner(data))),
        }
    }

    /// Reserve space for `additional` rows.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.n_columns);
    }

    /// The size of the rows accepted by this buffer.
    pub(crate) fn arity(&self) -> usize {
        self.n_columns
    }

    pub(crate) fn raw_rows(&self) -> *const Value {
        self.data.as_ptr() as *const Value
    }

    /// Blindly set the length of the RowBuffer to the given number of rows.
    ///
    /// # Safety
    /// `count` must be within the capacity of the RowBuffer and the resized buffer must point to
    /// initialized memory. (Analogous to [`Vec::set_len`]).
    pub(crate) unsafe fn set_len(&mut self, count: usize) {
        self.data.set_len(count * self.n_columns);
        self.total_rows = count;
    }

    /// Return an iterator over the non-stale rows in the buffer.
    pub(crate) fn non_stale(&self) -> impl Iterator<Item = &[Value]> {
        self.data
            .chunks(self.n_columns)
            .filter(|row| !row[0].get().is_stale())
            // SAFETY: This kind of transmutation is safe so long as no one
            // modifies any of the values behind the `Cell` while this value is
            // borrowed.
            //
            // The only time we modify these values is in safe methods requiring
            // a mutable reference (`set_stale`, `get_row_mut`), or in the
            // unsafe `set_stale_shared` method whose safety requirements imply
            // that no call will overlap with borrowing such a row.
            .map(|row| unsafe { mem::transmute::<&[Cell<Value>], &[Value]>(row) })
    }

    pub(crate) fn non_stale_mut(&mut self) -> impl Iterator<Item = &mut [Value]> {
        self.data
            .chunks_mut(self.n_columns)
            .filter(|row| !row[0].get().is_stale())
            // SAFETY: This kind of transmutation is safe so long as no one
            // modifies any of the values behind the `Cell` while this value is
            // borrowed.
            //
            // The only time we modify these values is in safe methods requiring
            // a mutable reference (`set_stale`, `get_row_mut`), or in the
            // unsafe `set_stale_shared` method whose safety requirements imply
            // that no call will overlap with borrowing such a row.
            .map(|row| unsafe { mem::transmute::<&mut [Cell<Value>], &mut [Value]>(row) })
    }

    /// A parallel version of [`RowBuffer::iter`].
    pub(crate) fn parallel_iter(&self) -> impl ParallelIterator<Item = &[Value]> {
        use rayon::prelude::*;
        // SAFETY: This kind of transmutation is safe so long as no one
        // modifies any of the values behind the `Cell` while this value is
        // borrowed.
        //
        // The only time we modify these values is in safe methods requiring
        // a mutable reference (`set_stale`, `get_row_mut`), or in the
        // unsafe `set_stale_shared` method whose safety requirements imply
        // that no call will overlap with borrowing such a row.
        unsafe { mem::transmute::<&[Cell<Value>], &[Value]>(&self.data) }.par_chunks(self.n_columns)
    }

    /// Return an iterator over all rows in the buffer.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &[Value]> {
        self.data
            .chunks(self.n_columns)
            // SAFETY: see comment in `non_stale`.
            .map(|row| unsafe { mem::transmute::<&[Cell<Value>], &[Value]>(row) })
    }

    /// Clear the contents of the buffer.
    pub(crate) fn clear(&mut self) {
        self.data.clear();
        self.total_rows = 0;
    }

    /// The number of rows in the buffer.
    pub(crate) fn len(&self) -> usize {
        self.total_rows
    }

    /// Mark a row as stale in the buffer with shared access to it. Returns
    /// whether the row was already stale.
    ///
    /// # Safety
    /// This method is unsafe because we implement `Send` and `Sync` for the
    /// `RowBuffer` type. That means that you can call `set_stale_shared(row)`
    /// and `get_row(row)` concurrently, which would be a data race.
    ///
    /// To safely use this method, you must ensure that there are no concurrent reads or writes to
    /// `row`. Indeed, that is what this method is for: parallel writes to exclusive rows in a
    /// shared `RowBuffer`. Any other use-case should use the [`RowBuffer::set_stale`] method,
    /// which requires a mutable reference.
    pub(crate) unsafe fn set_stale_shared(&self, row: RowId) -> bool {
        let cells = &self.data[row.index() * self.n_columns..(row.index() + 1) * self.n_columns];
        let was_stale = cells[0].get().is_stale();
        cells[0].set(Value::stale());
        was_stale
    }

    /// Get the row corresponding to the given RowId.
    ///
    /// # Panics
    /// This method panics if `row` is out of bounds.
    pub(crate) fn get_row(&self, row: RowId) -> &[Value] {
        // SAFETY: see the comment in `non_stale`.
        unsafe { get_row(&self.data, self.n_columns, row) }
    }

    /// Get the row corresponding to the given RowId without bounds checking.
    pub(crate) unsafe fn get_row_unchecked(&self, row: RowId) -> &[Value] {
        slice::from_raw_parts(
            self.data.as_ptr().add(row.index() * self.n_columns) as *const Value,
            self.n_columns,
        )
    }

    /// Get a mutable reference to the row corresponding to the given RowId.
    ///
    /// # Panics
    /// This method panics if `row` is out of bounds.
    pub(crate) fn get_row_mut(&mut self, row: RowId) -> &mut [Value] {
        // SAFETY: see the comment in `non_stale`.
        unsafe {
            mem::transmute::<&mut [Cell<Value>], &mut [Value]>(
                &mut self.data[row.index() * self.n_columns..(row.index() + 1) * self.n_columns],
            )
        }
    }

    /// Set the given row to be stale. By convention, this calls `set_stale` on
    /// the first column in the row. Returns whether the row was already stale.
    ///
    /// # Panics
    /// This method panics if `row` is out of bounds.
    pub(crate) fn set_stale(&mut self, row: RowId) -> bool {
        let row = self.get_row_mut(row);
        let res = row[0].is_stale();
        row[0].set_stale();
        res
    }

    /// Insert a row into a buffer, returning the RowId for this row.
    ///
    /// # Panics
    /// This method panics if the length of `row` does not match the arity of
    /// the RowBuffer.
    pub(crate) fn add_row(&mut self, row: &[Value]) -> RowId {
        assert_eq!(
            row.len(),
            self.n_columns,
            "attempting to add a row with mismatched arity to table"
        );
        if self.total_rows == 0 {
            Pooled::refresh(&mut self.data);
        }
        let res = RowId::from_usize(self.total_rows);
        self.data.extend(row.iter().copied().map(Cell::new));
        self.total_rows += 1;
        res
    }

    /// Remove any stale entries in the buffer. This invalidates existing
    /// RowIds. This method calls `remap` with the old and new RowIds for all
    /// non-stale rows.
    pub(crate) fn remove_stale(&mut self, mut remap: impl FnMut(&[Value], RowId, RowId)) {
        let mut within_row = 0;
        let mut row_in = 0;
        let mut row_out = 0;
        let mut keep_row = true;
        let mut scratch = SmallVec::<[Value; 8]>::new();
        self.data.retain(|entry| {
            if within_row == 0 {
                keep_row = !entry.get().is_stale();
                if keep_row {
                    scratch.push(entry.get());
                    row_out += 1;
                }
                row_in += 1;
            } else if keep_row {
                scratch.push(entry.get());
            }
            within_row += 1;
            if within_row == self.n_columns {
                within_row = 0;
                if keep_row {
                    remap(&scratch, RowId::new(row_in - 1), RowId::new(row_out - 1));
                    scratch.clear();
                }
            }
            keep_row
        });
        self.total_rows = row_out as usize;
    }
}

/// A `TaggedRowBuffer` wraps a `RowBuffer` but also keeps track of a _source_
/// `RowId` for the row it contains. This makes it useful for materializing
/// the contents of a `Subset` of a table.
pub struct TaggedRowBuffer {
    inner: RowBuffer,
}

impl TaggedRowBuffer {
    /// Create a new buffer with the given arity.
    pub fn new(n_columns: usize) -> TaggedRowBuffer {
        TaggedRowBuffer {
            inner: RowBuffer::new(n_columns + 1),
        }
    }

    /// Clear the contents of the buffer.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// The number of rows in the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn base_arity(&self) -> usize {
        self.inner.n_columns - 1
    }

    /// Add the given row and RowId to the buffer, returning the RowId (in
    /// `self`) for the new row.
    pub fn add_row(&mut self, row_id: RowId, row: &[Value]) -> RowId {
        // Variant of `RowBuffer::add_row` that also stores the given `RowId` inline.
        //
        // Changes to the implementation of one method should probably also
        // change the other.
        assert_eq!(
            row.len(),
            self.base_arity(),
            "attempting to add a row with mismatched arity to table"
        );
        if self.inner.total_rows == 0 {
            Pooled::refresh(&mut self.inner.data);
        }
        let res = RowId::from_usize(self.inner.total_rows);
        self.inner.data.extend(row.iter().copied().map(Cell::new));
        self.inner.data.push(Cell::new(Value::new(row_id.rep())));
        self.inner.total_rows += 1;
        res
    }

    /// Get the row (and the id it was associated with at insertion time) at the
    /// offset associated with `row`.
    pub fn get_row(&self, row: RowId) -> (RowId, &[Value]) {
        self.unwrap_row(self.inner.get_row(row))
    }

    pub fn get_row_mut(&mut self, row: RowId) -> (RowId, &mut [Value]) {
        let base_arity = self.base_arity();
        let row = self.inner.get_row_mut(row);
        let row_id = row[base_arity];
        let row = &mut row[..base_arity];
        (RowId::new(row_id.rep()), row)
    }

    /// Iterate over the contents of the buffer.
    pub fn iter(&self) -> impl Iterator<Item = (RowId, &[Value])> {
        self.inner.iter().map(|row| self.unwrap_row(row))
    }

    /// Iterate over the contents of the buffer in parallel.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (RowId, &[Value])> {
        self.inner.parallel_iter().map(|row| self.unwrap_row(row))
    }

    /// Iterate over all rows in the buffer, except for the stale ones.
    pub fn non_stale(&self) -> impl Iterator<Item = (RowId, &[Value])> {
        self.inner.non_stale().map(|row| self.unwrap_row(row))
    }

    /// Iterate over all rows in the buffer, except for the stale ones.
    pub fn non_stale_mut(&mut self) -> impl Iterator<Item = (RowId, &mut [Value])> {
        let base_arity = self.base_arity();
        self.inner
            .non_stale_mut()
            .map(move |row| Self::unwrap_row_mut(base_arity, row))
    }

    pub fn set_stale(&mut self, row: RowId) -> bool {
        self.inner.set_stale(row)
    }

    fn unwrap_row<'a>(&self, row: &'a [Value]) -> (RowId, &'a [Value]) {
        let row_id = row[self.base_arity()];
        let row = &row[..self.base_arity()];
        (RowId::new(row_id.rep()), row)
    }
    fn unwrap_row_mut(base_arity: usize, row: &mut [Value]) -> (RowId, &mut [Value]) {
        let row_id = row[base_arity];
        let row = &mut row[..base_arity];
        (RowId::new(row_id.rep()), row)
    }
}

/// # Safety
/// This function is safe so long as there are no concurrent writes to the given
/// row.
unsafe fn get_row(data: &[Cell<Value>], n_columns: usize, row: RowId) -> &[Value] {
    mem::transmute::<&[Cell<Value>], &[Value]>(
        &data[row.index() * n_columns..(row.index() + 1) * n_columns],
    )
}

/// A wrapper for a RowBuffer that allows it to be written to in parallel, based
/// on [`ParallelVecWriter`].
///
/// This is a type that is used to speed up parallel `merge` operations on
/// `SortedWritesTable`. It uses a low-level interface that should be avoided in
/// most cases.
pub(crate) struct ParallelRowBufWriter {
    buf: RowBuffer,
    // This is only an option so we can move out of it in `drop`. It is always
    // populated.
    vec: Option<ParallelVecWriter<Cell<Value>>>,
}

impl ParallelRowBufWriter {
    pub(crate) fn read_handle(&self) -> ReadHandle<'_, impl Deref<Target = [Cell<Value>]> + '_> {
        ReadHandle {
            buf: &self.buf,
            data: self.vec.as_ref().unwrap().read_access(),
        }
    }
    pub(crate) fn write_raw_values(
        &self,
        vals: impl ExactSizeIterator<Item = Value>,
        new_rows: usize,
    ) -> RowId {
        debug_assert_eq!(vals.len() % self.buf.n_columns, 0);
        debug_assert_eq!(vals.len() / self.buf.n_columns, new_rows);
        let start_off = self
            .vec
            .as_ref()
            .unwrap()
            .write_contents(vals.map(Cell::new));
        debug_assert_eq!(start_off % self.buf.n_columns, 0);
        RowId::from_usize(start_off / self.buf.n_columns)
    }

    pub(crate) fn finish(mut self) -> RowBuffer {
        self.buf.data = Pooled::new(self.vec.take().unwrap().finish());
        self.buf.total_rows = self.buf.data.len() / self.buf.n_columns;
        self.buf
    }
}

/// A handle granting read access to a row buffer's contents.
pub(crate) struct ReadHandle<'a, T> {
    buf: &'a RowBuffer,
    data: T,
}

impl<T: Deref<Target = [Cell<Value>]>> ReadHandle<'_, T> {
    /// Get the row corresponding to the given RowId without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that either `row` is within bounds of the buffer at the creation of
    /// this handle, or that the row was successfully written to the buffer before it was called.
    ///
    /// Furthermore, no calls to `set_stale_shared` may overlap with this call.
    pub(crate) unsafe fn get_row_unchecked(&self, row: RowId) -> &[Value] {
        // SAFETY: ParallelVecWriter guarantees that data within bounds is not
        // being modified concurrently.
        std::slice::from_raw_parts(
            self.data.as_ptr().add(row.index() * self.buf.n_columns) as *const Value,
            self.buf.n_columns,
        )
    }

    /// See the documentation for [`RowBuffer::set_stale_shared`].
    ///
    /// In addition to the requirements there, `row` is allowed to be out of bounds of the initial
    /// length of the wrapped vector, but any out-of-bounds row must be in bounds of a (previously
    /// completed) write.
    pub(crate) unsafe fn set_stale_shared(&self, row: RowId) -> bool {
        let cells: &[Cell<Value>] = &self.data;
        let cell_ptr: *const Cell<Value> = cells.as_ptr();
        let to_set: &Cell<Value> = &*cell_ptr.add(row.index() * self.buf.n_columns);
        let was_stale = to_set.get().is_stale();
        to_set.set(Value::stale());
        was_stale
    }
}
