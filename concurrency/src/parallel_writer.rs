//! A Utility Struct for Writing to a Vector in parallel without blocking reads.

use std::{
    mem,
    ops::{Deref, Range},
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{MutexReader, ReadOptimizedLock};

/// A struct that wraps a vector and allows for parallel writes to it.
///
/// While the writes happen, reads to the vector can proceed without being
/// blocked by writes (except during a vector resize). The final vector can be
/// extracted using the `finish` method. Elements written to the vector behind a
/// ParallelVecWriter will not be dropped unless `finish` is called.
pub struct ParallelVecWriter<T> {
    data: ReadOptimizedLock<Vec<T>>,
    end_len: AtomicUsize,
}

/// A handle that can be used to read arbitrary locations in a vector wrapped by a
/// [`ParallelVecWriter`], even if they weren't
/// initialized when the [`ParallelVecWriter`] was created.
pub struct UnsafeReadAccess<'a, T> {
    reader: MutexReader<'a, Vec<T>>,
}

impl<T> UnsafeReadAccess<'_, T> {
    /// Get a reference to the given index in the vector.
    ///
    /// # Safety
    /// `idx` must be either less than the length of the vector when the underlying
    /// [`ParallelVecWriter`] was created, or it must be within bounds of a completed write to
    /// [`ParallelVecWriter::write_contents`].
    pub unsafe fn get_unchecked(&self, idx: usize) -> &T {
        &*self.reader.as_ptr().add(idx)
    }

    /// Get a subslice of given index in the vector.
    ///
    /// # Safety
    /// `slice`'s contents must be either within the vector when the underlying
    /// [`ParallelVecWriter`] was created, or they must be within bounds of a completed write to
    /// [`ParallelVecWriter::write_contents`].
    pub unsafe fn get_unchecked_slice(&self, slice: Range<usize>) -> &[T] {
        let start: *const T = self.reader.as_ptr().add(slice.start);
        std::slice::from_raw_parts(start, slice.end - slice.start)
    }
}

impl<T> ParallelVecWriter<T> {
    pub fn new(data: Vec<T>) -> Self {
        let start_len = data.len();
        let end_len = AtomicUsize::new(start_len);
        Self {
            data: ReadOptimizedLock::new(data),
            end_len,
        }
    }

    /// Get read access to the portion of the vector that was present before the
    /// ParallelVecWriter was created. Unlike the `with_` methods, callers
    /// should be careful about keeping the object returned from this method
    /// around for too long.
    pub fn read_access(&self) -> impl Deref<Target = [T]> + '_ {
        struct PrefixReader<'a, T> {
            reader: MutexReader<'a, Vec<T>>,
        }
        impl<T> Deref for PrefixReader<'_, T> {
            type Target = [T];

            fn deref(&self) -> &[T] {
                self.reader.as_slice()
            }
        }
        PrefixReader {
            reader: self.data.read(),
        }
    }

    /// Get unsafe read access to the vector.
    ///
    /// This handle allows for reads past the end of the wrapped vector. Callers must guarantee
    /// that any cells read are covered by a corresponding call to
    /// [`ParallelVecWriter::write_contents`].
    pub fn unsafe_read_access(&self) -> UnsafeReadAccess<'_, T> {
        UnsafeReadAccess {
            reader: self.data.read(),
        }
    }

    /// Runs `f` with access to the element at `idx`.
    ///
    /// # Panics
    /// This method panics if `idx` is greater than or equal to the length of
    /// the vector when the ParallelVecWriter was created.
    pub fn with_index<R>(&self, idx: usize, f: impl FnOnce(&T) -> R) -> R {
        f(&self.read_access()[idx])
    }

    /// Runs `f` with access to the slice of elements in the range `slice`.
    ///
    /// # Panics
    /// This method panics if `slice.end` is greater than or equal to the length
    /// of the vector when the ParallelVecWriter was created.
    pub fn with_slice<R>(&self, slice: Range<usize>, f: impl FnOnce(&[T]) -> R) -> R {
        f(&self.read_access()[slice])
    }

    /// Write the contents of `items` to a contiguous chunk of the vector,
    /// returning the index of the first element in `items`.
    ///
    /// *Panics* It is very important that `items` does not lie about its
    /// length. This method panics if the actual length does not match the
    /// length method.
    pub fn write_contents(&self, items: impl ExactSizeIterator<Item = T>) -> usize {
        let start = self.end_len.fetch_add(items.len(), Ordering::AcqRel);
        let end = start + items.len();
        let reader = self.data.read();
        let current_len = reader.len();
        let current_cap = reader.capacity();
        mem::drop(reader);
        if current_cap < end {
            let mut writer = self.data.lock();
            if writer.capacity() < end {
                let new_cap = std::cmp::max(end, current_cap * 2);
                writer.reserve(new_cap - current_len);
            }
        }
        // SAFETY: the unsafe operations that `write_contents_at` performs are:
        // * Writing to a shared buffer: this is safe because the `fetch_add` we
        // perform gives us unique access to the subslice.
        // * Writing past the length of the vector: this is safe because the
        // above code pre-reseves sufficient capacity for `items` to write.
        unsafe { self.write_contents_at(items, start) };
        start
    }

    pub fn finish(self) -> Vec<T> {
        let mut res = self.data.into_inner();
        // SAFETY: this value is incremented past the original length of the
        // vector once for each item written to it.
        unsafe {
            res.set_len(self.end_len.load(Ordering::Acquire));
        }
        res
    }

    pub fn take(&mut self) -> Vec<T> {
        let mut res = mem::take(self.data.as_mut_ref());
        // SAFETY: this value is incremented past the original length of the
        // vector once for each item written to it.
        unsafe {
            res.set_len(self.end_len.load(Ordering::Acquire));
        }
        self.end_len.store(0, Ordering::Release);
        res
    }

    unsafe fn write_contents_at(&self, items: impl ExactSizeIterator<Item = T>, start: usize) {
        let mut written = 0;
        let expected = items.len();
        let reader = self.data.read();
        debug_assert!(reader.capacity() >= start + items.len());
        let mut mut_ptr = (reader.as_ptr() as *mut T).add(start);
        for item in items {
            written += 1;
            std::ptr::write(mut_ptr, item);
            mut_ptr = mut_ptr.offset(1);
        }
        assert_eq!(
            written, expected,
            "passed ExactSizeIterator with incorrect number of items"
        );
    }
}
