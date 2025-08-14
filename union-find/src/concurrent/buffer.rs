//! A concurrent array supporting resizing.
//!
//! This is a basic implementation of a concurrent vector object. When no resize
//! is required, it provides constant-time, wait-free access to elements and
//! provides a means for threads to safely block on resizing events. When
//! resizing is required, readers and writers will block until resizing is
//! complete.
//!
//! This means that the data-structure doesn't _really_ have any firm progress
//! guarantees when it comes to egglog, where resizing will be needed as the
//! egraph grows and new ids are generated. Still, this should scale much better
//! compared with a simple RwLock-based implementation.

use std::mem;

use concurrency::ReadOptimizedLock;

use super::atomic_int::AtomicInt;

pub struct Buffer<T> {
    data: ReadOptimizedLock<Vec<T>>,
}

impl<T: AtomicInt> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        let reader = self.data.read();
        let mut new = Vec::with_capacity(reader.capacity());
        let mut i = 0;
        // A bit of a hack... should we just memcopy and call it a day?
        new.resize_with(reader.len(), || {
            let result = T::from_usize(T::as_usize(reader[i].load()));
            i += 1;
            result
        });
        Self {
            data: ReadOptimizedLock::new(new),
        }
    }
}

impl<T: Send + Sync + 'static> Buffer<T> {
    /// Initialize a new buffer with the given capacity.
    pub fn new(capacity: usize, mut init: impl FnMut(usize) -> T) -> Buffer<T> {
        let mut vec = Vec::with_capacity(capacity);
        let mut i = 0;
        vec.resize_with(capacity, || {
            let res = init(i);
            i += 1;
            res
        });
        Buffer {
            data: ReadOptimizedLock::new(vec),
        }
    }

    /// Run `f` with access to the underlying buffer.
    ///
    /// This method will ensure that the underlying buffer has capacity at least
    /// `len` by the time `f` is run. If `len` is greater than the current
    /// capacity of the buffer, this method will block until a resize has been
    /// performed.
    ///
    /// If the calling thread is the one that performs resizing, it will use
    /// `init` to initialize new elements in the buffer.
    pub fn with_access<R>(
        &self,
        len: usize,
        mut f: impl FnMut(&[T]) -> R,
        mut init: impl FnMut(usize) -> T + Send + Clone + 'static,
    ) -> R {
        let data = self.data.read();
        if data.len() >= len {
            return f(&data);
        }
        mem::drop(data);
        let mut data = self.data.lock();
        if data.len() < len {
            let len = len.next_power_of_two();
            let mut i = data.len();
            data.resize_with(len, || {
                let result = init(i);
                i += 1;
                result
            });
        }
        mem::drop(data);
        self.with_access(len, f, init)
    }

    /// Overwrite the buffer with the contents specified by `init`.
    pub fn re_init(&self, mut init: impl FnMut(usize) -> T + Send + Clone + 'static) {
        let mut data = self.data.lock();
        let mut i = 0;
        data.iter_mut().for_each(|x| {
            *x = init(i);
            i += 1;
        });
    }
}
