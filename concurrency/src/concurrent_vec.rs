//! A variant of a vector supporting pushes that do not block reads.

use std::{
    cell::UnsafeCell,
    mem::{self, MaybeUninit},
    ops::Deref,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use serde::{Deserialize, Deserializer, Serialize, ser::SerializeSeq};

use crate::{MutexReader, ReadOptimizedLock};

// NB: probably don't need to do SyncUnsafeCell here. Can probably just do MaybeUninit?

/// A simple concurrent vector type supporting push operations that do not block
/// reads. Concurrent pushes are serialized, but reads need not wait for writes
/// to complete, except when the vector needs to be resized.
pub struct ConcurrentVec<T> {
    data: ReadOptimizedLock<Vec<MaybeUninit<SyncUnsafeCell<T>>>>,
    /// The index of the next element to be pushed. This is used to determine
    /// how many cells have been written to successfully without grabbing any
    /// exclusive locks.
    head: AtomicUsize,
    /// Used to synchronize writes.
    write_lock: Mutex<()>,
}

impl<T: Serialize> Serialize for ConcurrentVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let len = self.head.load(std::sync::atomic::Ordering::Acquire);

        let data = self.data.read();

        let vec = &*data;

        // Serialize only initialized entries.
        let mut seq = serializer.serialize_seq(Some(len))?;
        for i in 0..len {
            unsafe {
                // SAFETY: `i < head`, so the cell is fully initialized and contains valid `T`.
                let cell_ptr = vec[i].as_ptr();
                let t_ref: &T = &*(*cell_ptr).0.get();
                seq.serialize_element(t_ref)?;
            }
        }

        seq.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ConcurrentVec<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values: Vec<T> = Vec::<T>::deserialize(deserializer)?;
        let head = values.len();

        // Allocate uninitialized backing store
        let mut backing = Vec::with_capacity(head);
        for _ in 0..head {
            backing.push(MaybeUninit::uninit());
        }

        // Fill initialized portion
        for (i, v) in values.into_iter().enumerate() {
            backing[i] = MaybeUninit::new(SyncUnsafeCell(UnsafeCell::new(v)));
        }

        Ok(ConcurrentVec {
            data: ReadOptimizedLock::new(backing),
            head: AtomicUsize::new(head),
            write_lock: Mutex::new(()),
        })
    }
}

impl<T> Default for ConcurrentVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for ConcurrentVec<T> {
    fn drop(&mut self) {
        let mut writer = self.data.lock();
        let len = self.head.load(Ordering::Acquire);
        if mem::needs_drop::<T>() {
            for i in 0..len {
                // SAFETY: we own the data, have exclusive access, and know that the
                // data is valid (you need a valid reference to the vector to
                // increment `head`, any such call to `head` must have exited by
                // now).
                unsafe { writer[i].as_mut_ptr().drop_in_place() };
            }
        }
    }
}

impl<T> ConcurrentVec<T> {
    /// Create a new `AsyncVec` with the default capacity (128).
    pub fn new() -> Self {
        Self::with_capacity(128)
    }

    /// Create a new `AsyncVec` with the given starting capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        Self {
            data: ReadOptimizedLock::new(Vec::with_capacity(capacity)),
            head: AtomicUsize::new(0),
            write_lock: Mutex::new(()),
        }
    }
    /// Push `item` onto the vector. Other calls to `push` may have to complete
    /// in order for this item to be visible.
    pub fn push(&self, item: T) -> usize {
        let _guard = self.write_lock.lock().unwrap();
        let index = self.head.load(Ordering::Acquire);
        self.push_at(item, index);
        self.head.store(index + 1, Ordering::Release);
        index
    }

    fn push_at(&self, item: T, index: usize) {
        let handle = self.data.read();
        if let Some(slot) = handle.get(index) {
            // SAFETY: we are tansferring ownership of `item` to the slot.
            unsafe { ((*slot.as_ptr()).0.get()).write(item) };
            return;
        }
        // `index` is out of bounds. Need to resize.
        mem::drop(handle);
        let mut writer = self.data.lock();
        if index >= writer.len() {
            writer.resize_with((index + 1).next_power_of_two(), MaybeUninit::uninit);
        }
        mem::drop(writer);
        self.push_at(item, index);
    }

    pub fn read(&self) -> impl Deref<Target = [T]> + '_ {
        let valid_prefix = self.head.load(Ordering::Acquire);
        let reader = self.data.read();
        ReadHandle {
            valid_prefix,
            reader,
        }
    }
}

struct ReadHandle<'a, T> {
    valid_prefix: usize,
    reader: MutexReader<'a, Vec<MaybeUninit<SyncUnsafeCell<T>>>>,
}

impl<T> Deref for ReadHandle<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: all elements up to `prefix` are valid, and MaybeUninit<T> has
        // a compatible layout with T so long as T is properly initialized.
        //
        // NB: transmuting an UnsafeCell<T> to <T> may not be safe long-term,
        // even though this code passes miri.
        unsafe {
            mem::transmute::<&[MaybeUninit<SyncUnsafeCell<T>>], &[T]>(
                &self.reader[0..self.valid_prefix],
            )
        }
    }
}

struct SyncUnsafeCell<T>(UnsafeCell<T>);

unsafe impl<T: Send> Send for SyncUnsafeCell<T> {}
unsafe impl<T: Sync> Sync for SyncUnsafeCell<T> {}
