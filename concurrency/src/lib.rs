//! Handy routines for managing concurrency.

pub(crate) mod bitset;
pub(crate) mod concurrent_vec;
pub(crate) mod notification;
pub mod parallel_writer;
use arc_swap::{ArcSwap, Guard};

pub use bitset::BitSet;
pub use concurrent_vec::ConcurrentVec;
pub use notification::Notification;
pub use parallel_writer::ParallelVecWriter;

#[cfg(test)]
mod tests;

use std::{
    cell::UnsafeCell,
    mem,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{fence, Ordering},
        Arc,
    },
};

/// A mutex lock optimized for low-contention read access.
/// The RwLock type in the standard library allows multiple readers to make
/// progress concurrently, but scalability is limited if these read-only
/// critical sections are small because acquiring a read-side lock still
/// requires a read-modify-write atomic operation.
///
/// This crate uses RCU techniques from the ArcSwap crate to avoid expensive
/// atomic operations when acquiring a shared lock. The downside is that
/// acquiring a write lock is a good deal more expensive. The intent here is to
/// use it for parallel egglog where write-side critical sections are fairly
/// large and uncontended, but we need high scalability for read-only
/// operations.
pub struct ReadOptimizedLock<T> {
    token: ArcSwap<ReadToken>,
    data: UnsafeCell<T>,
}

/// A handle granting read access to the data guarded by a [`ReadOptimizedLock`].
pub struct MutexReader<'lock, T> {
    data: &'lock T,
    _guard: Guard<Arc<ReadToken>>,
}

impl<T> Deref for MutexReader<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.data
    }
}

/// A handle granting exclusive access to the data guarded by a [`ReadOptimizedLock`].
pub struct MutexWriter<'lock, T> {
    lock: &'lock ReadOptimizedLock<T>,
    unblock: Arc<Notification>,
}

impl<T> Deref for MutexWriter<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `data` is valid as long as we have a valid reference to it.
        // We only create a `Writer` after the current thread has exclusive
        // access to data.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for MutexWriter<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `data` is valid as long as we have a valid reference to it.
        // We only create a `Writer` after the current thread has exclusive
        // access to data.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for MutexWriter<'_, T> {
    fn drop(&mut self) {
        self.lock
            .token
            .store(Arc::new(ReadToken::ReadOk(TriggerWhenDone::default())));
        self.unblock.notify();
    }
}

impl<T> ReadOptimizedLock<T> {
    pub fn new(data: T) -> Self {
        Self {
            token: ArcSwap::from_pointee(ReadToken::ReadOk(TriggerWhenDone::default())),
            data: UnsafeCell::new(data),
        }
    }

    /// Extract the inner data from the lock.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Get mutable access to the underlying data. This operation does no synchronization as the
    /// mutable receiver guarantees exclusive access for safe code.
    pub fn as_mut_ref(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Create a `Reader` object that grants read access to the data.
    ///
    /// This method will block for any ongoing writes to complete.
    pub fn read(&self) -> MutexReader<T> {
        loop {
            let guard = self.token.load();
            match guard.as_ref() {
                ReadToken::ReadOk(..) => {
                    // This fence ensures that we see the outcome of any
                    // preceeding writes. (I think...).
                    fence(Ordering::SeqCst);
                    return MutexReader {
                        // SAFETY: We guarantee that as long as a guard that's
                        // in scope observes a ReadOk token
                        data: unsafe { &*self.data.get() },
                        _guard: guard,
                    };
                }
                ReadToken::WriteOngoing(n) => {
                    let n = n.clone();
                    mem::drop(guard);
                    n.wait();
                    continue;
                }
            }
        }
    }

    pub fn lock(&self) -> MutexWriter<T> {
        loop {
            let guard = self.token.load();
            match guard.as_ref() {
                ReadToken::ReadOk(n) => {
                    let unblock_waiters = Arc::new(Notification::default());
                    let write_token = ReadToken::WriteOngoing(unblock_waiters.clone());
                    let readers_done = n.0.clone();
                    let prev = self.token.compare_and_swap(&guard, Arc::new(write_token));
                    if prev.as_ref() as *const _ != guard.as_ref() as *const _ {
                        // CAS failed, retry.
                        continue;
                    }
                    mem::drop((guard, prev));
                    // Do an RCU to trigger an underlying "wait for readers" operation.
                    self.token.rcu(|x| x.clone());
                    // NB: this wait not be necessary... it isn't clear to me if
                    // this is documented behavior of the crate.
                    readers_done.wait();
                    return MutexWriter {
                        lock: self,
                        unblock: unblock_waiters,
                    };
                }
                ReadToken::WriteOngoing(n) => {
                    let n = n.clone();
                    mem::drop(guard);
                    n.wait();
                }
            }
        }
    }
}

unsafe impl<T: Send> Send for ReadOptimizedLock<T> {}
unsafe impl<T: Send> Sync for ReadOptimizedLock<T> {}

enum ReadToken {
    ReadOk(TriggerWhenDone),
    WriteOngoing(Arc<Notification>),
}

#[derive(Default)]
struct TriggerWhenDone(Arc<Notification>);

impl Drop for TriggerWhenDone {
    fn drop(&mut self) {
        self.0.notify();
    }
}
