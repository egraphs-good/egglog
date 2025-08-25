//! An abstraction providing high scalability of shared reads and limited support for interior
//! mutability and shared writes.
//!
//! The main thing we use this for is maintaining cached indexes in the `core-relations` crate.

use std::{cell::UnsafeCell, sync::Once};

/// `ResettableOnceLock` provides thread-safe access to a value of type `T` via a specific state
/// machine.
///
/// * Fresh values of the lock start in a "to be updated" state.
/// * When values are in "to be updated" then calls to `get` will return None.
/// * In the to be updated state, the first call to `get_or_update` will run `update` on the object
///   stored in the lock and transition the lock to the "updated" state.
/// * Once in the updated state, calls to `get` will return a reference to the shared object,
///   future calls to `get_or_update` will behave just like a call to `get` in this way.
/// * A call to `reset` will transition the lock back to the "to be updated" state. Crucially, this
///   requires mutable access to the lock object, making the patterns expressable via this lock less
///   expressive than a regular mutex.
pub struct ResettableOnceLock<T> {
    data: UnsafeCell<T>,
    update: Once,
}

unsafe impl<T: Send> Send for ResettableOnceLock<T> {}
unsafe impl<T: Send> Sync for ResettableOnceLock<T> {}

impl<T> ResettableOnceLock<T> {
    pub fn new(elt: T) -> ResettableOnceLock<T> {
        ResettableOnceLock {
            data: UnsafeCell::new(elt),
            update: Once::new(),
        }
    }

    pub fn get(&self) -> Option<&T> {
        if self.update.is_completed() {
            // SAFETY: if `is_completed` is true, then no one will access this value mutably until
            // a call to `reset`, which requires mutable access to the underlying lock.
            unsafe { Some(&*self.data.get()) }
        } else {
            None
        }
    }

    pub fn get_or_update(&self, update: impl FnOnce(&mut T)) -> &T {
        if let Some(elt) = self.get() {
            return elt;
        }
        self.update.call_once_force(|_| {
            // SAFETY: We are calling this within `call_once_force` which guarantees a single
            // thread will ever run at a given time.
            //
            // Further mutable accesses have to wait for a call to `reset`.
            let mut_elt = unsafe { &mut *self.data.get() };

            update(mut_elt);
        });
        // SAFETY: same as `get`. We know that is_completed() will return true if we get to this
        // point.
        unsafe { &*self.data.get() }
    }

    /// Reset the state of the lock to gate further accesses to the underlying value on another
    /// call to `get_or_update`.
    pub fn reset(&mut self) {
        self.update = Once::new();
    }
}
