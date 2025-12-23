//! A type for maintaining a "notified" subset of dense integer identifiers.
//!
//! Clients can create a [`ModifiedList`] and then mark (dense) integer idenitifiers as "modified".
//! Then a client can `reset` the list, extracting the set of items that were marked as modified.
//!
//! This is used to efficiently handle collecting a subset of tables that have been modified when
//! running a set of egglog rules. egglog databases can have hundreds of tables, and tight loops
//! often only need to take a few tables into account.
//!
//! The main complexity in the implementation here is to avoid contention in the common case that
//! someone is marking a table as modified that has already been marked by a different thread.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::ConcurrentVec;

#[derive(Default, Clone)]
pub struct ModifiedList {
    inner: Arc<Inner>,
}

impl ModifiedList {
    pub fn mark_modified(&self, item: usize) {
        self.inner
            .states
            .resize_with(item + 1, ModifyState::default);
        {
            let read_guard = self.inner.states.read();
            let state = &read_guard[item];
            if state.already_notified() {
                return;
            }
            if state.attempt_notify() {
                return;
            }
        }
        // We were the first to mark this state as modified. Notify this `item`.
        self.inner.notified.lock().unwrap().push(item);
    }

    /// NB: this method is "lossy" in the presence of concurrent calls to `mark_modified`. The
    /// result will be a valid state, but clear ordering of mark_modified calls is not guaranteed
    /// when there is an intervening reset.
    pub fn reset(&self) -> Vec<usize> {
        // TODO: we can make an optimized version of this that takes advantage of the  exclusive
        // reference.
        let mut notified = Vec::new();
        {
            let mut handle = self.inner.notified.lock().unwrap();
            notified.extend_from_slice(&handle);
            handle.clear();
        }
        let handle = self.inner.states.read();
        for &item in &notified {
            handle[item].reset();
        }
        notified
    }
}

#[derive(Default)]
struct Inner {
    notified: Mutex<Vec<usize>>,
    states: ConcurrentVec<ModifyState>,
}

#[derive(Default)]
struct ModifyState {
    notified: AtomicBool,
}

impl ModifyState {
    fn reset(&self) {
        self.notified.store(true, Ordering::Relaxed)
    }

    fn already_notified(&self) -> bool {
        self.notified.load(Ordering::Relaxed)
    }

    fn attempt_notify(&self) -> bool {
        self.notified.swap(true, Ordering::Relaxed)
    }
}
