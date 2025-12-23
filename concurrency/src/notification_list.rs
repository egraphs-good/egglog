//! A type for maintaining a "notified" subset of dense numeric identifiers.
//!
//! Clients can create a [`NotificationList`] and then notify (dense) numeric identifiers.
//! Then a client can `reset` the list, extracting the set of items that were notified.
//!
//! This is used to efficiently handle collecting a subset of tables that have been notified when
//! running a set of egglog rules. egglog databases can have hundreds of tables, and tight loops
//! often only need to take a few tables into account.
//!
//! The main complexity in the implementation here is to avoid contention in the common case that
//! someone is notifying a table that has already been notified by a different thread.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::ConcurrentVec;
use egglog_numeric_id::NumericId;

/// Tracks which dense numeric identifiers have been notified since the last reset.
#[derive(Clone)]
pub struct NotificationList<K: NumericId> {
    inner: Arc<Inner<K>>,
}

impl<K: NumericId> Default for NotificationList<K> {
    fn default() -> Self {
        Self {
            inner: Arc::new(Inner::default()),
        }
    }
}

impl<K: NumericId> NotificationList<K> {
    /// Notify a given item.
    ///
    /// It is expected that the space of `item`s is fairly dense: this implementation will use O(n)
    /// space where n is the largest value of `item` passed.
    pub fn notify(&self, item: K) {
        let index = item.index();
        self.inner
            .states
            .resize_with(index + 1, NotificationState::default);
        {
            let read_guard = self.inner.states.read();
            let state = &read_guard[index];
            if state.already_notified() {
                return;
            }
            if state.attempt_notify() {
                return;
            }
        }
        // We were the first to notify this state. Record this `item`.
        self.inner.notified.lock().unwrap().push(item);
    }

    /// Clears all notification state and returns a list of notified items since the last `reset`.
    ///
    /// NB: this method will have unpredictable behavior when it comes to concurrent calls to
    /// `reset` and `notify`. In such a situation, events can be notified more than once.
    pub fn reset(&self) -> Vec<K> {
        // TODO: we can make an optimized version of this that takes advantage of the  exclusive
        // reference.
        let notified = {
            let mut handle = self.inner.notified.lock().unwrap();
            std::mem::take(&mut *handle)
        };
        let handle = self.inner.states.read();
        for &item in &notified {
            handle[item.index()].reset();
        }
        notified
    }
}

struct Inner<K: NumericId> {
    notified: Mutex<Vec<K>>,
    states: ConcurrentVec<NotificationState>,
}

impl<K: NumericId> Default for Inner<K> {
    fn default() -> Self {
        Self {
            notified: Mutex::new(Vec::new()),
            states: ConcurrentVec::default(),
        }
    }
}

#[derive(Default)]
struct NotificationState {
    notified: AtomicBool,
}

impl NotificationState {
    fn reset(&self) {
        self.notified.store(false, Ordering::Relaxed)
    }

    fn already_notified(&self) -> bool {
        self.notified.load(Ordering::Relaxed)
    }

    fn attempt_notify(&self) -> bool {
        self.notified.swap(true, Ordering::Relaxed)
    }
}
