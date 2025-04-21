//! A simple concurrent notification object, based on `absl::Notificiation` from
//! the absl C++ library.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Condvar, Mutex,
    },
    time::Duration,
};

/// A simple concurrent notification object, based on `absl::Notificiation` from
/// the absl library. Notifications happen at most once (with future
/// notifications being no-ops). Waiting threads can block, optionally with a
/// timeout.
pub struct Notification {
    has_been_notified: AtomicBool,
    mutex: Mutex<()>,
    cv: Condvar,
}

impl Default for Notification {
    fn default() -> Self {
        Self {
            has_been_notified: AtomicBool::new(false),
            mutex: Mutex::new(()),
            cv: Condvar::new(),
        }
    }
}

impl Drop for Notification {
    fn drop(&mut self) {
        // From absl: want to ensure that a thread running `notify` exits before
        // the object is destroyed.
        let _guard = self.mutex.lock();
    }
}

impl Notification {
    /// Create a fresh notification.
    pub fn new() -> Self {
        Self::default()
    }

    /// Block until `notify` is called.
    pub fn wait(&self) {
        if self.has_been_notified() {
            return;
        }
        let mut lock = self.mutex.lock().unwrap();
        while !self.has_been_notified() {
            lock = self.cv.wait(lock).unwrap();
        }
    }

    pub fn wait_with_timeout(&self, timeout: Duration) -> bool {
        if self.has_been_notified() {
            return true;
        }
        let mut lock = self.mutex.lock().unwrap();
        while !self.has_been_notified() {
            let (next, result) = self.cv.wait_timeout(lock, timeout).unwrap();
            if result.timed_out() {
                return false;
            }
            lock = next;
        }
        self.has_been_notified()
    }

    /// Notify all threads waiting on this notification, and unblock any future
    /// threads who may wait.
    pub fn notify(&self) {
        let _guard = self.mutex.lock().unwrap();
        self.has_been_notified.store(true, Ordering::SeqCst);
        self.cv.notify_all();
    }

    /// Query whether this notification has been notified, without blocking.
    pub fn has_been_notified(&self) -> bool {
        self.has_been_notified.load(Ordering::SeqCst)
    }
}
