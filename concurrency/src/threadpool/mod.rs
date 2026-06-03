//! A small scoped thread pool backed by a shared crossbeam channel.
//!
//! [`ThreadPool`] owns a fixed set of primary worker threads. Each worker
//! receives boxed `'static` jobs from a shared channel and exits when the
//! channel is closed. A primary worker that blocks waiting for scoped work
//! temporarily starts a backup worker so the pool can continue draining queued
//! jobs. [`Scope`] provides the safe scoped API on top of those `'static` jobs
//! by erasing task lifetimes when work is sent to the channel, then waiting for
//! all work in the scope before returning to the caller.
//!
//! The root callback runs on the caller thread, and spawned callbacks run on
//! worker threads. Spawned callbacks receive the same scope reference as the
//! root callback, so they can add nested work. The high half of the scope
//! counter tracks expected work and is incremented before each task is
//! enqueued; the low half tracks completed work. The root callback is counted
//! as one expected item, so the scope is complete once the callback and every
//! spawned task have completed.
//!
//! # Examples
//!
//! ```
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use egglog_concurrency::ThreadPool;
//!
//! let pool = ThreadPool::new(4);
//! let values = [1, 2, 3, 4];
//! let sum = AtomicUsize::new(0);
//!
//! pool.scope(|scope| {
//!     for value in &values {
//!         scope.spawn(|_| {
//!             sum.fetch_add(*value, Ordering::Relaxed);
//!         });
//!     }
//! });
//!
//! assert_eq!(sum.load(Ordering::Relaxed), 10);
//! ```
//!
//! A spawned callback cannot store references that would outlive the borrowed
//! data, even though the implementation erases the callback's lifetime before
//! sending it to a worker:
//!
//! ```compile_fail
//! use egglog_concurrency::ThreadPool;
//!
//! let pool = ThreadPool::new(1);
//! let mut escaped = None;
//!
//! pool.scope(|scope| {
//!     let local = 10;
//!     scope.spawn(|_| {
//!         escaped = Some(&local);
//!     });
//! });
//!
//! assert_eq!(escaped, Some(&10));
//! ```
//!
//! Nested spawned callbacks cannot borrow data owned by the parent spawned
//! callback, because the parent callback may return before its nested work runs:
//!
//! ```compile_fail
//! use egglog_concurrency::ThreadPool;
//!
//! let pool = ThreadPool::new(1);
//! pool.scope(|scope| {
//!     scope.spawn(|scope| {
//!         let local = 10;
//!         scope.spawn(|_| {
//!             println!("{local}");
//!         });
//!     });
//! });
//! ```
//!
//! The scope itself also cannot be stored outside the call to
//! [`ThreadPool::scope`]:
//!
//! ```compile_fail
//! use egglog_concurrency::{Scope, ThreadPool};
//!
//! let pool = ThreadPool::new(1);
//! let mut escaped: Option<&Scope<'_>> = None;
//!
//! pool.scope(|scope| {
//!     scope.spawn(|scope| {
//!         escaped = Some(scope);
//!     });
//! });
//!
//! escaped.unwrap().spawn(|_| {});
//! ```

use std::{
    any::Any,
    cell::Cell,
    marker::PhantomData,
    mem,
    panic::{self, AssertUnwindSafe},
    ptr::{self, NonNull},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
};

#[cfg(test)]
use std::sync::atomic::AtomicUsize;

use crossbeam::channel::{Receiver, Sender, bounded, select_biased, unbounded};

use crate::Notification;

const EXPECTED_SHIFT: u32 = 32;
const COMPLETED_MASK: u64 = u32::MAX as u64;

type Job = Box<dyn FnOnce() + Send + 'static>;
type ScopedJob<'scope> = Box<dyn FnOnce() + Send + 'scope>;
type PanicPayload = Box<dyn Any + Send + 'static>;

thread_local! {
    static CURRENT_POOL: Cell<*const ThreadPoolState> = const { Cell::new(ptr::null()) };
    static IS_BACKGROUND_WORKER: Cell<bool> = const { Cell::new(false) };
}

/// Return the number of threads in the currently installed thread pool.
///
/// Returns `1` when called outside [`ThreadPool::install`], [`ThreadPool::scope`],
/// a free [`scope`] callback, or a worker callback spawned from an installed
/// pool.
///
/// # Examples
///
/// ```
/// use egglog_concurrency::{ThreadPool, current_num_threads};
///
/// assert_eq!(current_num_threads(), 1);
///
/// let pool = ThreadPool::new(2);
/// pool.install(|| {
///     assert_eq!(current_num_threads(), 2);
/// });
///
/// assert_eq!(current_num_threads(), 1);
/// ```
pub fn current_num_threads() -> usize {
    with_current_pool(|pool| pool.map_or(1, ThreadPoolState::thread_count))
}

/// Run `f` in a scope on the currently installed thread pool.
///
/// This is the free-function counterpart to [`ThreadPool::scope`]. It keeps the
/// same in-place semantics: the root callback runs on the caller thread, and
/// spawned work runs on the installed pool's workers.
///
/// # Panics
///
/// Panics if no thread pool is currently installed.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use egglog_concurrency::{ThreadPool, scope};
///
/// let pool = ThreadPool::new(2);
/// let counter = AtomicUsize::new(0);
///
/// pool.install(|| {
///     scope(|scope| {
///         scope.spawn(|_| {
///             counter.fetch_add(1, Ordering::Relaxed);
///         });
///     });
/// });
///
/// assert_eq!(counter.load(Ordering::Relaxed), 1);
/// ```
///
/// ```should_panic
/// egglog_concurrency::scope(|_| {});
/// ```
pub fn scope<'scope, F, R>(f: F) -> R
where
    F: FnOnce(&Scope<'scope>) -> R,
{
    with_current_pool(|pool| match pool {
        Some(pool) => pool.scope(f),
        None => panic!("no egglog thread pool is currently installed"),
    })
}

/// A thread pool using a single shared work queue.
///
/// The pool owns a fixed number of primary workers. It may also create
/// short-lived backup workers while a primary worker is blocked waiting for
/// nested scoped work.
///
/// # Examples
///
/// ```
/// use egglog_concurrency::ThreadPool;
///
/// let pool = ThreadPool::new(2);
/// pool.scope(|scope| {
///     scope.spawn(|_| {});
/// });
/// ```
pub struct ThreadPool {
    state: Box<ThreadPoolState>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a thread pool with `thread_count` worker threads.
    ///
    /// # Panics
    ///
    /// Panics when `thread_count` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use egglog_concurrency::ThreadPool;
    ///
    /// let pool = ThreadPool::new(2);
    /// assert_eq!(pool.thread_count(), 2);
    /// ```
    pub fn new(thread_count: usize) -> Self {
        assert!(
            thread_count > 0,
            "thread pool must have at least one worker"
        );

        let (sender, receiver) = unbounded();
        let state = Box::new(ThreadPoolState::new(sender, receiver.clone(), thread_count));
        let state_ptr = ThreadPoolStatePtr::new(&state);
        let workers = (0..thread_count)
            .map(|_| spawn_worker(receiver.clone(), state_ptr))
            .collect();

        Self { state, workers }
    }

    /// Return the number of primary worker threads owned by this pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use egglog_concurrency::ThreadPool;
    ///
    /// let pool = ThreadPool::new(3);
    /// assert_eq!(pool.thread_count(), 3);
    /// ```
    pub fn thread_count(&self) -> usize {
        self.state.thread_count()
    }

    /// Run `f` with this thread pool installed as the current pool.
    ///
    /// The callback runs on the caller thread. While it is running,
    /// [`current_num_threads`] reports this pool's thread count, the free
    /// [`scope`] function uses this pool, and worker callbacks spawned from this
    /// pool also see this pool as current. If another pool was already
    /// installed on the current thread, it is restored before this method
    /// returns or unwinds.
    ///
    /// # Examples
    ///
    /// ```
    /// use egglog_concurrency::{ThreadPool, current_num_threads};
    ///
    /// let pool = ThreadPool::new(2);
    /// assert_eq!(current_num_threads(), 1);
    ///
    /// let value = pool.install(|| current_num_threads());
    ///
    /// assert_eq!(value, 2);
    /// assert_eq!(current_num_threads(), 1);
    /// ```
    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        install_pool(&self.state, f)
    }

    /// Run `f` in a scope and wait for all work spawned in that scope.
    ///
    /// The root callback runs on the caller thread. Tasks spawned from `f` may
    /// borrow stack data owned by the caller. If a spawned task panics, this
    /// method waits for all previously spawned work and then resumes unwinding
    /// with one of the worker panic payloads. If `f` itself panics, the outer
    /// panic is resumed after spawned work has completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use egglog_concurrency::ThreadPool;
    ///
    /// let pool = ThreadPool::new(2);
    /// let value = AtomicUsize::new(0);
    ///
    /// let result = pool.scope(|scope| {
    ///     scope.spawn(|_| {
    ///         value.store(7, Ordering::Relaxed);
    ///     });
    ///     11
    /// });
    ///
    /// assert_eq!(result, 11);
    /// assert_eq!(value.load(Ordering::Relaxed), 7);
    /// ```
    pub fn scope<'scope, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Scope<'scope>) -> R,
    {
        self.install(|| self.state.scope(f))
    }

    /// Apply `f` to each item in `iter` in parallel.
    ///
    /// Items are pulled from the iterator by the calling thread and enqueued
    /// one by one. The method returns only after every enqueued callback has
    /// completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use egglog_concurrency::ThreadPool;
    ///
    /// let pool = ThreadPool::new(2);
    /// let sum = AtomicUsize::new(0);
    ///
    /// pool.parallel_for_each(0..4, |value| {
    ///     sum.fetch_add(value, Ordering::Relaxed);
    /// });
    ///
    /// assert_eq!(sum.load(Ordering::Relaxed), 6);
    /// ```
    pub fn parallel_for_each<'scope, I, F>(&self, iter: I, f: F)
    where
        I: IntoIterator,
        I::Item: Send + 'scope,
        F: Fn(I::Item) + Sync + 'scope,
    {
        self.scope(|scope| {
            let f = &f;
            for item in iter {
                scope.spawn(move |_| f(item));
            }
        });
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.state.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct ThreadPoolState {
    sender: Option<Sender<Job>>,
    receiver: Receiver<Job>,
    thread_count: usize,
    #[cfg(test)]
    backup_workers_spawned: AtomicUsize,
    #[cfg(test)]
    backup_workers_live: AtomicUsize,
}

impl ThreadPoolState {
    fn new(sender: Sender<Job>, receiver: Receiver<Job>, thread_count: usize) -> Self {
        Self {
            sender: Some(sender),
            receiver,
            thread_count,
            #[cfg(test)]
            backup_workers_spawned: AtomicUsize::new(0),
            #[cfg(test)]
            backup_workers_live: AtomicUsize::new(0),
        }
    }

    fn thread_count(&self) -> usize {
        self.thread_count
    }

    fn scope<'scope, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Scope<'scope>) -> R,
    {
        let scope = Scope::new(self);
        let result = panic::catch_unwind(AssertUnwindSafe(|| f(&scope)));
        scope.complete_root_and_wait();
        let worker_panic = scope.state.take_panic();

        match result {
            Ok(value) => {
                if let Some(payload) = worker_panic {
                    panic::resume_unwind(payload);
                }
                value
            }
            Err(payload) => panic::resume_unwind(payload),
        }
    }

    fn wait_for_notification(&self, notification: &Notification) {
        if !is_background_worker_thread() || notification.has_been_notified() {
            notification.wait();
            return;
        }

        let _backup = BackupWorker::spawn(self);
        notification.wait();
    }

    #[cfg(test)]
    fn backup_workers_spawned(&self) -> usize {
        self.backup_workers_spawned.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn backup_workers_live(&self) -> usize {
        self.backup_workers_live.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy)]
struct ThreadPoolStatePtr(*const ThreadPoolState);

impl ThreadPoolStatePtr {
    fn new(state: &ThreadPoolState) -> Self {
        Self(state)
    }

    unsafe fn as_ref(&self) -> &ThreadPoolState {
        // SAFETY: callers only use this pointer while the owning `ThreadPool`
        // is alive. Worker threads that hold this pointer are joined before the
        // boxed state is dropped.
        unsafe { &*self.0 }
    }
}

// SAFETY: the pointer targets the boxed `ThreadPoolState` owned by
// `ThreadPool`. `ThreadPool::drop` closes the work channel, joins all workers
// holding this pointer, and only then drops the boxed state.
unsafe impl Send for ThreadPoolStatePtr {}

// SAFETY: shared access through this pointer only reads immutable pool state or
// uses internally synchronized channel and atomic operations. The pointed-to
// boxed state remains alive until all worker threads have been joined.
unsafe impl Sync for ThreadPoolStatePtr {}

fn install_pool<F, R>(pool: &ThreadPoolState, f: F) -> R
where
    F: FnOnce() -> R,
{
    install_pool_ptr(pool as *const ThreadPoolState, f)
}

fn install_pool_ptr<F, R>(pool: *const ThreadPoolState, f: F) -> R
where
    F: FnOnce() -> R,
{
    CURRENT_POOL.with(|current| {
        let previous = current.replace(pool);
        let _guard = CurrentPoolGuard { current, previous };
        f()
    })
}

struct CurrentPoolGuard<'a> {
    current: &'a Cell<*const ThreadPoolState>,
    previous: *const ThreadPoolState,
}

impl Drop for CurrentPoolGuard<'_> {
    fn drop(&mut self) {
        self.current.set(self.previous);
    }
}

fn install_background_worker<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    IS_BACKGROUND_WORKER.with(|current| {
        let previous = current.replace(true);
        let _guard = BackgroundWorkerGuard { current, previous };
        f()
    })
}

struct BackgroundWorkerGuard<'a> {
    current: &'a Cell<bool>,
    previous: bool,
}

impl Drop for BackgroundWorkerGuard<'_> {
    fn drop(&mut self) {
        self.current.set(self.previous);
    }
}

fn is_background_worker_thread() -> bool {
    IS_BACKGROUND_WORKER.with(Cell::get)
}

fn with_current_pool<F, R>(f: F) -> R
where
    F: FnOnce(Option<&ThreadPoolState>) -> R,
{
    CURRENT_POOL.with(|current| {
        let pool = current.get();
        // SAFETY: installed pointers are set only by `install_pool` for a
        // borrowed `ThreadPoolState`, or by worker threads whose owning
        // `ThreadPool` joins them before dropping the boxed state. The returned
        // reference is only exposed for the duration of this closure.
        let pool = unsafe { pool.as_ref() };
        f(pool)
    })
}

/// A scoped handle for spawning non-`'static` work onto a [`ThreadPool`].
///
/// `Scope` has an invariant lifetime. This permits spawned callbacks to borrow
/// local data while preventing nested callbacks from smuggling shorter-lived
/// borrows into work that may outlive them.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use egglog_concurrency::ThreadPool;
///
/// let pool = ThreadPool::new(2);
/// let counter = AtomicUsize::new(0);
///
/// pool.scope(|scope| {
///     scope.spawn(|_| {
///         counter.fetch_add(1, Ordering::Relaxed);
///     });
/// });
///
/// assert_eq!(counter.load(Ordering::Relaxed), 1);
/// ```
pub struct Scope<'scope> {
    sender: Sender<Job>,
    pool: ThreadPoolStatePtr,
    state: ScopeState,
    _scope: PhantomData<fn(&'scope ()) -> &'scope ()>,
}

impl<'scope> Scope<'scope> {
    fn new(pool: &ThreadPoolState) -> Self {
        Self {
            sender: pool.sender.as_ref().unwrap().clone(),
            pool: ThreadPoolStatePtr::new(pool),
            state: ScopeState::new(),
            _scope: PhantomData,
        }
    }

    /// Spawn a callback onto the pool.
    ///
    /// The callback receives the current scope and may add nested work. It may
    /// borrow values with the scope lifetime. The scope waits for the callback
    /// before returning, so those borrows cannot outlive their owners.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use egglog_concurrency::ThreadPool;
    ///
    /// let pool = ThreadPool::new(2);
    /// let counter = AtomicUsize::new(0);
    ///
    /// pool.scope(|scope| {
    ///     scope.spawn(|scope| {
    ///         counter.fetch_add(1, Ordering::Relaxed);
    ///         scope.spawn(|_| {
    ///             counter.fetch_add(1, Ordering::Relaxed);
    ///         });
    ///     });
    /// });
    ///
    /// assert_eq!(counter.load(Ordering::Relaxed), 2);
    /// ```
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce(&Scope<'scope>) + Send + 'scope,
    {
        let scope = ScopePtr::new(self);
        let job: ScopedJob<'scope> = Box::new(move || {
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: `Scope::complete_root_and_wait` waits for this job
                // to call `complete_one` before the stack-allocated scope can
                // be destroyed.
                let scope = unsafe { scope.as_ref() };
                f(scope);
            }));
            // SAFETY: `Scope::complete_root_and_wait` waits for this job to call
            // `complete_one` before the stack-allocated scope state can be
            // destroyed.
            let scope = unsafe { scope.as_ref() };
            if let Err(payload) = result {
                scope.state.record_panic(payload);
            }
            scope.state.complete_one();
        });

        self.state.expect_one();
        // SAFETY: every erased job records completion in the scope state, and
        // `Scope::complete_root_and_wait` waits for all expected completions
        // before `ThreadPool::scope` returns.
        enqueue(&self.sender, unsafe { erase_job_lifetime(job) });
    }

    fn complete_root_and_wait(&self) {
        if !self.state.complete_one() {
            // SAFETY: the scope is created from a live `ThreadPoolState`, and
            // `ThreadPool::drop` joins all workers before dropping that boxed
            // state. Scopes cannot outlive the `ThreadPool::scope` call that
            // created them.
            let pool = unsafe { self.pool.as_ref() };
            self.state.wait(pool);
        }
    }
}

struct ScopeState {
    completion: AtomicCounts,
    finished: Notification,
    panic: Mutex<Option<PanicPayload>>,
}

impl ScopeState {
    fn new() -> Self {
        Self {
            completion: AtomicCounts::with_root_callback(),
            finished: Notification::new(),
            panic: Mutex::new(None),
        }
    }

    fn expect_one(&self) {
        self.completion.expect_one();
    }

    fn complete_one(&self) -> bool {
        let previous = self.completion.complete_one();
        let completed = completed(previous) + 1;
        let expected = expected(previous);
        debug_assert!(completed <= expected);

        if completed == expected {
            self.finished.notify();
            true
        } else {
            false
        }
    }

    fn wait(&self, pool: &ThreadPoolState) {
        pool.wait_for_notification(&self.finished);
    }

    fn record_panic(&self, payload: PanicPayload) {
        let mut slot = self.panic.lock().unwrap_or_else(|err| err.into_inner());
        if slot.is_none() {
            *slot = Some(payload);
        }
    }

    fn take_panic(&self) -> Option<PanicPayload> {
        self.panic
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take()
    }
}

struct AtomicCounts(AtomicU64);

impl AtomicCounts {
    fn with_root_callback() -> Self {
        Self(AtomicU64::new(1 << EXPECTED_SHIFT))
    }

    fn expect_one(&self) {
        loop {
            let current = self.0.load(Ordering::Acquire);
            let expected = expected(current);
            assert!(
                expected < u32::MAX,
                "thread pool scope launched more than u32::MAX tasks"
            );

            let next = current + (1 << EXPECTED_SHIFT);
            if self
                .0
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    fn complete_one(&self) -> u64 {
        self.0.fetch_add(1, Ordering::AcqRel)
    }
}

#[derive(Clone, Copy)]
struct ScopePtr<'scope>(NonNull<Scope<'scope>>);

impl<'scope> ScopePtr<'scope> {
    fn new(scope: &Scope<'scope>) -> Self {
        Self(NonNull::from(scope))
    }

    unsafe fn as_ref(&self) -> &Scope<'scope> {
        // SAFETY: callers only dereference this pointer from jobs whose
        // completion is counted by the pointed-to scope. The parent scope waits
        // for every counted completion before the stack-allocated scope is
        // destroyed.
        unsafe { self.0.as_ref() }
    }
}

// SAFETY: `ScopePtr` is only created for a live scope. Each job that receives
// this pointer must call `complete_one`, and the parent scope waits for all
// completions before the scope is destroyed.
unsafe impl Send for ScopePtr<'_> {}

unsafe fn erase_job_lifetime<'scope>(job: ScopedJob<'scope>) -> Job {
    // SAFETY: `ThreadPool::scope` waits for the root callback and every
    // dynamically expected spawned job to finish before returning to the
    // caller, so the erased callback cannot run after its captured `'scope`
    // borrows expire.
    unsafe { mem::transmute::<ScopedJob<'scope>, Job>(job) }
}

struct BackupWorker {
    shutdown: Sender<()>,
    worker: Option<JoinHandle<()>>,
}

impl BackupWorker {
    fn spawn(pool: &ThreadPoolState) -> Self {
        #[cfg(test)]
        pool.backup_workers_spawned.fetch_add(1, Ordering::AcqRel);

        let (shutdown, shutdown_receiver) = bounded(1);
        let receiver = pool.receiver.clone();
        let pool = ThreadPoolStatePtr::new(pool);
        let worker = thread::spawn(move || {
            // SAFETY: backup workers are joined by the worker that spawned them
            // before that worker resumes, and primary workers are joined before
            // the boxed pool state is dropped.
            let pool = unsafe { pool.as_ref() };
            #[cfg(test)]
            let _live = BackupWorkerLiveGuard::new(pool);

            install_pool(pool, || {
                install_background_worker(|| {
                    loop {
                        select_biased! {
                            recv(shutdown_receiver) -> _ => break,
                            recv(receiver) -> message => match message {
                                Ok(job) => job(),
                                Err(_) => break,
                            },
                        }
                    }
                });
            });
        });

        Self {
            shutdown,
            worker: Some(worker),
        }
    }

    fn shutdown_and_join(&mut self) {
        let _ = self.shutdown.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for BackupWorker {
    fn drop(&mut self) {
        self.shutdown_and_join();
    }
}

#[cfg(test)]
struct BackupWorkerLiveGuard<'a> {
    pool: &'a ThreadPoolState,
}

#[cfg(test)]
impl<'a> BackupWorkerLiveGuard<'a> {
    fn new(pool: &'a ThreadPoolState) -> Self {
        pool.backup_workers_live.fetch_add(1, Ordering::AcqRel);
        Self { pool }
    }
}

#[cfg(test)]
impl Drop for BackupWorkerLiveGuard<'_> {
    fn drop(&mut self) {
        self.pool.backup_workers_live.fetch_sub(1, Ordering::AcqRel);
    }
}

fn spawn_worker(receiver: Receiver<Job>, pool: ThreadPoolStatePtr) -> JoinHandle<()> {
    thread::spawn(move || {
        // SAFETY: each worker is joined before the boxed pool state is dropped.
        let pool = unsafe { pool.as_ref() };
        install_pool(pool, || {
            install_background_worker(|| {
                for job in receiver {
                    job();
                }
            });
        });
    })
}

fn enqueue(sender: &Sender<Job>, job: Job) {
    match sender.send(job) {
        Ok(()) => {}
        Err(error) => {
            let job = error.0;
            job();
        }
    }
}

fn completed(value: u64) -> u32 {
    (value & COMPLETED_MASK) as u32
}

fn expected(value: u64) -> u32 {
    (value >> EXPECTED_SHIFT) as u32
}

#[cfg(test)]
mod tests;
