//! A small scoped thread pool with global and worker-local work queues.
//!
//! [`ThreadPool`] owns a fixed set of primary worker threads. Each worker
//! owns a private deque in addition to receiving boxed `'static` jobs from a
//! shared channel. Local work is pushed and popped at the back, giving the
//! owner depth-first execution. At a bounded cadence, a worker with stalled
//! peers packages the oldest half of its private work as one global batch. A
//! primary worker that blocks waiting for nested scoped work helps drain its
//! local deque first and then the shared queue until that nested scope
//! completes, so the pool does not lose a worker while work it needs may still
//! be queued.
//! [`Scope`] provides the safe scoped API on top of those `'static` jobs by
//! erasing task lifetimes when work is queued, then waiting for all work in the
//! scope before returning to the caller.
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
    cell::{Cell, UnsafeCell},
    collections::VecDeque,
    marker::PhantomData,
    mem,
    panic::{self, AssertUnwindSafe},
    ptr::{self, NonNull},
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam::channel::{Receiver, Sender, bounded, select_biased, unbounded};
use crossbeam::utils::CachePadded;

const EXPECTED_SHIFT: u32 = 32;
const COMPLETED_MASK: u64 = u32::MAX as u64;
// Keep inline helping bounded so pathological nested scopes move onto a fresh
// stack before exhausting the current worker stack.
const MAX_INLINE_SCOPE_HELP_DEPTH: usize = 64;
const LOCAL_DONATION_INTERVAL: Duration = Duration::from_millis(10);

type Job = Box<dyn FnOnce() + Send + 'static>;
type ScopedJob<'scope> = Box<dyn FnOnce() + Send + 'scope>;
type PanicPayload = Box<dyn Any + Send + 'static>;

thread_local! {
    static CURRENT_POOL: Cell<*const ThreadPoolState> = const { Cell::new(ptr::null()) };
    static CURRENT_WORKER: Cell<*const WorkerContext> = const { Cell::new(ptr::null()) };
    static IS_BACKGROUND_WORKER: Cell<bool> = const { Cell::new(false) };
    static INLINE_SCOPE_HELP_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// A cumulative snapshot of a thread pool's scheduler instrumentation.
///
/// Counter fields and `stalled_time` are cumulative since the pool was created
/// or [`ThreadPool::reset_scheduler_metrics`] was called. Queue-depth fields
/// are high-water marks over that same period. `stalled_workers` is a gauge at
/// the instant of the snapshot rather than a cumulative value.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SchedulerMetrics {
    /// Jobs pushed onto the shared global queue, including donations.
    pub global_pushes: u64,
    /// Jobs popped from the shared global queue.
    pub global_pops: u64,
    /// Jobs successfully pushed onto worker-private queues.
    pub local_pushes: u64,
    /// Jobs popped by their owning worker from a private queue.
    pub local_pops: u64,
    /// Jobs moved from private queues to the global queue.
    pub donated_jobs: u64,
    /// Global queue batches created from donated private jobs.
    pub donation_batches: u64,
    /// Aggregate wall-clock time for which primary workers were stalled.
    ///
    /// This includes the elapsed portion of stalls that are still in progress
    /// when the snapshot is taken, which makes before/after snapshots useful
    /// even when the pool is idle at both endpoints.
    pub stalled_time: Duration,
    /// Number of primary workers currently waiting for global work.
    pub stalled_workers: usize,
    /// Largest number of jobs observed in any one worker-private queue.
    pub max_local_queue_depth: usize,
    /// Largest number of jobs observed in the global queue.
    pub max_global_queue_depth: usize,
}

impl SchedulerMetrics {
    /// Return the cumulative counter and stalled-time difference from `earlier`.
    ///
    /// Queue high-water marks cannot in general be subtracted. The returned
    /// value therefore retains the later snapshot's high-water marks and its
    /// current `stalled_workers` gauge. Call
    /// [`ThreadPool::reset_scheduler_metrics`] immediately before an isolated
    /// measurement when interval-specific queue high-water marks are needed.
    pub fn delta_since(self, earlier: Self) -> Self {
        Self {
            global_pushes: self.global_pushes.saturating_sub(earlier.global_pushes),
            global_pops: self.global_pops.saturating_sub(earlier.global_pops),
            local_pushes: self.local_pushes.saturating_sub(earlier.local_pushes),
            local_pops: self.local_pops.saturating_sub(earlier.local_pops),
            donated_jobs: self.donated_jobs.saturating_sub(earlier.donated_jobs),
            donation_batches: self
                .donation_batches
                .saturating_sub(earlier.donation_batches),
            stalled_time: self
                .stalled_time
                .checked_sub(earlier.stalled_time)
                .unwrap_or_default(),
            stalled_workers: self.stalled_workers,
            max_local_queue_depth: self.max_local_queue_depth,
            max_global_queue_depth: self.max_global_queue_depth,
        }
    }
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

/// Run `f` with no current thread pool installed on this thread.
///
/// This is useful for serial operations that must not inherit an ambient pool.
/// The previous current pool, if any, is restored before this function returns
/// or unwinds.
///
/// # Examples
///
/// ```
/// use egglog_concurrency::{ThreadPool, current_num_threads, without_current_pool};
///
/// let pool = ThreadPool::new(2);
/// pool.install(|| {
///     assert_eq!(current_num_threads(), 2);
///     without_current_pool(|| {
///         assert_eq!(current_num_threads(), 1);
///     });
///     assert_eq!(current_num_threads(), 2);
/// });
/// ```
pub fn without_current_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    install_pool_ptr(ptr::null(), f)
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

/// A thread pool with a shared queue and a private deque per primary worker.
///
/// The pool owns a fixed number of primary workers. A primary worker that
/// blocks waiting for nested scoped work helps drain its private deque before
/// the shared queue until the nested scope completes.
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
            .map(|worker_id| spawn_worker(receiver.clone(), state_ptr, worker_id))
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

    /// Return a cumulative snapshot of scheduler instrumentation.
    ///
    /// A snapshot includes the elapsed portion of stalls that are still active
    /// at the instant it is taken. Consequently, subtracting a before snapshot
    /// from an after snapshot with [`SchedulerMetrics::delta_since`] does not
    /// attribute idle time before the first snapshot to the measured interval.
    pub fn scheduler_metrics(&self) -> SchedulerMetrics {
        self.state.instrumentation.snapshot()
    }

    /// Reset cumulative scheduler instrumentation and queue high-water marks.
    ///
    /// Active worker stalls are split at the reset instant, so idle time before
    /// the reset is not charged to later snapshots. For meaningful queue
    /// high-water marks, call this while no callbacks are running or being
    /// scheduled; this method does not pause the pool.
    pub fn reset_scheduler_metrics(&self) {
        self.state.instrumentation.reset()
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

#[derive(Default)]
struct StallTimerState {
    completed: Duration,
    started: Option<Instant>,
}

struct WorkerInstrumentation {
    local_pushes: AtomicU64,
    local_pops: AtomicU64,
    donated_jobs: AtomicU64,
    donation_batches: AtomicU64,
    max_local_queue_depth: AtomicUsize,
    stall_timer: Mutex<StallTimerState>,
}

impl WorkerInstrumentation {
    fn new() -> Self {
        Self {
            local_pushes: AtomicU64::new(0),
            local_pops: AtomicU64::new(0),
            donated_jobs: AtomicU64::new(0),
            donation_batches: AtomicU64::new(0),
            max_local_queue_depth: AtomicUsize::new(0),
            stall_timer: Mutex::new(StallTimerState::default()),
        }
    }
}

struct SchedulerInstrumentation {
    global_pushes: AtomicU64,
    global_pops: AtomicU64,
    stalled_workers: AtomicUsize,
    workers: Vec<CachePadded<WorkerInstrumentation>>,
    global_queue_depth: AtomicUsize,
    max_global_queue_depth: AtomicUsize,
}

impl SchedulerInstrumentation {
    fn new(thread_count: usize) -> Self {
        Self {
            global_pushes: AtomicU64::new(0),
            global_pops: AtomicU64::new(0),
            stalled_workers: AtomicUsize::new(0),
            workers: (0..thread_count)
                .map(|_| CachePadded::new(WorkerInstrumentation::new()))
                .collect(),
            global_queue_depth: AtomicUsize::new(0),
            max_global_queue_depth: AtomicUsize::new(0),
        }
    }

    fn snapshot(&self) -> SchedulerMetrics {
        let now = Instant::now();
        let stalled_time = self
            .workers
            .iter()
            .map(|worker| {
                let timer = worker
                    .stall_timer
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                timer.completed.saturating_add(
                    timer
                        .started
                        .map_or(Duration::ZERO, |started| now.duration_since(started)),
                )
            })
            .fold(Duration::ZERO, Duration::saturating_add);
        let local_pushes = self
            .workers
            .iter()
            .map(|worker| worker.local_pushes.load(Ordering::Relaxed))
            .sum();
        let local_pops = self
            .workers
            .iter()
            .map(|worker| worker.local_pops.load(Ordering::Relaxed))
            .sum();
        let donated_jobs = self
            .workers
            .iter()
            .map(|worker| worker.donated_jobs.load(Ordering::Relaxed))
            .sum();
        let donation_batches = self
            .workers
            .iter()
            .map(|worker| worker.donation_batches.load(Ordering::Relaxed))
            .sum();
        let max_local_queue_depth = self
            .workers
            .iter()
            .map(|worker| worker.max_local_queue_depth.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);

        SchedulerMetrics {
            global_pushes: self.global_pushes.load(Ordering::Relaxed),
            global_pops: self.global_pops.load(Ordering::Relaxed),
            local_pushes,
            local_pops,
            donated_jobs,
            donation_batches,
            stalled_time,
            stalled_workers: self.stalled_workers.load(Ordering::Acquire),
            max_local_queue_depth,
            max_global_queue_depth: self.max_global_queue_depth.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        let now = Instant::now();
        for worker in &self.workers {
            let mut timer = worker
                .stall_timer
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            timer.completed = Duration::ZERO;
            if timer.started.is_some() {
                timer.started = Some(now);
            }
            drop(timer);

            worker.local_pushes.store(0, Ordering::Relaxed);
            worker.local_pops.store(0, Ordering::Relaxed);
            worker.donated_jobs.store(0, Ordering::Relaxed);
            worker.donation_batches.store(0, Ordering::Relaxed);
            worker.max_local_queue_depth.store(0, Ordering::Relaxed);
        }

        self.global_pushes.store(0, Ordering::Relaxed);
        self.global_pops.store(0, Ordering::Relaxed);
        self.max_global_queue_depth.store(
            self.global_queue_depth.load(Ordering::Acquire),
            Ordering::Relaxed,
        );
    }

    fn record_global_push(&self) {
        self.global_pushes.fetch_add(1, Ordering::Relaxed);
        let depth = self.global_queue_depth.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_global_queue_depth
            .fetch_max(depth, Ordering::Relaxed);
    }

    fn cancel_global_push(&self) {
        self.global_pushes.fetch_sub(1, Ordering::Relaxed);
        self.global_queue_depth.fetch_sub(1, Ordering::AcqRel);
    }

    fn record_global_pop(&self) {
        self.global_pops.fetch_add(1, Ordering::Relaxed);
        self.global_queue_depth.fetch_sub(1, Ordering::AcqRel);
    }

    fn record_local_push(&self, worker_id: usize, depth: usize) {
        let worker = &self.workers[worker_id];
        worker.local_pushes.fetch_add(1, Ordering::Relaxed);
        worker
            .max_local_queue_depth
            .fetch_max(depth, Ordering::Relaxed);
    }

    fn record_local_pop(&self, worker_id: usize) {
        self.workers[worker_id]
            .local_pops
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_donation(&self, worker_id: usize, jobs: usize, batches: usize) {
        let worker = &self.workers[worker_id];
        worker
            .donated_jobs
            .fetch_add(jobs as u64, Ordering::Relaxed);
        worker
            .donation_batches
            .fetch_add(batches as u64, Ordering::Relaxed);
    }

    fn stalled_workers(&self) -> usize {
        // This counter is only an advisory scheduling hint. Queue operations
        // provide the synchronization needed to publish and consume jobs.
        self.stalled_workers.load(Ordering::Relaxed)
    }

    fn has_stalled_workers(&self) -> bool {
        self.stalled_workers() != 0
    }

    fn begin_stall(&self, worker_id: usize) {
        let mut timer = self.workers[worker_id]
            .stall_timer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        debug_assert!(timer.started.is_none());
        timer.started = Some(Instant::now());
        drop(timer);
        self.stalled_workers.fetch_add(1, Ordering::AcqRel);
    }

    fn end_stall(&self, worker_id: usize) {
        let now = Instant::now();
        let mut timer = self.workers[worker_id]
            .stall_timer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let started = timer
            .started
            .take()
            .expect("primary worker ended a stall it did not begin");
        timer.completed += now.duration_since(started);
        drop(timer);
        self.stalled_workers.fetch_sub(1, Ordering::AcqRel);
    }
}

struct StalledWorkerGuard<'pool> {
    instrumentation: &'pool SchedulerInstrumentation,
    worker_id: usize,
}

impl<'pool> StalledWorkerGuard<'pool> {
    fn new(instrumentation: &'pool SchedulerInstrumentation, worker_id: usize) -> Self {
        instrumentation.begin_stall(worker_id);
        Self {
            instrumentation,
            worker_id,
        }
    }
}

impl Drop for StalledWorkerGuard<'_> {
    fn drop(&mut self) {
        self.instrumentation.end_stall(self.worker_id);
    }
}

struct ThreadPoolState {
    sender: Option<Sender<Job>>,
    receiver: Receiver<Job>,
    thread_count: usize,
    instrumentation: SchedulerInstrumentation,
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
            instrumentation: SchedulerInstrumentation::new(thread_count),
            #[cfg(test)]
            backup_workers_spawned: AtomicUsize::new(0),
            #[cfg(test)]
            backup_workers_live: AtomicUsize::new(0),
        }
    }

    fn thread_count(&self) -> usize {
        self.thread_count
    }

    fn enqueue_global(&self, job: Job) {
        let Some(sender) = self.sender.as_ref() else {
            job();
            return;
        };

        self.instrumentation.record_global_push();
        if let Err(error) = sender.send(job) {
            self.instrumentation.cancel_global_push();
            error.0();
        }
    }

    fn try_recv_global(&self) -> Result<Job, crossbeam::channel::TryRecvError> {
        let result = self.receiver.try_recv();
        if result.is_ok() {
            self.instrumentation.record_global_pop();
        }
        result
    }

    fn recv_global(&self, worker_id: usize) -> Result<Job, crossbeam::channel::RecvError> {
        let result = {
            let _stalled = StalledWorkerGuard::new(&self.instrumentation, worker_id);
            self.receiver.recv()
        };
        if result.is_ok() {
            self.instrumentation.record_global_pop();
        }
        result
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

    fn wait_for_scope_completion(&self, done: &Receiver<()>) {
        if !is_background_worker_thread() {
            receive_scope_completion(done);
            return;
        }

        let Some(_guard) = InlineScopeHelpGuard::try_enter() else {
            let worker_id = current_worker_id(self);
            donate_all_from_current_worker(self);
            let _backup = BackupWorker::spawn(self);
            let _stalled = worker_id
                .map(|worker_id| StalledWorkerGuard::new(&self.instrumentation, worker_id));
            receive_scope_completion(done);
            return;
        };

        loop {
            match done.try_recv() {
                Ok(()) => break,
                Err(crossbeam::channel::TryRecvError::Empty) => {}
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    panic!("scope completion sender dropped before the scope completed");
                }
            }

            if let Some(job) = pop_current_worker_local(self) {
                job();
                continue;
            }

            match self.try_recv_global() {
                Ok(job) => {
                    job();
                    continue;
                }
                Err(crossbeam::channel::TryRecvError::Empty) => {}
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    receive_scope_completion(done);
                    break;
                }
            }

            enum WaitEvent {
                Scope(Result<(), crossbeam::channel::RecvError>),
                Global(Result<Job, crossbeam::channel::RecvError>),
            }

            let worker_id = current_worker_id(self);
            let event = {
                let _stalled = worker_id
                    .map(|worker_id| StalledWorkerGuard::new(&self.instrumentation, worker_id));
                select_biased! {
                    recv(done) -> message => WaitEvent::Scope(message),
                    recv(self.receiver) -> message => WaitEvent::Global(message),
                }
            };

            match event {
                WaitEvent::Scope(message) => {
                    expect_scope_completion(message);
                    break;
                }
                WaitEvent::Global(message) => match message {
                    Ok(job) => {
                        self.instrumentation.record_global_pop();
                        job();
                    }
                    Err(_) => {
                        receive_scope_completion(done);
                        break;
                    }
                },
            }
        }
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

/// State owned and accessed exclusively by one primary worker thread.
///
/// The deque uses interior mutability so no Rust borrow of the queue is held
/// while a job is running. Jobs may recursively call `spawn_local`, which then
/// reaches the same deque through `CURRENT_WORKER` on that thread.
struct WorkerContext {
    pool: ThreadPoolStatePtr,
    worker_id: usize,
    local: UnsafeCell<VecDeque<Job>>,
    last_donation: Cell<Instant>,
}

impl WorkerContext {
    fn new(pool: ThreadPoolStatePtr, worker_id: usize) -> Self {
        Self {
            pool,
            worker_id,
            local: UnsafeCell::new(VecDeque::new()),
            last_donation: Cell::new(Instant::now()),
        }
    }

    fn belongs_to(&self, pool: &ThreadPoolState) -> bool {
        ptr::eq(self.pool.0, pool)
    }

    fn local_queue_depth(&self) -> usize {
        // SAFETY: a `WorkerContext` is installed only on its owning worker and
        // `CURRENT_WORKER` is never made available to another thread.
        unsafe { (&*self.local.get()).len() }
    }

    fn push_local(&self, pool: &ThreadPoolState, job: Job) {
        // SAFETY: a `WorkerContext` is installed only on its owning worker and
        // `CURRENT_WORKER` is never made available to another thread.
        let (depth, was_empty) = unsafe {
            let queue = &mut *self.local.get();
            let was_empty = queue.is_empty();
            queue.push_back(job);
            (queue.len(), was_empty)
        };
        // A worker that has been idle for longer than the interval should
        // still get one full collection window. Otherwise its first two local
        // jobs can be donated immediately using a stale timestamp.
        if was_empty {
            self.last_donation.set(Instant::now());
        }
        pool.instrumentation
            .record_local_push(self.worker_id, depth);
        self.donate_half_if_due(pool);
    }

    fn pop_local(&self, pool: &ThreadPoolState) -> Option<Job> {
        self.donate_half_if_due(pool);
        // The owner takes the newest work, retaining depth-first locality.
        // SAFETY: see `push_local`.
        let job = unsafe { (&mut *self.local.get()).pop_back() };
        if job.is_some() {
            pool.instrumentation.record_local_pop(self.worker_id);
        }
        job
    }

    fn donate_half_if_due(&self, pool: &ThreadPoolState) {
        if !pool.instrumentation.has_stalled_workers() {
            return;
        }

        let now = Instant::now();
        if now.saturating_duration_since(self.last_donation.get()) < LOCAL_DONATION_INTERVAL {
            return;
        }

        // Donation takes the opposite end from the owner. Package the oldest
        // half as one global job so its recipient can process related siblings
        // without repeated channel traffic or cache migration.
        // SAFETY: see `push_local`.
        let donated = unsafe {
            let queue = &mut *self.local.get();
            let count = queue.len() / 2;
            queue.drain(..count).collect::<Vec<_>>()
        };
        if donated.is_empty() {
            return;
        }

        self.last_donation.set(now);
        self.publish_donation_batch(pool, donated);
    }

    fn donate_all(&self, pool: &ThreadPoolState) {
        // The inline-help depth escape hatch blocks this worker and starts a
        // backup consumer. Publish everything so private work remains live.
        // SAFETY: see `push_local`.
        let donated = unsafe { (&mut *self.local.get()).drain(..).collect::<Vec<_>>() };
        self.publish_donation_individually(pool, donated);
    }

    fn publish_donation_batch(&self, pool: &ThreadPoolState, donated: Vec<Job>) {
        if donated.is_empty() {
            return;
        }

        pool.instrumentation
            .record_donation(self.worker_id, donated.len(), 1);
        pool.enqueue_global(Box::new(move || {
            for job in donated {
                job();
            }
        }));
    }

    fn publish_donation_individually(&self, pool: &ThreadPoolState, donated: Vec<Job>) {
        if donated.is_empty() {
            return;
        }

        pool.instrumentation
            .record_donation(self.worker_id, donated.len(), donated.len());
        for job in donated {
            pool.enqueue_global(job);
        }
    }
}

fn install_worker_context<F, R>(worker: &WorkerContext, f: F) -> R
where
    F: FnOnce() -> R,
{
    CURRENT_WORKER.with(|current| {
        let previous = current.replace(worker);
        let _guard = CurrentWorkerGuard { current, previous };
        f()
    })
}

struct CurrentWorkerGuard<'a> {
    current: &'a Cell<*const WorkerContext>,
    previous: *const WorkerContext,
}

impl Drop for CurrentWorkerGuard<'_> {
    fn drop(&mut self) {
        self.current.set(self.previous);
    }
}

fn with_current_worker<F, R>(f: F) -> R
where
    F: FnOnce(Option<&WorkerContext>) -> R,
{
    CURRENT_WORKER.with(|current| {
        let worker = current.get();
        // SAFETY: the pointer is installed from a live `WorkerContext` on this
        // same thread and is restored before that context is dropped.
        f(unsafe { worker.as_ref() })
    })
}

fn current_worker_id(pool: &ThreadPoolState) -> Option<usize> {
    with_current_worker(|worker| {
        worker
            .filter(|worker| worker.belongs_to(pool))
            .map(|worker| worker.worker_id)
    })
}

fn push_current_worker_local(pool: &ThreadPoolState, job: Job) -> Result<(), Job> {
    with_current_worker(
        |worker| match worker.filter(|worker| worker.belongs_to(pool)) {
            Some(worker) => {
                worker.push_local(pool, job);
                Ok(())
            }
            None => Err(job),
        },
    )
}

fn pop_current_worker_local(pool: &ThreadPoolState) -> Option<Job> {
    with_current_worker(|worker| {
        worker
            .filter(|worker| worker.belongs_to(pool))
            .and_then(|worker| worker.pop_local(pool))
    })
}

fn current_worker_local_queue_depth(pool: &ThreadPoolState) -> usize {
    with_current_worker(|worker| {
        worker
            .filter(|worker| worker.belongs_to(pool))
            .map_or(0, WorkerContext::local_queue_depth)
    })
}

fn donate_all_from_current_worker(pool: &ThreadPoolState) {
    with_current_worker(|worker| {
        if let Some(worker) = worker.filter(|worker| worker.belongs_to(pool)) {
            worker.donate_all(pool);
        }
    });
}

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

struct InlineScopeHelpGuard {
    previous: usize,
}

impl InlineScopeHelpGuard {
    fn try_enter() -> Option<Self> {
        INLINE_SCOPE_HELP_DEPTH.with(|depth| {
            let previous = depth.get();
            if previous >= MAX_INLINE_SCOPE_HELP_DEPTH {
                return None;
            }

            depth.set(previous + 1);
            Some(Self { previous })
        })
    }
}

impl Drop for InlineScopeHelpGuard {
    fn drop(&mut self) {
        INLINE_SCOPE_HELP_DEPTH.with(|depth| depth.set(self.previous));
    }
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
    pool: ThreadPoolStatePtr,
    state: ScopeState,
    _scope: PhantomData<fn(&'scope ()) -> &'scope ()>,
}

impl<'scope> Scope<'scope> {
    fn new(pool: &ThreadPoolState) -> Self {
        Self {
            pool: ThreadPoolStatePtr::new(pool),
            state: ScopeState::new(),
            _scope: PhantomData,
        }
    }

    /// Spawn a callback onto the pool's global queue.
    ///
    /// The callback receives the current scope and may add nested work. It may
    /// borrow values with the scope lifetime. The scope waits for the callback
    /// before returning, so those borrows cannot outlive their owners. This is
    /// retained as a compatibility alias for [`Scope::spawn_global`].
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
        self.spawn_global(f);
    }

    /// Spawn a callback onto the pool's global queue.
    ///
    /// Global work is visible to every worker immediately and is appropriate
    /// for coarse top-level partitioning.
    pub fn spawn_global<F>(&self, f: F)
    where
        F: FnOnce(&Scope<'scope>) + Send + 'scope,
    {
        let job = self.prepare_job(f);
        // SAFETY: this scope was created from a live pool, and the pool cannot
        // be dropped while its borrowed `scope` call is active.
        unsafe { self.pool.as_ref() }.enqueue_global(job);
    }

    /// Spawn a callback onto the current worker's private deque.
    ///
    /// The owning worker executes local jobs depth-first before consulting the
    /// global queue. At a bounded cadence, a worker with stalled peers packages
    /// the oldest half of its private queue as one global batch. Calls made
    /// outside a primary worker of this same pool—including calls from the root
    /// callback and from a worker of a nested, different pool—fall back to the
    /// global queue.
    pub fn spawn_local<F>(&self, f: F)
    where
        F: FnOnce(&Scope<'scope>) + Send + 'scope,
    {
        let job = self.prepare_job(f);
        // SAFETY: see `spawn_global`.
        let pool = unsafe { self.pool.as_ref() };
        if let Err(job) = push_current_worker_local(pool, job) {
            pool.enqueue_global(job);
        }
    }

    /// Return the number of primary workers currently waiting for global work.
    ///
    /// This is a cheap, relaxed, advisory snapshot of the scheduler's atomic
    /// stalled worker gauge. The value may become stale immediately and must
    /// not be used for correctness decisions or as a reservation of idle
    /// capacity. Unlike [`ThreadPool::scheduler_metrics`], this method does not
    /// collect counters, lock stall timers, or scan the pool's workers.
    pub fn stalled_workers(&self) -> usize {
        // SAFETY: this scope was created from a live pool, and the pool cannot
        // be dropped while its borrowed `scope` call is active.
        unsafe { self.pool.as_ref() }
            .instrumentation
            .stalled_workers()
    }

    /// Return whether any primary worker currently appears to need global work.
    ///
    /// This is the boolean convenience form of [`Scope::stalled_workers`] and
    /// has the same advisory, inherently racy semantics.
    pub fn has_stalled_workers(&self) -> bool {
        // SAFETY: see `Scope::stalled_workers`.
        unsafe { self.pool.as_ref() }
            .instrumentation
            .has_stalled_workers()
    }

    /// Return the current worker's private queue depth.
    ///
    /// The result is exact for the current worker because its deque is private.
    /// Calls made outside a primary worker of this pool return zero.
    pub fn local_queue_depth(&self) -> usize {
        // SAFETY: see `Scope::stalled_workers`.
        current_worker_local_queue_depth(unsafe { self.pool.as_ref() })
    }

    fn prepare_job<F>(&self, f: F) -> Job
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
        unsafe { erase_job_lifetime(job) }
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
    done_sender: Sender<()>,
    done_receiver: Receiver<()>,
    panic: Mutex<Option<PanicPayload>>,
}

impl ScopeState {
    fn new() -> Self {
        let (done_sender, done_receiver) = bounded(1);
        Self {
            completion: AtomicCounts::with_root_callback(),
            done_sender,
            done_receiver,
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
            // Keep the channel alive while signaling completion. Once the
            // message is visible, the waiting root may receive it and drop the
            // `ScopeState` before `try_send` returns on this worker.
            let done_sender = self.done_sender.clone();
            done_sender
                .try_send(())
                .expect("scope completion should only be signaled once");
            true
        } else {
            false
        }
    }

    fn wait(&self, pool: &ThreadPoolState) {
        pool.wait_for_scope_completion(&self.done_receiver);
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

fn receive_scope_completion(done: &Receiver<()>) {
    expect_scope_completion(done.recv());
}

fn expect_scope_completion(message: Result<(), crossbeam::channel::RecvError>) {
    message.expect("scope completion sender dropped before the scope completed");
}

/// A short-lived queue consumer used once inline scope helping gets too deep.
///
/// A worker that waits for nested scoped work usually helps by running queued
/// jobs on the same stack. Deeply nested scopes can make that recursive
/// execution chain large enough to overflow the worker stack. `BackupWorker`
/// provides the same liveness escape hatch as the original implementation, but
/// only after the waiting worker has already helped inline up to
/// [`MAX_INLINE_SCOPE_HELP_DEPTH`].
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
                                Ok(job) => {
                                    pool.instrumentation.record_global_pop();
                                    job();
                                }
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

fn spawn_worker(
    receiver: Receiver<Job>,
    pool: ThreadPoolStatePtr,
    worker_id: usize,
) -> JoinHandle<()> {
    thread::spawn(move || {
        // SAFETY: each worker is joined before the boxed pool state is dropped.
        let pool = unsafe { pool.as_ref() };
        let worker = WorkerContext::new(ThreadPoolStatePtr::new(pool), worker_id);
        install_pool(pool, || {
            install_worker_context(&worker, || {
                install_background_worker(|| {
                    loop {
                        if let Some(job) = worker.pop_local(pool) {
                            job();
                            continue;
                        }

                        match receiver.try_recv() {
                            Ok(job) => {
                                pool.instrumentation.record_global_pop();
                                job();
                                continue;
                            }
                            Err(crossbeam::channel::TryRecvError::Empty) => {}
                            Err(crossbeam::channel::TryRecvError::Disconnected) => break,
                        }

                        let Ok(job) = pool.recv_global(worker_id) else {
                            break;
                        };
                        job();
                    }
                })
            });
        });
    })
}

fn completed(value: u64) -> u32 {
    (value & COMPLETED_MASK) as u32
}

fn expected(value: u64) -> u32 {
    (value >> EXPECTED_SHIFT) as u32
}

#[cfg(test)]
mod tests;
