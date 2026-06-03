use std::{
    panic,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use crate::{Notification, Scope, ThreadPool, current_num_threads, scope as threadpool_scope};

#[test]
fn scope_runs_spawned_work_and_waits() {
    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);

    pool.scope(|scope| {
        for _ in 0..128 {
            scope.spawn(|_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }
    });

    assert_eq!(counter.load(Ordering::Relaxed), 128);
}

#[test]
fn scope_allows_borrowing_stack_data() {
    let pool = ThreadPool::new(2);
    let values = [1usize, 3, 5, 7, 11];
    let sum = AtomicUsize::new(0);

    pool.scope(|scope| {
        for value in &values {
            scope.spawn(|_| {
                sum.fetch_add(*value, Ordering::Relaxed);
            });
        }
    });

    assert_eq!(sum.load(Ordering::Relaxed), values.iter().sum());
}

#[test]
fn scope_allows_one_task_to_mutate_stack_data() {
    let pool = ThreadPool::new(1);
    let mut value = 0;

    pool.scope(|scope| {
        scope.spawn(|_| {
            value = 42;
        });
    });

    assert_eq!(value, 42);
}

#[test]
fn scope_handles_work_completed_before_expected_is_published() {
    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);
    const TASKS: usize = 64;

    pool.scope(|scope| {
        for _ in 0..TASKS {
            scope.spawn(|_| {
                counter.fetch_add(1, Ordering::Release);
            });
        }

        while counter.load(Ordering::Acquire) != TASKS {
            thread::yield_now();
        }
    });

    assert_eq!(counter.load(Ordering::Acquire), TASKS);
}

#[test]
fn scope_waits_for_last_work_after_expected_is_published() {
    let pool = ThreadPool::new(2);
    let release = Arc::new(Notification::new());
    let entered = Arc::new(Notification::new());
    let scope_returned = Arc::new(AtomicBool::new(false));

    thread::scope(|thread_scope| {
        let release_inner = release.clone();
        let entered_inner = entered.clone();
        let scope_returned_inner = scope_returned.clone();
        let pool = &pool;
        thread_scope.spawn(move || {
            pool.scope(|scope| {
                scope.spawn(move |_| {
                    entered_inner.notify();
                    release_inner.wait();
                });
            });
            scope_returned_inner.store(true, Ordering::Release);
        });

        entered.wait();
        thread::sleep(Duration::from_millis(10));
        assert!(!scope_returned.load(Ordering::Acquire));
        release.notify();
    });

    assert!(scope_returned.load(Ordering::Acquire));
}

#[test]
fn many_scopes_cover_completion_races() {
    let pool = ThreadPool::new(4);

    for round in 0..512 {
        let counter = AtomicUsize::new(0);
        let tasks = round % 65;

        pool.scope(|scope| {
            for task in 0..tasks {
                let counter = &counter;
                scope.spawn(move |_| {
                    if task % 3 == 0 {
                        thread::yield_now();
                    }
                    counter.fetch_add(1, Ordering::AcqRel);
                });
            }

            if round % 2 == 0 {
                while counter.load(Ordering::Acquire) != tasks {
                    thread::yield_now();
                }
            }
        });

        assert_eq!(counter.load(Ordering::Acquire), tasks);
    }
}

#[test]
fn blocked_tasks_complete_together() {
    let pool = ThreadPool::new(8);
    let barrier = Arc::new(Barrier::new(8));
    let counter = AtomicUsize::new(0);

    pool.scope(|scope| {
        for _ in 0..8 {
            let barrier = barrier.clone();
            let counter = &counter;
            scope.spawn(move |_| {
                barrier.wait();
                counter.fetch_add(1, Ordering::AcqRel);
            });
        }
    });

    assert_eq!(counter.load(Ordering::Acquire), 8);
}

#[test]
fn nested_spawn_runs_to_completion() {
    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);

    pool.scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|scope| {
                counter.fetch_add(1, Ordering::AcqRel);
                for _ in 0..8 {
                    scope.spawn(|_| {
                        counter.fetch_add(1, Ordering::AcqRel);
                    });
                }
            });
        }
    });

    assert_eq!(counter.load(Ordering::Acquire), 72);
}

#[test]
fn nested_spawn_can_borrow_scope_data() {
    let pool = ThreadPool::new(4);
    let values = [2usize, 4, 8, 16];
    let sum = AtomicUsize::new(0);

    pool.scope(|scope| {
        for value in &values {
            let sum = &sum;
            scope.spawn(move |scope| {
                scope.spawn(move |_| {
                    sum.fetch_add(*value, Ordering::AcqRel);
                });
            });
        }
    });

    assert_eq!(sum.load(Ordering::Acquire), values.iter().sum());
}

#[test]
fn nested_work_spawned_after_root_callback_returns_is_waited_for() {
    let pool = ThreadPool::new(2);
    let root_returned = Arc::new(Notification::new());
    let release_nested_spawn = Arc::new(Notification::new());
    let nested_ran = Arc::new(AtomicBool::new(false));
    let scope_returned = Arc::new(AtomicBool::new(false));

    thread::scope(|thread_scope| {
        let root_returned_inner = root_returned.clone();
        let release_nested_spawn_inner = release_nested_spawn.clone();
        let nested_ran_inner = nested_ran.clone();
        let scope_returned_inner = scope_returned.clone();
        let pool = &pool;

        thread_scope.spawn(move || {
            pool.scope(|scope| {
                scope.spawn(move |scope| {
                    root_returned_inner.wait();
                    release_nested_spawn_inner.wait();
                    scope.spawn(move |_| {
                        nested_ran_inner.store(true, Ordering::Release);
                    });
                });
            });
            scope_returned_inner.store(true, Ordering::Release);
        });

        root_returned.notify();
        thread::sleep(Duration::from_millis(10));
        assert!(!scope_returned.load(Ordering::Acquire));
        release_nested_spawn.notify();
    });

    assert!(nested_ran.load(Ordering::Acquire));
    assert!(scope_returned.load(Ordering::Acquire));
}

#[test]
fn scope_root_callback_can_borrow_shorter_scheduling_data() {
    fn increment<'slice, 'counter>(pool: &ThreadPool, counters: &'slice [&'counter AtomicUsize]) {
        pool.scope(move |scope: &Scope<'counter>| {
            for &counter in counters {
                scope.spawn(move |_| {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
    }

    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);
    increment(&pool, &[&counter; 64]);

    assert_eq!(counter.load(Ordering::Relaxed), 64);
}

#[test]
fn static_scope_still_allows_local_scheduling_iterator() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let pool = ThreadPool::new(4);
    let mut range = 0..32;
    let expected = range.clone().sum();
    let iter = &mut range;

    COUNTER.store(0, Ordering::Relaxed);
    pool.scope(|scope: &Scope<'static>| {
        for value in iter {
            scope.spawn(move |_| {
                COUNTER.fetch_add(value, Ordering::Relaxed);
            });
        }
    });

    assert_eq!(COUNTER.load(Ordering::Relaxed), expected);
}

#[test]
fn worker_panic_is_propagated_after_other_tasks_finish() {
    let pool = ThreadPool::new(2);
    let completed = Arc::new(AtomicUsize::new(0));

    let result = {
        let completed = completed.clone();
        panic::catch_unwind(panic::AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.spawn(|_| panic!("worker panic"));
                scope.spawn(move |_| {
                    completed.fetch_add(1, Ordering::Release);
                });
            });
        }))
    };

    assert!(result.is_err());
    assert_eq!(completed.load(Ordering::Acquire), 1);
}

#[test]
fn outer_panic_still_waits_for_spawned_work() {
    let pool = ThreadPool::new(1);
    let completed = AtomicBool::new(false);

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        pool.scope(|scope| {
            scope.spawn(|_| {
                completed.store(true, Ordering::Release);
            });
            panic!("outer panic");
        });
    }));

    assert!(result.is_err());
    assert!(completed.load(Ordering::Acquire));
}

#[test]
fn parallel_for_each_processes_borrowed_items() {
    let pool = ThreadPool::new(4);
    let values = vec![1usize, 2, 3, 4, 5, 6];
    let seen = Mutex::new(Vec::new());

    pool.parallel_for_each(values.iter(), |value| {
        seen.lock().unwrap().push(*value);
    });

    let mut seen = seen.into_inner().unwrap();
    seen.sort_unstable();
    assert_eq!(seen, values);
}

#[test]
fn parallel_for_each_waits_for_all_work() {
    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);

    pool.parallel_for_each(0..256, |_| {
        counter.fetch_add(1, Ordering::AcqRel);
    });

    assert_eq!(counter.load(Ordering::Acquire), 256);
}

#[test]
fn dropping_pool_closes_workers() {
    let pool = ThreadPool::new(4);
    assert_eq!(pool.thread_count(), 4);
    drop(pool);
}

#[test]
fn current_num_threads_is_one_without_installed_pool() {
    assert_eq!(current_num_threads(), 1);
}

#[test]
fn install_sets_and_restores_current_pool() {
    let outer = ThreadPool::new(2);
    let inner = ThreadPool::new(3);

    assert_eq!(current_num_threads(), 1);
    outer.install(|| {
        assert_eq!(current_num_threads(), 2);
        inner.install(|| {
            assert_eq!(current_num_threads(), 3);
        });
        assert_eq!(current_num_threads(), 2);
    });
    assert_eq!(current_num_threads(), 1);
}

#[test]
fn pool_scope_installs_pool_for_root_and_spawned_callbacks() {
    let pool = ThreadPool::new(4);

    pool.scope(|scope| {
        assert_eq!(current_num_threads(), 4);
        scope.spawn(|_| {
            assert_eq!(current_num_threads(), 4);
        });
    });
}

#[test]
fn free_scope_uses_installed_pool() {
    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);

    pool.install(|| {
        threadpool_scope(|scope| {
            assert_eq!(current_num_threads(), 4);
            scope.spawn(|_| {
                assert_eq!(current_num_threads(), 4);
                counter.fetch_add(1, Ordering::Relaxed);
            });
        });
    });

    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[test]
fn free_scope_works_from_spawned_callback() {
    let pool = ThreadPool::new(4);
    let counter = AtomicUsize::new(0);

    pool.scope(|scope| {
        scope.spawn(|_| {
            threadpool_scope(|scope| {
                scope.spawn(|_| {
                    assert_eq!(current_num_threads(), 4);
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            });
        });
    });

    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[test]
#[should_panic(expected = "no egglog thread pool is currently installed")]
fn free_scope_panics_without_installed_pool() {
    threadpool_scope(|_| {});
}
