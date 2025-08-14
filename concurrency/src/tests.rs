use std::{
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, sleep},
    time::Duration,
};

use crate::{
    ConcurrentVec, Notification, ParallelVecWriter, ReadOptimizedLock, ResettableOnceLock,
};

#[test]
fn notification_single_threaded() {
    let n = Notification::default();
    n.notify();
    n.wait();
}

#[test]
fn notification_wakes_up_multiple() {
    let n = Arc::new(Notification::default());
    let ctr = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..20)
        .map(|_| {
            let n = n.clone();
            let ctr = ctr.clone();
            std::thread::spawn(move || {
                n.wait();
                ctr.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert_eq!(ctr.load(Ordering::SeqCst), 0);
    n.notify();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(ctr.load(Ordering::SeqCst), 20);
}

#[test]
fn notification_times_out() {
    let n = Arc::new(Notification::default());
    let threads: Vec<_> = (0..20)
        .map(|_| {
            let n = n.clone();
            std::thread::spawn(move || assert!(!n.wait_with_timeout(Duration::from_millis(10))))
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn notification_race() {
    let n = Arc::new(Notification::default());
    let ctr = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..20)
        .map(|i| {
            let n = n.clone();
            let ctr = ctr.clone();
            std::thread::spawn(move || {
                if i == 19 {
                    n.notify();
                } else {
                    n.wait();
                }
                ctr.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(ctr.load(Ordering::SeqCst), 20);
}

// We get more test coverage in the union-find crate

#[test]
fn simple_mutex() {
    for _ in 0..50 {
        let m = Arc::new(ReadOptimizedLock::new(0));
        let read_guard_1 = m.read();
        let read_guard_2 = m.read();
        assert_eq!(*read_guard_1, 0);
        assert_eq!(*read_guard_2, 0);
        let locked = Arc::new(Notification::new());
        let locked_inner = locked.clone();
        let m_inner = m.clone();
        let writer = thread::spawn(move || {
            let mut lock = m_inner.lock();
            locked_inner.notify();
            *lock = 5;
        });
        sleep(Duration::from_millis(1));
        assert!(!locked.has_been_notified());
        mem::drop(read_guard_1);
        sleep(Duration::from_millis(1));
        assert!(!locked.has_been_notified());
        mem::drop(read_guard_2);
        locked.wait();
        writer.join().unwrap();
        assert_eq!(*m.read(), 5);
    }
}

#[test]
fn basic_parallel_vec_push() {
    const N_THREADS: usize = 10;
    const PER_THREAD: usize = 10;
    let v = Arc::new(ConcurrentVec::<usize>::with_capacity(0));
    let threads: Vec<_> = (0..N_THREADS)
        .map(|i| {
            let v = v.clone();
            thread::spawn(move || {
                let mut got = Vec::new();
                for j in 0..PER_THREAD {
                    got.push(v.push(i * PER_THREAD + j));
                }
                got
            })
        })
        .collect();
    let mut results = threads
        .into_iter()
        .flat_map(|x| x.join().unwrap())
        .collect::<Vec<usize>>();
    results.sort();
    assert_eq!(results.len(), N_THREADS * PER_THREAD);
    assert_eq!(
        results,
        (0..(N_THREADS * PER_THREAD)).collect::<Vec<usize>>()
    );
    let slice = v.read();
    assert_eq!(slice.len(), N_THREADS * PER_THREAD);
    let mut sorted = slice.to_vec();
    sorted.sort();
    assert_eq!(
        results,
        (0..(N_THREADS * PER_THREAD)).collect::<Vec<usize>>()
    );
}

#[test]
fn basic_parallel_vec_write() {
    const N_THREADS: usize = 10;
    const PER_THREAD: usize = 10;
    let finish = Arc::new(Notification::new());
    let v = (0..100).collect::<Vec<usize>>();
    let v = Arc::new(ParallelVecWriter::new(v));
    let threads: Vec<_> = (0..N_THREADS)
        .map(|i| {
            let finish = finish.clone();
            let v = v.clone();
            thread::spawn(move || {
                let dst = v.write_contents((0..PER_THREAD).map(|j| i * PER_THREAD + j + 100));
                assert!(dst.is_multiple_of(10));
                finish.wait();
            })
        })
        .collect();
    thread::sleep(Duration::from_millis(100));
    for i in 0..100 {
        v.with_index(i, |x| assert_eq!(*x, i));
    }
    v.with_slice(0..100, |x| assert_eq!(x, (0..100).collect::<Vec<usize>>()));
    finish.notify();
    threads.into_iter().for_each(|x| x.join().unwrap());
    let mut v = Arc::try_unwrap(v).ok().unwrap().finish();
    v.sort();
    assert_eq!(v, (0..200).collect::<Vec<usize>>());
}

#[test]
fn resettable_once_lock_basic() {
    let lock = ResettableOnceLock::new(0);

    // Initially, get should return None
    assert!(lock.get().is_none());

    // First call to get_or_update should run the update function
    let value = lock.get_or_update(|x| *x = 42);
    assert_eq!(*value, 42);

    // Subsequent calls to get should return Some
    assert_eq!(*lock.get().unwrap(), 42);

    // get_or_update should just return the value without updating
    let value = lock.get_or_update(|x| *x = 100);
    assert_eq!(*value, 42); // Should not have been updated
}

#[test]
fn resettable_once_lock_reset() {
    let mut lock = ResettableOnceLock::new(0);

    // Update the value
    lock.get_or_update(|x| *x = 42);
    assert_eq!(*lock.get().unwrap(), 42);

    // Reset the lock
    lock.reset();

    // After reset, get should return None
    assert!(lock.get().is_none());

    // get_or_update should work again with a new update
    let value = lock.get_or_update(|x| *x = 100);
    assert_eq!(*value, 100);

    assert_eq!(*lock.get().unwrap(), 100);
}

#[test]
fn resettable_once_lock_concurrent_readers() {
    let lock = Arc::new(ResettableOnceLock::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(10));

    // Update the value first
    lock.get_or_update(|x| *x = 42);

    let threads: Vec<_> = (0..10)
        .map(|_| {
            let lock = lock.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                // All threads should be able to read concurrently
                let value = lock.get().unwrap();
                assert_eq!(*value, 42);
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn resettable_once_lock_concurrent_get_or_update() {
    let lock = Arc::new(ResettableOnceLock::new(0));
    let counter = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(10));

    let threads: Vec<_> = (0..10)
        .map(|_| {
            let lock = lock.clone();
            let counter = counter.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                // Only one thread should actually run the update function
                let value = lock.get_or_update(|x| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    *x = 42;
                });
                assert_eq!(*value, 42);
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    // The update function should have been called exactly once
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert_eq!(*lock.get().unwrap(), 42);
}

#[test]
fn resettable_once_lock_update_mutability() {
    #[derive(Debug, PartialEq)]
    struct TestStruct {
        value: i32,
        updated: bool,
    }

    let lock = ResettableOnceLock::new(TestStruct {
        value: 0,
        updated: false,
    });

    // Update multiple fields
    lock.get_or_update(|data| {
        data.value = 100;
        data.updated = true;
    });

    let result = lock.get().unwrap();
    assert_eq!(result.value, 100);
    assert!(result.updated);
}

#[test]
fn resettable_once_lock_multiple_resets() {
    let mut lock = ResettableOnceLock::new(0);

    for i in 1..=5 {
        // Each iteration: update, verify, reset
        lock.get_or_update(|x| *x = i * 10);
        assert_eq!(*lock.get().unwrap(), i * 10);
        lock.reset();
        assert!(lock.get().is_none());
    }
}

#[test]
fn resettable_once_lock_send_sync() {
    // Test that ResettableOnceLock can be sent between threads
    let lock = ResettableOnceLock::new(42);

    let handle = thread::spawn(move || {
        lock.get_or_update(|x| *x += 1);
        *lock.get().unwrap()
    });

    let result = handle.join().unwrap();
    assert_eq!(result, 43);
}
