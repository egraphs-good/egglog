use std::{
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, sleep},
    time::Duration,
};

use crate::{ConcurrentVec, Notification, ParallelVecWriter, ReadOptimizedLock};

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
                assert!(dst % 10 == 0);
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
