use std::{
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use concurrency::Notification;
use numeric_id::{define_id, NumericId};

use crate::concurrent::UnionFind;

use super::buffer::Buffer;

#[derive(Clone)]
struct Dropper<T> {
    item: T,
    dropped: Arc<AtomicUsize>,
}

impl<T> Drop for Dropper<T> {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn buffer_single_threaded() {
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let dropped = drop_counter.clone();
    let init = move |i| Dropper {
        item: i,
        dropped: dropped.clone(),
    };
    let buffer = Buffer::new(4, init.clone());
    // No resizing should occur.
    buffer.with_access(
        4,
        |arr| {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0].item, 0);
            assert_eq!(arr[1].item, 1);
            assert_eq!(arr[2].item, 2);
            assert_eq!(arr[3].item, 3);
        },
        init.clone(),
    );
    assert_eq!(drop_counter.load(Ordering::SeqCst), 0);
    buffer.with_access(
        10,
        |arr| {
            assert!(arr.len() >= 10);
            assert!(arr.iter().enumerate().all(|(i, v)| { v.item == i }));
        },
        init.clone(),
    );
    assert_eq!(drop_counter.load(Ordering::SeqCst), 0);
    mem::drop(buffer);
    assert_eq!(
        drop_counter.load(Ordering::SeqCst),
        // We round up to amortize calls to `with_access`.
        10usize.next_power_of_two()
    );
}

#[test]
fn buffer_multi_threaded() {
    let init = move |i| i;
    let buffer = Arc::new(Buffer::new(1, init));

    let threads = (0..100)
        .map(|x| {
            let buffer = buffer.clone();
            std::thread::spawn(move || {
                let len = x * 1000;
                buffer.with_access(
                    len,
                    |arr| {
                        assert!(arr.len() >= len);
                        assert!(arr.iter().enumerate().all(|(i, v)| { *v == i }));
                    },
                    init,
                );
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
}

define_id!(pub(crate) Value, u32, "a value for testing the UF");

fn v(u: usize) -> Value {
    Value::from_usize(u)
}

#[test]
fn uf_single_theaded() {
    let uf = UnionFind::<Value>::default();
    let ids1 = (0..100).map(v).collect::<Vec<_>>();
    let ids2 = (100..200).map(v).collect::<Vec<_>>();

    for ids in [&ids1, &ids2] {
        ids.windows(2).for_each(|w| {
            let (parent, child) = uf.union(w[0], w[1]);
            assert_eq!(uf.find(w[0]), parent);
            assert_eq!(uf.find(w[1]), parent);
            assert!(child == w[0] || child == w[1]);
        });
    }

    assert!(ids1
        .windows(2)
        .all(|w| uf.find(w[0]) == uf.find(w[1]) && uf.same_set(w[0], w[1])));
    assert!(ids2.windows(2).all(|w| uf.find(w[0]) == uf.find(w[1])));
    assert_ne!(uf.find(ids1[0]), uf.find(ids2[0]));

    uf.union(ids1[5], ids2[20]);

    let target = uf.find(ids1[0]);
    assert!(ids1
        .iter()
        .chain(ids2.iter())
        .all(|&id| uf.find(id) == target));
}

#[test]
fn uf_multi_threaded() {
    let uf = UnionFind::<Value>::default();
    let n1 = Arc::new(Notification::new());
    let ids1 = (0..100).map(v).collect::<Vec<_>>();
    let ids2 = (100..200).map(v).collect::<Vec<_>>();
    let threads_1 = (0..100)
        .map(|_| {
            let n = n1.clone();
            let uf = uf.clone();
            let ids1 = ids1.clone();
            let ids2 = ids2.clone();
            thread::spawn(move || {
                n.wait();
                for ids in [&ids1, &ids2] {
                    ids.windows(2).for_each(|w| {
                        let (parent, child) = uf.union(w[0], w[1]);
                        assert!(uf.same_set(parent, child));
                        assert!(uf.same_set(w[0], child));
                        assert!(uf.same_set(w[0], parent));
                        assert!(uf.same_set(w[1], child));
                        assert!(uf.same_set(w[1], parent));
                    });
                }
            })
        })
        .collect::<Vec<_>>();

    n1.notify();
    threads_1.into_iter().for_each(|t| t.join().unwrap());

    assert!(ids1
        .windows(2)
        .all(|w| uf.find(w[0]) == uf.find(w[1]) && uf.same_set(w[0], w[1])));
    assert!(ids2.windows(2).all(|w| uf.find(w[0]) == uf.find(w[1])));
    assert_ne!(uf.find(ids1[0]), uf.find(ids2[0]));
    let threads_2 = (0..100)
        .map(|tid| {
            let uf = uf.clone();
            let ids1 = ids1.clone();
            let ids2 = ids2.clone();
            thread::spawn(move || {
                uf.union(ids1[tid], ids2[tid]);
            })
        })
        .collect::<Vec<_>>();

    threads_2.into_iter().for_each(|t| t.join().unwrap());

    let target = uf.find(ids1[0]);
    assert!(ids1
        .iter()
        .chain(ids2.iter())
        .all(|&id| uf.find(id) == target));
}
