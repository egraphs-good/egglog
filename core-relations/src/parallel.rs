//! Small parallel helpers backed by the installed egglog thread pool.

use std::sync::OnceLock;

use crate::numeric_id::{DenseIdMap, IdVec, NumericId};

const DEFAULT_TASKS_PER_THREAD: usize = 1;
const TASKS_PER_THREAD_ENV: &str = "EGGLOG_PARALLEL_TASKS_PER_THREAD";

static TASKS_PER_THREAD: OnceLock<usize> = OnceLock::new();

pub(crate) fn current_num_threads() -> usize {
    egglog_concurrency::current_num_threads()
}

pub(crate) fn enabled_for_len(len: usize) -> bool {
    len > 1 && current_num_threads() > 1
}

fn chunk_len(len: usize) -> usize {
    let tasks = current_num_threads()
        .saturating_mul(tasks_per_thread())
        .max(1);
    len.div_ceil(tasks).max(1)
}

fn tasks_per_thread() -> usize {
    *TASKS_PER_THREAD.get_or_init(|| {
        read_usize_env(TASKS_PER_THREAD_ENV)
            .unwrap_or(DEFAULT_TASKS_PER_THREAD)
            .max(1)
    })
}

fn read_usize_env(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn default_vec<T: Default>(len: usize) -> Vec<T> {
    vec_with(len, T::default)
}

fn vec_with<T>(len: usize, init: impl FnMut() -> T) -> Vec<T> {
    let mut values = Vec::with_capacity(len);
    values.resize_with(len, init);
    values
}

pub(crate) fn for_each_mut<T, F>(items: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync,
{
    if !enabled_for_len(items.len()) {
        for (index, item) in items.iter_mut().enumerate() {
            f(index, item);
        }
        return;
    }

    let chunk_len = chunk_len(items.len());
    egglog_concurrency::scope(|scope| {
        let f = &f;
        for (chunk_index, chunk) in items.chunks_mut(chunk_len).enumerate() {
            let base = chunk_index * chunk_len;
            scope.spawn(move |_| {
                for (offset, item) in chunk.iter_mut().enumerate() {
                    f(base + offset, item);
                }
            });
        }
    });
}

pub(crate) fn map_mut<T, R, F>(items: &mut [T], f: F) -> Vec<R>
where
    T: Send,
    R: Default + Send,
    F: Fn(usize, &mut T) -> R + Sync,
{
    map_mut_with(items, R::default, f)
}

pub(crate) fn map_mut_with<T, R, Init, F>(items: &mut [T], init: Init, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    Init: FnMut() -> R,
    F: Fn(usize, &mut T) -> R + Sync,
{
    if !enabled_for_len(items.len()) {
        return items
            .iter_mut()
            .enumerate()
            .map(|(index, item)| f(index, item))
            .collect();
    }

    let chunk_len = chunk_len(items.len());
    let mut results = vec_with(items.len(), init);
    egglog_concurrency::scope(|scope| {
        let f = &f;
        for (chunk_index, (chunk, out)) in items
            .chunks_mut(chunk_len)
            .zip(results.chunks_mut(chunk_len))
            .enumerate()
        {
            let base = chunk_index * chunk_len;
            scope.spawn(move |_| {
                for (offset, item) in chunk.iter_mut().enumerate() {
                    out[offset] = f(base + offset, item);
                }
            });
        }
    });

    results
}

pub(crate) fn map_dense_id_map_mut<K, V, R, F>(map: &mut DenseIdMap<K, V>, f: F) -> Vec<R>
where
    K: NumericId,
    V: Send,
    R: Send,
    F: Fn(K, &mut V) -> R + Sync,
{
    map_mut(map.raw_mut(), |index, slot| {
        slot.as_mut().map(|value| f(K::from_usize(index), value))
    })
    .into_iter()
    .flatten()
    .collect()
}

pub(crate) fn for_each_id_vec_mut<K, V, F>(map: &mut IdVec<K, V>, f: F)
where
    K: NumericId,
    V: Send,
    F: Fn(K, &mut V) + Sync,
{
    for_each_mut(map.as_mut_slice(), |index, value| {
        f(K::from_usize(index), value);
    });
}

pub(crate) fn map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Default + Send,
    F: Fn(usize, &T) -> R + Sync,
{
    if !enabled_for_len(items.len()) {
        return items
            .iter()
            .enumerate()
            .map(|(index, item)| f(index, item))
            .collect();
    }

    let chunk_len = chunk_len(items.len());
    let mut results = default_vec(items.len());
    egglog_concurrency::scope(|scope| {
        let f = &f;
        for (chunk_index, (chunk, out)) in items
            .chunks(chunk_len)
            .zip(results.chunks_mut(chunk_len))
            .enumerate()
        {
            let base = chunk_index * chunk_len;
            scope.spawn(move |_| {
                for (offset, item) in chunk.iter().enumerate() {
                    out[offset] = f(base + offset, item);
                }
            });
        }
    });

    results
}
