use super::*;

#[test]
fn arena_handle_is_created_only_on_first_packed_allocation() {
    let arena = SharedArena::new();
    let handle = LazyArenaHandle::new(&arena);
    assert!(handle.handle.get().is_none());
    handle.get();
    assert!(handle.handle.get().is_some());
}

use crate::numeric_id::NumericId;
use std::cell::Cell;

#[test]
fn sorted_multi_probe_keeps_a_monotone_cursor_across_batches() {
    let target = [1, 2, 4, 8, 9, 20, 21].map(Value::from_usize);
    let first = [0, 2, 3, 8].map(Value::from_usize);
    let second = [9, 10, 20, 22].map(Value::from_usize);
    let mut cursor = 0;
    let mut matches = Vec::new();

    for (input, key) in first.into_iter().enumerate() {
        if seek_sorted_key(key, target.len(), &mut cursor, |index| target[index]) {
            matches.push((input, cursor));
            cursor += 1;
        }
    }
    assert_eq!(matches, vec![(1, 1), (3, 3)]);
    assert_eq!(cursor, 4);

    matches.clear();
    for (input, key) in second.into_iter().enumerate() {
        if seek_sorted_key(key, target.len(), &mut cursor, |index| target[index]) {
            matches.push((input, cursor));
            cursor += 1;
        }
    }
    assert_eq!(matches, vec![(0, 4), (2, 5)]);
    assert_eq!(cursor, target.len());
}

#[test]
fn sorted_multi_probe_gallops_across_skewed_gaps() {
    let target_len = 100_000;
    let keys = [2, 199_998].map(Value::from_usize);
    let calls = Cell::new(0);
    let mut cursor = 0;
    let mut matches = Vec::new();

    for (input, key) in keys.into_iter().enumerate() {
        if seek_sorted_key(key, target_len, &mut cursor, |index| {
            calls.set(calls.get() + 1);
            Value::from_usize(index * 2)
        }) {
            matches.push((input, cursor));
            cursor += 1;
        }
    }

    assert_eq!(matches, vec![(0, 1), (1, 99_999)]);
    assert!(
        calls.get() < 100,
        "galloping should not linearly scan a large key gap"
    );
}

#[test]
fn sorted_seek_reads_an_immediate_hit_once() {
    let target = [2, 4, 6].map(Value::from_usize);
    let calls = Cell::new(0);
    let mut cursor = 0;
    assert!(seek_sorted_key(
        Value::from_usize(2),
        target.len(),
        &mut cursor,
        |index| {
            calls.set(calls.get() + 1);
            target[index]
        }
    ));
    assert_eq!(calls.get(), 1);
}
