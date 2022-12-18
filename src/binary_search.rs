use std::{cmp::Ordering, ops::Range};

use crate::{table::Table, util::IndexMap};

/// Binary search an IndexMap sorted by `SortKey`. Returns the index of the
/// smallest element greater than or equal to `target`, if there is one.
pub(crate) fn binary_search_by_key<K, V, SortKey, F>(
    data: &IndexMap<K, V>,
    f: F,
    target: &SortKey,
) -> Option<usize>
where
    SortKey: Ord,
    F: Fn(&V) -> &SortKey,
{
    if data.is_empty() {
        return None;
    }
    let (_, last) = data.last().unwrap();
    if f(last) < target {
        return None;
    }
    let (_, first) = data.first().unwrap();
    if f(first) > target {
        return Some(0);
    }
    // adapted from std::slice::binary_search_by
    let mut size = data.len();
    let mut left = 0;
    let mut right = size;
    while left < right {
        let mut mid = left + size / 2;
        let cmp = f(data.get_index(mid).unwrap().1).cmp(target);

        // The std implementation claims that if/else generates better code than match.
        if cmp == Ordering::Less {
            left = mid + 1;
        } else if cmp == Ordering::Greater {
            right = mid;
        } else {
            // We need to march back to the start of the matching elements. We
            // could have jumped into the middle of a run.
            //
            // TODO: this makes the algorithm O(n); we can use a variant of
            // gallop to get it back to log(n) if needed. See
            // https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md
            while mid > 0 {
                let next_mid = mid - 1;
                if f(data.get_index(next_mid).unwrap().1) != target {
                    break;
                }
                mid = next_mid;
            }
            return Some(mid);
        }
        size = right - left;
    }
    Some(left)
}

pub(crate) fn binary_search_table_by_key(data: &Table, target: u32) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    if data.max_ts() < target {
        return None;
    }
    if data.min_ts().unwrap() > target {
        return Some(0);
    }
    // adapted from std::slice::binary_search_by
    let mut size = data.len();
    let mut left = 0;
    let mut right = size;
    while left < right {
        let mut mid = left + size / 2;
        let cmp = data.get_timestamp(mid).unwrap().cmp(&target);

        // The std implementation claims that if/else generates better code than match.
        if cmp == Ordering::Less {
            left = mid + 1;
        } else if cmp == Ordering::Greater {
            right = mid;
        } else {
            // We need to march back to the start of the matching elements. We
            // could have jumped into the middle of a run.
            //
            // TODO: this makes the algorithm O(n); we can use a variant of
            // gallop to get it back to log(n) if needed. See
            // https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md
            while mid > 0 {
                let next_mid = mid - 1;
                if data.get_timestamp(next_mid).unwrap() != target {
                    break;
                }
                mid = next_mid;
            }
            return Some(mid);
        }
        size = right - left;
    }
    Some(left)
}

pub(crate) fn transform_range<K, V, SortKey, F>(
    data: &IndexMap<K, V>,
    f: F,
    range: &Range<SortKey>,
) -> Range<usize>
where
    SortKey: Ord,
    F: Fn(&V) -> &SortKey,
{
    if let Some(start) = binary_search_by_key(data, &f, &range.start) {
        if let Some(end) = binary_search_by_key(data, &f, &range.end) {
            start..end
        } else {
            start..data.len()
        }
    } else {
        0..0
    }
}
