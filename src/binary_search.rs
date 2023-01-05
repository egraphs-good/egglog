use std::cmp::Ordering;

use crate::table::Table;

/// Binary search a [`Table`] for the smallest index with a timestamp greater
/// than or equal to `target`.
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
