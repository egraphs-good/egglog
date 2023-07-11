use std::cmp::Ordering;

use super::table::Table;

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
    let mut size = data.num_offsets();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    fn make_value(bits: u32) -> Value {
        Value {
            tag: "testing".into(),
            bits: bits as u64,
        }
    }

    fn insert_to_map(table: &mut Table, i: u32, ts: u32) {
        let v = make_value(i);
        table.insert(&[v], v, ts);
    }

    #[test]
    fn binary_search() {
        let mut map = Table::default();
        assert_eq!(binary_search_table_by_key(&map, 0), None);
        insert_to_map(&mut map, 1, 1);
        assert_eq!(binary_search_table_by_key(&map, 0), Some(0));
        map.clear();
        for i in 0..128 {
            // have a run of 4 24s and then skip to 26
            let v = if i == 50 || i == 51 { 24 } else { i / 2 };
            insert_to_map(&mut map, i, v);
        }

        assert_eq!(binary_search_table_by_key(&map, 3), Some(6));
        assert_eq!(binary_search_table_by_key(&map, 0), Some(0));
        assert_eq!(binary_search_table_by_key(&map, 63), Some(126));
        assert_eq!(binary_search_table_by_key(&map, 200), None);
        assert_eq!(binary_search_table_by_key(&map, 24), Some(48));
        assert_eq!(binary_search_table_by_key(&map, 25), Some(52));
    }
}
