use crate::numeric_id::NumericId;

use crate::{OffsetRange, Subset, common::HashSet, pool::with_pool_set};

use super::{Offsets, RowId, SortedOffsetVector, SubsetRef};

fn o(u: usize) -> RowId {
    RowId::from_usize(u)
}

fn collect<T: Clone>(range: &impl Offsets, elts: &[T]) -> Vec<T> {
    let mut res = Vec::new();
    range.offsets(|off| res.push(elts[off.index()].clone()));
    if !res.is_empty() {
        range.bounds().expect("nonempty range should have bounds");
    }
    res
}

#[test]
fn subset_push() {
    let elts = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut s = Subset::Dense(OffsetRange::new(o(1), o(2)));
    assert_eq!(collect(&s, &elts), vec![1]);
    s.add_row_sorted(o(2));
    s.add_row_sorted(o(3));
    assert_eq!(collect(&s, &elts), vec![1, 2, 3]);
    s.add_row_sorted(o(7));
    assert_eq!(collect(&s, &elts), vec![1, 2, 3, 7]);
}

#[test]
fn subset_clone_reuses_pooled_vector_allocation() {
    use super::SortedOffsetVector;
    use crate::pool::Pooled;

    let pool = with_pool_set(|ps| ps.get_pool::<SortedOffsetVector>());
    pool.clear();
    let source = Subset::Sparse(Pooled::new(SortedOffsetVector(vec![o(1), o(3), o(7)])));

    let mut spare = pool.get();
    spare.fill_from_dense(&OffsetRange::new(o(0), o(64)));
    let pointer = spare.0.as_ptr();
    let capacity = spare.0.capacity();
    drop(spare);

    let cloned = source.clone();
    let Subset::Sparse(rows) = &cloned else {
        panic!("cloning a sparse subset changed its representation");
    };
    assert_eq!(rows.slice().inner(), &[o(1), o(3), o(7)]);
    assert_eq!(rows.0.capacity(), capacity);
    assert_eq!(rows.0.as_ptr(), pointer);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic]
fn bad_offset_range() {
    OffsetRange::new(o(3), o(2));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic]
fn bad_offset_stride() {
    OffsetRange::new(o(3), o(2));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic]
fn not_sorted_vec_push() {
    use crate::offsets::SortedOffsetVector;
    let mut v = SortedOffsetVector::default();
    v.push(o(3));
    v.push(o(1));
    v.push(o(2));
}

#[test]
fn intersect() {
    let elts = [
        vec![1, 3, 4, 8, 9, 12, 20],
        vec![3, 4, 8, 9, 11, 12, 15, 18, 20],
        vec![3, 4, 8, 9, 11, 12, 15, 18, 20, 22, 24],
        vec![1, 4, 8, 9, 11, 12, 15, 18, 20],
        vec![],
        Vec::from_iter(0..20),
        Vec::from_iter(2..100),
        Vec::from_iter(4..50),
    ];

    // Also include size-skewed pairs that exercise the two-pointer/galloping boundary.
    // Two-pointer fires when max/min < 4; galloping fires otherwise.
    let skewed: Vec<Vec<usize>> = vec![
        Vec::from_iter(0..5),
        Vec::from_iter(0..100), // 20x skew → galloping path
        Vec::from_iter((0..100).filter(|x| x % 7 == 0)),
        // Both sides Sparse (i.e. non-contiguous) with >4x skew: galloping in
        // `cur` while compacting it in place.
        Vec::from_iter((0..6).chain(11..22)),
        vec![13, 20, 21],
    ];
    let all_elts: Vec<&Vec<usize>> = elts.iter().chain(skewed.iter()).collect();

    for l in &all_elts {
        for r in &all_elts {
            let mut l_sub = Subset::empty();
            let mut r_sub = Subset::empty();
            let l_set = HashSet::from_iter(l.iter().copied().map(o));
            let r_set = HashSet::from_iter(r.iter().copied().map(o));
            for row in l.iter() {
                l_sub.add_row_sorted(o(*row));
            }
            for row in r.iter() {
                r_sub.add_row_sorted(o(*row));
            }
            let mut expected = Vec::from_iter(l_set.intersection(&r_set).copied());
            l_sub.intersect(
                r_sub.as_ref(),
                &with_pool_set(|pool_set| pool_set.get_pool()),
            );
            expected.sort();
            let mut got = Vec::new();
            l_sub.offsets(|row| got.push(row));
            assert_eq!(expected, got, "l: {l:?}, r: {r:?}");
        }
    }
}

#[test]
fn iter_bounded() {
    let mut s1 = Subset::empty();
    assert!(
        s1.as_ref()
            .iter_bounded(0, 100, |_| panic!("this should never be called"))
            .is_none()
    );
    let mut s2 = Subset::empty();
    for i in 0..100 {
        s1.add_row_sorted(RowId::new(i));
        s2.add_row_sorted(RowId::new(i * 2));
    }
    assert!(matches!(s1, Subset::Dense(..)));
    assert!(matches!(s2, Subset::Sparse(..)));

    let mut got = Vec::new();
    assert_eq!(
        Some(12),
        s1.as_ref().iter_bounded(2, 12, |row| got.push(row))
    );
    let expected = Vec::from_iter((2..12).map(RowId::new));
    assert_eq!(got, expected);

    got.clear();
    assert_eq!(
        Some(12),
        s2.as_ref().iter_bounded(2, 12, |row| got.push(row))
    );
    let expected = Vec::from_iter((2..12).map(|x| RowId::new(x * 2)));
    assert_eq!(got, expected);
}

#[test]
fn intersect_sparse_sparse_skewed() {
    // Regression test for #971. `cur` must be Sparse and >4x longer than
    // `other` (which must also be Sparse) to reach the "gallop in cur" branch
    // of `Subset::intersect`.
    let cur_rows: Vec<usize> = (0..6).chain(11..22).collect();
    let other_rows: Vec<usize> = vec![13, 20, 21];

    let mut cur = Subset::empty();
    for row in &cur_rows {
        cur.add_row_sorted(o(*row));
    }
    let mut other = Subset::empty();
    for row in &other_rows {
        other.add_row_sorted(o(*row));
    }
    assert!(matches!(cur, Subset::Sparse(..)));
    assert!(matches!(other, Subset::Sparse(..)));

    cur.intersect(
        other.as_ref(),
        &with_pool_set(|pool_set| pool_set.get_pool()),
    );
    let mut got = Vec::new();
    cur.offsets(|row| got.push(row.index()));
    assert_eq!(got, other_rows);
}

fn sparse_rows(rows: &[usize]) -> Subset {
    let mut offsets = with_pool_set(|ps| ps.get::<SortedOffsetVector>());
    for row in rows {
        offsets.push(o(*row));
    }
    Subset::Sparse(offsets)
}

fn row_ids(subset: &Subset) -> Vec<usize> {
    let mut rows = Vec::new();
    subset.offsets(|row| rows.push(row.index()));
    rows
}

#[test]
fn empty_sparse_results_stay_dense() {
    let pool = with_pool_set(|ps| ps.get_pool());
    let empty = sparse_rows(&[]);
    assert!(empty.as_ref().to_owned(&pool).is_dense());

    let mut retained = sparse_rows(&[1, 3]);
    retained.retain(|_| false);
    assert!(retained.is_dense());
    assert_eq!(retained.size(), 0);

    for left in [vec![], vec![1, 3], vec![7, 9]] {
        for right in [vec![], vec![4, 6]] {
            let mut left = sparse_rows(&left);
            left.intersect(sparse_rows(&right).as_ref(), &pool);
            assert!(left.is_dense());
            assert_eq!(left.size(), 0);
        }
    }
}

#[test]
fn singleton_sparse_results_stay_dense_under_set_intersections() {
    let pool = with_pool_set(|ps| ps.get_pool());
    let mut retained = sparse_rows(&[3, 5]);
    retained.retain(|row| row == o(3));
    let mut intersected = sparse_rows(&[1, 3]);
    intersected.intersect(SubsetRef::Dense(OffsetRange::new(o(3), o(4))), &pool);
    let mut intersected_sparse = sparse_rows(&[1, 3]);
    intersected_sparse.intersect(sparse_rows(&[3, 5]).as_ref(), &pool);
    let mut intersected_dense = Subset::Dense(OffsetRange::new(o(3), o(4)));
    intersected_dense.intersect(sparse_rows(&[1, 3, 5]).as_ref(), &pool);
    let owned = sparse_rows(&[3]).as_ref().to_owned(&pool);
    for singleton in [
        retained,
        intersected,
        intersected_sparse,
        intersected_dense,
        owned,
    ] {
        assert!(singleton.is_dense());
        assert_eq!(row_ids(&singleton), vec![3]);
        let mut containing = sparse_rows(&[1, 3, 5]);
        containing.intersect(singleton.as_ref(), &pool);
        assert!(containing.is_dense());
        assert_eq!(row_ids(&containing), vec![3]);
        let mut disjoint = sparse_rows(&[2, 4]);
        disjoint.intersect(singleton.as_ref(), &pool);
        assert!(disjoint.is_dense());
        assert_eq!(disjoint.size(), 0);
    }
}

#[test]
fn intersect_small_multisets_and_dense_ranges() {
    let pool = with_pool_set(|ps| ps.get_pool());
    // Enumerate every sorted multiset containing up to two of each row 1..=4.
    let multisets: Vec<Vec<usize>> = (0..3usize.pow(4))
        .map(|mut code| {
            let mut rows = Vec::new();
            for row in 1..=4 {
                rows.extend(std::iter::repeat_n(row, code % 3));
                code /= 3;
            }
            rows
        })
        .collect();

    for left in &multisets {
        for right in &multisets {
            let mut actual = sparse_rows(left);
            actual.intersect(sparse_rows(right).as_ref(), &pool);
            let expected: Vec<usize> = (1..=4)
                .flat_map(|row| {
                    let left_count = left.iter().filter(|r| **r == row).count();
                    let right_count = right.iter().filter(|r| **r == row).count();
                    std::iter::repeat_n(row, left_count.min(right_count))
                })
                .collect();
            assert_eq!(row_ids(&actual), expected, "{left:?} intersect {right:?}");
        }
        for low in 0..=6 {
            for high in low..=6 {
                let range = OffsetRange::new(o(low), o(high));
                let expected: Vec<usize> = left
                    .iter()
                    .copied()
                    .filter(|row| (low..high).contains(row))
                    .collect();
                let mut sparse = sparse_rows(left);
                sparse.intersect(SubsetRef::Dense(range), &pool);
                let mut dense = Subset::Dense(range);
                dense.intersect(sparse_rows(left).as_ref(), &pool);
                assert_eq!(row_ids(&sparse), expected, "{left:?} intersect {range:?}");
                assert_eq!(row_ids(&dense), expected, "{range:?} intersect {left:?}");
                if expected.len() <= 1 {
                    assert!(sparse.is_dense());
                    assert!(dense.is_dense());
                }
            }
        }
    }
}

#[test]
fn intersect_skewed_sparse_duplicates_and_maximum_row_id() {
    let pool = with_pool_set(|ps| ps.get_pool());
    let large: Vec<usize> = (0..128)
        .flat_map(|row| std::iter::repeat_n(row * 2, 2))
        .collect();
    let small = vec![4, 4, 4, 64, 254, 255];
    for (left, right) in [(&large, &small), (&small, &large)] {
        let mut actual = sparse_rows(left);
        actual.intersect(sparse_rows(right).as_ref(), &pool);
        assert_eq!(row_ids(&actual), vec![4, 4, 64, 254]);
    }

    // Endpoint checks must not compute an overflowing exclusive upper bound.
    let max = u32::MAX as usize;
    let mut dense = Subset::Dense(OffsetRange::new(o(0), o(max)));
    dense.intersect(sparse_rows(&[max]).as_ref(), &pool);
    assert_eq!(dense.size(), 0);
    let mut sparse = sparse_rows(&[max]);
    sparse.intersect(sparse_rows(&[max]).as_ref(), &pool);
    assert_eq!(row_ids(&sparse), vec![max]);
    assert!(!sparse.is_dense(), "MAX has no dense exclusive endpoint");
    let owned = sparse.as_ref().to_owned(&pool);
    assert_eq!(row_ids(&owned), vec![max]);
    assert!(!owned.is_dense());
}
