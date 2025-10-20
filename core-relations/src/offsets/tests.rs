use crate::numeric_id::NumericId;

use crate::{OffsetRange, Subset, common::HashSet, pool::with_pool_set};

use super::{Offsets, RowId, SortedOffsetVector};

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
#[should_panic(expected = "attempting to create malformed range")]
fn bad_offset_range() {
    let _ = OffsetRange::new(o(3), o(2));
}

#[test]
#[should_panic(expected = "attempting to create malformed range")]
fn bad_offset_stride() {
    let _ = OffsetRange::new(o(3), o(2));
}

#[test]
#[should_panic(expected = "assertion failed")]
fn not_sorted_vec_push() {
    let mut v = SortedOffsetVector::default();
    v.push(o(3));
    v.push(o(1));
    v.push(o(2));
}

#[test]
fn intersect() {
    let elts = vec![
        vec![1, 3, 4, 8, 9, 12, 20],
        vec![3, 4, 8, 9, 11, 12, 15, 18, 20],
        vec![3, 4, 8, 9, 11, 12, 15, 18, 20, 22, 24],
        vec![1, 4, 8, 9, 11, 12, 15, 18, 20],
        vec![],
        (0..20).collect::<Vec<_>>(),
        (2..100).collect::<Vec<_>>(),
        (4..50).collect::<Vec<_>>(),
    ];

    for l in &elts {
        for r in &elts {
            let mut l_sub = Subset::empty();
            let mut r_sub = Subset::empty();
            let l_set = l.iter().copied().map(o).collect::<HashSet<_>>();
            let r_set = r.iter().copied().map(o).collect::<HashSet<_>>();
            for row in l {
                l_sub.add_row_sorted(o(*row));
            }
            for row in r {
                r_sub.add_row_sorted(o(*row));
            }
            let mut expected = l_set.intersection(&r_set).copied().collect::<Vec<_>>();
            l_sub.intersect(
                r_sub.as_ref(),
                &with_pool_set(|pool_set| pool_set.get_pool().clone()),
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
    let expected = (2..12).map(RowId::new).collect::<Vec<_>>();
    assert_eq!(got, expected);

    got.clear();
    assert_eq!(
        Some(12),
        s2.as_ref().iter_bounded(2, 12, |row| got.push(row))
    );
    let expected = (2..12).map(|x| RowId::new(x * 2)).collect::<Vec<_>>();
    assert_eq!(got, expected);
}
