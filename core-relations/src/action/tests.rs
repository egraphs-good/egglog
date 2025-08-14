use crate::{
    action::mask::{IterResult, ValueSource},
    pool::{with_pool_set, PoolSet},
};

use super::mask::{Mask, MaskIter};

#[test]
fn mask_iter() {
    let ps = PoolSet::default();
    let offs = Vec::from_iter(0..100);
    let mut mask = Mask::new(0..100, &ps);
    let mut res = Vec::new();
    mask.iter(&offs).for_each(|x| res.push(*x));
    assert_eq!(offs, res);
}

#[test]
fn mask_iter_zip() {
    let ps = PoolSet::default();
    let offs1 = Vec::from_iter(0..100);
    let offs2 = Vec::from_iter(100..200);
    let mut mask = Mask::new(0..100, &ps);
    let mut res = Vec::new();
    mask.iter(&offs1)
        .zip(&offs2)
        .for_each(|(x, y)| res.push((*x, *y)));
    assert_eq!(
        Vec::from_iter(offs1.iter().copied().zip(offs2.iter().copied())),
        res
    );
}

#[test]
fn mask_iter_dyn() {
    let ps = PoolSet::default();
    let mut mask = Mask::new(0..3, &ps);
    let mut iter_dyn = mask.iter_dynamic(
        with_pool_set(|x| x.get_pool()),
        vec![
            ValueSource::Const(1),
            ValueSource::Slice(&[1, 3, 5]),
            ValueSource::Const(1),
            ValueSource::Slice(&[2, 4, 6]),
        ]
        .into_iter(),
    );
    match iter_dyn.get_at(2) {
        IterResult::Item(item) => {
            let v = item.as_slice();
            assert_eq!(v, [1, 5, 1, 6])
        }
        _ => unreachable!(),
    }
}

#[test]
fn retain() {
    let ps = PoolSet::default();
    let offs = Vec::from_iter(0..100);
    let mut mask = Mask::new(0..100, &ps);
    mask.iter(&offs).retain(|x| *x % 2 == 0);
    let mut got = Vec::new();
    mask.iter(&offs).for_each(|x| got.push(*x));
    assert_eq!(
        Vec::from_iter(offs.iter().copied().filter(|x| *x % 2 == 0)),
        got
    );
}

#[test]
fn fill_vec() {
    let ps = PoolSet::default();
    let offs = Vec::from_iter(0..100);
    let mut mask = Mask::new(0..100, &ps);
    let mut out = Vec::new();
    mask.iter(&offs).fill_vec(
        &mut out,
        || i32::MAX,
        |row, x| {
            assert_eq!(row, *x as usize);
            if *x % 2 == 0 {
                Some(*x)
            } else {
                None
            }
        },
    );
    // We should filter the mas for the entries for which we returned 'None'
    let mut got = Vec::new();
    mask.iter(&offs).for_each(|x| got.push(*x));
    assert_eq!(
        Vec::from_iter(offs.iter().copied().filter(|x| *x % 2 == 0)),
        got
    );

    assert_eq!(out.len(), 100);

    // The vector itself should have i32::MAX in for the odd indexes.
    for (i, x) in out.iter().copied().enumerate() {
        if i.is_multiple_of(2) {
            assert_eq!(x, i as i32);
        } else {
            assert_eq!(x, i32::MAX);
        }
    }
}
