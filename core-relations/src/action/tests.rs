use crate::{
    TableId, Value,
    action::mask::{IterResult, ValueSource},
    numeric_id::NumericId,
    pool::Clear,
    pool::{PoolSet, with_pool_set},
};

use super::{
    PredictedEntry, PredictedVals,
    mask::{Mask, MaskIter},
};

#[test]
fn predicted_vals_store_rows_contiguously() {
    let mut predicted = PredictedVals::default();
    let table = TableId::from_usize(3);
    let other_table = TableId::from_usize(4);
    let key = [
        Value::from_usize(10),
        Value::from_usize(11),
        Value::from_usize(12),
        Value::from_usize(13),
    ];
    let row = [key[0], key[1], key[2], key[3], Value::from_usize(99)];
    let mut default_calls = 0;

    let (got, inserted) = predicted.get_or_insert_with(table, &key, row.len(), |values, _| {
        default_calls += 1;
        values.extend_from_slice(&row[key.len()..]);
    });
    assert_eq!(got, row);
    assert!(inserted);

    let (got, inserted) = predicted.get_or_insert_with(table, &key, row.len(), |_, _| {
        default_calls += 1;
    });
    assert_eq!(got, row);
    assert!(!inserted);
    assert_eq!(default_calls, 1);

    let other_row = [key[0], key[1], key[2], key[3], Value::from_usize(100)];
    let (got, inserted) =
        predicted.get_or_insert_with(other_table, &key, other_row.len(), |values, _| {
            default_calls += 1;
            values.extend_from_slice(&other_row[key.len()..]);
        });
    assert_eq!(got, other_row);
    assert!(inserted);
    assert_eq!(predicted.index.len(), 2);
    assert_eq!(predicted.values.len(), 2 * row.len());

    predicted.clear();
    assert!(predicted.index.is_empty());
    assert!(predicted.values.is_empty());
}

#[test]
fn predicted_vals_resolve_hash_collisions_with_the_backing_rows() {
    let mut predicted = PredictedVals::default();
    let table = TableId::from_usize(3);
    let collision_key = [Value::from_usize(10), Value::from_usize(11)];
    let key = [Value::from_usize(20), Value::from_usize(21)];
    let collision_row = [collision_key[0], collision_key[1], Value::from_usize(30)];
    let row = [key[0], key[1], Value::from_usize(31)];
    let hash = PredictedVals::hash(table, &key);

    predicted.values.extend_from_slice(&collision_row);
    predicted.index.insert_unique(
        hash,
        PredictedEntry {
            hash,
            index: 0,
            table,
            key_arity: collision_key.len() as u32,
        },
        |entry| entry.hash,
    );

    let (got, inserted) = predicted.get_or_insert_with(table, &key, row.len(), |values, _| {
        values.extend_from_slice(&row[key.len()..]);
    });
    assert_eq!(got, row);
    assert!(inserted);

    let (got, inserted) = predicted.get_or_insert_with(table, &key, row.len(), |_, _| {
        unreachable!("the inserted row should be found through the collision")
    });
    assert_eq!(got, row);
    assert!(!inserted);
}

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
            if *x % 2 == 0 { Some(*x) } else { None }
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

#[test]
fn test_early_stop_initial_state() {
    empty_execution_state!(state);
    assert!(!state.should_stop());
}

#[test]
fn test_early_stop_trigger() {
    empty_execution_state!(state);
    assert!(!state.should_stop());

    state.trigger_early_stop();

    assert!(state.should_stop());
}

#[test]
fn test_early_stop_shared_across_clones() {
    empty_execution_state!(state1);
    let state2 = state1.clone();

    assert!(!state1.should_stop());
    assert!(!state2.should_stop());

    state1.trigger_early_stop();

    assert!(state1.should_stop());
    assert!(state2.should_stop());
}

#[test]
fn test_early_stop_multiple_clones() {
    empty_execution_state!(state1);
    let state2 = state1.clone();
    let state3 = state2.clone();

    assert!(!state1.should_stop());
    assert!(!state2.should_stop());
    assert!(!state3.should_stop());

    state2.trigger_early_stop();

    assert!(state1.should_stop());
    assert!(state2.should_stop());
    assert!(state3.should_stop());
}
