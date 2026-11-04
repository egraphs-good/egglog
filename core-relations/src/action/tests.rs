use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    ContainerValueId, Database, MutationBuffer, TableId, Value,
    action::mask::{IterResult, ValueSource},
    numeric_id::NumericId,
    pool::Clear,
    pool::{PoolSet, with_pool_set},
};

use super::{
    PredictedEntry, PredictedVals, PredictedValueEntry, PredictionOwner,
    mask::{Mask, MaskIter},
};

#[derive(Clone)]
struct RecordingMutationBuffer {
    rows: Arc<Mutex<Vec<Vec<Value>>>>,
    fresh_calls: Arc<AtomicUsize>,
}

impl MutationBuffer for RecordingMutationBuffer {
    fn stage_insert(&mut self, row: &[Value]) {
        self.rows.lock().unwrap().push(row.to_vec());
    }

    fn stage_remove(&mut self, _key: &[Value]) {
        unreachable!("container prediction tests only stage inserts")
    }

    fn fresh_handle(&self) -> Box<dyn MutationBuffer> {
        self.fresh_calls.fetch_add(1, Ordering::Relaxed);
        Box::new(self.clone())
    }
}

fn recording_buffer(
    rows: &Arc<Mutex<Vec<Vec<Value>>>>,
    creates: &Arc<AtomicUsize>,
    fresh_calls: &Arc<AtomicUsize>,
) -> impl FnOnce() -> Box<dyn MutationBuffer> + 'static {
    let rows = Arc::clone(rows);
    let creates = Arc::clone(creates);
    let fresh_calls = Arc::clone(fresh_calls);
    move || {
        creates.fetch_add(1, Ordering::Relaxed);
        Box::new(RecordingMutationBuffer { rows, fresh_calls })
    }
}

#[test]
fn predicted_entry_keeps_its_original_layout_size() {
    assert_eq!(std::mem::size_of::<PredictedEntry>(), 24);
}

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
    assert!(predicted.by_value.is_empty());

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
    assert!(predicted.by_value.is_empty());
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
            row_end: collision_row.len() as u32,
            owner: PredictionOwner::table(table),
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
fn predicted_vals_reverse_lookup_supports_variable_rows_and_nested_use() {
    let mut predicted = PredictedVals::default();
    let inner_container = ContainerValueId::from_usize(3);
    let outer_container = ContainerValueId::from_usize(4);
    let inner_key = [Value::from_usize(10), Value::from_usize(11)];
    let inner_id = Value::from_usize(90);
    let inner_row = [inner_key[0], inner_key[1], inner_id];

    let (row, inserted) =
        predicted.get_or_insert_container_with(inner_container, &inner_key, |values, _| {
            values.push(inner_id);
        });
    assert!(inserted);
    assert_eq!(row, inner_row);

    // The inner identity is available before any staged write is merged, so it
    // can immediately be embedded in another variable-length predicted row.
    let inner = predicted
        .get_container_by_value(inner_container, inner_id)
        .unwrap();
    let outer_key = [inner[2], Value::from_usize(20), Value::from_usize(21)];
    let outer_id = Value::from_usize(91);
    let outer_row = [outer_key[0], outer_key[1], outer_key[2], outer_id];
    let (row, inserted) =
        predicted.get_or_insert_container_with(outer_container, &outer_key, |values, _| {
            values.push(outer_id);
        });
    assert!(inserted);
    assert_eq!(row, outer_row);
    assert_eq!(
        predicted.get_container_by_value(outer_container, outer_id),
        Some(outer_row.as_slice())
    );
    assert_eq!(
        predicted.get_container_by_value(inner_container, outer_id),
        None
    );
}

#[test]
#[should_panic(expected = "one locally predicted value cannot identify two different rows")]
fn predicted_vals_reverse_lookup_rejects_a_conflicting_identity() {
    let mut predicted = PredictedVals::default();
    let container = ContainerValueId::from_usize(3);
    let identity = Value::from_usize(90);
    let first_key = [Value::from_usize(10)];
    let second_key = [Value::from_usize(11), Value::from_usize(12)];

    predicted.get_or_insert_container_with(container, &first_key, |values, _| {
        values.push(identity);
    });
    predicted.get_or_insert_container_with(container, &second_key, |values, _| {
        values.push(identity);
    });
}

#[test]
fn predicted_vals_keep_table_and_container_namespaces_distinct() {
    let mut predicted = PredictedVals::default();
    let table = TableId::from_usize(3);
    let container = ContainerValueId::from_usize(3);
    let key = [Value::from_usize(10)];
    let table_row = [key[0], Value::from_usize(80)];
    let container_row = [key[0], Value::from_usize(90)];

    let (row, inserted) =
        predicted.get_or_insert_with(table, &key, table_row.len(), |values, _| {
            values.push(table_row[1]);
        });
    assert!(inserted);
    assert_eq!(row, table_row);

    let (row, inserted) = predicted.get_or_insert_container_with(container, &key, |values, _| {
        values.push(container_row[1]);
    });
    assert!(inserted);
    assert_eq!(row, container_row);
    assert_eq!(predicted.index.len(), 2);
    assert_eq!(
        predicted.get_container_by_value(container, container_row[1]),
        Some(container_row.as_slice())
    );

    let (row, inserted) = predicted.get_or_insert_with(table, &key, table_row.len(), |_, _| {
        unreachable!("the table prediction should remain independently addressable")
    });
    assert!(!inserted);
    assert_eq!(row, table_row);
}

#[test]
#[should_panic(expected = "predicted row builder produced the wrong arity")]
fn predicted_container_rows_require_exactly_one_identity_value() {
    let mut predicted = PredictedVals::default();
    let container = ContainerValueId::from_usize(3);
    let key = [Value::from_usize(10)];
    predicted.get_or_insert_container_with(container, &key, |values, _| {
        values.extend_from_slice(&[Value::from_usize(11), Value::from_usize(90)]);
    });
}

#[test]
fn predicted_vals_reverse_lookup_resolves_raw_hash_collisions() {
    let mut predicted = PredictedVals::default();
    let container = ContainerValueId::from_usize(3);
    let owner = PredictionOwner::container(container);
    let collision_row = [Value::from_usize(10), Value::from_usize(80)];
    let key = [Value::from_usize(20), Value::from_usize(21)];
    let identity = Value::from_usize(90);
    let row = [key[0], key[1], identity];
    let hash = PredictedVals::value_hash(owner, identity);

    predicted.values.extend_from_slice(&collision_row);
    predicted.by_value.insert_unique(
        hash,
        PredictedValueEntry {
            hash,
            index: 0,
            row_end: collision_row.len() as u32,
            owner,
        },
        |entry| entry.hash,
    );

    let (got, inserted) = predicted.get_or_insert_container_with(container, &key, |values, _| {
        values.push(identity);
    });
    assert_eq!(got, row);
    assert!(inserted);
    assert_eq!(
        predicted.get_container_by_value(container, identity),
        Some(row.as_slice())
    );
    assert_eq!(predicted.by_value.len(), 2);
    assert!(
        predicted
            .by_value
            .find(hash, |entry| predicted.value_row(*entry) == collision_row)
            .is_some()
    );
}

#[test]
fn predicted_vals_are_execution_local_across_clones() {
    empty_execution_state!(state);
    let container = ContainerValueId::from_usize(3);
    let key = [Value::from_usize(10)];
    let identity = Value::from_usize(90);
    state
        .predicted
        .get_or_insert_container_with(container, &key, |values, _| values.push(identity));

    let cloned = state.clone();
    assert_eq!(
        state.predicted.get_container_by_value(container, identity),
        Some([key[0], identity].as_slice())
    );
    assert_eq!(
        cloned.predicted.get_container_by_value(container, identity),
        None
    );
}

#[test]
fn execution_state_predicts_and_stages_each_local_container_key_once() {
    let mut db = Database::default();
    let counter = db.add_reservable_counter(8);
    let mut state = super::ExecutionState::new(db.read_only_view(), Default::default());
    let container = ContainerValueId::from_usize(3);
    let rows = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
    let creates = Arc::new(AtomicUsize::new(0));
    let fresh_calls = Arc::new(AtomicUsize::new(0));
    let key = [Value::from_usize(10), Value::from_usize(11)];

    let first = state.predict_container_value(
        container,
        &key,
        counter,
        recording_buffer(&rows, &creates, &fresh_calls),
    );
    let repeated = state.predict_container_value(
        container,
        &key,
        counter,
        recording_buffer(&rows, &creates, &fresh_calls),
    );
    assert_eq!(first, repeated);
    assert!(state.changed);
    assert_eq!(creates.load(Ordering::Relaxed), 1);
    assert_eq!(fresh_calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        state.predicted_container_row(container, first),
        Some([key[0], key[1], first].as_slice())
    );
    assert_eq!(*rows.lock().unwrap(), vec![vec![key[0], key[1], first]]);

    let other_key = [Value::from_usize(20)];
    let second = state.predict_container_value(
        container,
        &other_key,
        counter,
        recording_buffer(&rows, &creates, &fresh_calls),
    );
    assert_ne!(first, second);
    assert_eq!(second.index(), first.index() + 1);
    assert_eq!(creates.load(Ordering::Relaxed), 1);
    assert_eq!(
        *rows.lock().unwrap(),
        vec![vec![key[0], key[1], first], vec![other_key[0], second]]
    );
}

#[test]
fn execution_state_clone_uses_fresh_container_buffer_handles() {
    let mut db = Database::default();
    let counter = db.add_reservable_counter(8);
    let mut state = super::ExecutionState::new(db.read_only_view(), Default::default());
    let container = ContainerValueId::from_usize(3);
    let rows = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
    let creates = Arc::new(AtomicUsize::new(0));
    let fresh_calls = Arc::new(AtomicUsize::new(0));
    let key = [Value::from_usize(10)];
    let original = state.predict_container_value(
        container,
        &key,
        counter,
        recording_buffer(&rows, &creates, &fresh_calls),
    );

    let mut cloned = state.clone();
    assert_eq!(fresh_calls.load(Ordering::Relaxed), 1);
    assert_eq!(cloned.predicted_container_row(container, original), None);

    let cloned_key = [Value::from_usize(20)];
    let cloned_value = cloned.predict_container_value(container, &cloned_key, counter, || {
        unreachable!("the cloned state should already have a fresh container buffer")
    });
    assert_ne!(original, cloned_value);
    assert!(cloned.changed);
    assert_eq!(creates.load(Ordering::Relaxed), 1);
    assert_eq!(
        *rows.lock().unwrap(),
        vec![vec![key[0], original], vec![cloned_key[0], cloned_value]]
    );
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
