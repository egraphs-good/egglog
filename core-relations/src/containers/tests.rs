//! Basic container operations get test coverage in `src/tests.rs`.
//!
//! This module has tests that verify specific behavior in a multithreaded setting that are harder
//! to exercise deterministically when testing end to end.

use std::{
    iter,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::numeric_id::NumericId;

use crate::{
    ColumnId, ContainerValue, Database, DisplacedTable, SortedWritesTable, Value, ValueRebuilder,
};

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct VecContainer(Vec<Value>);

fn cont<const N: usize>(values: [usize; N]) -> VecContainer {
    VecContainer(values.iter().map(|&v| Value::from_usize(v)).collect())
}

impl ContainerValue for VecContainer {
    fn encode_sequence(&self, _base_values: &crate::BaseValues, out: &mut Vec<Value>) {
        out.extend_from_slice(&self.0);
    }

    fn decode_sequence(sequence: &[Value], _base_values: &crate::BaseValues) -> Self {
        Self(sequence.to_vec())
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        sequence
    }

    fn rebuild_sequence(
        sequence: &[Value],
        _base_values: &crate::BaseValues,
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool {
        out.extend_from_slice(sequence);
        if rebuilder.rebuild_slice(out) {
            true
        } else {
            out.clear();
            false
        }
    }
}

#[test]
fn type_erased_registry_reports_container_lengths() {
    let mut db = Database::new();
    let counter = db.add_reservable_counter(8);
    let container_id = db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );

    assert_eq!(db.container_values().env_ids(), vec![container_id]);

    db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(VecContainer(vec![Value::from_usize(2)]), state);
    });
    assert_eq!(db.container_values().env_len(container_id), 0);

    assert!(db.merge_all());
    assert_eq!(db.container_values().env_len(container_id), 1);
    assert_eq!(db.container_values().container_len(), 1);
}

#[test]
fn relations_and_containers_share_one_table_id_namespace() {
    let mut db = Database::new();
    let first_relation = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );
    let counter = db.add_reservable_counter(8);
    let container = db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );
    let second_relation = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );

    assert_eq!(first_relation.index(), 0);
    assert_eq!(crate::TableId::from(container).index(), 1);
    assert_eq!(second_relation.index(), 2);
    assert_eq!(db.next_table_id().index(), 3);
    assert_eq!(db.container_values().env_ids(), [container]);
}

#[test]
fn shared_rebuild_defers_relations_until_nested_container_collisions_converge() {
    let mut db = Database::new();
    let uf = db.add_table(DisplacedTable::default(), iter::empty(), iter::empty());
    let collision_merges = Arc::new(AtomicUsize::new(0));
    let collision_merges_for_callback = Arc::clone(&collision_merges);
    let counter = db.add_reservable_counter(8);
    db.register_container_type::<VecContainer>(
        counter,
        move |state, left, right| {
            assert!(
                state.get_table(uf).rebuilder(&[]).is_some(),
                "the rebuild source must remain readable by container collision callbacks"
            );
            collision_merges_for_callback.fetch_add(1, Ordering::Relaxed);
            state.stage_insert(uf, &[left, right, Value::from_usize(2)]);
            left.min(right)
        },
        [uf],
        [uf],
    );
    let parents = db.add_table(
        SortedWritesTable::new(
            1,
            3,
            Some(ColumnId::new(2)),
            vec![ColumnId::new(0), ColumnId::new(1)],
            Box::new(|_, current, incoming, _| {
                assert_eq!(current[1], incoming[1]);
                false
            }),
        ),
        [uf],
        iter::empty(),
    );

    let child_a = Value::from_usize(10_000);
    let child_b = Value::from_usize(20_000);
    let tag_a = Value::from_usize(30_000);
    let tag_b = Value::from_usize(40_000);
    let (inner_a, inner_b, outer_a, outer_b) = db.with_execution_state(|state| {
        let containers = state.container_values();
        let inner_a = containers.register_val(VecContainer(vec![child_a]), state);
        let inner_b = containers.register_val(VecContainer(vec![child_b]), state);
        let outer_a = containers.register_val(VecContainer(vec![inner_a]), state);
        let outer_b = containers.register_val(VecContainer(vec![inner_b]), state);
        state.stage_insert(parents, &[tag_a, outer_a, Value::from_usize(0)]);
        state.stage_insert(parents, &[tag_b, outer_b, Value::from_usize(0)]);
        state.stage_insert(uf, &[child_a, child_b, Value::from_usize(1)]);
        state.stage_insert(uf, &[tag_a, tag_b, Value::from_usize(1)]);
        (inner_a, inner_b, outer_a, outer_b)
    });
    assert!(db.merge_all());

    // The first container round merges the inner Vec identities. The relation
    // must remain untouched because its keys would collapse while the outer
    // Vec identities are still distinct.
    assert!(db.apply_rebuild(uf, &[parents], Value::from_usize(2)));
    assert_eq!(collision_merges.load(Ordering::Relaxed), 1);
    assert!(db.get_table(parents).get_row(&[tag_a]).is_some());
    assert!(db.get_table(parents).get_row(&[tag_b]).is_some());

    let rebuilder = db.get_table(uf).rebuilder(&[]).unwrap();
    assert_eq!(
        rebuilder.rebuild_val(inner_a),
        rebuilder.rebuild_val(inner_b)
    );
    assert_ne!(
        rebuilder.rebuild_val(outer_a),
        rebuilder.rebuild_val(outer_b)
    );
    drop(rebuilder);

    // The next outer round can now collide the outer Vec keys. It likewise
    // defers the strict relation until that new identity union is published.
    assert!(db.apply_rebuild(uf, &[parents], Value::from_usize(3)));
    assert_eq!(collision_merges.load(Ordering::Relaxed), 2);
    assert!(db.get_table(parents).get_row(&[tag_a]).is_some());
    assert!(db.get_table(parents).get_row(&[tag_b]).is_some());

    // With the container/source chain stable, the third round rebuilds the
    // relation from one coherent snapshot. Both key and value canonicalize
    // together, so the :no-merge-style assertion above is satisfied.
    assert!(db.apply_rebuild(uf, &[parents], Value::from_usize(4)));

    let (canonical_outer_a, canonical_outer_b, canonical_tag_a, canonical_tag_b) = {
        let rebuilder = db.get_table(uf).rebuilder(&[]).unwrap();
        (
            rebuilder.rebuild_val(outer_a),
            rebuilder.rebuild_val(outer_b),
            rebuilder.rebuild_val(tag_a),
            rebuilder.rebuild_val(tag_b),
        )
    };
    assert_eq!(canonical_outer_a, canonical_outer_b);
    assert_eq!(canonical_tag_a, canonical_tag_b);
    let row = db
        .get_table(parents)
        .get_row(&[canonical_tag_a])
        .expect("canonical relation row must be published after source convergence");
    assert_eq!(row.vals[1], canonical_outer_a);
    assert_eq!(db.get_table(parents).len(), 1);
}

#[test]
fn containers_round_trip_predictions_through_database_merge() {
    let mut db = Database::new();
    let counter = db.add_reservable_counter(8);
    db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );
    let expected = cont([1, 2, 3]);

    let first = db.with_execution_state(|state| {
        let containers = state.container_values();
        let id = containers.register_val(expected.clone(), state);
        assert_eq!(
            state.get_container::<VecContainer>(id),
            Some(expected.clone())
        );
        assert_eq!(
            state.container_sequence::<VecContainer>(id),
            Some(&expected.0[..])
        );
        id
    });
    let second = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected.clone(), state)
    });
    assert_ne!(
        first, second,
        "predictions are intentionally execution-local"
    );

    assert!(db.merge_all());
    let winner = first.min(second);
    assert_eq!(
        db.container_values()
            .get_val::<VecContainer>(winner)
            .as_deref(),
        Some(&expected)
    );
    assert!(
        db.container_values()
            .get_val::<VecContainer>(first.max(second))
            .is_none()
    );
}

#[test]
fn cloned_databases_have_independent_container_notifications() {
    let mut original = Database::new();
    let counter = original.add_reservable_counter(8);
    original.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );
    let expected = cont([4, 5, 6]);
    let id = original.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected.clone(), state)
    });
    let mut cloned = original.clone();

    assert!(original.merge_all());
    assert!(cloned.merge_all());
    assert_eq!(
        original
            .container_values()
            .get_val::<VecContainer>(id)
            .as_deref(),
        Some(&expected)
    );
    assert_eq!(
        cloned
            .container_values()
            .get_val::<VecContainer>(id)
            .as_deref(),
        Some(&expected)
    );
}

#[test]
fn container_merge_writes_participate_in_simple_fixed_point() {
    let mut db = Database::new();
    let sink = db.add_table(
        SortedWritesTable::new(
            2,
            2,
            None,
            vec![],
            Box::new(|_, current, incoming, _| {
                assert_eq!(current, incoming);
                false
            }),
        ),
        iter::empty(),
        iter::empty(),
    );
    let gate = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );
    let counter = db.add_reservable_counter(8);
    db.register_container_type::<VecContainer>(
        counter,
        move |state, left, right| {
            assert_eq!(
                state.get_table(gate).len(),
                1,
                "the simple scheduler must honor container read dependencies"
            );
            state.stage_insert(sink, &[left, right]);
            left.min(right)
        },
        [gate],
        [sink],
    );
    db.with_execution_state(|state| {
        state.stage_insert(gate, &[Value::from_usize(1)]);
    });
    let expected = cont([7, 8]);
    let first = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected.clone(), state)
    });
    let second = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected, state)
    });
    assert_ne!(first, second);

    assert!(db.merge_all());
    assert_eq!(db.get_table(sink).len(), 1);
}

#[test]
fn container_merge_writes_participate_in_dependency_aware_fixed_point() {
    let mut db = Database::new();
    let sink = db.add_table(
        SortedWritesTable::new(
            2,
            2,
            None,
            vec![],
            Box::new(|_, current, incoming, _| {
                assert_eq!(current, incoming);
                false
            }),
        ),
        iter::empty(),
        iter::empty(),
    );
    let mut add_flag_table = || {
        db.add_table(
            SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
            iter::empty(),
            iter::empty(),
        )
    };
    let gate = add_flag_table();
    let first_bystander = add_flag_table();
    let second_bystander = add_flag_table();
    let counter = db.add_reservable_counter(8);
    db.register_container_type::<VecContainer>(
        counter,
        move |state, left, right| {
            assert_eq!(
                state.get_table(gate).len(),
                1,
                "the container merge must run after its read dependency"
            );
            state.stage_insert(sink, &[left, right]);
            left.min(right)
        },
        [gate],
        [sink],
    );

    db.with_execution_state(|state| {
        state.stage_insert(gate, &[Value::from_usize(1)]);
        state.stage_insert(first_bystander, &[Value::from_usize(2)]);
        state.stage_insert(second_bystander, &[Value::from_usize(3)]);
    });
    let expected = cont([9, 10]);
    let first = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected.clone(), state)
    });
    let second = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected, state)
    });
    assert_ne!(first, second);

    // The container environment plus the three ordinary tables above force
    // `merge_all` through its dependency-aware (four-or-more participant)
    // path. The container merge then notifies the previously inactive sink,
    // which must be consumed by the same call's fixed-point loop.
    assert!(db.merge_all());
    assert_eq!(db.get_table(sink).len(), 1);
}

#[test]
fn relation_merge_can_depend_on_nonzero_level_container() {
    let mut db = Database::new();
    let gate = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );
    let bystander = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );
    let counter = db.add_reservable_counter(8);
    let container = db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        [gate],
        iter::empty(),
    );
    let expected = cont([11, 12]);
    let relation = db.add_table(
        SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new({
                let expected = expected.clone();
                move |state, current, incoming, out| {
                    let winner = current[1].min(incoming[1]);
                    assert_eq!(
                        state.container_sequence::<VecContainer>(winner),
                        Some(expected.0.as_slice()),
                        "the relation merge must run after its container dependency"
                    );
                    if current[1] == winner {
                        false
                    } else {
                        out.extend_from_slice(&[current[0], winner]);
                        true
                    }
                }
            }),
        ),
        [container.into()],
        iter::empty(),
    );

    db.with_execution_state(|state| {
        state.stage_insert(gate, &[Value::from_usize(1)]);
        state.stage_insert(bystander, &[Value::from_usize(2)]);
    });
    let first = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected.clone(), state)
    });
    let second = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected, state)
    });
    assert_ne!(first, second);
    db.with_execution_state(|state| {
        state.stage_insert(relation, &[Value::from_usize(0), first]);
        state.stage_insert(relation, &[Value::from_usize(0), second]);
    });

    assert!(db.merge_all());
    assert_eq!(db.get_table(relation).len(), 1);
}

#[test]
fn same_stratum_relation_merge_can_read_active_container() {
    let mut db = Database::new();
    let counter = db.add_reservable_counter(8);
    db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );
    let relation = db.add_table(
        SortedWritesTable::new(
            1,
            2,
            None,
            vec![],
            Box::new(|state, current, incoming, _| {
                assert!(
                    state
                        .container_sequence::<VecContainer>(current[1])
                        .is_some(),
                    "a relation merge must be able to read an active container"
                );
                assert!(
                    state
                        .container_sequence::<VecContainer>(incoming[1])
                        .is_some(),
                    "a relation merge must be able to read an active container"
                );
                false
            }),
        ),
        iter::empty(),
        iter::empty(),
    );
    let mut add_bystander = || {
        db.add_table(
            SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
            iter::empty(),
            iter::empty(),
        )
    };
    let first_bystander = add_bystander();
    let second_bystander = add_bystander();
    let first = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(cont([21]), state)
    });
    let second = db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(cont([22]), state)
    });
    db.with_execution_state(|state| {
        state.stage_insert(relation, &[Value::from_usize(0), first]);
        state.stage_insert(relation, &[Value::from_usize(0), second]);
        state.stage_insert(first_bystander, &[Value::from_usize(1)]);
        state.stage_insert(second_bystander, &[Value::from_usize(2)]);
    });

    // Four active same-level participants force the stratum path. Container
    // reads by arbitrary primitive callbacks are ambient rather than declared
    // in the dependency graph, so the environment must remain installed while
    // the relation merge runs.
    assert!(db.merge_all());
    assert_eq!(db.get_table(relation).len(), 1);
}

#[test]
fn failed_table_dependency_registration_does_not_mutate_database() {
    let mut db = Database::new();
    let expected_id = db.next_table_id();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.add_table(
            SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
            [expected_id],
            iter::empty(),
        );
    }));
    assert!(result.is_err());

    let actual_id = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );
    assert_eq!(actual_id, expected_id);
    db.with_execution_state(|state| {
        state.stage_insert(actual_id, &[Value::from_usize(1)]);
    });
    assert!(db.merge_all());
    assert_eq!(db.get_table(actual_id).len(), 1);
}

#[test]
fn failed_container_dependency_registration_can_be_retried() {
    let mut db = Database::new();
    let counter = db.add_reservable_counter(8);
    let missing_table = db.next_table_id();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.register_container_type::<VecContainer>(
            counter,
            |_state, left, right| left.min(right),
            [missing_table],
            iter::empty(),
        );
    }));
    assert!(result.is_err());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.register_container_type::<VecContainer>(
            counter,
            |_state, left, right| left.min(right),
            iter::empty(),
            [missing_table],
        );
    }));
    assert!(result.is_err());

    let container = db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );
    let duplicate = db.register_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.max(right),
        [missing_table],
        [missing_table],
    );
    assert_eq!(duplicate, container, "the first registration owns the deps");
    let expected = cont([13, 14]);
    db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(expected, state);
    });
    assert!(db.merge_all());
    assert_eq!(db.container_values().env_len(container), 1);
}
