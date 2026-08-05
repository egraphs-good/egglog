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
use egglog_concurrency::Notification;

use crate::{
    ColumnId, Database, DisplacedTable, ExecutionState, Rebuilder, RowId, SequenceContainerValue,
    SortedWritesTable, Value, ValueRebuilder, row_buffer::RowBuffer, table_spec::WrappedTableRef,
};

use super::{
    ContainerBackend, ContainerEnv, ContainerRebuildSummary, ContainerValue, hash_container,
};

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct VecContainer(Vec<Value>);

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct LegacyVecContainer(Vec<Value>);

fn cont<const N: usize>(values: [usize; N]) -> VecContainer {
    VecContainer(values.iter().map(|&v| Value::from_usize(v)).collect())
}

impl ContainerValue for VecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        rebuilder.rebuild_slice(&mut self.0)
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.0.iter().copied()
    }
}

impl ContainerValue for LegacyVecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        rebuilder.rebuild_slice(&mut self.0)
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.0.iter().copied()
    }
}

impl SequenceContainerValue for VecContainer {
    fn encode_sequence(&self, out: &mut Vec<Value>) {
        out.extend_from_slice(&self.0);
    }

    fn decode_sequence(sequence: &[Value]) -> Self {
        Self(sequence.to_vec())
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        sequence
    }

    fn rebuild_sequence(
        sequence: &[Value],
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

/// A tiny rebuilder used to isolate outer-id canonicalization from inner
/// container rewrites in the unit tests below.
struct FakeRebuilder {
    old_outer_id: Option<Value>,
    new_outer_id: Option<Value>,
    old_inner_val: Option<Value>,
    new_inner_val: Option<Value>,
}

impl ValueRebuilder for FakeRebuilder {
    fn rebuild_val(&self, val: Value) -> Value {
        match (self.old_outer_id, self.new_outer_id) {
            (Some(old), Some(new)) if val == old => new,
            _ => val,
        }
    }

    fn rebuild_slice(&self, vals: &mut [Value]) -> bool {
        let mut changed = false;
        for val in vals {
            if let (Some(old), Some(new)) = (self.old_inner_val, self.new_inner_val)
                && *val == old
            {
                *val = new;
                changed = true;
            }
        }
        changed
    }
}

// Also exercised via the table-level rebuild path, so it implements the full
// `Rebuilder`; that path only calls the value-level methods for containers.
impl Rebuilder for FakeRebuilder {
    fn hint_col(&self) -> Option<ColumnId> {
        None
    }

    fn rebuild_buf(
        &self,
        _buf: &RowBuffer,
        _start: RowId,
        _end: RowId,
        _out: &mut crate::TaggedRowBuffer,
        _exec_state: &mut ExecutionState,
    ) {
        unreachable!("FakeRebuilder does not support rebuild_buf")
    }

    fn rebuild_subset(
        &self,
        _other: WrappedTableRef,
        _subset: crate::SubsetRef,
        _out: &mut crate::TaggedRowBuffer,
        _exec_state: &mut ExecutionState,
    ) {
        unreachable!("FakeRebuilder does not support rebuild_subset")
    }
}

#[test]
fn racing_inserts() {
    let mut db = Database::new();
    let counter = db.add_counter();
    let db = Arc::new(db);
    let start = Arc::new(Notification::new());
    let env = Arc::new(ContainerEnv::<VecContainer>::new(
        Box::new(|_state, v1, v2| {
            assert_eq!(v1, v2, "this test shouldn't merge anything");
            v1
        }),
        counter,
    ));
    let threads = (0..10)
        .map(|_| {
            let start = start.clone();
            let env = env.clone();
            let db = db.clone();
            std::thread::spawn(move || {
                db.with_execution_state(|es| {
                    start.wait();
                    env.get_or_insert(&cont([1, 2, 3]), es)
                })
            })
        })
        .collect::<Vec<_>>();
    start.notify();
    let results = Vec::from_iter(threads.into_iter().map(|t| t.join().unwrap()));

    for result in &results {
        assert_eq!(
            &*env.get_container(*result).unwrap_or_else(|| {
                panic!("container {result:?} not found");
            }),
            &cont([1, 2, 3])
        );
    }
    assert!(
        results.windows(2).all(|w| w[0] == w[1]),
        "all containers should be the same, got {results:?}"
    );
}

#[test]
fn incremental_reinsert_canonicalizes_displaced_outer_id() {
    let mut db = Database::new();
    let counter = db.add_counter();
    let mut env = ContainerEnv::<VecContainer>::new(
        Box::new(|_state, v1, v2| {
            assert_eq!(v1, v2, "this test shouldn't merge anything");
            v1
        }),
        counter,
    );
    let container = cont([1, 2, 3]);

    db.with_execution_state(|es| {
        let old_id = env.get_or_insert(&container, es);
        let new_id = Value::from_usize(old_id.index() + 1000);
        let hc = hash_container(&container);
        let target_map = env.to_id.determine_map(&container);
        let shard_mut = env.to_id.shards_mut()[target_map].get_mut();
        let (container, _) = shard_mut
            .remove_entry(hc as u64, |(_, v)| *v.get() == old_id)
            .expect("container should be present before reinsertion");

        let mut summary = ContainerRebuildSummary::default();
        env.reinsert_incremental(container, old_id, new_id, false, es, &mut summary);

        assert!(summary.changed());
        assert!(summary.dirty_ids().is_empty());
        assert!(env.get_container(old_id).is_none());
        assert_eq!(&*env.get_container(new_id).unwrap(), &cont([1, 2, 3]));
    });
}

#[test]
fn nonincremental_dirty_ids_only_include_stable_ids() {
    let mut db = Database::new();
    let counter = db.add_counter();
    let old_inner = Value::from_usize(1);
    let new_inner = Value::from_usize(2);

    let run_case = |outer_id_changes: bool| {
        let mut env = ContainerEnv::<VecContainer>::new(
            Box::new(|_state, v1, v2| {
                assert_eq!(v1, v2, "this test shouldn't merge anything");
                v1
            }),
            counter,
        );
        db.with_execution_state(|es| {
            let old_id = env.get_or_insert(&VecContainer(vec![old_inner]), es);
            let new_id = if outer_id_changes {
                Value::from_usize(old_id.index() + 1000)
            } else {
                old_id
            };
            let rebuilder = FakeRebuilder {
                old_outer_id: outer_id_changes.then_some(old_id),
                new_outer_id: outer_id_changes.then_some(new_id),
                old_inner_val: Some(old_inner),
                new_inner_val: Some(new_inner),
            };

            let summary = env.apply_rebuild_nonincremental(&rebuilder, es);
            assert!(summary.changed());
            if outer_id_changes {
                assert!(summary.dirty_ids().is_empty());
                assert!(env.get_container(old_id).is_none());
                assert_eq!(
                    &*env.get_container(new_id).unwrap(),
                    &VecContainer(vec![new_inner])
                );
            } else {
                assert_eq!(
                    summary.dirty_ids().iter().copied().collect::<Vec<_>>(),
                    vec![old_id]
                );
                assert_eq!(
                    &*env.get_container(old_id).unwrap(),
                    &VecContainer(vec![new_inner])
                );
            }
        });
    };

    run_case(false);
    run_case(true);
}

#[test]
fn type_erased_backend_metadata_reports_sequence_lengths() {
    let mut db = Database::new();
    let legacy_counter = db.add_counter();
    let sequence_counter = db.add_reservable_counter(8);
    let legacy_id = db
        .register_container_type::<LegacyVecContainer>(legacy_counter, |_state, left, right| {
            left.min(right)
        });
    let sequence_id = db.register_sequence_container_type::<VecContainer>(
        sequence_counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );

    assert_eq!(
        db.container_values().env_backend(legacy_id),
        ContainerBackend::Legacy
    );
    assert_eq!(
        db.container_values().env_backend(sequence_id),
        ContainerBackend::Sequence
    );
    assert_eq!(db.container_values().sequence_env_ids(), vec![sequence_id]);

    db.with_execution_state(|state| {
        let containers = state.container_values();
        containers.register_val(LegacyVecContainer(vec![Value::from_usize(1)]), state);
        containers.register_val(VecContainer(vec![Value::from_usize(2)]), state);
    });
    assert_eq!(db.container_values().env_len(legacy_id), 1);
    assert_eq!(db.container_values().env_len(sequence_id), 0);

    assert!(db.merge_all());
    assert_eq!(db.container_values().env_len(sequence_id), 1);
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
    let container =
        db.register_container_type::<VecContainer>(counter, |_state, left, right| left.min(right));
    let second_relation = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        iter::empty(),
        iter::empty(),
    );

    assert_eq!(first_relation.index(), 0);
    assert_eq!(crate::TableId::from(container).index(), 1);
    assert_eq!(second_relation.index(), 2);
    assert_eq!(db.next_table_id().index(), 3);
    assert_eq!(
        db.container_values().env_backend(container),
        ContainerBackend::Legacy
    );
}

#[test]
fn shared_rebuild_defers_relations_until_nested_container_collisions_converge() {
    let mut db = Database::new();
    let uf = db.add_table(DisplacedTable::default(), iter::empty(), iter::empty());
    let collision_merges = Arc::new(AtomicUsize::new(0));
    let collision_merges_for_callback = Arc::clone(&collision_merges);
    let counter = db.add_reservable_counter(8);
    db.register_sequence_container_type::<VecContainer>(
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
    db.register_sequence_container_type::<VecContainer>(
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
fn cloned_databases_have_independent_sequence_notifications() {
    let mut original = Database::new();
    let counter = original.add_reservable_counter(8);
    original.register_sequence_container_type::<VecContainer>(
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
fn sequence_merge_writes_participate_in_simple_fixed_point() {
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
    db.register_sequence_container_type::<VecContainer>(
        counter,
        move |state, left, right| {
            assert_eq!(
                state.get_table(gate).len(),
                1,
                "the simple scheduler must honor sequence read dependencies"
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
fn sequence_merge_writes_participate_in_dependency_aware_fixed_point() {
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
    db.register_sequence_container_type::<VecContainer>(
        counter,
        move |state, left, right| {
            assert_eq!(
                state.get_table(gate).len(),
                1,
                "the sequence merge must run after its read dependency"
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

    // The sequence environment plus the three ordinary tables above force
    // `merge_all` through its dependency-aware (four-or-more participant)
    // path. The sequence merge then notifies the previously inactive sink,
    // which must be consumed by the same call's fixed-point loop.
    assert!(db.merge_all());
    assert_eq!(db.get_table(sink).len(), 1);
}

#[test]
fn relation_merge_can_depend_on_nonzero_level_sequence_container() {
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
    let container = db.register_sequence_container_type::<VecContainer>(
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
                        "the relation merge must run after its sequence dependency"
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
    db.register_sequence_container_type::<VecContainer>(
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
fn failed_sequence_dependency_registration_can_be_retried() {
    let mut db = Database::new();
    let counter = db.add_reservable_counter(8);
    let missing_table = db.next_table_id();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.register_sequence_container_type::<VecContainer>(
            counter,
            |_state, left, right| left.min(right),
            [missing_table],
            iter::empty(),
        );
    }));
    assert!(result.is_err());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.register_sequence_container_type::<VecContainer>(
            counter,
            |_state, left, right| left.min(right),
            iter::empty(),
            [missing_table],
        );
    }));
    assert!(result.is_err());

    let container = db.register_sequence_container_type::<VecContainer>(
        counter,
        |_state, left, right| left.min(right),
        iter::empty(),
        iter::empty(),
    );
    let duplicate = db.register_sequence_container_type::<VecContainer>(
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
