//! Basic container operations get test coverage in `src/tests.rs`.
//!
//! This module has tests that verify specific behavior in a multithreaded setting that are harder
//! to exercise deterministically when testing end to end.

use std::sync::Arc;

use crate::numeric_id::NumericId;
use egglog_concurrency::Notification;

use crate::{
    ColumnId, Database, ExecutionState, Rebuilder, RowId, Value, row_buffer::RowBuffer,
    table_spec::WrappedTableRef,
};

use super::{ContainerEnv, ContainerRebuildSummary, ContainerValue, hash_container};

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct VecContainer(Vec<Value>);

fn cont<const N: usize>(values: [usize; N]) -> VecContainer {
    VecContainer(values.iter().map(|&v| Value::from_usize(v)).collect())
}

impl ContainerValue for VecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        rebuilder.rebuild_slice(&mut self.0)
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.0.iter().copied()
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

impl Rebuilder for FakeRebuilder {
    fn hint_col(&self) -> Option<ColumnId> {
        None
    }

    fn rebuild_val(&self, val: Value) -> Value {
        match (self.old_outer_id, self.new_outer_id) {
            (Some(old), Some(new)) if val == old => new,
            _ => val,
        }
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
