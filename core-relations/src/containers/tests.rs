//! Basic container operations get test coverage in `src/tests.rs`.
//!
//! This module has tests that verify specific behavior in a multithreaded setting that are harder
//! to exercise deterministically when testing end to end.

use std::sync::Arc;

use concurrency::Notification;
use numeric_id::NumericId;

use crate::{Database, Rebuilder, Value};

use super::{Container, ContainerEnv};

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct VecContainer(Vec<Value>);

fn cont<const N: usize>(values: [usize; N]) -> VecContainer {
    VecContainer(values.iter().map(|&v| Value::from_usize(v)).collect())
}

impl Container for VecContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        rebuilder.rebuild_slice(&mut self.0)
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.0.iter().copied()
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
