use super::*;

#[test]
fn shared_roots_build_once_including_empty_intersections() {
    use std::sync::Barrier;

    for empty in [false, true] {
        let cache = TrieCache::with_shared(Default::default());
        let builds = AtomicUsize::new(0);
        let barrier = Barrier::new(8);
        let roots = std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        cache.root(TableId::new(0), HeaderConstraintId::new(0), || {
                            builds.fetch_add(1, Ordering::Relaxed);
                            (!empty).then(|| {
                                Subset::Dense(OffsetRange::new(RowId::new(2), RowId::new(8))).into()
                            })
                        })
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        for root in &roots {
            if empty {
                assert!(root.is_none());
            } else {
                assert!(Arc::ptr_eq(
                    root.as_ref().unwrap(),
                    roots[0].as_ref().unwrap()
                ));
            }
        }
        assert!(
            cache
                .root(TableId::new(1), HeaderConstraintId::new(0), || {
                    builds.fetch_add(1, Ordering::Relaxed);
                    None
                })
                .is_none()
        );
        assert_eq!(builds.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn packed_root_slots_belong_to_one_run_and_one_root() {
    let rows = Subset::Dense(crate::OffsetRange::new(
        crate::RowId::new(0),
        crate::RowId::new(8),
    ));
    let first = OwnedAtomRows::new_shared(rows.clone());
    let next_run = OwnedAtomRows::new_shared(rows);
    let family = FamilyId::new(0);
    first.packed_root_slot(family, 2).unwrap().set(123).unwrap();
    assert!(
        first
            .packed_root_slot(FamilyId::new(1), 2)
            .unwrap()
            .get()
            .is_none()
    );
    assert!(
        next_run
            .packed_root_slot(family, 2)
            .unwrap()
            .get()
            .is_none()
    );
}
