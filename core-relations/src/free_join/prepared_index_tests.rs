use super::*;
use crate::free_join::probe::CatalogContinuation;
use std::{mem, sync::Barrier};

#[test]
fn continuation_positions_remain_compact() {
    assert_eq!(
        mem::size_of::<ContinuationPosition>(),
        2 * mem::size_of::<u32>()
    );
    assert_eq!(
        mem::size_of::<CatalogContinuation<'_>>(),
        mem::size_of::<&RootContinuationCache>() + mem::size_of::<ContinuationPosition>()
    );
}

#[test]
fn root_continuation_cache_reuses_direct_and_dynamic_slots() {
    let shard_lens = [2, 0, 3];

    let direct = RootContinuationCache::default();
    direct.prepare(ChildShape::Direct, shard_lens.len(), |shard| {
        shard_lens[shard]
    });
    direct.prepare(ChildShape::Direct, shard_lens.len(), |shard| {
        shard_lens[shard]
    });
    assert_eq!(direct.slots(AccessId::new(0)).len(), 3);

    let dynamic = RootContinuationCache::default();
    dynamic.prepare(
        ChildShape::Dynamic { families: 3 },
        shard_lens.len(),
        |shard| shard_lens[shard],
    );

    let barrier = Barrier::new(16);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..16 {
            let dynamic = &dynamic;
            let barrier = &barrier;
            handles.push(scope.spawn(move || {
                barrier.wait();
                dynamic.slots(AccessId::new(1)).as_ptr() as usize
            }));
        }
        let addresses = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
    });

    assert!(!std::ptr::eq(
        dynamic.slots(AccessId::new(1)),
        dynamic.slots(AccessId::new(2)),
    ));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "root continuation shape changed after initialization")]
fn root_continuation_prepare_revalidates_initialized_shape() {
    let cache = RootContinuationCache::default();
    cache.prepare(ChildShape::Direct, 1, |_| 1);
    cache.prepare(ChildShape::Dynamic { families: 2 }, 1, |_| 1);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "direct root continuation was used by multiple indexed accesses")]
fn direct_root_continuation_rejects_different_successors() {
    let cache = RootContinuationCache::default();
    cache.prepare(ChildShape::Direct, 1, |_| 1);
    let _ = cache.slots(AccessId::new(0));
    let _ = cache.slots(AccessId::new(1));
}
