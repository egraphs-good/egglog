use super::*;
use crate::free_join::Variable;
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
    assert_eq!(direct.slots(0).len(), 3);

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
                dynamic.slots(1).as_ptr() as usize
            }));
        }
        let addresses = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
    });

    assert!(!std::ptr::eq(dynamic.slots(1), dynamic.slots(2)));
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
    let _ = cache.slots(0);
    let _ = cache.slots(1);
}

#[test]
fn cover_only_stages_skip_prepared_index_state() {
    let stages = JoinStages::new(vec![JoinStage::Intersect {
        var: Variable::from_usize(0),
        scans: SmallVec::new(),
    }]);
    let atoms = Arc::new(DenseIdMap::new());
    assert!(matches!(
        PreparedJoinIndexes::new(&Database::new(), &atoms, &stages),
        PreparedJoinIndexes::NoIndexes
    ));
}

#[test]
fn cached_plan_layout_reuses_analysis_but_not_execution_state() {
    use crate::{PlanStrategy, table::SortedWritesTable, table_shortcuts::v};

    let mut db = Database::new();
    let table = db.add_table(
        SortedWritesTable::new(1, 1, None, vec![], Box::new(|_, _, _, _| false)),
        std::iter::empty(),
        std::iter::empty(),
    );
    let mut rsb = db.new_rule_set();
    let mut query = rsb.new_rule();
    query.set_plan_strategy(PlanStrategy::Gj);
    query.set_no_decomp(true);
    let x = query.new_var_named("x");
    query.add_atom(table, &[x.into()], &[]).unwrap();
    query.add_atom(table, &[x.into()], &[]).unwrap();
    let mut rule = query.build();
    rule.insert(table, &[x.into()]).unwrap();
    rule.build();
    let rules = rsb.build();
    let (plan, _, _) = rules.plans.values().next().unwrap();
    let Plan::SinglePlan(plan) = plan else {
        unreachable!()
    };
    let cloned_stages = plan.stages.clone();
    assert!(plan.stages.prepared_layout.get().is_none());
    let first = PreparedJoinIndexes::new(&db, &plan.atoms, &plan.stages);
    let PreparedJoinIndexes::Indexed { layout, states, .. } = &first else {
        panic!("fixture must prepare indexed accesses")
    };
    // Execution addresses and retained catalog handles must never enter the
    // shared plan layout, even when plans differ only in seminaive headers.
    assert!(states.iter().all(|state| state.root.get().is_none()));
    assert_eq!(first.resolve(&first.stage(0)[0]).packed_root(|| 123), 123);
    let next = PreparedJoinIndexes::new(&db, &plan.atoms, &cloned_stages);
    let PreparedJoinIndexes::Indexed {
        layout: next_layout,
        states: next_states,
        ..
    } = &next
    else {
        unreachable!()
    };
    assert!(std::ptr::eq(*layout, *next_layout));
    assert!(next_states[0].root.get().is_none());
    assert!(!std::ptr::eq(states.as_ptr(), next_states.as_ptr()));
    drop(first);
    drop(next);
    // The same layout remains valid after table mutations; catalog handles
    // are reacquired from the fresh execution state.
    let mut buffer = db.new_buffer(table);
    buffer.stage_insert(&[v(1)]);
    drop(buffer);
    db.merge_all();
    let refreshed = PreparedJoinIndexes::new(&db, &plan.atoms, &cloned_stages);
    let slot = refreshed.stage(0)[0];
    assert_eq!(
        refreshed
            .resolve(&slot)
            .column_index(&db.tables[table], ColumnId::new(0))
            .len(),
        1
    );
}
