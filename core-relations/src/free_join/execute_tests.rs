use std::{
    mem,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
};

use egglog_concurrency::SharedArena;
use smallvec::{SmallVec, smallvec};

use crate::{
    common::{IndexMap, Value},
    free_join::{
        AtomId, SubAtom, Variable,
        plan::{JoinStage, JoinStages, MatId, MatScanMode, ScanSpec, SingleScanSpec},
    },
    numeric_id::{DenseIdMap, NumericId},
    offsets::Subset,
    row_buffer::RowBuffer,
    table_spec::ColumnId,
};

use super::{
    AccessId, BindingInfo, CatalogContinuation, ContinuationPosition, Database, InlineRows,
    InstrOrder, LazyArenaHandle, PreparedIndexKind, PreparedIndexSlot, PreparedIndexState,
    PreparedIndexStateId, PreparedJoinIndexes, PreparedTailMasks, RootContinuationCache,
    RootProjection, SMALL_RESIDUAL, SmallColumnIndex, SmallColumnSink, TrieNode,
    for_each_stage_atom, materialization_is_live_in_tail, packed_child_shape_in_tail,
    scan_atom_tail_use, sort_plan_by_size_inner, top_index_shape_is_eligible,
};
use crate::free_join::packed_trie::ChildShape;

#[test]
fn cover_only_stages_skip_prepared_index_state() {
    let stages = JoinStages {
        instrs: Arc::new(vec![JoinStage::Intersect {
            var: Variable::from_usize(0),
            scans: SmallVec::new(),
        }]),
    };
    let atoms = Arc::new(DenseIdMap::new());
    assert!(matches!(
        PreparedJoinIndexes::new(&Database::new(), &atoms, &stages),
        PreparedJoinIndexes::NoIndexes
    ));
}

#[test]
fn arena_handle_is_created_only_on_first_packed_allocation() {
    let arena = SharedArena::new();
    let handle = LazyArenaHandle::new(&arena);
    assert!(handle.handle.get().is_none());
    handle.get();
    assert!(handle.handle.get().is_some());
}

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
fn small_column_index_groups_keys_and_sorts_inline_rows() {
    assert_eq!(
        mem::size_of::<InlineRows>(),
        mem::size_of::<u32>() + SMALL_RESIDUAL * mem::size_of::<crate::RowId>()
    );

    let mut sink = SmallColumnSink::default();
    for (value, row) in [(2, 7), (1, 5), (2, 3), (1, 4), (3, 6)] {
        sink.rows[sink.len] = (Value::from_usize(value), crate::RowId::from_usize(row));
        sink.len += 1;
    }

    let index = SmallColumnIndex::from_projected(sink);
    assert_eq!(index.len(), 3);
    assert_eq!(index.find(Value::from_usize(0)), None);
    assert_eq!(
        index
            .rows_at(index.find(Value::from_usize(1)).unwrap())
            .rows()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![4, 5]
    );
    assert_eq!(
        index
            .rows_at(index.find(Value::from_usize(2)).unwrap())
            .rows()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![3, 7]
    );
    assert_eq!(
        index
            .rows_at(index.find(Value::from_usize(3)).unwrap())
            .rows()[0]
            .index(),
        6
    );
}

#[test]
fn root_projection_is_a_final_dense_or_sparse_index() {
    let index = RootProjection::from_sorted_pairs(vec![
        (Value::from_usize(1), crate::RowId::from_usize(4)),
        (Value::from_usize(1), crate::RowId::from_usize(5)),
        (Value::from_usize(2), crate::RowId::from_usize(3)),
        (Value::from_usize(2), crate::RowId::from_usize(7)),
        (Value::from_usize(9), crate::RowId::from_usize(11)),
    ]);

    assert_eq!(index.len(), 3);
    assert_eq!(index.rows.len(), 5);
    assert_eq!(
        index
            .keys
            .iter()
            .map(|&(key, offset)| (key.index(), offset))
            .collect::<Vec<_>>(),
        vec![(1, 0), (2, 2), (9, 4), (0, 5)]
    );
    assert_eq!(index.find(Value::from_usize(0)), None);
    assert_eq!(index.find(Value::from_usize(8)), None);

    let dense = index.subset_at(index.find(Value::from_usize(1)).unwrap());
    let crate::SubsetRef::Dense(dense) = dense else {
        panic!("contiguous root-index rows must use a dense subset")
    };
    assert_eq!((dense.start.index(), dense.end.index()), (4, 6));

    let sparse = index.subset_at(index.find(Value::from_usize(2)).unwrap());
    let crate::SubsetRef::Sparse(sparse) = sparse else {
        panic!("noncontiguous root-index rows must use a sparse subset")
    };
    assert_eq!(
        sparse
            .inner()
            .iter()
            .map(|row| row.index())
            .collect::<Vec<_>>(),
        vec![3, 7]
    );

    let singleton = index.subset_at(index.find(Value::from_usize(9)).unwrap());
    let crate::SubsetRef::Dense(singleton) = singleton else {
        panic!("a singleton root-index group must use a dense subset")
    };
    assert_eq!((singleton.start.index(), singleton.end.index()), (11, 12));

    let empty = RootProjection::from_sorted_pairs(Vec::new());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.rows.len(), 0);
    assert_eq!(
        empty.keys.len(),
        1,
        "empty indexes retain only the sentinel"
    );
    assert_eq!(empty.find(Value::from_usize(0)), None);
}

#[test]
fn shared_root_projection_keys_are_canonical_and_single_flight() {
    let root = Arc::new(TrieNode::new_shared(Subset::Dense(
        crate::OffsetRange::new(crate::RowId::from_usize(0), crate::RowId::from_usize(1)),
    )));
    let lower = crate::Constraint::GtConst {
        col: ColumnId::from_usize(1),
        val: Value::from_usize(10),
    };
    let upper = crate::Constraint::LtConst {
        col: ColumnId::from_usize(1),
        val: Value::from_usize(20),
    };
    let forward = root
        .projection_slot(ColumnId::from_usize(0), &[lower.clone(), upper.clone()])
        .unwrap();
    let reversed = root
        .projection_slot(ColumnId::from_usize(0), &[upper.clone(), lower.clone()])
        .unwrap();
    assert!(Arc::ptr_eq(&forward, &reversed));
    assert!(!Arc::ptr_eq(
        &forward,
        &root
            .projection_slot(ColumnId::from_usize(1), &[lower.clone(), upper.clone()])
            .unwrap()
    ));

    let builds = AtomicUsize::new(0);
    let barrier = Barrier::new(16);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..16 {
            let root = root.clone();
            let lower = lower.clone();
            let upper = upper.clone();
            let builds = &builds;
            let barrier = &barrier;
            handles.push(scope.spawn(move || {
                barrier.wait();
                // Race both the canonicalized DashMap lookup and the lazy
                // projection publication, as parallel plans do.
                let slot = root
                    .projection_slot(ColumnId::from_usize(0), &[upper, lower])
                    .unwrap();
                slot.get_or_init(|| {
                    builds.fetch_add(1, Ordering::Relaxed);
                    RootProjection::from_sorted_pairs(Vec::new())
                }) as *const RootProjection as usize
            }));
        }
        let addresses = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
    });
    assert_eq!(builds.load(Ordering::Relaxed), 1);
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

fn scan(atom: usize) -> ScanSpec {
    ScanSpec {
        to_index: SubAtom {
            atom: AtomId::from_usize(atom),
            vars: smallvec![ColumnId::from_usize(0)],
        },
        constraints: Vec::new(),
    }
}

fn mat_stage(mat_id: usize) -> JoinStage {
    JoinStage::FusedIntersectMat {
        cover: MatId::from_usize(mat_id),
        mode: MatScanMode::KeyOnly,
        bind: SmallVec::new(),
        to_intersect: Vec::new(),
    }
}

fn intersect_stage(atom: usize, column: usize) -> JoinStage {
    JoinStage::Intersect {
        var: Variable::from_usize(column),
        scans: smallvec![SingleScanSpec {
            atom: AtomId::from_usize(atom),
            column: ColumnId::from_usize(column),
            cs: Vec::new(),
        }],
    }
}

#[test]
fn mixed_recursive_dvo_keeps_the_plan_prefix_as_its_refinement_anchor() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(1, 0),
        intersect_stage(1, 1),
        mat_stage(0),
    ];
    let mut binding_info = BindingInfo::default();
    for atom in 0..2 {
        binding_info.insert_subset(
            AtomId::from_usize(atom),
            Subset::Dense(crate::OffsetRange::new(
                crate::RowId::from_usize(0),
                crate::RowId::from_usize(100),
            )),
        );
    }

    // Stage 1 happened to run first in this branch. The stable plan prefix
    // still anchors the recursive ordering to stage 0 / atom 0. Using the
    // physical prefix here would instead promote stage 2 / atom 1.
    let mut order = InstrOrder::from_iter([1, 0, 2, 3].into_iter());
    sort_plan_by_size_inner(&mut order, 1..3, &stages, &mut binding_info);

    assert_eq!(order.data.as_slice(), &[1, 0, 2, 3]);
}

fn prepared_for(stages: &[JoinStage]) -> PreparedJoinIndexes {
    let mut access_counts = crate::numeric_id::DenseIdMap::new();
    let mut states = Vec::new();
    let prepared_stages: Box<[SmallVec<[PreparedIndexSlot; 4]>]> = stages
        .iter()
        .map(|stage| {
            let atoms = match stage {
                JoinStage::Intersect { scans, .. } => {
                    scans.iter().map(|scan| scan.atom).collect::<Vec<_>>()
                }
                JoinStage::FusedIntersect { to_intersect, .. }
                | JoinStage::FusedIntersectMat { to_intersect, .. } => to_intersect
                    .iter()
                    .map(|(scan, _)| scan.to_index.atom)
                    .collect(),
            };
            atoms
                .into_iter()
                .map(|atom| {
                    let next = access_counts.get_or_default(atom);
                    let access = AccessId::from_usize(*next);
                    *next += 1;
                    let kind = PreparedIndexKind::Uncacheable;
                    let state = PreparedIndexStateId::from_usize(states.len());
                    states.push(PreparedIndexState::new(kind));
                    PreparedIndexSlot::new(kind, access, state)
                })
                .collect()
        })
        .collect();
    let tail_masks = PreparedTailMasks::new(stages, &prepared_stages, access_counts.n_ids());
    PreparedJoinIndexes::Indexed {
        stages: prepared_stages,
        states: states.into_boxed_slice(),
        access_counts,
        tail_masks,
    }
}

fn permutations(values: &mut [usize], start: usize, result: &mut Vec<Vec<usize>>) {
    if start == values.len() {
        result.push(values.to_vec());
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        permutations(values, start + 1, result);
        values.swap(start, index);
    }
}

#[test]
fn prepared_tail_masks_match_scanner_for_every_permutation_and_suffix() {
    let stages = vec![
        intersect_stage(0, 0),
        JoinStage::Intersect {
            var: Variable::from_usize(1),
            scans: smallvec![
                SingleScanSpec {
                    atom: AtomId::from_usize(0),
                    column: ColumnId::from_usize(1),
                    cs: Vec::new(),
                },
                SingleScanSpec {
                    atom: AtomId::from_usize(1),
                    column: ColumnId::from_usize(0),
                    cs: Vec::new(),
                }
            ],
        },
        intersect_stage(1, 1),
        intersect_stage(2, 0),
    ];
    let prepared = prepared_for(&stages);
    let masks = prepared.tail_masks().unwrap();
    let mut orders = Vec::new();
    permutations(&mut [0, 1, 2, 3], 0, &mut orders);
    for order in orders {
        let instr_order = InstrOrder::from_iter(order.iter().copied());
        for resume_pos in 0..=order.len() {
            let remaining = order[resume_pos..]
                .iter()
                .fold(0u64, |mask, &stage| mask | (1u64 << stage));
            for atom_index in 0..=3 {
                let atom = AtomId::from_usize(atom_index);
                assert_eq!(
                    masks.atom_tail_use(atom, remaining, prepared.access_count(atom)),
                    scan_atom_tail_use(atom, &stages, &prepared, &instr_order, resume_pos,),
                    "tail metadata diverged for order {order:?}, suffix {resume_pos}, atom {atom_index}"
                );
            }
        }
    }
}

#[test]
fn prepared_tail_masks_use_u64_boundary_and_fallback_after_it() {
    let stages_64 = (0..64)
        .map(|column| intersect_stage(0, column))
        .collect::<Vec<_>>();
    let prepared_64 = prepared_for(&stages_64);
    assert_eq!(prepared_64.tail_masks().unwrap().all_stages, u64::MAX);

    let stages_65 = (0..65)
        .map(|column| intersect_stage(0, column))
        .collect::<Vec<_>>();
    assert!(prepared_for(&stages_65).tail_masks().is_none());
}

#[test]
fn packed_tail_shape_preserves_direct_graph_path() {
    let stages = vec![intersect_stage(0, 0), intersect_stage(0, 1)];
    let prepared = prepared_for(&stages);
    let order = InstrOrder::from_iter(0..stages.len());

    assert_eq!(
        packed_child_shape_in_tail(AtomId::from_usize(0), &stages, &prepared, &order, 1,),
        ChildShape::Direct
    );
    assert_eq!(
        packed_child_shape_in_tail(AtomId::from_usize(0), &stages, &prepared, &order, 2,),
        ChildShape::Leaf
    );
}

#[test]
fn packed_tail_shape_uses_dynamic_families_for_dvo_choice() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(0, 1),
        intersect_stage(0, 2),
    ];
    let prepared = prepared_for(&stages);
    let order = InstrOrder::from_iter([0, 2, 1].into_iter());

    assert_eq!(
        packed_child_shape_in_tail(AtomId::from_usize(0), &stages, &prepared, &order, 1,),
        ChildShape::Dynamic { families: 3 }
    );
}

#[test]
fn packed_tail_shape_stops_at_cover_and_reorder_barriers() {
    let atom = AtomId::from_usize(0);
    let stages = vec![
        intersect_stage(0, 0),
        JoinStage::FusedIntersect {
            cover: scan(0),
            bind: SmallVec::new(),
            to_intersect: Vec::new(),
        },
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(0),
            mode: MatScanMode::Full,
            bind: SmallVec::new(),
            to_intersect: Vec::new(),
        },
        intersect_stage(0, 1),
    ];
    let prepared = prepared_for(&stages);
    let order = InstrOrder::from_iter(0..stages.len());
    assert_eq!(
        packed_child_shape_in_tail(atom, &stages, &prepared, &order, 1),
        ChildShape::Leaf,
        "the cover consumes the packed residual before the later phase"
    );

    let stages = vec![
        intersect_stage(0, 0),
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(0),
            mode: MatScanMode::Full,
            bind: SmallVec::new(),
            to_intersect: vec![(scan(0), SmallVec::new())],
        },
        intersect_stage(0, 2),
    ];
    let prepared = prepared_for(&stages);
    let order = InstrOrder::from_iter(0..stages.len());
    assert_eq!(
        packed_child_shape_in_tail(atom, &stages, &prepared, &order, 1),
        ChildShape::Direct,
        "a singleton barrier hides indexed accesses in later phases"
    );
}

#[test]
fn top_index_partitioning_rejects_serial_tiny_and_skewed_shapes() {
    assert!(!top_index_shape_is_eligible(1, 10_000, 8, 64));
    assert!(!top_index_shape_is_eligible(4, 255, 8, 64));
    assert!(!top_index_shape_is_eligible(4, 10_000, 3, 64));
    assert!(top_index_shape_is_eligible(4, 256, 4, 64));
    assert!(top_index_shape_is_eligible(4, 40, 4, 10));
}

#[test]
fn task_clone_keeps_only_atoms_in_the_dynamic_join_tail() {
    let stages = vec![
        JoinStage::Intersect {
            var: Variable::from_usize(0),
            scans: smallvec![SingleScanSpec {
                atom: AtomId::from_usize(0),
                column: ColumnId::from_usize(0),
                cs: Vec::new(),
            }],
        },
        JoinStage::FusedIntersect {
            cover: scan(1),
            bind: SmallVec::new(),
            // Repeat the cover atom to verify that it is cloned once.
            to_intersect: vec![(scan(2), SmallVec::new()), (scan(1), SmallVec::new())],
        },
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(0),
            mode: MatScanMode::Full,
            bind: SmallVec::new(),
            to_intersect: vec![(scan(3), SmallVec::new())],
        },
    ];
    // The physical tail is stages 0 and 1, not the lexical suffix 1 and 2.
    let order = InstrOrder::from_iter([2, 0, 1].into_iter());

    let nodes = (0..4)
        .map(|_| Arc::new(TrieNode::new(Subset::empty())))
        .collect::<Vec<_>>();
    let mut source = BindingInfo::default();
    for (atom, node) in nodes.iter().enumerate() {
        source.insert_node(AtomId::from_usize(atom), Arc::clone(node));
    }
    let materializations = (0..2)
        .map(|_| Arc::new(IndexMap::<Vec<Value>, RowBuffer>::default()))
        .collect::<Vec<_>>();
    for (mat_id, materialization) in materializations.iter().enumerate() {
        source
            .materializations
            .insert(MatId::from_usize(mat_id), Arc::clone(materialization));
    }

    let child = source.clone_for_join_tail(&stages, &order, 1);
    for (atom, node) in nodes.iter().enumerate().take(3) {
        let cloned = child.subsets.get(AtomId::from_usize(atom)).unwrap();
        assert!(Arc::ptr_eq(cloned.root_arc(), node));
        assert_eq!(Arc::strong_count(node), 3);
    }
    assert!(!child.subsets.contains_key(AtomId::from_usize(3)));
    assert!(child.materializations.is_empty());
    assert_eq!(Arc::strong_count(&nodes[3]), 2);
    assert!(!materialization_is_live_in_tail(
        &stages,
        &order,
        1,
        MatId::from_usize(0)
    ));
    drop(child);
    assert!(nodes.iter().all(|node| Arc::strong_count(node) == 2));

    // Top-level partition jobs resume at zero and therefore retain the
    // driver stage as well as the rest of the dynamically ordered plan.
    let top = source.clone_for_join_tail(&stages, &order, 0);
    assert!((0..4).all(|atom| top.subsets.contains_key(AtomId::from_usize(atom))));
    assert!(top.materializations.contains_key(MatId::from_usize(0)));
    assert!(!top.materializations.contains_key(MatId::from_usize(1)));
    assert!(materialization_is_live_in_tail(
        &stages,
        &order,
        0,
        MatId::from_usize(0)
    ));
    assert_eq!(Arc::strong_count(&materializations[0]), 3);
    assert_eq!(Arc::strong_count(&materializations[1]), 2);

    // Exercise the exhaustive dependency visitor directly: materialized
    // covers are MatIds, so only their atom probes are reported.
    let mut mat_atoms = Vec::new();
    for_each_stage_atom(&stages[2], |atom| mat_atoms.push(atom));
    assert_eq!(mat_atoms, vec![AtomId::from_usize(3)]);
}

#[test]
fn task_clone_keeps_each_live_materialization_once_in_dynamic_order() {
    let stages = vec![mat_stage(0), mat_stage(1), mat_stage(0)];
    // The dynamic tail after the first stage contains Mat0 twice, while
    // lexical stage 1 (Mat1) has already executed.
    let order = InstrOrder::from_iter([1, 0, 2].into_iter());
    let materializations = (0..2)
        .map(|_| Arc::new(IndexMap::<Vec<Value>, RowBuffer>::default()))
        .collect::<Vec<_>>();
    let mut source = BindingInfo::default();
    for (mat_id, materialization) in materializations.iter().enumerate() {
        source
            .materializations
            .insert(MatId::from_usize(mat_id), Arc::clone(materialization));
    }

    let child = source.clone_for_join_tail(&stages, &order, 1);
    assert!(child.materializations.contains_key(MatId::from_usize(0)));
    assert!(!child.materializations.contains_key(MatId::from_usize(1)));
    assert_eq!(Arc::strong_count(&materializations[0]), 3);
    assert_eq!(Arc::strong_count(&materializations[1]), 2);
    assert!(materialization_is_live_in_tail(
        &stages,
        &order,
        1,
        MatId::from_usize(0)
    ));
    assert!(!materialization_is_live_in_tail(
        &stages,
        &order,
        1,
        MatId::from_usize(1)
    ));
}
