use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};

use smallvec::{SmallVec, smallvec};

use crate::{
    common::{IndexMap, Value},
    free_join::{
        AtomId, SubAtom, Variable,
        plan::{JoinStage, MatId, MatScanMode, ScanSpec, SingleScanSpec},
    },
    numeric_id::NumericId,
    offsets::Subset,
    row_buffer::RowBuffer,
    table_spec::ColumnId,
};

use crate::free_join::{
    join_tail::{
        BindingInfo, InstrOrder, for_each_stage_atom, materialization_is_live_in_tail,
        packed_child_shape_in_tail, scan_atom_tail_use, sort_plan_by_size_inner,
    },
    packed_cache::{RootProjection, TrieRoot},
    packed_trie::ChildShape,
    prepared_index::{
        AccessId, PreparedIndexKind, PreparedIndexSlot, PreparedJoinIndexes, PreparedTailMasks,
    },
};

#[test]
fn catalog_candidates_require_live_rows_before_terminal_match() {
    terminal_catalog_filter_cases(true);
}

#[test]
fn catalog_candidates_require_constraints_before_terminal_match() {
    terminal_catalog_filter_cases(false);
}

fn terminal_catalog_filter_cases(stale: bool) {
    for workers in [1, 4] {
        let pool = egglog_concurrency::ThreadPool::new(workers);
        pool.install(|| {
            for arity in [1, 2] {
                terminal_catalog_filter_case(stale, arity);
            }
        });
    }
}

fn terminal_catalog_filter_case(stale: bool, arity: usize) {
    use crate::{
        PlanStrategy,
        free_join::{
            Database, get_column_index_from_tableinfo, get_index_from_tableinfo, plan::Plan,
        },
        table::SortedWritesTable,
        table_shortcuts::v,
        table_spec::Constraint,
    };
    use egglog_reports::ReportLevel;

    let mut db = Database::new();
    let tables = (0..3)
        .map(|_| {
            db.add_table(
                SortedWritesTable::new(
                    arity,
                    arity,
                    None,
                    vec![],
                    Box::new(|_, old, new, _| {
                        assert_eq!(old, new);
                        false
                    }),
                ),
                std::iter::empty(),
                std::iter::empty(),
            )
        })
        .collect::<Vec<_>>();
    let [facts, gate, output] = tables.as_slice() else {
        unreachable!()
    };
    let key = |x: usize| {
        (0..arity)
            .map(|column| v(x + 100 * column))
            .collect::<Vec<_>>()
    };
    {
        let mut facts_buf = db.new_buffer(*facts);
        let mut gate_buf = db.new_buffer(*gate);
        for x in 0..64 {
            facts_buf.stage_insert(&key(x));
            if x < 16 {
                gate_buf.stage_insert(&key(x));
            }
        }
    }
    db.merge_all();

    let columns = (0..arity).map(ColumnId::from_usize).collect::<Vec<_>>();
    // Populate the persistent index before deleting anything. Rebuilding a
    // fresh index after deletion would conceal its stale-key candidates.
    if arity == 1 {
        drop(get_column_index_from_tableinfo(
            db.get_table_info(*facts),
            columns[0],
        ));
    } else {
        drop(get_index_from_tableinfo(
            db.get_table_info(*facts),
            &columns,
        ));
    }
    if stale {
        {
            let mut facts_buf = db.new_buffer(*facts);
            for x in 0..3 {
                facts_buf.stage_remove(&key(x));
            }
        }
        db.merge_all();
        assert!(db.get_table(*facts).has_stale_rows());
    }
    assert_eq!(db.get_table(*facts).all().size(), 64);
    // The index still has a candidate for key 0, but that candidate must not
    // establish existence: its row is stale or fails the remaining constraint.
    if arity == 1 {
        let index = get_column_index_from_tableinfo(db.get_table_info(*facts), columns[0]);
        assert!(index.get().unwrap().get_subset(&v(0)).is_some());
    } else {
        let index = get_index_from_tableinfo(db.get_table_info(*facts), &columns);
        assert!(index.get().unwrap().get_subset(key(0).as_slice()).is_some());
    }

    let constraints = if stale {
        vec![]
    } else {
        vec![Constraint::GtConst {
            col: columns[0],
            val: v(5),
        }]
    };
    let mut rsb = db.new_rule_set();
    let mut query = rsb.new_rule();
    query.set_plan_strategy(PlanStrategy::PureSize);
    query.set_no_decomp(true);
    let variables = (0..arity).map(|_| query.new_var()).collect::<Vec<_>>();
    let args = variables
        .iter()
        .map(|var| (*var).into())
        .collect::<Vec<_>>();
    query.add_atom(*gate, &args, &[]).unwrap();
    query.add_atom(*facts, &args, &constraints).unwrap();
    let mut rule = query.build();
    rule.insert(*output, &args).unwrap();
    rule.build_with_description("terminal-catalog-filter");
    let rules = rsb.build();

    let (plan, _, _) = rules.plans.values().next().unwrap();
    let Plan::SinglePlan(plan) = plan else {
        panic!("expected a single plan")
    };
    assert_eq!(
        plan.stages.instrs.len(),
        1,
        "the facts probe must be existence-only"
    );
    match &plan.stages.instrs[0] {
        JoinStage::Intersect { scans, .. } => {
            assert_eq!(arity, 1);
            assert_eq!(scans.len(), 2);
            let scan = scans
                .iter()
                .find(|scan| plan.atoms[scan.atom].table == *facts)
                .unwrap();
            assert_eq!(scan.column, columns[0]);
            assert_eq!(scan.cs.is_empty(), stale);
        }
        JoinStage::FusedIntersect {
            cover,
            to_intersect,
            ..
        } => {
            assert_eq!(plan.atoms[cover.to_index.atom].table, *gate);
            assert_eq!(to_intersect.len(), 1);
            let scan = &to_intersect[0].0;
            assert_eq!(plan.atoms[scan.to_index.atom].table, *facts);
            assert_eq!(scan.to_index.vars.len(), arity);
            assert_eq!(scan.constraints.is_empty(), stale);
        }
        _ => panic!("expected a terminal root probe"),
    }

    let expected = ((if stale { 3 } else { 6 })..16)
        .map(key)
        .collect::<Vec<_>>();
    let report = db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
    assert_eq!(
        report.num_matches("terminal-catalog-filter"),
        expected.len()
    );
    let table = db.get_table(*output);
    let mut actual = table
        .scan(table.all().as_ref())
        .iter()
        .map(|(_, row)| row.to_vec())
        .collect::<Vec<_>>();
    actual.sort();
    assert_eq!(actual, expected);
}

#[test]
fn shared_root_projection_keys_are_canonical_and_single_flight() {
    let unshared = TrieRoot::new(Subset::Dense(crate::OffsetRange::new(
        crate::RowId::from_usize(0),
        crate::RowId::from_usize(1),
    )));
    let prepared = PreparedIndexSlot::new(PreparedIndexKind::Uncacheable, AccessId::new(0));
    assert!(
        prepared
            .get_or_init_root_projection(&unshared, ColumnId::from_usize(0), &[], || {
                panic!("an unshared root must not build a shared projection")
            })
            .is_none()
    );

    let root = Arc::new(TrieRoot::new_shared(Subset::Dense(
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
                let prepared =
                    PreparedIndexSlot::new(PreparedIndexKind::Uncacheable, AccessId::new(0));
                barrier.wait();
                // Race both the canonicalized DashMap lookup and the lazy
                // projection publication, as parallel plans do.
                let projection = prepared
                    .get_or_init_root_projection(
                        &root,
                        ColumnId::from_usize(0),
                        &[upper.clone(), lower.clone()],
                        || {
                            builds.fetch_add(1, Ordering::Relaxed);
                            RootProjection::from_sorted_pairs(Vec::new())
                        },
                    )
                    .unwrap();
                let reused = prepared
                    .get_or_init_root_projection(
                        &root,
                        ColumnId::from_usize(0),
                        &[lower, upper],
                        || panic!("a retained projection must not be rebuilt"),
                    )
                    .unwrap();
                assert!(std::ptr::eq(projection, reused));
                projection as *const RootProjection as usize
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

    assert_eq!(order, InstrOrder::from_iter([1, 0, 2, 3].into_iter()));
}

fn prepared_for(stages: &[JoinStage]) -> PreparedJoinIndexes {
    let mut access_counts = crate::numeric_id::DenseIdMap::new();
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
                    PreparedIndexSlot::new(PreparedIndexKind::Uncacheable, access)
                })
                .collect()
        })
        .collect();
    let tail_masks = PreparedTailMasks::new(stages, &prepared_stages, access_counts.n_ids());
    PreparedJoinIndexes {
        stages: prepared_stages,
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
    let masks = prepared.tail_masks.as_ref().unwrap();
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
    assert_eq!(prepared_64.all_stage_mask(), Some(u64::MAX));

    let stages_65 = (0..65)
        .map(|column| intersect_stage(0, column))
        .collect::<Vec<_>>();
    assert!(prepared_for(&stages_65).tail_masks.is_none());
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
        .map(|_| Arc::new(TrieRoot::new(Subset::empty())))
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
