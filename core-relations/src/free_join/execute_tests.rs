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
        packed_child_shape_in_tail, scan_atom_tail_use, select_next_stage, suffix_stage_mask,
    },
    packed_cache::{FamilyId, OwnedAtomRows},
    packed_trie::ChildShape,
    prepared_index::{
        AccessId, PreparedIndexKind, PreparedIndexSlot, PreparedIndexStateId, PreparedJoinIndexes,
        PreparedJoinLayout, PreparedTailMaskWidth, StageMask,
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
        table_shortcuts::TimestampedTable,
        table_shortcuts::v,
        table_spec::Constraint,
    };
    use egglog_reports::ReportLevel;

    let mut db = Database::new();
    let tables = (0..3)
        .map(|_| {
            db.add_table(
                TimestampedTable::new(
                    arity,
                    arity,
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
fn materialized_full_and_key_only_morsels_preserve_rejected_frames_and_join_tails() {
    use super::{InPlaceMaterializer, JoinState, MatSpec, ScopedMaterializer};
    use crate::{
        Database,
        free_join::{ProcessedConstraints, plan::JoinStages},
        numeric_id::DenseIdMap,
        query::{Atom, VarColumnMap},
        table_shortcuts::{TimestampedTable, v},
    };

    const KEYS: usize = 1025;
    const VALUES: usize = 40;
    let variables = [0, 1, 2].map(Variable::from_usize);
    let [x, y, _] = variables;
    let input_mat = MatId::from_usize(0);
    let output_mat = MatId::from_usize(1);
    let mut expected = Vec::new();
    for key in 0..KEYS {
        if key % 3 != 0 {
            for value in 0..VALUES {
                expected.push(vec![v(key), v(value), v(100_000 + key * VALUES + value)]);
            }
        }
    }
    // KeyOnly has 683 surviving frames: two full 256-frame morsels and a
    // remainder. Full also crosses boundaries inside one materialized group.
    assert!(KEYS - KEYS.div_ceil(3) > 2 * 256);

    for mode in [MatScanMode::Full, MatScanMode::KeyOnly] {
        for workers in [1, 4] {
            let pool = egglog_concurrency::ThreadPool::new(workers);
            let mut actual = pool.install(|| {
                let mut db = Database::default();
                let mut add_table = |arity| {
                    db.add_table(
                        TimestampedTable::new(arity, arity, Box::new(|_, _, _, _| false)),
                        std::iter::empty(),
                        std::iter::empty(),
                    )
                };
                let facts = add_table(3);
                let gate = add_table(1);
                let mut materialization = IndexMap::default();
                {
                    let mut fact_buf = db.new_buffer(facts);
                    let mut gate_buf = db.new_buffer(gate);
                    for key in 0..KEYS {
                        let mut values = RowBuffer::new(1);
                        for value in 0..VALUES {
                            fact_buf.stage_insert(&[
                                v(key),
                                v(value),
                                v(100_000 + key * VALUES + value),
                            ]);
                            values.add_row(&[v(value)]);
                        }
                        materialization.insert(vec![v(key)], values);
                        if key % 3 != 0 {
                            gate_buf.stage_insert(&[v(key)]);
                        }
                    }
                }
                db.merge_all();
                let mut atoms = DenseIdMap::new();
                for (atom, table, vars) in [
                    (AtomId::from_usize(0), facts, variables.as_slice()),
                    (AtomId::from_usize(1), gate, variables[..1].as_ref()),
                ] {
                    let mut var_columns = VarColumnMap::default();
                    for (column, var) in vars.iter().enumerate() {
                        var_columns.insert(*var, ColumnId::from_usize(column));
                    }
                    atoms.insert(
                        atom,
                        Atom {
                            table,
                            var_columns,
                            constraints: ProcessedConstraints::dummy(),
                        },
                    );
                }
                let atoms = Arc::new(atoms);
                let full = mode == MatScanMode::Full;
                let columns = if full {
                    smallvec![ColumnId::new(0), ColumnId::new(1)]
                } else {
                    smallvec![ColumnId::new(0)]
                };
                let mut bind = smallvec![(ColumnId::new(0), x)];
                if full {
                    bind.push((ColumnId::new(1), y));
                }
                let mut instrs = vec![JoinStage::FusedIntersectMat {
                    cover: input_mat,
                    mode: mode.clone(),
                    bind,
                    // Reject after the first probe has buffered a refinement,
                    // exercising rollback of both instructions and side rows.
                    to_intersect: vec![
                        (
                            ScanSpec {
                                to_index: SubAtom {
                                    atom: AtomId::new(0),
                                    vars: columns.clone(),
                                },
                                constraints: vec![],
                            },
                            columns,
                        ),
                        (scan(1), smallvec![ColumnId::new(0)]),
                    ],
                }];
                if !full {
                    instrs.push(intersect_stage(0, 1));
                }
                instrs.push(intersect_stage(0, 2));
                let stages = JoinStages::new(instrs);
                let layout = prepared_for(&stages.instrs);
                let prepared = PreparedJoinIndexes::from_layout(&layout, &[]);
                let mut binding_info = BindingInfo::default();
                for (atom, info) in atoms.iter() {
                    binding_info.insert_subset(atom, db.get_table(info.table).all());
                }
                binding_info
                    .materializations
                    .insert(input_mat, Arc::new(materialization));
                let arena = egglog_concurrency::SharedArena::new();
                let exec = super::ExecutionState::new(db.read_only_view(), Default::default());
                let join = JoinState::new(&db, exec.seed(), None, &arena);
                let mut order = InstrOrder::from_iter(0..stages.instrs.len());
                let mut leaves = smallvec![false; stages.instrs.len()];
                let remaining = prepared.all_stage_mask::<u64>();
                let mut specs = DenseIdMap::new();
                specs.insert(
                    output_mat,
                    MatSpec {
                        msg_vars: variables.into_iter().collect(),
                        val_vars: smallvec![],
                    },
                );
                let output = if workers == 1 {
                    let mut materializations = DenseIdMap::new();
                    materializations.insert(output_mat, IndexMap::default());
                    let mut buf = InPlaceMaterializer {
                        specs: &specs,
                        materializations,
                        scratch_key: vec![],
                        scratch_val: vec![],
                    };
                    join.run_plan(
                        &stages,
                        &prepared,
                        &atoms,
                        output_mat,
                        &mut order,
                        &mut leaves,
                        0,
                        None,
                        remaining,
                        &mut binding_info,
                        &mut buf,
                    );
                    buf.materializations
                        .unwrap_val(output_mat)
                        .into_iter()
                        .collect::<Vec<_>>()
                } else {
                    let retired = super::RetiredLocalStates::default();
                    let mut outputs = DenseIdMap::new();
                    outputs.insert(output_mat, Arc::new(crate::common::DashMap::default()));
                    let outputs = Arc::new(outputs);
                    let specs = Arc::new(specs);
                    pool.scope(|scope| {
                        let mut buf = ScopedMaterializer {
                            scope,
                            retired_states: &retired,
                            specs,
                            materializations: outputs.clone(),
                            scratch_key: vec![],
                            scratch_val: vec![],
                        };
                        join.run_plan(
                            &stages,
                            &prepared,
                            &atoms,
                            output_mat,
                            &mut order,
                            &mut leaves,
                            0,
                            None,
                            remaining,
                            &mut binding_info,
                            &mut buf,
                        );
                    });
                    outputs[output_mat]
                        .iter()
                        .map(|entry| (entry.key().clone(), entry.value().clone()))
                        .collect()
                };
                assert!(binding_info.materializations.contains_key(input_mat));
                for (atom, info) in atoms.iter() {
                    assert_eq!(
                        binding_info.subsets[atom].size(),
                        db.get_table(info.table).all().size()
                    );
                }
                output
                    .into_iter()
                    .map(|(key, rows)| {
                        assert_eq!(rows.len(), 1, "each binding must be emitted exactly once");
                        key
                    })
                    .collect::<Vec<_>>()
            });
            actual.sort();
            assert_eq!(actual, expected, "{mode:?}, workers={workers}");
        }
    }
}

#[test]
fn shared_packed_root_keys_are_canonical_and_single_flight() {
    let unshared = OwnedAtomRows::new(Subset::Dense(crate::OffsetRange::new(
        crate::RowId::from_usize(0),
        crate::RowId::from_usize(1),
    )));
    assert!(unshared.packed_root_slot(FamilyId::new(0), 0).is_none());

    let root = Arc::new(OwnedAtomRows::new_shared(Subset::Dense(
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
    let mut db = crate::free_join::Database::new();
    let table = db.add_table(
        crate::table_shortcuts::TimestampedTable::new(2, 2, Box::new(|_, _, _, _| false)),
        std::iter::empty(),
        std::iter::empty(),
    );
    let info = &mut db.tables[table];
    let family = info.successor_family(ColumnId::new(0), &[lower.clone(), upper.clone()]);
    let reversed_family = info.successor_family(ColumnId::new(0), &[upper.clone(), lower.clone()]);
    let other_column = info.successor_family(ColumnId::new(1), &[lower.clone(), upper]);
    let other_constraints = info.successor_family(ColumnId::new(0), &[lower]);
    let family_count = info.successor_families.len();
    assert_eq!(family, reversed_family);
    let forward = root.packed_root_slot(family, family_count).unwrap();
    let reversed = root
        .packed_root_slot(reversed_family, family_count)
        .unwrap();
    assert!(std::ptr::eq(forward, reversed));
    for other in [other_column, other_constraints] {
        assert!(!std::ptr::eq(
            forward,
            root.packed_root_slot(other, family_count).unwrap()
        ));
    }

    let builds = AtomicUsize::new(0);
    let barrier = Barrier::new(16);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..16 {
            let root = root.clone();
            let builds = &builds;
            let barrier = &barrier;
            handles.push(scope.spawn(move || {
                barrier.wait();
                let slot = root.packed_root_slot(family, family_count).unwrap();
                let address = slot.get_or_init(|| {
                    builds.fetch_add(1, Ordering::Relaxed);
                    123
                });
                assert_eq!(
                    *slot.get_or_init(|| panic!("a packed root must not be rebuilt")),
                    123
                );
                *address
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
    intersect_atoms_stage(&[atom], column)
}

fn intersect_atoms_stage(atoms: &[usize], column: usize) -> JoinStage {
    JoinStage::Intersect {
        var: Variable::from_usize(column),
        scans: atoms
            .iter()
            .map(|&atom| SingleScanSpec {
                atom: AtomId::from_usize(atom),
                column: ColumnId::from_usize(column),
                cs: Vec::new(),
            })
            .collect(),
    }
}

fn binding_info_with_sizes(sizes: &[usize]) -> BindingInfo<'static, 'static> {
    let mut binding_info = BindingInfo::default();
    for (atom, &rows) in sizes.iter().enumerate() {
        binding_info.insert_subset(
            AtomId::from_usize(atom),
            Subset::Dense(crate::OffsetRange::new(
                crate::RowId::from_usize(0),
                crate::RowId::from_usize(rows),
            )),
        );
    }
    binding_info
}

#[test]
fn initial_dvo_selects_only_the_first_stage() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(1, 0),
        intersect_stage(0, 1),
        intersect_stage(2, 0),
    ];
    let binding_info = binding_info_with_sizes(&[160, 80, 66]);
    let mut order = InstrOrder::from_iter(0..stages.len());

    select_next_stage(
        &mut order,
        &mut smallvec![false; stages.len()],
        0,
        &stages,
        &binding_info,
    );

    // Choose the root before partitioning it, but defer the remaining choices
    // until that root has actually filtered their inputs.
    assert_eq!(order, InstrOrder::from_iter([3, 0, 1, 2].into_iter()));
}

#[test]
fn initial_dvo_preserves_materialized_key_root_priority() {
    let stages = vec![
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(0),
            mode: MatScanMode::KeyOnly,
            bind: smallvec![(ColumnId::from_usize(0), Variable::from_usize(0))],
            to_intersect: Vec::new(),
        },
        intersect_stage(0, 1),
    ];
    for (rows, expected) in [(1, [0, 1]), (3, [0, 1])] {
        let mut binding_info = binding_info_with_sizes(&[rows]);
        let keys = (0..2)
            .map(|key| (vec![Value::from_usize(key)], RowBuffer::new(1)))
            .collect();
        binding_info
            .materializations
            .insert(MatId::from_usize(0), Arc::new(keys));
        let mut order = InstrOrder::from_iter(0..stages.len());

        select_next_stage(
            &mut order,
            &mut smallvec![false; stages.len()],
            0,
            &stages,
            &binding_info,
        );

        assert_eq!(order, InstrOrder::from_iter(expected.into_iter()));
    }
}

#[test]
fn runtime_dvo_preserves_materialized_key_priority_over_smaller_rows() {
    let stages = vec![
        intersect_stage(1, 0),
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(0),
            mode: MatScanMode::KeyOnly,
            bind: smallvec![(ColumnId::from_usize(0), Variable::from_usize(1))],
            to_intersect: Vec::new(),
        },
        intersect_stage(0, 2),
    ];
    let mut binding_info = binding_info_with_sizes(&[1, 10]);
    let keys = (0..2)
        .map(|key| (vec![Value::from_usize(key)], RowBuffer::new(1)))
        .collect();
    binding_info
        .materializations
        .insert(MatId::from_usize(0), Arc::new(keys));
    let mut order = InstrOrder::from_iter(0..stages.len());

    select_next_stage(
        &mut order,
        &mut smallvec![false; stages.len()],
        1,
        &stages,
        &binding_info,
    );

    // The common key applies after the root too. This intentionally differs
    // from a first-only policy that uses row counts alone at runtime.
    assert_eq!(order, InstrOrder::from_iter([0, 1, 2].into_iter()));
}

fn key_only_stage(mat_id: usize, arity: usize, probes: usize) -> JoinStage {
    JoinStage::FusedIntersectMat {
        cover: MatId::from_usize(mat_id),
        mode: MatScanMode::KeyOnly,
        bind: (0..arity)
            .map(|i| (ColumnId::from_usize(i), Variable::from_usize(i)))
            .collect(),
        to_intersect: (0..probes).map(|atom| (scan(atom), smallvec![])).collect(),
    }
}

fn insert_materialized_keys(binding_info: &mut BindingInfo<'_, '_>, id: usize, rows: usize) {
    let keys = (0..rows)
        .map(|key| (vec![Value::from_usize(key)], RowBuffer::new(1)))
        .collect();
    binding_info
        .materializations
        .insert(MatId::from_usize(id), Arc::new(keys));
}

#[test]
fn common_dvo_key_orders_by_arity_then_rows_then_relations_then_physical_ties() {
    // Arity beats the ordinary singleton and one-key materialization; rows
    // break the two-key tie, then relations, then current physical order.
    for prefix in [0, 1] {
        let mut stages: Vec<_> = (0..prefix).map(|_| intersect_stage(0, 9)).collect();
        stages.push(intersect_stage(0, 8));
        stages.extend([
            key_only_stage(0, 1, 0),
            key_only_stage(1, 2, 0),
            key_only_stage(2, 2, 0),
            key_only_stage(3, 2, 1),
            key_only_stage(4, 2, 1),
        ]);
        let mut binding_info = binding_info_with_sizes(&[1]);
        for (id, rows) in [1, 4, 2, 2, 2].into_iter().enumerate() {
            insert_materialized_keys(&mut binding_info, id, rows);
        }
        for reversed_tie in [false, true] {
            let mut positions: Vec<_> = (0..stages.len()).collect();
            if reversed_tie {
                positions.swap(prefix + 4, prefix + 5);
            }
            let winner = positions[prefix + 4];
            let mut expected = positions.clone();
            expected.remove(prefix + 4);
            expected.insert(prefix, winner);
            let mut order = InstrOrder::from_iter(positions.into_iter());
            select_next_stage(
                &mut order,
                &mut smallvec![false; stages.len()],
                prefix,
                &stages,
                &binding_info,
            );
            assert_eq!(order, InstrOrder::from_iter(expected.into_iter()));
        }
    }
}

#[test]
fn common_dvo_key_cannot_cross_barriers_but_applies_in_the_next_phase() {
    for mode in [
        MatScanMode::Full,
        MatScanMode::Value(smallvec![Variable::from_usize(9)]),
        MatScanMode::Lookup(smallvec![Variable::from_usize(9)]),
    ] {
        let stages = vec![
            intersect_stage(0, 9),
            JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(0),
                mode,
                bind: SmallVec::new(),
                to_intersect: Vec::new(),
            },
            intersect_stage(0, 8),
            key_only_stage(1, 2, 0),
        ];
        let mut binding_info = binding_info_with_sizes(&[1]);
        insert_materialized_keys(&mut binding_info, 1, 100);
        let mut order = InstrOrder::from_iter(0..stages.len());
        let mut leaf_scans = smallvec![false; stages.len()];
        for start in [0, 1] {
            select_next_stage(&mut order, &mut leaf_scans, start, &stages, &binding_info);
            assert_eq!(order, InstrOrder::from_iter(0..stages.len()));
        }
        select_next_stage(&mut order, &mut leaf_scans, 2, &stages, &binding_info);
        assert_eq!(order, InstrOrder::from_iter([0, 1, 3, 2].into_iter()));
    }
}

#[test]
fn common_dvo_key_keeps_materialization_leaf_flags_consistent() {
    use crate::free_join::join_tail::recompute_leaf_scans;
    let stages = vec![
        intersect_stage(0, 9),
        intersect_stage(1, 8),
        key_only_stage(0, 1, 0),
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(1),
            mode: MatScanMode::Lookup(smallvec![Variable::from_usize(0)]),
            bind: SmallVec::new(),
            to_intersect: Vec::new(),
        },
    ];
    let mut binding_info = binding_info_with_sizes(&[1, 2]);
    insert_materialized_keys(&mut binding_info, 0, 100);
    let mut order = InstrOrder::from_iter(0..stages.len());
    let mut leaf_scans = smallvec![false; stages.len()];
    recompute_leaf_scans(&order, &mut leaf_scans, &stages, 0);
    for _ in 0..2 {
        select_next_stage(&mut order, &mut leaf_scans, 1, &stages, &binding_info);
        assert_eq!(order, InstrOrder::from_iter([0, 2, 1, 3].into_iter()));
        let mut expected = smallvec![false; stages.len()];
        recompute_leaf_scans(&order, &mut expected, &stages, 0);
        assert_eq!(leaf_scans, expected);
        assert!(
            !leaf_scans[1],
            "the lookup still needs the materialized key scalar"
        );
    }
}

#[test]
fn initial_dvo_keeps_materialization_barriers_and_later_phases_fixed() {
    for mode in [
        MatScanMode::Full,
        MatScanMode::Value(smallvec![Variable::from_usize(0)]),
        MatScanMode::Lookup(smallvec![Variable::from_usize(0)]),
    ] {
        let stages = vec![
            intersect_stage(0, 0),
            intersect_stage(1, 0),
            JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(0),
                mode,
                bind: SmallVec::new(),
                to_intersect: Vec::new(),
            },
            intersect_stage(0, 1),
            intersect_stage(2, 0),
        ];
        let binding_info = binding_info_with_sizes(&[160, 66, 1]);
        let mut order = InstrOrder::from_iter(0..stages.len());

        select_next_stage(
            &mut order,
            &mut smallvec![false; stages.len()],
            0,
            &stages,
            &binding_info,
        );

        // The singleton after the barrier cannot be selected or moved yet.
        assert_eq!(order, InstrOrder::from_iter([1, 0, 2, 3, 4].into_iter()));
    }
}

#[test]
fn runtime_dvo_compares_residual_rows_regardless_of_prior_refinement() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(1, 0),
        intersect_stage(1, 1),
    ];
    for (residual_sizes, expected) in [
        ([7_582, 70], [1, 2, 0]),
        ([70, 7_582], [1, 0, 2]),
        ([1, 2], [1, 0, 2]),
        ([100, 100], [1, 0, 2]),
    ] {
        let binding_info = binding_info_with_sizes(&residual_sizes);
        // Only atom 1 has been refined. The smaller residual wins, with the
        // current physical order deciding a tie, even if atom 0 is unrefined.
        let mut order = InstrOrder::from_iter([1, 0, 2].into_iter());
        select_next_stage(
            &mut order,
            &mut smallvec![false; stages.len()],
            1,
            &stages,
            &binding_info,
        );
        assert_eq!(
            order,
            InstrOrder::from_iter(expected.into_iter()),
            "residual sizes: {residual_sizes:?}"
        );
    }
}

#[test]
fn runtime_dvo_breaks_equal_row_counts_by_number_of_scans() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(0, 1),
        intersect_atoms_stage(&[1, 2], 2),
    ];
    let binding_info = binding_info_with_sizes(&[100, 100, 200]);
    let mut order = InstrOrder::from_iter(0..stages.len());

    // Both candidates can scan 100 rows. The two unrefined scans take
    // precedence over a single scan of the already refined atom.
    select_next_stage(
        &mut order,
        &mut smallvec![false; stages.len()],
        1,
        &stages,
        &binding_info,
    );
    assert_eq!(order, InstrOrder::from_iter([0, 2, 1].into_iter()));
}

#[test]
fn runtime_dvo_preserves_physical_order_when_rows_and_scans_tie() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(1, 0),
        intersect_stage(1, 1),
        intersect_stage(0, 1),
        intersect_stage(1, 2),
    ];
    for physical_order in [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3]] {
        let binding_info = binding_info_with_sizes(&[100, 100]);
        let original = InstrOrder::from_iter(physical_order.into_iter());
        let mut order = original.clone();

        // Atom 0 has depth one and atom 1 has depth two. Neither depth nor
        // logical stage number should break a tie in the immediate choice.
        select_next_stage(
            &mut order,
            &mut smallvec![false; stages.len()],
            3,
            &stages,
            &binding_info,
        );
        assert_eq!(order, original);
    }
}

#[test]
fn runtime_dvo_uses_live_sizes_only_for_the_next_stage() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(0, 1),
        intersect_stage(0, 2),
        intersect_stage(1, 0),
        intersect_stage(0, 3),
        intersect_stage(1, 1),
        intersect_stage(2, 0),
    ];
    for small_rows in [1, 2] {
        let binding_info = binding_info_with_sizes(&[65, small_rows, 1]);
        let original = InstrOrder::from_iter(0..stages.len());

        // Atom 2's unrefined singleton wins when atom 1 has two rows; when
        // both have one row, physical order selects atom 1. Insert the winning
        // stage before the current one, preserving every other relative order.
        let mut order = original;
        let mut leaf_scans = smallvec![false; stages.len()];
        select_next_stage(&mut order, &mut leaf_scans, 4, &stages, &binding_info);
        let expected = if small_rows == 1 {
            [0, 1, 2, 3, 5, 4, 6]
        } else {
            [0, 1, 2, 3, 6, 4, 5]
        };
        assert_eq!(order, InstrOrder::from_iter(expected.into_iter()));
    }
}

#[test]
fn runtime_dvo_leaves_unexecuted_suffix_in_its_existing_order() {
    let stages = vec![
        intersect_atoms_stage(&[0, 2], 0),
        intersect_atoms_stage(&[0, 1], 1),
        intersect_atoms_stage(&[2, 4], 2),
        intersect_atoms_stage(&[0, 3], 3),
    ];
    let binding_info = binding_info_with_sizes(&[160, 66, 80, 160, 80]);
    let mut order = InstrOrder::from_iter(0..stages.len());
    let mut leaf_scans = smallvec![false; stages.len()];

    // Stage 1 wins with 66 rows versus 80 and 160. Speculatively counting its
    // refinements would move stage 3 ahead of stage 2. Runtime ordering must
    // leave both alone until stage 1 has executed and its residuals are known.
    select_next_stage(&mut order, &mut leaf_scans, 1, &stages, &binding_info);
    assert_eq!(order, InstrOrder::from_iter([0, 1, 2, 3].into_iter()));
}

#[test]
fn runtime_dvo_reconsiders_the_next_stage_after_residuals_change() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(0, 1),
        intersect_stage(1, 0),
        intersect_stage(2, 0),
    ];
    let mut binding_info = binding_info_with_sizes(&[160, 66, 80]);
    let mut order = InstrOrder::from_iter(0..stages.len());
    let mut leaf_scans = smallvec![false; stages.len()];
    select_next_stage(&mut order, &mut leaf_scans, 1, &stages, &binding_info);
    assert_eq!(order, InstrOrder::from_iter([0, 2, 1, 3].into_iter()));

    // Model the residuals after the chosen stage executes. The immediately
    // following choice must observe them even at physical position two.
    binding_info = binding_info_with_sizes(&[5, 66, 2]);
    select_next_stage(&mut order, &mut leaf_scans, 2, &stages, &binding_info);
    assert_eq!(order, InstrOrder::from_iter([0, 2, 3, 1].into_iter()));
}

#[test]
fn runtime_dvo_does_not_restart_live_size_priority_after_a_barrier() {
    let stages = vec![
        intersect_stage(0, 0),
        intersect_stage(0, 1),
        intersect_stage(1, 0),
        JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(0),
            mode: MatScanMode::Lookup(smallvec![Variable::from_usize(0)]),
            bind: SmallVec::new(),
            to_intersect: Vec::new(),
        },
        intersect_stage(0, 2),
        intersect_stage(1, 1),
    ];
    for start in [2, 3] {
        let binding_info = binding_info_with_sizes(&[65, 1]);
        let original = InstrOrder::from_iter([0, 1, 2, 3, 5, 4].into_iter());
        let mut order = original.clone();
        let mut leaf_scans = smallvec![false; stages.len()];

        // Whether the next position is the sole stage before the barrier or
        // the barrier itself, the later phase keeps its existing order. Even
        // the old speculative depth preference must not move atom 0 ahead.
        select_next_stage(&mut order, &mut leaf_scans, start, &stages, &binding_info);
        assert_eq!(order, original);
    }

    // Once the barrier has executed, the new phase can select its own next
    // stage from fresh residuals, without changing the executed prefix.
    let binding_info = binding_info_with_sizes(&[65, 1]);
    let mut order = InstrOrder::from_iter(0..stages.len());
    let mut leaf_scans = smallvec![false; stages.len()];
    select_next_stage(&mut order, &mut leaf_scans, 4, &stages, &binding_info);
    assert_eq!(order, InstrOrder::from_iter([0, 1, 2, 3, 5, 4].into_iter()));
}

fn prepared_for(stages: &[JoinStage]) -> PreparedJoinLayout {
    let mut access_counts = crate::numeric_id::DenseIdMap::new();
    let mut state_count = 0;
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
                    let state = PreparedIndexStateId::from_usize(state_count);
                    state_count += 1;
                    PreparedIndexSlot::new(kind, access, state)
                })
                .collect()
        })
        .collect();
    let tail_masks = PreparedTailMaskWidth::new(stages, &prepared_stages, access_counts.n_ids());
    PreparedJoinLayout {
        stages: prepared_stages,
        state_count,
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

/// Check the prepared masks against the exact scanner for every suffix of
/// every `order` and every atom in `0..atom_count`.
fn assert_tail_masks_match_scanner<M: StageMask>(
    stages: &[JoinStage],
    orders: &[Vec<usize>],
    atom_count: usize,
) {
    let prepared_layout = prepared_for(stages);
    let prepared = PreparedJoinIndexes::from_layout(&prepared_layout, &[]);
    let masks = prepared.tail_masks::<M>().unwrap();
    for order in orders {
        let instr_order = InstrOrder::from_iter(order.iter().copied());
        for resume_pos in 0..=order.len() {
            let remaining = suffix_stage_mask::<M>(&instr_order, resume_pos);
            for atom_index in 0..atom_count {
                let atom = AtomId::from_usize(atom_index);
                assert_eq!(
                    masks.atom_tail_use(atom, remaining),
                    scan_atom_tail_use(atom, stages, &prepared, &instr_order, resume_pos),
                    "tail metadata diverged for order {order:?}, suffix {resume_pos}, atom {atom_index}"
                );
            }
        }
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
    let mut orders = Vec::new();
    permutations(&mut [0, 1, 2, 3], 0, &mut orders);
    assert_tail_masks_match_scanner::<u64>(&stages, &orders, 4);
}

#[test]
fn wide_prepared_tail_masks_match_scanner_past_the_u64_boundary() {
    // 70 stages: atom 0 is probed at every stage, atoms 1 and 2 only beyond
    // bit 64, and stage 66 holds two accesses to atom 0 so the dynamic shape
    // is exercised in the high word.
    let mut stages = (0..70)
        .map(|column| intersect_stage(0, column))
        .collect::<Vec<_>>();
    stages[65] = intersect_stage(1, 0);
    stages[66] = JoinStage::Intersect {
        var: Variable::from_usize(66),
        scans: smallvec![
            SingleScanSpec {
                atom: AtomId::from_usize(0),
                column: ColumnId::from_usize(66),
                cs: Vec::new(),
            },
            SingleScanSpec {
                atom: AtomId::from_usize(0),
                column: ColumnId::from_usize(67),
                cs: Vec::new(),
            },
        ],
    };
    stages[69] = intersect_stage(2, 0);
    let identity = (0..stages.len()).collect::<Vec<_>>();
    let reversed = identity.iter().rev().copied().collect::<Vec<_>>();
    let rotated = identity[64..]
        .iter()
        .chain(&identity[..64])
        .copied()
        .collect::<Vec<_>>();
    assert_tail_masks_match_scanner::<u128>(&stages, &[identity, reversed, rotated], 3);
}

#[test]
fn prepared_tail_masks_pick_the_narrowest_width_that_fits() {
    fn stages(count: usize) -> Vec<JoinStage> {
        (0..count)
            .map(|column| intersect_stage(0, column))
            .collect()
    }

    let narrow_full_layout = prepared_for(&stages(64));
    let narrow_full = PreparedJoinIndexes::from_layout(&narrow_full_layout, &[]);
    assert!(!narrow_full.uses_wide_stage_mask());
    assert_eq!(narrow_full.all_stage_mask::<u64>(), Some(u64::MAX));
    assert!(narrow_full.tail_masks::<u128>().is_none());

    let wide_layout = prepared_for(&stages(65));
    let wide = PreparedJoinIndexes::from_layout(&wide_layout, &[]);
    assert!(wide.uses_wide_stage_mask());
    assert!(wide.tail_masks::<u64>().is_none());
    assert_eq!(wide.all_stage_mask::<u128>(), Some((1u128 << 65) - 1));

    let wide_full_layout = prepared_for(&stages(128));
    let wide_full = PreparedJoinIndexes::from_layout(&wide_full_layout, &[]);
    assert_eq!(wide_full.all_stage_mask::<u128>(), Some(u128::MAX));

    let over_layout = prepared_for(&stages(129));
    let over = PreparedJoinIndexes::from_layout(&over_layout, &[]);
    assert!(!over.uses_wide_stage_mask());
    assert!(over.tail_masks::<u64>().is_none());
    assert!(over.tail_masks::<u128>().is_none());
}

#[test]
fn packed_tail_shape_preserves_direct_graph_path() {
    let stages = vec![intersect_stage(0, 0), intersect_stage(0, 1)];
    let prepared_layout = prepared_for(&stages);
    let prepared = PreparedJoinIndexes::from_layout(&prepared_layout, &[]);
    let order = InstrOrder::from_iter(0..stages.len());

    assert_eq!(
        packed_child_shape_in_tail::<u64>(AtomId::from_usize(0), &stages, &prepared, &order, 1,),
        ChildShape::Direct
    );
    assert_eq!(
        packed_child_shape_in_tail::<u64>(AtomId::from_usize(0), &stages, &prepared, &order, 2,),
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
    let prepared_layout = prepared_for(&stages);
    let prepared = PreparedJoinIndexes::from_layout(&prepared_layout, &[]);
    let order = InstrOrder::from_iter([0, 2, 1].into_iter());

    assert_eq!(
        packed_child_shape_in_tail::<u64>(AtomId::from_usize(0), &stages, &prepared, &order, 1,),
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
    let prepared_layout = prepared_for(&stages);
    let prepared = PreparedJoinIndexes::from_layout(&prepared_layout, &[]);
    let order = InstrOrder::from_iter(0..stages.len());
    assert_eq!(
        packed_child_shape_in_tail::<u64>(atom, &stages, &prepared, &order, 1),
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
    let prepared_layout = prepared_for(&stages);
    let prepared = PreparedJoinIndexes::from_layout(&prepared_layout, &[]);
    let order = InstrOrder::from_iter(0..stages.len());
    assert_eq!(
        packed_child_shape_in_tail::<u64>(atom, &stages, &prepared, &order, 1),
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
        .map(|_| Arc::new(OwnedAtomRows::new(Subset::empty())))
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
        assert!(Arc::ptr_eq(cloned.owned_arc(), node));
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

use super::{
    TOP_INDEX_RANGES_PER_WORKER, TopLevelPartition, cover_scan_bounds, top_cover_partitions,
    top_index_range_partitions,
};

#[test]
fn fused_cover_partitions_are_complete_disjoint_and_coarse() {
    assert!(top_cover_partitions(10_000, 1, 16).is_none());
    assert!(top_cover_partitions(63, 4, 16).is_none());

    for (cover_rows, workers) in [(64, 4), (1_003, 4), (585_276, 12)] {
        let partitions = top_cover_partitions(cover_rows, workers, 16).unwrap();
        assert!(partitions.len() <= workers * 4);

        let mut expected_start = 0;
        for partition in &partitions {
            let TopLevelPartition::CoverRange { start, scan_size } = *partition else {
                panic!("fused covers must produce ordinal ranges")
            };
            assert_eq!(start, expected_start, "ranges must be contiguous");
            assert!(scan_size > 0);
            if start + scan_size != cover_rows {
                assert!(
                    scan_size >= 16,
                    "all non-tail partitions must remain coarse"
                );
            }
            expected_start += scan_size;
        }
        assert_eq!(expected_start, cover_rows, "ranges must cover every row");
    }
}

#[test]
fn fused_cover_scan_bounds_preserve_nonzero_subset_origins() {
    let partition = TopLevelPartition::CoverRange {
        start: 17,
        scan_size: 23,
    };
    let (start, len) = cover_scan_bounds(Some(partition), 100);
    assert_eq!(start.index(), 17);
    assert_eq!(len, 23);

    let (start, len) = cover_scan_bounds(None, 100);
    assert_eq!(start.index(), 0);
    assert_eq!(len, 100);
}

#[test]
fn filtered_index_ranges_are_complete_disjoint_and_coarse() {
    assert!(top_index_range_partitions(10_000, 1, 16).is_none());
    assert!(top_index_range_partitions(63, 4, 16).is_none());

    for (leader_keys, workers) in [(64, 4), (1_003, 4), (585_276, 12)] {
        let partitions = top_index_range_partitions(leader_keys, workers, 16).unwrap();
        assert!(partitions.len() <= workers * TOP_INDEX_RANGES_PER_WORKER);

        let mut expected_start = 0;
        for partition in &partitions {
            let TopLevelPartition::IndexRange { start, scan_size } = *partition else {
                panic!("an unsharded index must produce key-ordinal ranges")
            };
            assert_eq!(start, expected_start, "ranges must be contiguous");
            assert!(scan_size > 0);
            if start + scan_size != leader_keys {
                assert!(
                    scan_size >= 16,
                    "all non-tail partitions must remain coarse"
                );
            }
            expected_start += scan_size;
        }
        assert_eq!(expected_start, leader_keys, "ranges must cover every key");
    }
}

#[test]
fn row_handle_cardinality_tracks_each_storage_representation() {
    use crate::{
        free_join::{
            packed_trie::{PackedCursor, TrieNode},
            probe::AtomRows,
            residual_index::InlineRows,
        },
        offsets::{OffsetRange, RowId, SortedOffsetSlice, SubsetRef},
    };
    let arena = egglog_concurrency::SharedArena::new();
    let handle = arena.new_handle();
    let rows = [RowId::new_const(1), RowId::new_const(3)];
    // This deliberately has a gap: cardinality counts rows, not the span.
    let sparse = SubsetRef::Sparse(unsafe { SortedOffsetSlice::new_unchecked(&rows) });
    let pairs = [
        (Value::new_const(0), rows[0]),
        (Value::new_const(0), rows[1]),
    ];
    let packed = TrieNode::build_from_sorted_pairs(&handle, &pairs, ChildShape::Leaf, false);
    let range = OffsetRange::new(RowId::new_const(5), RowId::new_const(8));
    let cases = [
        AtomRows::owned(Arc::new(OwnedAtomRows::new(Subset::Dense(range)))),
        AtomRows::catalog(sparse, None),
        AtomRows::packed(PackedCursor::new(packed, 0)),
        AtomRows::inline(InlineRows::from_sorted(&rows)),
        AtomRows::dense(range),
        AtomRows::dense(OffsetRange::new(rows[0], rows[0])),
    ];
    for rows in cases {
        assert_eq!(rows.size(), rows.subset().size());
        assert_eq!(rows.is_empty(), rows.subset().size() == 0);
        let clone = rows.clone();
        drop(rows);
        assert_eq!(clone.size(), clone.subset().size());
    }
}
