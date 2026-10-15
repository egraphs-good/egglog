//! Completed bag results shared by seminaive variants during one ruleset run.

use std::{
    fmt::Write,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use web_time::Instant;

use crate::{
    common::{HashMap, IndexMap, Value},
    numeric_id::{DenseIdMap, NumericId},
    row_buffer::RowBuffer,
    table_spec::Constraint,
};

use super::{
    ActionId, AtomId, TableId, Variable,
    join_tail::for_each_stage_atom,
    plan::{JoinStage, JoinStageData, MatId, Plan},
};

type Materialization = IndexMap<Vec<Value>, RowBuffer>;

#[derive(Hash, PartialEq, Eq)]
struct BagKey {
    atoms: usize,
    stages: usize,
    msg_vars: Vec<Variable>,
    val_vars: Vec<Variable>,
    roots: Vec<(AtomId, TableId, Vec<Constraint>)>,
    // Child keys also distinguish a changed child program or output schema.
    children: Vec<usize>,
}

struct Slot {
    cacheable: bool,
    result: OnceLock<Arc<Materialization>>,
    stats: Option<Box<KeyStats>>,
}

#[derive(Default)]
struct Counters {
    attempts: AtomicU64,
    hits: AtomicU64,
    empty_hits: AtomicU64,
    builds: AtomicU64,
    empty_builds: AtomicU64,
    repeated_builds: AtomicU64,
    build_ns: AtomicU64,
    duplicate_build_ns: AtomicU64,
    result_keys: AtomicU64,
    result_rows: AtomicU64,
    result_payload_bytes: AtomicU64,
    repeated_result_keys: AtomicU64,
    repeated_result_rows: AtomicU64,
    repeated_result_payload_bytes: AtomicU64,
    max_result_rows: AtomicU64,
    max_result_payload_bytes: AtomicU64,
}

struct KeyInfo {
    occurrences: usize,
    action: ActionId,
    mat: MatId,
}

struct KeyStats {
    info: KeyInfo,
    counters: Counters,
}

#[derive(Default)]
struct PlannedStats {
    planned_decomposed: usize,
    planned_single: usize,
    planned_blocks: usize,
    distinct_keys: usize,
    repeated_keys: usize,
    repeated_occurrences: usize,
    direct_atom_refs: usize,
    transitive_atom_refs: usize,
}

/// Slots contain only completed results: a miss never waits for another join.
/// The caller must keep plans and the frozen database alive until this cache
/// is dropped, and must not publish results after early stopping.
pub(super) struct BagCache {
    enabled: bool,
    by_action: DenseIdMap<ActionId, Vec<Option<usize>>>,
    slots: Vec<Slot>,
    stats: Option<PlannedStats>,
}

impl BagCache {
    pub(super) fn new<'a>(plans: impl Iterator<Item = &'a Plan>) -> Self {
        Self::with_options(
            plans,
            std::env::var("EGGLOG_BAG_CACHE").as_deref() != Ok("off"),
            std::env::var("EGGLOG_BAG_STATS").as_deref() == Ok("1"),
        )
    }

    fn with_options<'a>(
        plans: impl Iterator<Item = &'a Plan>,
        enabled: bool,
        collect_stats: bool,
    ) -> Self {
        let mut cache = Self {
            enabled,
            by_action: DenseIdMap::new(),
            slots: Vec::new(),
            stats: collect_stats.then(PlannedStats::default),
        };
        if !enabled && !collect_stats {
            return cache;
        }

        let mut keys = HashMap::<BagKey, usize>::default();
        let mut info = Vec::<KeyInfo>::new();
        for plan in plans {
            let Plan::DecomposedPlan(plan) = plan else {
                if let Some(stats) = &mut cache.stats {
                    stats.planned_single += 1;
                }
                continue;
            };
            if let Some(stats) = &mut cache.stats {
                stats.planned_decomposed += 1;
            }
            let mut dependencies = Vec::<Vec<AtomId>>::with_capacity(plan.stages.blocks.len());
            let mut block_keys = Vec::<Option<usize>>::with_capacity(plan.stages.blocks.len());
            for (index, (stages, spec)) in plan.stages.blocks.iter().enumerate() {
                let mut atoms = Vec::new();
                let mut direct_atoms = Vec::new();
                let mut children = Vec::new();
                for stage in &stages.instrs {
                    for_each_stage_atom(stage, |atom| direct_atoms.push(atom));
                    if let JoinStage::FusedIntersectMat { cover, .. } = stage {
                        assert!(
                            cover.index() < index,
                            "bag dependencies must precede their users"
                        );
                        atoms.extend_from_slice(&dependencies[cover.index()]);
                        children.push(block_keys[cover.index()].unwrap());
                    }
                }
                direct_atoms.sort_unstable();
                direct_atoms.dedup();
                atoms.extend_from_slice(&direct_atoms);
                atoms.sort_unstable();
                atoms.dedup();
                if let Some(stats) = &mut cache.stats {
                    stats.direct_atom_refs += direct_atoms.len();
                    stats.transitive_atom_refs += atoms.len();
                }
                let roots = atoms
                    .iter()
                    .map(|atom| {
                        let mut constraints: Vec<_> = plan
                            .header
                            .iter()
                            .filter(|header| header.atom == *atom)
                            .flat_map(|header| header.constraints.iter().cloned())
                            .collect();
                        constraints.sort_unstable();
                        (*atom, plan.atoms[*atom].table, constraints)
                    })
                    .collect();
                let key = BagKey {
                    atoms: Arc::as_ptr(&plan.atoms) as usize,
                    stages: &**stages as *const JoinStageData as usize,
                    msg_vars: spec.msg_vars.to_vec(),
                    val_vars: spec.val_vars.to_vec(),
                    roots,
                    children,
                };
                let next = info.len();
                let key_id = *keys.entry(key).or_insert(next);
                if key_id == next {
                    info.push(KeyInfo {
                        occurrences: 0,
                        action: plan.actions,
                        mat: MatId::from_usize(index),
                    });
                }
                info[key_id].occurrences += 1;
                block_keys.push(Some(key_id));
                dependencies.push(atoms);
            }
            let previous = cache.by_action.insert(plan.actions, block_keys);
            assert!(previous.is_none(), "each plan must have its own action id");
        }
        let mut remap = Vec::with_capacity(info.len());
        for info in info {
            let repeated = info.occurrences > 1;
            if let Some(stats) = &mut cache.stats {
                stats.planned_blocks += info.occurrences;
                stats.distinct_keys += 1;
                if repeated {
                    stats.repeated_keys += 1;
                    stats.repeated_occurrences += info.occurrences;
                }
            }
            if repeated || collect_stats {
                remap.push(Some(cache.slots.len()));
                cache.slots.push(Slot {
                    cacheable: repeated,
                    result: OnceLock::new(),
                    stats: collect_stats.then(|| {
                        Box::new(KeyStats {
                            info,
                            counters: Counters::default(),
                        })
                    }),
                });
            } else {
                remap.push(None);
            }
        }
        for (_, blocks) in cache.by_action.iter_mut() {
            for slot in blocks {
                *slot = remap[slot.unwrap()];
            }
        }
        cache
    }

    fn slot(&self, action: ActionId, mat: MatId) -> Option<&Slot> {
        let index = *self.by_action.get(action)?.get(mat.index())?;
        Some(&self.slots[index?])
    }

    pub(super) fn get(&self, action: ActionId, mat: MatId) -> Option<Arc<Materialization>> {
        let slot = self.slot(action, mat)?;
        if let Some(stats) = &slot.stats {
            stats.counters.attempts.fetch_add(1, Ordering::Relaxed);
        }
        if !self.enabled || !slot.cacheable {
            return None;
        }
        let result = slot.result.get()?;
        if let Some(stats) = &slot.stats {
            stats.counters.hits.fetch_add(1, Ordering::Relaxed);
            if result.is_empty() {
                stats.counters.empty_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
        Some(result.clone())
    }

    pub(super) fn start_timer(&self) -> Option<Instant> {
        self.stats.as_ref().map(|_| Instant::now())
    }

    pub(super) fn finish(
        &self,
        action: ActionId,
        mat: MatId,
        result: &Materialization,
        started: Option<Instant>,
    ) {
        let Some(stats) = self.slot(action, mat).and_then(|slot| slot.stats.as_ref()) else {
            return;
        };
        let elapsed = started.map_or(0, |start| nanos(start.elapsed()));
        let counts = &stats.counters;
        let repeated = counts.builds.fetch_add(1, Ordering::Relaxed) > 0;
        let (keys, rows, bytes) = result_size(result);
        counts.build_ns.fetch_add(elapsed, Ordering::Relaxed);
        counts.result_keys.fetch_add(keys, Ordering::Relaxed);
        counts.result_rows.fetch_add(rows, Ordering::Relaxed);
        counts
            .result_payload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        counts.max_result_rows.fetch_max(rows, Ordering::Relaxed);
        counts
            .max_result_payload_bytes
            .fetch_max(bytes, Ordering::Relaxed);
        if result.is_empty() {
            counts.empty_builds.fetch_add(1, Ordering::Relaxed);
        }
        if repeated {
            counts.repeated_builds.fetch_add(1, Ordering::Relaxed);
            counts
                .duplicate_build_ns
                .fetch_add(elapsed, Ordering::Relaxed);
            counts
                .repeated_result_keys
                .fetch_add(keys, Ordering::Relaxed);
            counts
                .repeated_result_rows
                .fetch_add(rows, Ordering::Relaxed);
            counts
                .repeated_result_payload_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        }
    }

    pub(super) fn publish(
        &self,
        action: ActionId,
        mat: MatId,
        result: Arc<Materialization>,
    ) -> Arc<Materialization> {
        if self.enabled
            && let Some(slot) = self.slot(action, mat)
            && slot.cacheable
        {
            // A concurrent miss may have finished first. Only publishing the
            // already-built Arc is synchronized; construction never waits.
            let _ = slot.result.set(result);
            return slot.result.get().unwrap().clone();
        }
        result
    }

    #[allow(clippy::disallowed_macros)]
    pub(super) fn report(&self, elapsed: Duration) {
        let Some(planned) = &self.stats else {
            return;
        };
        let mut json = format!(
            "{{\"enabled\":{},\"elapsed_ns\":{},\"planned_decomposed\":{},\"planned_single\":{},\"planned_blocks\":{},\"distinct_keys\":{},\"repeated_keys\":{},\"repeated_occurrences\":{},\"eligible_reuses\":{},\"direct_atom_refs\":{},\"transitive_atom_refs\":{}",
            self.enabled,
            nanos(elapsed),
            planned.planned_decomposed,
            planned.planned_single,
            planned.planned_blocks,
            planned.distinct_keys,
            planned.repeated_keys,
            planned.repeated_occurrences,
            planned.repeated_occurrences - planned.repeated_keys,
            planned.direct_atom_refs,
            planned.transitive_atom_refs,
        );
        let all: Vec<_> = self
            .slots
            .iter()
            .filter_map(|slot| slot.stats.as_ref())
            .collect();
        macro_rules! sum {
            ($($field:ident),* $(,)?) => {$(
                let total: u64 = all.iter().map(|s| s.counters.$field.load(Ordering::Relaxed)).sum();
                write!(json, ",\"{}\":{}", stringify!($field), total).unwrap();
            )*};
        }
        sum!(
            attempts,
            hits,
            empty_hits,
            builds,
            empty_builds,
            repeated_builds,
            build_ns,
            duplicate_build_ns,
            result_keys,
            result_rows,
            result_payload_bytes,
            repeated_result_keys,
            repeated_result_rows,
            repeated_result_payload_bytes
        );
        let max_rows = all
            .iter()
            .map(|s| s.counters.max_result_rows.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);
        let max_bytes = all
            .iter()
            .map(|s| s.counters.max_result_payload_bytes.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);
        let (mut retained_keys, mut retained_rows, mut retained_bytes) = (0, 0, 0);
        for result in self.slots.iter().filter_map(|slot| slot.result.get()) {
            let (keys, rows, bytes) = result_size(result);
            retained_keys += keys;
            retained_rows += rows;
            retained_bytes += bytes;
        }
        write!(json, ",\"max_result_rows\":{max_rows},\"max_result_payload_bytes\":{max_bytes},\"retained_keys\":{retained_keys},\"retained_rows\":{retained_rows},\"retained_payload_bytes\":{retained_bytes},\"top_repeated_keys\":[").unwrap();
        let mut repeated: Vec<_> = all.into_iter().filter(|s| s.info.occurrences > 1).collect();
        repeated.sort_unstable_by_key(|s| {
            std::cmp::Reverse(s.counters.duplicate_build_ns.load(Ordering::Relaxed))
        });
        for (i, stats) in repeated.into_iter().take(8).enumerate() {
            if i > 0 {
                json.push(',');
            }
            write!(json, "{{\"first_action\":{},\"first_mat\":{},\"planned\":{},\"attempts\":{},\"hits\":{},\"builds\":{},\"build_ns\":{},\"duplicate_build_ns\":{}}}",
                stats.info.action.index(), stats.info.mat.index(), stats.info.occurrences,
                stats.counters.attempts.load(Ordering::Relaxed), stats.counters.hits.load(Ordering::Relaxed),
                stats.counters.builds.load(Ordering::Relaxed), stats.counters.build_ns.load(Ordering::Relaxed),
                stats.counters.duplicate_build_ns.load(Ordering::Relaxed)).unwrap();
        }
        json.push_str("]}");
        eprintln!("EGGLOG_BAG_STATS {json}");
    }
}

fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

// Logical payload, excluding container capacity, hashes, and allocation metadata.
fn result_size(result: &Materialization) -> (u64, u64, u64) {
    let mut rows = 0;
    let mut values = 0;
    for (key, buffer) in result {
        rows += buffer.len() as u64;
        values += (key.len() + buffer.len() * buffer.arity()) as u64;
    }
    (
        result.len() as u64,
        rows,
        values * size_of::<Value>() as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Subset,
        free_join::{
            ProcessedConstraints,
            plan::{
                DecomposedPlan, JoinHeader, JoinStageBlocks, JoinStages, MatScanMode, MatSpec,
                SingleScanSpec,
            },
        },
        pool::Pooled,
        query::Atom,
        table_shortcuts::v,
        table_spec::ColumnId,
    };
    use smallvec::smallvec;

    fn fixture() -> Plan {
        let mut atoms = DenseIdMap::new();
        for id in 0..3 {
            atoms.insert(
                AtomId::new(id),
                Atom {
                    table: TableId::new(id),
                    var_columns: Default::default(),
                    constraints: ProcessedConstraints::dummy(),
                },
            );
        }
        let scan = |atom| JoinStage::Intersect {
            var: Variable::new(atom),
            scans: smallvec![SingleScanSpec {
                atom: AtomId::new(atom),
                column: ColumnId::new(0),
                cs: vec![],
            }],
        };
        let spec = |var| MatSpec {
            msg_vars: smallvec![],
            val_vars: smallvec![Variable::new(var)],
        };
        Plan::DecomposedPlan(DecomposedPlan {
            atoms: Arc::new(atoms),
            header: vec![],
            stages: JoinStageBlocks {
                blocks: vec![
                    (JoinStages::new(vec![scan(0)]), spec(0)),
                    (
                        JoinStages::new(vec![
                            scan(1),
                            JoinStage::FusedIntersectMat {
                                cover: MatId::new(0),
                                mode: MatScanMode::Lookup(smallvec![Variable::new(1)]),
                                bind: smallvec![],
                                to_intersect: vec![],
                            },
                        ]),
                        spec(1),
                    ),
                    (JoinStages::new(vec![scan(2)]), spec(2)),
                ],
            },
            result_block: JoinStages::new(vec![]),
            actions: ActionId::new(0),
        })
    }

    fn variant(plan: &Plan, action: u32, headers: &[(u32, Vec<Constraint>)]) -> Plan {
        let mut plan = plan.clone();
        let Plan::DecomposedPlan(p) = &mut plan else {
            unreachable!()
        };
        p.actions = ActionId::new(action);
        p.header = headers
            .iter()
            .map(|(atom, constraints)| JoinHeader {
                atom: AtomId::new(*atom),
                constraints: Pooled::new(constraints.clone()),
                subset: Subset::empty().into(),
            })
            .collect();
        plan
    }

    fn output() -> Arc<Materialization> {
        let mut rows = RowBuffer::new(1);
        rows.add_row(&[v(7)]);
        rows.add_row(&[v(7)]);
        Arc::new([(vec![v(3)], rows)].into_iter().collect())
    }

    #[test]
    fn keys_follow_lookup_dependencies_but_ignore_unrelated_headers() {
        let base = fixture();
        let lo = Constraint::GeConst {
            col: ColumnId::new(0),
            val: v(2),
        };
        let hi = Constraint::LtConst {
            col: ColumnId::new(0),
            val: v(8),
        };
        let a = variant(&base, 0, &[(0, vec![lo.clone(), hi.clone()])]);
        let b = variant(
            &base,
            1,
            &[(0, vec![hi, lo.clone()]), (2, vec![lo.clone()])],
        );
        let c = variant(&base, 2, &[(0, vec![lo])]);
        let cache = BagCache::with_options([&a, &b, &c].into_iter(), true, true);
        for mat in [MatId::new(0), MatId::new(1)] {
            let result = cache.publish(ActionId::new(0), mat, output());
            assert!(Arc::ptr_eq(
                &result,
                &cache.get(ActionId::new(1), mat).unwrap()
            ));
            assert!(cache.get(ActionId::new(2), mat).is_none());
        }
        assert_ne!(
            cache.by_action[ActionId::new(0)][2],
            cache.by_action[ActionId::new(1)][2]
        );
        assert_eq!(
            cache.by_action[ActionId::new(0)][2],
            cache.by_action[ActionId::new(2)][2]
        );
    }

    #[test]
    fn keys_include_program_atoms_schema_and_child_program() {
        let base = fixture();
        let same = variant(&base, 1, &[]);
        let mut new_atoms = variant(&base, 2, &[]);
        let Plan::DecomposedPlan(plan) = &mut new_atoms else {
            unreachable!()
        };
        plan.atoms = Arc::new((*plan.atoms).clone());
        let mut new_schema = variant(&base, 3, &[]);
        let Plan::DecomposedPlan(plan) = &mut new_schema else {
            unreachable!()
        };
        plan.stages.blocks[0].1.val_vars.push(Variable::new(9));
        let mut new_child = variant(&base, 4, &[]);
        let Plan::DecomposedPlan(plan) = &mut new_child else {
            unreachable!()
        };
        plan.stages.blocks[0].0 = JoinStages::new(plan.stages.blocks[0].0.instrs.clone());
        let cache = BagCache::with_options(
            [&base, &same, &new_atoms, &new_schema, &new_child].into_iter(),
            true,
            true,
        );
        for mat in 0..2 {
            let first = cache.by_action[ActionId::new(0)][mat];
            assert_eq!(first, cache.by_action[ActionId::new(1)][mat]);
            for action in 2..5 {
                assert_ne!(first, cache.by_action[ActionId::new(action)][mat]);
            }
        }
    }

    #[test]
    fn completed_results_preserve_rows_and_empty_results_and_do_not_cross_runs() {
        let base = fixture();
        let second = variant(&base, 1, &[]);
        let cache = BagCache::with_options([&base, &second].into_iter(), true, false);
        let mat = MatId::new(0);
        assert!(cache.get(ActionId::new(0), mat).is_none());
        let result = output();
        let winner = cache.publish(ActionId::new(0), mat, result.clone());
        let concurrent = cache.publish(ActionId::new(1), mat, output());
        assert!(Arc::ptr_eq(&winner, &concurrent));
        assert!(Arc::ptr_eq(
            &result,
            &cache.get(ActionId::new(1), mat).unwrap()
        ));
        assert_eq!(
            winner[&vec![v(3)]].iter().collect::<Vec<_>>(),
            vec![&[v(7)][..], &[v(7)][..]]
        );
        let empty = Arc::new(Materialization::default());
        cache.publish(ActionId::new(0), MatId::new(1), empty.clone());
        assert!(Arc::ptr_eq(
            &empty,
            &cache.get(ActionId::new(1), MatId::new(1)).unwrap()
        ));
        let fresh = BagCache::with_options([&base, &second].into_iter(), true, false);
        assert!(fresh.get(ActionId::new(0), mat).is_none());
        assert!(fresh.start_timer().is_none());
        let unique = BagCache::with_options([&base].into_iter(), true, false);
        assert!(unique.slots.is_empty());
    }

    #[test]
    fn disabled_cache_counts_duplicate_work_without_retaining_results() {
        let base = fixture();
        let second = variant(&base, 1, &[]);
        let cache = BagCache::with_options([&base, &second].into_iter(), false, true);
        let mat = MatId::new(0);
        for action in [ActionId::new(0), ActionId::new(1)] {
            assert!(cache.get(action, mat).is_none());
            let result = output();
            cache.finish(action, mat, &result, cache.start_timer());
            assert!(Arc::ptr_eq(
                &result,
                &cache.publish(action, mat, result.clone())
            ));
        }
        let counts = &cache
            .slot(ActionId::new(0), mat)
            .unwrap()
            .stats
            .as_ref()
            .unwrap()
            .counters;
        assert_eq!(counts.attempts.load(Ordering::Relaxed), 2);
        assert_eq!(counts.builds.load(Ordering::Relaxed), 2);
        assert_eq!(counts.repeated_builds.load(Ordering::Relaxed), 1);
        assert_eq!(counts.result_rows.load(Ordering::Relaxed), 4);
        assert_eq!(counts.repeated_result_rows.load(Ordering::Relaxed), 2);
        assert_eq!(
            counts.result_payload_bytes.load(Ordering::Relaxed),
            6 * size_of::<Value>() as u64
        );
        assert!(
            cache
                .slot(ActionId::new(0), mat)
                .unwrap()
                .result
                .get()
                .is_none()
        );
    }
}
