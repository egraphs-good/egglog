use std::{collections::BTreeMap, iter, mem, sync::Arc};

use crate::{
    TableId,
    free_join::ProcessedConstraints,
    numeric_id::{DenseIdMap, NumericId},
    query::SymbolMap,
};
use egglog_numeric_id::define_id;
use fixedbitset::FixedBitSet;
use smallvec::{SmallVec, smallvec};

use crate::{
    common::{HashMap, HashSet, IndexSet},
    offsets::Subset,
    pool::Pooled,
    query::{Atom, Query},
    table_spec::Constraint,
};

use super::{ActionId, AtomId, ColumnId, SubAtom, VarInfo, Variable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScanSpec {
    pub to_index: SubAtom,
    // Only yield rows where the given constraints match.
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SingleScanSpec {
    pub atom: AtomId,
    pub column: ColumnId,
    pub cs: Vec<Constraint>,
}

define_id!(pub(crate) MatId, u32, "An identifier for materialization within a decomposed plan.");

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MatScanMode {
    Full,
    KeyOnly,
    Value(Vec<Variable>),
    Lookup(Vec<Variable>),
}

#[allow(unused)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ScanMatSpec {
    Scan(ScanSpec),
    Materialized(MatId),
}

/// Join headers evaluate constraints on a single atom; they prune the search space before the rest
/// of the join plan is executed.
pub(crate) struct JoinHeader {
    pub atom: AtomId,
    /// We currently aren't using these at all. The plan is to use this to
    /// dedup plan stages later (it also helps for debugging).
    #[allow(unused)]
    pub constraints: Pooled<Vec<Constraint>>,
    /// A pre-computed table subset that we can use to filter the table,
    /// given these constaints.
    ///
    /// Why use the constraints at all? Because we want to use them to
    /// discover common plan nodes from different queries (subsets can be
    /// large).
    pub subset: Subset,
}

impl std::fmt::Debug for JoinHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinHeader")
            .field("atom", &self.atom)
            .field("constraints", &self.constraints)
            .field(
                "subset",
                &format_args!("Subset(size={})", self.subset.size()),
            )
            .finish()
    }
}

impl Clone for JoinHeader {
    fn clone(&self) -> Self {
        JoinHeader {
            atom: self.atom,
            constraints: Pooled::cloned(&self.constraints),
            subset: self.subset.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum JoinStage {
    /// `Intersect` takes a variable and intersects a set of atoms
    /// on that variable.
    /// This corresponds to the classic generic join algorithm.
    Intersect {
        var: Variable,
        scans: SmallVec<[SingleScanSpec; 3]>,
    },
    /// `FusedIntersect` takes a "cover" (sub)atom and use it to probe other (sub)atoms.
    /// This corresponds to the free join algorithm, or when to_intersect.len() == 1 and cover is
    /// the entire atom, a hash join.
    FusedIntersect {
        cover: ScanSpec,
        bind: SmallVec<[(ColumnId, Variable); 2]>,
        // to_intersect.1 is the index into the cover atom.
        to_intersect: Vec<(ScanSpec, SmallVec<[ColumnId; 2]>)>,
    },
    FusedIntersectMat {
        cover: MatId,
        mode: MatScanMode,
        bind: SmallVec<[(ColumnId, Variable); 2]>,
        to_intersect: Vec<(ScanMatSpec, SmallVec<[ColumnId; 2]>)>,
    },
}

impl JoinStage {
    /// Attempt to fuse two stages into one.
    ///
    /// This operation is very conservative right now, it only fuses multiple
    /// scans that do no filtering whatsoever.
    fn fuse(&mut self, other: &JoinStage) -> bool {
        use JoinStage::*;
        match (self, other) {
            (
                FusedIntersect {
                    cover,
                    bind,
                    to_intersect,
                },
                Intersect { var, scans },
            ) if to_intersect.is_empty()
                && scans.len() == 1
                && cover.to_index.atom == scans[0].atom
                && scans[0].cs.is_empty() =>
            {
                let col = scans[0].column;
                bind.push((col, *var));
                cover.to_index.vars.push(col);
                true
            }
            (
                x,
                Intersect {
                    var: var2,
                    scans: scans2,
                },
            ) => {
                // This is all somewhat mangled because of the borrowing rules
                // when we pass &mut self into a tuple.
                let (var1, mut scans1) = if let Intersect {
                    var: var1,
                    scans: scans1,
                } = x
                {
                    if !(scans1.len() == 1
                        && scans2.len() == 1
                        && scans1[0].atom == scans2[0].atom
                        && scans2[0].cs.is_empty())
                    {
                        return false;
                    }
                    (*var1, mem::take(scans1))
                } else {
                    return false;
                };
                let atom = scans1[0].atom;
                let col1 = scans1[0].column;
                let col2 = scans2[0].column;
                *x = FusedIntersect {
                    cover: ScanSpec {
                        to_index: SubAtom {
                            atom,
                            vars: smallvec![col1, col2],
                        },
                        constraints: mem::take(&mut scans1[0].cs),
                    },
                    bind: smallvec![(col1, var1), (col2, *var2)],
                    to_intersect: Default::default(),
                };
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Plan {
    SinglePlan(SinglePlan),
    DecomposedPlan(DecomposedPlan),
}
impl Plan {
    pub fn actions(&self) -> ActionId {
        match self {
            Plan::SinglePlan(p) => p.actions,
            Plan::DecomposedPlan(p) => p.actions,
        }
    }

    pub fn atoms(&self) -> Arc<DenseIdMap<AtomId, Atom>> {
        match self {
            Plan::SinglePlan(p) => p.atoms.clone(),
            Plan::DecomposedPlan(p) => p.atoms.clone(),
        }
    }

    pub(crate) fn to_report(&self, _symbol_map: &SymbolMap) -> egglog_reports::Plan {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SinglePlan {
    pub atoms: Arc<DenseIdMap<AtomId, Atom>>,
    pub stages: JoinStages,
    pub actions: ActionId,
}

#[derive(Debug, Clone)]
pub(crate) struct JoinStages {
    pub header: Vec<JoinHeader>,
    pub instrs: Arc<Vec<JoinStage>>,
}

/// Specification of the materialization of the intermediate results, as required by tree decomposition.
#[derive(Debug, Clone)]
pub(crate) struct MatSpec {
    pub msg_vars: Vec<Variable>,
    pub val_vars: Vec<Variable>,
}

#[derive(Debug, Clone)]
pub(crate) struct JoinStageBlocks {
    // each block is a list of instructions and how to yield
    // TODO: Arc<MatSpec>
    pub blocks: Vec<(JoinStages, MatSpec)>,
}

#[derive(Debug, Clone)]
pub(crate) struct DecomposedPlan {
    pub atoms: Arc<DenseIdMap<AtomId, Atom>>,
    pub atom_to_bag: Arc<DenseIdMap<AtomId, usize>>,
    pub stages: JoinStageBlocks,
    pub result_block: JoinStages,
    pub actions: ActionId,
}

impl SinglePlan {
    pub(crate) fn to_report(&self, symbol_map: &SymbolMap) -> egglog_reports::Plan {
        use egglog_reports::{
            Plan as ReportPlan, Scan as ReportScan, SingleScan as ReportSingleScan,
            Stage as ReportStage,
        };
        const INTERNAL_PREFIX: &str = "@";
        let get_var = |var: Variable| {
            symbol_map
                .vars
                .get(&var)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{INTERNAL_PREFIX}x{var:?}"))
        };
        let get_atom = |atom: AtomId| {
            symbol_map
                .atoms
                .get(&atom)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{INTERNAL_PREFIX}R{atom:?}"))
        };
        let mut stages = Vec::new();
        for (i, stage) in self.stages.instrs.iter().enumerate() {
            let report_stage = match stage {
                JoinStage::Intersect { var, scans } => {
                    let var_name = get_var(*var);
                    let report_scans = scans
                        .iter()
                        .map(|scan| {
                            let atom_name = get_atom(scan.atom);
                            ReportSingleScan(
                                atom_name,
                                (var_name.clone(), scan.column.index() as i64),
                            )
                        })
                        .collect();
                    ReportStage::Intersect {
                        scans: report_scans,
                    }
                }
                JoinStage::FusedIntersect {
                    cover,
                    bind: _,
                    to_intersect,
                } => {
                    let cover_atom_name = get_atom(cover.to_index.atom);
                    let cover_cols: Vec<(String, i64)> = cover
                        .to_index
                        .vars
                        .iter()
                        .map(|col| {
                            let var_name =
                                get_var(self.atoms[cover.to_index.atom].column_to_var[*col]);
                            (var_name, col.index() as i64)
                        })
                        .collect();
                    let report_cover = ReportScan(cover_atom_name, cover_cols);
                    let report_to_intersect = to_intersect
                        .iter()
                        .map(|(scan, key_spec)| {
                            let atom_name = get_atom(scan.to_index.atom);
                            let cols: Vec<(String, i64)> = key_spec
                                .iter()
                                .map(|col| {
                                    let var_name =
                                        get_var(self.atoms[scan.to_index.atom].column_to_var[*col]);
                                    (var_name, col.index() as i64)
                                })
                                .collect();
                            ReportScan(atom_name, cols)
                        })
                        .collect();
                    ReportStage::FusedIntersect {
                        cover: report_cover,
                        to_intersect: report_to_intersect,
                    }
                }
                JoinStage::FusedIntersectMat {
                    cover: _,
                    mode: _,
                    bind: _,
                    to_intersect: _,
                } => {
                    todo!("materialization")
                }
            };
            let next = if i == self.stages.instrs.len() - 1 {
                vec![]
            } else {
                vec![i + 1]
            };
            stages.push((report_stage, None, next));
        }
        ReportPlan { stages }
    }
}

type VarSet = FixedBitSet;
type AtomSet = FixedBitSet;

/// The algorithm used to produce a join plan.
#[derive(Default, Copy, Clone)]
pub enum PlanStrategy {
    /// Free Join: Iteratively pick the smallest atom as the cover for the next
    /// stage, until all subatoms have been visited.
    PureSize,

    /// Free Join: Pick an approximate minimal set of covers, then order those
    /// covers in increasing order of size.
    ///
    /// This is similar to PureSize but we first limit the potential atoms that
    /// can act as covers so as to minimize the total number of stages in the
    /// plan. This is only an approximate minimum: the problem of finding the
    /// exact minimum ("set cover") is NP-hard.
    MinCover,

    /// Generate a plan for the classic Generic Join algorithm, constraining a
    /// single variable per stage.
    #[default]
    Gj,
}

type AtomToBag = DenseIdMap<AtomId, usize>;

/// Computes the next variable to eliminate and the subquery of its neighborhood.
///
/// Uses a min-fill heuristic that prioritizes variables whose elimination creates
/// neighborhoods that are more subsumed by existing bags. This is a generalization
/// of the "min-fill" heuristic to hypergraphs.
///
/// Variables occurring in only one atom are deprioritized until absolutely necessary
/// (i.e., when the atom is not joined with any other relation).
fn next_var_to_eliminate(
    vars: &DenseIdMap<Variable, VarInfo>,
    atoms: &DenseIdMap<AtomId, Atom>,
    bags: &[PlanningContext],
) -> Option<IndexSet<Variable>> {
    vars.iter()
        .map(|(var, vinfo)| {
            let subquery_vars: IndexSet<Variable> = atoms
                .iter()
                .filter(|(_, atom)| atom.column_to_var.iter().any(|(_, avar)| *avar == var))
                .flat_map(|(_, atom)| atom.column_to_var.iter().map(|(_, var)| *var))
                .collect::<IndexSet<_>>();
            let is_private = vinfo.occurrences.len() == 1;
            // At most how many variables of the proposed hyperedge are subsumed by any existing bag?
            let min_fill = subquery_vars.len()
                - bags
                    .iter()
                    .map(|bag| {
                        bag.vars
                            .iter()
                            .filter(|(v, _)| subquery_vars.contains(v))
                            .count()
                    })
                    .max()
                    .unwrap_or(0);
            (is_private, min_fill, subquery_vars)
        })
        .min_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)))
        .map(|(_is_private, _min_fill, subquery_vars)| subquery_vars)
}

/// Builds a fake covering atom that bridges the subquery back to the remaining main query.
///
/// The covering atom contains variables that appear in both the subquery and atoms
/// outside the subquery.
fn create_covering_atom(
    subquery_vars: &IndexSet<Variable>,
    vars: &mut DenseIdMap<Variable, VarInfo>,
    atoms: &mut DenseIdMap<AtomId, Atom>,
) -> AtomId {
    let covering_vars: Vec<_> = subquery_vars
        .iter()
        .filter(|var| {
            vars[**var].occurrences.iter().any(|occ| {
                atoms[occ.atom]
                    .var_to_column
                    .iter()
                    .any(|(ov, _)| !subquery_vars.contains(ov))
            })
        })
        .copied()
        .collect();

    let fake_atom_id = atoms.push(Atom {
        column_to_var: covering_vars
            .iter()
            .enumerate()
            .map(|(ix, var)| (ColumnId::from_usize(ix), *var))
            .collect(),
        var_to_column: covering_vars
            .iter()
            .enumerate()
            .map(|(ix, var)| (*var, ColumnId::from_usize(ix)))
            .collect(),
        constraints: ProcessedConstraints::dummy(),
        table: TableId::dummy(),
    });

    // Update variable occurrences to include the covering atom
    for &subquery_var in subquery_vars.iter() {
        if covering_vars.contains(&subquery_var) {
            vars[subquery_var].occurrences.push(SubAtom {
                atom: fake_atom_id,
                vars: smallvec![ColumnId::from_usize(
                    covering_vars
                        .iter()
                        .position(|v| *v == subquery_var)
                        .unwrap()
                )],
            });
        }
    }

    fake_atom_id
}

/// Remove variable occurrences from the remaining main query, returning the variable information for the subquery.
///
/// For each variable in the subquery, this function:
/// - Moves occurrences that are fully contained in the subquery to the subquery's VarInfo
/// - Removes those occurrences from the main query's VarInfo
/// - Prunes variables with no remaining occurrences from the main query
fn remove_occurrences(
    subquery_vars: &IndexSet<Variable>,
    vars: &mut DenseIdMap<Variable, VarInfo>,
    atoms: &DenseIdMap<AtomId, Atom>,
) -> DenseIdMap<Variable, VarInfo> {
    DenseIdMap::from_iter(subquery_vars.iter().filter_map(|&subq_var| {
        let vinfo = &vars[subq_var];
        let mut subquery_vinfo = VarInfo {
            occurrences: vec![],
            // TODO: this makes certain columns like timestamp and subsumed always used,
            // and undoes the used_in_rhs optimization that skips scanning these columns.
            // Maybe we can say if a variable is not used_in_rhs and comes up only in val columns,
            // then they can stay !used_in_rhs and the materialization won't include them.
            used_in_rhs: true,
            defined_in_rhs: vinfo.defined_in_rhs,
            name: vinfo.name.clone(),
        };

        // Separate occurrences into subquery and main query parts
        vars[subq_var].occurrences.retain_mut(|occ| {
            // If the variable only occurs in the current bag, we remove this variable from the main
            // query.
            if !atoms[occ.atom]
                .var_to_column
                .iter()
                .all(|(ov, _)| subquery_vars.contains(ov))
            {
                true
            } else {
                // TODO: CRITICAL: is this correct? We are not using the subquery from the tree decomposition?
                // Should move this to main so that it's planed based on subquery_atoms
                if !atoms[occ.atom].table.is_dummy() {
                    subquery_vinfo
                        .occurrences
                        .push(mem::replace(occ, SubAtom::dummy()));
                }
                false
            }
        });

        if vars[subq_var].occurrences.is_empty() {
            vars.unwrap_val(subq_var);
        }

        if !subquery_vinfo.occurrences.is_empty() {
            Some((subq_var, subquery_vinfo))
        } else {
            None
        }
    }))
}

/// Performs variable elimination to decompose the query into tree-structured bags.
///
/// This is the core tree decomposition algorithm. It iteratively:
/// 1. Selects the next variable to eliminate using a min-fill heuristic
/// 2. Creates a bag for the subquery of that variable's neighborhood
/// 3. Updates variable and atom information for the remaining main query
fn decompose_into_bags(original_ctx: &PlanningContext) -> Vec<PlanningContext> {
    let mut atoms = original_ctx.atoms.clone();
    let mut vars = original_ctx.vars.clone();

    // Prune variables with no occurrences
    for (var, vinfo) in original_ctx.vars.iter() {
        if vinfo.occurrences.is_empty() {
            vars.take(var).unwrap();
        }
    }

    let mut bags = Vec::new();

    // Variable elimination loop
    while let Some(subquery_vars) = next_var_to_eliminate(&vars, &atoms, &bags) {
        // Collect atoms that only contain subquery variables
        let subquery_atoms: IndexSet<AtomId> = original_ctx
            .atoms
            .iter()
            .filter(|(_, atom)| {
                atom.column_to_var
                    .iter()
                    .all(|(_, var)| subquery_vars.contains(var))
            })
            .map(|(atom_id, _)| atom_id)
            .collect();

        // Create a fake covering atom to bridge back to the main query
        create_covering_atom(&subquery_vars, &mut vars, &mut atoms);

        // Split variable occurrences and extract subquery variables
        let subquery_var_map = remove_occurrences(&subquery_vars, &mut vars, &atoms);

        // Extract subquery atoms
        let subquery_atom_map = DenseIdMap::from_iter(
            subquery_atoms
                .iter()
                .map(|&atom_id| (atom_id, original_ctx.atoms[atom_id].clone())),
        );

        // Remove subquery atoms from main query
        atoms.retain(|_, atom| {
            !atom
                .var_to_column
                .iter()
                .all(|(var, _)| subquery_var_map.contains_key(*var))
        });

        // Add bag if it's not already covered by an existing bag
        if !bags.iter().any(|bag| {
            subquery_var_map
                .iter()
                .all(|(var, _)| bag.vars.contains_key(var))
        }) {
            bags.push(PlanningContext {
                vars: subquery_var_map,
                atoms: subquery_atom_map,
            });
        }
    }

    assert!(
        atoms
            .iter()
            .filter(
                |(_, atom_info)| !atom_info.table.is_dummy() && !atom_info.var_to_column.is_empty()
            )
            .next()
            .is_none(),
        "All atoms should be put into bags"
    );

    bags
}

/// Topologically sorts bags based on variable dependencies.
///
/// A child bag depends on its parent if they share variables. The result is
/// ordered from leaves to root, enabling bottom-up processing.
fn topologically_sort_bags(bags: Vec<PlanningContext>) -> Vec<PlanningContext> {
    let mut bags_opt = bags.into_iter().map(Some).collect::<Vec<_>>();
    let mut bags_topo = Vec::with_capacity(bags_opt.len());
    let mut visited = vec![false; bags_opt.len()];
    let mut stack = Vec::new();

    for i in 0..bags_opt.len() {
        if visited[i] {
            continue;
        }
        stack.push(i);
        visited[i] = true;

        while !stack.is_empty() {
            let bag_id = stack.pop().unwrap();
            let bag = mem::take(&mut bags_opt[bag_id]).unwrap();

            // Find child bags that share variables with this bag
            for (i, child_bag) in bags_opt.iter().enumerate().filter(|(_, b)| b.is_some()) {
                let child_bag = child_bag.as_ref().unwrap();
                if child_bag
                    .vars
                    .iter()
                    .any(|(var, _)| bag.vars.contains_key(var))
                    && !visited[i]
                {
                    visited[i] = true;
                    stack.push(i);
                }
            }
            bags_topo.push(bag);
        }
    }

    bags_topo.reverse();
    bags_topo
}

/// Counts how many bags each variable appears in.
///
/// This is used to determine whether a variable should be passed as a message
/// variable (if used in later bags) or a value variable (if only used in the current bag).
fn count_variable_usage_per_bag(bags: &[PlanningContext]) -> DenseIdMap<Variable, usize> {
    let mut n_used_in_bag = DenseIdMap::new();
    for bag in bags {
        for (var, _) in bag.vars.iter() {
            if !n_used_in_bag.contains_key(var) {
                n_used_in_bag.insert(var, 0);
            }
            n_used_in_bag[var] += 1;
        }
    }
    n_used_in_bag
}

/// Plans the execution stages for a single bag.
///
/// This involves:
/// - Dividing variables into message variables (passed to later stages) and value variables
/// - Planning join stages within the bag
/// - Adding epilogue instructions to look up previous materialized bags
fn plan_single_bag(
    bag: &PlanningContext,
    blocks: &[(JoinStages, MatSpec)],
    n_used_in_bag: &mut DenseIdMap<Variable, usize>,
    strat: PlanStrategy,
) -> (JoinStages, MatSpec) {
    let mut msg_vars = vec![];
    let mut val_vars = vec![];

    // Classify variables as message or value variables
    for (var, _) in bag.vars.iter() {
        n_used_in_bag[var] -= 1;
        if n_used_in_bag[var] > 0 {
            msg_vars.push(var);
        } else {
            val_vars.push(var);
        }
    }

    let (header, mut instrs) = plan_stages(bag, strat);

    // Add epilogue instructions to look up previous materialized bags
    let mut epilogue = Vec::new();
    for (i, prev_block) in blocks.iter().enumerate().rev() {
        if prev_block
            .1
            .msg_vars
            .iter()
            .all(|var| bag.vars.contains_key(*var))
        {
            epilogue.push(JoinStage::FusedIntersectMat {
                cover: MatId::from_usize(i),
                mode: MatScanMode::Lookup(prev_block.1.msg_vars.clone()),
                bind: smallvec![],
                to_intersect: vec![],
            });
        }
    }
    instrs.extend(epilogue.into_iter());

    let stages = JoinStages {
        header,
        instrs: Arc::new(instrs),
    };

    (stages, MatSpec { msg_vars, val_vars })
}

/// Builds the final result block that collects results from all materialized bags.
///
/// This performs a bottom-up pass through the materialized bags, binding value
/// variables and gathering results. Each block is scanned at most once.
fn build_result_block(blocks: &[(JoinStages, MatSpec)]) -> JoinStages {
    let mut result_block = Vec::new();
    let mut pinned_vars = DenseIdMap::<Variable, ()>::new();

    for (i, (_stages, mat_spec)) in blocks.iter().enumerate().rev() {
        let to_bind: SmallVec<[(ColumnId, Variable); 2]> = mat_spec
            .val_vars
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, var)| !pinned_vars.contains_key(*var))
            .map(|(i, var)| (ColumnId::from_usize(i), var))
            .collect();

        if to_bind.is_empty() {
            continue;
        }

        for (_, var) in to_bind.iter() {
            pinned_vars.insert(*var, ());
        }

        result_block.push(JoinStage::FusedIntersectMat {
            cover: MatId::from_usize(i),
            // TODO: optimization to switch to MatScanMode::KeyOnly or MatScanMode::Value
            mode: if i == blocks.len() - 1 {
                MatScanMode::Full
            } else {
                MatScanMode::Value(mat_spec.msg_vars.clone())
            },
            bind: to_bind,
            to_intersect: vec![],
        });
    }

    JoinStages {
        header: vec![],
        instrs: Arc::new(result_block),
    }
}

pub(crate) fn tree_decompose_and_plan(
    ctx: &PlanningContext,
    strat: PlanStrategy,
) -> (
    /* Intermediate materializations */ Vec<(JoinStages, MatSpec)>,
    /* Final block that produces the final results */ JoinStages,
    /* Mapping from an atom to the bag it belongs to */ AtomToBag,
) {
    // Step 1: Decompose the query into tree-structured bags
    let bags = decompose_into_bags(ctx);
    let mut atom_to_bag = AtomToBag::new();

    // Step 2: Sort bags topologically
    let bags = topologically_sort_bags(bags);

    // Map atoms to their bag indices
    for (i, bag) in bags.iter().enumerate() {
        for (atom_id, _) in bag.atoms.iter() {
            atom_to_bag.insert(atom_id, i);
        }
    }

    // Step 3: Count variable usage across bags
    let mut n_used_in_bag = count_variable_usage_per_bag(&bags);

    // Step 4: Plan each bag and create materialization blocks
    let mut blocks = Vec::new();
    for bag in bags.iter() {
        let (stages, mat_spec) = plan_single_bag(bag, &blocks, &mut n_used_in_bag, strat);
        blocks.push((stages, mat_spec));
    }

    // Step 5: Build the final result block
    let result_block = build_result_block(&blocks);

    (blocks, result_block, atom_to_bag)
}

const TREE_DECOMPOSE: bool = true;

pub(crate) fn plan_query(query: Query) -> Plan {
    let atoms = query.atoms;
    let ctx = PlanningContext {
        vars: query.var_info,
        atoms,
    };
    if TREE_DECOMPOSE {
        let (blocks, result_block, atom_to_bag) =
            tree_decompose_and_plan(&ctx, query.plan_strategy);
        Plan::DecomposedPlan(DecomposedPlan {
            atoms: Arc::new(ctx.atoms),
            atom_to_bag: Arc::new(atom_to_bag),
            stages: JoinStageBlocks { blocks },
            result_block,
            actions: query.action,
        })
    } else {
        let (header, instrs) = plan_stages(&ctx, query.plan_strategy);
        let stages = JoinStages {
            header: header,
            instrs: Arc::new(instrs),
        };

        Plan::SinglePlan(SinglePlan {
            atoms: Arc::new(ctx.atoms),
            stages,
            actions: query.action,
        })
    }
}

/// StageInfo is an intermediate stage used to describe the ordering of
/// operations. One of these contains enough information to "expand" it to a
/// JoinStage, but it still contains variable information.
///
/// This separation makes it easier for us to iterate with different planning
/// algorithms while sharing the same "backend" that generates a concrete plan.
#[derive(Debug)]
struct StageInfo {
    cover: SubAtom,
    vars: SmallVec<[Variable; 1]>,
    filters: Vec<(
        SubAtom,                 /* the subatom to index */
        SmallVec<[ColumnId; 2]>, /* how to build a key for that index from the cover atom */
    )>,
}

/// Immutable context for query planning containing references to query metadata.
#[derive(Debug)]
pub(crate) struct PlanningContext {
    vars: DenseIdMap<Variable, VarInfo>,
    atoms: DenseIdMap<AtomId, Atom>,
}

/// Mutable state tracked during query planning.
#[derive(Clone)]
pub(crate) struct PlanningState {
    used_vars: VarSet,
    constrained_atoms: AtomSet,
}

impl PlanningState {
    fn new(n_vars: usize, n_atoms: usize) -> Self {
        Self {
            used_vars: VarSet::with_capacity(n_vars),
            constrained_atoms: AtomSet::with_capacity(n_atoms),
        }
    }

    fn mark_var_used(&mut self, var: Variable) {
        self.used_vars.insert(var.index());
    }

    fn is_var_used(&self, var: Variable) -> bool {
        self.used_vars.contains(var.index())
    }

    fn mark_atom_constrained(&mut self, atom: AtomId) {
        self.constrained_atoms.insert(atom.index());
    }

    fn is_atom_constrained(&self, atom: AtomId) -> bool {
        self.constrained_atoms.contains(atom.index())
    }
}

/// Datastructure used to greedily solve the set cover problem for a given free
/// join plan.
struct BucketQueue<'a> {
    var_info: &'a DenseIdMap<Variable, VarInfo>,
    cover: VarSet,
    atom_info: DenseIdMap<AtomId, VarSet>,
    sizes: BTreeMap<usize, IndexSet<AtomId>>,
}

impl<'a> BucketQueue<'a> {
    fn new(var_info: &'a DenseIdMap<Variable, VarInfo>, atoms: &DenseIdMap<AtomId, Atom>) -> Self {
        let cover = VarSet::with_capacity(var_info.n_ids());
        let mut atom_info = DenseIdMap::with_capacity(atoms.n_ids());
        let mut sizes = BTreeMap::<usize, IndexSet<AtomId>>::new();
        for (id, atom) in atoms.iter() {
            let mut bitset = VarSet::with_capacity(var_info.n_ids());
            for (_, var) in atom.column_to_var.iter() {
                bitset.insert(var.index());
            }
            sizes.entry(bitset.count_ones(..)).or_default().insert(id);
            atom_info.insert(id, bitset);
        }
        BucketQueue {
            var_info,
            cover,
            atom_info,
            sizes,
        }
    }

    /// Return the atom with the largest number of uncovered variables. A
    /// variable is "covered" if a previous call to `pop_min` returned an atom
    /// referencing that variable.
    fn pop_min(&mut self) -> Option<AtomId> {
        // Pick an arbitrary atom from the smallest bucket.
        let (_, atoms) = self.sizes.iter_mut().next_back()?;
        let res = atoms.pop().unwrap();
        let vars = self.atom_info[res].clone();
        // For each variable that we added to the cover, remove it from the
        // entries in atom_info referencing it and update `sizes` to reflect the
        // new ordering.
        for new_var in vars.difference(&self.cover).map(Variable::from_usize) {
            for subatom in &self.var_info[new_var].occurrences {
                let cur_set = &mut self.atom_info[subatom.atom];
                let old_size = cur_set.count_ones(..);
                cur_set.difference_with(&vars);
                let new_size = cur_set.count_ones(..);
                if old_size == new_size {
                    continue;
                }
                if let Some(old_size_set) = self.sizes.get_mut(&old_size) {
                    old_size_set.swap_remove(&subatom.atom);
                    if old_size_set.is_empty() {
                        self.sizes.remove(&old_size);
                    }
                }
                if new_size > 0 {
                    self.sizes.entry(new_size).or_default().insert(subatom.atom);
                }
            }
        }
        self.cover.union_with(&vars);
        Some(res)
    }
}

/// Build join headers from fast constraints and compute remaining constraints for planning.
/// Returns (headers, remaining_constraints) tuple.
fn plan_headers(
    ctx: &PlanningContext,
) -> (
    Vec<JoinHeader>,
    DenseIdMap<
        AtomId,
        (
            usize, /* The approx size of the subset matching the constraints. */
            &Pooled<Vec<Constraint>>,
        ),
    >,
) {
    let mut header = Vec::new();
    let mut remaining_constraints: DenseIdMap<AtomId, (usize, &Pooled<Vec<Constraint>>)> =
        Default::default();

    for (atom, atom_info) in ctx.atoms.iter() {
        remaining_constraints.insert(
            atom,
            (
                atom_info.constraints.approx_size(),
                &atom_info.constraints.slow,
            ),
        );
        if !atom_info.constraints.fast.is_empty() {
            header.push(JoinHeader {
                atom,
                constraints: Pooled::cloned(&atom_info.constraints.fast),
                subset: atom_info.constraints.subset.clone(),
            });
        }
    }

    (header, remaining_constraints)
}

/// Plan query execution stages using the specified strategy.
/// Returns (header, instructions) tuple that can be assembled into a Plan by the caller.
/// It does not directly return the plan because the caller may want to further modify the stages.
fn plan_stages(ctx: &PlanningContext, strat: PlanStrategy) -> (Vec<JoinHeader>, Vec<JoinStage>) {
    let (header, remaining_constraints) = plan_headers(ctx);
    let mut instrs = Vec::new();
    let mut state = PlanningState::new(ctx.vars.n_ids(), ctx.atoms.n_ids());

    match strat {
        PlanStrategy::PureSize | PlanStrategy::MinCover => {
            plan_free_join(ctx, &mut state, strat, &remaining_constraints, &mut instrs)
        }
        PlanStrategy::Gj => plan_gj(ctx, &mut state, &remaining_constraints, &mut instrs),
    };

    (header, instrs)
}

/// Plan free join queries using pure size or minimal cover strategy.
fn plan_free_join(
    ctx: &PlanningContext,
    state: &mut PlanningState,
    strat: PlanStrategy,
    remaining_constraints: &DenseIdMap<AtomId, (usize, &Pooled<Vec<Constraint>>)>,
    stages: &mut Vec<JoinStage>,
) {
    let mut size_info = Vec::<(AtomId, usize)>::new();

    match strat {
        PlanStrategy::PureSize => {
            for (atom, (size, _)) in remaining_constraints.iter() {
                size_info.push((atom, *size));
            }
        }
        PlanStrategy::MinCover => {
            let mut eligible_covers = HashSet::default();
            let mut queue = BucketQueue::new(&ctx.vars, &ctx.atoms);
            while let Some(atom) = queue.pop_min() {
                eligible_covers.insert(atom);
            }
            for (atom, (size, _)) in remaining_constraints
                .iter()
                .filter(|(atom, _)| eligible_covers.contains(atom))
            {
                size_info.push((atom, *size));
            }
        }
        PlanStrategy::Gj => unreachable!(),
    };

    size_info.sort_by_key(|(_, size)| *size);
    let mut atoms = size_info.iter().map(|(atom, _)| *atom);

    while let Some(info) = get_next_freejoin_stage(ctx, state, &mut atoms) {
        let stage = compile_stage(ctx, state, info);
        stages.push(stage);
    }
}

/// Generate the next free join stage by picking an atom from the ordering.
/// Returns the stage info and updated state, or None if all atoms are covered.
fn get_next_freejoin_stage(
    ctx: &PlanningContext,
    state: &mut PlanningState,
    ordering: &mut impl Iterator<Item = AtomId>,
) -> Option<StageInfo> {
    let mut scratch_subatom: HashMap<AtomId, SmallVec<[ColumnId; 2]>> = Default::default();

    loop {
        let mut covered = false;
        let atom = ordering.next()?;
        let atom_info = &ctx.atoms[atom];
        let mut cover = SubAtom::new(atom);
        let mut vars = SmallVec::<[Variable; 1]>::new();

        for (ix, var) in atom_info.column_to_var.iter() {
            if state.is_var_used(*var) {
                continue;
            }
            // This atom is not completely covered by previous stages.
            covered = true;
            state.mark_var_used(*var);
            vars.push(*var);
            cover.vars.push(ix);

            for subatom in ctx.vars[*var].occurrences.iter() {
                if subatom.atom == atom {
                    continue;
                }
                scratch_subatom
                    .entry(subatom.atom)
                    .or_default()
                    .extend(subatom.vars.iter().copied());
            }
        }

        if !covered {
            // Search the next atom.
            continue;
        }

        let mut filters = Vec::new();
        for (atom, cols) in scratch_subatom.drain() {
            let mut form_key = SmallVec::<[ColumnId; 2]>::new();
            for var_ix in &cols {
                let var = ctx.atoms[atom].column_to_var[*var_ix];
                // form_key is an index _into the subatom forming the cover_.
                let cover_col = vars.iter().position(|v| *v == var).unwrap();
                form_key.push(ColumnId::from_usize(cover_col));
            }
            filters.push((SubAtom { atom, vars: cols }, form_key));
        }

        return Some(StageInfo {
            cover,
            vars,
            filters,
        });
    }
}

/// Plan generic join queries (one variable per stage).
fn plan_gj(
    ctx: &PlanningContext,
    state: &mut PlanningState,
    remaining_constraints: &DenseIdMap<AtomId, (usize, &Pooled<Vec<Constraint>>)>,
    stages: &mut Vec<JoinStage>,
) {
    // First, map all variables to the size of the smallest atom in which they appear:
    let mut min_sizes = Vec::with_capacity(ctx.vars.n_ids());
    let mut atoms_hit = AtomSet::with_capacity(ctx.atoms.n_ids());
    for (var, var_info) in ctx.vars.iter() {
        let n_occs = var_info.occurrences.len();
        if n_occs == 1 && !var_info.used_in_rhs {
            // Do not plan this one. Unless (see below).
            continue;
        }
        if let Some(min_size) = var_info
            .occurrences
            .iter()
            .map(|subatom| {
                atoms_hit.set(subatom.atom.index(), true);
                remaining_constraints[subatom.atom].0
            })
            .min()
        {
            min_sizes.push((var, min_size, n_occs));
        }
        // If the variable has no ocurrences, it may be bound on the RHS of a
        // rule (or it may just be unused). Either way, we will ignore it when
        // planning the query.
    }
    for (var, var_info) in ctx.vars.iter() {
        if var_info.occurrences.len() == 1 && !var_info.used_in_rhs {
            // We skipped this variable the first time around because it
            // looks "unused". If it belongs to an atom that otherwise has
            // gone unmentioned, though, we need to plan it anyway.
            let atom = var_info.occurrences[0].atom;
            if !atoms_hit.contains(atom.index()) {
                min_sizes.push((var, remaining_constraints[atom].0, 1));
            }
        }
    }
    // Sort ascending by size, then descending by number of occurrences.
    min_sizes.sort_by_key(|(_, size, occs)| (*size, -(*occs as i64)));
    for (var, _, _) in min_sizes {
        let occ = ctx.vars[var].occurrences[0].clone();
        let mut info = StageInfo {
            cover: occ,
            vars: smallvec![var],
            filters: Default::default(),
        };
        for occ in &ctx.vars[var].occurrences[1..] {
            info.filters
                .push((occ.clone(), smallvec![ColumnId::new(0); occ.vars.len()]));
        }

        let next_stage = compile_stage(ctx, state, info);
        if let Some(prev) = stages.last_mut() {
            if prev.fuse(&next_stage) {
                continue;
            }
        }
        stages.push(next_stage);
    }
}

/// Compile a stage info into a concrete join stage, updating constraint state.
fn compile_stage(
    ctx: &PlanningContext,
    state: &mut PlanningState,
    StageInfo {
        cover,
        vars,
        filters,
    }: StageInfo,
) -> JoinStage {
    fn take_atom_constraints_if_new(
        ctx: &PlanningContext,
        state: &mut PlanningState,
        atom: AtomId,
    ) -> Vec<Constraint> {
        if state.is_atom_constrained(atom) {
            Default::default()
        } else {
            state.mark_atom_constrained(atom);
            ctx.atoms[atom].constraints.slow.clone()
        }
    }

    if vars.len() == 1 {
        let scans = SmallVec::<[SingleScanSpec; 3]>::from_iter(
            iter::once(&cover)
                .chain(filters.iter().map(|(x, _)| x))
                .map(|subatom| {
                    let atom = subatom.atom;
                    SingleScanSpec {
                        atom,
                        column: subatom.vars[0],
                        cs: take_atom_constraints_if_new(ctx, state, atom),
                    }
                }),
        );

        return JoinStage::Intersect {
            var: vars[0],
            scans,
        };
    }

    // FusedIntersect case
    let atom = cover.atom;

    let cover_spec = ScanSpec {
        to_index: cover,
        constraints: take_atom_constraints_if_new(ctx, state, atom),
    };

    let mut bind = SmallVec::new();
    let var_set = &ctx.atoms[atom].var_to_column;
    for var in vars {
        bind.push((var_set[&var], var));
    }

    let mut to_intersect = Vec::with_capacity(filters.len());
    for (subatom, key_spec) in filters {
        let atom = subatom.atom;
        let scan = ScanSpec {
            to_index: subatom,
            constraints: take_atom_constraints_if_new(ctx, state, atom),
        };
        to_intersect.push((scan, key_spec));
    }

    JoinStage::FusedIntersect {
        cover: cover_spec,
        bind,
        to_intersect,
    }
}
