//! This module defines query optimization for egglog. The main entry point is `plan_query`, which takes a `Query` and produces a `Plan`.
//!
//! At a high level, the query planner has two phases: **(hyper)tree decomposition** and **join planning for each bag**.
//! Both phases are very subtle, and heuristics are heavily used for good performance.
//!
//! # (Hyper)tree Decomposition
//!
//! A conjunctive query can be viewed as a hypergraph where variables are vertices and atoms (relations) are hyperedges.
//! The idea of tree decomposition is to break this hypergraph into a tree of overlapping subqueries called *bags*,
//! each of which is cheaper to evaluate independently. This is the classical idea behind tree decomposition and the
//! Yannakakis algorithm.
//!
//! The decomposition proceeds via *variable elimination*: we iteratively pick a variable `v` and eliminate the neighborhood
//! `N(v)` (which also includes `v`) from the hypergraph, and add back a hyperedge consisting of `N(v) - {v}`, until
//! there are no variables left. Each elimination step gives us a bag. A min-fill heuristic
//! (`next_var_to_eliminate`) guides the order of elimination to keep bags small. After all variables are eliminated,
//! redundant bags are pruned: bags subsumed by another (all their variables are covered) are merged, and "ears"
//! are merged into their parent.
//!
//! We then topologically sort the bags and decide which variables are "message variables" and which are private to the bag.
//! The materialized result of each bag has its output keyed on the *message variables* it shares with
//! its parent, and the parent uses that materialization to prune its own search space.
//!
//! When the query hypergraph is a single connected component with no beneficial decomposition, the planner falls back to
//! a `SinglePlan` with no materialization steps.
//!
//! # Join Planning for a Single Bag
//!
//! Once each bag (subquery) is isolated, the planner generates a sequence of `JoinStage` instructions that enumerate
//! all satisfying tuples for that bag. Two heuristics are supported:
//!
//! - **Generic Join** (`PlanStrategy::Gj`): The classic worst-case optimal join algorithm. Each stage picks one variable
//!   and intersects the columns of atoms that correspond to this variable (`JoinStage::Intersect`).
//!
//! - **Free Join** (`PlanStrategy::PureSize` / `PlanStrategy::MinCover`): From Remy's paper. The planning algorithm
//!   does the following: Each stage it selects a *cover* — a (sub)atom whose columns span the variables being bound in that step — and
//!   uses it to probe all other atoms that share those variables (`JoinStage::FusedIntersect`). When the cover is an
//!   entire atom and there is only one relation to probe, this degenerates to a hash join; when covers are single-column
//!   scans it ~ recovers generic join*.
//!
//!   *: although this is not worst-case optimal because it does not necessarily picks the smallest side to scan.
//!
//! Both strategies produce a flat list of `JoinStage` instructions that are fused where possible (`JoinStage::fuse`) to
//! reduce the number of passes over the data. A `JoinHeader` is prepended to each plan to apply constant constraints and
//! pre-filter the driving relation before the main join loop begins.
//!
use std::{collections::BTreeMap, iter, mem, sync::Arc};

use crate::{
    TableId,
    free_join::ProcessedConstraints,
    numeric_id::{DenseIdMap, NumericId},
    query::{FunDeps, SymbolMap},
};
use egglog_numeric_id::define_id;
use fixedbitset::FixedBitSet;
use smallvec::{SmallVec, smallvec};

use crate::{
    common::{HashMap, HashSet, IndexSet},
    offsets::Subset,
    pool::Pooled,
    query::{Atom, Query, VarColumnMap},
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
    Value(SmallVec<[Variable; 16]>),
    Lookup(SmallVec<[Variable; 16]>),
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
        to_intersect: Vec<(ScanSpec, SmallVec<[ColumnId; 2]>)>,
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

    pub(crate) fn to_report(
        &self,
        symbol_map: &SymbolMap,
        stage_stats: Option<&[egglog_reports::StageStats]>,
    ) -> egglog_reports::Plan {
        match self {
            Plan::SinglePlan(p) => p.to_report(symbol_map, stage_stats),
            Plan::DecomposedPlan(p) => p.to_report(symbol_map, stage_stats),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SinglePlan {
    pub atoms: Arc<DenseIdMap<AtomId, Atom>>,
    pub header: Vec<JoinHeader>,
    pub stages: JoinStages,
    pub actions: ActionId,
}

#[derive(Debug, Clone)]
pub(crate) struct JoinStages {
    pub instrs: Arc<Vec<JoinStage>>,
}

/// Specification of the materialization of the intermediate results, as required by tree decomposition.
/// A materialization has two parts. The message variables are variables that are passed to and joined with later stages,
/// and the value/private variables are variables that only occur in the current (and maybe previous) bags.
///
/// A materialization thus looks like a map from values of the message variables to sets of values of the private variables,
/// and when we evaluate other bags, only the keys (message variables) are looked up or enumerated. This is because
/// the private variables are not relevant to the evaluation of other bags. A key idea of tree decomposition is to separate
/// independent parts of a query and make sure they are evaluated independently.
#[derive(Debug, Clone)]
pub(crate) struct MatSpec {
    // Variables that are used by later stages
    pub msg_vars: SmallVec<[Variable; 16]>,
    // Variables that are not used by later stages.
    pub val_vars: SmallVec<[Variable; 16]>,
}

#[derive(Debug, Clone)]
pub(crate) struct JoinStageBlocks {
    // each block is a list of instructions and how to yield
    pub blocks: Vec<(JoinStages, MatSpec)>,
}

#[derive(Debug, Clone)]
pub(crate) struct DecomposedPlan {
    pub atoms: Arc<DenseIdMap<AtomId, Atom>>,
    pub header: Vec<JoinHeader>,
    pub stages: JoinStageBlocks,
    pub result_block: JoinStages,
    pub actions: ActionId,
}

impl SinglePlan {
    pub(crate) fn to_report(
        &self,
        symbol_map: &SymbolMap,
        stage_stats: Option<&[egglog_reports::StageStats]>,
    ) -> egglog_reports::Plan {
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
                                get_var(self.atoms[cover.to_index.atom].get_var(*col).unwrap());
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
                                    let var_name = get_var(
                                        self.atoms[scan.to_index.atom].get_var(*col).unwrap(),
                                    );
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
                JoinStage::FusedIntersectMat { .. } => {
                    unreachable!("FusedIntersectMat cannot appear in a SinglePlan")
                }
            };
            let next = if i == self.stages.instrs.len() - 1 {
                vec![]
            } else {
                vec![i + 1]
            };
            let stats = stage_stats.and_then(|s| s.get(i)).cloned();
            stages.push((report_stage, stats, next));
        }
        ReportPlan { stages }
    }
}

impl DecomposedPlan {
    pub(crate) fn to_report(
        &self,
        symbol_map: &SymbolMap,
        _stage_stats: Option<&[egglog_reports::StageStats]>,
    ) -> egglog_reports::Plan {
        use egglog_reports::{
            Plan as ReportPlan, Scan as ReportScan, SingleScan as ReportSingleScan,
            Stage as ReportStage,
        };
        const INTERNAL_PREFIX: &str = "@";
        let get_var = |var: Variable| -> String {
            symbol_map
                .vars
                .get(&var)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{INTERNAL_PREFIX}x{var:?}"))
        };
        let get_atom = |atom: AtomId| -> String {
            symbol_map
                .atoms
                .get(&atom)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{INTERNAL_PREFIX}R{atom:?}"))
        };

        let mut stages: Vec<(ReportStage, Option<egglog_reports::StageStats>, Vec<usize>)> =
            Vec::new();

        let convert_instrs =
            |instrs: &[JoinStage],
             stages: &mut Vec<(ReportStage, Option<egglog_reports::StageStats>, Vec<usize>)>| {
                let block_start = stages.len();
                let block_len = instrs.len();
                for (i, stage) in instrs.iter().enumerate() {
                    let report_stage = match stage {
                        JoinStage::Intersect { var, scans } => {
                            let var_name = get_var(*var);
                            let report_scans = scans
                                .iter()
                                .map(|scan| {
                                    ReportSingleScan(
                                        get_atom(scan.atom),
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
                                    let var_name = get_var(
                                        self.atoms[cover.to_index.atom].get_var(*col).unwrap(),
                                    );
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
                                            let var_name = get_var(
                                                self.atoms[scan.to_index.atom]
                                                    .get_var(*col)
                                                    .unwrap(),
                                            );
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
                            cover,
                            mode,
                            bind,
                            to_intersect: _,
                        } => {
                            let mode_str = match mode {
                                MatScanMode::Full => "Full",
                                MatScanMode::KeyOnly => "KeyOnly",
                                MatScanMode::Value(_) => "Value",
                                MatScanMode::Lookup(_) => "Lookup",
                            }
                            .to_string();
                            let vars: Vec<String> =
                                bind.iter().map(|(_, var)| get_var(*var)).collect();
                            ReportStage::MatLookup {
                                mat_id: cover.index(),
                                mode: mode_str,
                                vars,
                            }
                        }
                    };
                    let next = if i < block_len - 1 {
                        vec![block_start + i + 1]
                    } else {
                        vec![]
                    };
                    stages.push((report_stage, None, next));
                }
            };

        for (block_stages, _mat_spec) in &self.stages.blocks {
            convert_instrs(&block_stages.instrs, &mut stages);
        }
        convert_instrs(&self.result_block.instrs, &mut stages);

        ReportPlan { stages }
    }
}

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

/// Pick the next variable to eliminate and computes its neighborhood.
///
/// Each time, we pick a variable that has the least number of occurrences and find its neighborhood* (i.e.,
/// the set of variables that share an atom with it). We pick the neighborhood based on the "min-fill" heuristic,
/// which tries to eliminate neighborhood that would introduce the least number of new hyperedges.
/// A hyperedge is introduced during variable elimination if two variables that don't share an atom before are in the same neighborhood.
///
/// *: We find the closure of the neighborhood under functional dependencies, since these variables are "for free".
fn next_var_to_eliminate(
    vars: &DenseIdMap<Variable, VarInfo>,
    atoms: &DenseIdMap<AtomId, Atom>,
    fun_deps: &FunDeps,
) -> Option<IndexSet<Variable>> {
    let (_var, subquery_vars) = vars
        .iter()
        .map(|(var, _vinfo)| {
            let subquery_vars = atoms
                .iter()
                // every atom that contains this variable
                .filter(|(_, atom)| atom.get_col(var).is_some())
                // every variable of those atoms
                .flat_map(|(_, atom)| atom.vars());

            // Optimization: use functional dependencies to find all variables inferred by the
            // current neightborhood.
            let subquery_vars = fun_deps.closure(subquery_vars);

            let occ = atoms
                .iter()
                .filter(|(_, atom)| atom.vars().any(|v| subquery_vars.contains_key(v)))
                .count();
            (occ, var, subquery_vars)
        })
        .min_by_key(|a| a.0)
        .map(|a| (a.1, a.2))?;
    Some(IndexSet::from_iter(
        subquery_vars.into_iter().map(|(var, _)| var),
    ))
}

/// It updates the hypergraph with the given bag of variables by:
/// 1. Remove atoms that only contain variables in the bag and remove those atoms from variable's occurrences,
/// 2. Add a covering hyperedge that contains every non-private variable.
fn update_hypergraph(
    subquery_vars: &IndexSet<Variable>,
    vars: &mut DenseIdMap<Variable, VarInfo>,
    atoms: &mut DenseIdMap<AtomId, Atom>,
) {
    // Build the covering hyperedge before we remove from the hypergraph

    // Find variables that occur not just in the subquery
    let covering_vars: Vec<_> = subquery_vars
        .iter()
        .copied()
        .filter(|&var| {
            vars.contains_key(var)
                && vars[var].occurrences.iter().any(|occ| {
                    atoms[occ.atom]
                        .vars()
                        .any(|ov| !subquery_vars.contains(&ov))
                })
        })
        .collect();

    // Remove atoms from the hypergraph
    let mut removed = Vec::new();
    atoms.retain(|atom_id, atom| {
        if atom.vars().all(|var| subquery_vars.contains(&var)) {
            removed.push(atom_id);
            false
        } else {
            true
        }
    });

    // Update occurrences to reflect removed atoms
    for &subq_var in subquery_vars.iter() {
        if vars.contains_key(subq_var) {
            vars[subq_var]
                .occurrences
                .retain(|occ| !removed.contains(&occ.atom));

            if vars[subq_var].occurrences.is_empty() {
                vars.unwrap_val(subq_var);
            }
        }
    }

    // Add the covering atom to the hypergraph
    let mut var_columns = VarColumnMap::default();
    for (ix, var) in covering_vars.iter().enumerate() {
        var_columns.insert(*var, ColumnId::from_usize(ix));
    }
    let fake_atom_id = atoms.push(Atom {
        var_columns,
        constraints: ProcessedConstraints::dummy(),
        table: TableId::dummy(),
    });

    // Update variable occurrences to include the covering atom
    for (i, &covering_var) in covering_vars.iter().enumerate() {
        vars[covering_var].occurrences.push(SubAtom {
            atom: fake_atom_id,
            vars: smallvec![ColumnId::from_usize(i)],
        });
    }
}

/// This function does tree decomposition. At a high level, it takes a bag (equivalently, a `PlanningContext`, a subquery, a hypergraph,
/// or a set of variables + atoms), and returns a list of bags that forms a tree decomposition.
///
/// Recall that a bag is equivalent to a hypergraph, where vertices = variables and hyperedges = atoms.
///
/// The algorithm is based on the classical variable elimination, where it iteratively removes neighborhoods until no variables are left.
/// More specifically, it iteratively
///
/// 1. Select a variable `v` and its neighborhood `N(v)`, based on the "min-fill" heuristic. (`next_var_to_eliminate`)
/// 2. Remove the neighborhood from the working hypergraph. (`update_hypergraph`)
/// 3. Add a covering atom that contains variables `N(v) - {v}` to the working hypergraph. (`update_hypergraph`)
/// 4. Step 1-3 gives us a set of variables `N(v)`. We need to construct a subquery from it. This step is a bit subtle.
///
///    For example, consider the rectangle query `R(x, y), S(y, z), T(z, w), U(w, x)`. Let's say we pick variable `x` to eliminate.
///    The neighborhood `N(x)` of `x` is {x, y, z}. A naive approach is to subquery would be `R(x, y), S(y, z)`, but this query can have size quadratic,
///    even when the final output size is small. The issue here is `x` and `z` are not fully constrained in this subquery.
///    Another approach is to include every atom that contains variables in `N(x)`, but this gives us the entire query as the subquery for that rectangle query,
///    which is also not ideal because the rectangle query should be broken into two bags.
///
///    The solution is to include every atom that contains variables in `N(x)`, but only keep the variables in `N(x)` in those atoms. For the rectangle example,
///    this would be `R(x, y), S(y, z), T(z, -), U(-, x)`, where `-` means we don't expand this variable during evaluation. As a result, the produced PlanningContext
///    may have atoms whose variables are not in `PlanningContext::vars`. The query planner for a single bag handles this correctly.
///
/// Now we have collected a list of bags, but they are very redundant. (Remember the variable elimination loop is run |vars| steps, because each iteration eliminates
/// only one variable.) We need to prune these bags. See the comments in the code for details.
///
/// Another invariant we maintain is higher-indexed bags are heavier (closer to the root of the tree decomposition), so they will be evaluated later and constrained
/// by evaluation of earlier bags.
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
    while let Some(subquery_vars) = next_var_to_eliminate(&vars, &atoms, &original_ctx.fun_deps) {
        // Create a fake covering atom to bridge back to the main query
        // Remove hyperedges that only contain subquery variables.
        update_hypergraph(&subquery_vars, &mut vars, &mut atoms);

        // Collect atoms that only contain subquery variables.
        let subquery_atoms: DenseIdMap<AtomId, Atom> = original_ctx
            .atoms
            .iter()
            .filter(|(_, atom)| atom.vars().any(|var| subquery_vars.contains(&var)))
            .map(|(atom_id, atom)| (atom_id, atom.clone()))
            .collect();

        let subquery_var_map = DenseIdMap::from_iter(subquery_vars.iter().map(|var| {
            let mut var_info = original_ctx.vars[*var].clone();
            // NB: used_in_rhs is handled in [`plan_single_bag`]
            var_info
                .occurrences
                .retain(|occ| subquery_atoms.contains_key(occ.atom));
            (*var, var_info)
        }));

        bags.push(PlanningContext {
            vars: subquery_var_map,
            atoms: subquery_atoms,
            fun_deps: original_ctx.fun_deps.clone(),
        });
    }

    assert!(
        !atoms.iter().any(|(_, atom_info)| {
            !atom_info.table.is_dummy() && !atom_info.var_columns.is_empty()
        }),
        "All atoms should be put into bags"
    );

    // Iteratively prune the query
    let mut changed = true;
    while changed {
        changed = false;
        // Pruning 1: Remove bags that are subsumed by others. A bag is subsumed by another bag if all of its variables are contained in the other bag,
        // so the output of this bag must be a subset of the bigger bag.
        let mut pruned_bags: Vec<PlanningContext> = Vec::with_capacity(bags.len());
        for mut bag1 in bags.into_iter() {
            pruned_bags.retain_mut(|bag2| {
                let leq = bag1.is_subsumed_by(bag2);
                let geq = bag2.is_subsumed_by(&bag1);
                if leq || geq {
                    bag1.merge_bag(bag2);
                    changed = true;
                    false
                } else {
                    true
                }
            });
            pruned_bags.push(bag1);
        }

        // Pruning 2: Find "ears" and merge them with other bags. A bag is an ear if one of its atoms covers all of its variables, i.e., it only has one useful
        // relation. We can safely remove an ear if it shares variables with only one bag - in this case, that bag is necessarily the parent in the tree decomposition.
        //
        // Why removing ears? Let's say an ear has the form R(x, y, z) with message variable {x}. The evaluation of its parent will already intersect on `x` with `R(x, y, z)`,
        // so if `y` and `z` are expanded at the innermost loop of the evaluation, this does not incur any overhead. Versus if we keep this ear as a separate bag,
        // we would need to first build a map x -> (y, z) only to enumerate each x to get the corresponding (y, z) values.
        bags = pruned_bags;
        let is_ear = |bag: &PlanningContext| {
            bag.atoms.iter().any(|(_atom_id, atom)| {
                let all_vars = original_ctx.fun_deps.closure(atom.vars());
                bag.is_subsumed_by_vars(&all_vars)
            })
            // HACK: this weird condition says if there's exactly one atom whose variables are all wanted, then we can also treat it as an ear,
            // because other atoms in the bag are likely added only to constrain the bag. This is approximately what a bag is, but not really.
            // However, removing this condition makes some benchmark much worse...
            || bag
                .atoms
                .iter()
                .filter(|(_atom_id, atom)| bag.has_vars(atom.vars()))
                .count()
                == 1
        };

        let mut i = 0;
        while i < bags.len() {
            if !is_ear(&bags[i]) {
                i += 1;
                continue;
            }

            // Find the bag that shares the most variables with this ear bag, and merge the ear bag into it.
            let parent = bags
                .iter()
                .enumerate()
                .rev()
                .filter(|(j, _)| *j != i)
                .map(|(j, b)| (j, b.common_vars_with(&bags[i]).count()))
                .collect::<Vec<_>>();

            let j = parent.into_iter().max_by_key(|(_, count)| *count);
            if j.is_none() || j.unwrap().1 == 0 {
                i += 1;
                continue;
            }
            let j = j.unwrap().0;

            // Invariant: bigger-numbered bags are heavier and should stay at the root of the tree
            if i < j {
                let bag = mem::take(&mut bags[i]);
                bags[j].merge_bag(&bag);
                bags.remove(i);
            } else {
                let bag = mem::take(&mut bags[j]);
                bags[i].merge_bag(&bag);
                bags.remove(j);
                i += 1;
            }
            changed = true;
        }
    }
    bags
}

/// Topologically sorts bags based on variable dependencies.
///
/// This ensures that we evaluate bags in order.
///
/// This method also merges bags. The case where a bag has multiple children in the tree decomposition
/// has terrible performance currently, because all children but one has to be an innermost lookup.
/// So in this function, we merge all the non-first children into the parent. This improves HardBoiled benchmarks
/// significantly.
fn topologically_sort_bags(bags: Vec<PlanningContext>) -> Vec<PlanningContext> {
    let mut bags_opt = bags.into_iter().map(Some).collect::<Vec<_>>();
    let mut bags_topo = Vec::<PlanningContext>::with_capacity(bags_opt.len());
    let mut visited = vec![false; bags_opt.len()];
    let mut stack: Vec<(usize, Option<usize>)> = Vec::new();

    // Starting from the last, since early bags are more likely to be leaves and we don't
    // want a leafy bag to be a root.
    for i in (0..bags_opt.len()).rev() {
        if visited[i] {
            continue;
        }
        stack.push((i, None));
        visited[i] = true;

        while let Some((bag_id, parent)) = stack.pop() {
            let bag = mem::take(&mut bags_opt[bag_id]).unwrap();

            let this;
            if let Some(parent) = parent {
                bags_topo[parent].merge_bag(&bag);
                this = parent;
            } else {
                this = bags_topo.len();
            }

            let mut all_children: Vec<_> = bags_opt
                .iter()
                .enumerate()
                .filter_map(|(i, b)| Some((i, b.as_ref()?)))
                .map(|(i, b)| (i, b.common_vars_with(&bag).count()))
                .filter(|(i, count)| *count > 0 && !visited[*i])
                .collect();
            all_children.sort_unstable_by_key(|(_, count)| *count);

            if !all_children.is_empty() {
                visited[all_children[0].0] = true;
                stack.push((all_children[0].0, None));

                for &(i, _) in all_children[1..].iter() {
                    visited[i] = true;
                    stack.push((i, Some(this)))
                }
            }

            if parent.is_none() {
                bags_topo.push(bag);
            }
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
        for (var, _vinfo) in bag.vars.iter() {
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
/// - Adding prologue and epilogue instructions so that the bag is constrained by previous materializations.
///
/// This function also sets the `used_in_rhs` field for variables. A variable is not used in RHS during the planning
/// of a bag if it's not used in later bags.
fn plan_single_bag(
    bag: &mut PlanningContext,
    blocks: &[(JoinStages, MatSpec)],
    // If this bag has been used to prune its parent
    has_block_contributed: &mut [bool],
    n_used_in_bag: &mut DenseIdMap<Variable, usize>,
    strat: PlanStrategy,
) -> (Vec<JoinHeader>, JoinStages, MatSpec) {
    let mut msg_vars = smallvec![];
    let mut val_vars = smallvec![];

    // Classify variables as message or value variables
    for (var, vinfo) in bag.vars.iter_mut() {
        n_used_in_bag[var] -= 1;
        if n_used_in_bag[var] > 0 {
            // If this is a public variable, then we need to pass it on anyway
            vinfo.used_in_rhs = true;
            msg_vars.push(var);
        } else {
            // If this variable is not used in later and previous bag,
            // and it is not used in the right hand side,
            // this variable doesn't need to be expanded.
            if !vinfo.used_in_rhs
                && blocks.iter().all(|(_, spec)| !spec.msg_vars.contains(&var))
                && n_used_in_bag[var] == 0
            {
                continue;
            }
            val_vars.push(var);
            vinfo.used_in_rhs = true;
        }
    }

    let mut stripped_bag = bag.clone();

    // Add prologue and epilogue instructions to look up previous materialized bags
    // These are constraints from children blocks. If there's only one such block, it can be the header.
    // Otherwise, they have to be epilogue instructions doing filtering at the end, which is less efficient.
    let mut prologue = None;
    let mut epilogue = Vec::new();
    for (i, prev_block) in blocks.iter().enumerate().rev() {
        if prev_block.1.msg_vars.is_empty() {
            continue;
        }
        if !has_block_contributed[i]
            && prev_block
                .1
                .msg_vars
                .iter()
                .all(|var| bag.vars.contains_key(*var))
        {
            has_block_contributed[i] = true;
            if prologue.is_none() {
                let bind = prev_block
                    .1
                    .msg_vars
                    .iter()
                    .enumerate()
                    .map(|(j, var)| (ColumnId::from_usize(j), *var))
                    .collect();
                let mut to_intersect: Vec<(ScanSpec, SmallVec<[ColumnId; 2]>)> = vec![];
                for (col, var) in prev_block.1.msg_vars.iter().enumerate() {
                    let vinfo = &bag.vars[*var];
                    for occ in vinfo.occurrences.iter() {
                        let isect = match to_intersect
                            .iter_mut()
                            .find(|(spec, _)| spec.to_index.atom == occ.atom)
                        {
                            Some(isect) => isect,
                            None => {
                                to_intersect.push((
                                    ScanSpec {
                                        to_index: SubAtom {
                                            atom: occ.atom,
                                            vars: smallvec![],
                                        },
                                        constraints: vec![],
                                    },
                                    smallvec![],
                                ));
                                to_intersect.last_mut().unwrap()
                            }
                        };
                        isect.0.to_index.vars.extend(occ.vars.iter().copied());
                        isect
                            .1
                            .extend(occ.vars.iter().map(|_| ColumnId::from_usize(col)));
                    }
                }

                prologue = Some(JoinStage::FusedIntersectMat {
                    cover: MatId::from_usize(i),
                    mode: MatScanMode::KeyOnly,
                    bind,
                    to_intersect,
                });

                stripped_bag
                    .vars
                    .retain(|var, _vinfo| !prev_block.1.msg_vars.contains(&var));
            } else {
                epilogue.push(JoinStage::FusedIntersectMat {
                    cover: MatId::from_usize(i),
                    mode: MatScanMode::Lookup(prev_block.1.msg_vars.clone()),
                    bind: smallvec![],
                    to_intersect: vec![],
                });
            }
        }
    }

    let (header, mut instrs) = plan_stages(&stripped_bag, strat);
    instrs.splice(0..0, prologue);
    instrs.extend(epilogue);

    let stages = JoinStages {
        instrs: Arc::new(instrs),
    };

    (header, stages, MatSpec { msg_vars, val_vars })
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
        instrs: Arc::new(result_block),
    }
}

/// The last stage and the result block have the following structure:
///
/// for ...
///    yield [] -> x1, x2, ... as Mn
///
/// For x1, x2, ... in Mn:
///   ...
///
/// This can be fused into one loop
///
/// This is currently not used because somehow iterating the materialized RowBuffer is much faster than iterating the table
#[allow(unused)]
fn fuse_last_stage(
    mut blocks: Vec<(JoinStages, MatSpec)>,
    result_block: JoinStages,
) -> (Vec<(JoinStages, MatSpec)>, JoinStages) {
    if blocks.is_empty() {
        return (blocks, result_block);
    }

    let last_block = blocks.pop().unwrap();
    assert!(last_block.1.msg_vars.is_empty());
    if !matches!(
        result_block.instrs[0],
        JoinStage::FusedIntersectMat {
            cover,
            mode: MatScanMode::Full,
            ..
        } if cover == MatId::from_usize(blocks.len()
    )) {
        // If the first stage of the result block does not scan the last materialization
        return (blocks, result_block);
    }

    // Fuse the instructions
    let mut last_block = last_block.0;
    let mut instrs = Arc::unwrap_or_clone(last_block.instrs);
    instrs.extend(result_block.instrs[1..].iter().cloned());
    last_block.instrs = Arc::new(instrs);

    (blocks, last_block)
}

/// Eagerly lift materialization lookups up
///
/// For example, in the following, looking up of `r` can be lifted up before `z`
///
/// for x in R isec S:
///  R = R[x]; S = S[x]
///  for z in R:
///   if r in Mat[x]:
///     yield
fn loop_lifting(stages: JoinStages) -> JoinStages {
    let mut instrs = Arc::unwrap_or_clone(stages.instrs);
    for i in 1..instrs.len() {
        if let JoinStage::FusedIntersectMat {
            cover: _,
            mode: MatScanMode::Lookup(vars),
            bind,
            to_intersect,
        } = &instrs[i]
        {
            assert!(bind.is_empty() && to_intersect.is_empty());
            let vars = vars.clone();
            let mut j = i;
            while j > 0 {
                if matches!(
                    &instrs[j - 1], JoinStage::FusedIntersect { bind, .. } | JoinStage::FusedIntersectMat { bind, ..}
                        if bind.iter().all(|(_, var)| !vars.contains(var))
                ) || matches!(&instrs[j - 1], JoinStage::Intersect { var, .. } if !vars.contains(var))
                {
                    instrs.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }
    JoinStages {
        instrs: Arc::new(instrs),
    }
}

/// This is the main entry point for query optimization using tree decomposition.
pub(crate) fn tree_decompose_and_plan(
    ctx: PlanningContext,
    strat: PlanStrategy,
    actions: ActionId,
) -> Plan {
    macro_rules! fast_path {
        () => {{
            let (header, instrs) = plan_stages(&ctx, strat);
            let stages = JoinStages {
                instrs: Arc::new(instrs),
            };

            Plan::SinglePlan(SinglePlan {
                atoms: Arc::new(ctx.atoms),
                header,
                stages,
                actions,
            })
        }};
    }
    if ctx.atoms.len() <= 2 {
        return fast_path!();
    }

    // Step 1: Decompose the query into tree-structured bags
    let bags = decompose_into_bags(&ctx);
    if bags.len() <= 1 {
        // Don't do Yannakakis if it's just one bag
        return fast_path!();
    }

    // Step 2: Sort bags topologically and merge leafy bags with their parents
    let mut bags = topologically_sort_bags(bags);

    if bags.len() <= 1 {
        return fast_path!();
    }

    // Step 3: Count variable usage across bags. Used for deciding if a variable is public (i.e., message variables) or private.
    let mut n_used_in_bag = count_variable_usage_per_bag(&bags);
    let mut has_block_contributed = vec![false; bags.len()];

    // Step 4: Plan each bag and create materialization blocks
    let mut blocks = Vec::new();
    let mut header = vec![];
    for bag in bags.iter_mut() {
        let (bag_header, stages, mat_spec) = plan_single_bag(
            bag,
            &blocks,
            &mut has_block_contributed,
            &mut n_used_in_bag,
            strat,
        );
        blocks.push((stages, mat_spec));
        header.extend(bag_header);
    }

    // Step 5: Build the final result block
    let result_block = build_result_block(&blocks);

    // Optimization the avoids the last materialization
    // let (blocks, result_block) = fuse_last_stage(blocks, result_block);

    // Lifting variables
    let blocks = blocks
        .into_iter()
        .map(|(stages, mat_spec)| (loop_lifting(stages), mat_spec))
        .collect::<Vec<_>>();
    let result_block = loop_lifting(result_block);

    Plan::DecomposedPlan(DecomposedPlan {
        atoms: Arc::new(ctx.atoms),
        header,
        stages: JoinStageBlocks { blocks },
        result_block,
        actions,
    })
}

pub(crate) fn plan_query(query: Query) -> Plan {
    let atoms = query.atoms;
    let ctx = PlanningContext {
        vars: query.var_info,
        atoms,
        fun_deps: Arc::new(query.fun_deps),
    };
    tree_decompose_and_plan(ctx, query.plan_strategy, query.action)
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
#[derive(Debug, Clone, Default)]
pub(crate) struct PlanningContext {
    vars: DenseIdMap<Variable, VarInfo>,
    atoms: DenseIdMap<AtomId, Atom>,
    fun_deps: Arc<FunDeps>,
}

impl PlanningContext {
    fn is_subsumed_by(&self, bag2: &PlanningContext) -> bool {
        self.is_subsumed_by_vars(&bag2.vars)
    }

    fn is_subsumed_by_vars<I>(&self, bag2: &DenseIdMap<Variable, I>) -> bool {
        self.vars.iter().all(|(var, _)| bag2.contains_key(var))
    }

    fn merge_bag(&mut self, bag2: &PlanningContext) {
        for (var, vinfo) in bag2.vars.iter() {
            if self.vars.contains_key(var) {
                for new_occ in vinfo.occurrences.iter().cloned() {
                    if !self.vars[var]
                        .occurrences
                        .iter()
                        .any(|occ| occ.atom == new_occ.atom)
                    {
                        self.vars[var].occurrences.push(new_occ);
                    }
                }
            } else {
                self.vars.insert(var, vinfo.clone());
            }
        }
        for (atom_id, atom) in bag2.atoms.iter() {
            // atoms don't need to be merged
            if !self.atoms.contains_key(atom_id) {
                self.atoms.insert(atom_id, atom.clone());
            }
        }
    }

    fn common_vars_with<'a>(
        &'a self,
        other: &'a PlanningContext,
    ) -> impl Iterator<Item = Variable> + 'a {
        self.vars
            .iter()
            .filter(|(var, _)| other.vars.contains_key(*var))
            .map(|(var, _)| var)
    }

    fn has_vars(&self, mut vars: impl Iterator<Item = Variable>) -> bool {
        vars.all(|var| self.vars.contains_key(var))
    }
}

type VarSet = FixedBitSet;
type AtomSet = FixedBitSet;

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
            for var in atom.vars() {
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

        for (ix, var) in atom_info.var_columns.iter() {
            if state.is_var_used(var) {
                continue;
            }
            // This atom is not completely covered by previous stages.
            covered = true;
            state.mark_var_used(var);
            vars.push(var);
            cover.vars.push(ix);

            for subatom in ctx.vars[var].occurrences.iter() {
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
                let var = ctx.atoms[atom].get_var(*var_ix).unwrap();
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
        if let Some(prev) = stages.last_mut()
            && prev.fuse(&next_stage)
        {
            continue;
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
    for var in vars {
        bind.push((ctx.atoms[atom].get_col(var).unwrap(), var));
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
