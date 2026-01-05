use std::{collections::BTreeMap, iter, mem};

use crate::{
    numeric_id::{DenseIdMap, NumericId},
    query::SymbolMap,
};
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
pub(crate) struct Plan {
    pub atoms: DenseIdMap<AtomId, Atom>,
    pub stages: JoinStages,
}
impl Plan {
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

#[derive(Debug, Clone)]
pub(crate) struct JoinStages {
    pub header: Vec<JoinHeader>,
    pub instrs: Vec<JoinStage>,
    pub actions: ActionId,
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

pub(crate) fn plan_query(query: Query) -> Plan {
    let atoms = query.atoms;
    let ctx = PlanningContext {
        vars: query.var_info,
        atoms,
    };
    let (header, instrs) = plan_stages(&ctx, query.plan_strategy);

    Plan {
        atoms: ctx.atoms,
        stages: JoinStages {
            header,
            instrs,
            actions: query.action,
        },
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
struct PlanningContext {
    vars: DenseIdMap<Variable, VarInfo>,
    atoms: DenseIdMap<AtomId, Atom>,
}

/// Mutable state tracked during query planning.
#[derive(Clone)]
struct PlanningState {
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
    DenseIdMap<AtomId, (usize, &Pooled<Vec<Constraint>>)>,
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
