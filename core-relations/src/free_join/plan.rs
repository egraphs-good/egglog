use std::{collections::BTreeMap, iter, mem, sync::Arc};

use fixedbitset::FixedBitSet;
use numeric_id::{DenseIdMap, NumericId};
use smallvec::{smallvec, SmallVec};

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
#[derive(Debug)]
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
    Intersect {
        var: Variable,
        scans: SmallVec<[SingleScanSpec; 3]>,
    },
    FusedIntersect {
        cover: ScanSpec,
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
pub(crate) struct Plan {
    pub atoms: Arc<DenseIdMap<AtomId, Atom>>,
    pub stages: JoinStages,
}

#[derive(Debug, Clone)]
pub(crate) struct JoinStages {
    pub header: Vec<JoinHeader>,
    pub instrs: Arc<Vec<JoinStage>>,
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
    Planner::new(&query.var_info, &query.atoms).plan(query.plan_strategy, query.action)
}

struct Planner<'a> {
    // immutable
    vars: &'a DenseIdMap<Variable, VarInfo>,
    atoms: &'a DenseIdMap<AtomId, Atom>,

    // mutable
    used: VarSet,
    constrained: AtomSet,

    scratch_subatom: HashMap<AtomId, SmallVec<[ColumnId; 2]>>,
}

/// StageInfo is an intermediate stage used to describe the ordering of
/// operations. One of these contains enough information to "expand" it to a
/// JoinStage, but it still contains variable information.
///
/// This separation makes it easier for us to iterate with different planning
/// algorithms while sharing the same "backend" that generates a concrete plan.
struct StageInfo {
    cover: SubAtom,
    vars: SmallVec<[Variable; 1]>,
    filters: Vec<(
        SubAtom,                 /* the subatom to index */
        SmallVec<[ColumnId; 2]>, /* how to build a key for that index from the cover atom */
    )>,
}

impl<'a> Planner<'a> {
    pub(crate) fn new(
        vars: &'a DenseIdMap<Variable, VarInfo>,
        atoms: &'a DenseIdMap<AtomId, Atom>,
    ) -> Self {
        Planner {
            vars,
            atoms,
            used: VarSet::with_capacity(vars.n_ids()),
            constrained: AtomSet::with_capacity(atoms.n_ids()),
            scratch_subatom: Default::default(),
        }
    }

    fn plan_free_join(
        &mut self,
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
                let mut queue = BucketQueue::new(self.vars, self.atoms);
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
        while let Some(info) = self.get_next_freejoin_stage(&mut atoms) {
            stages.push(self.compile_stage(info))
        }
    }

    fn plan_gj(
        &mut self,
        remaining_constraints: &DenseIdMap<AtomId, (usize, &Pooled<Vec<Constraint>>)>,
        stages: &mut Vec<JoinStage>,
    ) {
        // First, map all variables to the size of the smallest atom in which they appear:
        let mut min_sizes = Vec::with_capacity(self.vars.n_ids());
        let mut atoms_hit = AtomSet::with_capacity(self.atoms.n_ids());
        for (var, var_info) in self.vars.iter() {
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
        for (var, var_info) in self.vars.iter() {
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
            let occ = self.vars[var].occurrences[0].clone();
            let mut info = StageInfo {
                cover: occ,
                vars: smallvec![var],
                filters: Default::default(),
            };
            for occ in &self.vars[var].occurrences[1..] {
                info.filters
                    .push((occ.clone(), smallvec![ColumnId::new(0)]));
            }
            let next_stage = self.compile_stage(info);
            if let Some(prev) = stages.last_mut() {
                if prev.fuse(&next_stage) {
                    continue;
                }
            }
            stages.push(next_stage);
        }
    }

    pub(crate) fn plan(&mut self, strat: PlanStrategy, actions: ActionId) -> Plan {
        let mut instrs = Vec::new();
        let mut header = Vec::new();
        self.used.clear();
        self.constrained.clear();
        let mut remaining_constraints: DenseIdMap<AtomId, (usize, &Pooled<Vec<Constraint>>)> =
            Default::default();
        // First, plan all the constants:
        for (atom, atom_info) in self.atoms.iter() {
            remaining_constraints.insert(
                atom,
                (
                    atom_info.constraints.approx_size(),
                    &atom_info.constraints.slow,
                ),
            );
            if atom_info.constraints.fast.is_empty() {
                continue;
            }
            header.push(JoinHeader {
                atom,
                constraints: Pooled::cloned(&atom_info.constraints.fast),
                subset: atom_info.constraints.subset.clone(),
            });
        }
        match strat {
            PlanStrategy::PureSize | PlanStrategy::MinCover => {
                self.plan_free_join(strat, &remaining_constraints, &mut instrs);
            }
            PlanStrategy::Gj => {
                self.plan_gj(&remaining_constraints, &mut instrs);
            }
        }
        Plan {
            atoms: self.atoms.clone().into(),
            stages: JoinStages {
                header,
                instrs: Arc::new(instrs),
                actions,
            },
        }
    }

    fn get_next_freejoin_stage(
        &mut self,
        ordering: &mut impl Iterator<Item = AtomId>,
    ) -> Option<StageInfo> {
        loop {
            let mut covered = false;
            let mut filters = Vec::new();
            let atom = ordering.next()?;
            let atom_info = &self.atoms[atom];
            let mut cover = SubAtom::new(atom);
            let mut vars = SmallVec::<[Variable; 1]>::new();
            for (ix, var) in atom_info.column_to_var.iter() {
                if self.used.contains(var.index()) {
                    continue;
                }
                // This atom is not completely covered by previous stages.
                covered = true;
                self.used.insert(var.index());
                vars.push(*var);
                cover.vars.push(ix);
                for subatom in self.vars[*var].occurrences.iter() {
                    if subatom.atom == atom {
                        continue;
                    }
                    self.scratch_subatom
                        .entry(subatom.atom)
                        .or_default()
                        .extend(subatom.vars.iter().copied());
                }
            }
            if !covered {
                // Search the next atom.
                continue;
            }
            for (atom, cols) in self.scratch_subatom.drain() {
                let mut form_key = SmallVec::<[ColumnId; 2]>::new();
                for var_ix in &cols {
                    let var = self.atoms[atom].column_to_var[*var_ix];
                    // form_key is an index _into the subatom forming the cover_.
                    let cover_col = vars
                        .iter()
                        .enumerate()
                        .find(|(_, v)| **v == var)
                        .map(|(ix, _)| ix)
                        .unwrap();
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

    fn compile_stage(
        &mut self,
        StageInfo {
            cover,
            vars,
            filters,
        }: StageInfo,
    ) -> JoinStage {
        if vars.len() == 1 {
            debug_assert!(
                filters
                    .iter()
                    .all(|(_, x)| x.len() == 1 && x[0] == ColumnId::new(0)),
                "filters={filters:?}"
            );
            let scans = SmallVec::<[SingleScanSpec; 3]>::from_iter(
                iter::once(&cover)
                    .chain(filters.iter().map(|(x, _)| x))
                    .map(|subatom| {
                        let atom = subatom.atom;
                        SingleScanSpec {
                            atom,
                            column: subatom.vars[0],
                            cs: if !self.constrained.put(atom.index()) {
                                self.atoms[atom].constraints.slow.clone()
                            } else {
                                Default::default()
                            },
                        }
                    }),
            );
            return JoinStage::Intersect {
                var: vars[0],
                scans,
            };
        }
        let atom = cover.atom;
        let cover = ScanSpec {
            to_index: cover,
            constraints: if !self.constrained.put(atom.index()) {
                self.atoms[atom].constraints.slow.clone()
            } else {
                Default::default()
            },
        };
        let mut bind = SmallVec::new();
        let var_set = &self.atoms[atom].var_to_column;
        for var in vars {
            bind.push((var_set[&var], var));
        }

        let mut to_intersect = Vec::with_capacity(filters.len());
        for (subatom, key_spec) in filters {
            let atom = subatom.atom;
            let scan = ScanSpec {
                to_index: subatom,
                constraints: if !self.constrained.put(atom.index()) {
                    self.atoms[atom].constraints.slow.clone()
                } else {
                    Default::default()
                },
            };
            to_intersect.push((scan, key_spec));
        }

        JoinStage::FusedIntersect {
            cover,
            bind,
            to_intersect,
        }
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
