use crate::ast::FunctionSubtype;
use crate::termdag::{TermDag, TermId};
use crate::util::{HashMap, HashSet};
use crate::*;
use std::collections::VecDeque;

/// An interface for custom cost model.
///
/// To use it with the default extractor, the cost type must also satisfy `Ord + Eq + Clone + Debug`.
/// Additionally, the cost model should guarantee that a term has a no-smaller cost
/// than its subterms to avoid cycles in the extracted terms for common case usages.
/// For more niche usages, a term can have a cost less than its subterms.
/// As long as there is no negative cost cycle,
/// the default extractor is guaranteed to terminate in computing the costs.
/// However, the user needs to be careful to guarantee acyclicity in the extracted terms.
pub trait CostModel<C: Cost> {
    /// The total cost of a term given the cost of the root e-node and its immediate children's total costs.
    fn fold(&self, head: &str, children_cost: &[C], head_cost: C) -> C;

    /// The cost of an enode (without the cost of children)
    fn enode_cost(&self, egraph: &EGraph, func: &Function, enode: &Enode<'_>) -> C;

    /// The cost of a container value given the costs of its elements.
    ///
    /// The default cost for containers is just the sum of all the elements inside
    fn container_cost(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        element_costs: &[C],
    ) -> C {
        let _egraph = egraph;
        let _sort = sort;
        let _value = value;
        element_costs
            .iter()
            .fold(C::identity(), |s, c| s.combine(c))
    }

    /// Compute the cost of a (non-container) primitive value.
    ///
    /// The default cost for base values is the constant one
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C {
        let _egraph = egraph;
        let _sort = sort;
        let _value = value;
        C::unit()
    }
}

/// Requirements for a type to be usable as a cost by a [`CostModel`].
pub trait Cost {
    /// An identity element, usually zero.
    fn identity() -> Self;

    /// The default cost for a node with no children, usually one.
    fn unit() -> Self;

    /// A binary operation to combine costs, usually addition.
    /// This operation must NOT overflow or panic when given large values!
    fn combine(self, other: &Self) -> Self;
}

macro_rules! cost_impl_int {
    ($($cost:ty),*) => {$(
        impl Cost for $cost {
            fn identity() -> Self { 0 }
            fn unit()     -> Self { 1 }
            fn combine(self, other: &Self) -> Self {
                self.saturating_add(*other)
            }
        }
    )*};
}
cost_impl_int!(u8, u16, u32, u64, u128, usize);
cost_impl_int!(i8, i16, i32, i64, i128, isize);

macro_rules! cost_impl_num {
    ($($cost:ty),*) => {$(
        impl Cost for $cost {
            fn identity() -> Self {
                use num::Zero;
                Self::zero()
            }
            fn unit() -> Self {
                use num::One;
                Self::one()
            }
            fn combine(self, other: &Self) -> Self {
                self + other
            }
        }
    )*};
}
cost_impl_num!(num::BigInt, num::BigRational);
use ordered_float::OrderedFloat;
cost_impl_num!(f32, f64, OrderedFloat<f32>, OrderedFloat<f64>);

pub type DefaultCost = u64;

/// A cost model that computes the cost by summing the cost of each node.
#[derive(Default, Clone)]
pub struct TreeAdditiveCostModel {}

impl CostModel<DefaultCost> for TreeAdditiveCostModel {
    fn fold(
        &self,
        _head: &str,
        children_cost: &[DefaultCost],
        head_cost: DefaultCost,
    ) -> DefaultCost {
        children_cost.iter().fold(head_cost, |s, c| s.combine(c))
    }

    fn enode_cost(&self, egraph: &EGraph, func: &Function, _enode: &Enode<'_>) -> DefaultCost {
        func.extraction_head_cost(egraph)
    }
}

/// The default, Bellman-Ford like extractor. This extractor is optimal for [`CostModel`].
///
/// Note that this assumes optimal substructure in the cost model, that is, a lower-cost
/// subterm should always lead to a non-worse superterm, to guarantee the extracted term
/// being optimal under the given cost model.
/// If this is not followed, the extractor may panic on reconstruction
pub struct Extractor<C: Cost + Ord + Eq + Clone + Debug> {
    rootsorts: Vec<ArcSort>,
    funcs: Vec<String>,
    cost_model: Box<dyn CostModel<C>>,
    /// Dense id assigned to each eq sort that some extractable function outputs;
    /// indexes into `costs`, `topo_rnk`, and `parent_edge`.
    sort_ids: HashMap<String, usize>,
    costs: Vec<HashMap<Value, C>>,
    topo_rnk_cnt: usize,
    topo_rnk: Vec<HashMap<Value, usize>>,
    parent_edge: Vec<HashMap<Value, (String, Vec<Value>)>>,
}

/// How extraction treats one child column of a function, resolved once so the
/// per-row cost loops avoid repeated sort dispatch and name-keyed lookups.
enum ChildKind {
    /// An eq-sort column; `Some` holds the sort's dense id, `None` means no
    /// extractable function outputs this sort (its terms have no cost).
    EqSort(Option<usize>),
    Container(ArcSort),
    Base(ArcSort),
}

/// Per-function data for [`Extractor::bellman_ford`]: resolved schema facts
/// plus a flat materialized copy of the table's non-subsumed rows, so each
/// relaxation pass iterates memory instead of re-scanning the table.
struct FuncData {
    name: String,
    /// Row width in `rows`; 0 iff the table had no (non-subsumed) rows.
    arity: usize,
    output_idx: usize,
    output_sort_id: usize,
    child_kinds: Vec<ChildKind>,
    rows: Vec<Value>,
}

impl<C: Cost + Ord + Eq + Clone + Debug> Extractor<C> {
    /// Bulk of the computation happens at initialization time.
    /// The later extractions only reuses saved results.
    /// This means a new extractor must be created if the egraph changes.
    /// Holding a reference to the egraph would enforce this but prevents the extractor being reused.
    ///
    /// For convenience, if the rootsorts is `None`, it defaults to extract all extractable rootsorts.
    pub fn compute_costs_from_rootsorts(
        rootsorts: Option<Vec<ArcSort>>,
        egraph: &EGraph,
        cost_model: impl CostModel<C> + 'static,
    ) -> Self {
        // We filter out tables unreachable from the root sorts
        let extract_all_sorts = rootsorts.is_none();

        let mut rootsorts = rootsorts.unwrap_or_default();

        // Built a reverse index from output sort to function head symbols
        // Only include constructors (not regular functions), and respect the user-facing
        // hidden and unextractable flags.
        let mut rev_index: HashMap<String, Vec<String>> = Default::default();
        for func in egraph.functions.iter() {
            let unextractable = func.1.decl.unextractable;
            let hidden = func.1.decl.internal_hidden;

            // Only extract constructors and view tables, which reconstruct as their
            // term_constructor. Proof extraction uses its own root-directed extractor
            // and does not need alternate behavior here.
            if !unextractable
                && !hidden
                && (func.1.decl.subtype == FunctionSubtype::Constructor
                    || func.1.decl.term_constructor.is_some())
            {
                let func_name = func.0.clone();
                // For view tables (with term_constructor in proof mode), the e-class is the last input column
                let output_sort_name = func.1.extraction_output_sort().name();
                if let Some(v) = rev_index.get_mut(output_sort_name) {
                    v.push(func_name);
                } else {
                    rev_index.insert(output_sort_name.to_owned(), vec![func_name]);
                    if extract_all_sorts {
                        rootsorts.push(func.1.extraction_output_sort().clone());
                    }
                }
            }
        }

        // Do a BFS to find reachable tables
        let mut q: VecDeque<ArcSort> = VecDeque::new();
        let mut seen: HashSet<String> = Default::default();
        for rootsort in rootsorts.iter() {
            q.push_back(rootsort.clone());
            seen.insert(rootsort.name().to_owned());
        }

        let mut funcs_set: HashSet<String> = Default::default();
        let mut funcs: Vec<String> = Vec::new();
        while !q.is_empty() {
            let sort = q.pop_front().unwrap();
            if sort.is_container_sort() {
                let inner_sorts = sort.inner_sorts();
                for s in inner_sorts {
                    if !seen.contains(s.name()) {
                        q.push_back(s.clone());
                        seen.insert(s.name().to_owned());
                    }
                }
            } else if sort.is_eq_sort()
                && let Some(head_symbols) = rev_index.get(sort.name())
            {
                for h in head_symbols {
                    if !funcs_set.contains(h) {
                        let func = egraph.functions.get(h).unwrap();
                        // For view tables, children are all but the last input (which is the e-class)
                        let num_children = func.extraction_num_children();
                        for ch in func.schema.input.iter().take(num_children) {
                            let ch_name = ch.name();
                            if !seen.contains(ch_name) {
                                q.push_back(ch.clone());
                                seen.insert(ch_name.to_owned());
                            }
                        }
                        funcs_set.insert(h.clone());
                        funcs.push(h.clone());
                    }
                }
            }
        }

        // Initialize the tables to have the reachable entries
        let mut sort_ids: HashMap<String, usize> = Default::default();
        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            let output_sort_name = func.extraction_output_sort().name();
            let next_id = sort_ids.len();
            sort_ids
                .entry(output_sort_name.to_owned())
                .or_insert(next_id);
        }
        let n_sorts = sort_ids.len();

        let mut extractor = Extractor {
            rootsorts,
            funcs,
            cost_model: Box::new(cost_model),
            sort_ids,
            costs: (0..n_sorts).map(|_| Default::default()).collect(),
            topo_rnk_cnt: 0,
            topo_rnk: (0..n_sorts).map(|_| Default::default()).collect(),
            parent_edge: (0..n_sorts).map(|_| Default::default()).collect(),
        };

        extractor.bellman_ford(egraph);

        extractor
    }

    /// Compute the cost of a single enode
    /// Recurse if container
    /// Returns None if contains an undefined eqsort term (potentially after unfolding)
    fn compute_cost_node(&self, egraph: &EGraph, value: Value, sort: &ArcSort) -> Option<C> {
        if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.container_values(), value);
            let mut ch_costs: Vec<C> = Vec::new();
            for ch in elements.iter() {
                ch_costs.push(self.compute_cost_node(egraph, ch.1, &ch.0)?);
            }
            Some(
                self.cost_model
                    .container_cost(egraph, sort, value, &ch_costs),
            )
        } else if sort.is_eq_sort() {
            self.costs[*self.sort_ids.get(sort.name())?]
                .get(&value)
                .cloned()
        } else {
            // Primitive
            Some(self.cost_model.base_value_cost(egraph, sort, value))
        }
    }

    /// A row in a constructor table is a hyperedge from the set of input terms to the constructed output term.
    fn compute_cost_hyperedge(
        &self,
        egraph: &EGraph,
        row: &egglog_bridge::ScanEntry,
        func: &Function,
    ) -> Option<C> {
        let mut ch_costs: Vec<C> = Vec::new();
        let sorts = &func.schema.input;
        let num_children = func.extraction_num_children();
        for (value, sort) in row.vals.iter().take(num_children).zip(sorts.iter()) {
            ch_costs.push(self.compute_cost_node(egraph, *value, sort)?);
        }
        let head_name = func.extraction_term_name();
        let output_idx = func.extraction_output_index();
        let enode = Enode {
            children: &row.vals[..output_idx],
            eclass: row.vals[output_idx],
            subsumed: row.subsumed,
        };
        Some(self.cost_model.fold(
            head_name,
            &ch_costs,
            self.cost_model.enode_cost(egraph, func, &enode),
        ))
    }

    fn compute_topo_rnk_node(&self, egraph: &EGraph, value: Value, sort: &ArcSort) -> usize {
        if sort.is_container_sort() {
            sort.inner_values(egraph.backend.container_values(), value)
                .iter()
                .fold(0, |ret, (sort, value)| {
                    usize::max(ret, self.compute_topo_rnk_node(egraph, *value, sort))
                })
        } else if sort.is_eq_sort() {
            if let Some(id) = self.sort_ids.get(sort.name()) {
                *self.topo_rnk[*id].get(&value).unwrap_or(&usize::MAX)
            } else {
                usize::MAX
            }
        } else {
            0
        }
    }

    /// We use Bellman-Ford to compute the costs of the relevant eq sorts' terms
    /// [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) is a shortest path algorithm.
    /// The version implemented here computes the shortest path from any node in a set of sources to all the reachable nodes.
    /// Computing the minimum cost for terms is treated as a shortest path problem on a hypergraph here.
    /// In this hypergraph, the nodes corresponde to eclasses, the distances are the costs to extract a term of those eclasses,
    /// and each enode is a hyperedge that goes from the set of children eclasses to the enode's eclass.
    /// The sources are the eclasses with known costs from the cost model.
    /// Additionally, to avoid cycles in the extraction even when the cost model can assign an equal cost to a term and its subterm.
    /// It computes a topological rank for each eclass
    /// and only allows each eclass to have children of classes of strictly smaller ranks in the extraction.
    /// Compute the child costs of one materialized row into `ch_costs`, and fold them
    /// into the row's total cost. Returns `None` if any child is uncomputed so far.
    fn row_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        f: &FuncData,
        row: &[Value],
        ch_costs: &mut Vec<C>,
    ) -> Option<C> {
        ch_costs.clear();
        for (kind, value) in f.child_kinds.iter().zip(row.iter()) {
            let cost = match kind {
                ChildKind::EqSort(Some(id)) => self.costs[*id].get(value)?.clone(),
                ChildKind::EqSort(None) => return None,
                ChildKind::Container(sort) => self.compute_cost_node(egraph, *value, sort)?,
                ChildKind::Base(sort) => self.cost_model.base_value_cost(egraph, sort, *value),
            };
            ch_costs.push(cost);
        }
        let enode = Enode {
            children: &row[..f.output_idx],
            eclass: row[f.output_idx],
            subsumed: false,
        };
        Some(self.cost_model.fold(
            func.extraction_term_name(),
            ch_costs,
            self.cost_model.enode_cost(egraph, func, &enode),
        ))
    }

    /// Report every (sort id, value) pair a container value's cost depends on:
    /// the eq values stored anywhere inside it, found by recursing through
    /// nested container values.
    fn register_container_deps(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        register: &mut impl FnMut(usize, Value),
    ) {
        if sort.is_container_sort() {
            for (inner_sort, inner_value) in
                sort.inner_values(egraph.backend.container_values(), value)
            {
                self.register_container_deps(egraph, &inner_sort, inner_value, register);
            }
        } else if sort.is_eq_sort()
            && let Some(id) = self.sort_ids.get(sort.name())
        {
            register(*id, value);
        }
    }

    fn bellman_ford(&mut self, egraph: &EGraph) {
        // Materialize each function's non-subsumed rows once, so the relaxation
        // passes below iterate plain memory instead of re-scanning every table.
        let func_data: Vec<FuncData> = self
            .funcs
            .iter()
            .map(|func_name| {
                let func = egraph.functions.get(func_name).unwrap();
                let num_children = func.extraction_num_children();
                let child_kinds = func.schema.input[..num_children]
                    .iter()
                    .map(|sort| {
                        if sort.is_container_sort() {
                            ChildKind::Container(sort.clone())
                        } else if sort.is_eq_sort() {
                            ChildKind::EqSort(self.sort_ids.get(sort.name()).copied())
                        } else {
                            ChildKind::Base(sort.clone())
                        }
                    })
                    .collect();
                let mut arity = 0;
                let mut rows = Vec::new();
                egraph.backend.for_each(func.backend_id, |row| {
                    if !row.subsumed {
                        arity = row.vals.len();
                        rows.extend_from_slice(row.vals);
                    }
                });
                FuncData {
                    name: func_name.clone(),
                    arity,
                    output_idx: func.extraction_output_index(),
                    output_sort_id: self.sort_ids[func.extraction_output_sort().name()],
                    child_kinds,
                    rows,
                }
            })
            .collect();

        // Reverse dependency index: (sort id, value) -> rows reading that value
        // as an eq child, directly or inside a container child.
        let mut child_index: HashMap<(usize, Value), Vec<(u32, u32)>> = Default::default();
        for (fi, f) in func_data.iter().enumerate() {
            if f.rows.is_empty() {
                continue;
            }
            for (ri, row) in f.rows.chunks_exact(f.arity).enumerate() {
                for (kind, value) in f.child_kinds.iter().zip(row.iter()) {
                    let mut register = |sid: usize, v: Value| {
                        child_index
                            .entry((sid, v))
                            .or_default()
                            .push((fi as u32, ri as u32));
                    };
                    match kind {
                        ChildKind::EqSort(Some(id)) => register(*id, *value),
                        ChildKind::EqSort(None) | ChildKind::Base(_) => {}
                        ChildKind::Container(sort) => {
                            self.register_container_deps(egraph, sort, *value, &mut register)
                        }
                    }
                }
            }
        }

        // Semi-naive relaxation: a row recomputes exactly the same cost against a
        // never-increasing target unless one of its child (sort, value) costs
        // changed since the row was last evaluated, so only such "dirty" rows are
        // (re)visited. Rows are swept in the same (function, row) order as the
        // naive pass loop, so the update trace — and hence topo ranks and
        // extracted terms — is unchanged.
        let mut dirty: Vec<Vec<bool>> = func_data
            .iter()
            .map(|f| {
                vec![
                    true;
                    if f.arity == 0 {
                        0
                    } else {
                        f.rows.len() / f.arity
                    }
                ]
            })
            .collect();
        let mut dirty_count: Vec<usize> = dirty.iter().map(|d| d.len()).collect();

        let mut ch_costs: Vec<C> = Vec::new();
        loop {
            let mut any = false;
            for (fi, f) in func_data.iter().enumerate() {
                if dirty_count[fi] == 0 {
                    continue;
                }
                any = true;
                let func = egraph.functions.get(&f.name).unwrap();
                // Marks set at or behind the sweep cursor (including a row
                // re-marking itself) stay for the next sweep; marks ahead of it
                // are picked up in this one, matching the naive pass exactly.
                let n_rows = dirty[fi].len();
                let mut ri = 0;
                while ri < n_rows {
                    if !dirty[fi][ri] {
                        ri += 1;
                        continue;
                    }
                    dirty[fi][ri] = false;
                    dirty_count[fi] -= 1;
                    let row = &f.rows[ri * f.arity..(ri + 1) * f.arity];
                    ri += 1;
                    let Some(new_cost) = self.row_cost(egraph, func, f, row, &mut ch_costs) else {
                        continue;
                    };
                    let target = row[f.output_idx];
                    let updated = match self.costs[f.output_sort_id].entry(target) {
                        HEntry::Vacant(e) => {
                            e.insert(new_cost);
                            true
                        }
                        HEntry::Occupied(mut e) => {
                            if new_cost < *(e.get()) {
                                e.insert(new_cost);
                                true
                            } else {
                                false
                            }
                        }
                    };
                    // record the chronological order of the updates
                    // which serves as a topological order that avoids cycles
                    // even when a term has a cost equal to its subterms
                    if updated {
                        self.topo_rnk_cnt += 1;
                        self.topo_rnk[f.output_sort_id].insert(target, self.topo_rnk_cnt);
                        if let Some(readers) = child_index.get(&(f.output_sort_id, target)) {
                            for &(dfi, dri) in readers {
                                let (dfi, dri) = (dfi as usize, dri as usize);
                                if !dirty[dfi][dri] {
                                    dirty[dfi][dri] = true;
                                    dirty_count[dfi] += 1;
                                }
                            }
                        }
                    }
                }
            }
            if !any {
                break;
            }
        }

        // Save the edges for reconstruction
        for f in &func_data {
            if f.rows.is_empty() {
                continue;
            }
            let func = egraph.functions.get(&f.name).unwrap();
            for row in f.rows.chunks_exact(f.arity) {
                let target = row[f.output_idx];
                let Some(best_cost) = self.costs[f.output_sort_id].get(&target) else {
                    continue;
                };
                if Some(best_cost.clone()) != self.row_cost(egraph, func, f, row, &mut ch_costs) {
                    continue;
                }
                // one of the possible best parent edges
                let target_topo_rnk = *self.topo_rnk[f.output_sort_id].get(&target).unwrap();
                let edge_topo_rnk =
                    f.child_kinds
                        .iter()
                        .zip(row.iter())
                        .fold(0, |ret, (kind, value)| {
                            usize::max(
                                ret,
                                match kind {
                                    ChildKind::EqSort(Some(id)) => {
                                        *self.topo_rnk[*id].get(value).unwrap_or(&usize::MAX)
                                    }
                                    ChildKind::EqSort(None) => usize::MAX,
                                    ChildKind::Container(sort) => {
                                        self.compute_topo_rnk_node(egraph, *value, sort)
                                    }
                                    ChildKind::Base(_) => 0,
                                },
                            )
                        });
                if target_topo_rnk > edge_topo_rnk {
                    // one of the parent edges that avoids cycles
                    if let HEntry::Vacant(e) = self.parent_edge[f.output_sort_id].entry(target) {
                        e.insert((func.decl.name.clone(), row.to_vec()));
                    }
                }
            }
        }
    }

    /// This recursively reconstruct the termdag that gives the minimum cost for eclass value.
    fn reconstruct_termdag_node(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
    ) -> TermId {
        self.reconstruct_termdag_node_helper(egraph, termdag, value, sort, &mut Default::default())
    }

    fn reconstruct_termdag_node_helper(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
        cache: &mut HashMap<(Value, String), TermId>,
    ) -> TermId {
        let key = (value, sort.name().to_owned());
        if let Some(term) = cache.get(&key) {
            return *term;
        }

        let term = if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.container_values(), value);
            let mut ch_terms: Vec<TermId> = Vec::new();
            for ch in elements.iter() {
                ch_terms.push(
                    self.reconstruct_termdag_node_helper(egraph, termdag, ch.1, &ch.0, cache),
                );
            }
            sort.reconstruct_termdag_container(
                egraph.backend.container_values(),
                value,
                termdag,
                ch_terms,
            )
        } else if sort.is_eq_sort() {
            let (func_name, hyperedge) = self.parent_edge[self.sort_ids[sort.name()]]
                .get(&value)
                .unwrap();
            let func = egraph.functions.get(func_name).unwrap();
            let ch_sorts = &func.schema.input;

            let num_children = func.extraction_num_children();
            let output_name = func.extraction_term_name();

            let mut ch_terms: Vec<TermId> = Vec::new();
            for (value, sort) in hyperedge.iter().take(num_children).zip(ch_sorts.iter()) {
                ch_terms.push(
                    self.reconstruct_termdag_node_helper(egraph, termdag, *value, sort, cache),
                );
            }
            termdag.app(output_name.to_string(), ch_terms)
        } else {
            // Base value case
            sort.reconstruct_termdag_base(egraph.backend.base_values(), value, termdag)
        };

        cache.insert(key, term);
        term
    }

    /// Extract the best term of a value from a given sort.
    ///
    /// This function expects the sort to be already computed,
    /// which can be one of the rootsorts, or reachable from rootsorts, or primitives, or containers of computed sorts.
    pub fn extract_best_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: ArcSort,
    ) -> Option<(C, TermId)> {
        // Canonicalize the value using the union-find if available (for term-encoding mode)
        let canonical_value = self.find_canonical(egraph, value, &sort);

        match self.compute_cost_node(egraph, canonical_value, &sort) {
            Some(best_cost) => {
                log::debug!("Best cost for the extract root: {best_cost:?}");

                let term = self.reconstruct_termdag_node(egraph, termdag, canonical_value, &sort);

                Some((best_cost, term))
            }
            None => {
                log::error!("Unextractable root {value:?} with sort {sort:?}",);
                None
            }
        }
    }

    /// A convenience method for extraction.
    ///
    /// This expects the value to be of the unique sort the extractor has been initialized with
    pub fn extract_best(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
    ) -> Option<(C, TermId)> {
        assert!(
            self.rootsorts.len() == 1,
            "extract_best requires a single rootsort"
        );
        self.extract_best_with_sort(
            egraph,
            termdag,
            value,
            self.rootsorts.first().unwrap().clone(),
        )
    }

    /// Find the canonical representative of a value using the union-find table.
    /// If no UF is registered for this sort, returns the original value.
    /// The UF table stores (value, canonical) pairs - one hop lookup.
    fn find_canonical(&self, egraph: &EGraph, value: Value, sort: &ArcSort) -> Value {
        // Check if there's a UF registered for this sort
        let Some(uf_name) = egraph.proof_state.uf_parent.get(sort.name()) else {
            return value;
        };

        // Get the UF function
        let Some(uf_func) = egraph.functions.get(uf_name) else {
            return value;
        };

        // Single lookup in UF table - it's guaranteed to be one hop to canonical
        let mut canonical = value;
        egraph
            .backend
            .for_each(uf_func.backend_id, |row: egglog_bridge::ScanEntry| {
                // UF table has (child, parent) as inputs
                if row.vals[0] == value {
                    canonical = row.vals[1];
                }
            });

        canonical
    }

    /// Extract variants of an e-class.
    ///
    /// The variants are selected by first picking `nvairants` e-nodes with the lowest cost from the e-class
    /// and then extracting a term from each e-node.
    pub fn extract_variants_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: ArcSort,
    ) -> Vec<(C, TermId)> {
        debug_assert!(self.rootsorts.iter().any(|s| { s.name() == sort.name() }));

        if sort.is_eq_sort() {
            // Canonicalize the value using the union-find if available
            let canonical_value = self.find_canonical(egraph, value, &sort);

            let mut root_variants: Vec<(C, String, Vec<Value>)> = Vec::new();

            let mut root_funcs: Vec<String> = Vec::new();

            for func_name in self.funcs.iter() {
                // Need an eq on sorts - use extraction_output_sort for view table support
                if sort.name()
                    == egraph
                        .functions
                        .get(func_name)
                        .unwrap()
                        .extraction_output_sort()
                        .name()
                {
                    root_funcs.push(func_name.clone());
                }
            }

            for func_name in root_funcs.iter() {
                let func = egraph.functions.get(func_name).unwrap();
                let output_idx = func.extraction_output_index();

                let find_root_variants = |row: egglog_bridge::ScanEntry| {
                    if !row.subsumed {
                        let target = &row.vals[output_idx];
                        if *target == canonical_value {
                            let cost = self.compute_cost_hyperedge(egraph, &row, func).unwrap();
                            root_variants.push((cost, func_name.clone(), row.vals.to_vec()));
                        }
                    }
                };

                egraph.backend.for_each(func.backend_id, find_root_variants);
            }

            let mut res: Vec<(C, TermId)> = Vec::new();
            let mut cache: HashMap<(Value, String), TermId> = Default::default();
            root_variants.sort();
            root_variants.truncate(nvariants);
            for (cost, func_name, hyperedge) in root_variants {
                let mut ch_terms: Vec<TermId> = Vec::new();
                let func = egraph.functions.get(&func_name).unwrap();
                let ch_sorts = &func.schema.input;
                let num_children = func.extraction_num_children();
                // For view tables, children are all but the last input (which is the e-class)
                for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()).take(num_children) {
                    ch_terms.push(self.reconstruct_termdag_node_helper(
                        egraph, termdag, *value, sort, &mut cache,
                    ));
                }
                // Use extraction_term_name for view tables (maps to the original constructor)
                res.push((
                    cost,
                    termdag.app(func.extraction_term_name().to_string(), ch_terms),
                ));
            }

            res
        } else {
            log::warn!(
                "extracting multiple variants for containers or primitives is not implemented, returning a single variant."
            );
            if let Some(res) = self.extract_best_with_sort(egraph, termdag, value, sort) {
                vec![res]
            } else {
                vec![]
            }
        }
    }

    /// A convenience method for extracting variants of a value.
    ///
    /// This expects the value to be of the unique sort the extractor has been initialized with.
    pub fn extract_variants(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
    ) -> Vec<(C, TermId)> {
        assert!(
            self.rootsorts.len() == 1,
            "extract_variants requires a single rootsort"
        );
        self.extract_variants_with_sort(
            egraph,
            termdag,
            value,
            nvariants,
            self.rootsorts.first().unwrap().clone(),
        )
    }
}

impl Function {
    /// Returns the extraction head cost for this table.
    /// View tables inherit the cost of their referenced hidden term constructor.
    pub(crate) fn extraction_head_cost(&self, egraph: &EGraph) -> DefaultCost {
        if let Some(term_constructor) = &self.decl.term_constructor {
            egraph
                .functions
                .get(term_constructor)
                .and_then(|func| func.decl.cost)
                .unwrap_or(DefaultCost::unit())
        } else {
            self.decl.cost.unwrap_or(DefaultCost::unit())
        }
    }

    /// For view tables (with term_constructor), the effective output sort is the last input column.
    /// For regular tables, it's the output sort.
    /// This is used by extraction to determine which sort a table produces values for.
    pub(crate) fn extraction_output_sort(&self) -> &ArcSort {
        if self.decl.term_constructor.is_some() {
            self.schema.input.last().unwrap()
        } else {
            &self.schema.output
        }
    }

    /// Returns the number of children for extraction purposes.
    /// For view tables, this excludes the last column (the e-class).
    pub(crate) fn extraction_num_children(&self) -> usize {
        if self.decl.term_constructor.is_some() {
            self.schema.input.len() - 1
        } else {
            self.schema.input.len()
        }
    }

    /// Returns the name to use when building terms during extraction.
    /// For view tables, this is the term_constructor name.
    pub(crate) fn extraction_term_name(&self) -> &str {
        self.decl
            .term_constructor
            .as_ref()
            .unwrap_or(&self.decl.name)
    }

    /// Returns the index of the output value in a row for extraction purposes.
    /// For view tables, the e-class is the last input column (second-to-last in the row).
    /// For regular tables, it's the last column (the actual output).
    pub(crate) fn extraction_output_index(&self) -> usize {
        if self.decl.term_constructor.is_some() {
            // For view tables: input is [children..., eclass], output is view_sort
            // Row is [children..., eclass, view_sort]
            // We want eclass which is at index input.len() - 1
            self.schema.input.len() - 1
        } else {
            // For regular tables: row is [inputs..., output]
            self.schema.input.len()
        }
    }
}

impl EGraph {
    /// Extract a value to a [`TermDag`] and [`TermId`] in the [`TermDag`] using the default cost model.
    /// See also [`EGraph::extract_value_with_cost_model`] for more control.
    pub fn extract_value(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<(TermDag, TermId, DefaultCost), Error> {
        self.extract_value_with_cost_model(sort, value, TreeAdditiveCostModel::default())
    }

    /// Extract a value to a [`TermDag`] and [`TermId`] in the [`TermDag`].
    /// Note that the `TermDag` may contain a superset of the nodes referenced by the returned `TermId`.
    /// See also [`EGraph::extract_value_to_string`] for convenience.
    pub fn extract_value_with_cost_model<CM: CostModel<DefaultCost> + 'static>(
        &self,
        sort: &ArcSort,
        value: Value,
        cost_model: CM,
    ) -> Result<(TermDag, TermId, DefaultCost), Error> {
        let extractor =
            Extractor::compute_costs_from_rootsorts(Some(vec![sort.clone()]), self, cost_model);
        let mut termdag = TermDag::default();
        let (cost, term) = extractor.extract_best(self, &mut termdag, value).unwrap();
        Ok((termdag, term, cost))
    }

    /// Extract a value to a string for printing.
    /// See also [`EGraph::extract_value`] for more control.
    pub fn extract_value_to_string(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<(String, DefaultCost), Error> {
        let (termdag, term, cost) = self.extract_value(sort, value)?;
        Ok((termdag.to_string(term), cost))
    }

    /// For constructors and relations, the output column can be ignored
    pub fn function_to_dag(
        &self,
        sym: &str,
        n: usize,
        include_output: bool,
    ) -> Result<(Vec<TermId>, Option<Vec<TermId>>, TermDag), Error> {
        let func = self
            .functions
            .get(sym)
            .ok_or(TypeError::UnboundFunction(sym.to_owned(), span!()))?;
        let mut rootsorts = func.schema.input.clone();
        if include_output {
            rootsorts.push(func.schema.output.clone());
        }
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(rootsorts),
            self,
            TreeAdditiveCostModel::default(),
        );

        let mut termdag = TermDag::default();
        let mut inputs: Vec<TermId> = Vec::new();
        let mut output: Option<Vec<TermId>> = if include_output {
            Some(Vec::new())
        } else {
            None
        };

        let extract_row = |row: egglog_bridge::ScanEntry| {
            if inputs.len() < n {
                // include subsumed rows
                let mut children: Vec<TermId> = Vec::new();
                for (value, sort) in row.vals.iter().zip(&func.schema.input) {
                    let (_, term_id) = extractor
                        .extract_best_with_sort(self, &mut termdag, *value, sort.clone())
                        .unwrap_or_else(|| (0, termdag.var("Unextractable".into())));
                    children.push(term_id);
                }
                inputs.push(termdag.app(sym.to_owned(), children));
                if include_output {
                    let value = row.vals[func.schema.input.len()];
                    let sort = &func.schema.output;
                    let (_, term) = extractor
                        .extract_best_with_sort(self, &mut termdag, value, sort.clone())
                        .unwrap_or_else(|| (0, termdag.var("Unextractable".into())));
                    output.as_mut().unwrap().push(term);
                }
                true
            } else {
                false
            }
        };

        self.backend.for_each_while(func.backend_id, extract_row);

        Ok((inputs, output, termdag))
    }
}
