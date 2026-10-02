//! Cost models used by extraction.
//!
//! [`Cost`] is the minimum requirement for ranking extracted terms.
//! [`TreeCostModel`] computes a node annotation and then folds it with selected
//! child costs. The annotation can carry context that should not be part of the
//! final cost, so tree extraction does not require costs to support addition.
//!
//! [`MonoidCost`] adds the associative and commutative combination operation
//! needed to charge shared dependencies once. [`DagCostModel`] computes the
//! marginal costs consumed by DAG extractors. [`TreeCostModelFromDag`] adapts
//! such a model to tree extraction by combining each node's marginal cost with
//! its selected child costs. [`DEFAULT_COST_MODEL`] is the default additive
//! tree model.

use crate::termdag::{TermDag, TermId};
use crate::util::{HashMap, HashSet};
use crate::*;
use std::collections::{BTreeSet, VecDeque};
use std::ops::Range;

#[cfg(test)]
mod dependency_tests;

/// A value that can be used to rank extraction candidates.
pub trait Cost: Clone + Ord {}

impl<T: Clone + Ord> Cost for T {}

/// An extraction cost with an identity and a combination operation.
///
/// Implementations must make `identity` a two-sided identity and `combine`
/// associative, commutative, deterministic, non-panicking, and monotone with
/// respect to [`Cost`]'s ordering. Rust cannot enforce these laws.
pub trait MonoidCost: Cost {
    /// The identity cost, usually zero.
    fn identity() -> Self;

    /// Combines two costs without overflowing or panicking.
    fn combine(self, other: &Self) -> Self;
}

macro_rules! monoid_unsigned_cost {
    ($($cost:ty),* $(,)?) => {$(
        impl MonoidCost for $cost {
            fn identity() -> Self {
                0
            }

            fn combine(self, other: &Self) -> Self {
                self.saturating_add(*other)
            }
        }
    )*};
}

monoid_unsigned_cost!(u8, u16, u32, u64, u128, usize);

macro_rules! monoid_exact_cost {
    ($($cost:ty),* $(,)?) => {$(
        impl MonoidCost for $cost {
            fn identity() -> Self {
                use num::Zero;
                Self::zero()
            }

            fn combine(self, other: &Self) -> Self {
                self + other
            }
        }
    )*};
}

monoid_exact_cost!(num::BigInt, num::BigRational);

/// Computes marginal costs for extraction that charges shared dependencies once.
///
/// The returned costs exclude selected child and container-element costs.
/// Implementations must return equal costs for equivalent arguments during one
/// extraction.
pub trait DagCostModel<C: MonoidCost> {
    /// Computes the cost of a non-container primitive value.
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C;
    /// Computes an enode's cost, excluding its children.
    fn enode_cost(&self, egraph: &EGraph, func: &Function, enode: &Enode<'_>) -> C;

    /// Computes a container's cost, excluding its elements.
    fn container_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C {
        let _ = (egraph, sort, value);
        C::identity()
    }
}

/// Computes tree-extraction costs from node annotations and selected child costs.
///
/// Models used by [`TreeExtractor`] must be deterministic and satisfy optimal
/// substructure: replacing a child with a lower-cost extraction must not make
/// its parent more expensive. They must also avoid making cyclic terms improve
/// indefinitely. Otherwise extraction may be non-optimal, fail to converge, or
/// fail during reconstruction. The annotation types let the first stage retain
/// context, such as the constructor kind, without encoding it in `C`.
pub trait TreeCostModel<C: Cost> {
    /// Context retained between [`TreeCostModel::enode_cost`] and its fold.
    type EnodeCost;
    /// Context retained between [`TreeCostModel::container_cost`] and its fold.
    type ContainerCost;

    /// Computes the cost of a non-container primitive value.
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C;

    /// Computes the annotation folded with an enode's selected child costs.
    fn enode_cost(&self, egraph: &EGraph, func: &Function, enode: &Enode<'_>) -> Self::EnodeCost;

    /// Computes the annotation folded with a container's selected element costs.
    fn container_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> Self::ContainerCost;

    /// Produces an enode's total cost from its annotation and child costs.
    fn fold_enode_cost(&self, enode_cost: Self::EnodeCost, child_costs: &[C]) -> C;

    /// Produces a container's total cost from its annotation and element costs.
    fn fold_container_cost(&self, container_cost: Self::ContainerCost, element_costs: &[C]) -> C;
}

trait TreeExtractorCostModel<C: Cost> {
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C;
    fn total_enode_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        enode: &Enode<'_>,
        child_costs: &[C],
    ) -> C;
    fn total_container_cost(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        element_costs: &[C],
    ) -> C;
}

impl<C: Cost, M: TreeCostModel<C>> TreeExtractorCostModel<C> for M {
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C {
        TreeCostModel::base_value_cost(self, egraph, sort, value)
    }

    fn total_enode_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        enode: &Enode<'_>,
        child_costs: &[C],
    ) -> C {
        self.fold_enode_cost(self.enode_cost(egraph, func, enode), child_costs)
    }

    fn total_container_cost(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        element_costs: &[C],
    ) -> C {
        self.fold_container_cost(self.container_cost(egraph, sort, value), element_costs)
    }
}

/// Adapts a [`DagCostModel`] to tree extraction by combining marginal costs.
#[derive(Clone, Debug)]
pub struct TreeCostModelFromDag<M>(pub M);

impl<C: MonoidCost, M: DagCostModel<C>> TreeCostModel<C> for TreeCostModelFromDag<M> {
    type EnodeCost = C;
    type ContainerCost = C;

    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C {
        self.0.base_value_cost(egraph, sort, value)
    }

    fn enode_cost(&self, egraph: &EGraph, func: &Function, enode: &Enode<'_>) -> C {
        self.0.enode_cost(egraph, func, enode)
    }

    fn container_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C {
        self.0.container_cost(egraph, sort, value)
    }

    fn fold_enode_cost(&self, enode_cost: C, child_costs: &[C]) -> C {
        child_costs
            .iter()
            .fold(enode_cost, |cost, child| cost.combine(child))
    }

    fn fold_container_cost(&self, container_cost: C, element_costs: &[C]) -> C {
        element_costs
            .iter()
            .fold(container_cost, |cost, element| cost.combine(element))
    }
}

/// The default extraction cost type.
pub type DefaultCost = u64;

/// The marginal-cost model underlying default tree and DAG extraction.
///
/// With [`DefaultCost`], constructor `:cost` declarations override `node_cost`
/// and selected child costs are combined with each constructor's cost.
#[derive(Clone, Debug)]
pub struct AdditiveCostModel {
    /// The fallback cost for primitive values and constructors without `:cost`.
    pub node_cost: DefaultCost,
}

impl Default for AdditiveCostModel {
    fn default() -> Self {
        Self { node_cost: 1 }
    }
}

/// The default additive model used by tree extraction.
pub const DEFAULT_COST_MODEL: TreeCostModelFromDag<AdditiveCostModel> =
    TreeCostModelFromDag(AdditiveCostModel { node_cost: 1 });

impl DagCostModel<DefaultCost> for AdditiveCostModel {
    fn base_value_cost(&self, _egraph: &EGraph, _sort: &ArcSort, _value: Value) -> DefaultCost {
        self.node_cost
    }

    fn enode_cost(&self, egraph: &EGraph, func: &Function, _enode: &Enode<'_>) -> DefaultCost {
        func.extraction_head_cost(egraph).unwrap_or(self.node_cost)
    }
}

/// One extracted root or root variant.
///
/// The cost is the objective value assigned by the selected extractor. Tree
/// extraction reports tree cost; DAG extractors may instead report a per-root
/// or per-variant DAG cost. `term` indexes the enclosing result's shared
/// [`TermDag`].
#[derive(Clone, Debug)]
pub struct ExtractedTerm<C> {
    /// The selected extractor's cost for this result.
    pub cost: C,
    /// The extracted root in the enclosing result's [`TermDag`].
    pub term: TermId,
}

/// Best-extraction results for a batch of requested roots.
///
/// All returned term ids index the shared [`ExtractedTerms::termdag`].
#[derive(Clone, Debug)]
pub struct ExtractedTerms<C> {
    /// Shared term storage for every extracted root.
    pub termdag: TermDag,
    /// One extraction result per requested root, in request order.
    ///
    /// `None` means that root is unextractable with the selected cost model and
    /// available constructors.
    pub terms: Vec<Option<ExtractedTerm<C>>>,
}

/// Root-variant extraction results for a batch of requested roots.
///
/// All returned term ids index the shared [`ExtractedTermVariants::termdag`].
#[derive(Clone, Debug)]
pub struct ExtractedTermVariants<C> {
    /// Shared term storage for every extracted variant.
    pub termdag: TermDag,
    /// Outer vector follows requested-root order; each inner vector contains that root's variants.
    pub variants: Vec<Vec<ExtractedTerm<C>>>,
}

/// Dependency-prepared tree extraction with reusable costs.
///
/// The prepared state borrows the e-graph because reconstruction still needs
/// its sort storage and constructor metadata. This prevents prepared costs from
/// being used after the e-graph is mutated.
pub struct TreeExtractor<'g, C: Cost> {
    egraph: &'g EGraph,
    funcs: Vec<&'g Function>,
    cost_model: Box<dyn TreeExtractorCostModel<C> + 'g>,
    costs: HashMap<String, HashMap<Value, RankedCost<C>>>,
    topo_rnk_cnt: usize,
    parent_edge: HashMap<String, HashMap<Value, (usize, Vec<Value>)>>,
}

// A successful relaxation updates the cost and its chronological rank together.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RankedCost<C> {
    cost: C,
    rank: usize,
}

struct ExtractionFunction<'g> {
    func: &'g Function,
    output_sort_name: &'g str,
    // The output follows the extracted children, including in view tables.
    output_index: usize,
}

struct PreparedHyperedge {
    function: usize,
    values: Range<usize>,
}

type ReverseDependencies = HashMap<String, HashMap<Value, Vec<usize>>>;

impl<'g, C: Cost> TreeExtractor<'g, C> {
    /// Prepares extraction costs for constructors reachable from `rootsorts`.
    ///
    /// Pass `None` to prepare every extractable root sort. Later calls to
    /// [`TreeExtractor::extract_best_with_sort`] and
    /// [`TreeExtractor::extract_variants_with_sort`] reuse the prepared best
    /// costs and producer choices.
    ///
    /// Primitive and container roots are costed when extracted. Variant
    /// extraction also uses the model to rescore candidate root enodes, so the
    /// model must return stable results for the extractor's lifetime.
    pub fn compute_costs_from_rootsorts(
        rootsorts: Option<Vec<ArcSort>>,
        egraph: &'g EGraph,
        cost_model: impl TreeCostModel<C> + 'g,
    ) -> Self {
        // We filter out tables unreachable from the root sorts
        let extract_all_sorts = rootsorts.is_none();

        let mut rootsorts = rootsorts.unwrap_or_default();

        // Index extractable tables by output sort. View tables reconstruct as
        // their term_constructor; their e-class is the last input column.
        let mut rev_index: HashMap<&str, Vec<&Function>> = Default::default();
        for func in egraph.functions.values() {
            if !func.decl.unextractable
                && !func.decl.internal_hidden
                && (func.decl.subtype == FunctionSubtype::Constructor
                    || func.decl.term_constructor.is_some())
            {
                let output_sort = func.extraction_output_sort();
                let functions = rev_index.entry(output_sort.name()).or_default();
                if functions.is_empty() && extract_all_sorts {
                    rootsorts.push(output_sort.clone());
                }
                functions.push(func);
            }
        }

        // Visit each sort once, retaining its tables in the original BFS order.
        // Each table belongs to one output sort, so no table deduplication is needed.
        let mut q: VecDeque<ArcSort> = VecDeque::new();
        let mut seen: HashSet<String> = Default::default();
        for rootsort in rootsorts {
            if seen.insert(rootsort.name().to_owned()) {
                q.push_back(rootsort);
            }
        }

        let mut funcs = Vec::new();
        while let Some(sort) = q.pop_front() {
            if sort.is_container_sort() {
                for s in sort.inner_sorts() {
                    if !seen.contains(s.name()) {
                        seen.insert(s.name().to_owned());
                        q.push_back(s.clone());
                    }
                }
            } else if sort.is_eq_sort()
                && let Some(functions) = rev_index.get(sort.name())
            {
                for &func in functions {
                    for ch in func
                        .func_type
                        .input
                        .iter()
                        .take(func.extraction_num_children())
                    {
                        if !seen.contains(ch.name()) {
                            seen.insert(ch.name().to_owned());
                            q.push_back(ch.clone());
                        }
                    }
                    funcs.push(func);
                }
            }
        }

        // Initialize the tables to have the reachable entries
        let mut costs = HashMap::default();
        let mut parent_edge = HashMap::default();

        for func in &funcs {
            let output_sort_name = func.extraction_output_sort().name();
            if !costs.contains_key(output_sort_name) {
                costs.insert(output_sort_name.to_owned(), Default::default());
                parent_edge.insert(output_sort_name.to_owned(), Default::default());
            }
        }

        let mut extractor = TreeExtractor {
            egraph,
            funcs,
            cost_model: Box::new(cost_model),
            costs,
            topo_rnk_cnt: 0,
            parent_edge,
        };
        let funcs: Vec<_> = extractor
            .funcs
            .iter()
            .map(|&func| ExtractionFunction {
                func,
                output_sort_name: func.extraction_output_sort().name(),
                output_index: func.extraction_output_index(),
            })
            .collect();
        extractor.dependency_relaxation(egraph, &funcs);
        extractor.save_best_parent_edges(egraph, &funcs);
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
                    .total_container_cost(egraph, sort, value, &ch_costs),
            )
        } else if sort.is_eq_sort() {
            Some(self.costs.get(sort.name())?.get(&value)?.cost.clone())
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
        let sorts = &func.func_type.input;
        let num_children = func.extraction_num_children();
        for (value, sort) in row.vals.iter().take(num_children).zip(sorts.iter()) {
            ch_costs.push(self.compute_cost_node(egraph, *value, sort)?);
        }
        let head_name = func.extraction_term_name();
        let output_idx = func.extraction_output_index();
        let enode = Enode {
            name: head_name,
            children: &row.vals[..output_idx],
            eclass: row.vals[output_idx],
            subsumed: row.subsumed,
        };
        let cost_func = func
            .decl
            .term_constructor
            .as_ref()
            .and_then(|name| egraph.functions.get(name))
            .unwrap_or(func);
        Some(
            self.cost_model
                .total_enode_cost(egraph, cost_func, &enode, &ch_costs),
        )
    }

    fn compute_topo_rnk_node(&self, egraph: &EGraph, value: Value, sort: &ArcSort) -> usize {
        if sort.is_container_sort() {
            sort.inner_values(egraph.backend.container_values(), value)
                .iter()
                .fold(0, |ret, (sort, value)| {
                    usize::max(ret, self.compute_topo_rnk_node(egraph, *value, sort))
                })
        } else if sort.is_eq_sort() {
            if let Some(costs) = self.costs.get(sort.name()) {
                costs.get(&value).map_or(usize::MAX, |entry| entry.rank)
            } else {
                usize::MAX
            }
        } else {
            0
        }
    }

    fn compute_topo_rnk_hyperedge(
        &self,
        egraph: &EGraph,
        row: &egglog_bridge::ScanEntry,
        func: &Function,
    ) -> usize {
        let sorts = &func.func_type.input;
        let num_children = func.extraction_num_children();
        row.vals
            .iter()
            .take(num_children)
            .zip(sorts.iter())
            .fold(0, |ret, (value, sort)| {
                usize::max(ret, self.compute_topo_rnk_node(egraph, *value, sort))
            })
    }

    fn collect_eq_dependencies(
        egraph: &EGraph,
        value: Value,
        sort: &ArcSort,
        parents: &mut ReverseDependencies,
        ordinal: usize,
    ) -> bool {
        if sort.is_container_sort() {
            let mut found = false;
            for (inner_sort, inner_value) in
                sort.inner_values(egraph.backend.container_values(), value)
            {
                found |= Self::collect_eq_dependencies(
                    egraph,
                    inner_value,
                    &inner_sort,
                    parents,
                    ordinal,
                );
            }
            found
        } else if sort.is_eq_sort() {
            let dependent_edges = parents
                .entry_ref(sort.name())
                .or_default()
                .entry(value)
                .or_default();
            // Rows register in ordinal order, so repeated leaves in this row
            // already have this ordinal at the end of their parent list.
            if dependent_edges.last() != Some(&ordinal) {
                dependent_edges.push(ordinal);
            }
            true
        } else {
            false
        }
    }

    fn relax_hyperedge(
        &mut self,
        egraph: &EGraph,
        row: &egglog_bridge::ScanEntry,
        descriptor: &ExtractionFunction<'g>,
    ) -> bool {
        let target = row.vals[descriptor.output_index];
        let Some(new_cost) = self.compute_cost_hyperedge(egraph, row, descriptor.func) else {
            return false;
        };
        let entry = self
            .costs
            .get_mut(descriptor.output_sort_name)
            .unwrap()
            .entry(target);
        if let HEntry::Occupied(old) = &entry
            && new_cost >= old.get().cost
        {
            return false;
        }
        self.topo_rnk_cnt += 1;
        entry.insert(RankedCost {
            cost: new_cost,
            rank: self.topo_rnk_cnt,
        });
        true
    }

    fn ordered_full_sweep(&mut self, egraph: &EGraph, funcs: &[ExtractionFunction<'g>]) -> bool {
        let mut updated = false;
        for descriptor in funcs {
            egraph.backend.for_each(descriptor.func.backend_id, |row| {
                if !row.subsumed && self.relax_hyperedge(egraph, &row, descriptor) {
                    updated = true;
                }
            });
        }
        updated
    }

    fn dependency_relaxation(&mut self, egraph: &EGraph, funcs: &[ExtractionFunction<'g>]) {
        // Stable costs let us skip rows with unchanged child costs. Keeping all
        // possible updates in sweep/row order preserves reconstruction ranks.
        // A small fixed prefix avoids indexing short-convergence workloads and
        // bounds full-scan overhead independently of dependency depth. The cutoff
        // affects preparation cost only; either schedule preserves the same ranks.
        const UNINDEXED_SWEEPS: usize = 3;
        for _ in 0..UNINDEXED_SWEEPS {
            if !self.ordered_full_sweep(egraph, funcs) {
                return;
            }
        }

        // A confirmation sweep must not pay for an unused dependency index.
        // Probe without updating costs: if work remains, the prepared sweep
        // starts from the same state and retains every successful update's order.
        let pending = funcs.iter().any(|descriptor| {
            let mut pending = false;
            egraph
                .backend
                .for_each_while(descriptor.func.backend_id, |row| {
                    pending = !row.subsumed
                        && self
                            .compute_cost_hyperedge(egraph, &row, descriptor.func)
                            .is_some_and(|cost| {
                                self.costs[descriptor.output_sort_name]
                                    .get(&row.vals[descriptor.output_index])
                                    .is_none_or(|old| cost < old.cost)
                            });
                    !pending
                });
            pending
        });
        if !pending {
            return;
        }

        // The final full sweep both relaxes costs and records immutable rows
        // and their reverse dependencies for subsequent sparse sweeps.
        // Ordinals rely on immutable function tables retaining their row scan
        // order: sparse sweeps must replay the full sweep's update chronology.
        let mut edges = Vec::new();
        let mut values = Vec::new();
        let mut parents: ReverseDependencies = Default::default();
        let mut current = BTreeSet::new();
        for (function, descriptor) in funcs.iter().enumerate() {
            let func = descriptor.func;
            egraph.backend.for_each(func.backend_id, |row| {
                if row.subsumed {
                    return;
                }

                let ordinal = edges.len();
                let mut has_dependencies = false;
                for (value, sort) in row
                    .vals
                    .iter()
                    .take(descriptor.output_index)
                    .zip(&func.func_type.input)
                {
                    has_dependencies |=
                        Self::collect_eq_dependencies(egraph, *value, sort, &mut parents, ordinal);
                }
                // Without e-class dependencies the cost was fixed on the first
                // sweep, so this row can never trigger another improvement.
                if !has_dependencies {
                    return;
                }
                let start = values.len();
                values.extend_from_slice(row.vals);
                edges.push(PreparedHyperedge {
                    function,
                    values: start..values.len(),
                });
                let target = row.vals[descriptor.output_index];
                if self.relax_hyperedge(egraph, &row, descriptor)
                    && let Some(dependent_edges) = parents
                        .get(descriptor.output_sort_name)
                        .and_then(|by_value| by_value.get(&target))
                {
                    // Registered parents have already been visited, or are this
                    // row itself. Future rows will be visited by this dense scan.
                    for &parent in dependent_edges {
                        current.insert(parent);
                    }
                }
            });
        }
        let mut next = BTreeSet::new();

        while !current.is_empty() {
            while let Some(ordinal) = current.pop_first() {
                let edge = &edges[ordinal];
                let descriptor = &funcs[edge.function];
                let row_values = &values[edge.values.clone()];
                let target = row_values[descriptor.output_index];
                let row = egglog_bridge::ScanEntry {
                    vals: row_values,
                    subsumed: false,
                };
                if self.relax_hyperedge(egraph, &row, descriptor)
                    && let Some(dependent_edges) = parents
                        .get(descriptor.output_sort_name)
                        .and_then(|by_value| by_value.get(&target))
                {
                    for &parent in dependent_edges {
                        // Preserve the original sweep and row order, including
                        // self-dependencies, to retain chronological ranks.
                        if parent > ordinal {
                            current.insert(parent);
                        } else {
                            next.insert(parent);
                        }
                    }
                }
            }

            // The drained current queue becomes the empty next-sweep queue.
            std::mem::swap(&mut current, &mut next);
        }
    }

    fn save_best_parent_edges(&mut self, egraph: &EGraph, funcs: &[ExtractionFunction<'g>]) {
        for (function, descriptor) in funcs.iter().enumerate() {
            let func = descriptor.func;
            egraph.backend.for_each(func.backend_id, |row| {
                if row.subsumed {
                    return;
                }
                let target = row.vals[descriptor.output_index];
                let Some(best) = self.costs[descriptor.output_sort_name].get(&target) else {
                    return;
                };
                // Keep the first minimum-cost producer whose children have
                // strictly earlier ranks, so reconstruction cannot cycle.
                if Some(best.cost.clone()) != self.compute_cost_hyperedge(egraph, &row, func)
                    || best.rank <= self.compute_topo_rnk_hyperedge(egraph, &row, func)
                {
                    return;
                }
                self.parent_edge
                    .get_mut(descriptor.output_sort_name)
                    .unwrap()
                    .entry(target)
                    .or_insert_with(|| (function, row.vals.to_vec()));
            });
        }
    }

    /// This recursively reconstruct the termdag that gives the minimum cost for eclass value.
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
            let (function, hyperedge) = self
                .parent_edge
                .get(sort.name())
                .unwrap()
                .get(&value)
                .unwrap();
            let func = self.funcs[*function];
            let ch_sorts = &func.func_type.input;

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

    /// Extracts the best term for `value` from a prepared sort.
    ///
    /// `sort` may be a requested root sort, a sort reachable from those roots,
    /// a primitive sort, or a container of prepared sorts.
    pub fn extract_best_with_sort(
        &self,
        termdag: &mut TermDag,
        value: Value,
        sort: ArcSort,
    ) -> Option<ExtractedTerm<C>> {
        self.extract_best_with_sort_cached(termdag, &mut Default::default(), value, sort)
    }

    fn extract_best_with_sort_cached(
        &self,
        termdag: &mut TermDag,
        cache: &mut HashMap<(Value, String), TermId>,
        value: Value,
        sort: ArcSort,
    ) -> Option<ExtractedTerm<C>> {
        let egraph = self.egraph;
        // Canonicalize the value using the union-find if available (for term-encoding mode)
        let canonical_value = self.find_canonical(egraph, value, &sort);

        let best_cost = self.compute_cost_node(egraph, canonical_value, &sort)?;
        let term =
            self.reconstruct_termdag_node_helper(egraph, termdag, canonical_value, &sort, cache);

        Some(ExtractedTerm {
            cost: best_cost,
            term,
        })
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

    /// Extracts up to `nvariants` root variants for `value` from a prepared sort.
    ///
    /// Variants are selected by first picking the lowest-cost root e-nodes and
    /// then extracting each e-node's children with their best prepared terms.
    pub fn extract_variants_with_sort(
        &self,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: ArcSort,
    ) -> Vec<ExtractedTerm<C>> {
        if nvariants == 0 {
            return vec![];
        }

        let egraph = self.egraph;
        if sort.is_eq_sort() {
            // Canonicalize the value using the union-find if available
            let canonical_value = self.find_canonical(egraph, value, &sort);

            let mut root_variants: Vec<(C, String, Vec<Value>)> = Vec::new();

            for &func in self
                .funcs
                .iter()
                .filter(|func| sort.name() == func.extraction_output_sort().name())
            {
                let output_idx = func.extraction_output_index();

                let find_root_variants = |row: egglog_bridge::ScanEntry| {
                    if !row.subsumed {
                        let target = &row.vals[output_idx];
                        // A variant whose cost is `None` has a child e-class with no
                        // finite extraction (e.g. a purely cyclic child); such a variant
                        // can never appear in a minimal extraction, so we skip it. The
                        // target e-class still extracts via its other, costed variants.
                        if *target == canonical_value
                            && let Some(cost) = self.compute_cost_hyperedge(egraph, &row, func)
                        {
                            root_variants.push((cost, func.name().to_owned(), row.vals.to_vec()));
                        }
                    }
                };

                egraph.backend.for_each(func.backend_id, find_root_variants);
            }

            let mut res: Vec<ExtractedTerm<C>> = Vec::new();
            let mut cache: HashMap<(Value, String), TermId> = Default::default();
            root_variants.sort();
            root_variants.truncate(nvariants);
            for (cost, func_name, hyperedge) in root_variants {
                let mut ch_terms: Vec<TermId> = Vec::new();
                let func = egraph.functions.get(&func_name).unwrap();
                let ch_sorts = &func.func_type.input;
                let num_children = func.extraction_num_children();
                // For view tables, children are all but the last input (which is the e-class)
                for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()).take(num_children) {
                    ch_terms.push(self.reconstruct_termdag_node_helper(
                        egraph, termdag, *value, sort, &mut cache,
                    ));
                }
                // Use extraction_term_name for view tables (maps to the original constructor)
                res.push(ExtractedTerm {
                    cost,
                    term: termdag.app(func.extraction_term_name().to_string(), ch_terms),
                });
            }

            res
        } else {
            log::warn!(
                "extracting multiple variants for containers or primitives is not implemented, returning a single variant."
            );
            if let Some(res) = self.extract_best_with_sort(termdag, value, sort) {
                vec![res]
            } else {
                vec![]
            }
        }
    }
}

impl Function {
    /// Returns the configured extraction head cost for this table.
    ///
    /// View tables inherit the cost of their referenced hidden term constructor.
    pub(crate) fn extraction_head_cost(&self, egraph: &EGraph) -> Option<DefaultCost> {
        if let Some(term_constructor) = &self.decl.term_constructor {
            egraph
                .functions
                .get(term_constructor)
                .and_then(|func| func.decl.cost)
        } else {
            self.decl.cost
        }
    }

    /// For view tables (with term_constructor), the effective output sort is the last input column.
    /// For regular tables, it's the output sort.
    /// This is used by extraction to determine which sort a table produces values for.
    pub(crate) fn extraction_output_sort(&self) -> &ArcSort {
        if self.decl.term_constructor.is_some() {
            self.func_type.input.last().unwrap()
        } else {
            &self.func_type.output
        }
    }

    /// Returns the number of children for extraction purposes.
    /// For view tables, this excludes the last column (the e-class).
    pub(crate) fn extraction_num_children(&self) -> usize {
        if self.decl.term_constructor.is_some() {
            self.func_type.input.len() - 1
        } else {
            self.func_type.input.len()
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
            self.func_type.input.len() - 1
        } else {
            // For regular tables: row is [inputs..., output]
            self.func_type.input.len()
        }
    }
}

impl EGraph {
    /// Extracts the best tree term for each requested `(sort, value)` root
    /// using [`DEFAULT_COST_MODEL`].
    pub fn extract_best(
        &self,
        roots: Vec<(ArcSort, Value)>,
    ) -> Result<ExtractedTerms<DefaultCost>, Error> {
        self.extract_best_with_cost_model(roots, DEFAULT_COST_MODEL)
    }

    /// Extracts the best tree term for each requested `(sort, value)` root.
    ///
    /// This is the normal user extraction path: it respects `:unextractable`
    /// and hidden internal functions. The cost model must satisfy the
    /// optimal-substructure and convergence requirements on [`TreeExtractor`].
    pub fn extract_best_with_cost_model<C: Cost, M: TreeCostModel<C>>(
        &self,
        roots: Vec<(ArcSort, Value)>,
        cost_model: M,
    ) -> Result<ExtractedTerms<C>, Error> {
        let rootsorts = roots.iter().map(|(sort, _)| sort.clone()).collect();
        let extractor =
            TreeExtractor::compute_costs_from_rootsorts(Some(rootsorts), self, cost_model);
        let mut termdag = TermDag::default();
        let mut cache = Default::default();
        let extracted_roots = roots
            .into_iter()
            .map(|(sort, value)| {
                extractor.extract_best_with_sort_cached(&mut termdag, &mut cache, value, sort)
            })
            .collect();

        Ok(ExtractedTerms {
            termdag,
            terms: extracted_roots,
        })
    }

    /// Extracts up to `nvariants` tree root variants for each requested root
    /// using [`DEFAULT_COST_MODEL`].
    pub fn extract_variants(
        &self,
        roots: Vec<(ArcSort, Value)>,
        nvariants: usize,
    ) -> Result<ExtractedTermVariants<DefaultCost>, Error> {
        self.extract_variants_with_cost_model(roots, nvariants, DEFAULT_COST_MODEL)
    }

    /// Extracts up to `nvariants` tree root variants for each requested root.
    ///
    /// The cost model must satisfy the optimal-substructure and convergence
    /// requirements on [`TreeExtractor`].
    pub fn extract_variants_with_cost_model<C: Cost, M: TreeCostModel<C>>(
        &self,
        roots: Vec<(ArcSort, Value)>,
        nvariants: usize,
        cost_model: M,
    ) -> Result<ExtractedTermVariants<C>, Error> {
        if nvariants == 0 {
            return Ok(ExtractedTermVariants {
                termdag: TermDag::default(),
                variants: roots.iter().map(|_| Vec::new()).collect(),
            });
        }

        let rootsorts = roots.iter().map(|(sort, _)| sort.clone()).collect();
        let extractor =
            TreeExtractor::compute_costs_from_rootsorts(Some(rootsorts), self, cost_model);
        let mut termdag = TermDag::default();
        let variants = roots
            .into_iter()
            .map(|(sort, value)| {
                extractor.extract_variants_with_sort(&mut termdag, value, nvariants, sort)
            })
            .collect();

        Ok(ExtractedTermVariants { termdag, variants })
    }

    /// Extracts the best term for one value using the default additive cost model.
    pub fn extract_value(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<(TermDag, TermId, DefaultCost), Error> {
        let sort_name = sort.name().to_owned();
        let mut extracted = self.extract_best(vec![(sort.clone(), value)])?;
        let root = extracted
            .terms
            .pop()
            .expect("one requested root produces one result")
            .ok_or_else(|| {
                Error::ExtractError(format!(
                    "Unable to find any valid extraction for sort {sort_name}"
                ))
            })?;
        Ok((extracted.termdag, root.term, root.cost))
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
        let mut rootsorts = func.func_type.input.clone();
        if include_output {
            rootsorts.push(func.func_type.output.clone());
        }
        let extractor =
            TreeExtractor::compute_costs_from_rootsorts(Some(rootsorts), self, DEFAULT_COST_MODEL);

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
                for (value, sort) in row.vals.iter().zip(&func.func_type.input) {
                    let term = extractor
                        .extract_best_with_sort(&mut termdag, *value, sort.clone())
                        .map(|extracted| extracted.term)
                        .unwrap_or_else(|| termdag.var("Unextractable".into()));
                    children.push(term);
                }
                inputs.push(termdag.app(sym.to_owned(), children));
                if include_output {
                    let value = row.vals[func.func_type.input.len()];
                    let term = extractor
                        .extract_best_with_sort(&mut termdag, value, func.func_type.output.clone())
                        .map(|extracted| extracted.term)
                        .unwrap_or_else(|| termdag.var("Unextractable".into()));
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
