use crate::ast::FunctionSubtype;
use crate::termdag::{TermDag, TermId};
use crate::util::{HashMap, HashSet};
use crate::*;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Shared primitive value cost hook for extraction cost models.
pub trait BaseCostModel<C: Cost> {
    /// Compute the cost of a (non-container) primitive value.
    ///
    /// Base values have no children, so there is no total-vs-marginal distinction.
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> C;
}

/// An interface for tree extraction cost models.
///
/// Cost models should usually make a term no cheaper than its subterms; that
/// rules out cycles in common extraction workloads. More specialized models may
/// assign lower costs to larger terms, but then the model is responsible for
/// avoiding negative-cost cycles and cyclic extracted terms.
pub trait TreeCostModel<C: Cost>: BaseCostModel<C> {
    /// The total cost of an e-node with its selected children.
    fn total_enode_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        enode: &Enode<'_>,
        child_costs: &[C],
    ) -> C;

    /// The total cost of a container value with its selected elements.
    ///
    /// The default cost for containers is the combined total cost of all elements.
    fn total_container_cost(
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
}

/// Values that can be accumulated in any insertion order.
///
/// Extraction cost accumulation relies on `identity` being neutral and
/// `combine` being associative and commutative for the values used during
/// extraction. Rust cannot enforce those laws, so custom cost implementations
/// are responsible for preserving them.
pub trait CommutativeMonoid: Clone {
    /// The neutral value for [`CommutativeMonoid::combine`], usually zero.
    fn identity() -> Self;

    /// Accumulates two values, usually addition.
    ///
    /// This operation must not overflow or panic when given large values.
    fn combine(self, other: &Self) -> Self;
}

/// Domain marker for values that can be used as extraction costs.
///
/// Implement [`CommutativeMonoid`] for custom cost types. `Cost` is provided
/// by a blanket impl so public extraction APIs can keep a domain-specific bound.
/// Raw floats can implement [`CommutativeMonoid`], but they do not satisfy this
/// bound because extractors need a total order. `OrderedFloat` is available for
/// approximate floating-point costs, but those can be order-sensitive because
/// floating-point addition is not strictly associative.
pub trait Cost: CommutativeMonoid + Ord + Eq + Debug {}

impl<T: CommutativeMonoid + Ord + Eq + Debug> Cost for T {}

macro_rules! cost_impl_int {
    ($($cost:ty),*) => {$(
        impl CommutativeMonoid for $cost {
            fn identity() -> Self { 0 }
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
        impl CommutativeMonoid for $cost {
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
cost_impl_num!(num::BigInt, num::BigRational);
use ordered_float::OrderedFloat;
cost_impl_num!(f32, f64, OrderedFloat<f32>, OrderedFloat<f64>);

pub type DefaultCost = u64;

/// A cost model that computes the cost by summing the cost of each node.
#[derive(Default, Clone)]
pub struct TreeAdditiveCostModel {}

impl BaseCostModel<DefaultCost> for TreeAdditiveCostModel {
    fn base_value_cost(&self, _egraph: &EGraph, _sort: &ArcSort, _value: Value) -> DefaultCost {
        1
    }
}

impl TreeCostModel<DefaultCost> for TreeAdditiveCostModel {
    fn total_enode_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        _enode: &Enode<'_>,
        child_costs: &[DefaultCost],
    ) -> DefaultCost {
        child_costs
            .iter()
            .fold(func.extraction_head_cost(egraph), |cost, child| {
                cost.combine(child)
            })
    }
}

#[derive(Clone, Debug)]
pub struct ExtractedTerm<C> {
    pub cost: C,
    pub term: TermId,
}

#[derive(Clone, Debug)]
pub struct ExtractedTerms<C> {
    /// Shared term storage for every extracted root.
    pub termdag: TermDag,
    /// One extraction result per requested root, in request order.
    ///
    /// `None` means that root is unextractable under the selected extraction
    /// policy.
    pub terms: Vec<Option<ExtractedTerm<C>>>,
}

#[derive(Clone, Debug)]
pub struct ExtractedTermVariants<C> {
    /// Shared term storage for every extracted variant.
    pub termdag: TermDag,
    /// Outer vector follows requested-root order; each inner vector contains that root's variants.
    pub variants: Vec<Vec<ExtractedTerm<C>>>,
}

/// The default, Bellman-Ford like extractor. This extractor is optimal for [`TreeCostModel`].
///
/// Note that this assumes optimal substructure in the cost model, that is, a lower-cost
/// subterm should always lead to a non-worse superterm, to guarantee the extracted term
/// being optimal under the given cost model.
/// If this is not followed, the extractor may panic on reconstruction
pub(crate) struct Extractor<C: Cost> {
    rootsorts: Vec<ArcSort>,
    funcs: Vec<String>,
    cost_model: Box<dyn TreeCostModel<C>>,
    costs: HashMap<String, HashMap<Value, C>>,
    topo_rnk_cnt: usize,
    topo_rnk: HashMap<String, HashMap<Value, usize>>,
    parent_edge: HashMap<String, HashMap<Value, (String, Vec<Value>)>>,
}

#[derive(Clone, Copy)]
enum ExtractionMode {
    Normal,
    Proof,
}

impl ExtractionMode {
    fn includes(self, func: &Function) -> bool {
        let constructor_or_view = func.decl.subtype == FunctionSubtype::Constructor
            || func.decl.term_constructor.is_some();
        match self {
            Self::Normal => {
                constructor_or_view && !func.decl.unextractable && !func.decl.internal_hidden
            }
            Self::Proof => constructor_or_view && func.decl.term_constructor.is_none(),
        }
    }
}

impl<C: Cost> Extractor<C> {
    /// Bulk of the computation happens at initialization time.
    /// The later extractions only reuses saved results.
    /// This means a new extractor must be created if the egraph changes.
    /// Holding a reference to the egraph would enforce this but prevents the extractor being reused.
    ///
    /// For convenience, if the rootsorts is `None`, it defaults to extract all extractable rootsorts.
    pub(crate) fn compute_costs_from_rootsorts(
        rootsorts: Option<Vec<ArcSort>>,
        egraph: &EGraph,
        cost_model: impl TreeCostModel<C> + 'static,
    ) -> Self {
        Self::compute_costs_from_rootsorts_internal(
            rootsorts,
            egraph,
            Box::new(cost_model),
            ExtractionMode::Normal,
        )
    }

    fn compute_costs_from_rootsorts_internal(
        rootsorts: Option<Vec<ArcSort>>,
        egraph: &EGraph,
        cost_model: Box<dyn TreeCostModel<C>>,
        mode: ExtractionMode,
    ) -> Self {
        // We filter out tables unreachable from the root sorts
        let extract_all_sorts = rootsorts.is_none();

        let mut rootsorts = rootsorts.unwrap_or_default();

        // Built a reverse index from output sort to function head symbols
        // Only include constructors (not regular functions) and respect unextractable flag
        let mut rev_index: HashMap<String, Vec<String>> = Default::default();
        for func in egraph.functions.iter() {
            if mode.includes(func.1) {
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
        let mut costs: HashMap<String, HashMap<Value, C>> = Default::default();
        let mut topo_rnk: HashMap<String, HashMap<Value, usize>> = Default::default();
        let mut parent_edge: HashMap<String, HashMap<Value, (String, Vec<Value>)>> =
            Default::default();

        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            let output_sort_name = func.extraction_output_sort().name();
            if !costs.contains_key(output_sort_name) {
                costs.insert(output_sort_name.to_owned(), Default::default());
                topo_rnk.insert(output_sort_name.to_owned(), Default::default());
                parent_edge.insert(output_sort_name.to_owned(), Default::default());
            }
        }

        let mut extractor = Extractor {
            rootsorts,
            funcs,
            cost_model,
            costs,
            topo_rnk_cnt: 0,
            topo_rnk,
            parent_edge,
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
                    .total_container_cost(egraph, sort, value, &ch_costs),
            )
        } else if sort.is_eq_sort() {
            self.costs.get(sort.name())?.get(&value).cloned()
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
        let output_idx = func.extraction_output_index();
        let enode = Enode {
            children: &row.vals[..output_idx],
            eclass: row.vals[output_idx],
            subsumed: row.subsumed,
        };
        Some(
            self.cost_model
                .total_enode_cost(egraph, func, &enode, &ch_costs),
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
            if let Some(t) = self.topo_rnk.get(sort.name()) {
                *t.get(&value).unwrap_or(&usize::MAX)
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
        let sorts = &func.schema.input;
        let num_children = func.extraction_num_children();
        row.vals
            .iter()
            .take(num_children)
            .zip(sorts.iter())
            .fold(0, |ret, (value, sort)| {
                usize::max(ret, self.compute_topo_rnk_node(egraph, *value, sort))
            })
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
    fn bellman_ford(&mut self, egraph: &EGraph) {
        let mut ensure_fixpoint = false;

        let funcs = self.funcs.clone();

        while !ensure_fixpoint {
            ensure_fixpoint = true;

            for func_name in funcs.iter() {
                let func = egraph.functions.get(func_name).unwrap();
                let target_sort = func.extraction_output_sort();

                let output_idx = func.extraction_output_index();
                let relax_hyperedge = |row: egglog_bridge::ScanEntry| {
                    if !row.subsumed {
                        let target = &row.vals[output_idx];
                        let mut updated = false;
                        if let Some(new_cost) = self.compute_cost_hyperedge(egraph, &row, func) {
                            match self
                                .costs
                                .get_mut(target_sort.name())
                                .unwrap()
                                .entry(*target)
                            {
                                HEntry::Vacant(e) => {
                                    updated = true;
                                    e.insert(new_cost);
                                }
                                HEntry::Occupied(mut e) => {
                                    if new_cost < *(e.get()) {
                                        updated = true;
                                        e.insert(new_cost);
                                    }
                                }
                            }
                        }
                        // record the chronological order of the updates
                        // which serves as a topological order that avoids cycles
                        // even when a term has a cost equal to its subterms
                        if updated {
                            ensure_fixpoint = false;
                            self.topo_rnk_cnt += 1;
                            self.topo_rnk
                                .get_mut(target_sort.name())
                                .unwrap()
                                .insert(*target, self.topo_rnk_cnt);
                        }
                    }
                };

                egraph.backend.for_each(func.backend_id, relax_hyperedge);
            }
        }

        // Save the edges for reconstruction
        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            let target_sort = func.extraction_output_sort();
            let output_idx = func.extraction_output_index();

            let save_best_parent_edge = |row: egglog_bridge::ScanEntry| {
                if !row.subsumed {
                    let target = &row.vals[output_idx];
                    if let Some(best_cost) = self.costs.get(target_sort.name()).unwrap().get(target)
                        && Some(best_cost.clone())
                            == self.compute_cost_hyperedge(egraph, &row, func)
                    {
                        // one of the possible best parent edges
                        let target_topo_rnk = *self
                            .topo_rnk
                            .get(target_sort.name())
                            .unwrap()
                            .get(target)
                            .unwrap();
                        if target_topo_rnk > self.compute_topo_rnk_hyperedge(egraph, &row, func) {
                            // one of the parent edges that avoids cycles
                            if let HEntry::Vacant(e) = self
                                .parent_edge
                                .get_mut(target_sort.name())
                                .unwrap()
                                .entry(*target)
                            {
                                e.insert((func.decl.name.clone(), row.vals.to_vec()));
                            }
                        }
                    }
                }
            };

            egraph
                .backend
                .for_each(func.backend_id, save_best_parent_edge);
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
            let (func_name, hyperedge) = self
                .parent_edge
                .get(sort.name())
                .unwrap()
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
    pub(crate) fn extract_best_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: ArcSort,
    ) -> Option<ExtractedTerm<C>> {
        // Canonicalize the value using the union-find if available (for term-encoding mode)
        let canonical_value = self.find_canonical(egraph, value, &sort);

        match self.compute_cost_node(egraph, canonical_value, &sort) {
            Some(best_cost) => {
                log::debug!("Best cost for the extract root: {best_cost:?}");

                let term = self.reconstruct_termdag_node(egraph, termdag, canonical_value, &sort);

                Some(ExtractedTerm {
                    cost: best_cost,
                    term,
                })
            }
            None => {
                log::error!("Unextractable root {value:?} with sort {sort:?}",);
                None
            }
        }
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
    pub(crate) fn extract_variants_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: ArcSort,
    ) -> Vec<ExtractedTerm<C>> {
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

            let mut res: Vec<ExtractedTerm<C>> = Vec::new();
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
            if let Some(res) = self.extract_best_with_sort(egraph, termdag, value, sort) {
                vec![res]
            } else {
                vec![]
            }
        }
    }
}

/// Extract proof terms with proof-internal visibility rules.
///
/// This stays separate from [`EGraph::extract_best`] because normal extraction
/// respects `:unextractable` and may use view tables, while proof extraction
/// must extract hidden proof term tables with their original names.
pub(crate) fn extract_best_for_proofs(
    egraph: &EGraph,
    roots: Vec<(ArcSort, Value)>,
) -> Result<ExtractedTerms<DefaultCost>, Error> {
    let extracted = egraph.extract_best_with_mode(
        roots,
        TreeAdditiveCostModel::default(),
        ExtractionMode::Proof,
    )?;

    if extracted.terms.iter().any(Option::is_none) {
        return Err(Error::ExtractError(
            "Unable to find any valid proof extraction".to_string(),
        ));
    }

    Ok(extracted)
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
                .unwrap_or(1)
        } else {
            self.decl.cost.unwrap_or(1)
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
    /// Extract the best tree term for each requested `(sort, value)` root.
    ///
    /// This is the normal user extraction path: it respects `:unextractable`
    /// and hidden internal functions.
    pub fn extract_best<C: Cost, M: TreeCostModel<C> + 'static>(
        &self,
        roots: Vec<(ArcSort, Value)>,
        cost_model: M,
    ) -> Result<ExtractedTerms<C>, Error> {
        self.extract_best_with_mode(roots, cost_model, ExtractionMode::Normal)
    }

    fn extract_best_with_mode<C: Cost, M: TreeCostModel<C> + 'static>(
        &self,
        roots: Vec<(ArcSort, Value)>,
        cost_model: M,
        mode: ExtractionMode,
    ) -> Result<ExtractedTerms<C>, Error> {
        let rootsorts = roots.iter().map(|(sort, _)| sort.clone()).collect();
        let extractor = Extractor::compute_costs_from_rootsorts_internal(
            Some(rootsorts),
            self,
            Box::new(cost_model),
            mode,
        );
        let mut termdag = TermDag::default();
        let extracted_roots = roots
            .into_iter()
            .map(|(sort, value)| extractor.extract_best_with_sort(self, &mut termdag, value, sort))
            .collect();

        Ok(ExtractedTerms {
            termdag,
            terms: extracted_roots,
        })
    }

    /// Extract up to `nvariants` default tree root variants for each requested root.
    pub fn extract_variants<C: Cost, M: TreeCostModel<C> + 'static>(
        &self,
        roots: Vec<(ArcSort, Value)>,
        nvariants: usize,
        cost_model: M,
    ) -> Result<ExtractedTermVariants<C>, Error> {
        let rootsorts = roots.iter().map(|(sort, _)| sort.clone()).collect();
        let extractor = Extractor::compute_costs_from_rootsorts(Some(rootsorts), self, cost_model);
        let mut termdag = TermDag::default();
        let variants = roots
            .into_iter()
            .map(|(sort, value)| {
                extractor.extract_variants_with_sort(self, &mut termdag, value, nvariants, sort)
            })
            .collect();

        Ok(ExtractedTermVariants { termdag, variants })
    }

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
    pub fn extract_value_with_cost_model<CM: TreeCostModel<DefaultCost> + 'static>(
        &self,
        sort: &ArcSort,
        value: Value,
        cost_model: CM,
    ) -> Result<(TermDag, TermId, DefaultCost), Error> {
        let ExtractedTerms { termdag, mut terms } =
            self.extract_best(vec![(sort.clone(), value)], cost_model)?;
        let extracted = terms
            .pop()
            .expect("extract_best returns one result for one root")
            .ok_or_else(|| {
                Error::ExtractError(format!(
                    "Unable to find any valid extraction for sort {}",
                    sort.name()
                ))
            })?;
        Ok((termdag, extracted.term, extracted.cost))
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
        let mut cell_sorts = func.schema.input.clone();
        if include_output {
            cell_sorts.push(func.schema.output.clone());
        }
        let mut rows: Vec<Vec<Value>> = Vec::new();
        let mut inputs: Vec<TermId> = Vec::new();
        let mut output: Option<Vec<TermId>> = if include_output {
            Some(Vec::new())
        } else {
            None
        };

        let extract_row = |row: egglog_bridge::ScanEntry| {
            if rows.len() < n {
                rows.push(row.vals.iter().take(cell_sorts.len()).copied().collect());
                true
            } else {
                false
            }
        };

        self.backend.for_each_while(func.backend_id, extract_row);

        let roots = rows
            .iter()
            .flat_map(|row| {
                row.iter()
                    .zip(&cell_sorts)
                    .map(|(value, sort)| (sort.clone(), *value))
            })
            .collect();
        let ExtractedTerms { mut termdag, terms } =
            self.extract_best(roots, TreeAdditiveCostModel::default())?;
        let mut terms = terms.into_iter();

        for row in rows {
            let mut children: Vec<TermId> = Vec::new();
            for _ in row.iter().take(func.schema.input.len()) {
                let term = terms
                    .next()
                    .expect("one extraction result per displayed input")
                    .map(|extracted| extracted.term)
                    .unwrap_or_else(|| termdag.var("Unextractable".into()));
                children.push(term);
            }
            inputs.push(termdag.app(sym.to_owned(), children));
            if include_output {
                let term = terms
                    .next()
                    .expect("one extraction result per displayed output")
                    .map(|extracted| extracted.term)
                    .unwrap_or_else(|| termdag.var("Unextractable".into()));
                output.as_mut().unwrap().push(term);
            }
        }

        Ok((inputs, output, termdag))
    }
}
