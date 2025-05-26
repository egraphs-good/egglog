use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::{HashMap, HashSet};
use crate::*;
use std::collections::VecDeque;

pub type Cost = usize;

/// An interface for custom cost model
/// For common case usage, the model should guarantee a term has a no-smaller cost
/// than its subterms to avoid cycles in the extracted terms.
/// For more niech usage, a term can have a cost less than its subterms.
/// As long as there is no negative cost cycle,
/// the default extractor is guaranteed to terminate in computing the costs.
/// However, the user needs to be careful to guarantee acyclicity in the extracted terms.
pub trait CostModel {
    /// Compute the cost of term given the costs of the head symbol and the subterms
    fn fold(&self, head: Symbol, children_cost: &[Cost], head_cost: Cost) -> Cost;

    /// Compute the cost of a particular enode.
    fn enode_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        row: &egglog_bridge::FunctionRow,
    ) -> Cost;

    /// Compute the cost of a container value given the costs of its elements
    fn container_primitive(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        element_costs: &[Cost],
    ) -> Cost;

    /// Compute the cost of a primitive, non-container, value
    fn leaf_primitive(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> Cost;
}

#[derive(Default, Clone)]
pub struct TreeAdditiveCostModel {}

impl CostModel for TreeAdditiveCostModel {
    fn fold(&self, _head: Symbol, children_cost: &[Cost], head_cost: Cost) -> Cost {
        children_cost
            .iter()
            .fold(head_cost, |s, c| s.saturating_add(*c))
    }

    fn enode_cost(
        &self,
        _egraph: &EGraph,
        func: &Function,
        _row: &egglog_bridge::FunctionRow,
    ) -> Cost {
        func.decl.cost.unwrap_or(1)
    }

    fn container_primitive(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: Value,
        element_costs: &[Cost],
    ) -> Cost {
        sort.default_container_cost(egraph.backend.containers(), value, element_costs)
    }

    fn leaf_primitive(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> Cost {
        sort.default_leaf_cost(egraph.backend.primitives(), value)
    }
}

pub struct ExtractorAlter {
    rootsorts: Vec<ArcSort>,
    funcs: Vec<Symbol>,
    cost_model: Box<dyn CostModel>,
    costs: HashMap<Symbol, HashMap<Value, Cost>>,
    topo_rnk_cnt: usize,
    topo_rnk: HashMap<Symbol, HashMap<Value, usize>>,
    parent_edge: HashMap<Symbol, HashMap<Value, (Symbol, Vec<Value>)>>,
}

impl ExtractorAlter {
    /// Bulk of the computation happens at initialization time.
    /// The later extractions only reuses saved results.
    /// This means a new extractor must be created if the egraph changes.
    /// Holding a reference to the egraph would enforce this but prevents the extractor being reused.
    /// For convenience, if the rootsorts is None, it defaults to extract all extractable rootsorts.
    pub fn compute_costs_from_rootsorts(
        rootsorts: Option<Vec<ArcSort>>,
        egraph: &EGraph,
        cost_model: impl CostModel + 'static,
    ) -> Self {
        // We filter out tables unreachable from the root sorts
        let extract_all_sorts = rootsorts.is_none();

        let mut rootsorts = rootsorts.unwrap_or_default();

        // Built a reverse index from output sort to function head symbols
        let mut rev_index: HashMap<Symbol, Vec<Symbol>> = Default::default();
        for func in egraph.functions.iter() {
            if !func.1.decl.unextractable {
                let func_name = *func.0;
                let output_sort_name = func.1.schema.output.name();
                if let Some(v) = rev_index.get_mut(&output_sort_name) {
                    v.push(func_name);
                } else {
                    rev_index.insert(output_sort_name, vec![func_name]);
                    if extract_all_sorts {
                        rootsorts.push(func.1.schema.output.clone());
                    }
                }
            }
        }

        // Do a BFS to find reachable tables
        let mut q: VecDeque<ArcSort> = VecDeque::new();
        let mut seen: HashSet<Symbol> = Default::default();
        for rootsort in rootsorts.iter() {
            q.push_back(rootsort.clone());
            seen.insert(rootsort.name());
        }

        let mut funcs_set: HashSet<Symbol> = Default::default();
        let mut funcs: Vec<Symbol> = Vec::new();
        while !q.is_empty() {
            let sort = q.pop_front().unwrap();
            if sort.is_container_sort() {
                let inner_sorts = sort.inner_sorts();
                for s in inner_sorts {
                    if !seen.contains(&s.name()) {
                        q.push_back(s.clone());
                        seen.insert(s.name());
                    }
                }
            } else if sort.is_eq_sort() {
                if let Some(head_symbols) = rev_index.get(&sort.name()) {
                    for h in head_symbols {
                        if !funcs_set.contains(h) {
                            let func = egraph.functions.get(h).unwrap();
                            for ch in &func.schema.input {
                                let ch_name = ch.name();
                                if !seen.contains(&ch_name) {
                                    q.push_back(ch.clone());
                                    seen.insert(ch_name);
                                }
                            }
                            funcs_set.insert(*h);
                            funcs.push(*h);
                        }
                    }
                }
            }
        }

        // Initialize the tables to have the reachable entries
        let mut costs: HashMap<Symbol, HashMap<Value, Cost>> = Default::default();
        let mut topo_rnk: HashMap<Symbol, HashMap<Value, usize>> = Default::default();
        let mut parent_edge: HashMap<Symbol, HashMap<Value, (Symbol, Vec<Value>)>> =
            Default::default();

        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            if !costs.contains_key(&func.schema.output.name()) {
                debug_assert!(func.schema.output.is_eq_sort());
                costs.insert(func.schema.output.name(), Default::default());
                topo_rnk.insert(func.schema.output.name(), Default::default());
                parent_edge.insert(func.schema.output.name(), Default::default());
            }
        }

        let mut extractor = ExtractorAlter {
            rootsorts,
            funcs,
            cost_model: Box::new(cost_model),
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
    fn compute_cost_node(&self, egraph: &EGraph, value: &Value, sort: &ArcSort) -> Option<Cost> {
        if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.containers(), value);
            let mut ch_costs: Vec<Cost> = Vec::new();
            for ch in elements.iter() {
                if let Some(c) = self.compute_cost_node(egraph, &ch.1, &ch.0) {
                    ch_costs.push(c);
                } else {
                    return None;
                }
            }
            Some(
                self.cost_model
                    .container_primitive(egraph, sort, *value, &ch_costs),
            )
        } else if sort.is_eq_sort() {
            if self
                .costs
                .get(&sort.name())
                .is_some_and(|t| t.get(value).is_some())
            {
                Some(*self.costs.get(&sort.name()).unwrap().get(value).unwrap())
            } else {
                None
            }
        } else {
            // Primitive
            Some(self.cost_model.leaf_primitive(egraph, sort, *value))
        }
    }

    /// A row in a [constructor] table is an hyperedge from the set of input terms to the constructed output term.
    fn compute_cost_hyperedge(
        &self,
        egraph: &EGraph,
        row: &egglog_bridge::FunctionRow,
        func: &Function,
    ) -> Option<Cost> {
        let mut ch_costs: Vec<Cost> = Vec::new();
        let sorts = &func.schema.input;
        //log::debug!("compute_cost_hyperedge head {} sorts {:?}", head, sorts);
        // Relying on .zip to truncate the values
        for (value, sort) in row.vals.iter().zip(sorts.iter()) {
            if let Some(c) = self.compute_cost_node(egraph, value, sort) {
                ch_costs.push(c);
            } else {
                return None;
            }
        }
        Some(self.cost_model.fold(
            func.decl.name,
            &ch_costs,
            self.cost_model.enode_cost(egraph, func, row),
        ))
    }

    fn compute_topo_rnk_node(&self, egraph: &EGraph, value: &Value, sort: &ArcSort) -> usize {
        if sort.is_container_sort() {
            sort.inner_values(egraph.backend.containers(), value)
                .iter()
                .fold(0, |ret, (sort, value)| {
                    usize::max(ret, self.compute_topo_rnk_node(egraph, value, sort))
                })
        } else if sort.is_eq_sort() {
            if let Some(t) = self.topo_rnk.get(&sort.name()) {
                *t.get(value).unwrap_or(&usize::MAX)
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
        row: &egglog_bridge::FunctionRow,
        func: &Function,
    ) -> usize {
        let sorts = &func.schema.input;
        row.vals
            .iter()
            .zip(sorts.iter())
            .fold(0, |ret, (value, sort)| {
                usize::max(ret, self.compute_topo_rnk_node(egraph, value, sort))
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
                let target_sort = func.schema.output.clone();

                let relax_hyperedge = |row: egglog_bridge::FunctionRow| {
                    log::debug!("Relaxing a new hyperedge: {:?}", row);
                    if !row.subsumed {
                        let target = row.vals.last().unwrap();
                        let mut updated = false;
                        if let Some(new_cost) = self.compute_cost_hyperedge(egraph, &row, func) {
                            match self
                                .costs
                                .get_mut(&target_sort.name())
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
                                .get_mut(&target_sort.name())
                                .unwrap()
                                .insert(*target, self.topo_rnk_cnt);
                        }
                    }
                };

                egraph.backend.dump_table(func.backend_id, relax_hyperedge);
            }
        }

        // Save the edges for reconstruction
        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            let target_sort = func.schema.output.clone();

            let save_best_parent_edge = |row: egglog_bridge::FunctionRow| {
                if !row.subsumed {
                    let target = row.vals.last().unwrap();
                    if let Some(best_cost) =
                        self.costs.get(&target_sort.name()).unwrap().get(target)
                    {
                        if Some(*best_cost) == self.compute_cost_hyperedge(egraph, &row, func) {
                            // one of the possible best parent edges
                            let target_topo_rnk = *self
                                .topo_rnk
                                .get(&target_sort.name())
                                .unwrap()
                                .get(target)
                                .unwrap();
                            if target_topo_rnk > self.compute_topo_rnk_hyperedge(egraph, &row, func)
                            {
                                // one of the parent edges that avoids cycles
                                if let HEntry::Vacant(e) = self
                                    .parent_edge
                                    .get_mut(&target_sort.name())
                                    .unwrap()
                                    .entry(*target)
                                {
                                    e.insert((func.decl.name, row.vals.to_vec()));
                                }
                            }
                        }
                    }
                }
            };

            egraph
                .backend
                .dump_table(func.backend_id, save_best_parent_edge);
        }
    }

    /// This recursively reconstruct the termdag that gives the minimum cost for eclass value.
    fn reconstruct_termdag_node(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: &Value,
        sort: &ArcSort,
    ) -> Term {
        self.reconstruct_termdag_node_helper(egraph, termdag, value, sort, &mut Default::default())
    }

    fn reconstruct_termdag_node_helper(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: &Value,
        sort: &ArcSort,
        cache: &mut HashMap<(Value, Symbol), Term>,
    ) -> Term {
        let key = (*value, sort.name());
        if let Some(term) = cache.get(&key) {
            return term.clone();
        }

        let term = if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.containers(), value);
            let mut ch_terms: Vec<Term> = Vec::new();
            for ch in elements.iter() {
                ch_terms.push(
                    self.reconstruct_termdag_node_helper(egraph, termdag, &ch.1, &ch.0, cache),
                );
            }
            sort.reconstruct_termdag_container(
                egraph.backend.containers(),
                value,
                termdag,
                ch_terms,
            )
        } else if sort.is_eq_sort() {
            let (func_name, hyperedge) = self
                .parent_edge
                .get(&sort.name())
                .unwrap()
                .get(value)
                .unwrap();
            let mut ch_terms: Vec<Term> = Vec::new();
            let ch_sorts = &egraph.functions.get(func_name).unwrap().schema.input;
            for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()) {
                ch_terms.push(
                    self.reconstruct_termdag_node_helper(egraph, termdag, value, sort, cache),
                );
            }
            termdag.app(*func_name, ch_terms)
        } else {
            // Primitive
            sort.reconstruct_termdag_leaf(egraph.backend.primitives(), value, termdag)
        };

        cache.insert(key, term.clone());
        term
    }

    /// This expects the sort to be already computed
    /// can be one of the rootsorts, or reachable from rootsorts, or primitives, or containers of computed sorts
    pub fn extract_best_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: ArcSort,
    ) -> Option<(Cost, Term)> {
        match self.compute_cost_node(egraph, &value, &sort) {
            Some(best_cost) => {
                log::debug!("Best cost for the extract root: {:?}", best_cost);

                let term = self.reconstruct_termdag_node(egraph, termdag, &value, &sort);

                Some((best_cost, term))
            }
            None => {
                log::error!("Unextractable root {:?} with sort {:?}", value, sort,);
                None
            }
        }
    }

    /// This expects the value to be of the single sort the extractor has been initialized with
    pub fn extract_best(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
    ) -> Option<(Cost, Term)> {
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

    /// This extract variants by selecting nvairants enodes with the lowest cost from the root eclass.
    pub fn extract_variants_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: ArcSort,
    ) -> Vec<(Cost, Term)> {
        debug_assert!(self.rootsorts.iter().any(|s| { s.name() == sort.name() }));

        if sort.is_eq_sort() {
            let mut root_variants: Vec<(Cost, Symbol, Vec<Value>)> = Vec::new();

            let mut root_funcs: Vec<Symbol> = Vec::new();

            for func_name in self.funcs.iter() {
                // Need an eq on sorts
                if sort.name()
                    == egraph
                        .functions
                        .get(func_name)
                        .unwrap()
                        .schema
                        .output
                        .name()
                {
                    root_funcs.push(*func_name);
                }
            }

            for func_name in root_funcs.iter() {
                let func = egraph.functions.get(func_name).unwrap();

                let find_root_variants = |row: egglog_bridge::FunctionRow| {
                    if !row.subsumed {
                        let target = row.vals.last().unwrap();
                        if *target == value {
                            let cost = self.compute_cost_hyperedge(egraph, &row, func).unwrap();
                            root_variants.push((cost, *func_name, row.vals.to_vec()));
                        }
                    }
                };

                egraph
                    .backend
                    .dump_table(func.backend_id, find_root_variants);
            }

            let mut res: Vec<(Cost, Term)> = Vec::new();
            root_variants.sort();
            root_variants.truncate(nvariants);
            for (cost, func_name, hyperedge) in root_variants {
                let mut ch_terms: Vec<Term> = Vec::new();
                let ch_sorts = &egraph.functions.get(&func_name).unwrap().schema.input;
                // zip truncates the row
                for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()) {
                    ch_terms.push(self.reconstruct_termdag_node(egraph, termdag, value, sort));
                }
                res.push((cost, termdag.app(func_name, ch_terms)));
            }

            res
        } else {
            log::warn!("extracting multiple variants for containers or primitives is not implemented, returning a single variant.");
            if let Some(res) = self.extract_best_with_sort(egraph, termdag, value, sort) {
                vec![res]
            } else {
                vec![]
            }
        }
    }

    /// This expects the value to be of the single sort the extractor has been initialized with
    pub fn extract_variants(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
    ) -> Vec<(Cost, Term)> {
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
