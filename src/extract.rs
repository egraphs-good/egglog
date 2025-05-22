use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::{HashMap, HashSet};
use crate::{ArcSort, EGraph, Error, Function, HEntry, Id, Value};
use queues::*;

pub type Cost = usize;

#[derive(Debug)]
pub(crate) struct Node<'a> {
    sym: Symbol,
    func: &'a Function,
    inputs: &'a [Value],
}

pub struct Extractor<'a> {
    pub costs: HashMap<Id, (Cost, Term)>,
    ctors: Vec<Symbol>,
    egraph: &'a EGraph,
}

impl EGraph {
    /// This example uses [`EGraph::extract`] to extract a term. The example is
    /// trivial, as there is only a single variant of the expression in the
    /// egraph.
    /// ```
    /// use egglog::{EGraph, TermDag};
    /// let mut egraph = EGraph::default();
    /// egraph
    ///     .parse_and_run_program(
    ///         None,
    ///         "(datatype Op (Add i64 i64))
    ///          (let expr (Add 1 1))",
    ///     )
    ///     .unwrap();
    /// let mut termdag = TermDag::default();
    /// let (sort, value) = egraph.eval_expr(&egglog::var!("expr")).unwrap();
    /// let (_, extracted) = egraph.extract(value, &mut termdag, &sort).unwrap();
    /// assert_eq!(termdag.to_string(&extracted), "(Add 1 1)");
    /// ```
    pub fn extract(
        &self,
        value: Value,
        termdag: &mut TermDag,
        arcsort: &ArcSort,
    ) -> Result<(Cost, Term), Error> {
        let extractor = Extractor::new(self, termdag);
        extractor.find_best(value, termdag, arcsort).ok_or_else(|| {
            log::error!("No cost for {:?}", value);
            for func in self.functions.values() {
                for (inputs, output) in func.nodes.iter(false) {
                    if output.value == value {
                        log::error!("Found unextractable function: {:?}", func.decl.name);
                        log::error!("Inputs: {:?}", inputs);

                        assert_eq!(inputs.len(), func.schema.input.len());
                        log::error!(
                            "{:?}",
                            inputs
                                .iter()
                                .zip(&func.schema.input)
                                .map(|(input, sort)| extractor
                                    .costs
                                    .get(&extractor.egraph.find(sort, *input).bits))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
            Error::ExtractError(String::new())
        })
    }

    pub fn extract_variants(
        &mut self,
        sort: &ArcSort,
        value: Value,
        limit: usize,
        termdag: &mut TermDag,
    ) -> Vec<Term> {
        let output_sort = sort.name();
        let output_value = self.find(sort, value);
        let ext = &Extractor::new(self, termdag);
        ext.ctors
            .iter()
            .flat_map(|&sym| {
                let func = &self.functions[&sym];
                if !func.schema.output.is_eq_sort() {
                    return vec![];
                }
                assert!(func.schema.output.is_eq_sort());

                func.nodes
                    .iter(false)
                    .filter(|&(_, output)| {
                        func.schema.output.name() == output_sort && output.value == output_value
                    })
                    .map(|(inputs, _output)| {
                        let node = Node { sym, func, inputs };
                        ext.expr_from_node(&node, termdag).expect(
                            "extract_variants should be called after extractor initialization",
                        )
                    })
                    .collect()
            })
            .take(limit)
            .collect()
    }
}

impl<'a> Extractor<'a> {
    pub fn new(egraph: &'a EGraph, termdag: &mut TermDag) -> Self {
        let mut extractor = Extractor {
            costs: HashMap::default(),
            egraph,
            ctors: vec![],
        };

        // only consider "extractable" functions
        extractor.ctors.extend(
            egraph
                .functions
                .keys()
                .filter(|func| !egraph.functions.get(*func).unwrap().decl.unextractable)
                .cloned(),
        );

        log::debug!("Extracting from ctors: {:?}", extractor.ctors);
        extractor.find_costs(termdag);
        extractor
    }

    fn expr_from_node(&self, node: &Node, termdag: &mut TermDag) -> Option<Term> {
        let mut children = vec![];

        let values = node.inputs;
        let arcsorts = &node.func.schema.input;
        assert_eq!(values.len(), arcsorts.len());

        for (value, arcsort) in values.iter().zip(arcsorts) {
            children.push(self.find_best(*value, termdag, arcsort)?.1)
        }

        Some(termdag.app(node.sym, children))
    }

    pub fn find_best(
        &self,
        value: Value,
        termdag: &mut TermDag,
        sort: &ArcSort,
    ) -> Option<(Cost, Term)> {
        if sort.is_eq_sort() {
            let id = self.egraph.find(sort, value).bits;
            let (cost, node) = self.costs.get(&id)?.clone();
            Some((cost, node))
        } else {
            let (cost, node) = sort.extract_term(self.egraph, value, self, termdag)?;
            Some((cost, node))
        }
    }

    fn node_total_cost(
        &mut self,
        function: &Function,
        children: &[Value],
        termdag: &mut TermDag,
    ) -> Option<(Vec<Term>, Cost)> {
        let mut cost = function.decl.cost.unwrap_or(1);
        let types = &function.schema.input;
        let mut terms: Vec<Term> = vec![];
        for (ty, value) in types.iter().zip(children) {
            let (term_cost, term) = self.find_best(*value, termdag, ty)?;
            terms.push(term.clone());
            cost = cost.saturating_add(term_cost);
        }
        Some((terms, cost))
    }

    fn find_costs(&mut self, termdag: &mut TermDag) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for sym in self.ctors.clone() {
                let func = &self.egraph.functions[&sym];
                if func.schema.output.is_eq_sort() {
                    for (inputs, output) in func.nodes.iter(false) {
                        if let Some((term_inputs, new_cost)) =
                            self.node_total_cost(func, inputs, termdag)
                        {
                            let make_new_pair = || (new_cost, termdag.app(sym, term_inputs));

                            let id = self.egraph.find(&func.schema.output, output.value).bits;
                            match self.costs.entry(id) {
                                HEntry::Vacant(e) => {
                                    did_something = true;
                                    e.insert(make_new_pair());
                                }
                                HEntry::Occupied(mut e) => {
                                    if new_cost < e.get().0 {
                                        did_something = true;
                                        e.insert(make_new_pair());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub trait CostModel {
    fn fold(&self, head: Symbol, children_cost: &[Cost], head_cost: Cost) -> Cost;

    fn enode_cost(
        &self,
        egraph: &EGraph,
        func: &Function,
        row: &egglog_bridge::FunctionRow,
    ) -> Cost;

    fn container_primitive(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: core_relations::Value,
        element_costs: &[Cost],
    ) -> Cost;

    fn leaf_primitive(&self, egraph: &EGraph, sort: &ArcSort, value: core_relations::Value)
        -> Cost;
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
        value: core_relations::Value,
        element_costs: &[Cost],
    ) -> Cost {
        sort.default_container_cost(egraph.backend.containers(), value, element_costs)
    }

    fn leaf_primitive(
        &self,
        egraph: &EGraph,
        sort: &ArcSort,
        value: core_relations::Value,
    ) -> Cost {
        sort.default_leaf_cost(egraph.backend.primitives(), value)
    }
}

pub struct ExtractorAlter {
    rootsorts: Vec<ArcSort>,
    funcs: Vec<Symbol>,
    cost_model: Box<dyn CostModel>,
    costs: HashMap<Symbol, HashMap<core_relations::Value, Cost>>,
    parent_edge:
        HashMap<Symbol, HashMap<core_relations::Value, (Symbol, Vec<core_relations::Value>)>>,
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
            if func.1.is_extractable() {
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
        let mut q: Queue<ArcSort> = Queue::new();
        let mut seen: HashSet<Symbol> = Default::default();
        for rootsort in rootsorts.iter() {
            let _ = q.add(rootsort.clone());
            seen.insert(rootsort.name());
        }

        let mut funcs_set: HashSet<Symbol> = Default::default();
        let mut funcs: Vec<Symbol> = Vec::new();
        while q.size() > 0 {
            let sort = q.remove().unwrap();
            if sort.is_container_sort() {
                let inner_sorts = sort.inner_sorts();
                for s in inner_sorts {
                    if !seen.contains(&s.name()) {
                        let _ = q.add(s.clone());
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
                                    let _ = q.add(ch.clone());
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
        let mut costs: HashMap<Symbol, HashMap<core_relations::Value, Cost>> = Default::default();
        let mut parent_edge: HashMap<
            Symbol,
            HashMap<core_relations::Value, (Symbol, Vec<core_relations::Value>)>,
        > = Default::default();

        for func_name in funcs.iter() {
            let func = egraph.functions.get(func_name).unwrap();
            if !costs.contains_key(&func.schema.output.name()) {
                debug_assert!(func.schema.output.is_eq_sort());
                costs.insert(func.schema.output.name(), Default::default());
                parent_edge.insert(func.schema.output.name(), Default::default());
            }
        }

        let mut extractor = ExtractorAlter {
            rootsorts,
            funcs,
            cost_model: Box::new(cost_model),
            costs,
            parent_edge,
        };

        extractor.bellman_ford(egraph);

        extractor
    }

    /// Compute the cost of a single enode
    /// Recurse if container
    /// Returns None if contains an undefined eqsort term (potentially after unfolding)
    fn compute_cost_node(
        &self,
        egraph: &EGraph,
        value: &core_relations::Value,
        sort: &ArcSort,
    ) -> Option<Cost> {
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

    fn bellman_ford(&mut self, egraph: &EGraph) {
        // We use Bellman-Ford to compute the costs of the relevant eq sorts' terms

        let mut ensure_fixpoint = false;
        let mut reconstruction_round = false;

        let funcs = self.funcs.clone();

        // Runs an extra round to copy the best hyperedges
        while !ensure_fixpoint || reconstruction_round {
            ensure_fixpoint = true;

            for func_name in funcs.iter() {
                let func = egraph.functions.get(func_name).unwrap();
                let target_sort = func.schema.output.clone();

                let relax_hyperedge = |row: egglog_bridge::FunctionRow| {
                    log::debug!("Relaxing a new hyperedge: {:?}", row);
                    if !row.subsumed {
                        let target = row.vals.last().unwrap();
                        if let Some(new_cost) = self.compute_cost_hyperedge(egraph, &row, func) {
                            if !reconstruction_round {
                                match self
                                    .costs
                                    .get_mut(&target_sort.name())
                                    .unwrap()
                                    .entry(*target)
                                {
                                    HEntry::Vacant(e) => {
                                        ensure_fixpoint = false;
                                        e.insert(new_cost);
                                    }
                                    HEntry::Occupied(mut e) => {
                                        if new_cost < *(e.get()) {
                                            ensure_fixpoint = false;
                                            e.insert(new_cost);
                                        }
                                    }
                                }
                            } else if new_cost
                                == *self
                                    .costs
                                    .get(&target_sort.name())
                                    .unwrap()
                                    .get(target)
                                    .unwrap()
                            {
                                // one of the possible best parent edges
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
                };

                egraph
                    .backend
                    .dump_table(func.new_backend_id, relax_hyperedge);
            }

            if ensure_fixpoint {
                reconstruction_round = !reconstruction_round;
            }
        }
    }

    fn reconstruct_termdag_node(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: &core_relations::Value,
        sort: &ArcSort,
    ) -> Term {
        if sort.is_container_sort() {
            let elements = sort.inner_values(egraph.backend.containers(), value);
            let mut ch_terms: Vec<Term> = Vec::new();
            for ch in elements.iter() {
                ch_terms.push(self.reconstruct_termdag_node(egraph, termdag, &ch.1, &ch.0));
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
                ch_terms.push(self.reconstruct_termdag_node(egraph, termdag, value, sort));
            }
            termdag.app(*func_name, ch_terms)
        } else {
            // Primitive
            sort.reconstruct_termdag_leaf(egraph.backend.primitives(), value, termdag)
        }
    }

    pub fn extract_best_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: core_relations::Value,
        sort: ArcSort,
    ) -> Option<(Cost, Term)> {
        // can be more accurate by considering all sorts extracted but this is ok
        debug_assert!(self.rootsorts.iter().any(|s| { s.name() == sort.name() }));
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

    pub fn extract_best(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: core_relations::Value,
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

    /// We extract variants by selecting nvairants enodes with the lowest cost from the root eclass
    pub fn extract_variants_with_sort(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: core_relations::Value,
        nvariants: usize,
        sort: ArcSort,
    ) -> Vec<(Cost, Term)> {
        debug_assert!(self.rootsorts.iter().any(|s| { s.name() == sort.name() }));

        if sort.is_eq_sort() {
            let mut root_variants: Vec<(Cost, Symbol, Vec<core_relations::Value>)> = Vec::new();

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
                    .dump_table(func.new_backend_id, find_root_variants);
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

    pub fn extract_variants(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: core_relations::Value,
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
