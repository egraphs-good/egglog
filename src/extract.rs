use crate::ast::Symbol;
use crate::function::ResolvedSchema;
use crate::termdag::{Term, TermDag};
use crate::util::{HashMap, HashSet};
use crate::IndexMap;
use crate::{ArcSort, EGraph, Error, Function, HEntry, Id, Value};
use egglog_bridge::SchemaMath;
use queues::*;
use std::sync::{Arc, Mutex};
use arc_swap::ArcSwap;

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
        //if true {
        //    todo!("extraction")
        //}

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
            Error::ExtractError(value)
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

pub trait PreExtractionWriter {

    fn extract_output_single(&self, termdag: &TermDag, term: &Term, cost: Cost);

    //fn extract_output_multiple(termdag: &TermDag, terms: Vec<Term>, costs: Vec<Cost>);
}

pub trait ExtractionWriter : PreExtractionWriter + Send + Sync {}

#[derive(Clone)]
pub struct EgraphMsgWriter {
    msgs : Arc<Mutex<Option<Vec<String>>>>,
    report : Arc<Mutex<Option<crate::ExtractReport>>>,
}

impl EgraphMsgWriter {
    pub fn new(msgs: Arc<Mutex<Option<Vec<String>>>>,
               report: Arc<Mutex<Option<crate::ExtractReport>>>) -> Self {
        EgraphMsgWriter { msgs: msgs, report: report }
    }

    fn print_msg(&self, msg: String) {
        match &mut *self.msgs.lock().unwrap() {
            Some (msgs) => {msgs.push(msg);}
            _ => {}
        }
    }
}

impl PreExtractionWriter for EgraphMsgWriter {
    fn extract_output_single(&self, termdag: &TermDag, term: &Term, cost: Cost) {
        let extracted = termdag.to_string(&term);
        log::info!("New extractor extracted with cost {cost}: {extracted}");
        self.print_msg(extracted);
        let _ = self.report.lock().unwrap().insert(crate::ExtractReport::Best { termdag: (termdag.clone()), cost: cost, term: (term.clone()) });
    }
}

impl ExtractionWriter for EgraphMsgWriter {}

#[derive(Clone)]
pub struct SaveTermDag {
    pub buffer : Arc<Mutex<Vec<(TermDag, Term, Cost)>>>,
}

impl PreExtractionWriter for SaveTermDag {
    fn extract_output_single(&self, termdag: &TermDag, term: &Term, cost: Cost) {
        self.buffer.lock().unwrap().push((termdag.clone(), term.clone(), cost));
    }
}

impl ExtractionWriter for SaveTermDag {}

pub trait PreCostModel {
    fn fold (
        &self,
        head : &Symbol,
        children_cost : &Vec<Cost>,
        head_cost : Cost,
    ) -> Cost;

    fn container_primitive (
        &self,
        exec_state: &ExecutionState,
        sort : &ArcSort,
        value : &core_relations::Value,
        element_costs : &Vec<Cost>,
    ) -> Cost;

    fn leaf_primitive (
        &self,
        exec_state: &ExecutionState,
        sort : &ArcSort,
        value : &core_relations::Value,
    ) -> Cost;
}

pub trait CostModel : PreCostModel + Send + Sync {}

#[derive(Default, Clone)]
pub struct TreeAdditiveCostModel {}

impl PreCostModel for TreeAdditiveCostModel {
    fn fold(
        &self,
        _head : &Symbol,
        children_cost : &Vec<Cost>,
        head_cost : Cost,
    ) -> Cost {
        children_cost.iter().fold(head_cost, |s, c| { s.saturating_add(*c) })
    }

    fn container_primitive (
        &self,
        exec_state: &ExecutionState,
        sort : &ArcSort,
        value : &core_relations::Value,
        element_costs : &Vec<Cost>,
    ) -> Cost {
        sort.default_container_cost(exec_state, value, element_costs)
    }

    fn leaf_primitive (
        &self,
        exec_state: &ExecutionState,
        sort : &ArcSort,
        value : &core_relations::Value,
    ) -> Cost {
        sort.default_leaf_cost(exec_state, value)
    }
}

impl CostModel for TreeAdditiveCostModel {}

/// Captures what the extractor need to know about a table/function
/// Should only be used for extractable functions
#[derive(Clone, Debug)]
pub struct ExtractorViewFunc {
    pub schema : ResolvedSchema,
    pub cost : Option<Cost>,
    pub backend_table_id : TableId,
    pub can_subsume : bool,
    pub schema_math : SchemaMath,
}

impl ExtractorViewFunc {
    fn new (f : &Function, backend: &egglog_bridge::EGraph) -> Self {
        debug_assert!(f.is_extractable(), "Unextractable function in ExtractorView");
        let (table_id, math) = backend.get_func_info(f.new_backend_id);
        ExtractorViewFunc {
            schema: f.schema.clone(),
            cost: f.decl.cost,
            backend_table_id: table_id,
            can_subsume: f.can_subsume,
            schema_math: math,
        }
    } 
}

// A struct for copying meta data required for extraction
#[derive(Default, Debug)]
pub struct ExtractorView {
    pub funcs : IndexMap<Symbol, ExtractorViewFunc>,
}

impl ExtractorView {
    pub fn new (function: &IndexMap<Symbol, Function>, backend: &egglog_bridge::EGraph) -> Self {
        let mut funcs : IndexMap<Symbol, ExtractorViewFunc> = Default::default();
        for func in function.iter() {
            if func.1.is_extractable() {
                funcs.insert(func.0.clone(), ExtractorViewFunc::new(func.1, backend));
            }
        }
        ExtractorView {
            funcs
        }
    }
}

#[derive(Clone)]
pub struct ExtractorAlter {
    rootsort : ArcSort,
    funcs : Arc<ArcSwap<ExtractorView>>,
    cost_model : Arc<dyn CostModel>,
    writer : Arc<dyn ExtractionWriter>,
}

impl ExtractorAlter {
    pub fn new(
        rootsort : ArcSort,
        funcs : Arc<ArcSwap<ExtractorView>>,
        cost_model : impl CostModel + 'static,
        writer : impl ExtractionWriter + 'static,
    ) -> Self {
        ExtractorAlter {  
            rootsort,
            funcs,
            cost_model: Arc::new(cost_model),
            writer : Arc::new(writer),
        }
    }
}


impl ExtractorAlter {

    /// Built a reverse index from output sort to function head symbols
    fn compute_sort_index(&self) -> IndexMap<Symbol, Vec<Symbol>> {
        let mut rev_index : IndexMap<Symbol, Vec<Symbol>> = Default::default();
        for func in self.funcs.load().funcs.iter() {
            let func_name = *func.0;
            let output_sort = func.1.schema.output.name();
            if let Some (v) = rev_index.get_mut(&output_sort) {
                v.push(func_name);
            } else {
                rev_index.insert(output_sort, vec![func_name]);
            }
        }
        return rev_index;
    }

    /// BFS from the rootsort to filter the tables relevant to this root
    fn compute_filtered_func(&self) -> ExtractorView {
        let sort_index = self.compute_sort_index();
        let mut q = queue![self.rootsort.clone()];
        let mut filtered = ExtractorView::default();
        let mut seen: HashSet<Symbol> = Default::default();
        seen.insert(self.rootsort.name());
        while q.size() > 0 {
            let sort = q.remove().expect("Impossible");
            if sort.is_container_sort() {
                let inner_sorts = sort.inner_sorts();
                for s in inner_sorts {
                    if !seen.contains(&s.name()) {
                        let _ = q.add(s.clone());
                        seen.insert(s.name().clone());
                    }
                }
            } else if sort.is_eq_sort() {
                if let Some (head_symbols) = sort_index.get(&sort.name()) {
                    for h in head_symbols {
                        if !filtered.funcs.contains_key(h) {
                            let func = self.funcs.load().funcs.get(h).expect("Impossible").clone();
                            for ch in &func.schema.input {
                                let ch_name = ch.name();
                                if !seen.contains(&ch_name) {
                                    let _ = q.add(ch.clone());
                                    seen.insert(ch_name);
                                }
                            }
                            filtered.funcs.insert(*h, func);
                        }
                    }
                }
            }
        }
        return filtered;
    }

    /// Compute the cost of a single enode
    /// Recurse if container
    /// Returns None if contains an undefined eqsort term (potentially after unfolding)
    fn compute_cost_node(
        &self,
        exec_state : &ExecutionState,
        value : &core_relations::Value,
        sort : &ArcSort,
        costs : &HashMap<Symbol, IndexMap<core_relations::Value, Cost>>
    ) -> Option<Cost> {
        if sort.is_container_sort() {
            let elements = sort.inner_values(exec_state.containers(), value);
            let mut ch_costs: Vec<Cost> = Vec::new();
            for ch in elements.iter() {
                if let Some (c) = self.compute_cost_node(exec_state,&ch.1, &ch.0, costs) {
                    ch_costs.push(c);
                } else {
                    return None
                }
            }
            Some (self.cost_model.container_primitive(exec_state, sort, value, &ch_costs))
        } else if sort.is_eq_sort() {
            if costs.get(&sort.name()).is_some_and(|t| {t.get(value).is_some()}) {
                Some(costs.get(&sort.name()).expect("Impossible").get(value).expect("Impossible").clone())
            } else {
                None
            }
        } else { // Primitive
            Some(self.cost_model.leaf_primitive(exec_state, sort, value))
        }
    }

    fn compute_cost_hyperedge(
        &self,
        exec_state : &ExecutionState,
        values : &[core_relations::Value],
        head : &Symbol,
        func : &ExtractorViewFunc,
        costs : &HashMap<Symbol, IndexMap<core_relations::Value, Cost>>
    ) -> Option<Cost> {
        let mut ch_costs: Vec<Cost> = Vec::new();
        let sorts = &func.schema.input;
        //log::debug!("compute_cost_hyperedge head {} sorts {:?}", head, sorts);
        debug_assert_eq!(sorts.len(), values.len());
        for (value, sort) in values.iter().zip(sorts.iter()) {
            if let Some (c) = self.compute_cost_node(exec_state, value, sort, costs) {
                ch_costs.push(c);
            } else {
                return None;
            }
        }
        Some (self.cost_model.fold(head, &ch_costs, func.cost.unwrap_or(1)))
    }

    fn reconstruct_termdag_node(
        &self,
        exec_state : &ExecutionState,
        termdag : &mut TermDag,
        value : &core_relations::Value,
        sort : &ArcSort,
        filtered_func : &ExtractorView,
        parent_edge : &HashMap<Symbol, HashMap<core_relations::Value, (Symbol, Vec<core_relations::Value>)>>
    ) -> Term {
        if sort.is_container_sort() {
            let elements = sort.inner_values(exec_state.containers(), value);
            let mut ch_terms: Vec<Term> = Vec::new();
            for ch in elements.iter() {
                ch_terms.push(self.reconstruct_termdag_node(exec_state, termdag, &ch.1, &ch.0, filtered_func, parent_edge));
            }
            sort.reconstruct_termdag_container(exec_state, value, termdag, ch_terms)
        } else if sort.is_eq_sort() {
            let (func_name, hyperedge) = parent_edge.get(&sort.name()).expect("Impossible")
                                                    .get(value).expect("Impossible");
            let mut ch_terms: Vec<Term> = Vec::new();
            let ch_sorts = &filtered_func.funcs.get(func_name).expect("Impossible").schema.input;
            for (value, sort) in hyperedge.iter().zip(ch_sorts.iter()) {
                ch_terms.push(self.reconstruct_termdag_node(exec_state, termdag, value, sort, filtered_func, parent_edge));
            }
            termdag.app(*func_name, ch_terms)
        } else { // Primitive
            sort.reconstruct_termdag_leaf(exec_state, value, termdag)
        }

    }

}

use core_relations::{ExecutionState, ExternalFunction, TableId};

impl ExternalFunction for ExtractorAlter {

    fn invoke(&self, exec_state: &mut ExecutionState, args: &[core_relations::Value]) -> Option<core_relations::Value> {
        debug_assert!(args.len() == 2);
        let target = args[0];
        let nvariants = exec_state.prims().unwrap::<i64>(args[1]);
        log::debug!("target = {:?}, rootsort = {:?}, nvariants = {}", target, self.rootsort, nvariants);
        log::debug!("func: {:?}", self.funcs);
        
        let filtered_func = self.compute_filtered_func();
        log::debug!("filtered_func = {:?}", filtered_func);
        
        // We use Bellman-Ford to compute the costs of the relevant eq sorts' terms
        let mut costs : HashMap<Symbol, HashMap<core_relations::Value, Cost>> = Default::default();
        let mut parent_edge : HashMap<Symbol, HashMap<core_relations::Value, (Symbol, Vec<core_relations::Value>)>> = Default::default();
        for func in filtered_func.funcs.iter() {
            if !costs.contains_key(&func.1.schema.output.name()) {
                debug_assert!(func.1.schema.output.is_eq_sort());
                costs.insert(func.1.schema.output.name(), Default::default());
                parent_edge.insert(func.1.schema.output.name(), Default::default());
            }
        }

        let mut ensure_fixpoint = false;
        let mut reconstruction_round = false;

        // Runs an extra round to copy the best hyperedges
        while !ensure_fixpoint || reconstruction_round {
            ensure_fixpoint = true;

            for func in filtered_func.funcs.iter() {
                let table = exec_state.get_table(func.1.backend_table_id);
                let all = table.all();
                // TODO: Make sure Offset::new_const works as expected
                let mut cur = core_relations::Offset::new_const(0);
                let mut buf = core_relations::TaggedRowBuffer::new(table.spec().arity());
                let func_cols = func.1.schema.input.len();
                let mut done = false;
                while !done {
                    if let Some(next) = table.scan_bounded(all.as_ref(), cur, 500, &mut buf) {
                        cur = next;
                    } else {
                        done = true;
                    }
                    for (_, row) in buf.non_stale() {
                        if func.1.can_subsume && row[func.1.schema_math.subsume_col()] == egglog_bridge::SUBSUMED {
                            continue;
                        }
                        // Output sort and value
                        let target_sort = &func.1.schema.output;
                        let target = row[func.1.schema_math.ret_val_col()];
                        if let Some (new_cost) = self.compute_cost_hyperedge(exec_state, &row[0..func.1.schema_math.num_keys()], func.0, func.1, &costs) {
                            if !reconstruction_round {
                                log::debug!("Got cost: {:?}", (target, new_cost));
                                match costs.get_mut(&target_sort.name()).expect("Impossible").entry(target) {
                                    HEntry::Vacant(e) => {
                                        log::debug!{"Inserted new cost"};
                                        ensure_fixpoint = false;
                                        e.insert(new_cost);
                                    }
                                    HEntry::Occupied(mut e) => {
                                        if new_cost < *(e.get()) {
                                            log::debug!{"Updated new cost"};
                                            ensure_fixpoint = false;
                                            e.insert(new_cost);
                                        }
                                    }
                                }
                            } else {
                                if new_cost == *costs.get(&target_sort.name()).expect("Impossible")
                                                     .get(&target).expect("Impossible") {
                                    // one of the possible best parent edges
                                    match parent_edge.get_mut(&target_sort.name()).expect("Impossible").entry(target) {
                                        HEntry::Vacant(e) => {
                                            e.insert((*func.0, row[0..func_cols].to_vec()));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    buf.clear();
                }
            }

            if ensure_fixpoint {
                reconstruction_round = !reconstruction_round;
            }
        }
        let best_cost = self.compute_cost_node(exec_state, &target, &self.rootsort, &costs).expect("Failed to extract root!");
        log::debug!("Best cost for the extract root: {:?}", best_cost);
        log::debug!("Dumping costs: {:?}", costs);
        log::debug!{"Dumping parent_edge: {:?}", parent_edge};

        let mut termdag : TermDag = Default::default(); 
        let term = self.reconstruct_termdag_node(exec_state, &mut termdag, &target, &self.rootsort, &filtered_func, &parent_edge);

        log::debug!("Got termdag: {:?}, term: {:?}", termdag, term);

        self.writer.extract_output_single(&termdag, &term, best_cost);
        Some(exec_state.prims().get::<()>(()))
    }
}