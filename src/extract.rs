use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::HashMap;
use crate::{ArcSort, EGraph, Error, Function, HEntry, Id, Value};

pub type Cost = usize;
pub(crate) type CostMap = HashMap<Id, (Cost, Term)>;

#[derive(Debug)]
pub(crate) struct Node<'a> {
    sym: Symbol,
    func: &'a Function,
    inputs: &'a [Value],
}

pub struct Extractor<'a> {
    pub costs: CostMap,
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
    pub fn extract(&mut self, value: Value, arcsort: &ArcSort) -> Result<(Cost, Term), Error> {
        let mut cost_map = None;
        let mut termdag = std::mem::take(&mut self.termdag);
        if self.cost_cache.is_some() {
            let cost_map_ts = std::mem::take(&mut self.cost_cache).unwrap();
            if cost_map_ts.0 == self.timestamp {
                cost_map = Some(cost_map_ts.1);
            }
        }
        let extractor = Extractor::new(self, &mut termdag, cost_map);
        let result = extractor
            .find_best(value, &mut termdag, arcsort)
            .ok_or_else(|| {
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
            });
        self.cost_cache = Some((self.timestamp, extractor.cost_map()));
        self.termdag = termdag;
        result
    }

    /// Extracts up to `limit` terms for a given `value`.
    pub fn extract_variants(&mut self, sort: &ArcSort, value: Value, limit: usize) -> Vec<Term> {
        let mut cost_map = None;
        let mut termdag = std::mem::take(&mut self.termdag);
        if self.cost_cache.is_some() {
            let cost_map_ts = std::mem::take(&mut self.cost_cache).unwrap();
            if cost_map_ts.0 == self.timestamp {
                cost_map = Some(cost_map_ts.1);
            }
        }
        let extractor = Extractor::new(self, &mut termdag, cost_map);
        let result = extractor.find_variants(value, &mut termdag, sort, limit);
        self.cost_cache = Some((self.timestamp, extractor.cost_map()));
        self.termdag = termdag;
        result
    }
}

impl<'a> Extractor<'a> {
    pub fn new(egraph: &'a EGraph, termdag: &mut TermDag, cost_map: Option<CostMap>) -> Self {
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
        if let Some(cost_map) = cost_map {
            extractor.costs = cost_map;
        } else {
            extractor.find_costs(termdag);
        }
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

    /// Extracts up to `limit` terms for a given `value`.
    pub fn find_variants(
        &self,
        value: Value,
        termdag: &mut TermDag,
        sort: &ArcSort,
        limit: usize,
    ) -> Vec<Term> {
        let output_sort = sort.name();
        let output_value = self.egraph.find(sort, value);
        let terms = self
            .ctors
            .iter()
            .flat_map(|&sym| {
                let func = &self.egraph.functions[&sym];
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
                        self.expr_from_node(&node, termdag).expect(
                            "extract_variants should be called after extractor initialization",
                        )
                    })
                    .collect()
            })
            .take(limit)
            .collect::<Vec<Term>>();

        // TODO: what happens if `terms` is empty?
        terms
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

    pub fn cost_map(self) -> CostMap {
        self.costs
    }
}
