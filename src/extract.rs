use hashbrown::hash_map::Entry;

use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::HashMap;
use crate::{ArcSort, EGraph, Function, Id, Value};

pub type Cost = usize;

#[derive(Debug)]
pub(crate) struct Node<'a> {
    sym: Symbol,
    inputs: &'a [Value],
}

#[derive(Debug, Clone)]
pub struct CostSet {
    pub total: Cost,
    // TODO this would be more efficient as a
    // persistent data structure
    pub costs: HashMap<Value, Cost>,
    pub term: Term,
}

pub struct Extractor<'a> {
    // TODO why is this public?
    pub costs: HashMap<Id, CostSet>,
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
    ///         "(datatype Op (Add i64 i64))
    ///          (let expr (Add 1 1))",
    ///     )
    ///     .unwrap();
    /// let mut termdag = TermDag::default();
    /// let (sort, value) = egraph
    ///     .eval_expr(&egglog::ast::Expr::Var("expr".into()), None, true)
    ///     .unwrap();
    /// let costset = egraph.extract(value, &mut termdag, &sort);
    /// assert_eq!(termdag.to_string(&costset.term), "(Add 1 1)");
    /// ```
    pub fn extract(&self, value: Value, termdag: &mut TermDag, arcsort: &ArcSort) -> CostSet {
        let extractor = Extractor::new(self, termdag);
        extractor
            .find_best(value, termdag, arcsort)
            .unwrap_or_else(|| {
                log::error!("No cost for {:?}", value);
                for func in self.functions.values() {
                    for (inputs, output) in func.nodes.iter() {
                        if output.value == value {
                            log::error!("Found unextractable function: {:?}", func.decl.name);
                            log::error!("Inputs: {:?}", inputs);
                            log::error!(
                                "{:?}",
                                inputs
                                    .iter()
                                    .map(|input| extractor.costs.get(&extractor.find_id(*input)))
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }

                panic!("No cost for {:?}", value)
            })
    }

    pub fn extract_variants(
        &mut self,
        value: Value,
        limit: usize,
        termdag: &mut TermDag,
    ) -> Vec<Term> {
        let output_value = self.find(value);
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
                    .iter()
                    .filter(|&(_inputs, output)| (output.value == output_value))
                    .map(|(inputs, _output)| {
                        let node = Node { sym, inputs };
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
        for value in node.inputs {
            let arcsort = self.egraph.get_sort_from_value(value).unwrap();
            children.push(self.find_best(*value, termdag, arcsort)?.term)
        }

        Some(termdag.app(node.sym, children))
    }

    pub fn find_best(
        &self,
        value: Value,
        termdag: &mut TermDag,
        sort: &ArcSort,
    ) -> Option<CostSet> {
        if sort.is_eq_sort() {
            let id = self.find_id(value);
            Some(self.costs.get(&id)?.clone())
        } else {
            sort.extract_expr(self.egraph, value, self, termdag)
        }
    }

    fn node_total_cost(
        &mut self,
        function: &Function,
        children: &[Value],
        value: Value,
        termdag: &mut TermDag,
    ) -> Option<CostSet> {
        let types = &function.schema.input;

        let mut costs = HashMap::default();
        let mut new_children: Vec<Term> = vec![];
        for (ty, value) in types.iter().zip(children) {
            let child_res = self.find_best(*value, termdag, ty)?;
            costs.extend(child_res.costs.clone());
            new_children.push(child_res.term.clone());
        }
        let cost = function.decl.cost.unwrap_or(1);
        costs.insert(value, cost);

        Some(CostSet {
            total: costs.values().sum::<Cost>(),
            costs,
            term: termdag.app(function.decl.name, new_children),
        })
    }

    fn find(&self, value: Value) -> Value {
        self.egraph.find(value)
    }

    fn find_id(&self, value: Value) -> Id {
        Id::from(self.find(value).bits as usize)
    }

    fn find_costs(&mut self, termdag: &mut TermDag) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for sym in self.ctors.clone() {
                let func = &self.egraph.functions[&sym];
                if func.schema.output.is_eq_sort() {
                    for (inputs, output) in func.nodes.iter() {
                        if let Some(costset) =
                            self.node_total_cost(func, inputs, output.value, termdag)
                        {
                            let id = self.find_id(output.value);
                            match self.costs.entry(id) {
                                Entry::Vacant(e) => {
                                    did_something = true;
                                    e.insert(costset);
                                }
                                Entry::Occupied(mut e) => {
                                    if costset.total < e.get().total {
                                        did_something = true;
                                        e.insert(costset);
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
