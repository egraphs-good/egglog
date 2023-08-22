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

pub struct Extractor<'a> {
    pub costs: HashMap<Id, (Cost, Term)>,
    ctors: Vec<Symbol>,
    egraph: &'a EGraph,
}

impl EGraph {
    pub fn value_to_id(&self, value: Value) -> Option<(Symbol, Id)> {
        if let Some(sort) = self.get_sort(&value) {
            if sort.is_eq_sort() {
                let id = Id::from(value.bits as usize);
                return Some((sort.name(), self.find(id)));
            }
        }
        None
    }

    pub fn extract(&self, value: Value, termdag: &mut TermDag, arcsort: &ArcSort) -> (Cost, Term) {
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
                                    .map(|input| extractor.costs.get(&extractor.find(input)))
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
        let (tag, id) = self.value_to_id(value).unwrap();
        let output_value = &Value::from_id(tag, id);
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
                    .filter_map(|(inputs, output)| {
                        (&output.value == output_value).then(|| {
                            let node = Node { sym, inputs };
                            ext.expr_from_node(&node, termdag).expect(
                                "extract_variants should be called after extractor initialization",
                            )
                        })
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
            let arcsort = self.egraph.get_sort(value).unwrap();
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
            let id = self.find(&value);
            let (cost, node) = self.costs.get(&id)?.clone();
            Some((cost, node))
        } else {
            let (cost, node) = sort.extract_expr(self.egraph, value, self, termdag)?;
            Some((cost, termdag.expr_to_term(&node)))
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

    fn find(&self, value: &Value) -> Id {
        self.egraph.find(Id::from(value.bits as usize))
    }

    fn find_costs(&mut self, termdag: &mut TermDag) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for sym in self.ctors.clone() {
                let func = &self.egraph.functions[&sym];
                if func.schema.output.is_eq_sort() {
                    for (inputs, output) in func.nodes.iter() {
                        if let Some((term_inputs, new_cost)) =
                            self.node_total_cost(func, inputs, termdag)
                        {
                            let make_new_pair = || (new_cost, termdag.app(sym, term_inputs));

                            let id = self.find(&output.value);
                            match self.costs.entry(id) {
                                Entry::Vacant(e) => {
                                    did_something = true;
                                    e.insert(make_new_pair());
                                }
                                Entry::Occupied(mut e) => {
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
