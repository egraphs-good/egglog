use hashbrown::hash_map::Entry;

use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::HashMap;
use crate::{EGraph, Function, Id, Value};

type Cost = usize;

pub(crate) struct Extractor<'a> {
    costs: HashMap<Id, (Cost, Term)>,
    ctors: Vec<Symbol>,
    egraph: &'a EGraph,
    pub(crate) termdag: &'a mut TermDag,
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

    pub fn extract(&mut self, value: Value, termdag: &mut TermDag) -> (Cost, Term) {
        Extractor::new(self, termdag).find_best(value)
    }

    pub fn extract_variants(
        &mut self,
        value: Value,
        limit: usize,
        termdag: &mut TermDag,
    ) -> Vec<Term> {
        let (tag, id) = self.value_to_id(value).unwrap();
        let output_value = &Value::from_id(tag, id);
        let mut ext = Extractor::new(self, termdag);

        let mut result = vec![];
        for sym in ext.ctors.clone() {
            let func = &self.functions[&sym];
            if !func.schema.output.is_eq_sort() {
                return vec![];
            }
            assert!(func.schema.output.is_eq_sort());

            for (inputs, output) in func.nodes.iter() {
                if result.len() >= limit {
                    return result;
                }
                if &output.value == output_value {
                    let mut children = vec![];
                    for input in inputs {
                        let node = ext.find_best(*input).1;
                        children.push(ext.termdag.lookup(&node));
                    }
                    result.push(Term::App(sym, children))
                }
            }
        }
        result
    }
}

impl<'a> Extractor<'a> {
    pub fn new(egraph: &'a EGraph, termdag: &'a mut TermDag) -> Self {
        let mut extractor = Extractor {
            costs: HashMap::default(),
            egraph,
            ctors: vec![],
            termdag,
        };

        // HACK
        // just consider all functions constructors for now...
        extractor.ctors.extend(
            egraph
                .functions
                .keys()
                .filter(|func| !egraph.functions.get(*func).unwrap().decl.unextractable)
                .cloned(),
        );

        log::debug!("Extracting from ctors: {:?}", extractor.ctors);
        extractor.find_costs();
        extractor
    }

    pub fn find_best(&mut self, value: Value) -> (Cost, Term) {
        let sort = self.egraph.get_sort(&value).unwrap();
        if sort.is_eq_sort() {
            let id = self.egraph.find(Id::from(value.bits as usize));
            let (cost, node) = self
                .costs
                .get(&id)
                .unwrap_or_else(|| panic!("No cost for {:?}", value))
                .clone();
            (cost, node)
        } else {
            (0, self.termdag.expr_to_term(&sort.make_expr(value)))
        }
    }

    fn node_total_cost(
        &mut self,
        function: &Function,
        children: &[Value],
    ) -> Option<(Vec<Term>, Cost)> {
        let mut cost = function.decl.cost.unwrap_or(1);
        let types = &function.schema.input;
        let mut terms: Vec<Term> = vec![];
        for (ty, value) in types.iter().zip(children) {
            cost = cost.saturating_add(if ty.is_eq_sort() {
                let id = self.egraph.find(Id::from(value.bits as usize));
                // TODO costs should probably map values?
                let (cost, term) = self.costs.get(&id)?;
                terms.push(term.clone());
                *cost
            } else {
                let term = self.termdag.expr_to_term(&ty.make_expr(*value));
                terms.push(term);
                1
            });
        }
        Some((terms, cost))
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for sym in self.ctors.clone() {
                println!("{}", sym);
                let func = &self.egraph.functions[&sym];
                if func.schema.output.is_eq_sort() {
                    for (inputs, output) in func.nodes.iter() {
                        if let Some((term_inputs, new_cost)) = self.node_total_cost(func, inputs) {
                            let make_new_pair = || (new_cost, self.termdag.make(sym, term_inputs));

                            let id = self.egraph.find(Id::from(output.value.bits as usize));
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
