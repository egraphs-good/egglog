use hashbrown::hash_map::Entry;

use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::HashMap;
use crate::{ArcSort, EGraph, Function, Id, Value};

type Cost = usize;

#[derive(Debug)]
struct Node<'a> {
    sym: Symbol,
    inputs: &'a [Value],
}

pub(crate) struct Extractor<'a> {
    costs: HashMap<Id, (Cost, Term)>,
    ctors: Vec<Symbol>,
    egraph: &'a EGraph,
    use_eq_relation: bool,
}

impl EGraph {
    pub fn value_to_id(&self, value: Value) -> Option<(Symbol, Id)> {
        if let Some(sort) = self.get_sort(&value) {
            if sort.is_eq_sort() {
                return Some((sort.name(), Id::from(self.find(value).bits as usize)));
            }
        }
        None
    }

    pub fn extract(&self, value: Value, termdag: &mut TermDag, arcsort: &ArcSort) -> (Cost, Term) {
        Extractor::new(self, termdag, true).find_best(value, termdag, arcsort)
    }

    pub fn extract_variants(
        &mut self,
        value: Value,
        limit: usize,
        termdag: &mut TermDag,
    ) -> Vec<Term> {
        let (tag, id) = self.value_to_id(value).unwrap();
        let output_value = &Value::from_id(tag, id);
        let ext = &Extractor::new(self, termdag, true);
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
                            ext.expr_from_node(&node, termdag)
                        })
                    })
                    .collect()
            })
            .take(limit)
            .collect()
    }
}

impl<'a> Extractor<'a> {
    pub fn new(egraph: &'a EGraph, termdag: &mut TermDag, use_eq_relation: bool) -> Self {
        let mut extractor = Extractor {
            costs: HashMap::default(),
            egraph,
            ctors: vec![],
            use_eq_relation,
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

    fn expr_from_node(&self, node: &Node, termdag: &mut TermDag) -> Term {
        let mut children = vec![];
        for value in node.inputs {
            let arcsort = self.egraph.get_sort(value).unwrap();
            children.push(self.find_best(*value, termdag, arcsort).1)
        }
        termdag.make(node.sym, children)
    }

    fn find(&self, value: Value) -> Id {
        if self.use_eq_relation {
            Id::from(self.egraph.find(value).bits as usize)
        } else {
            Id::from(value.bits as usize)
        }
    }

    pub fn find_best(&self, value: Value, termdag: &mut TermDag, sort: &ArcSort) -> (Cost, Term) {
        if sort.is_eq_sort() {
            let id = self.find(value);
            let (cost, node) = self
                .costs
                .get(&id)
                .unwrap_or_else(|| {
                    eprintln!("No cost for {:?}", value);
                    for func in self.egraph.functions.values() {
                        for (inputs, output) in func.nodes.iter() {
                            if output.value == value {
                                eprintln!("Found unextractable function: {:?}", func.decl.name);
                                eprintln!("Inputs: {:?}", inputs);
                                eprintln!(
                                    "{:?}",
                                    inputs
                                        .iter()
                                        .map(|input| self.costs.get(&self.find(*input)))
                                        .collect::<Vec<_>>()
                                );
                            }
                        }
                    }

                    panic!("No cost for {:?}", value)
                })
                .clone();
            (cost, node)
        } else {
            (0, termdag.expr_to_term(&sort.make_expr(self.egraph, value)))
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
            cost = cost.saturating_add(if ty.is_eq_sort() {
                let id = self.find(*value);
                // TODO costs should probably map values?
                let (cost, term) = self.costs.get(&id)?;
                terms.push(term.clone());
                *cost
            } else {
                let term = termdag.expr_to_term(&ty.make_expr(self.egraph, *value));
                terms.push(term);
                1
            });
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
                    for (inputs, output) in func.nodes.iter() {
                        if let Some((term_inputs, new_cost)) =
                            self.node_total_cost(func, inputs, termdag)
                        {
                            let make_new_pair = || (new_cost, termdag.make(sym, term_inputs));

                            let id = self.find(output.value);
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
