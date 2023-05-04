use hashbrown::hash_map::Entry;

use crate::ast::Symbol;
use crate::termdag::{Term, TermDag};
use crate::util::HashMap;
use crate::{EGraph, Expr, Function, Id, Value};

type Cost = usize;

#[derive(Debug, Clone)]
pub(crate) struct Node<'a> {
    pub(crate) sym: Symbol,
    pub(crate) inputs: &'a [Value],
}

struct Extractor<'a> {
    costs: HashMap<Id, (Cost, Node<'a>)>,
    ctors: Vec<Symbol>,
    egraph: &'a EGraph,
    termdag: &'a mut TermDag,
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
                    .filter_map(move |(inputs, output)| {
                        (&output.value == output_value).then(|| {
                            let node = Node { sym, inputs };
                            ext.expr_from_node(&node)
                        })
                    })
                    .collect()
            })
            .take(limit)
            .collect()
    }
}

impl<'a> Extractor<'a> {
    fn new(egraph: &'a EGraph, termdag: &'a mut TermDag) -> Self {
        let mut extractor = Extractor {
            costs: HashMap::default(),
            egraph,
            ctors: vec![],
            termdag,
        };

        // HACK
        // just consider all functions constructors for now...
        extractor.ctors.extend(egraph.functions.keys().cloned());

        log::debug!("Extracting from ctors: {:?}", extractor.ctors);
        extractor.find_costs();
        extractor
    }

    fn expr_from_node(&mut self, node: &Node) -> Term {
        let children = node.inputs.iter().map(|&value| self.find_best(value).1);
        self.termdag.make(node.sym, children.collect())
    }

    fn find_best(&mut self, value: Value) -> (Cost, Term) {
        let sort = self.egraph.get_sort(&value).unwrap();
        if sort.is_eq_sort() {
            let id = self.egraph.find(Id::from(value.bits as usize));
            let (cost, node) = &self
                .costs
                .get(&id)
                .unwrap_or_else(|| panic!("No cost for {:?}", value));
            (*cost, self.expr_from_node(node))
        } else {
            (0, self.termdag.from_expr(&sort.make_expr(value)))
        }
    }

    fn node_total_cost(&self, function: &Function, children: &[Value]) -> Option<Cost> {
        let mut cost = function.decl.cost.unwrap_or(1);
        let types = &function.schema.input;
        for (ty, value) in types.iter().zip(children) {
            cost = cost.saturating_add(if ty.is_eq_sort() {
                let id = self.egraph.find(Id::from(value.bits as usize));
                // TODO costs should probably map values?
                self.costs.get(&id)?.0
            } else {
                1
            });
        }
        Some(cost)
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for &sym in &self.ctors {
                let func = &self.egraph.functions[&sym];
                if func.schema.output.is_eq_sort() {
                    for (inputs, output) in func.nodes.iter() {
                        if let Some(new_cost) = self.node_total_cost(func, inputs) {
                            let make_new_pair = || (new_cost, Node { sym, inputs });

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
