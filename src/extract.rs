use hashbrown::hash_map::Entry;

use crate::ast::Symbol;
use crate::util::HashMap;
use crate::{EGraph, Expr, Function, Id, Value};

type Cost = usize;

#[derive(Debug)]
struct Node<'a> {
    sym: Symbol,
    inputs: &'a [Value],
}

struct Extractor<'a> {
    costs: HashMap<Id, (Cost, Node<'a>)>,
    ctors: Vec<Symbol>,
    egraph: &'a EGraph,
}

impl EGraph {
    pub fn value_to_id(&self, value: Value) -> Option<(Symbol, Id)> {
        if let Some(sort) = self.sorts.get(&value.tag) {
            if sort.is_eq_sort() {
                let id = Id::from(value.bits as usize);
                return Some((sort.name(), self.find(id)));
            }
        }
        None
    }

    pub fn extract(&mut self, value: Value) -> (Cost, Expr) {
        Extractor::new(self).find_best(value)
    }

    pub fn extract_variants(&mut self, value: Value, limit: usize) -> Vec<Expr> {
        let (tag, id) = self.value_to_id(value).unwrap();
        let output_value = &Value::from_id(tag, id);
        let ext = &Extractor::new(self);
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
                            let node = Node {
                                sym,
                                inputs: inputs.data(),
                            };
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
    fn new(egraph: &'a EGraph) -> Self {
        let mut extractor = Extractor {
            costs: HashMap::default(),
            egraph,
            ctors: vec![],
        };

        // HACK
        // just consider all functions constructors for now...
        extractor.ctors.extend(egraph.functions.keys().cloned());

        log::debug!("Extracting from ctors: {:?}", extractor.ctors);
        extractor.find_costs();
        extractor
    }

    fn expr_from_node(&self, node: &Node) -> Expr {
        let children = node.inputs.iter().map(|&value| self.find_best(value).1);
        Expr::call(node.sym, children)
    }

    fn find_best(&self, value: Value) -> (Cost, Expr) {
        let sort = self.egraph.sorts.get(&value.tag).unwrap();
        if sort.is_eq_sort() {
            let id = self.egraph.find(Id::from(value.bits as usize));
            let (cost, node) = &self
                .costs
                .get(&id)
                .unwrap_or_else(|| panic!("No cost for {:?}", value));
            (*cost, self.expr_from_node(node))
        } else {
            (0, sort.make_expr(value))
        }
    }

    fn node_total_cost(&self, function: &Function, children: &[Value]) -> Option<Cost> {
        let mut cost = function.decl.cost.unwrap_or(1);
        let types = &function.schema.input;
        for (ty, value) in types.iter().zip(children) {
            cost += if ty.is_eq_sort() {
                let id = self.egraph.find(Id::from(value.bits as usize));
                // TODO costs should probably map values?
                self.costs.get(&id)?.0
            } else {
                1
            }
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
                    for (inputs, output) in &func.nodes {
                        let inputs = inputs.data();
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
