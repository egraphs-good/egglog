use hashbrown::hash_map::Entry;

use crate::ast::{Symbol, Type};
use crate::util::HashMap;
use crate::{EGraph, Expr, Id, Value};

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
    pub fn extract(&mut self, id: Id) -> (Cost, Expr) {
        Extractor::new(self).find_best(id)
    }

    pub fn extract_variants(&mut self, id: Id, limit: usize) -> Vec<Expr> {
        let id = self.find(id);
        let output_value = &Value::from(id);
        let ext = &Extractor::new(self);
        ext.ctors
            .iter()
            .flat_map(|&sym| {
                let func = &self.functions[&sym];
                assert!(func.decl.schema.output.is_sort());
                func.nodes.iter().filter_map(move |(inputs, output)| {
                    (output == output_value).then(|| {
                        let node = Node { sym, inputs };
                        ext.expr_from_node(&node)
                    })
                })
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

        for ctors in egraph.sorts.values() {
            extractor.ctors.extend(ctors.iter().copied())
        }

        log::debug!("Extracting from ctors: {:?}", extractor.ctors);
        extractor.find_costs();
        extractor
    }

    fn expr_from_node(&self, node: &Node) -> Expr {
        let mut children = vec![];
        for (ty, value) in self.egraph.functions[&node.sym]
            .decl
            .schema
            .input
            .iter()
            .zip(node.inputs)
        {
            if ty.is_sort() {
                children.push(self.find_best(Id::from(value.clone())).1)
            } else {
                children.push(Expr::Lit(value.to_literal()))
            }
        }
        Expr::call(node.sym, children)
    }

    fn find_best(&self, id: Id) -> (Cost, Expr) {
        let id = self.egraph.find(id);
        let (cost, node) = &self.costs[&id];
        (*cost, self.expr_from_node(node))
    }

    fn node_total_cost(&self, types: &[Type], children: &[Value]) -> Option<Cost> {
        let mut cost = 1;
        for (ty, value) in types.iter().zip(children) {
            cost += if ty.is_sort() {
                self.costs.get(&Id::from(value.clone()))?.0
            } else {
                0
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
                assert!(func.decl.schema.output.is_sort());
                for (inputs, output) in &func.nodes {
                    if let Some(new_cost) = self.node_total_cost(&func.decl.schema.input, inputs) {
                        let make_new_pair = || (new_cost, Node { sym, inputs });
                        match self.costs.entry(Id::from(output.clone())) {
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
