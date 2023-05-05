use symbolic_expressions::Sexp;

use crate::{
    ast::{Expr, Literal},
    util::HashMap,
    Symbol,
};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum TermNode {
    Lit(Literal),
    Var(Symbol),
    App(Symbol, Vec<usize>),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Term(TermNode);

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TermDag {
    nodes: Vec<Term>,
    hashcons: HashMap<Term, usize>,
}

impl TermDag {
    // users can't construct a termnode, so just
    // look it up
    fn lookup(&self, node: &Term) -> usize {
        *self.hashcons.get(node).unwrap()
    }

    pub fn make(&mut self, sym: Symbol, children: Vec<Term>) -> Term {
        let node = Term(TermNode::App(
            sym,
            children.iter().map(|c| self.lookup(c)).collect(),
        ));

        if self.hashcons.get(&node).is_none() {
            let idx = self.nodes.len();
            self.nodes.push(node.clone());
            self.hashcons.insert(node.clone(), idx);
        }

        node
    }

    pub fn from_expr(&mut self, expr: &Expr) -> Term {
        match expr {
            Expr::Lit(lit) => Term(TermNode::Lit(lit.clone())),
            Expr::Var(v) => Term(TermNode::Var(*v)),
            Expr::Call(op, args) => {
                let args = args
                    .iter()
                    .map(|a| {
                        let term = self.from_expr(a);
                        self.lookup(&term)
                    })
                    .collect();
                Term(TermNode::App(*op, args))
            }
        }
    }

    pub fn to_string(&self, term: &Term) -> String {
        let mut stored = HashMap::<usize, String>::default();
        let id = self.lookup(term);
        // use a stack to avoid stack overflow ):
        let mut stack = vec![id];
        while !stack.is_empty() {
            let next = stack.pop().unwrap();

            match self.nodes[next].0.clone() {
                TermNode::App(name, children) => {
                    if children.is_empty() || stored.contains_key(&children[0]) {
                        let mut str = String::new();
                        str.push_str(&format!("({}", name));
                        for c in children.iter() {
                            str.push_str(&format!(" {}", stored[c]));
                        }
                        str.push_str(")");
                        stored.insert(next, str);
                    } else {
                        stack.push(next);
                        for c in children.iter().rev() {
                            stack.push(*c);
                        }
                    }
                }
                TermNode::Lit(lit) => {
                    stored.insert(next, format!("{}", lit));
                }
                TermNode::Var(v) => {
                    stored.insert(next, format!("{}", v));
                }
            }
        }

        stored.get(&id).unwrap().clone()
    }
}
