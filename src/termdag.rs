use symbolic_expressions::Sexp;

use crate::{
    ast::{Expr, Literal},
    util::{HashMap, HashSet},
    Symbol,
};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    Lit(Literal),
    Var(Symbol),
    App(Symbol, Vec<usize>),
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TermDag {
    nodes: Vec<Term>,
    hashcons: HashMap<Term, usize>,
}

#[macro_export]
macro_rules! match_term_app {
    ($e:expr; { $(
        ($head:expr, $args:pat) => $body:expr $(,)?
    ),*}) => {
        match $e {
            Term::App(head, args) => {
                $(
                    if head.as_str() == $head {
                        match args.as_slice() {
                            $args => $body,
                            _ => panic!("arg mismatch"),
                        }
                    } else
                )* {
                    panic!("Failed to match any of the heads of the patterns. Got: {}", head);
                }
            }
            _ => panic!("not an app")
        }
    }
}

impl TermDag {
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    // users can't construct a termnode, so just
    // look it up
    pub fn lookup(&self, node: &Term) -> usize {
        *self.hashcons.get(node).unwrap()
    }

    pub fn get(&self, idx: usize) -> Term {
        self.nodes[idx].clone()
    }

    pub fn make(&mut self, sym: Symbol, children: Vec<Term>) -> Term {
        let node = Term::App(sym, children.iter().map(|c| self.lookup(c)).collect());

        self.add_node(&node);

        node
    }

    fn add_node(&mut self, node: &Term) {
        if self.hashcons.get(node).is_none() {
            let idx = self.nodes.len();
            self.nodes.push(node.clone());
            self.hashcons.insert(node.clone(), idx);
        }
    }

    pub fn expr_to_term(&mut self, expr: &Expr) -> Term {
        let res = match expr {
            Expr::Lit(lit) => Term::Lit(lit.clone()),
            Expr::Var(v) => Term::Var(*v),
            Expr::Call(op, args) => {
                let args = args
                    .iter()
                    .map(|a| {
                        let term = self.expr_to_term(a);
                        self.lookup(&term)
                    })
                    .collect();
                Term::App(*op, args)
            }
        };
        self.add_node(&res);
        res
    }

    pub fn to_string(&self, term: &Term) -> String {
        let mut stored = HashMap::<usize, String>::default();
        let mut seen = HashSet::<usize>::default();
        let id = self.lookup(term);
        // use a stack to avoid stack overflow
        let mut stack = vec![id];
        while !stack.is_empty() {
            let next = stack.pop().unwrap();

            match self.nodes[next].clone() {
                Term::App(name, children) => {
                    if seen.contains(&next) {
                        let mut str = String::new();
                        str.push_str(&format!("({}", name));
                        for c in children.iter() {
                            str.push_str(&format!(" {}", stored[c]));
                        }
                        str.push_str(")");
                        stored.insert(next, str);
                    } else {
                        seen.insert(next);
                        stack.push(next);
                        for c in children.iter().rev() {
                            stack.push(*c);
                        }
                    }
                }
                Term::Lit(lit) => {
                    stored.insert(next, format!("{}", lit));
                }
                Term::Var(v) => {
                    stored.insert(next, format!("{}", v));
                }
            }
        }

        stored.get(&id).unwrap().clone()
    }
}
