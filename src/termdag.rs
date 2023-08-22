use crate::{
    ast::{Expr, Literal},
    util::{HashMap, HashSet},
    Symbol,
};

/// Like [`Expr`]s but with sharing and deduplication.
///
/// Terms refer to their children indirectly as indexes into an ambient [`TermDag`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    Lit(Literal),
    Var(Symbol),
    /// usize is the index of the child in the nodes vec
    App(Symbol, Vec<usize>),
}

/// A hashconsing arena for [`Term`]s.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TermDag {
    // think of nodes as a map from indices to Terms.
    // invariant: the nodes map and the hashcons map are inverses.
    // note that this implies:
    // - no duplicates in nodes
    // - every element of node is a key in hashcons
    // - every key of hashcons is in nodes
    pub nodes: Vec<Term>,
    /// maps a term to its index in the nodes vec, for fast lookup
    pub hashcons: HashMap<Term, usize>,
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
    /// Returns the number of nodes in this DAG.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Convert the given term to its index.
    ///
    /// Panics if the term does not already exist in this [TermDag].
    pub fn lookup(&self, node: &Term) -> usize {
        *self.hashcons.get(node).unwrap()
    }

    /// Convert the given index to the corresponding term.
    ///
    /// Panics if the index is not valid.
    pub fn get(&self, idx: usize) -> Term {
        self.nodes[idx].clone()
    }

    /// Make and return a [`Term::App`] with the given head symbol and children,
    /// and insert into the DAG if it is not already present.
    ///
    /// Panics if any of the children are not already in the DAG.
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

    /// Recursively converts the given expression to a term.
    ///
    /// This involves inserting every subexpression into this DAG. Because
    /// TermDags are hashconsed, the resulting term is guaranteed to maximally
    /// share subterms.
    pub fn expr_to_term(&self, expr: &Expr) -> Term {
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

    /// Recursively converts the given term to an expression.
    ///
    /// Panics if the term contains subterms that are not in the DAG.
    pub fn term_to_expr(&self, term: &Term) -> Expr {
        match term {
            Term::Lit(lit) => Expr::Lit(lit.clone()),
            Term::Var(v) => Expr::Var(*v),
            Term::App(op, args) => {
                let args = args
                    .iter()
                    .map(|a| {
                        let term = self.get(*a);
                        self.term_to_expr(&term)
                    })
                    .collect();
                Expr::Call(*op, args)
            }
        }
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: &Term) -> String {
        let mut stored = HashMap::<usize, String>::default();
        let mut seen = HashSet::<usize>::default();
        let id = self.lookup(term);
        // use a stack to avoid stack overflow
        let mut stack = vec![id];
        while let Some(next) = stack.pop() {
            match self.nodes[next].clone() {
                Term::App(name, children) => {
                    if seen.contains(&next) {
                        let mut str = String::new();
                        str.push_str(&format!("({}", name));
                        for c in children.iter() {
                            str.push_str(&format!(" {}", stored[c]));
                        }
                        str.push(')');
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
