use crate::{
    ast::Literal,
    util::{HashMap, HashSet, IndexSet},
    Expr, GenericExpr, Symbol,
};

pub type TermId = usize;

#[allow(rustdoc::private_intra_doc_links)]
/// Like [`Expr`]s but with sharing and deduplication.
///
/// Terms refer to their children indirectly via opaque [TermId]s (internally
/// these are just `usize`s) that map into an ambient [`TermDag`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    Lit(Literal),
    Var(Symbol),
    App(Symbol, Vec<TermId>),
}

/// A hashconsing arena for [`Term`]s.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TermDag {
    /// A bidirectional map between deduplicated `Term`s and indices.
    nodes: IndexSet<Term>,
}

#[macro_export]
macro_rules! match_term_app {
    ($e:expr; $body:tt) => {
        match $e {
            Term::App(head, args) => {
                match (head.as_str(), args.as_slice())
                    $body
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

    /// Convert the given term to its id.
    ///
    /// Panics if the term does not already exist in this [TermDag].
    pub fn lookup(&self, node: &Term) -> TermId {
        self.nodes.get_index_of(node).unwrap()
    }

    /// Convert the given id to the corresponding term.
    ///
    /// Panics if the id is not valid.
    pub fn get(&self, id: TermId) -> &Term {
        self.nodes.get_index(id).unwrap()
    }

    /// Make and return a [`Term::App`] with the given head symbol and children,
    /// and insert into the DAG if it is not already present.
    ///
    /// Panics if any of the children are not already in the DAG.
    pub fn app(&mut self, sym: Symbol, children: Vec<Term>) -> Term {
        let node = Term::App(sym, children.iter().map(|c| self.lookup(c)).collect());

        self.add_node(&node);

        node
    }

    /// Make and return a [`Term::Lit`] with the given literal, and insert into
    /// the DAG if it is not already present.
    pub fn lit(&mut self, lit: Literal) -> Term {
        let node = Term::Lit(lit);

        self.add_node(&node);

        node
    }

    /// Make and return a [`Term::Var`] with the given symbol, and insert into
    /// the DAG if it is not already present.
    pub fn var(&mut self, sym: Symbol) -> Term {
        let node = Term::Var(sym);

        self.add_node(&node);

        node
    }

    fn add_node(&mut self, node: &Term) {
        if self.nodes.get(node).is_none() {
            self.nodes.insert(node.clone());
        }
    }

    /// Recursively converts the given expression to a term.
    ///
    /// This involves inserting every subexpression into this DAG. Because
    /// TermDags are hashconsed, the resulting term is guaranteed to maximally
    /// share subterms.
    pub fn expr_to_term(&mut self, expr: &GenericExpr<Symbol, Symbol>) -> Term {
        let res = match expr {
            GenericExpr::Lit(_, lit) => Term::Lit(lit.clone()),
            GenericExpr::Var(_, v) => Term::Var(*v),
            GenericExpr::Call(_, op, args) => {
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
            Term::Lit(lit) => Expr::lit_no_span(lit.clone()),
            Term::Var(v) => Expr::var_no_span(*v),
            Term::App(op, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.term_to_expr(self.get(*a)))
                    .collect();
                Expr::call_no_span(*op, args)
            }
        }
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: &Term) -> String {
        let mut stored = HashMap::<TermId, String>::default();
        let mut seen = HashSet::<TermId>::default();
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

#[cfg(test)]
mod tests {
    use crate::{ast, DUMMY_SPAN};

    use super::*;

    fn parse_term(s: &str) -> (TermDag, Term) {
        let e = crate::ast::parse_expr(s).unwrap();
        let mut td = TermDag::default();
        let t = td.expr_to_term(&e);
        (td, t)
    }

    #[test]
    fn test_to_from_expr() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let e = crate::ast::parse_expr(s).unwrap();
        let mut td = TermDag::default();
        assert_eq!(td.size(), 0);
        let t = td.expr_to_term(&e);
        assert_eq!(td.size(), 4);
        // the expression above has 4 distinct subterms.
        // in left-to-right, depth-first order, they are:
        //     x, y, (g x y), and the root call to f
        // so we can compute expected answer by hand:
        assert_eq!(
            td.nodes.as_slice().iter().cloned().collect::<Vec<_>>(),
            vec![
                Term::Var("x".into()),
                Term::Var("y".into()),
                Term::App("g".into(), vec![0, 1]),
                Term::App("f".into(), vec![2, 0, 1, 2]),
            ]
        );
        let e2 = td.term_to_expr(&t);
        // This is tested using Sexp's equality because e1 and e2 have different
        // annotations. A better way to test this would be to implement a map_ann
        // function for GenericExpr.
        assert_eq!(e.to_sexp(), e2.to_sexp()); // roundtrip
    }

    #[test]
    fn test_match_term_app() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        match_term_app!(t; {
            ("f", [_, x, _, _]) =>
                assert_eq!(td.term_to_expr(td.get(*x)), ast::GenericExpr::Var(DUMMY_SPAN.clone(), Symbol::new("x"))),
            (head, _) => panic!("unexpected head {}, in {}:{}:{}", head, file!(), line!(), column!())
        })
    }

    #[test]
    fn test_to_string() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        assert_eq!(td.to_string(&t), s);
    }

    #[test]
    fn test_lookup() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        assert_eq!(td.lookup(&t), td.size() - 1);
    }

    #[test]
    fn test_app_var_lit() {
        let s = r#"(f (g x y) x 7 (g x y))"#;
        let (mut td, t) = parse_term(s);
        let x = td.var("x".into());
        let y = td.var("y".into());
        let seven = td.lit(7.into());
        let g = td.app("g".into(), vec![x.clone(), y.clone()]);
        let t2 = td.app("f".into(), vec![g.clone(), x, seven, g]);
        assert_eq!(t, t2);
    }
}
