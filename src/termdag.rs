use crate::*;
use std::io::Write;

pub type TermId = usize;

#[allow(rustdoc::private_intra_doc_links)]
/// Like [`Expr`]s but with sharing and deduplication.
///
/// Terms refer to their children indirectly via opaque [TermId]s (internally
/// these are just `usize`s) that map into an ambient [`TermDag`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Term {
    Lit(Literal),
    Var(String),
    App(String, Vec<TermId>),
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
    pub fn app(&mut self, sym: String, children: Vec<Term>) -> Term {
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
    pub fn var(&mut self, sym: String) -> Term {
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
    pub fn expr_to_term(&mut self, expr: &GenericExpr<String, String>) -> Term {
        let res = match expr {
            GenericExpr::Lit(_, lit) => Term::Lit(lit.clone()),
            GenericExpr::Var(_, v) => Term::Var(v.to_owned()),
            GenericExpr::Call(_, op, args) => {
                let args = args
                    .iter()
                    .map(|a| {
                        let term = self.expr_to_term(a);
                        self.lookup(&term)
                    })
                    .collect();
                Term::App(op.clone(), args)
            }
        };
        self.add_node(&res);
        res
    }

    /// Recursively converts the given term to an expression.
    ///
    /// Panics if the term contains subterms that are not in the DAG.
    pub fn term_to_expr(&self, term: &Term, span: Span) -> Expr {
        match term {
            Term::Lit(lit) => Expr::Lit(span, lit.clone()),
            Term::Var(v) => Expr::Var(span, v.clone()),
            Term::App(op, args) => {
                let args: Vec<_> = args
                    .iter()
                    .map(|a| self.term_to_expr(self.get(*a), span.clone()))
                    .collect();
                Expr::Call(span, op.clone(), args)
            }
        }
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: &Term) -> String {
        // Vec is used here instead of String as String doesn't have it's
        // extend_from_within method stabilized.
        let mut result = vec![];
        // subranges of the `result` string containing already stringified subterms
        let mut ranges = HashMap::<TermId, (usize, usize)>::default();
        let id = self.lookup(term);
        // use a stack to avoid stack overflow

        let mut stack = vec![(id, false, None)];
        while let Some((id, space_before, mut start_index)) = stack.pop() {
            if space_before {
                result.push(b' ');
            }

            if let Some((start, end)) = ranges.get(&id) {
                result.extend_from_within(*start..*end);
                continue;
            }

            match self.nodes[id].clone() {
                Term::App(name, children) => {
                    if start_index.is_some() {
                        result.push(b')');
                    } else {
                        stack.push((id, false, Some(result.len())));
                        write!(&mut result, "({}", name).unwrap();
                        for c in children.iter().rev() {
                            stack.push((*c, true, None));
                        }
                    }
                }
                Term::Lit(lit) => {
                    start_index = Some(result.len());
                    write!(&mut result, "{lit}").unwrap();
                }
                Term::Var(v) => {
                    start_index = Some(result.len());
                    write!(&mut result, "{v}").unwrap();
                }
            }

            if let Some(start_index) = start_index {
                ranges.insert(id, (start_index, result.len()));
            }
        }

        String::from_utf8(result).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast::*, span};

    fn parse_term(s: &str) -> (TermDag, Term) {
        let e = Parser::default().get_expr_from_string(None, s).unwrap();
        let mut td = TermDag::default();
        let t = td.expr_to_term(&e);
        (td, t)
    }

    #[test]
    fn test_to_from_expr() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let e = Parser::default().get_expr_from_string(None, s).unwrap();
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
        // This is tested using string equality because e1 and e2 have different
        let e2 = td.term_to_expr(&t, span!());
        // annotations. A better way to test this would be to implement a map_ann
        // function for GenericExpr.
        assert_eq!(format!("{e}"), format!("{e2}")); // roundtrip
    }

    #[test]
    fn test_match_term_app() {
        let s = r#"(f (g x y) x y (g x y))"#;
        let (td, t) = parse_term(s);
        match_term_app!(t; {
            ("f", [_, x, _, _]) => {
                let span = span!();
                assert_eq!(
                    td.term_to_expr(td.get(*x), span.clone()),
                    crate::ast::GenericExpr::Var(span, "x".to_owned())
                )
            }
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
