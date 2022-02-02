mod gj;
mod unionfind;
mod util;

use std::{collections::HashMap, hash::Hash};

use util::Symbol;

pub type Id = usize;
type IndexVar = usize;

pub trait Operator: Hash + Eq + Clone {}

#[derive(Default)]
pub struct Relation {
    nodes: HashMap<Vec<Id>, Id>,
}

impl Relation {
    fn for_each(&self, mut f: impl FnMut(&[Id])) {
        let mut buf = vec![];
        for (ids, id) in &self.nodes {
            buf.clear();
            buf.extend_from_slice(ids);
            buf.push(*id);
            f(&buf)
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Expr<T> {
    op: T,
    children: Vec<Self>,
}

impl<Op> Expr<Op> {
    pub fn new(op: Op, children: Vec<Self>) -> Self {
        Self { op, children }
    }

    pub fn leaf(op: Op) -> Self {
        Self::new(op, vec![])
    }
}

pub struct EGraph<T> {
    unionfind: unionfind::UnionFind,
    relations: HashMap<T, Relation>,
}

impl<T> Default for EGraph<T> {
    fn default() -> Self {
        Self {
            relations: Default::default(),
            unionfind: Default::default(),
        }
    }
}

impl<T: Operator> EGraph<T> {
    pub fn add_expr(&mut self, expr: &Expr<T>) -> Id {
        // TODO prevent expensive recursive additions
        let children_ids: Vec<Id> = expr
            .children
            .iter()
            .map(|child| self.add_expr(child))
            .collect();
        self.add_node(expr.op.clone(), &children_ids)
    }

    pub fn add_node(&mut self, op: T, children: &[Id]) -> Id {
        let r = self.relations.entry(op).or_default();
        // TODO use raw_entry API https://docs.rs/hashbrown/latest/hashbrown/hash_map/enum.RawEntryMut.html
        *r.nodes
            .entry(children.to_vec())
            .or_insert_with(|| self.unionfind.make_set())
    }

    pub fn query(&self, query: &[Atom<T>], callback: impl FnMut(&[Id])) {
        let compiled_query = self.compile_query(query);
        self.eval(&compiled_query, callback)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpOrVar<Op> {
    Op(Op),
    Var(Symbol),
}

impl<Op: Operator> Operator for OpOrVar<Op> {}

pub struct Query<Op> {
    terms: Vec<Expr<OpOrVar<Op>>>,
}

// impl<Op> Query<Op> {
//     fn flatten(&self) -> Vec<Atom<Op>> {}
// }

#[derive(Clone)]
pub struct Atom<T> {
    op: T,
    vars: Vec<IndexVar>,
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! expr {
        (($($arg:tt)*)) => { expr!($($arg)*) }; // unpack parenthesized things
        ($op:tt $($arg:tt)*) => { Expr::new(stringify!($op), vec![$(expr!($arg)),*]) };
    }

    macro_rules! pattern {
        ($x:ident) => { Expr::leaf(OpOrVar::Var(Symbol::from(stringify!($x)))) };
        (($($arg:tt)*)) => { pattern!($($arg)*) }; // unpack parenthesized things
        ($op:tt $($arg:tt)*) => { Expr::new(OpOrVar::Op(stringify!($op)), vec![$(pattern!($arg)),*]) };
    }

    #[test]
    fn test_macro() {
        assert_eq!(
            expr!(+ (+ 1 2) 3),
            Expr::new(
                "+",
                vec![
                    Expr::new("+", vec![Expr::leaf("1"), Expr::leaf("2")]),
                    Expr::leaf("3")
                ]
            )
        );

        use OpOrVar::*;
        assert_eq!(
            pattern!(+ x (+ 3 y)),
            Expr::new(
                Op("+"),
                vec![
                    Expr::leaf(Var(Symbol::from("x"))),
                    Expr::new(
                        Op("+"),
                        vec![Expr::leaf(Op("3")), Expr::leaf(Var(Symbol::from("y"))),]
                    ),
                ]
            )
        );
    }
}
