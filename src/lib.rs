mod gj;
mod unionfind;
mod util;

use std::fmt::Debug;
use std::{collections::HashMap, hash::Hash};

pub use util::Symbol;

use util::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Id(usize);

impl Id {
    pub(crate) fn fake() -> Self {
        Id(0xbadbeef)
    }
}

impl From<usize> for Id {
    fn from(n: usize) -> Self {
        Id(n)
    }
}

impl From<Id> for usize {
    fn from(id: Id) -> Self {
        id.0
    }
}

type IndexVar = usize;

pub trait Operator: Hash + Eq + Clone + Debug {}

impl Operator for Symbol {}

#[derive(Default)]
pub struct Relation {
    nodes: HashMap<Vec<Id>, Id>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
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

    pub fn walk(&self, pre: &mut impl FnMut(&Self), post: &mut impl FnMut(&Self)) {
        pre(self);
        self.children.iter().for_each(|child| child.walk(pre, post));
        post(self);
    }

    pub fn fold<T>(&self, f: &mut impl FnMut(&Self, Vec<T>) -> T) -> T {
        let ts = self.children.iter().map(|child| child.fold(f)).collect();
        f(self, ts)
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

impl<Op: Operator> EGraph<Op> {
    pub fn union(&mut self, id1: Id, id2: Id) -> Id {
        self.unionfind.union(id1, id2)
    }

    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    pub fn rebuild(&mut self) {
        while self.rebuild_one() != 0 {}
        for (op, r) in &self.relations {
            for (children, value) in &r.nodes {
                println!("{op:?}{children:?} = {value:?}");
            }
        }
    }

    fn rebuild_one(&mut self) -> usize {
        let n_unions = self.unionfind.n_unions();
        for relation in self.relations.values_mut() {
            let old_nodes = std::mem::take(&mut relation.nodes);
            for (mut args, mut value) in old_nodes {
                args.iter_mut()
                    .for_each(|id| *id = self.unionfind.find_mut(*id));
                value = self.unionfind.find_mut(value);
                relation
                    .nodes
                    .entry(args)
                    .and_modify(|value2| *value2 = self.unionfind.union(value, *value2))
                    .or_insert(value);
            }
        }
        self.unionfind.n_unions() - n_unions
    }

    pub fn add_expr(&mut self, expr: &Expr<Op>) -> Id {
        // TODO prevent expensive recursive additions
        let children_ids: Vec<Id> = expr
            .children
            .iter()
            .map(|child| self.add_expr(child))
            .collect();
        self.add_node(expr.op.clone(), &children_ids)
    }

    pub fn add_node(&mut self, op: Op, children: &[Id]) -> Id {
        let r = self.relations.entry(op).or_default();
        // TODO use raw_entry API https://docs.rs/hashbrown/latest/hashbrown/hash_map/enum.RawEntryMut.html
        *r.nodes
            .entry(children.to_vec())
            .or_insert_with(|| self.unionfind.make_set())
    }

    fn lookup_node(&self, op: &Op, children: &[Id]) -> Option<Id> {
        let id = self.relations.get(op)?.nodes.get(children)?;
        Some(self.find(*id))
    }

    pub fn lookup_expr(&self, expr: &Expr<Op>) -> Option<Id> {
        let children: Option<Vec<Id>> = expr
            .children
            .iter()
            .map(|child| self.lookup_expr(child))
            .collect();
        self.lookup_node(&expr.op, &children?)
    }

    fn query(&self, query: &Query<Op>, callback: impl FnMut(&[Id])) {
        let compiled_query = self.compile_query(&query.atoms);
        self.eval(&compiled_query, callback)
    }

    fn apply_pat(&mut self, pattern: &Pattern<Op>, subst: &IndexMap<Symbol, Id>) -> Id {
        match &pattern.op {
            OpOrVar::Var(var) => {
                assert!(pattern.children.is_empty());
                self.find(subst[var])
            }
            OpOrVar::Op(op) => {
                let children: Vec<Id> = pattern
                    .children
                    .iter()
                    .map(|child| self.apply_pat(child, subst))
                    .collect();
                let r = self.relations.entry(op.clone()).or_default();
                let id = *r
                    .nodes
                    .entry(children)
                    .or_insert_with(|| self.unionfind.make_set());
                self.find(id)
            }
        }
    }

    fn apply(&mut self, applier: &Applier<Op>, subst: &IndexMap<Symbol, Id>) {
        let mut subst = subst.clone();
        for (var, pattern) in &applier.assignments {
            let id = self.apply_pat(pattern, &subst);
            subst
                .entry(*var)
                .and_modify(|old| *old = self.union(id, *old))
                .or_insert(id);
        }
    }

    pub fn run_rules(&mut self, limit: usize, rules: &[Rule<Op>]) -> usize {
        let mut fingerprint = (self.unionfind.size(), self.unionfind.n_unions());
        for i in 0..limit {
            self.step_rules(rules);
            self.rebuild();
            let new_fingerprint = (self.unionfind.size(), self.unionfind.n_unions());
            println!(
                "Made {} new nodes, {} new unions",
                new_fingerprint.0 - fingerprint.0,
                new_fingerprint.1 - fingerprint.1
            );
            if new_fingerprint == fingerprint {
                return i + 1;
            }
            fingerprint = new_fingerprint;
        }
        limit
    }

    fn step_rules(&mut self, rules: &[Rule<Op>]) {
        let searched: Vec<_> = rules
            .iter()
            .map(|rule| {
                let mut substs = Vec::<Subst>::new();
                self.query(&rule.query, |ids| {
                    substs.push(
                        rule.query
                            .bindings
                            .iter()
                            .zip(ids)
                            .map(|(s, i)| (*s, *i))
                            .collect(),
                    )
                });
                substs
            })
            .collect();

        for (rule, substs) in rules.iter().zip(searched) {
            for subst in substs {
                self.apply(&rule.applier, &subst);
            }
        }
    }

    fn for_each_canonicalized(&self, relation: &Op, mut f: impl FnMut(&[Id])) {
        let mut ids = vec![];
        for (children, &value) in &self.relations[relation].nodes {
            ids.clear();
            ids.extend(children.iter().map(|id| self.find(*id)));
            ids.push(self.find(value));
            f(&ids);
        }
    }
}

type Subst = IndexMap<Symbol, Id>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpOrVar<Op> {
    Op(Op),
    Var(Symbol),
}

impl<Op: Operator> Operator for OpOrVar<Op> {}

pub type Pattern<Op> = Expr<OpOrVar<Op>>;

// TODO allow applier to bind new variables for its own use
pub struct Applier<Op> {
    assignments: Vec<(Symbol, Pattern<Op>)>,
}

pub struct Query<Op> {
    #[allow(dead_code)]
    patterns: Vec<(Symbol, Pattern<Op>)>,
    bindings: IndexSet<Symbol>,
    atoms: Vec<Atom<Op>>,
}

impl<Op: Operator> Query<Op> {
    pub fn from_patterns(patterns: Vec<(Symbol, Pattern<Op>)>) -> Self {
        let mut bindings = IndexSet::default();
        let mut atoms = vec![];

        for (root, pattern) in &patterns {
            match pattern.op.clone() {
                OpOrVar::Var(_) => panic!("Can't bind a var to a var"),
                OpOrVar::Op(op) => {
                    let mut children_roots: Vec<IndexVar> = pattern
                        .children
                        .iter()
                        .map(|child| {
                            child.fold(&mut |p, mut vars| match p.op.clone() {
                                OpOrVar::Var(var) => bindings.insert_full(var).0,
                                OpOrVar::Op(op) => {
                                    let var = Symbol::from(format!("_aux_{}", bindings.len()));
                                    let (i, was_new) = bindings.insert_full(var);
                                    assert!(was_new);
                                    vars.push(i);
                                    atoms.push(Atom { op, vars });
                                    i
                                }
                            })
                        })
                        .collect();

                    let (i, was_new) = bindings.insert_full(*root);
                    assert!(was_new);
                    children_roots.push(i);
                    atoms.push(Atom {
                        op,
                        vars: children_roots,
                    })
                }
            }
        }
        Self {
            bindings,
            atoms,
            patterns,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom<T> {
    op: T,
    vars: Vec<IndexVar>,
}

pub struct Rule<Op> {
    query: Query<Op>,
    applier: Applier<Op>,
}

impl<Op: Operator> Rule<Op> {
    pub fn new(query: Query<Op>, applier: Applier<Op>) -> Self {
        Self { query, applier }
    }

    pub fn rewrite(lhs: Pattern<Op>, rhs: Pattern<Op>) -> Self {
        let root = Symbol::from("__root");
        let query = Query::from_patterns(vec![(root, lhs)]);
        let applier = Applier {
            assignments: vec![(root, rhs)],
        };
        Self::new(query, applier)
    }
}
