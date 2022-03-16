mod gj;
mod unionfind;
mod util;
mod value;

use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::{collections::HashMap, hash::Hash};

pub use util::Symbol;
pub use value::Value;

use gj::*;
use unionfind::*;
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

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Sort(Symbol),
    Int,
}

impl Type {
    pub fn is_sort(&self) -> bool {
        matches!(self, Self::Sort(..))
    }
}

pub struct Schema {
    pub input: Vec<Type>,
    pub output: Type,
}

pub struct Relation {
    schema: Schema,
    nodes: HashMap<Vec<Value>, Value>,
}

impl Relation {
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            nodes: Default::default(),
        }
    }

    pub fn get(&mut self, uf: &mut UnionFind, args: &[Value]) -> Value {
        // TODO typecheck?
        self.nodes
            .entry(args.into())
            .or_insert_with(|| uf.make_set().into())
            .clone()
    }

    pub fn set(&mut self, uf: &mut UnionFind, args: &[Value], value: Value) -> Value {
        // TODO typecheck?
        match self.nodes.entry(args.into()) {
            Entry::Occupied(mut e) => {
                let old = e.get().clone();
                e.insert(uf.union_values(old, value))
            }
            Entry::Vacant(e) => e.insert(value).clone(),
        }
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind) -> usize {
        let n_unions = uf.n_unions();
        let old_nodes = std::mem::take(&mut self.nodes);
        for (mut args, mut value) in old_nodes {
            for (a, ty) in args.iter_mut().zip(&self.schema.input) {
                if ty.is_sort() {
                    *a = uf.find_mut_value(a.clone())
                }
            }
            if self.schema.output.is_sort() {
                value = uf.find_mut(value.into()).into();
            }
            self.nodes
                .entry(args)
                .and_modify(|value2| *value2 = uf.union_values(value.clone(), value2.clone()))
                .or_insert(value);
        }
        uf.n_unions() - n_unions
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Expr<T = Value> {
    Leaf(T),
    Var(Symbol),
    Node(Symbol, Vec<Self>),
}

impl<T> Expr<T> {
    pub fn new(op: impl Into<Symbol>, children: impl IntoIterator<Item = Self>) -> Self {
        Self::Node(op.into(), children.into_iter().collect())
    }

    pub fn leaf(op: impl Into<T>) -> Self {
        Self::Leaf(op.into())
    }

    fn children(&self) -> &[Self] {
        match self {
            Expr::Var(_) | Expr::Leaf(_) => &[],
            Expr::Node(_, children) => children,
        }
    }

    pub fn walk(&self, pre: &mut impl FnMut(&Self), post: &mut impl FnMut(&Self)) {
        pre(self);
        self.children()
            .iter()
            .for_each(|child| child.walk(pre, post));
        post(self);
    }

    pub fn fold<Out>(&self, f: &mut impl FnMut(&Self, Vec<Out>) -> Out) -> Out {
        let ts = self.children().iter().map(|child| child.fold(f)).collect();
        f(self, ts)
    }
}

pub type Subst = IndexMap<Symbol, Value>;

#[derive(Default)]
pub struct EGraph {
    unionfind: UnionFind,
    sorts: HashSet<Symbol>,
    relations: HashMap<Symbol, Relation>,
}

impl EGraph {
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
        let mut new_unions = 0;
        for relation in self.relations.values_mut() {
            new_unions += relation.rebuild(&mut self.unionfind);
        }
        new_unions
    }

    pub fn declare_sort(&mut self, symbol: impl Into<Symbol>) {
        let symbol = symbol.into();
        assert!(
            self.sorts.insert(symbol),
            "Relation '{}' already exists",
            symbol
        );
    }

    pub fn declare_function(&mut self, op: impl Into<Symbol>, schema: Schema) -> &mut Relation {
        let op = op.into();
        match self.relations.entry(op) {
            Entry::Vacant(e) => e.insert(Relation::new(schema)),
            Entry::Occupied(_) => panic!("Relation '{}' already exists", op),
        }
    }

    // this must be &mut because it'll call "make_set",
    // but it'd be nice if that didn't have to happen
    pub fn eval_expr(&mut self, ctx: &Subst, expr: &Expr) -> Value {
        match expr {
            Expr::Var(var) => ctx[var].clone(),
            Expr::Leaf(value) => value.clone(),
            Expr::Node(op, args) => {
                let values: Vec<Value> = args.iter().map(|a| self.eval_expr(ctx, a)).collect();
                self.relations
                    .get_mut(op)
                    .unwrap_or_else(|| panic!("Relation '{}' doesn't exist", op))
                    .get(&mut self.unionfind, &values)
            }
        }
    }

    pub fn eval_closed_expr(&mut self, expr: &Expr) -> Value {
        self.eval_expr(&Default::default(), expr)
    }

    // pub fn set_expr(&mut self, ctx: &Subst, expr: &Expr, value: Value) -> Value {
    //     match expr {
    //         Expr::Var(var) => ctx[var].clone(),
    //         Expr::Leaf(value) => value.clone(),
    //         Expr::Node(op, args) => {
    //             let values: Vec<Value> = args.iter().map(|a| self.eval_expr(ctx, a)).collect();
    //             self.relations[op].get(&mut self.unionfind, &values)
    //         }
    //     }
    // }

    fn query(&self, query: &Query, callback: impl FnMut(&[Value])) {
        let compiled_query = self.compile_query(&query.atoms);
        self.run_query(&compiled_query, callback)
    }

    fn apply(&mut self, applier: &Applier, subst: &Subst) {
        let mut subst = subst.clone();
        for (var, pattern) in &applier.assignments {
            let value = self.eval_expr(&subst, pattern);
            // FIXME this is bad
            subst
                .entry(*var)
                .and_modify(|old| *old = self.unionfind.union_values(value.clone(), old.clone()))
                .or_insert(value);
        }
    }

    pub fn run_rules(&mut self, limit: usize, rules: &[Rule]) -> usize {
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

    fn step_rules(&mut self, rules: &[Rule]) {
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
                            .map(|(s, i)| (*s, i.clone()))
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

    fn for_each_canonicalized(&self, relation: Symbol, mut f: impl FnMut(&[Value])) {
        let mut ids = vec![];
        for (children, value) in &self.relations[&relation].nodes {
            ids.clear();
            // FIXME canonicalize
            // ids.extend(children.iter().map(|id| self.find(value)));
            ids.extend(children.iter().cloned());
            ids.push(value.clone());
            f(&ids);
        }
    }
}

pub type Pattern = Expr;

// TODO allow applier to bind new variables for its own use
pub struct Applier {
    assignments: Vec<(Symbol, Pattern)>,
}

pub struct Query {
    #[allow(dead_code)]
    patterns: Vec<(Symbol, Pattern)>,
    bindings: IndexSet<Symbol>,
    atoms: Vec<Atom>,
}

impl Query {
    pub fn from_patterns(patterns: Vec<(Symbol, Pattern)>) -> Self {
        let mut bindings = IndexSet::default();
        let mut atoms = vec![];

        for (root, pattern) in &patterns {
            match pattern {
                Expr::Var(_) => panic!("Not allowed to bind a var to a var for now"),
                Expr::Leaf(_) => panic!("Not allowed to bind a leaf to a var for now"),
                Expr::Node(op, args) => {
                    let mut rootterms: Vec<AtomTerm> = args
                        .iter()
                        .map(|child| {
                            child.fold(&mut |p, mut atomterms| match p {
                                Expr::Leaf(val) => AtomTerm::Value(val.clone()),
                                Expr::Var(var) => AtomTerm::Var(bindings.insert_full(*var).0),
                                Expr::Node(op, _args) => {
                                    let aux = Symbol::from(format!("_aux_{}", bindings.len()));
                                    let (i, was_new) = bindings.insert_full(aux);
                                    assert!(was_new);
                                    let var = AtomTerm::Var(i);
                                    atomterms.push(var.clone());
                                    atoms.push(Atom(*op, atomterms));
                                    var
                                }
                            })
                        })
                        .collect();

                    let (i, was_new) = bindings.insert_full(*root);
                    assert!(was_new);
                    let root = AtomTerm::Var(i);
                    rootterms.push(root);
                    atoms.push(Atom(*op, rootterms));
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

pub struct Rule {
    query: Query,
    applier: Applier,
}

impl Rule {
    pub fn new(query: Query, applier: Applier) -> Self {
        Self { query, applier }
    }

    pub fn rewrite(lhs: Pattern, rhs: Pattern) -> Self {
        let root = Symbol::from("__root");
        let query = Query::from_patterns(vec![(root, lhs)]);
        let applier = Applier {
            assignments: vec![(root, rhs)],
        };
        Self::new(query, applier)
    }
}
