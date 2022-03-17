mod gj;
mod unionfind;
mod util;
mod value;

mod ast;
use lalrpop_util::lalrpop_mod;
lalrpop_mod!(
    #[allow(clippy::all)]
    grammar
);

use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::{collections::HashMap, hash::Hash};

pub use util::Symbol;
pub use value::Value;

pub use ast::*;
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

pub type Subst = IndexMap<Symbol, Value>;

#[derive(Default)]
pub struct EGraph {
    unionfind: UnionFind,
    sorts: HashSet<Symbol>,
    relations: HashMap<Symbol, Relation>,
    rules: HashMap<Symbol, Rule>,
    globals: HashMap<Symbol, Value>,
}

impl EGraph {
    pub fn union(&mut self, id1: Id, id2: Id) -> Id {
        self.unionfind.union(id1, id2)
    }

    pub fn union_exprs(&mut self, ctx: &Subst, exprs: &[Expr]) -> Option<Value> {
        let mut exprs = exprs.iter();
        if let Some(e) = exprs.next() {
            let mut val = self.eval_expr(ctx, e);
            for e2 in exprs {
                let val2 = self.eval_expr(ctx, e2);
                val = self.unionfind.union_values(val, val2);
            }
            Some(val)
        } else {
            None
        }
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

    pub fn declare_sort(&mut self, name: impl Into<Symbol>) -> Type {
        let name = name.into();
        assert!(self.sorts.insert(name), "Sort '{}' already exists", name);
        Type::Sort(name)
    }

    pub fn declare_function(&mut self, name: impl Into<Symbol>, schema: Schema) -> &mut Relation {
        let name = name.into();
        match self.relations.entry(name) {
            Entry::Vacant(e) => e.insert(Relation::new(schema)),
            Entry::Occupied(_) => panic!("Relation '{}' already exists", name),
        }
    }

    pub fn declare_constructor(
        &mut self,
        name: impl Into<Symbol>,
        types: Vec<Type>,
    ) -> &mut Relation {
        let name = name.into();
        self.declare_function(
            name,
            Schema {
                input: types,
                output: Type::Sort(name),
            },
        )
    }

    // this must be &mut because it'll call "make_set",
    // but it'd be nice if that didn't have to happen
    pub fn eval_expr(&mut self, ctx: &Subst, expr: &Expr) -> Value {
        match expr {
            // TODO should we canonicalize here?
            Expr::Var(var) => ctx
                .get(var)
                .or_else(|| self.globals.get(var))
                .cloned()
                .unwrap_or_else(|| panic!("Couldn't find variable '{var}'")),
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

    fn apply(&mut self, actions: &[Action], subst: &Subst) {
        let mut subst = subst.clone(); // This is slow
        for action in actions {
            match action {
                Action::Define(v, e) => {
                    let value = self.eval_expr(&subst, e);
                    subst
                        .entry(*v)
                        .and_modify(|old| {
                            *old = self.unionfind.union_values(value.clone(), old.clone())
                        })
                        .or_insert(value);
                }
                Action::Union(exprs) => {
                    self.union_exprs(&subst, exprs);
                }
            }
        }
    }

    pub fn run_rules(&mut self, limit: usize) -> usize {
        let mut fingerprint = (self.unionfind.size(), self.unionfind.n_unions());
        for i in 0..limit {
            self.step_rules();
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

    fn step_rules(&mut self) {
        let searched: Vec<_> = self
            .rules
            .values()
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

        let rules = std::mem::take(&mut self.rules);
        for (rule, substs) in rules.values().zip(searched) {
            for subst in substs {
                self.apply(&rule.actions, &subst);
            }
        }
        self.rules = rules;
    }

    pub fn add_named_rule(&mut self, name: impl Into<Symbol>, rule: Rule) {
        let name = name.into();
        match self.rules.entry(name) {
            Entry::Occupied(_) => panic!("Rule '{name}' was already present"),
            Entry::Vacant(e) => e.insert(rule),
        };
    }

    pub fn add_rule(&mut self, rule: Rule) -> Symbol {
        let name = Symbol::from(format!("{:?}", rule));
        self.add_named_rule(name, rule);
        name
    }

    fn for_each_canonicalized(&self, relation: Symbol, mut f: impl FnMut(&[Value])) {
        let mut ids = vec![];
        for (children, value) in &self.relations[&relation].nodes {
            ids.clear();
            // FIXME canonicalize, do we need to with rebuilding?
            // ids.extend(children.iter().map(|id| self.find(value)));
            ids.extend(children.iter().cloned());
            ids.push(value.clone());
            f(&ids);
        }
    }

    pub fn run_command(&mut self, command: Command) {
        match command {
            Command::Datatype { name, variants } => {
                self.declare_sort(name);
                for variant in variants {
                    self.declare_constructor(variant.name, variant.types);
                }
            }
            Command::Function(name, schema) => {
                self.declare_function(name, schema);
            }
            Command::Rule(name, rule) => {
                if let Some(name) = name {
                    self.add_named_rule(name, rule);
                } else {
                    self.add_rule(rule);
                }
            }
            Command::Action(a) => match a {
                Action::Define(v, e) => {
                    let val = self.eval_closed_expr(&e);
                    self.globals.insert(v, val);
                }
                Action::Union(exprs) => {
                    let ctx = Default::default();
                    self.union_exprs(&ctx, &exprs);
                }
            },
            Command::Run(limit) => {
                self.run_rules(limit);
            }
            Command::Extract(_) => todo!(),
            Command::CheckEq(exprs) => {
                let mut exprs = exprs.iter();
                if let Some(first) = exprs.next() {
                    let val = self.eval_closed_expr(first);
                    let val = self.unionfind.find_mut_value(val);
                    for e2 in exprs {
                        let v2 = self.eval_closed_expr(e2);
                        let v2 = self.unionfind.find_mut_value(v2);
                        assert_eq!(val, v2);
                    }
                }
            }
        }
    }

    pub fn run_program(&mut self, input: &str) {
        let parser = grammar::ProgramParser::new();
        let program = parser.parse(input).unwrap();
        for command in program {
            self.run_command(command)
        }
    }
}

pub type Pattern = Expr;

#[derive(Clone, Debug)]
pub struct Query {
    #[allow(dead_code)]
    patterns: Vec<(Symbol, Pattern)>,
    bindings: IndexSet<Symbol>,
    atoms: Vec<Atom>,
}

impl Query {
    pub fn from_groups(groups: Vec<Vec<Pattern>>) -> Self {
        let mut bindings = vec![];
        for (i, group) in groups.into_iter().enumerate() {
            let var = group
                .iter()
                .find_map(|e| e.get_var())
                .unwrap_or_else(|| format!("__group_{}", i).into());
            for e in group {
                if e != Expr::Var(var) {
                    bindings.push((var, e))
                }
            }
        }
        Self::from_patterns(bindings)
    }

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
