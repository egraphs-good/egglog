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
    updates: usize,
}

impl Relation {
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            nodes: Default::default(),
            updates: 0,
        }
    }

    pub fn get(&mut self, uf: &mut UnionFind, args: &[Value]) -> Value {
        // TODO typecheck?
        self.nodes
            .entry(args.into())
            .or_insert_with(|| {
                self.updates += 1;
                match self.schema.output {
                    Type::Sort(_) => uf.make_set().into(),
                    Type::Bool => false.into(),
                    Type::Int => todo!(),
                }
            })
            .clone()
    }

    pub fn set(&mut self, uf: &mut UnionFind, args: &[Value], value: Value) -> Value {
        // TODO typecheck?
        match self.nodes.entry(args.into()) {
            Entry::Occupied(mut e) => {
                let old = e.get().clone();
                if old == value {
                    value
                } else {
                    self.updates += 1;
                    if self.schema.output.is_sort() {
                        e.insert(uf.union_values(old, value))
                    } else {
                        e.insert(value)
                    }
                }
            }
            Entry::Vacant(e) => {
                self.updates += 1;
                e.insert(value).clone()
            }
        }
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind) -> usize {
        // FIXME this doesn't compute updates properly
        let n_unions = uf.n_unions();
        let old_nodes = std::mem::take(&mut self.nodes);
        for (mut args, value) in old_nodes {
            for (a, ty) in args.iter_mut().zip(&self.schema.input) {
                if ty.is_sort() {
                    *a = uf.find_mut_value(a.clone())
                }
            }
            if self.schema.output.is_sort() {
                self.nodes
                    .entry(args)
                    .and_modify(|value2| *value2 = uf.union_values(value.clone(), value2.clone()))
                    .or_insert_with(|| uf.find_mut_value(value));
            } else {
                self.nodes
                    .entry(args)
                    // .and_modify(|value2| *value2 = uf.union_values(value.clone(), value2.clone()))
                    .or_insert(value);
            }
        }
        uf.n_unions() - n_unions + std::mem::take(&mut self.updates)
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

    pub fn assert_exprs(&mut self, ctx: &Subst, exprs: &[Expr]) {
        for e in exprs {
            self.set_expr(ctx, e, true.into());
        }
    }

    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    pub fn rebuild(&mut self) -> usize {
        let mut updates = 0;
        loop {
            let new = self.rebuild_one();
            updates += new;
            if new == 0 {
                return updates;
            }
        }
        // for (op, r) in &self.relations {
        //     for (children, value) in &r.nodes {
        //         println!("{op:?}{children:?} = {value:?}");
        //     }
        // }
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

    pub fn set_expr(&mut self, ctx: &Subst, expr: &Expr, value: Value) -> Value {
        match expr {
            Expr::Var(var) => ctx[var].clone(),
            Expr::Leaf(value) => value.clone(),
            Expr::Node(op, args) => {
                let values: Vec<Value> = args.iter().map(|a| self.eval_expr(ctx, a)).collect();
                self.relations
                    .get_mut(op)
                    .unwrap()
                    .set(&mut self.unionfind, &values, value)
            }
        }
    }

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
                Action::Assert(exprs) => {
                    self.assert_exprs(&subst, exprs);
                }
            }
        }
    }

    pub fn run_rules(&mut self, limit: usize) {
        for _ in 0..limit {
            self.step_rules();
            let updates = self.rebuild();
            println!("Made {updates} updates",);
            if updates == 0 {
                break;
            }
        }

        // TODO detect functions
        for (name, r) in &self.relations {
            println!("{name}:");
            for (args, val) in &r.nodes {
                println!("  {args:?} = {val:?}");
            }
        }
    }

    fn step_rules(&mut self) {
        let searched: Vec<_> = self
            .rules
            .values()
            .map(|rule| {
                let mut substs = Vec::<Subst>::new();
                self.query(&rule.query, |values| {
                    let get = |a: &AtomTerm| -> Value {
                        match a {
                            AtomTerm::Var(i) => values[*i].clone(),
                            AtomTerm::Value(val) => val.clone(),
                        }
                    };
                    substs.push(
                        rule.query
                            .bindings
                            .iter()
                            .map(|(sym, a)| (*sym, get(a)))
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
                Action::Assert(exprs) => {
                    let ctx = Default::default();
                    self.assert_exprs(&ctx, &exprs);
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
                    let val = self.bad_find_value(val);
                    for e2 in exprs {
                        let v2 = self.eval_closed_expr(e2);
                        let v2 = self.bad_find_value(v2);
                        assert_eq!(val, v2);
                    }
                }
            }
        }
    }

    // this is bad because we shouldn't inspect values like this, we should use type information
    fn bad_find_value(&self, value: Value) -> Value {
        match value.0 {
            value::ValueInner::Bool(b) => b.into(),
            value::ValueInner::Id(id) => self.unionfind.find(id).into(),
            value::ValueInner::Int(i) => i.into(),
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
    groups: Vec<Vec<Pattern>>,
    bindings: IndexMap<Symbol, AtomTerm>,
    atoms: Vec<Atom>,
}

impl Query {
    pub fn from_groups(groups: Vec<Vec<Pattern>>) -> Self {
        #[derive(PartialEq, Eq, Hash, Clone)]
        enum VarOrValue {
            Var(Symbol),
            Value(Value),
        }

        let mut aux_counter = 0;
        let mut uf = SparseUnionFind::<VarOrValue>::default();
        let mut pre_atoms: Vec<(Symbol, Vec<VarOrValue>)> = vec![];

        for (i, group) in groups.iter().enumerate() {
            let group_var = VarOrValue::Var(Symbol::from(format!("__group_{i}")));
            uf.insert(group_var.clone());
            for expr in group {
                let vv = expr.fold(&mut |expr, mut child_pre_atoms| -> VarOrValue {
                    let vv = match expr {
                        Expr::Leaf(value) => VarOrValue::Value(value.clone()),
                        Expr::Var(var) => VarOrValue::Var(*var),
                        Expr::Node(op, _) => {
                            let aux = VarOrValue::Var(format!("_aux_{}", aux_counter).into());
                            aux_counter += 1;
                            child_pre_atoms.push(aux.clone());
                            pre_atoms.push((*op, child_pre_atoms));
                            aux
                        }
                    };
                    uf.insert(vv.clone());
                    vv
                });
                uf.union(group_var.clone(), vv);
            }
        }

        let mut next_var_index = 0;
        let mut bindings = IndexMap::default();

        for set in uf.sets() {
            let mut values: Vec<Value> = set
                .iter()
                .filter_map(|vv| match vv {
                    VarOrValue::Var(_) => None,
                    VarOrValue::Value(val) => Some(val.clone()),
                })
                .collect();

            if values.len() > 1 {
                panic!("too many values")
            }

            let atom_term = if let Some(value) = values.pop() {
                AtomTerm::Value(value)
            } else {
                debug_assert!(set.iter().all(|vv| matches!(vv, VarOrValue::Var(_))));
                let a = AtomTerm::Var(next_var_index);
                next_var_index += 1;
                a
            };

            assert!(values.is_empty());
            for vv in set {
                if let VarOrValue::Var(var) = vv {
                    let old = bindings.insert(var, atom_term.clone());
                    assert!(old.is_none());
                }
            }
        }

        let vv_to_atomterm = |vv: VarOrValue| match vv {
            VarOrValue::Var(v) => bindings[&v].clone(),
            VarOrValue::Value(val) => AtomTerm::Value(val),
        };
        let atoms = pre_atoms
            .into_iter()
            .map(|(sym, vvs)| Atom(sym, vvs.into_iter().map(vv_to_atomterm).collect()))
            .collect();

        println!("atoms: {:?}", atoms);
        Self {
            bindings,
            atoms,
            groups,
        }
    }
}
