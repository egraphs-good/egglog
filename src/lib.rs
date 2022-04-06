mod gj;
mod unionfind;
mod util;
mod value;

mod ast;

use thiserror::Error;

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

pub struct Function {
    schema: Schema,
    nodes: HashMap<Vec<Value>, Value>,
    updates: usize,
    merge_fn: Option<MergeFn>,
}

impl Function {
    pub fn new(schema: Schema, merge_fn: Option<MergeFn>) -> Self {
        Self {
            schema,
            nodes: Default::default(),
            updates: 0,
            merge_fn,
        }
    }

    pub fn get(&mut self, uf: &mut UnionFind, args: &[Value]) -> Option<Value> {
        // TODO typecheck?
        if self.schema.output.is_sort() {
            let id = self
                .nodes
                .entry(args.into())
                .or_insert_with(|| uf.make_set().into());
            Some(id.clone())
        } else {
            self.nodes.get(args).cloned()
        }
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
                    let new_value = if self.schema.output.is_sort() {
                        uf.union_values(old, value)
                    } else if let Some(merge) = &self.merge_fn {
                        match merge.expr {
                            Expr::Node(s, _) if s.as_str() == "max" => {
                                i64::max(old.into(), value.into()).into()
                            }
                            Expr::Node(s, _) if s.as_str() == "min" => {
                                i64::min(old.into(), value.into()).into()
                            }
                            _ => panic!("Unsupported merge for now"),
                        }
                    } else {
                        panic!("no merge!")
                    };

                    e.insert(new_value.clone());
                    new_value
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

pub struct Primitive {
    _schema: Schema,
    f: fn(&[Value]) -> Value,
}

fn default_primitives() -> HashMap<Symbol, Primitive> {
    macro_rules! prim {
        (@type Int) => { i64 };
        (@type Bool) => { bool };
        (|$($param:ident : $t:ident),*| -> $output:ident { $body:expr }) => {
            Primitive {
                _schema: Schema {
                    input: vec![$(Type::$t),*],
                    output: Type::$output,
                },
                f: |values: &[Value]| -> Value {
                    let mut values = values.iter();
                    $(
                        let $param: prim!(@type $t) = values.next().unwrap().clone().into();
                    )*
                    Value::from($body)
                }
            }
        };
    }

    [
        ("+", prim!(|a: Int, b: Int| -> Int { a + b })),
        ("-", prim!(|a: Int, b: Int| -> Int { a - b })),
        ("*", prim!(|a: Int, b: Int| -> Int { a * b })),
        ("max", prim!(|a: Int, b: Int| -> Int { a.max(b) })),
        ("min", prim!(|a: Int, b: Int| -> Int { a.min(b) })),
    ]
    .into_iter()
    .map(|(k, v)| (Symbol::from(k), v))
    .collect()
}

pub struct EGraph {
    unionfind: UnionFind,
    sorts: HashSet<Symbol>,
    primitives: HashMap<Symbol, Primitive>,
    functions: HashMap<Symbol, Function>,
    rules: HashMap<Symbol, Rule>,
    globals: HashMap<Symbol, Value>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self {
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            globals: Default::default(),
            primitives: default_primitives(),
        }
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(Expr);

impl EGraph {
    pub fn union(&mut self, id1: Id, id2: Id) -> Id {
        self.unionfind.union(id1, id2)
    }

    pub fn union_exprs(&mut self, ctx: &Subst, exprs: &[Expr]) -> Result<Value, NotFoundError> {
        let mut exprs = exprs.iter();
        let e = exprs.next().expect("shouldn't be empty");
        let mut val = self.eval_expr(ctx, e)?;
        for e2 in exprs {
            let val2 = self.eval_expr(ctx, e2)?;
            val = self.unionfind.union_values(val, val2);
        }
        Ok(val)
    }

    pub fn assert_exprs(&mut self, ctx: &Subst, exprs: &[Expr]) -> Result<(), NotFoundError> {
        assert!(!exprs.is_empty());
        for e in exprs {
            self.set_expr(ctx, e, &[true.into()])?;
        }
        Ok(())
    }

    fn set_exprs(
        &mut self,
        ctx: &Subst,
        dst: &Expr,
        srcs: &[Expr],
    ) -> Result<Value, NotFoundError> {
        assert!(!srcs.is_empty());
        let values: Vec<Value> = srcs
            .iter()
            .map(|e| self.eval_expr(ctx, e))
            .collect::<Result<_, _>>()?;
        self.set_expr(ctx, dst, &values)
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
        // for (op, r) in &self.functions {
        //     for (children, value) in &r.nodes {
        //         log::debug!("{op:?}{children:?} = {value:?}");
        //     }
        // }
    }

    fn rebuild_one(&mut self) -> usize {
        let mut new_unions = 0;
        for function in self.functions.values_mut() {
            new_unions += function.rebuild(&mut self.unionfind);
        }
        new_unions
    }

    pub fn declare_sort(&mut self, name: impl Into<Symbol>) -> Type {
        let name = name.into();
        assert!(self.sorts.insert(name), "Sort '{}' already exists", name);
        Type::Sort(name)
    }

    pub fn declare_function(
        &mut self,
        name: impl Into<Symbol>,
        schema: Schema,
        merge: Option<MergeFn>,
    ) -> &mut Function {
        let name = name.into();
        match self.functions.entry(name) {
            Entry::Vacant(e) => e.insert(Function::new(schema, merge)),
            Entry::Occupied(_) => panic!("Function '{}' already exists", name),
        }
    }

    pub fn declare_constructor(
        &mut self,
        name: impl Into<Symbol>,
        types: Vec<Type>,
    ) -> &mut Function {
        let name = name.into();
        self.declare_function(
            name,
            Schema {
                input: types,
                output: Type::Sort(name),
            },
            None,
        )
    }

    // this must be &mut because it'll call "make_set",
    // but it'd be nice if that didn't have to happen
    pub fn eval_expr(&mut self, ctx: &Subst, expr: &Expr) -> Result<Value, NotFoundError> {
        match expr {
            // TODO should we canonicalize here?
            Expr::Var(var) => Ok(ctx
                .get(var)
                .or_else(|| self.globals.get(var))
                .cloned()
                .unwrap_or_else(|| panic!("Couldn't find variable '{var}'"))),
            Expr::Leaf(value) => Ok(value.clone()),
            Expr::Node(op, args) => {
                let values: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(ctx, a))
                    .collect::<Result<_, _>>()?;
                if let Some(rel) = self.functions.get_mut(op) {
                    rel.get(&mut self.unionfind, &values)
                        .ok_or_else(|| NotFoundError(expr.clone()))
                } else if let Some(prim) = self.primitives.get(op) {
                    Ok((prim.f)(&values))
                } else {
                    panic!("Couldn't find function/primitive: {op}")
                }
            }
        }
    }

    pub fn eval_closed_expr(&mut self, expr: &Expr) -> Result<Value, NotFoundError> {
        self.eval_expr(&Default::default(), expr)
    }

    pub fn set_expr(
        &mut self,
        ctx: &Subst,
        expr: &Expr,
        values: &[Value],
    ) -> Result<Value, NotFoundError> {
        assert!(!values.is_empty());
        match expr {
            Expr::Var(var) => Ok(ctx[var].clone()),
            Expr::Leaf(value) => Ok(value.clone()),
            Expr::Node(op, args) => {
                let args: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(ctx, a))
                    .collect::<Result<_, _>>()?;
                let func = self.functions.get_mut(op).unwrap();
                Ok(values
                    .iter()
                    .map(|v| func.set(&mut self.unionfind, &args, v.clone()))
                    .last()
                    .unwrap())
            }
        }
    }

    fn query(&self, query: &Query, callback: impl FnMut(&[Value])) {
        let compiled_query = self.compile_query(&query.atoms);
        self.run_query(&compiled_query, callback)
    }

    fn apply(&mut self, actions: &[Action], subst: &Subst) -> Result<(), NotFoundError> {
        let mut subst = subst.clone(); // This is slow
        for action in actions {
            match action {
                Action::Define(v, e) => {
                    let value = self.eval_expr(&subst, e)?;
                    subst
                        .entry(*v)
                        .and_modify(|old| {
                            *old = self.unionfind.union_values(value.clone(), old.clone())
                        })
                        .or_insert(value);
                }
                Action::Union(exprs) => {
                    self.union_exprs(&subst, exprs)?;
                }
                Action::Assert(exprs) => {
                    self.assert_exprs(&subst, exprs)?;
                }
                Action::Set(dst, srcs) => {
                    self.set_exprs(&subst, dst, srcs)?;
                }
            }
        }
        Ok(())
    }

    pub fn run_rules(&mut self, limit: usize) {
        for _ in 0..limit {
            self.step_rules();
            let updates = self.rebuild();
            log::debug!("Made {updates} updates",);
            // if updates == 0 {
            //     log::debug!("Breaking early!");
            //     break;
            // }
        }

        // TODO detect functions
        for (name, r) in &self.functions {
            log::debug!("{name}:");
            for (args, val) in &r.nodes {
                log::debug!("  {args:?} = {val}");
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
                // we ignore the result here because rule applications are best effort
                let _result: Result<_, NotFoundError> = self.apply(&rule.actions, &subst);
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

    fn for_each_canonicalized(&self, name: Symbol, mut f: impl FnMut(&[Value])) {
        let mut ids = vec![];
        for (children, value) in &self.functions[&name].nodes {
            ids.clear();
            // FIXME canonicalize, do we need to with rebuilding?
            // ids.extend(children.iter().map(|id| self.find(value)));
            ids.extend(children.iter().cloned());
            ids.push(value.clone());
            f(&ids);
        }
    }

    pub fn run_command(&mut self, command: Command) -> Result<String, Error> {
        #[allow(clippy::useless_format)]
        match command {
            Command::Datatype { name, variants } => {
                self.declare_sort(name);
                for variant in variants {
                    self.declare_constructor(variant.name, variant.types);
                }
                Ok(format!("Declared datatype {name}."))
            }
            Command::Function {
                name,
                schema,
                merge,
            } => {
                self.declare_function(name, schema, merge);
                Ok(format!("Declared function {name}."))
            }
            Command::Rule(name, rule) => {
                if let Some(name) = name {
                    self.add_named_rule(name, rule);
                    Ok(format!("Declared rule {name}."))
                } else {
                    self.add_rule(rule);
                    Ok(format!("Declared rule."))
                }
            }
            Command::Action(a) => match a {
                Action::Define(v, e) => {
                    let val = self.eval_closed_expr(&e)?;
                    self.globals.insert(v, val.clone());
                    Ok(format!("Defined {v} = {val}."))
                }
                Action::Union(exprs) => {
                    let ctx = Default::default();
                    self.union_exprs(&ctx, &exprs)?;
                    Ok(format!("Unioned."))
                }
                Action::Assert(exprs) => {
                    let ctx = Default::default();
                    self.assert_exprs(&ctx, &exprs)?;
                    Ok(format!("Asserted."))
                }
                Action::Set(dst, srcs) => {
                    let ctx = Default::default();
                    self.set_exprs(&ctx, &dst, &srcs)?;
                    Ok(format!("Set."))
                }
            },
            Command::Run(limit) => {
                self.run_rules(limit);
                Ok(format!("Ran {limit}."))
            }
            Command::Extract(_) => todo!(),
            Command::CheckEq(exprs) => {
                let mut iter = exprs.iter();
                if let Some(first) = iter.next() {
                    let val = self.eval_closed_expr(first)?;
                    let val = self.bad_find_value(val);
                    for e2 in iter {
                        let v2 = self.eval_closed_expr(e2)?;
                        let v2 = self.bad_find_value(v2);
                        assert_eq!(val, v2);
                    }
                    Ok(format!("Checked {} exprs equal to {val}.", exprs.len()))
                } else {
                    Ok("Empty assert.".into())
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

    pub fn run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let parser = grammar::ProgramParser::new();
        let program = parser
            .parse(input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?;
        program
            .into_iter()
            .map(|cmd| self.run_command(cmd))
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] lalrpop_util::ParseError<usize, String, &'static str>),
    #[error(transparent)]
    NotFoundError(#[from] NotFoundError),
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

        log::debug!("atoms: {:?}", atoms);
        Self {
            bindings,
            atoms,
            groups,
        }
    }
}
