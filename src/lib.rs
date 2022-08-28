pub mod ast;
mod extract;
mod gj;
pub mod sort;
mod typecheck;
mod unionfind;
mod util;
mod value;

use hashbrown::hash_map::Entry;
use instant::{Duration, Instant};
use sort::*;
use thiserror::Error;

use ast::*;

use std::fmt::Write;
use std::hash::Hash;
use std::iter::once;
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use std::{fmt::Debug, sync::Arc};
use typecheck::{AtomTerm, Bindings};

type ArcSort = Arc<dyn Sort>;

pub use value::*;

use gj::*;

use unionfind::*;
use util::*;

use crate::typecheck::TypeError;

#[derive(Default, Clone, Debug)]
struct FunctionData {
    nodes: HashMap<Rc<[Value]>, Value>,
    index: HashMap<Id, HashSet<Rc<[Value]>>>,
}

impl FunctionData {
    fn add_to_index(
        id_index: &mut HashMap<Id, HashSet<Rc<[Value]>>>,
        schema: &ResolvedSchema,
        vs: Rc<[Value]>,
        ret: &Value,
    ) {
        for (v, ty) in vs
            .iter()
            .zip(&schema.input)
            .chain(once((ret, &schema.output)))
        {
            if !ty.is_eq_sort() {
                continue;
            }
            // XXX: Hack
            let id = Id::from(v.bits as usize);
            id_index.entry(id).or_default().insert(vs.clone());
        }
    }
    fn remove_from_index(
        id_index: &mut HashMap<Id, HashSet<Rc<[Value]>>>,
        schema: &ResolvedSchema,
        vs: Rc<[Value]>,
        ret: &Value,
    ) {
        for (v, ty) in vs
            .iter()
            .zip(&schema.input)
            .chain(once((ret, &schema.output)))
        {
            if !ty.is_eq_sort() {
                continue;
            }
            // XXX: Hack
            let id = Id::from(v.bits as usize);
            if let Some(set) = id_index.get_mut(&id) {
                set.remove(&vs);
                if set.is_empty() {
                    id_index.remove(&id);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Function {
    decl: FunctionDecl,
    schema: ResolvedSchema,
    data: FunctionData,
    scratch_indexes: Vec<Rc<[Value]>>,
}

#[derive(Clone, Debug)]
struct ResolvedSchema {
    input: Vec<ArcSort>,
    output: ArcSort,
}

impl Function {
    fn insert_value(&mut self, uf: &mut UnionFind, vs: Rc<[Value]>, ret: Value) -> Option<Value> {
        self.insert_value_maybe_unify(uf, vs, |_, prev| (ret, prev))
    }
    fn canonicalize_args(&self, uf: &mut UnionFind, vs: Rc<[Value]>) -> Rc<[Value]> {
        let mut is_canon = true;
        for (ty, v) in self.schema.input.iter().zip(vs.iter()) {
            if ty.is_eq_sort() {
                let id = Id::from(v.bits as usize);
                if id != uf.find(id) {
                    is_canon = false;
                    break;
                }
            }
        }
        if is_canon {
            return vs;
        }
        let mut scratch = Vec::from_iter(vs.iter().copied());
        for (ty, v) in self.schema.input.iter().zip(scratch.iter_mut()) {
            if ty.is_eq_sort() {
                *v = uf.find_mut_value(*v);
            }
        }
        scratch.into()
    }
    fn canonicalize(
        &self,
        uf: &mut UnionFind,
        vs: Rc<[Value]>,
        mut ret: Value,
    ) -> (Rc<[Value]>, Value) {
        let vs = self.canonicalize_args(uf, vs);
        if self.schema.output.is_eq_sort() {
            ret = uf.find_mut_value(ret);
        }
        (vs, ret)
    }
    fn insert_value_maybe_unify(
        &mut self,
        uf: &mut UnionFind,
        vs: Rc<[Value]>,
        mut merge: impl FnMut(&mut UnionFind, Option<Value>) -> (Value, Option<Value>),
    ) -> Option<Value> {
        if let Some(v) = self.data.nodes.get_mut(&vs) {
            let (to_insert, to_return) = merge(uf, Some(*v));
            if !self.schema.output.is_eq_sort() {
                *v = to_insert;
                return to_return;
            }
            // We are removing an eq sort value from the map and replacing
            // it with a new one. We have to do the same to the index first.
            // XXX: hack.
            let id = Id::from(v.bits as usize);
            let ixs = self
                .data
                .index
                .get_mut(&id)
                .expect("all return values must be mapped");
            let _prev = ixs.remove(&vs);
            debug_assert!(_prev, "all return values should be present in the index");
            if ixs.is_empty() {
                self.data.index.remove(&id);
            }
            // XXX: hack.
            let new_id = Id::from(to_insert.bits as usize);
            self.data
                .index
                .entry(new_id)
                .or_default()
                .insert(vs.clone());
            *v = to_insert;
            return to_return;
        }
        let (to_insert, to_return) = merge(uf, None);
        let (canon, ret) = self.canonicalize(uf, vs, to_insert);
        FunctionData::add_to_index(
            &mut self.data.index,
            &self.schema,
            canon.clone(),
            &to_insert,
        );
        self.data.nodes.insert(canon, ret);
        to_return
    }

    fn remove_value(&mut self, vs: &[Value]) -> Option<Value> {
        let (vs, prev) = self.data.nodes.remove_entry(vs)?;
        // First, remove all the mappings for the old entries.
        FunctionData::remove_from_index(&mut self.data.index, &self.schema, vs, &prev);
        Some(prev)
    }

    fn get(&mut self, vs: &[Value]) -> Option<&Value> {
        self.data.nodes.get(vs)
    }

    fn clear(&mut self) {
        self.data.nodes.clear();
        self.data.index.clear();
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind) {
        let mut scratch = mem::take(&mut self.scratch_indexes);
        // For each Id that has a new root.
        for id in &*uf.recently_merged_dyn().borrow() {
            // Get all of the argument tuples that mention it.
            scratch.extend(
                self.data
                    .index
                    .get(id)
                    .into_iter()
                    .flat_map(|x| x.iter().cloned()),
            );
            for vs in scratch.drain(..) {
                // Remove them from the function mapping.
                let ret = if let Some(ret) = self.remove_value(&vs) {
                    ret
                } else {
                    continue;
                };
                let (vs, ret) = self.canonicalize(uf, vs, ret);
                let is_eq_output = self.schema.output.is_eq_sort();
                // Insert them back, unifying the old return value whatever is
                // already there.
                self.insert_value_maybe_unify(uf, vs, |uf, prev| {
                    if let Some(prev) = prev {
                        if is_eq_output {
                            (uf.union_values(prev, ret), None)
                        } else {
                            (prev, None)
                        }
                    } else {
                        (ret, None)
                    }
                });
            }
        }
    }
}

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    fn accept(&self, types: &[&dyn Sort]) -> Option<ArcSort>;
    fn apply(&self, values: &[Value]) -> Option<Value>;
}

#[derive(Clone)]
pub struct Primitive(Arc<dyn PrimitiveLike>);

impl Deref for Primitive {
    type Target = dyn PrimitiveLike;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl Hash for Primitive {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Eq for Primitive {}
impl PartialEq for Primitive {
    fn eq(&self, other: &Self) -> bool {
        // this is a bit of a hack, but clippy says we don't want to compare the
        // vtables, just the data pointers
        std::ptr::eq(
            Arc::as_ptr(&self.0) as *const u8,
            Arc::as_ptr(&other.0) as *const u8,
        )
    }
}

impl Debug for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prim({})", self.0.name())
    }
}

impl<T: PrimitiveLike + 'static> From<T> for Primitive {
    fn from(p: T) -> Self {
        Self(Arc::new(p))
    }
}

pub struct SimplePrimitive {
    name: Symbol,
    input: Vec<ArcSort>,
    output: ArcSort,
    f: fn(&[Value]) -> Option<Value>,
}

impl PrimitiveLike for SimplePrimitive {
    fn name(&self) -> Symbol {
        self.name
    }
    fn accept(&self, types: &[&dyn Sort]) -> Option<ArcSort> {
        if self.input.len() != types.len() {
            return None;
        }
        // TODO can we use a better notion of equality than just names?
        self.input
            .iter()
            .zip(types)
            .all(|(a, b)| a.name() == b.name())
            .then(|| self.output.clone())
    }
    fn apply(&self, values: &[Value]) -> Option<Value> {
        (self.f)(values)
    }
}

#[derive(Clone)]
pub struct EGraph {
    egraphs: Vec<Self>,
    unionfind: UnionFind,
    presorts: HashMap<Symbol, PreSort>,
    sorts: HashMap<Symbol, Arc<dyn Sort>>,
    primitives: HashMap<Symbol, Vec<Primitive>>,
    functions: HashMap<Symbol, Function>,
    rules: HashMap<Symbol, Rule>,
    globals: HashMap<Symbol, Value>,
}

#[derive(Clone, Debug)]
struct Rule {
    query: CompiledQuery,
    bindings: Bindings,
    head: Vec<Action>,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            egraphs: vec![],
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            globals: Default::default(),
            primitives: Default::default(),
            presorts: Default::default(),
        };
        egraph.add_sort(UnitSort::new("Unit".into()));
        egraph.add_sort(StringSort::new("String".into()));
        egraph.add_sort(I64Sort::new("i64".into()));
        egraph.add_sort(RationalSort::new("Rational".into()));
        egraph.presorts.insert("Map".into(), MapSort::make_sort);
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(Expr);

impl EGraph {
    pub fn add_sort<S: Sort + 'static>(&mut self, sort: S) {
        self.add_arcsort(Arc::new(sort));
    }

    pub fn push(&mut self) {
        self.egraphs.push(self.clone());
    }

    pub fn pop(&mut self) -> Result<(), Error> {
        match self.egraphs.pop() {
            Some(e) => {
                *self = e;
                Ok(())
            }
            None => Err(Error::Pop),
        }
    }

    pub fn add_arcsort(&mut self, sort: ArcSort) {
        match self.sorts.entry(sort.name()) {
            Entry::Occupied(_) => panic!(),
            Entry::Vacant(e) => {
                e.insert(sort.clone());
                sort.register_primitives(self);
            }
        };
    }

    fn get_sort<S: Sort + Send + Sync>(&self) -> Arc<S> {
        for sort in self.sorts.values() {
            let sort = sort.clone().as_arc_any();
            if let Ok(sort) = Arc::downcast(sort) {
                return sort;
            }
        }
        // TODO handle if multiple match?
        // could handle by type id??
        panic!("Failed to lookup sort: {}", std::any::type_name::<S>());
    }

    fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        let prim = prim.into();
        self.primitives.entry(prim.name()).or_default().push(prim);
    }

    pub fn union(&mut self, id1: Id, id2: Id) -> Id {
        self.unionfind.union(id1, id2)
    }

    #[track_caller]
    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        for (name, function) in self.functions.iter() {
            for (inputs, output) in function.data.nodes.iter() {
                for input in inputs.iter() {
                    assert_eq!(
                        input,
                        &self.bad_find_value(*input),
                        "{name}({inputs:?}) = {output:?}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output,
                    &self.bad_find_value(*output),
                    "{name}({inputs:?}) = {output:?}\n{:?}",
                    function.schema,
                )
            }
        }
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

    pub fn eval_actions(
        &mut self,
        mut ctx: Option<Subst>,
        actions: &[Action],
    ) -> Result<(), Error> {
        let default = Subst::default();
        for action in actions {
            match action {
                Action::Panic(msg) => {
                    panic!("panic {} {:?}", msg, ctx.as_ref().unwrap_or(&default))
                }
                Action::Expr(e) => {
                    self.eval_expr(ctx.as_ref().unwrap_or(&default), e)?;
                }
                Action::Define(x, e) => {
                    if let Some(ctx) = ctx.as_mut() {
                        let value = self.eval_expr(ctx, e)?;
                        ctx.insert(*x, value);
                    } else {
                        let value = self.eval_expr(&default, e)?;
                        self.globals.insert(*x, value);
                    }
                }
                Action::Set(f, args, e) => {
                    let ctx = ctx.as_ref().unwrap_or(&default);
                    let values = args
                        .iter()
                        .map(|a| self.eval_expr(ctx, a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = self.eval_expr(ctx, e)?;
                    let function = self
                        .functions
                        .get_mut(f)
                        .ok_or_else(|| NotFoundError(e.clone()))?;
                    let old_value =
                        function.insert_value(&mut self.unionfind, values.clone().into(), value);

                    if let Some(old_value) = old_value {
                        if value != old_value {
                            let out = &function.schema.output;
                            match function.decl.merge.as_ref() {
                                None if out.name().as_str() == "Unit" => (),
                                None if out.is_eq_sort() => {
                                    self.unionfind.union_values(old_value, value);
                                }
                                Some(expr) => {
                                    let mut ctx = Subst::default();
                                    ctx.insert("old".into(), old_value);
                                    ctx.insert("new".into(), value);
                                    let expr = expr.clone(); // break the borrow of `function`
                                    let new_value = self.eval_expr(&ctx, &expr)?;
                                    self.functions.get_mut(f).unwrap().insert_value(
                                        &mut self.unionfind,
                                        values.into(),
                                        new_value,
                                    );
                                }
                                _ => panic!("invalid merge for {}", function.decl.name),
                            }
                        }
                    }
                }
                Action::Union(a, b) => {
                    let ctx = ctx.as_ref().unwrap_or(&default);
                    let a = self.eval_expr(ctx, a)?;
                    let b = self.eval_expr(ctx, b)?;
                    self.unionfind.union_values(a, b);
                }
                Action::Delete(sym, args) => {
                    let ctx = ctx.as_ref().unwrap_or(&default);
                    let values = args
                        .iter()
                        .map(|a| self.eval_expr(ctx, a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let function = self
                        .functions
                        .get_mut(sym)
                        .ok_or(TypeError::Unbound(*sym))?;
                    function.remove_value(&values);
                }
            }
        }
        Ok(())
    }

    pub fn check_with(&mut self, ctx: &Subst, fact: &Fact) -> Result<(), Error> {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() > 1);
                let values: Vec<Value> = exprs
                    .iter()
                    .map(|e| self.eval_expr(ctx, e).map(|v| self.bad_find_value(v)))
                    .collect::<Result<_, _>>()?;
                for v in &values[1..] {
                    if &values[0] != v {
                        println!("Check failed");
                        // the check failed, so print out some useful info
                        self.rebuild();
                        for value in &values {
                            if let Some((_tag, id)) = self.value_to_id(*value) {
                                let best = self.extract(*value).1;
                                println!("{}: {}", id, best);
                            }
                        }
                        return Err(Error::CheckError(values[0], *v));
                    }
                }
            }
            Fact::Fact(expr) => match expr {
                Expr::Lit(_) => panic!("can't assert a literal"),
                Expr::Var(_) => panic!("can't assert a var"),
                Expr::Call(sym, args) => {
                    let values: Vec<Value> = args
                        .iter()
                        .map(|e| self.eval_expr(ctx, e).map(|v| self.bad_find_value(v)))
                        .collect::<Result<_, _>>()?;
                    if let Some(f) = self.functions.get_mut(sym) {
                        // FIXME We don't have a unit value
                        assert_eq!(f.schema.output.name().as_str(), "Unit");
                        f.get(&values).ok_or_else(|| NotFoundError(expr.clone()))?;
                    } else if self.primitives.contains_key(sym) {
                        // HACK
                        // we didn't typecheck so we don't know which prim to call
                        let val = self.eval_expr(ctx, expr)?;
                        assert_eq!(val, Value::unit());
                    } else {
                        return Err(Error::TypeError(TypeError::Unbound(*sym)));
                    }
                }
            },
        }
        Ok(())
    }

    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    pub fn rebuild(&mut self) -> usize {
        let mut updates = 0;
        loop {
            let new = self.rebuild_one();
            log::debug!("{new} rebuilds?");
            updates += new;
            if new == 0 {
                break;
            }
        }
        self.debug_assert_invariants();
        updates
    }

    fn rebuild_one(&mut self) -> usize {
        self.unionfind.update();
        for function in self.functions.values_mut() {
            function.rebuild(&mut self.unionfind);
        }
        self.unionfind.recently_merged().len()
    }

    pub fn declare_sort(&mut self, name: impl Into<Symbol>) -> Result<(), Error> {
        let name = name.into();
        match self.sorts.entry(name) {
            Entry::Occupied(_) => Err(Error::SortAlreadyBound(name)),
            Entry::Vacant(e) => {
                e.insert(Arc::new(EqSort { name }));
                Ok(())
            }
        }
    }

    pub fn declare_function(&mut self, decl: &FunctionDecl) -> Result<(), Error> {
        let mut input = Vec::with_capacity(decl.schema.input.len());
        for s in &decl.schema.input {
            input.push(match self.sorts.get(s) {
                Some(sort) => sort.clone(),
                None => return Err(Error::TypeError(TypeError::Unbound(*s))),
            })
        }

        let output = match self.sorts.get(&decl.schema.output) {
            Some(sort) => sort.clone(),
            None => return Err(Error::TypeError(TypeError::Unbound(decl.schema.output))),
        };

        let function = Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            data: Default::default(),
            scratch_indexes: Default::default(),
            // TODO figure out merge and default here
        };

        let old = self.functions.insert(decl.name, function);
        if old.is_some() {
            return Err(TypeError::FunctionAlreadyBound(decl.name).into());
        }

        Ok(())
    }

    pub fn declare_constructor(
        &mut self,
        name: impl Into<Symbol>,
        types: Vec<Symbol>,
        sort: impl Into<Symbol>,
    ) -> Result<(), Error> {
        let name = name.into();
        let sort = sort.into();
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: types,
                output: sort,
            },
            merge: None,
            default: None,
        })?;
        // if let Some(ctors) = self.sorts.get_mut(&sort) {
        //     ctors.push(name);
        // }
        Ok(())
    }

    pub fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(i) => i.store(&self.get_sort()).unwrap(),
            Literal::String(s) => s.store(&self.get_sort()).unwrap(),
            Literal::Unit => ().store(&self.get_sort()).unwrap(),
        }
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
            Expr::Lit(lit) => Ok(self.eval_lit(lit)),
            Expr::Call(op, args) => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.eval_expr(ctx, arg)?);
                }
                if let Some(function) = self.functions.get_mut(op) {
                    if let Some(value) = function.get(&values) {
                        Ok(*value)
                    } else {
                        let out = &function.schema.output;
                        match function.decl.default.as_ref() {
                            None if out.name() == "Unit".into() => {
                                function.insert_value(
                                    &mut self.unionfind,
                                    values.into(),
                                    Value::unit(),
                                );
                                Ok(Value::unit())
                            }
                            None if out.is_eq_sort() => {
                                let id = self.unionfind.make_set();
                                let value = Value::from_id(out.name(), id);
                                function.insert_value(&mut self.unionfind, values.into(), value);
                                Ok(value)
                            }
                            Some(default) => {
                                let default = default.clone(); // break the borrow
                                let value = self.eval_expr(ctx, &default)?;
                                let function = self.functions.get_mut(op).unwrap();
                                function.insert_value(&mut self.unionfind, values.into(), value);
                                Ok(value)
                            }
                            _ => panic!("invalid default for {:?}", function.decl.name),
                        }
                    }
                } else if let Some(prims) = self.primitives.get(op) {
                    let mut res = None;
                    for prim in prims.iter() {
                        // HACK
                        let types = values
                            .iter()
                            .map(|v| &*self.sorts[&v.tag])
                            .collect::<Vec<_>>();
                        if prim.accept(&types).is_some() {
                            if res.is_none() {
                                res = prim.apply(&values);
                            } else {
                                panic!("multiple implementation matches primitives {op}");
                            }
                        }
                    }
                    res.ok_or_else(|| NotFoundError(expr.clone()))
                } else {
                    panic!("Couldn't find function/primitive: {op}")
                }
            }
        }
    }

    pub fn eval_closed_expr(&mut self, expr: &Expr) -> Result<Value, NotFoundError> {
        self.eval_expr(&Default::default(), expr)
    }

    pub fn run_rules(&mut self, limit: usize) -> [Duration; 3] {
        let mut search_time = Duration::default();
        let mut apply_time = Duration::default();
        let mut rebuild_time = Duration::default();
        for _ in 0..limit {
            let [st, at] = self.step_rules();
            search_time += st;
            apply_time += at;

            let rebuild_start = Instant::now();
            let updates = self.rebuild();
            log::debug!("Made {updates} updates",);
            rebuild_time += rebuild_start.elapsed();
            // if updates == 0 {
            //     log::debug!("Breaking early!");
            //     break;
            // }
        }

        // TODO detect functions
        for (name, r) in &self.functions {
            log::debug!("{name}:");
            for (args, val) in &r.data.nodes {
                log::debug!("  {args:?} = {val:?}");
            }
        }
        [search_time, apply_time, rebuild_time]
    }

    fn step_rules(&mut self) -> [Duration; 2] {
        fn make_subst(rule: &Rule, values: &[Value]) -> Subst {
            let get_val = |t: &AtomTerm| match t {
                AtomTerm::Var(sym) => {
                    let i = rule
                        .query
                        .vars
                        .get_index_of(sym)
                        .unwrap_or_else(|| panic!("Couldn't find variable '{sym}'"));
                    values[i]
                }
                AtomTerm::Value(val) => *val,
            };

            rule.bindings
                .iter()
                .map(|(k, t)| (*k, get_val(t)))
                .collect()
        }

        let search_start = Instant::now();
        let mut searched = vec![];
        for rule in self.rules.values() {
            let mut all_values = vec![];
            self.run_query(&rule.query, |values| {
                assert_eq!(values.len(), rule.query.vars.len());
                all_values.extend_from_slice(values);
            });
            searched.push(all_values);
        }
        let search_elapsed = search_start.elapsed();

        let apply_start = Instant::now();
        let rules = std::mem::take(&mut self.rules);
        for (rule, all_values) in rules.values().zip(searched) {
            let n = rule.query.vars.len();
            for values in all_values.chunks(n) {
                let subst = make_subst(rule, values);
                let _result: Result<_, _> = self.eval_actions(Some(subst), &rule.head);
            }
        }
        self.rules = rules;
        let apply_elapsed = apply_start.elapsed();
        [search_elapsed, apply_elapsed]
    }

    fn add_rule_with_name(&mut self, name: String, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let mut ctx = typecheck::Context::new(self);
        let (atoms, bindings) = ctx.typecheck_query(&rule.body).map_err(Error::TypeErrors)?;
        let query = self.compile_gj_query(atoms, ctx.types);
        let compiled_rule = Rule {
            query,
            bindings,
            head: rule.head,
        };
        match self.rules.entry(name) {
            Entry::Occupied(_) => panic!("Rule '{name}' was already present"),
            Entry::Vacant(e) => e.insert(compiled_rule),
        };
        Ok(name)
    }

    pub fn add_rule(&mut self, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = format!("{}", rule);
        self.add_rule_with_name(name, rule)
    }

    pub fn clear_rules(&mut self) {
        self.rules = Default::default();
    }

    pub fn add_rewrite(&mut self, rewrite: ast::Rewrite) -> Result<Symbol, Error> {
        let name = format!("{} -> {}", rewrite.lhs, rewrite.rhs);
        let var = Symbol::from("__rewrite_var");
        let rule = ast::Rule {
            body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs])]
                .into_iter()
                .chain(rewrite.conditions)
                .collect(),
            head: vec![Action::Union(Expr::Var(var), rewrite.rhs)],
        };
        self.add_rule_with_name(name, rule)
    }

    fn for_each_canonicalized(&self, name: Symbol, mut cb: impl FnMut(&[Value])) {
        let mut ids = vec![];
        let f = self
            .functions
            .get(&name)
            .unwrap_or_else(|| panic!("No function {name}"));
        for (children, value) in &f.data.nodes {
            ids.clear();
            // FIXME canonicalize, do we need to with rebuilding?
            // ids.extend(children.iter().map(|id| self.find(value)));
            ids.extend(children.iter().cloned());
            ids.push(*value);
            cb(&ids);
        }
    }

    fn run_command(&mut self, command: Command, should_run: bool) -> Result<String, Error> {
        Ok(match command {
            Command::Datatype { name, variants } => {
                self.declare_sort(name)?;
                for variant in variants {
                    self.declare_constructor(variant.name, variant.types, name)?;
                }
                format!("Declared datatype {name}.")
            }
            Command::Sort(name, presort, args) => {
                // TODO extract this into a function
                assert!(!self.sorts.contains_key(&name));
                let mksort = self.presorts[&presort];
                let sort = mksort(self, name, &args)?;
                self.add_arcsort(sort);
                format!(
                    "Declared sort {name} = ({presort} {})",
                    ListDisplay(&args, " ")
                )
            }
            Command::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                format!("Declared function {}.", fdecl.name)
            }
            Command::Rule(rule) => {
                let name = self.add_rule(rule)?;
                format!("Declared rule {name}.")
            }
            Command::Rewrite(rewrite) => {
                let name = self.add_rewrite(rewrite)?;
                format!("Declared rw {name}.")
            }
            Command::Run(limit) => {
                if should_run {
                    let [st, at, rt] = self.run_rules(limit);
                    let st = st.as_secs_f64();
                    let at = at.as_secs_f64();
                    let rt = rt.as_secs_f64();
                    format!(
                        "Ran {limit} in {total:10.6}s.\n\
                        Search:  {st:10.6}s\n\
                        Apply:   {at:10.6}s\n\
                        Rebuild: {rt:10.6}s",
                        total = st + at + rt,
                    )
                } else {
                    log::info!("Skipping running!");
                    format!("Skipped run {limit}.")
                }
            }
            Command::Extract { e, variants } => {
                if should_run {
                    // TODO typecheck
                    self.rebuild();
                    let value = self.eval_closed_expr(&e)?;
                    log::info!("Extracting {e} at {value:?}");
                    let (cost, expr) = self.extract(value);
                    let mut msg = format!("Extracted with cost {cost}: {expr}");
                    if variants > 0 {
                        let exprs = self.extract_variants(value, variants);
                        let line = "\n    ";
                        let v_exprs = ListDisplay(&exprs, line);
                        write!(msg, "\nVariants of {expr}:{line}{v_exprs}").unwrap();
                    }
                    msg
                } else {
                    "Skipping extraction.".into()
                }
            }
            Command::Check(fact) => {
                if should_run {
                    self.check_with(&Default::default(), &fact)?;
                    "Checked.".into()
                } else {
                    "Skipping check.".into()
                }
            }
            Command::Action(action) => {
                if should_run {
                    self.eval_actions(None, std::slice::from_ref(&action))?;
                    format!("Run {action}.")
                } else {
                    format!("Skipping running {action}.")
                }
            }
            Command::Define(name, expr) => {
                if should_run {
                    let value = self.eval_closed_expr(&expr)?;
                    let old = self.globals.insert(name, value);
                    assert!(old.is_none());
                    format!("Defined {name}")
                } else {
                    format!("Skipping define {name}")
                }
            }
            Command::ClearRules => {
                self.clear_rules();
                "Clearing rules.".into()
            }
            Command::Query(_q) => {
                // let qsexp = sexp::Sexp::List(
                //     q.iter()
                //         .map(|fact| sexp::parse(&fact.to_string()).unwrap())
                //         .collect(),
                // );
                // let qcomp = self
                //     .compile_query(q)
                //     .unwrap_or_else(|_| panic!("Could not compile query"));
                // let mut res = vec![];
                // self.query(&qcomp, |v| {
                //     res.push(sexp::Sexp::List(
                //         v.iter()
                //             .map(|val| sexp::Sexp::Atom(sexp::Atom::S(format!("{}", val))))
                //             .collect(),
                //     ));
                // });
                // format!(
                //     "Query: {}\n  Bindings: {:?}\n  Results: {}",
                //     qsexp,
                //     qcomp,
                //     sexp::Sexp::List(res)
                // )
                todo!()
            }
            Command::Clear => {
                for f in self.functions.values_mut() {
                    f.clear();
                }
                "Cleared.".into()
            }
            Command::Push(n) => {
                (0..n).for_each(|_| self.push());
                format!("Pushed {n} levels.")
            }
            Command::Pop(n) => {
                for _ in 0..n {
                    self.pop()?;
                }
                format!("Popped {n} levels.")
            }
        })
    }

    fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<String>, Error> {
        let mut msgs = vec![];
        let should_run = true;

        for command in program {
            let msg = self.run_command(command, should_run)?;
            log::info!("{}", msg);
            msgs.push(msg);
        }

        Ok(msgs)
    }

    // this is bad because we shouldn't inspect values like this, we should use type information
    fn bad_find_value(&self, value: Value) -> Value {
        if let Some((tag, id)) = self.value_to_id(value) {
            Value::from_id(tag, self.find(id))
        } else {
            value
        }
    }

    pub fn parse_and_run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let parser = ast::parse::ProgramParser::new();
        let program = parser
            .parse(input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?;
        self.run_program(program)
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] lalrpop_util::ParseError<usize, String, String>),
    #[error(transparent)]
    NotFoundError(#[from] NotFoundError),
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error("Errors:\n{}", ListDisplay(.0, "\n"))]
    TypeErrors(Vec<TypeError>),
    #[error("Check failed: {0:?} != {1:?}")]
    CheckError(Value, Value),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
    #[error("Tried to pop too much")]
    Pop,
}
