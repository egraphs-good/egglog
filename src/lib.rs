pub mod ast;
mod extract;
mod gj;
pub mod sort;
mod typecheck;
mod unionfind;
mod util;
mod value;

use hashbrown::hash_map::Entry;
use indexmap::map::Entry as IEntry;
use instant::{Duration, Instant};
use smallvec::SmallVec;
use sort::*;
use thiserror::Error;

use ast::*;

use std::fmt::Write;
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::rc::Rc;
use std::{fmt::Debug, sync::Arc};
use typecheck::Program;

type ArcSort = Arc<dyn Sort>;

pub use value::*;

use gj::*;

use unionfind::*;
use util::*;

use crate::typecheck::TypeError;

type ValueVec = SmallVec<[Value; 3]>;

#[derive(Clone)]
pub struct Function {
    decl: FunctionDecl,
    schema: ResolvedSchema,
    merge: MergeFn,
    nodes: IndexMap<ValueVec, TupleOutput>,
    updates: usize,
}

#[derive(Clone)]
enum MergeFn {
    AssertEq,
    Union,
    // the rc is make sure it's cheaply clonable, since calling the merge fn
    // requires a clone
    Expr(Rc<Program>),
}

#[derive(Debug, Clone)]
struct TupleOutput {
    value: Value,
    timestamp: u32,
}

#[derive(Clone, Debug)]
struct ResolvedSchema {
    input: Vec<ArcSort>,
    output: ArcSort,
}

impl Function {
    pub fn insert(&mut self, inputs: ValueVec, value: Value, timestamp: u32) -> Option<Value> {
        match self.nodes.entry(inputs) {
            IEntry::Occupied(mut entry) => {
                let old = entry.get_mut();
                if old.value == value {
                    Some(value)
                } else {
                    let saved = old.value;
                    old.value = value;
                    assert!(old.timestamp <= timestamp);
                    old.timestamp = timestamp;
                    self.updates += 1;
                    debug_assert_ne!(saved, value);
                    Some(saved)
                }
            }
            IEntry::Vacant(entry) => {
                entry.insert(TupleOutput { value, timestamp });
                self.updates += 1;
                None
            }
        }
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind, timestamp: u32) -> usize {
        if self.schema.input.iter().all(|s| !s.is_eq_sort()) && !self.schema.output.is_eq_sort() {
            return std::mem::take(&mut self.updates);
        }

        // FIXME this doesn't compute updates properly
        let n_unions = uf.n_unions();
        let mut to_add = vec![];
        self.nodes.retain(|args, out| {
            assert!(out.timestamp <= timestamp);
            let mut new_args = args.clone();
            let mut modified = false;
            for (a, ty) in new_args.iter_mut().zip(&self.schema.input) {
                if ty.is_eq_sort() {
                    let new_a = uf.find_mut_value(*a);
                    if new_a != *a {
                        *a = new_a;
                        modified = true;
                    }
                }
            }
            if self.schema.output.is_eq_sort() {
                let new_value = uf.find_mut_value(out.value);
                if out.value != new_value {
                    modified = true;
                }
            }

            if modified {
                to_add.push((new_args, out.clone()));
            }
            !modified
        });

        for (args, out) in to_add {
            let value = out.value;
            // TODO call the merge fn!!!
            let _new_value = if self.schema.output.is_eq_sort() {
                self.nodes
                    .entry(args)
                    .and_modify(|out2| {
                        let new_value = uf.union_values(value, out2.value);
                        if out2.value != new_value {
                            out2.value = new_value;
                            out2.timestamp = timestamp;
                        }
                        assert!(out2.timestamp <= timestamp);
                    })
                    .or_insert_with(|| {
                        let new_value = uf.find_mut_value(value);
                        TupleOutput {
                            value: new_value,
                            timestamp,
                        }
                    })
            } else {
                self.nodes
                    .entry(args)
                    .and_modify(|out2| {
                        // out2.value = uf.union_values(value, out2.value);
                        out2.timestamp = out2.timestamp.max(out.timestamp);
                    })
                    .or_insert(TupleOutput { value, timestamp })
            };
        }

        // if cfg!(debug_assertions) {
        //     let mut seen = 0;
        //     for out in self.nodes.values() {
        //         assert!(seen <= out.timestamp, "Timestamps should be ordered");
        //         seen = out.timestamp;
        //     }
        // }

        uf.n_unions() - n_unions + std::mem::take(&mut self.updates)
    }

    pub(crate) fn get_size(&self, range: &Range<u32>) -> usize {
        self.nodes
            .values()
            .filter(|out| range.contains(&out.timestamp))
            .count()
        // if range.start == 0 {
        //     if range.end == u32::MAX {
        //         self.nodes.len()
        //     } else {
        //         // TODO binary search or something
        //         self.nodes
        //             .values()
        //             .filter(|out| out.timestamp < range.end)
        //             .count()
        //     }
        // } else {
        //     assert_eq!(range.end, u32::MAX);
        //     self.nodes
        //         .values()
        //         .filter(|out| out.timestamp >= range.end)
        //         .count()
        // }
    }
}

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort>;
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
    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
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
    saturated: bool,
    timestamp: u32,
    pub match_limit: usize,
    pub fact_directory: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct Rule {
    query: CompiledQuery,
    program: Program,
    matches: usize,
    times_banned: usize,
    banned_until: usize,
    todo_timestamp: u32,
    search_time: Duration,
    apply_time: Duration,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            egraphs: vec![],
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            primitives: Default::default(),
            presorts: Default::default(),
            match_limit: 10_000_000,
            timestamp: 0,
            saturated: false,
            fact_directory: None,
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
            for (inputs, output) in function.nodes.iter() {
                for input in inputs {
                    assert_eq!(
                        input,
                        &self.bad_find_value(*input),
                        "{name}({inputs:?}) = {output:?}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output.value,
                    self.bad_find_value(output.value),
                    "{name}({inputs:?}) = {output:?}\n{:?}",
                    function.schema,
                )
            }
        }
    }

    pub fn check_fact(&mut self, fact: &Fact) -> Result<(), Error> {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() > 1);
                let values: Vec<(ArcSort, Value)> = exprs
                    .iter()
                    .map(|e| self.eval_expr(e, None, false))
                    .collect::<Result<_, _>>()?;

                let (_t0, v0) = &values[0];
                for (_t, v) in &values[1..] {
                    if v0 != v {
                        log::error!("Check failed");
                        // the check failed, so print out some useful info
                        self.rebuild();
                        for (_t, value) in &values {
                            if let Some((_tag, id)) = self.value_to_id(*value) {
                                let best = self.extract(*value).1;
                                log::error!("{}: {}", id, best);
                            }
                        }
                        return Err(Error::CheckError(values[0].1, *v));
                    }
                }
            }
            Fact::Fact(expr) => match expr {
                Expr::Lit(_) => panic!("can't check a literal"),
                Expr::Var(_) => panic!("can't check a var"),
                Expr::Call(_, _) => {
                    // println!("Checking fact: {}", expr);
                    let unit = self.get_sort::<UnitSort>();
                    self.eval_expr(expr, Some(unit), true)?;
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
        let mut new_unions = 0;
        for function in self.functions.values_mut() {
            new_unions += function.rebuild(&mut self.unionfind, self.timestamp);
        }
        new_unions
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

        let merge = if let Some(merge_expr) = &decl.merge {
            let mut types = IndexMap::<Symbol, ArcSort>::default();
            types.insert("old".into(), output.clone());
            types.insert("new".into(), output.clone());
            let (_, program) = self
                .compile_expr(&types, merge_expr, Some(output.clone()))
                .map_err(Error::TypeErrors)?;
            MergeFn::Expr(Rc::new(program))
        } else if output.is_eq_sort() {
            MergeFn::Union
        } else {
            MergeFn::AssertEq
        };

        let function = Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            nodes: Default::default(),
            updates: 0,
            merge,
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
        variant: Variant,
        sort: impl Into<Symbol>,
    ) -> Result<(), Error> {
        let name = variant.name;
        let sort = sort.into();
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: variant.types,
                output: sort,
            },
            merge: None,
            default: None,
            cost: variant.cost,
        })?;
        // if let Some(ctors) = self.sorts.get_mut(&sort) {
        //     ctors.push(name);
        // }
        Ok(())
    }

    pub fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(i) => i.store(&*self.get_sort()).unwrap(),
            Literal::String(s) => s.store(&*self.get_sort()).unwrap(),
            Literal::Unit => ().store(&*self.get_sort()).unwrap(),
        }
    }

    fn print_function(&mut self, sym: Symbol, n: usize) -> Result<String, Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        let schema = f.schema.clone();
        let nodes = f
            .nodes
            .iter()
            .take(n)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();

        let out_is_unit = f.schema.output.name() == "Unit".into();

        let mut buf = String::new();
        let s = &mut buf;
        for (ins, out) in nodes {
            write!(s, "({}", sym).unwrap();
            for (a, t) in ins.iter().zip(&schema.input) {
                s.push(' ');
                let e = if t.is_eq_sort() {
                    self.extract(*a).1
                } else {
                    t.make_expr(*a)
                };
                write!(s, "{}", e).unwrap();
            }

            if out_is_unit {
                s.push(')');
            } else {
                let e = if schema.output.is_eq_sort() {
                    self.extract(out.value).1
                } else {
                    schema.output.make_expr(out.value)
                };
                write!(s, ") -> {}", e).unwrap();
            }
            s.push('\n');
            // write!(s, "{}(", self.decl.name)?;
            // for (i, arg) in args.iter().enumerate() {
            //     if i > 0 {
            //         write!(s, ", ")?;
            //     }
            //     write!(s, "{}", arg)?;
            // }
            // write!(s, ") = {}", value)?;
            // println!("{}", s);
        }

        Ok(buf)
    }

    pub fn print_size(&self, sym: Symbol) -> Result<String, Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        Ok(format!("Function {} has size {}", sym, f.nodes.len()))
    }

    pub fn run_rules(&mut self, limit: usize) -> [Duration; 3] {
        let mut search_time = Duration::default();
        let mut apply_time = Duration::default();

        // we might have to do a rebuild before starting,
        // because the use can manually do stuff
        let initial_rebuild_start = Instant::now();
        self.rebuild();
        let mut rebuild_time = initial_rebuild_start.elapsed();

        for i in 0..limit {
            self.saturated = true;
            let [st, at] = self.step_rules(i);
            search_time += st;
            apply_time += at;

            let rebuild_start = Instant::now();
            let updates = self.rebuild();
            log::info!("database size: {}", self.num_tuples());
            log::info!("Made {updates} updates",);
            rebuild_time += rebuild_start.elapsed();
            self.timestamp += 1;
            if self.saturated {
                log::info!("Breaking early at iteration {}!", i);
                break;
            }
        }

        // Report the worst offenders
        log::debug!("Slowest rules:\n{}", {
            let mut msg = String::new();
            let mut vec = self.rules.iter().collect::<Vec<_>>();
            vec.sort_by_key(|(_, r)| r.search_time + r.apply_time);
            for (name, rule) in vec.iter().rev().take(5) {
                write!(
                    msg,
                    "{name}\n  Search: {}\n  Apply: {}\n",
                    rule.search_time.as_secs_f64(),
                    rule.apply_time.as_secs_f64()
                )
                .unwrap();
            }
            msg
        });

        // // TODO detect functions
        // for (name, r) in &self.functions {
        //     log::debug!("{name}:");
        //     for (args, val) in &r.nodes {
        //         log::debug!("  {args:?} = {val:?}");
        //     }
        // }
        [search_time, apply_time, rebuild_time]
    }

    fn step_rules(&mut self, iteration: usize) -> [Duration; 2] {
        // fn make_subst(rule: &Rule, values: &[Value]) -> Subst {
        //     let get_val = |t: &AtomTerm| match t {
        //         AtomTerm::Var(sym) => {
        //             let i = rule
        //                 .query
        //                 .vars
        //                 .get_index_of(sym)
        //                 .unwrap_or_else(|| panic!("Couldn't find variable '{sym}'"));
        //             values[i]
        //         }
        //         AtomTerm::Value(val) => *val,
        //     };

        //     todo!()
        //     // rule.bindings
        //     //     .iter()
        //     //     .map(|(k, t)| (*k, get_val(t)))
        //     //     .collect()
        // }

        let ban_length = 5;

        let mut rules = std::mem::take(&mut self.rules);
        let search_start = Instant::now();
        let mut searched = vec![];
        for (&name, rule) in rules.iter_mut() {
            let mut all_values = vec![];
            if rule.banned_until <= iteration {
                let rule_search_start = Instant::now();
                self.run_query(&rule.query, rule.todo_timestamp, |values| {
                    assert_eq!(values.len(), rule.query.vars.len());
                    all_values.extend_from_slice(values);
                });
                rule.todo_timestamp = self.timestamp;
                let rule_search_time = rule_search_start.elapsed();
                log::trace!(
                    "Searched for {name} in {} ({} results)",
                    rule_search_time.as_secs_f64(),
                    all_values.len()
                );
                rule.search_time += rule_search_time;
                searched.push((name, all_values));
            }
        }
        let search_elapsed = search_start.elapsed();

        let apply_start = Instant::now();
        'outer: for (name, all_values) in searched {
            let rule = rules.get_mut(&name).unwrap();
            let n = rule.query.vars.len();

            // backoff logic
            let len = all_values.len() / n;
            let threshold = self.match_limit << rule.times_banned;
            if len > threshold {
                rule.times_banned += 1;
                rule.banned_until = iteration + (ban_length << rule.times_banned);
                log::info!("Banning rule {name} for {ban_length} iterations, matched {len} times");
                continue;
            }

            let rule_apply_start = Instant::now();

            let stack = &mut vec![];
            for values in all_values.chunks(n) {
                rule.matches += 1;
                if rule.matches > 10_000_000 {
                    log::warn!("Rule {} has matched {} times, bailing!", name, rule.matches);
                    break 'outer;
                }
                // we can ignore results here
                stack.clear();
                let _ = self.run_actions(stack, values, &rule.program, true);
            }

            rule.apply_time += rule_apply_start.elapsed();
        }
        self.rules = rules;
        let apply_elapsed = apply_start.elapsed();
        [search_elapsed, apply_elapsed]
    }

    fn add_rule_with_name(&mut self, name: String, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let mut ctx = typecheck::Context::new(self);
        let query0 = ctx.typecheck_query(&rule.body).map_err(Error::TypeErrors)?;
        let query = self.compile_gj_query(query0, &ctx.types);
        let program = self
            .compile_actions(&ctx.types, &rule.head)
            .map_err(Error::TypeErrors)?;
        // println!(
        //     "Compiled rule {rule:?}\n{subst:?}to {program:#?}",
        //     subst = &ctx.types
        // );
        let compiled_rule = Rule {
            query,
            matches: 0,
            times_banned: 0,
            banned_until: 0,
            todo_timestamp: 0,
            program,
            search_time: Duration::default(),
            apply_time: Duration::default(),
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
        let mut name = format!("{} -> {}", rewrite.lhs, rewrite.rhs);
        if !rewrite.conditions.is_empty() {
            write!(name, " if {}", ListDisplay(&rewrite.conditions, ", ")).unwrap();
        }
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

    fn eval_actions(&mut self, actions: &[Action]) -> Result<(), Error> {
        let types = Default::default();
        let program = self
            .compile_actions(&types, actions)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, true)?;
        Ok(())
    }

    fn eval_expr(
        &mut self,
        expr: &Expr,
        expected_type: Option<ArcSort>,
        make_defaults: bool,
    ) -> Result<(ArcSort, Value), Error> {
        let types = Default::default();
        let (t, program) = self
            .compile_expr(&types, expr, expected_type)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, make_defaults)?;
        assert_eq!(stack.len(), 1);
        Ok((t, stack.pop().unwrap()))
    }

    fn run_command(&mut self, command: Command, should_run: bool) -> Result<String, Error> {
        Ok(match command {
            Command::Datatype { name, variants } => {
                self.declare_sort(name)?;
                for variant in variants {
                    self.declare_constructor(variant, name)?;
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
                    let total = st + at + rt;
                    let size = self.num_tuples();
                    format!(
                        "Ran {limit} in {total:10.6}s.\n\
                        Search:  ({:.02}) {st:10.6}s\n\
                        Apply:   ({:.02}) {at:10.6}s\n\
                        Rebuild: ({:.02}) {rt:10.6}s\n\
                        Database size: {size}",
                        st / total,
                        at / total,
                        rt / total,
                    )
                } else {
                    log::info!("Skipping running!");
                    format!("Skipped run {limit}.")
                }
            }
            Command::Extract { e, variants } => {
                if should_run {
                    // TODO typecheck
                    let (cost, expr, exprs) = self.extract_expr(e, variants)?;
                    let mut msg = format!("Extracted with cost {cost}: {expr}");
                    if variants > 0 {
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
                    self.check_fact(&fact)?;
                    "Checked.".into()
                } else {
                    "Skipping check.".into()
                }
            }
            Command::Action(action) => {
                if should_run {
                    self.eval_actions(std::slice::from_ref(&action))?;
                    format!("Run {action}.")
                } else {
                    format!("Skipping running {action}.")
                }
            }
            Command::Define { name, expr, cost } => {
                if should_run {
                    let sort = self.define(name, expr, cost)?;
                    format!("Defined {name}: {sort:?}")
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
                    f.nodes.clear();
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
            Command::Print(f, n) => {
                let msg = self.print_function(f, n)?;
                println!("{}", msg);
                msg
            }
            Command::PrintSize(f) => {
                let msg = self.print_size(f)?;
                println!("{}", msg);
                msg
            }
            Command::Input { name, file } => {
                let func = self.functions.get_mut(&name).unwrap();
                let is_unit = func.schema.output.name().as_str() == "Unit";

                let mut filename = self.fact_directory.clone().unwrap_or_default();
                filename.push(file.as_str());

                // check that the function uses supported types
                for t in &func.schema.input {
                    match t.name().as_str() {
                        "i64" | "String" => {}
                        s => panic!("Unsupported type {} for input", s),
                    }
                }
                match func.schema.output.name().as_str() {
                    "i64" | "String" | "Unit" => {}
                    s => panic!("Unsupported type {} for input", s),
                }

                log::info!("Opening file '{:?}'...", filename);
                let mut f = File::open(filename).unwrap();
                let mut contents = String::new();
                f.read_to_string(&mut contents).unwrap();

                let mut actions: Vec<Action> = vec![];
                let mut str_buf: Vec<&str> = vec![];
                for line in contents.lines() {
                    str_buf.clear();
                    str_buf.extend(line.split('\t').map(|s| s.trim()));
                    if str_buf.is_empty() {
                        continue;
                    }

                    let parse = |s: &str| -> Expr {
                        if let Ok(i) = s.parse() {
                            Expr::Lit(Literal::Int(i))
                        } else {
                            Expr::Lit(Literal::String(s.into()))
                        }
                    };

                    let mut exprs: Vec<Expr> = str_buf.iter().map(|&s| parse(s)).collect();

                    actions.push(if is_unit {
                        Action::Expr(Expr::Call(name, exprs))
                    } else {
                        let out = exprs.pop().unwrap();
                        Action::Set(name, exprs, out)
                    });
                }
                self.eval_actions(&actions)?;
                format!("Read {} facts into {name} from '{file}'.", actions.len())
            }
        })
    }

    // Extract an expression from the current state, returning the cost, the extracted expression and some number
    // of other variants, if variants is not zero.
    pub fn extract_expr(&mut self, e: Expr, variants: usize) -> Result<(usize, Expr, Vec<Expr>), Error> {
        self.rebuild();
        let (_t, value) = self.eval_expr(&e, None, true)?;
        let (cost, expr) = self.extract(value);
        let exprs = if variants > 0 {
            self.extract_variants(value, variants)
        } else {
            vec![]
        };
        Ok((cost, expr, exprs))
    }

    pub fn define(
        &mut self,
        name: Symbol,
        expr: Expr,
        cost: Option<usize>,
    ) -> Result<ArcSort, Error> {
        let (sort, value) = self.eval_expr(&expr, None, true)?;
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: vec![],
                output: value.tag,
            },
            default: None,
            merge: None,
            cost,
        })?;
        let f = self.functions.get_mut(&name).unwrap();
        f.insert(ValueVec::default(), value, self.timestamp);
        Ok(sort)
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

    pub fn num_tuples(&self) -> usize {
        self.functions.values().map(|f| f.nodes.len()).sum()
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
