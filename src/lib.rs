pub mod ast;
mod desugar;
mod extract;
mod function;
mod gj;
pub mod sort;
mod typecheck;
mod unionfind;
pub mod util;
mod value;

use hashbrown::hash_map::Entry;
use index::ColumnIndex;
use instant::{Duration, Instant};
use sort::*;
use thiserror::Error;

use desugar::desugar_program;

use symbolic_expressions::Sexp;

use ast::*;

use std::fmt::{Formatter, Write};
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::iter::once;
use std::mem;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::rc::Rc;
use std::{fmt::Debug, sync::Arc};
use typecheck::Program;

type ArcSort = Arc<dyn Sort>;

pub use value::*;

use function::*;
use gj::*;
use unionfind::*;
use util::*;

use crate::typecheck::TypeError;

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
    rulesets: IndexMap<Symbol, Vec<(Symbol, Rule)>>,
    saturated: bool,
    timestamp: u32,
    unit_sym: Symbol,
    pub match_limit: usize,
    pub node_limit: usize,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
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
        let unit_sym = "Unit".into();
        let mut egraph = Self {
            egraphs: vec![],
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            rulesets: Default::default(),
            primitives: Default::default(),
            presorts: Default::default(),
            unit_sym,
            match_limit: 10_000_000,
            node_limit: 100_000_000,
            timestamp: 0,
            saturated: false,
            fact_directory: None,
            seminaive: true,
        };
        egraph.add_sort(UnitSort::new(unit_sym));
        egraph.add_sort(StringSort::new("String".into()));
        egraph.add_sort(I64Sort::new("i64".into()));
        egraph.add_sort(F64Sort::new("f64".into()));
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
        self.add_arcsort(Arc::new(sort)).unwrap()
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

    pub fn add_arcsort(&mut self, sort: ArcSort) -> Result<(), Error> {
        let name = sort.name();
        match self.sorts.entry(name) {
            Entry::Occupied(_) => Err(Error::SortAlreadyBound(name)),
            Entry::Vacant(e) => {
                e.insert(sort.clone());
                sort.register_primitives(self);
                Ok(())
            }
        }
    }

    pub fn get_sort<S: Sort + Send + Sync>(&self) -> Arc<S> {
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

    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        let prim = prim.into();
        self.primitives.entry(prim.name()).or_default().push(prim);
    }

    pub fn union(&mut self, id1: Id, id2: Id, sort: Symbol) -> Id {
        self.unionfind.union(id1, id2, sort)
    }

    #[track_caller]
    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        for (name, function) in self.functions.iter() {
            function.nodes.assert_sorted();
            for (i, inputs, output) in function.nodes.iter_range(0..function.nodes.len()) {
                for input in inputs {
                    assert_eq!(
                        input,
                        &self.bad_find_value(*input),
                        "[{i}] {name}({inputs:?}) = {output:?}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output.value,
                    self.bad_find_value(output.value),
                    "[{i}] {name}({inputs:?}) = {output:?}\n{:?}",
                    function.schema,
                )
            }
            for ix in &function.indexes {
                for (_, offs) in ix.iter() {
                    for off in offs {
                        assert!(
                            (*off as usize) < function.nodes.len(),
                            "index contains offset {off:?}, which is out of range for function {name}"
                        );
                    }
                }
            }
        }
    }

    pub fn check_fact(&mut self, fact: &Fact, log: bool) -> Result<(), Error> {
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
                        if log {
                            log::error!("Check failed");
                            // the check failed, so print out some useful info
                            self.rebuild()?;
                            for (_t, value) in &values {
                                if let Some((_tag, id)) = self.value_to_id(*value) {
                                    let best = self.extract(*value).1;
                                    log::error!("{}: {}", id, best);
                                }
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
                    self.eval_expr(expr, Some(unit), false)?;
                }
            },
        }
        Ok(())
    }

    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    pub fn rebuild_nofail(&mut self) -> usize {
        match self.rebuild() {
            Ok(updates) => updates,
            Err(e) => {
                panic!("Unsoundness detected during rebuild. Exiting: {e}")
            }
        }
    }

    pub fn rebuild(&mut self) -> Result<usize, Error> {
        self.unionfind.clear_recent_ids();
        let mut updates = 0;
        loop {
            let new = self.rebuild_one()?;
            log::debug!("{new} rebuilds?");
            self.unionfind.clear_recent_ids();
            updates += new;
            if new == 0 {
                break;
            }
        }
        self.debug_assert_invariants();
        Ok(updates)
    }

    fn rebuild_one(&mut self) -> Result<usize, Error> {
        let mut new_unions = 0;
        let mut deferred_merges = Vec::new();
        for function in self.functions.values_mut() {
            let (unions, merges) = function.rebuild(&mut self.unionfind, self.timestamp)?;
            if !merges.is_empty() {
                deferred_merges.push((function.decl.name, merges));
            }
            new_unions += unions;
        }
        for (func, merges) in deferred_merges {
            new_unions += self.apply_merges(func, &merges);
        }
        Ok(new_unions)
    }

    fn apply_merges(&mut self, func: Symbol, merges: &[DeferredMerge]) -> usize {
        let mut stack = Vec::new();
        let mut function = self.functions.get_mut(&func).unwrap();
        let n_unions = self.unionfind.n_unions();
        let merge_prog = match &function.merge.merge_vals {
            MergeFn::Expr(e) => Some(e.clone()),
            MergeFn::AssertEq | MergeFn::Union => None,
        };
        for (inputs, old, new) in merges {
            if let Some(prog) = function.merge.on_merge.clone() {
                self.run_actions(&mut stack, &[*old, *new], &prog, true)
                    .unwrap();
                function = self.functions.get_mut(&func).unwrap();
                stack.clear();
            }
            if let Some(prog) = &merge_prog {
                // TODO: error handling?
                self.run_actions(&mut stack, &[*old, *new], prog, true)
                    .unwrap();
                let merged = stack.pop().expect("merges should produce a value");
                stack.clear();
                function = self.functions.get_mut(&func).unwrap();
                function.insert(inputs, merged, self.timestamp);
            }
        }
        self.unionfind.n_unions() - n_unions + function.clear_updates()
    }

    pub fn declare_sort(
        &mut self,
        name: impl Into<Symbol>,
        presort_and_args: Option<(Symbol, &[Expr])>,
    ) -> Result<(), Error> {
        let name = name.into();
        let sort = match presort_and_args {
            Some((presort, args)) => {
                let mksort = self
                    .presorts
                    .get(&presort)
                    .ok_or(Error::PresortNotFound(presort))?;
                mksort(self, name, args)?
            }
            None => Arc::new(EqSort { name }),
        };
        self.add_arcsort(sort)
    }

    pub fn declare_function(&mut self, decl: &FunctionDecl) -> Result<(), Error> {
        let function = Function::new(self, decl)?;
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
            merge_action: vec![],
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
            Literal::Int(i) => i.store(&self.get_sort()).unwrap(),
            Literal::F64(f) => f.store(&self.get_sort()).unwrap(),
            Literal::String(s) => s.store(&self.get_sort()).unwrap(),
            Literal::Unit => ().store(&self.get_sort()).unwrap(),
        }
    }

    pub fn print_function(&mut self, sym: Symbol, n: usize) -> Result<String, Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        let schema = f.schema.clone();
        let nodes = f
            .nodes
            .iter()
            .take(n)
            .map(|(k, v)| (ValueVec::from(k), v.clone()))
            .collect::<Vec<_>>();

        let out_is_unit = f.schema.output.name() == self.unit_sym;

        let mut buf = String::new();
        let s = &mut buf;
        for (ins, out) in nodes {
            write!(s, "({}", sym).unwrap();
            for (a, t) in ins.iter().copied().zip(&schema.input) {
                s.push(' ');
                let e = if t.is_eq_sort() {
                    self.extract(a).1
                } else {
                    t.make_expr(a)
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

    // returns whether the egraph was updated
    pub fn run_schedule(&mut self, sched: &Schedule) -> bool {
        match sched {
            Schedule::Ruleset(ruleset_name) => {

                self.saturated = true;

                let mut rules = HashMap::default();
                rules.extend(self.rulesets.get(ruleset_name).unwrap().clone()); 
                let mut searched = vec![];
                for (&name, rule) in rules.iter_mut() {
                    let mut all_values = vec![];
                    self.run_query(&rule.query, rule.todo_timestamp, |values| {
                        assert_eq!(values.len(), rule.query.vars.len());
                        all_values.extend_from_slice(values);
                        Ok(())
                    });
                    searched.push((name, all_values));
                }
                

                for (name, all_values) in searched {
                    let rule = rules.get_mut(&name).unwrap();
                    let n = rule.query.vars.len();
                    let stack = &mut vec![];
                    for values in all_values.chunks(n) {
                        // we can ignore results here
                        stack.clear();
                        let _ = self.run_actions(stack, values, &rule.program, true);
                    }
                }

                self.rebuild_nofail();
                return !self.saturated;
            },
            Schedule::Repeat(limit, sched) => {
                let mut updated = false;
                for _ in 0..*limit {
                    updated |= self.run_schedule(sched);
                }
                return updated;
            },
            Schedule::Saturate(sched) => {
                let mut updated = false;

                let mut still_updating = true;
                while still_updating {
                    still_updating = self.run_schedule(sched);
                    updated |= still_updating;
                }

                return updated;
            },
            Schedule::Sequence(scheds) => {
                let mut updated = false;
                for sched in scheds {
                    updated |= self.run_schedule(sched);
                }

                return updated;
            },

        }
    }

    pub fn run_rules(&mut self, config: &RunConfig) -> [Duration; 3] {
        let RunConfig { limit, until } = config;
        let mut search_time = Duration::default();
        let mut apply_time = Duration::default();

        // we might have to do a rebuild before starting,
        // because the use can manually do stuff
        let initial_rebuild_start = Instant::now();
        self.rebuild_nofail();
        let mut rebuild_time = initial_rebuild_start.elapsed();

        for i in 0..*limit {
            self.saturated = true;
            let [st, at] = self.step_rules(i);
            search_time += st;
            apply_time += at;

            let rebuild_start = Instant::now();
            let updates = self.rebuild_nofail();
            log::debug!("database size: {}", self.num_tuples());
            log::debug!("Made {updates} updates (iteration {i})");
            rebuild_time += rebuild_start.elapsed();
            self.timestamp += 1;
            if self.saturated {
                log::info!("Breaking early at iteration {}!", i);
                break;
            }
            if let Some(fact) = until {
                if self.check_fact(fact, false).is_ok() {
                    log::info!(
                        "Breaking early at iteration {} because of fact {}!",
                        i,
                        fact
                    );
                    break;
                }
            }
            if self.num_tuples() > self.node_limit {
                log::warn!(
                    "Node limit reached at iteration {}, {} nodes. Stopping!",
                    i,
                    self.num_tuples()
                );
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
                let mut fuel = self.match_limit << rule.times_banned;
                let rule_search_start = Instant::now();
                self.run_query(&rule.query, rule.todo_timestamp, |values| {
                    assert_eq!(values.len(), rule.query.vars.len());
                    all_values.extend_from_slice(values);
                    if fuel > 0 {
                        fuel -= 1;
                        Ok(())
                    } else {
                        Err(())
                    }
                });
                let rule_search_time = rule_search_start.elapsed();
                log::trace!(
                    "Searched for {name} in {} ({} results)",
                    rule_search_time.as_secs_f64(),
                    all_values.len()
                );
                rule.search_time += rule_search_time;
                searched.push((name, all_values));
            } else {
                self.saturated = false;
            }
        }
        let search_elapsed = search_start.elapsed();

        let apply_start = Instant::now();
        'outer: for (name, all_values) in searched {
            let rule = rules.get_mut(&name).unwrap();
            let num_vars = rule.query.vars.len();

            // the query doesn't require matches
            if num_vars != 0 {
                // backoff logic
                let len = all_values.len() / num_vars;
                let threshold = self.match_limit << rule.times_banned;
                if len > threshold {
                    let ban_length = ban_length << rule.times_banned;
                    rule.times_banned += 1;
                    rule.banned_until = iteration + ban_length;
                    log::info!("Banning rule {name} for {ban_length} iterations, matched {len} > {threshold} times");
                    self.saturated = false;
                    continue;
                }
            }

            rule.todo_timestamp = self.timestamp;
            let rule_apply_start = Instant::now();

            let stack = &mut vec![];
            // run one iteration when n == 0
            if num_vars == 0 {
                rule.matches += 1;
                // we can ignore results here
                stack.clear();
                let _ = self.run_actions(stack, &[], &rule.program, true);
            } else {
                for values in all_values.chunks(num_vars) {
                    rule.matches += 1;
                    if rule.matches > 10_000_000 {
                        log::warn!("Rule {} has matched {} times, bailing!", name, rule.matches);
                        break 'outer;
                    }
                    // we can ignore results here
                    stack.clear();
                    let _ = self.run_actions(stack, values, &rule.program, true);
                }
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
        let (query0, action0) = ctx
            .typecheck_query(&rule.body, &rule.head)
            .map_err(Error::TypeErrors)?;
        let query = self.compile_gj_query(query0, &ctx.types);
        let program = self
            .compile_actions(&ctx.types, &action0)
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

    pub fn add_ruleset(&mut self, name: Symbol) {
        if self.rulesets.contains_key(&name) {
            panic!("Ruleset '{name}' was already present");
        }
        self.rulesets.insert(
            name,
            self.rules
                .iter()
                .map(|pair| (*pair.0, pair.1.clone()))
                .collect(),
        );
    }

    pub fn load_ruleset(&mut self, name: Symbol) {
        self.rules.extend(self.rulesets.get(&name).unwrap().clone());
    }

    pub fn eval_actions(&mut self, actions: &[Action]) -> Result<(), Error> {
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
            Command::Datatype {
                name: _,
                variants: _,
            } => {
                panic!("Datatype should have been desugared");
            }
            Command::Sort(name, presort_and_args) => match presort_and_args {
                Some((presort, args)) => {
                    self.declare_sort(name, Some((presort, &args)))?;
                    format!(
                        "Declared sort {name} = ({presort} {})",
                        ListDisplay(&args, " ")
                    )
                }
                None => {
                    self.declare_sort(name, None)?;
                    format!("Declared sort {name}.")
                }
            },
            Command::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                format!("Declared function {}.", fdecl.name)
            }
            Command::Rule(rule) => {
                let name = self.add_rule(rule)?;
                format!("Declared rule {name}.")
            }
            Command::Rewrite(_rewrite) => {
                panic!("Rewrite should have been desugared");
            }
            Command::BiRewrite(_rewrite) => {
                panic!("Birewrite should have been desugared");
            }
            Command::Run(config) => {
                let limit = config.limit;
                if should_run {
                    let [st, at, rt] = self.run_rules(&config);
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
            Command::RunSchedule(sched) => {
                if should_run {
                    self.run_schedule(&sched);
                    format!("Ran schedule")
                } else {
                    format!("Skipping schedule")
                }
            }
            Command::Calc(idents, exprs) => {
                self.calc(idents.clone(), exprs.clone())?;
                format!(
                    "Calc proof succeeded: forall {}, {}",
                    ListDisplay(idents, " "),
                    ListDisplay(exprs, " = ")
                )
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
                    self.check_fact(&fact, true)?;
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
            Command::AddRuleset(name) => {
                self.add_ruleset(name);
                format!("Added ruleset {}", name)
            }
            Command::LoadRuleset(name) => {
                self.load_ruleset(name);
                format!("Loaded ruleset {}", name)
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
                self.clear();
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
            Command::Fail(c) => {
                if self.run_command(*c, should_run).is_ok() {
                    return Err(Error::ExpectFail);
                }
                "Command failed as expected.".into()
            }
            Command::Include(file) => {
                let s = std::fs::read_to_string(&file)
                    .unwrap_or_else(|_| panic!("Failed to read file {file}"));
                self.parse_and_run_program(&s)?;
                format!("Included file {file}")
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
            Command::Output { file, exprs } => {
                let mut filename = self.fact_directory.clone().unwrap_or_default();
                filename.push(file.as_str());
                // append to file
                let mut f = File::options()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(&filename)
                    .map_err(|e| Error::IoError(filename.clone(), e))?;

                for expr in exprs {
                    use std::io::Write;
                    let (_cost, expr, _exprs) = self.extract_expr(expr, 1)?;
                    writeln!(f, "{expr}").map_err(|e| Error::IoError(filename.clone(), e))?;
                }

                format!("Output to '{filename:?}'.")
            }
        })
    }

    pub fn clear(&mut self) {
        for f in self.functions.values_mut() {
            f.clear();
        }
    }

    fn calc_helper(
        &mut self,
        idents: Vec<IdentSort>,
        exprs: Vec<Expr>,
        depth: &mut i64,
    ) -> Result<(), Error> {
        self.push();
        *depth += 1;
        // Insert fresh symbols for locally universally quantified reasoning.
        for IdentSort { ident, sort } in idents {
            let sort = self.sorts.get(&sort).unwrap().clone();
            self.declare_const(ident, &sort)?;
        }
        // Insert each expression pair and run until they match.
        for ab in exprs.windows(2) {
            let a = &ab[0];
            let b = &ab[1];
            self.push();
            *depth += 1;
            self.eval_expr(a, None, true)?;
            self.eval_expr(b, None, true)?;
            let cond = Fact::Eq(vec![a.clone(), b.clone()]);
            self.run_command(
                Command::Run(RunConfig {
                    limit: 100000,
                    until: Some(cond.clone()),
                }),
                true,
            )?;
            self.run_command(Command::Check(cond), true)?;
            self.pop().unwrap();
            *depth -= 1;
        }
        self.pop().unwrap();
        *depth -= 1;
        Ok(())
    }

    // Prove a sequence of equalities universally quantified over idents
    pub fn calc(&mut self, idents: Vec<IdentSort>, exprs: Vec<Expr>) -> Result<(), Error> {
        if exprs.len() < 2 {
            Ok(())
        } else {
            let mut depth = 0;
            let res = self.calc_helper(idents, exprs, &mut depth);
            if res.is_err() {
                // pop egraph back to original state if error
                for _ in 0..depth {
                    self.pop()?;
                }
            } else {
                assert!(depth == 0);
            }
            res
        }
    }

    // Extract an expression from the current state, returning the cost, the extracted expression and some number
    // of other variants, if variants is not zero.
    pub fn extract_expr(
        &mut self,
        e: Expr,
        variants: usize,
    ) -> Result<(usize, Expr, Vec<Expr>), Error> {
        self.rebuild()?;
        let (_t, value) = self.eval_expr(&e, None, true)?;
        let (cost, expr) = self.extract(value);
        let exprs = match variants {
            0 => vec![],
            1 => vec![expr.clone()],
            _ => self.extract_variants(value, variants),
        };
        Ok((cost, expr, exprs))
    }

    pub fn declare_const(&mut self, name: Symbol, sort: &ArcSort) -> Result<(), Error> {
        assert!(sort.is_eq_sort());
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: vec![],
                output: sort.name(),
            },
            default: None,
            merge: None,
            merge_action: vec![],
            cost: None,
        })?;
        let f = self.functions.get_mut(&name).unwrap();
        let id = self.unionfind.make_set();
        let value = Value::from_id(sort.name(), id);
        f.insert(&[], value, self.timestamp);
        Ok(())
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
            merge_action: vec![],
            cost,
        })?;
        let f = self.functions.get_mut(&name).unwrap();
        f.insert(&[], value, self.timestamp);
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
    #[cfg(debug_assertions)]
    fn bad_find_value(&self, value: Value) -> Value {
        if let Some((tag, id)) = self.value_to_id(value) {
            Value::from_id(tag, self.find(id))
        } else {
            value
        }
    }

    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        let parser = ast::parse::ProgramParser::new();
        let program = parser
            .parse(input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?;
        Ok(desugar_program(program))
    }

    pub fn parse_and_run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let program = self.parse_program(input)?;

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
    #[error("Illegal merge attempted for function {0}, {1:?} != {2:?}")]
    MergeError(Symbol, Value, Value),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
    #[error("Presort {0} not found.")]
    PresortNotFound(Symbol),
    #[error("Tried to pop too much")]
    Pop,
    #[error("Command should have failed.")]
    ExpectFail,
    #[error("IO error: {0}: {1}")]
    IoError(PathBuf, std::io::Error),
}
