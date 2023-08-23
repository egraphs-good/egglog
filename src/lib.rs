pub mod ast;
mod extract;
mod function;
mod gj;
mod proofs;
mod serialize;
pub mod sort;
mod termdag;
mod typecheck;
mod typechecking;
mod unionfind;
pub mod util;
mod value;

use extract::Extractor;
use hashbrown::hash_map::Entry;
use index::ColumnIndex;
use instant::{Duration, Instant};
pub use serialize::SerializeConfig;
use sort::*;
pub use termdag::{Term, TermDag};
use thiserror::Error;

use proofs::ProofState;

use symbolic_expressions::Sexp;

use ast::*;
pub use typechecking::{TypeInfo, UNIT_SYM};

use std::fmt::{Display, Formatter, Write};
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::iter::once;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use std::{fmt::Debug, sync::Arc};
use typecheck::Program;

pub type ArcSort = Arc<dyn Sort>;

pub use value::*;

use function::*;
use gj::*;
use unionfind::*;
use util::*;

use crate::typechecking::TypeError;

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort>;
    fn apply(&self, values: &[Value]) -> Option<Value>;
}

#[derive(Debug, Clone, Default)]
pub struct RunReport {
    pub updated: bool,
    pub search_time: Duration,
    pub apply_time: Duration,
    pub rebuild_time: Duration,
}

/// A report of the results of an extract action.
#[derive(Debug, Clone)]
pub enum ExtractReport {
    Best {
        termdag: TermDag,
        cost: usize,
        expr: Term,
    },
    Variants {
        termdag: TermDag,
        variants: Vec<Term>,
    },
}

impl RunReport {
    pub fn union(&self, other: &Self) -> Self {
        Self {
            updated: self.updated || other.updated,
            search_time: self.search_time + other.search_time,
            apply_time: self.apply_time + other.apply_time,
            rebuild_time: self.rebuild_time + other.rebuild_time,
        }
    }
}

pub const HIGH_COST: usize = i64::MAX as usize;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum CompilerPassStop {
    Desugar,
    TypecheckDesugared,
    TermEncoding,
    TypecheckTermEncoding,
    Proofs,
    TypecheckProofs,
    All,
}

impl Display for CompilerPassStop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerPassStop::Desugar => write!(f, "desugar"),
            CompilerPassStop::TypecheckDesugared => write!(f, "typecheck_desugared"),
            CompilerPassStop::TermEncoding => write!(f, "term_encoding"),
            CompilerPassStop::TypecheckTermEncoding => write!(f, "typecheck_term_encoding"),
            CompilerPassStop::Proofs => write!(f, "proofs"),
            CompilerPassStop::TypecheckProofs => write!(f, "typecheck_proofs"),
            CompilerPassStop::All => write!(f, "all"),
        }
    }
}

impl FromStr for CompilerPassStop {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "desugar" => Ok(CompilerPassStop::Desugar),
            "typecheck_desugared" => Ok(CompilerPassStop::TypecheckDesugared),
            "term_encoding" => Ok(CompilerPassStop::TermEncoding),
            "typecheck_term_encoding" => Ok(CompilerPassStop::TypecheckTermEncoding),
            "proofs" => Ok(CompilerPassStop::Proofs),
            "typecheck_proofs" => Ok(CompilerPassStop::TypecheckProofs),
            "all" => Ok(CompilerPassStop::All),
            _ => Err(format!("Unknown compiler pass stop: {}", s)),
        }
    }
}

#[derive(Clone)]
pub struct EGraph {
    egraphs: Vec<Self>,
    unionfind: UnionFind,
    pub(crate) proof_state: ProofState,
    functions: HashMap<Symbol, Function>,
    rulesets: HashMap<Symbol, HashMap<Symbol, Rule>>,
    ruleset_iteration: HashMap<Symbol, usize>,
    proofs_enabled: bool,
    interactive_mode: bool,
    timestamp: u32,
    pub test_proofs: bool,
    pub match_limit: usize,
    pub node_limit: usize,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
    // sort, value, and timestamp
    pub global_bindings: HashMap<Symbol, (ArcSort, Value, u32)>,
    extract_report: Option<ExtractReport>,
    run_report: Option<RunReport>,
    msgs: Vec<String>,
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
            functions: Default::default(),
            rulesets: Default::default(),
            ruleset_iteration: Default::default(),
            proof_state: ProofState::default(),
            global_bindings: Default::default(),
            match_limit: usize::MAX,
            node_limit: usize::MAX,
            timestamp: 0,
            proofs_enabled: false,
            interactive_mode: false,
            test_proofs: false,
            fact_directory: None,
            seminaive: true,
            extract_report: None,
            run_report: None,
            msgs: Default::default(),
        };
        egraph.rulesets.insert("".into(), Default::default());
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(Expr);

impl EGraph {
    pub fn is_interactive_mode(&self) -> bool {
        self.interactive_mode
    }

    pub fn push(&mut self) {
        self.egraphs.push(self.clone());
    }

    pub fn pop(&mut self) -> Result<(), Error> {
        match self.egraphs.pop() {
            Some(e) => {
                // Copy the reports and messages from the popped egraph
                let extract_report = self.extract_report.clone();
                let run_report = self.run_report.clone();
                let messages = self.msgs.clone();
                *self = e;
                if let Some(report) = extract_report {
                    self.extract_report = Some(report);
                }
                if let Some(report) = run_report {
                    self.run_report = Some(report);
                }
                self.msgs.extend(messages);
                Ok(())
            }
            None => Err(Error::Pop),
        }
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
                            (*off as usize) < function.nodes.num_offsets(),
                            "index contains offset {off:?}, which is out of range for function {name}"
                        );
                    }
                }
            }
            for (rix, sort) in function.rebuild_indexes.iter().zip(
                function
                    .schema
                    .input
                    .iter()
                    .chain(once(&function.schema.output)),
            ) {
                assert!(sort.is_eq_container_sort() == rix.is_some());
                if sort.is_eq_container_sort() {
                    let rix = rix.as_ref().unwrap();
                    for ix in rix.iter() {
                        for (_, offs) in ix.iter() {
                            for off in offs {
                                assert!(
                                (*off as usize) < function.nodes.num_offsets(),
                                "index contains offset {off:?}, which is out of range for function {name}"
                            );
                            }
                        }
                    }
                }
            }
        }
    }

    // find the leader term for this term
    // in the corresponding table
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

        // now update global bindings
        let mut new_global_bindings = std::mem::take(&mut self.global_bindings);
        for (_sym, (sort, value, ts)) in new_global_bindings.iter_mut() {
            if sort.canonicalize(value, &self.unionfind) {
                *ts = self.timestamp;
            }
        }
        self.global_bindings = new_global_bindings;

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

    pub fn declare_function(&mut self, decl: &FunctionDecl) -> Result<(), Error> {
        let function = Function::new(self, decl)?;
        let old = self.functions.insert(decl.name, function);
        if old.is_some() {
            panic!(
                "Typechecking should have caught function already bound: {}",
                decl.name
            );
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
            unextractable: false,
        })?;
        // if let Some(ctors) = self.sorts.get_mut(&sort) {
        //     ctors.push(name);
        // }
        Ok(())
    }

    pub fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(i) => i.store(&self.proof_state.type_info.get_sort()).unwrap(),
            Literal::F64(f) => f.store(&self.proof_state.type_info.get_sort()).unwrap(),
            Literal::String(s) => s.store(&self.proof_state.type_info.get_sort()).unwrap(),
            Literal::Unit => ().store(&self.proof_state.type_info.get_sort()).unwrap(),
        }
    }

    pub fn function_to_dag(
        &mut self,
        sym: Symbol,
        n: usize,
    ) -> Result<(Vec<(Term, Term)>, TermDag), Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        let schema = f.schema.clone();
        let nodes = f
            .nodes
            .iter()
            .take(n)
            .map(|(k, v)| (ValueVec::from(k), v.clone()))
            .collect::<Vec<_>>();

        let mut termdag = TermDag::default();
        let extractor = Extractor::new(self, &mut termdag);
        let mut terms = Vec::new();
        for (ins, out) in nodes {
            let mut children = Vec::new();
            for (a, a_type) in ins.iter().copied().zip(&schema.input) {
                if a_type.is_eq_sort() {
                    children.push(extractor.find_best(a, &mut termdag, a_type).unwrap().1);
                } else {
                    children.push(termdag.expr_to_term(&a_type.make_expr(self, a).1));
                };
            }

            let out = if schema.output.is_eq_sort() {
                extractor
                    .find_best(out.value, &mut termdag, &schema.output)
                    .unwrap()
                    .1
            } else {
                termdag.expr_to_term(&schema.output.make_expr(self, out.value).1)
            };
            terms.push((termdag.make(sym, children), out));
        }
        drop(extractor);

        Ok((terms, termdag))
    }

    pub fn print_function(&mut self, sym: Symbol, n: usize) -> Result<(), Error> {
        log::info!("Printing up to {n} tuples of table {sym}: ");
        let (terms_with_outputs, termdag) = self.function_to_dag(sym, n)?;
        let f = self
            .functions
            .get(&sym)
            .ok_or(TypeError::UnboundFunction(sym))?;
        let out_is_unit = f.schema.output.name() == UNIT_SYM.into();

        let mut buf = String::new();
        let s = &mut buf;
        s.push_str("(\n");
        if terms_with_outputs.is_empty() {
            log::info!("   (none)");
        }
        for (term, output) in terms_with_outputs {
            let tuple_str = format!(
                "   {}{}",
                termdag.to_string(&term),
                if !out_is_unit {
                    format!(" -> {}", termdag.to_string(&output))
                } else {
                    "".into()
                },
            );
            log::info!("{}", tuple_str);
            s.push_str(&tuple_str);
            s.push('\n');
        }
        s.push_str(")\n");
        self.print_msg(buf);
        Ok(())
    }

    pub fn print_size(&mut self, sym: Symbol) -> Result<(), Error> {
        let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
        log::info!("Function {} has size {}", sym, f.nodes.len());
        self.print_msg(f.nodes.len().to_string());
        Ok(())
    }

    // returns whether the egraph was updated
    pub fn run_schedule(&mut self, sched: &NormSchedule) -> RunReport {
        match sched {
            NormSchedule::Run(config) => self.run_rules(config),
            NormSchedule::Repeat(limit, sched) => {
                let mut report = RunReport::default();
                for _i in 0..*limit {
                    let rec = self.run_schedule(sched);
                    report = report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                report
            }
            NormSchedule::Saturate(sched) => {
                let mut report = RunReport::default();
                loop {
                    let rec = self.run_schedule(sched);
                    report = report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                report
            }
            NormSchedule::Sequence(scheds) => {
                let mut report = RunReport::default();
                for sched in scheds {
                    report = report.union(&self.run_schedule(sched));
                }
                report
            }
        }
    }

    pub fn run_rules_once(&mut self, config: &NormRunConfig, report: &mut RunReport) {
        // first rebuild
        let rebuild_start = Instant::now();
        let updates = self.rebuild_nofail();
        log::debug!("database size: {}", self.num_tuples());
        log::debug!("Made {updates} updates");
        report.rebuild_time += rebuild_start.elapsed();
        self.timestamp += 1;

        let NormRunConfig { ruleset, until } = config;

        if let Some(facts) = until {
            if self.check_facts(facts).is_ok() {
                log::info!(
                    "Breaking early because of facts:\n {}!",
                    ListDisplay(facts, "\n")
                );
                return;
            }
        }

        let subreport = self.step_rules(*ruleset);
        *report = report.union(&subreport);

        log::debug!("database size: {}", self.num_tuples());
        self.timestamp += 1;

        if self.num_tuples() > self.node_limit {
            log::warn!("Node limit reached, {} nodes. Stopping!", self.num_tuples());
        }
    }

    pub fn run_rules(&mut self, config: &NormRunConfig) -> RunReport {
        let mut report: RunReport = Default::default();

        self.run_rules_once(config, &mut report);

        // Report the worst offenders
        log::debug!("Slowest rules:\n{}", {
            let mut msg = String::new();
            let mut vec = self
                .rulesets
                .iter()
                .flat_map(|(_name, rules)| rules)
                .collect::<Vec<_>>();
            vec.sort_by_key(|(_, r)| r.search_time + r.apply_time);
            for (name, rule) in vec.iter().rev().take(5) {
                write!(
                    msg,
                    "{name}\n  Search: {:.3}s\n  Apply: {:.3}s\n",
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
        report
    }

    fn step_rules(&mut self, ruleset: Symbol) -> RunReport {
        let n_unions_before = self.unionfind.n_unions();
        // don't ban parent or rebuilding
        let match_limit =
            if ruleset.as_str().contains("parent_") || ruleset.as_str().contains("rebuilding_") {
                usize::MAX
            } else {
                self.match_limit
            };
        let mut report = RunReport::default();

        let ban_length = 5;

        if !self.rulesets.contains_key(&ruleset) {
            panic!("run: No ruleset named '{ruleset}'");
        }
        let mut rules: HashMap<Symbol, Rule> =
            std::mem::take(self.rulesets.get_mut(&ruleset).unwrap());
        let iteration = *self.ruleset_iteration.entry(ruleset).or_default();
        self.ruleset_iteration.insert(ruleset, iteration + 1);
        // TODO why did I have to copy the rules here for the first for loop?
        let copy_rules = rules.clone();
        let search_start = Instant::now();
        let mut searched = vec![];
        for (name, rule) in copy_rules.iter() {
            let mut all_values = vec![];
            if rule.banned_until <= iteration {
                let mut fuel = safe_shl(match_limit, rule.times_banned);
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
                    "Searched for {name} in {:.3}s ({} results)",
                    rule_search_time.as_secs_f64(),
                    all_values.len()
                );
                searched.push((name, all_values, rule_search_time));
            }
        }

        let search_elapsed = search_start.elapsed();
        report.search_time += search_elapsed;

        let apply_start = Instant::now();
        for (name, all_values, time) in searched {
            let rule = rules.get_mut(name).unwrap();
            rule.search_time += time;
            let num_vars = rule.query.vars.len();

            // the query doesn't require matches
            if num_vars != 0 {
                // backoff logic
                let len = all_values.len() / num_vars;
                let threshold = safe_shl(match_limit, rule.times_banned);
                if len > threshold {
                    let ban_length = safe_shl(ban_length, rule.times_banned);
                    rule.times_banned = rule.times_banned.saturating_add(1);
                    rule.banned_until = iteration + ban_length;
                    log::info!("Banning rule {name} for {ban_length} iterations, matched {len} > {threshold} times");
                    report.updated = true;
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
                    // we can ignore results here
                    stack.clear();
                    let _ = self.run_actions(stack, values, &rule.program, true);
                }
            }

            rule.apply_time += rule_apply_start.elapsed();
        }
        self.rulesets.insert(ruleset, rules);
        let apply_elapsed = apply_start.elapsed();
        report.apply_time += apply_elapsed;
        report.updated |= self.did_change_tables() || n_unions_before != self.unionfind.n_unions();

        report
    }

    fn did_change_tables(&self) -> bool {
        for (_name, function) in &self.functions {
            if function.nodes.max_ts() >= self.timestamp {
                return true;
            }
        }

        false
    }

    fn add_rule_with_name(
        &mut self,
        name: String,
        rule: ast::Rule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
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
        if let Some(rules) = self.rulesets.get_mut(&ruleset) {
            match rules.entry(name) {
                Entry::Occupied(_) => panic!("Rule '{name}' was already present"),
                Entry::Vacant(e) => e.insert(compiled_rule),
            };
        } else {
            panic!("No such ruleset {ruleset}");
        }
        Ok(name)
    }

    pub fn add_rule(&mut self, rule: ast::Rule, ruleset: Symbol) -> Result<Symbol, Error> {
        let name = format!("{}", rule);
        self.add_rule_with_name(name, rule, ruleset)
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

    pub fn eval_expr(
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

    fn add_ruleset(&mut self, name: Symbol) {
        match self.rulesets.entry(name) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Default::default()),
        };
    }

    pub fn set_option(&mut self, name: &str, value: Expr) {
        match name {
            "enable_proofs" => {
                self.proofs_enabled = true;
            }
            "interactive_mode" => {
                if let Expr::Lit(Literal::Int(i)) = value {
                    self.interactive_mode = i != 0;
                } else {
                    panic!("interactive_mode must be an integer");
                }
            }
            "match_limit" => {
                if let Expr::Lit(Literal::Int(i)) = value {
                    self.match_limit = i as usize;
                } else {
                    panic!("match_limit must be an integer");
                }
            }
            "node_limit" => {
                if let Expr::Lit(Literal::Int(i)) = value {
                    self.node_limit = i as usize;
                } else {
                    panic!("node_limit must be an integer");
                }
            }
            _ => panic!("Unknown option '{}'", name),
        }
    }

    fn check_facts(&mut self, facts: &[NormFact]) -> Result<(), Error> {
        let mut ctx = typecheck::Context::new(self);
        let converted_facts = facts.iter().map(|f| f.to_fact()).collect::<Vec<Fact>>();
        let empty_actions = vec![];
        let (query0, _) = ctx
            .typecheck_query(&converted_facts, &empty_actions)
            .map_err(Error::TypeErrors)?;
        let query = self.compile_gj_query(query0, &ctx.types);

        let mut matched = false;
        // TODO what timestamp to use?
        self.run_query(&query, 0, |values| {
            assert_eq!(values.len(), query.vars.len());
            matched = true;
            Err(())
        });
        if !matched {
            // TODO add useful info here
            Err(Error::CheckError(facts.to_vec()))
        } else {
            Ok(())
        }
    }

    fn run_command(&mut self, command: NCommand, should_run: bool) -> Result<(), Error> {
        let pre_rebuild = Instant::now();
        let rebuild_num = self.rebuild()?;
        if rebuild_num > 0 {
            log::info!(
                "Rebuild before command: {:10}ms",
                pre_rebuild.elapsed().as_millis()
            );
        }

        self.debug_assert_invariants();

        match command {
            NCommand::SetOption { name, value } => {
                let str = format!("Set option {} to {}", name, value);
                self.set_option(name.into(), value);
                log::info!("{}", str)
            }
            // Sorts are already declared during typechecking
            NCommand::Sort(name, _presort_and_args) => log::info!("Declared sort {}.", name),
            NCommand::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                log::info!("Declared function {}.", fdecl.name)
            }
            NCommand::AddRuleset(name) => {
                self.add_ruleset(name);
                log::info!("Declared ruleset {name}.");
            }
            NCommand::NormRule {
                ruleset,
                rule,
                name,
            } => {
                self.add_rule(rule.to_rule(), ruleset)?;
                log::info!("Declared rule {name}.")
            }
            NCommand::RunSchedule(sched) => {
                if should_run {
                    self.run_report = Some(self.run_schedule(&sched));
                    log::info!("Ran schedule {}.", sched)
                } else {
                    log::warn!("Skipping schedule.")
                }
            }
            NCommand::Check(facts) => {
                if should_run {
                    self.check_facts(&facts)?;
                    log::info!("Checked fact {:?}.", facts);
                } else {
                    log::warn!("Skipping check.")
                }
            }
            NCommand::CheckProof => log::error!("TODO implement proofs"),
            NCommand::NormAction(action) => {
                if should_run {
                    match &action {
                        NormAction::Let(name, contents) => {
                            let (etype, value) = self.eval_expr(&contents.to_expr(), None, true)?;
                            let present = self
                                .global_bindings
                                .insert(*name, (etype, value, self.timestamp));
                            if present.is_some() {
                                panic!("Variable {name} was already present in global bindings");
                            }
                        }
                        NormAction::LetVar(var1, var2) => {
                            let value = self.global_bindings.get(var2).unwrap();
                            let present = self.global_bindings.insert(*var1, value.clone());
                            if present.is_some() {
                                panic!("Variable {var1} was already present in global bindings");
                            }
                        }
                        NormAction::LetLit(var, lit) => {
                            let value = self.eval_lit(lit);
                            let etype = self.proof_state.type_info.infer_literal(lit);
                            let present = self
                                .global_bindings
                                .insert(*var, (etype, value, self.timestamp));

                            if present.is_some() {
                                panic!("Variable {var} was already present in global bindings");
                            }
                        }
                        _ => {
                            self.eval_actions(std::slice::from_ref(&action.to_action()))?;
                        }
                    }
                } else {
                    log::warn!("Skipping running {action}.")
                }
            }
            NCommand::Push(n) => {
                (0..n).for_each(|_| self.push());
                log::info!("Pushed {n} levels.")
            }
            NCommand::Pop(n) => {
                for _ in 0..n {
                    self.pop()?;
                }
                log::info!("Popped {n} levels.")
            }
            NCommand::PrintTable(f, n) => {
                self.print_function(f, n)?;
            }
            NCommand::PrintSize(f) => {
                self.print_size(f)?;
            }
            NCommand::Fail(c) => {
                let result = self.run_command(*c, should_run);
                if let Err(e) = result {
                    log::info!("Command failed as expected: {}", e);
                } else {
                    return Err(Error::ExpectFail);
                }
            }
            NCommand::Input { name, file } => {
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
                log::info!("Read {} facts into {name} from '{file}'.", actions.len())
            }
            NCommand::Output { file, exprs } => {
                let mut filename = self.fact_directory.clone().unwrap_or_default();
                filename.push(file.as_str());
                // append to file
                let mut f = File::options()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(&filename)
                    .map_err(|e| Error::IoError(filename.clone(), e))?;
                let mut termdag = TermDag::default();
                for expr in exprs {
                    let (t, value) = self.eval_expr(&expr, None, true)?;
                    let expr = self.extract(value, &mut termdag, &t).1;
                    use std::io::Write;
                    writeln!(f, "{}", termdag.to_string(&expr))
                        .map_err(|e| Error::IoError(filename.clone(), e))?;
                }

                log::info!("Output to '{filename:?}'.")
            }
        };
        Ok(())
    }

    pub fn clear(&mut self) {
        for f in self.functions.values_mut() {
            f.clear();
        }
    }

    pub fn process_commands(
        &mut self,
        program: Vec<Command>,
        stop: CompilerPassStop,
    ) -> Result<Vec<NormCommand>, Error> {
        let mut result = vec![];

        for command in program {
            match command {
                Command::Push(num) => {
                    for _ in 0..num {
                        self.push();
                    }
                }
                Command::Pop(num) => {
                    for _ in 0..num {
                        self.pop()
                            .expect("Failed to desugar, popped too many times");
                    }
                }
                _ => {}
            }
            result.extend(self.process_command(command, stop)?);
        }
        Ok(result)
    }

    pub fn set_underscores_for_desugaring(&mut self, underscores: usize) {
        self.proof_state.desugar.number_underscores = underscores;
    }

    fn process_command(
        &mut self,
        command: Command,
        stop: CompilerPassStop,
    ) -> Result<Vec<NormCommand>, Error> {
        let program = self.proof_state.desugar.desugar_program(
            vec![command],
            self.test_proofs,
            self.seminaive,
        )?;
        if stop == CompilerPassStop::Desugar {
            return Ok(program);
        }

        let type_info_before = self.proof_state.type_info.clone();

        self.proof_state.type_info.typecheck_program(&program)?;
        if stop == CompilerPassStop::TypecheckDesugared {
            return Ok(program);
        }

        // reset type info
        self.proof_state.type_info = type_info_before;
        self.proof_state.type_info.typecheck_program(&program)?;
        if stop == CompilerPassStop::TypecheckTermEncoding {
            return Ok(program);
        }

        Ok(program)
    }

    pub fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<String>, Error> {
        let should_run = true;

        for command in program {
            // Important to process each command individually
            // because push and pop create new scopes
            for processed in self.process_command(command, CompilerPassStop::All)? {
                self.run_command(processed.command, should_run)?;
            }
        }
        log::logger().flush();

        // remove consecutive empty lines
        Ok(self.flush_msgs())
    }

    // this is bad because we shouldn't inspect values like this, we should use type information
    fn bad_find_value(&self, value: Value) -> Value {
        if let Some((tag, id)) = self.value_to_id(value) {
            Value::from_id(tag, self.find(id))
        } else {
            value
        }
    }

    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        self.proof_state.desugar.parse_program(input)
    }

    pub fn parse_and_run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let parsed = self.proof_state.desugar.parse_program(input)?;
        self.run_program(parsed)
    }

    pub fn num_tuples(&self) -> usize {
        self.functions.values().map(|f| f.nodes.len()).sum()
    }

    pub(crate) fn get_sort(&self, value: &Value) -> Option<&ArcSort> {
        self.proof_state.type_info.sorts.get(&value.tag)
    }

    pub fn add_arcsort(&mut self, arcsort: ArcSort) -> Result<(), TypeError> {
        self.proof_state.type_info.add_arcsort(arcsort)
    }

    /// Gets the last extract report and returns it, if the last command saved it.
    pub fn get_extract_report(&self) -> &Option<ExtractReport> {
        &self.extract_report
    }

    /// Gets the last run report and returns it, if the last command saved it.
    pub fn get_run_report(&self) -> &Option<RunReport> {
        &self.run_report
    }

    /// Serializes the egraph for export to graphviz.
    pub fn serialize_for_graphviz(&self) -> egraph_serialize::EGraph {
        let mut serialized = self.serialize(SerializeConfig::default());
        serialized.inline_leaves();
        serialized
    }

    pub(crate) fn print_msg(&mut self, msg: String) {
        self.msgs.push(msg);
    }

    fn flush_msgs(&mut self) -> Vec<String> {
        self.msgs.dedup_by(|a, b| a.is_empty() && b.is_empty());
        std::mem::take(&mut self.msgs)
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
    #[error("Check failed: \n{}", ListDisplay(.0, "\n"))]
    CheckError(Vec<NormFact>),
    #[error("Evaluating primitive {0:?} failed. ({0:?} {:?})", ListDebug(.1, " "))]
    PrimitiveError(Primitive, Vec<Value>),
    #[error("Illegal merge attempted for function {0}, {1:?} != {2:?}")]
    MergeError(Symbol, Value, Value),
    #[error("Tried to pop too much")]
    Pop,
    #[error("Command should have failed.")]
    ExpectFail,
    #[error("IO error: {0}: {1}")]
    IoError(PathBuf, std::io::Error),
}

fn safe_shl(a: usize, b: usize) -> usize {
    a.checked_shl(b.try_into().unwrap()).unwrap_or(usize::MAX)
}
