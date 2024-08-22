//! # egglog
//! egglog is a language specialized for writing equality saturation
//! applications. It is the successor to the rust library [egg](https://github.com/egraphs-good/egg).
//! egglog is faster and more general than egg.
//!
//! # Documentation
//! Documentation for the egglog language can be found
//! here: [`Command`]
//!
//! # Tutorial
//! [Here](https://www.youtube.com/watch?v=N2RDQGRBrSY) is the video tutorial on what egglog is and how to use it.
//! We plan to have a text tutorial here soon, PRs welcome!
//!
mod actions;
pub mod ast;
pub mod constraint;
mod core;
mod extract;
mod function;
mod gj;
mod serialize;
pub mod sort;
mod termdag;
mod typechecking;
mod unionfind;
pub mod util;
mod value;

use ast::desugar::Desugar;
use ast::remove_globals::remove_globals;
use extract::Extractor;
use hashbrown::hash_map::Entry;
use index::ColumnIndex;
use instant::{Duration, Instant};
pub use serialize::SerializeConfig;
use sort::*;
pub use termdag::{Term, TermDag, TermId};
use thiserror::Error;

use generic_symbolic_expressions::Sexp;

use ast::*;
pub use typechecking::{TypeInfo, UNIT_SYM};

use crate::core::{AtomTerm, ResolvedCall};
use actions::Program;
use constraint::{Constraint, SimpleTypeConstraint, TypeConstraint};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::iter::once;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use std::{fmt::Debug, sync::Arc};

pub type ArcSort = Arc<dyn Sort>;

pub use value::*;

pub use function::Function;
use function::*;
use gj::*;
use unionfind::*;
use util::*;

use crate::constraint::Problem;
use crate::typechecking::TypeError;

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    /// Constructs a type constraint for the primitive that uses the span information
    /// for error localization.
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint>;
    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value>;
}

/// Running a schedule produces a report of the results.
/// This includes rough timing information and whether
/// the database was updated.
/// Calling `union` on two run reports adds the timing
/// information together.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// If any changes were made to the database, this is
    /// true.
    pub updated: bool,
    /// The time it took to run the query, for each rule.
    pub search_time_per_rule: HashMap<Symbol, Duration>,
    pub apply_time_per_rule: HashMap<Symbol, Duration>,
    pub search_time_per_ruleset: HashMap<Symbol, Duration>,
    pub num_matches_per_rule: HashMap<Symbol, usize>,
    pub apply_time_per_ruleset: HashMap<Symbol, Duration>,
    pub rebuild_time_per_ruleset: HashMap<Symbol, Duration>,
}

impl RunReport {
    /// add a ... and a maximum size to the name
    /// for printing, since they may be the rule itself
    fn truncate_rule_name(sym: Symbol) -> String {
        let mut s = sym.to_string();
        // replace newlines in s with a space
        s = s.replace('\n', " ");
        if s.len() > 80 {
            s.truncate(80);
            s.push_str("...");
        }
        s
    }

    fn add_rule_search_time(&mut self, rule: Symbol, time: Duration) {
        *self.search_time_per_rule.entry(rule).or_default() += time;
    }

    fn add_ruleset_search_time(&mut self, ruleset: Symbol, time: Duration) {
        *self.search_time_per_ruleset.entry(ruleset).or_default() += time;
    }

    fn add_rule_apply_time(&mut self, rule: Symbol, time: Duration) {
        *self.apply_time_per_rule.entry(rule).or_default() += time;
    }

    fn add_ruleset_apply_time(&mut self, ruleset: Symbol, time: Duration) {
        *self.apply_time_per_ruleset.entry(ruleset).or_default() += time;
    }

    fn add_ruleset_rebuild_time(&mut self, ruleset: Symbol, time: Duration) {
        *self.rebuild_time_per_ruleset.entry(ruleset).or_default() += time;
    }

    fn add_rule_num_matches(&mut self, rule: Symbol, num_matches: usize) {
        *self.num_matches_per_rule.entry(rule).or_default() += num_matches;
    }
}

impl Display for RunReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let all_rules = self
            .search_time_per_rule
            .keys()
            .chain(self.apply_time_per_rule.keys())
            .collect::<HashSet<_>>();
        let mut all_rules_vec = all_rules.iter().cloned().collect::<Vec<_>>();
        // sort rules by search and apply time
        all_rules_vec.sort_by_key(|rule| {
            let search_time = self
                .search_time_per_rule
                .get(*rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_millis();
            let apply_time = self
                .apply_time_per_rule
                .get(*rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_millis();
            search_time + apply_time
        });

        for rule in all_rules_vec {
            let truncated = Self::truncate_rule_name(*rule);
            // print out the search and apply time for rule
            let search_time = self
                .search_time_per_rule
                .get(rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let apply_time = self
                .apply_time_per_rule
                .get(rule)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let num_matches = self.num_matches_per_rule.get(rule).cloned().unwrap_or(0);
            writeln!(
                f,
                "Rule {truncated}: search {search_time:.3}s, apply {apply_time:.3}s, num matches {num_matches}",
            )?;
        }

        let rulesets = self
            .search_time_per_ruleset
            .keys()
            .chain(self.apply_time_per_ruleset.keys())
            .chain(self.rebuild_time_per_ruleset.keys())
            .collect::<HashSet<_>>();

        for ruleset in rulesets {
            // print out the search and apply time for rule
            let search_time = self
                .search_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let apply_time = self
                .apply_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            let rebuild_time = self
                .rebuild_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::default())
                .as_secs_f64();
            writeln!(
                f,
                "Ruleset {ruleset}: search {search_time:.3}s, apply {apply_time:.3}s, rebuild {rebuild_time:.3}s",
            )?;
        }

        Ok(())
    }
}

/// A report of the results of an extract action.
#[derive(Debug, Clone)]
pub enum ExtractReport {
    Best {
        termdag: TermDag,
        cost: usize,
        term: Term,
    },
    Variants {
        termdag: TermDag,
        terms: Vec<Term>,
    },
}

impl RunReport {
    fn union_times(
        times: &HashMap<Symbol, Duration>,
        other_times: &HashMap<Symbol, Duration>,
    ) -> HashMap<Symbol, Duration> {
        let mut new_times = times.clone();
        for (k, v) in other_times {
            let entry = new_times.entry(*k).or_default();
            *entry += *v;
        }
        new_times
    }

    fn union_counts(
        counts: &HashMap<Symbol, usize>,
        other_counts: &HashMap<Symbol, usize>,
    ) -> HashMap<Symbol, usize> {
        let mut new_counts = counts.clone();
        for (k, v) in other_counts {
            let entry = new_counts.entry(*k).or_default();
            *entry += *v;
        }
        new_counts
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            updated: self.updated || other.updated,
            search_time_per_rule: Self::union_times(
                &self.search_time_per_rule,
                &other.search_time_per_rule,
            ),
            apply_time_per_rule: Self::union_times(
                &self.apply_time_per_rule,
                &other.apply_time_per_rule,
            ),
            num_matches_per_rule: Self::union_counts(
                &self.num_matches_per_rule,
                &other.num_matches_per_rule,
            ),
            search_time_per_ruleset: Self::union_times(
                &self.search_time_per_ruleset,
                &other.search_time_per_ruleset,
            ),
            apply_time_per_ruleset: Self::union_times(
                &self.apply_time_per_ruleset,
                &other.apply_time_per_ruleset,
            ),
            rebuild_time_per_ruleset: Self::union_times(
                &self.rebuild_time_per_ruleset,
                &other.rebuild_time_per_ruleset,
            ),
        }
    }
}

pub const HIGH_COST: usize = i64::MAX as usize;

#[derive(Clone)]
pub struct Primitive(Arc<dyn PrimitiveLike>);
impl Primitive {
    // Takes the full signature of a primitive (including input and output types)
    // Returns whether the primitive is compatible with this signature
    fn accept(&self, tys: &[Arc<dyn Sort>], typeinfo: &TypeInfo) -> bool {
        let mut constraints = vec![];
        let lits: Vec<_> = (0..tys.len())
            .map(|i| AtomTerm::Literal(DUMMY_SPAN.clone(), Literal::Int(i as i64)))
            .collect();
        for (lit, ty) in lits.iter().zip(tys.iter()) {
            constraints.push(Constraint::Assign(lit.clone(), ty.clone()))
        }
        constraints.extend(self.get_type_constraints(&DUMMY_SPAN).get(&lits, typeinfo));
        let problem = Problem {
            constraints,
            range: HashSet::default(),
        };
        problem.solve(|sort| sort.name()).is_ok()
    }
}

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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let sorts: Vec<_> = self
            .input
            .iter()
            .chain(once(&self.output as &ArcSort))
            .cloned()
            .collect();
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        (self.f)(values)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum RunMode {
    Normal,
    ShowDesugaredEgglog,
    // TODO: supporting them needs to refactor the way NCommand is organized.
    // There is no version of NCommand where CoreRule is used in place of Rule.
    // As a result, we cannot just call to_lower_rule and get a NCommand with lowered CoreRule in it
    // and print it out.
    // A refactoring that allows NCommand to contain CoreRule can make this possible.
    // ShowCore,
    // ShowResugaredCore,
}
impl RunMode {
    fn show_egglog(&self) -> bool {
        matches!(self, RunMode::ShowDesugaredEgglog)
    }
}

impl Display for RunMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // A little bit unintuitive but RunMode is specified as command-line
        // argument with flag `--show`, so `--show none` means a normal run.
        match self {
            RunMode::Normal => write!(f, "none"),
            RunMode::ShowDesugaredEgglog => write!(f, "desugared-egglog"),
            // RunMode::ShowCore => write!(f, "core"),
            // RunMode::ShowResugaredCore => write!(f, "resugared-core"),
        }
    }
}

impl FromStr for RunMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(RunMode::Normal),
            "desugared-egglog" => Ok(RunMode::ShowDesugaredEgglog),
            // "core" => Ok(RunMode::ShowCore),
            // "resugared-core" => Ok(RunMode::ShowResugaredCore),
            _ => Err(format!("Unknown run mode: {s}")),
        }
    }
}

#[derive(Clone)]
pub struct EGraph {
    egraphs: Vec<Self>,
    unionfind: UnionFind,
    pub(crate) desugar: Desugar,
    pub functions: HashMap<Symbol, Function>,
    rulesets: HashMap<Symbol, Ruleset>,
    rule_last_run_timestamp: HashMap<Symbol, u32>,
    proofs_enabled: bool,
    terms_enabled: bool,
    interactive_mode: bool,
    timestamp: u32,
    pub run_mode: RunMode,
    pub test_proofs: bool,
    pub match_limit: usize,
    pub node_limit: usize,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
    type_info: TypeInfo,
    extract_report: Option<ExtractReport>,
    /// The run report for the most recent run of a schedule.
    recent_run_report: Option<RunReport>,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,
    msgs: Vec<String>,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            egraphs: vec![],
            unionfind: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            rule_last_run_timestamp: Default::default(),
            desugar: Desugar::default(),
            match_limit: usize::MAX,
            node_limit: usize::MAX,
            timestamp: 0,
            run_mode: RunMode::Normal,
            proofs_enabled: false,
            terms_enabled: false,
            interactive_mode: false,
            test_proofs: false,
            fact_directory: None,
            seminaive: true,
            extract_report: None,
            recent_run_report: None,
            overall_run_report: Default::default(),
            msgs: Default::default(),
            type_info: Default::default(),
        };
        egraph
            .rulesets
            .insert("".into(), Ruleset::Rules("".into(), Default::default()));
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(String);

/// For each rule, we produce a `SearchResult`
/// storing data about that rule's matches.
/// When a rule has no variables, it may still match- in this case
/// the `did_match` field is used.
struct SearchResult {
    all_matches: Vec<Value>,
    did_match: bool,
}

impl EGraph {
    /// Use the rust backend implimentation of eqsat,
    /// including a rust implementation of the union-find
    /// data structure and the rust implementation of
    /// the rebuilding algorithm (maintains congruence closure).
    pub fn enable_terms_encoding(&mut self) {
        self.terms_enabled = true;
    }

    pub fn is_interactive_mode(&self) -> bool {
        self.interactive_mode
    }

    pub fn push(&mut self) {
        self.egraphs.push(self.clone());
    }

    /// Pop the current egraph off the stack, replacing
    /// it with the previously pushed egraph.
    /// It preserves the run report and messages from the popped
    /// egraph.
    pub fn pop(&mut self) -> Result<(), Error> {
        match self.egraphs.pop() {
            Some(e) => {
                // Copy the reports and messages from the popped egraph
                let extract_report = self.extract_report.clone();
                let recent_run_report = self.recent_run_report.clone();
                let overall_run_report = self.overall_run_report.clone();
                let messages = self.msgs.clone();

                *self = e;
                self.extract_report = extract_report.or(self.extract_report.clone());
                // We union the run reports, meaning
                // that statistics are shared across
                // push/pop
                self.recent_run_report = recent_run_report.or(self.recent_run_report.clone());
                self.overall_run_report = overall_run_report;
                self.msgs = messages;
                Ok(())
            }
            None => Err(Error::Pop(DUMMY_SPAN.clone())),
        }
    }

    pub fn union(&mut self, id1: Id, id2: Id, sort: Symbol) -> Id {
        self.unionfind.union(id1, id2, sort)
    }

    #[track_caller]
    fn debug_assert_invariants(&self) {
        // we can't use find before something
        // is added to the parent table, so this
        // is disabled in terms mode
        if self.terms_enabled {
            return;
        }
        #[cfg(debug_assertions)]
        for (name, function) in self.functions.iter() {
            function.nodes.assert_sorted();
            for (i, inputs, output) in function.nodes.iter_range(0..function.nodes.len(), true) {
                for input in inputs {
                    assert_eq!(
                        input,
                        &self.find(*input),
                        "[{i}] {name}({inputs:?}) = {output:?}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output.value,
                    self.find(output.value),
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

    /// find the leader value for a particular eclass
    pub fn find(&self, value: Value) -> Value {
        if self.terms_enabled {
            // HACK using value tag for parent table name
            let parent_name = self
                .desugar
                .lookup_parent_name(value.tag)
                .unwrap_or_else(|| {
                    panic!(
                        "Term encoding should have created parent table for {}",
                        value.tag
                    )
                });
            if let Some(func) = self.functions.get(&parent_name) {
                func.get(&[value])
                    .unwrap_or_else(|| panic!("No value {:?} in {parent_name}.", value,))
            } else {
                value
            }
        } else {
            if let Some(sort) = self.get_sort_from_value(&value) {
                if sort.is_eq_sort() {
                    return Value {
                        tag: value.tag,
                        bits: usize::from(self.unionfind.find(Id::from(value.bits as usize)))
                            as u64,
                    };
                }
            }
            value
        }
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

    fn declare_function(&mut self, decl: &ResolvedFunctionDecl) -> Result<(), Error> {
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

    pub fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(i) => i.store(&self.type_info().get_sort_nofail()).unwrap(),
            Literal::F64(f) => f.store(&self.type_info().get_sort_nofail()).unwrap(),
            Literal::String(s) => s.store(&self.type_info().get_sort_nofail()).unwrap(),
            Literal::Unit => ().store(&self.type_info().get_sort_nofail()).unwrap(),
            Literal::Bool(b) => b.store(&self.type_info().get_sort_nofail()).unwrap(),
        }
    }

    pub fn function_to_dag(
        &mut self,
        sym: Symbol,
        n: usize,
    ) -> Result<(Vec<(Term, Term)>, TermDag), Error> {
        let f = self
            .functions
            .get(&sym)
            .ok_or(TypeError::UnboundFunction(sym, DUMMY_SPAN.clone()))?;
        let schema = f.schema.clone();
        let nodes = f
            .nodes
            .iter(true)
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
            terms.push((termdag.app(sym, children), out));
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
            // function_to_dag should have checked this
            .unwrap();
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

    pub fn print_size(&mut self, sym: Option<Symbol>) -> Result<(), Error> {
        if let Some(sym) = sym {
            let f = self
                .functions
                .get(&sym)
                .ok_or(TypeError::UnboundFunction(sym, DUMMY_SPAN.clone()))?;
            log::info!("Function {} has size {}", sym, f.nodes.len());
            self.print_msg(f.nodes.len().to_string());
            Ok(())
        } else {
            // Print size of all functions
            let mut lens = self
                .functions
                .iter()
                .map(|(sym, f)| (*sym, f.nodes.len()))
                .collect::<Vec<_>>();

            // Function name's alphabetical order
            lens.sort_by_key(|(name, _)| name.as_str());

            for (sym, len) in &lens {
                log::info!("Function {} has size {}", sym, len);
            }

            self.print_msg(
                lens.into_iter()
                    .map(|(name, len)| format!("{}: {}", name, len))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );

            Ok(())
        }
    }

    // returns whether the egraph was updated
    fn run_schedule(&mut self, sched: &ResolvedSchedule) -> RunReport {
        match sched {
            ResolvedSchedule::Run(span, config) => self.run_rules(span, config),
            ResolvedSchedule::Repeat(_span, limit, sched) => {
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
            ResolvedSchedule::Saturate(_span, sched) => {
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
            ResolvedSchedule::Sequence(_span, scheds) => {
                let mut report = RunReport::default();
                for sched in scheds {
                    report = report.union(&self.run_schedule(sched));
                }
                report
            }
        }
    }

    /// Extract a value to a [`TermDag`] and [`Term`]
    /// in the [`TermDag`].
    /// See also extract_value_to_string for convenience.
    pub fn extract_value(&self, value: Value) -> (TermDag, Term) {
        let mut termdag = TermDag::default();
        let sort = self.type_info().sorts.get(&value.tag).unwrap();
        let term = self.extract(value, &mut termdag, sort).1;
        (termdag, term)
    }

    /// Extract a value to a string for printing.
    /// See also extract_value for more control.
    pub fn extract_value_to_string(&self, value: Value) -> String {
        let (termdag, term) = self.extract_value(value);
        termdag.to_string(&term)
    }

    fn run_rules(&mut self, span: &Span, config: &ResolvedRunConfig) -> RunReport {
        let mut report: RunReport = Default::default();

        // first rebuild
        let rebuild_start = Instant::now();
        let updates = self.rebuild_nofail();
        log::debug!("database size: {}", self.num_tuples());
        log::debug!("Made {updates} updates");
        // add to the rebuild time for this ruleset
        report.add_ruleset_rebuild_time(config.ruleset, rebuild_start.elapsed());
        self.timestamp += 1;

        let GenericRunConfig { ruleset, until } = config;

        if let Some(facts) = until {
            if self.check_facts(span, facts).is_ok() {
                log::info!(
                    "Breaking early because of facts:\n {}!",
                    ListDisplay(facts, "\n")
                );
                return report;
            }
        }

        let subreport = self.step_rules(*ruleset);
        report = report.union(&subreport);

        log::debug!("database size: {}", self.num_tuples());
        self.timestamp += 1;

        if self.num_tuples() > self.node_limit {
            log::warn!("Node limit reached, {} nodes. Stopping!", self.num_tuples());
        }

        report
    }

    /// Search all the rules in a ruleset.
    /// Add the search results for a rule to search_results, a map indexed by rule name.
    fn search_rules(
        &self,
        ruleset: Symbol,
        run_report: &mut RunReport,
        search_results: &mut HashMap<Symbol, SearchResult>,
    ) {
        let rules = self
            .rulesets
            .get(&ruleset)
            .unwrap_or_else(|| panic!("ruleset does not exist: {}", &ruleset));
        match rules {
            Ruleset::Rules(_ruleset_name, rule_names) => {
                let copy_rules = rule_names.clone();
                let search_start = Instant::now();

                for (rule_name, rule) in copy_rules.iter() {
                    let mut all_matches = vec![];
                    let rule_search_start = Instant::now();
                    let mut did_match = false;
                    let timestamp = self.rule_last_run_timestamp.get(rule_name).unwrap_or(&0);
                    self.run_query(&rule.query, *timestamp, false, |values| {
                        did_match = true;
                        assert_eq!(values.len(), rule.query.vars.len());
                        all_matches.extend_from_slice(values);
                        Ok(())
                    });
                    let rule_search_time = rule_search_start.elapsed();
                    log::trace!(
                        "Searched for {rule_name} in {:.3}s ({} results)",
                        rule_search_time.as_secs_f64(),
                        all_matches.len()
                    );
                    run_report.add_rule_search_time(*rule_name, rule_search_time);
                    search_results.insert(
                        *rule_name,
                        SearchResult {
                            all_matches,
                            did_match,
                        },
                    );
                }

                let search_time = search_start.elapsed();
                run_report.add_ruleset_search_time(ruleset, search_time);
            }
            Ruleset::Combined(_name, sub_rulesets) => {
                let start_time = Instant::now();
                for sub_ruleset in sub_rulesets {
                    self.search_rules(*sub_ruleset, run_report, search_results);
                }
                let search_time = start_time.elapsed();
                run_report.add_ruleset_search_time(ruleset, search_time);
            }
        }
    }

    fn apply_rules(
        &mut self,
        ruleset: Symbol,
        run_report: &mut RunReport,
        search_results: &HashMap<Symbol, SearchResult>,
    ) {
        // TODO this clone is not efficient
        let rules = self.rulesets.get(&ruleset).unwrap().clone();
        match rules {
            Ruleset::Rules(_name, compiled_rules) => {
                let apply_start = Instant::now();
                let rule_names = compiled_rules.keys().cloned().collect::<Vec<_>>();
                for rule_name in rule_names {
                    let SearchResult {
                        all_matches,
                        did_match,
                    } = search_results.get(&rule_name).unwrap();
                    let rule = compiled_rules.get(&rule_name).unwrap();
                    let num_vars = rule.query.vars.len();

                    // make sure the query requires matches
                    if num_vars != 0 {
                        run_report.add_rule_num_matches(rule_name, all_matches.len() / num_vars);
                    }

                    self.rule_last_run_timestamp
                        .insert(rule_name, self.timestamp);
                    let rule_apply_start = Instant::now();

                    let stack = &mut vec![];

                    // when there are no variables, a query can still fail to match
                    // here we handle that case
                    if num_vars == 0 {
                        if *did_match {
                            stack.clear();
                            self.run_actions(stack, &[], &rule.program, true)
                                .unwrap_or_else(|e| {
                                    panic!("error while running actions for {rule_name}: {e}")
                                });
                        }
                    } else {
                        for values in all_matches.chunks(num_vars) {
                            stack.clear();
                            self.run_actions(stack, values, &rule.program, true)
                                .unwrap_or_else(|e| {
                                    panic!("error while running actions for {rule_name}: {e}")
                                });
                        }
                    }

                    // add to the rule's apply time
                    run_report.add_rule_apply_time(rule_name, rule_apply_start.elapsed());
                }
                run_report.add_ruleset_apply_time(ruleset, apply_start.elapsed());
            }
            Ruleset::Combined(_name, sub_rulesets) => {
                let start_time = Instant::now();
                for sub_ruleset in sub_rulesets {
                    self.apply_rules(sub_ruleset, run_report, search_results);
                }
                let apply_time = start_time.elapsed();
                run_report.add_ruleset_apply_time(ruleset, apply_time);
            }
        }
    }

    fn step_rules(&mut self, ruleset: Symbol) -> RunReport {
        let n_unions_before = self.unionfind.n_unions();
        let mut run_report = Default::default();
        let mut search_results = HashMap::<Symbol, SearchResult>::default();
        self.search_rules(ruleset, &mut run_report, &mut search_results);
        self.apply_rules(ruleset, &mut run_report, &search_results);
        run_report.updated |=
            self.did_change_tables() || n_unions_before != self.unionfind.n_unions();

        run_report
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
        rule: ast::ResolvedRule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let core_rule = rule.to_canonicalized_core_rule(self.type_info())?;
        let (query, actions) = (core_rule.body, core_rule.head);

        let vars = query.get_vars();
        let query = self.compile_gj_query(query, &vars);

        let program = self
            .compile_actions(&vars, &actions)
            .map_err(Error::TypeErrors)?;
        let compiled_rule = CompiledRule { query, program };
        if let Some(rules) = self.rulesets.get_mut(&ruleset) {
            match rules {
                Ruleset::Rules(_, rules) => {
                    match rules.entry(name) {
                        indexmap::map::Entry::Occupied(_) => {
                            panic!("Rule '{name}' was already present")
                        }
                        indexmap::map::Entry::Vacant(e) => e.insert(compiled_rule),
                    };
                    Ok(name)
                }
                Ruleset::Combined(_, _) => Err(Error::CombinedRulesetError(ruleset, rule.span)),
            }
        } else {
            Err(Error::NoSuchRuleset(ruleset, rule.span))
        }
    }

    pub(crate) fn add_rule(
        &mut self,
        rule: ast::ResolvedRule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
        let name = format!("{}", rule);
        self.add_rule_with_name(name, rule, ruleset)
    }

    fn eval_actions(&mut self, actions: &ResolvedActions) -> Result<(), Error> {
        let (actions, _) = actions.to_core_actions(
            self.type_info(),
            &mut Default::default(),
            &mut ResolvedGen::new("$".to_string()),
        )?;
        let program = self
            .compile_actions(&Default::default(), &actions)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, true)?;
        Ok(())
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, Value), Error> {
        let fresh_name = self.desugar.get_fresh();
        let command = Command::Action(Action::Let(DUMMY_SPAN.clone(), fresh_name, expr.clone()));
        self.run_program(vec![command])?;
        // find the table with the same name as the fresh name
        let func = self.functions.get(&fresh_name).unwrap();
        let value = func.nodes.get(&[]).unwrap().value;
        let sort = func.schema.output.clone();
        Ok((sort, value))
    }

    // TODO make a public version of eval_expr that makes a command,
    // then returns the value at the end.
    fn eval_resolved_expr(
        &mut self,
        expr: &ResolvedExpr,
        make_defaults: bool,
    ) -> Result<Value, Error> {
        let (actions, mapped_expr) = expr.to_core_actions(
            self.type_info(),
            &mut Default::default(),
            &mut ResolvedGen::new("$".to_string()),
        )?;
        let target = mapped_expr.get_corresponding_var_or_lit(self.type_info());
        let program = self
            .compile_expr(&Default::default(), &actions, &target)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, make_defaults)?;
        Ok(stack.pop().unwrap())
    }

    fn add_combined_ruleset(&mut self, name: Symbol, rulesets: Vec<Symbol>) {
        match self.rulesets.entry(name) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Combined(name, rulesets)),
        };
    }

    fn add_ruleset(&mut self, name: Symbol) {
        match self.rulesets.entry(name) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Rules(name, Default::default())),
        };
    }

    fn set_option(&mut self, name: &str, value: ResolvedExpr) {
        match name {
            "enable_proofs" => {
                self.proofs_enabled = true;
            }
            "interactive_mode" => {
                if let ResolvedExpr::Lit(_ann, Literal::Int(i)) = value {
                    self.interactive_mode = i != 0;
                } else {
                    panic!("interactive_mode must be an integer");
                }
            }
            "match_limit" => {
                if let ResolvedExpr::Lit(_ann, Literal::Int(i)) = value {
                    self.match_limit = i as usize;
                } else {
                    panic!("match_limit must be an integer");
                }
            }
            "node_limit" => {
                if let ResolvedExpr::Lit(_ann, Literal::Int(i)) = value {
                    self.node_limit = i as usize;
                } else {
                    panic!("node_limit must be an integer");
                }
            }
            _ => panic!("Unknown option '{}'", name),
        }
    }

    fn check_facts(&mut self, span: &Span, facts: &[ResolvedFact]) -> Result<(), Error> {
        let rule = ast::ResolvedRule {
            span: span.clone(),
            head: ResolvedActions::default(),
            body: facts.to_vec(),
        };
        let core_rule = rule.to_canonicalized_core_rule(self.type_info())?;
        let query = core_rule.body;
        let ordering = &query.get_vars();
        let query = self.compile_gj_query(query, ordering);

        let mut matched = false;
        self.run_query(&query, 0, true, |values| {
            assert_eq!(values.len(), query.vars.len());
            matched = true;
            Err(())
        });
        if !matched {
            Err(Error::CheckError(
                facts.iter().map(|f| f.clone().make_unresolved()).collect(),
                span.clone(),
            ))
        } else {
            Ok(())
        }
    }

    fn run_command(&mut self, command: ResolvedNCommand) -> Result<(), Error> {
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
            ResolvedNCommand::SetOption { name, value } => {
                let str = format!("Set option {} to {}", name, value);
                self.set_option(name.into(), value);
                log::info!("{}", str)
            }
            // Sorts are already declared during typechecking
            ResolvedNCommand::Sort(_span, name, _presort_and_args) => {
                log::info!("Declared sort {}.", name)
            }
            ResolvedNCommand::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                log::info!("Declared function {}.", fdecl.name)
            }
            ResolvedNCommand::AddRuleset(name) => {
                self.add_ruleset(name);
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::UnstableCombinedRuleset(name, others) => {
                self.add_combined_ruleset(name, others);
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::NormRule {
                ruleset,
                rule,
                name,
            } => {
                self.add_rule(rule, ruleset)?;
                log::info!("Declared rule {name}.")
            }
            ResolvedNCommand::RunSchedule(sched) => {
                let report = self.run_schedule(&sched);
                log::info!("Ran schedule {}.", sched);
                log::info!("Report: {}", report);
                self.overall_run_report = self.overall_run_report.union(&report);
                self.recent_run_report = Some(report);
            }
            ResolvedNCommand::PrintOverallStatistics => {
                log::info!("Overall statistics:\n{}", self.overall_run_report);
                self.print_msg(format!("Overall statistics:\n{}", self.overall_run_report));
            }
            ResolvedNCommand::Check(span, facts) => {
                self.check_facts(&span, &facts)?;
                log::info!("Checked fact {:?}.", facts);
            }
            ResolvedNCommand::CheckProof => log::error!("TODO implement proofs"),
            ResolvedNCommand::CoreAction(action) => match &action {
                ResolvedAction::Let(_, name, contents) => {
                    panic!("Globals should have been desugared away: {name} = {contents}")
                }
                _ => {
                    self.eval_actions(&ResolvedActions::new(vec![action.clone()]))?;
                }
            },
            ResolvedNCommand::Push(n) => {
                (0..n).for_each(|_| self.push());
                log::info!("Pushed {n} levels.")
            }
            ResolvedNCommand::Pop(span, n) => {
                for _ in 0..n {
                    self.pop().map_err(|err| {
                        if let Error::Pop(_) = err {
                            Error::Pop(span.clone())
                        } else {
                            err
                        }
                    })?;
                }
                log::info!("Popped {n} levels.")
            }
            ResolvedNCommand::PrintTable(span, f, n) => {
                self.print_function(f, n).map_err(|e| match e {
                    Error::TypeError(TypeError::UnboundFunction(f, _)) => {
                        Error::TypeError(TypeError::UnboundFunction(f, span.clone()))
                    }
                    // This case is currently impossible
                    _ => e,
                })?;
            }
            ResolvedNCommand::PrintSize(span, f) => {
                self.print_size(f).map_err(|e| match e {
                    Error::TypeError(TypeError::UnboundFunction(f, _)) => {
                        Error::TypeError(TypeError::UnboundFunction(f, span.clone()))
                    }
                    // This case is currently impossible
                    _ => e,
                })?;
            }
            ResolvedNCommand::Fail(span, c) => {
                let result = self.run_command(*c);
                if let Err(e) = result {
                    log::info!("Command failed as expected: {e}");
                } else {
                    return Err(Error::ExpectFail(span));
                }
            }
            ResolvedNCommand::Input {
                span: _,
                name,
                file,
            } => {
                self.input_file(name, file)?;
            }
            ResolvedNCommand::Output { span, file, exprs } => {
                let mut filename = self.fact_directory.clone().unwrap_or_default();
                filename.push(file.as_str());
                // append to file
                let mut f = File::options()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(&filename)
                    .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                let mut termdag = TermDag::default();
                for expr in exprs {
                    let value = self.eval_resolved_expr(&expr, true)?;
                    let expr_type = expr.output_type(self.type_info());
                    let term = self.extract(value, &mut termdag, &expr_type).1;
                    use std::io::Write;
                    writeln!(f, "{}", termdag.to_string(&term))
                        .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                }

                log::info!("Output to '{filename:?}'.")
            }
        };
        Ok(())
    }

    fn input_file(&mut self, func_name: Symbol, file: String) -> Result<(), Error> {
        let function_type = self
            .type_info()
            .lookup_user_func(func_name)
            .unwrap_or_else(|| panic!("Unrecognized function name {}", func_name));
        let func = self.functions.get_mut(&func_name).unwrap();

        let mut filename = self.fact_directory.clone().unwrap_or_default();
        filename.push(file.as_str());

        // check that the function uses supported types

        for t in &func.schema.input {
            match t.name().as_str() {
                "i64" | "f64" | "String" => {}
                s => panic!("Unsupported type {} for input", s),
            }
        }

        if !function_type.is_datatype {
            match func.schema.output.name().as_str() {
                "i64" | "String" | "Unit" => {}
                s => panic!("Unsupported type {} for input", s),
            }
        }

        log::info!("Opening file '{:?}'...", filename);
        let mut f = File::open(filename).unwrap();
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();

        let span: Span = DUMMY_SPAN.clone();
        let mut actions: Vec<Action> = vec![];
        let mut str_buf: Vec<&str> = vec![];
        for line in contents.lines() {
            str_buf.clear();
            str_buf.extend(line.split('\t').map(|s| s.trim()));
            if str_buf.is_empty() {
                continue;
            }

            let parse = |s: &str| -> Expr {
                match s.parse::<i64>() {
                    Ok(i) => Expr::Lit(span.clone(), Literal::Int(i)),
                    Err(_) => match s.parse::<f64>() {
                        Ok(f) => Expr::Lit(span.clone(), Literal::F64(f.into())),
                        Err(_) => Expr::Lit(span.clone(), Literal::String(s.into())),
                    },
                }
            };

            let mut exprs: Vec<Expr> = str_buf.iter().map(|&s| parse(s)).collect();

            actions.push(
                if function_type.is_datatype || function_type.output.name() == UNIT_SYM.into() {
                    Action::Expr(span.clone(), Expr::Call(span.clone(), func_name, exprs))
                } else {
                    let out = exprs.pop().unwrap();
                    Action::Set(span.clone(), func_name, exprs, out)
                },
            );
        }
        let num_facts = actions.len();
        let commands = actions
            .into_iter()
            .map(NCommand::CoreAction)
            .collect::<Vec<_>>();
        let commands: Vec<_> = self.type_info_mut().typecheck_program(&commands)?;
        for command in commands {
            self.run_command(command)?;
        }
        log::info!("Read {num_facts} facts into {func_name} from '{file}'.");
        Ok(())
    }

    pub fn clear(&mut self) {
        for f in self.functions.values_mut() {
            f.clear();
        }
    }

    pub fn set_reserved_symbol(&mut self, sym: Symbol) {
        assert!(
            !self.desugar.fresh_gen.has_been_used(),
            "Reserved symbol must be set before any symbols are generated"
        );
        self.desugar.fresh_gen = SymbolGen::new(sym.to_string());
    }

    fn process_command(&mut self, command: Command) -> Result<Vec<ResolvedNCommand>, Error> {
        let program =
            self.desugar
                .desugar_program(vec![command], self.test_proofs, self.seminaive)?;

        let program = self.type_info_mut().typecheck_program(&program)?;

        let program = remove_globals(&self.type_info, program, &mut self.desugar.fresh_gen);

        Ok(program)
    }

    /// Run a program, represented as an AST.
    /// Return a list of messages.
    pub fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<String>, Error> {
        for command in program {
            // Important to process each command individually
            // because push and pop create new scopes
            for processed in self.process_command(command)? {
                if self.run_mode.show_egglog() {
                    // In show_egglog mode, we still need to run scope-related commands (Push/Pop) to make
                    // the program well-scoped.
                    match &processed {
                        ResolvedNCommand::Push(..) | ResolvedNCommand::Pop(..) => {
                            self.run_command(processed.clone())?;
                        }
                        _ => {}
                    };
                    self.print_msg(processed.to_command().to_string());
                    continue;
                }

                self.run_command(processed)?;
            }
        }
        log::logger().flush();

        Ok(self.flush_msgs())
    }

    pub fn parse_program(
        &self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<Command>, Error> {
        self.desugar.parse_program(filename, input)
    }

    /// Takes a source program `input`, parses it, runs it, and returns a list of messages.
    ///
    /// `filename` is an optional argument to indicate the source of
    /// the program for error reporting. If `filename` is `None`,
    /// a default name will be used.
    pub fn parse_and_run_program(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<String>, Error> {
        let parsed = self.desugar.parse_program(filename, input)?;
        self.run_program(parsed)
    }

    pub fn num_tuples(&self) -> usize {
        self.functions.values().map(|f| f.nodes.len()).sum()
    }

    pub(crate) fn get_sort_from_value(&self, value: &Value) -> Option<&ArcSort> {
        self.type_info().sorts.get(&value.tag)
    }

    /// Returns the first sort that satisfies the type and predicate if there's one.
    /// Otherwise returns none.
    pub fn get_sort_by<S: Sort + Send + Sync>(
        &self,
        pred: impl Fn(&Arc<S>) -> bool,
    ) -> Option<Arc<S>> {
        self.type_info().get_sort_by(pred)
    }

    /// Returns a sort based on the type
    pub fn get_sort<S: Sort + Send + Sync>(&self) -> Option<Arc<S>> {
        self.type_info().get_sort_by(|_| true)
    }

    /// Add a user-defined sort
    pub fn add_arcsort(&mut self, arcsort: ArcSort) -> Result<(), TypeError> {
        self.type_info_mut()
            .add_arcsort(arcsort, DUMMY_SPAN.clone())
    }

    /// Add a user-defined primitive
    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        self.type_info_mut().add_primitive(prim)
    }

    /// Gets the last extract report and returns it, if the last command saved it.
    pub fn get_extract_report(&self) -> &Option<ExtractReport> {
        &self.extract_report
    }

    /// Gets the last run report and returns it, if the last command saved it.
    pub fn get_run_report(&self) -> &Option<RunReport> {
        &self.recent_run_report
    }

    /// Gets the overall run report and returns it.
    pub fn get_overall_run_report(&self) -> &RunReport {
        &self.overall_run_report
    }

    /// Serializes the egraph for export to graphviz.
    ///
    /// This will limit the total number of nodes so that visualization does not blow up.
    pub fn serialize_for_graphviz(
        &self,
        split_primitive_outputs: bool,
        max_functions: usize,
        max_calls_per_function: usize,
    ) -> egraph_serialize::EGraph {
        let config = SerializeConfig {
            split_primitive_outputs,
            max_functions: Some(max_functions),
            max_calls_per_function: Some(max_calls_per_function),
            ..Default::default()
        };
        self.serialize(config)
    }

    pub(crate) fn print_msg(&mut self, msg: String) {
        self.msgs.push(msg);
    }

    fn flush_msgs(&mut self) -> Vec<String> {
        self.msgs.dedup_by(|a, b| a.is_empty() && b.is_empty());
        std::mem::take(&mut self.msgs)
    }

    pub(crate) fn type_info(&self) -> &TypeInfo {
        &self.type_info
    }

    pub(crate) fn type_info_mut(&mut self) -> &mut TypeInfo {
        &mut self.type_info
    }
}

// Currently, only the following errors can thrown without location information:
// * PrimitiveError
// * MergeError
// * SubsumeMergeError
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
    #[error("{}\nCheck failed: \n{}", .1.get_quote(), ListDisplay(.0, "\n"))]
    CheckError(Vec<Fact>, Span),
    #[error("{}\nNo such ruleset: {0}", .1.get_quote())]
    NoSuchRuleset(Symbol, Span),
    #[error("{}\nAttempted to add a rule to combined ruleset {0}. Combined rulesets may only depend on other rulesets.", .1.get_quote())]
    CombinedRulesetError(Symbol, Span),
    #[error("Evaluating primitive {0:?} failed. ({0:?} {:?})", ListDebug(.1, " "))]
    PrimitiveError(Primitive, Vec<Value>),
    #[error("Illegal merge attempted for function {0}, {1:?} != {2:?}")]
    MergeError(Symbol, Value, Value),
    #[error("{}\nTried to pop too much", .0.get_quote())]
    Pop(Span),
    #[error("{}\nCommand should have failed.", .0.get_quote())]
    ExpectFail(Span),
    #[error("{}\nIO error: {0}: {1}", .2.get_quote())]
    IoError(PathBuf, std::io::Error, Span),
    #[error("Cannot subsume function with merge: {0}")]
    SubsumeMergeError(Symbol),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        constraint::SimpleTypeConstraint,
        sort::{FromSort, I64Sort, IntoSort, Sort, VecSort},
        EGraph, PrimitiveLike, Span, Value,
    };

    struct InnerProduct {
        ele: Arc<I64Sort>,
        vec: Arc<VecSort>,
    }

    impl PrimitiveLike for InnerProduct {
        fn name(&self) -> symbol_table::GlobalSymbol {
            "inner-product".into()
        }

        fn get_type_constraints(&self, span: &Span) -> Box<dyn crate::constraint::TypeConstraint> {
            SimpleTypeConstraint::new(
                self.name(),
                vec![self.vec.clone(), self.vec.clone(), self.ele.clone()],
                span.clone(),
            )
            .into_box()
        }

        fn apply(
            &self,
            values: &[crate::Value],
            _egraph: Option<&mut EGraph>,
        ) -> Option<crate::Value> {
            let mut sum = 0;
            let vec1 = Vec::<Value>::load(&self.vec, &values[0]);
            let vec2 = Vec::<Value>::load(&self.vec, &values[1]);
            assert_eq!(vec1.len(), vec2.len());
            for (a, b) in vec1.iter().zip(vec2.iter()) {
                let a = i64::load(&self.ele, a);
                let b = i64::load(&self.ele, b);
                sum += a * b;
            }
            sum.store(&self.ele)
        }
    }

    #[test]
    fn test_user_defined_primitive() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                "
                (sort IntVec (Vec i64))
            ",
            )
            .unwrap();
        let i64_sort: Arc<I64Sort> = egraph.get_sort().unwrap();
        let int_vec_sort: Arc<VecSort> = egraph
            .get_sort_by(|s: &Arc<VecSort>| s.element_name() == i64_sort.name())
            .unwrap();
        egraph.add_primitive(InnerProduct {
            ele: i64_sort,
            vec: int_vec_sort,
        });
        egraph
            .parse_and_run_program(
                None,
                "
                (let a (vec-of 1 2 3 4 5 6))
                (let b (vec-of 6 5 4 3 2 1))
                (check (= (inner-product a b) 56))
            ",
            )
            .unwrap();
    }
}
