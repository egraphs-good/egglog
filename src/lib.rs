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
mod cli;
pub mod constraint;
mod core;
pub mod extract;
mod function;
mod gj;
mod serialize;
pub mod sort;
mod termdag;
mod typechecking;
mod unionfind;
pub mod util;
mod value;

// This is used to allow the `add_primitive` macro to work in
// both this crate and other crates by referring to `::egglog`.
extern crate self as egglog;
pub use add_primitive::add_primitive;

use crate::constraint::Problem;
use crate::core::{AtomTerm, ResolvedCall};
use crate::typechecking::TypeError;
use actions::Program;
use ast::remove_globals::remove_globals;
use ast::*;
#[cfg(feature = "bin")]
pub use cli::bin::*;
use constraint::{Constraint, SimpleTypeConstraint, TypeConstraint};
use core::ResolvedAtomTerm;
use core_relations::{ExternalFunctionId, PrimitivePrinter};
use egglog_bridge::{ColumnTy, QueryEntry};
use extract::Extractor;
pub use function::Function;
use function::*;
use gj::*;
use index::ColumnIndex;
use indexmap::map::Entry;
use instant::{Duration, Instant};
pub use serialize::{SerializeConfig, SerializedNode};
use sort::*;
use std::fmt::Debug;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::iter::once;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
pub use termdag::{Term, TermDag, TermId};
use thiserror::Error;
pub use typechecking::TypeInfo;
use unionfind::*;
use util::*;
pub use value::*;

pub type ArcSort = Arc<dyn Sort>;

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    /// Constructs a type constraint for the primitive that uses the span information
    /// for error localization.
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint>;
    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        egraph: Option<&mut EGraph>,
    ) -> Option<Value>;
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

#[derive(Clone)]
pub struct Primitive(Arc<dyn PrimitiveLike + Send + Sync>, ExternalFunctionId);
impl Primitive {
    // Takes the full signature of a primitive (including input and output types)
    // Returns whether the primitive is compatible with this signature
    fn accept(&self, tys: &[Arc<dyn Sort>], typeinfo: &TypeInfo) -> bool {
        let mut constraints = vec![];
        let lits: Vec<_> = (0..tys.len())
            .map(|i| AtomTerm::Literal(Span::Panic, Literal::Int(i as i64)))
            .collect();
        for (lit, ty) in lits.iter().zip(tys.iter()) {
            constraints.push(constraint::assign(lit.clone(), ty.clone()))
        }
        constraints.extend(self.get_type_constraints(&Span::Panic).get(&lits, typeinfo));
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
    pub backend: egglog_bridge::EGraph,
    pub parser: Parser,
    egraphs: Vec<Self>,
    unionfind: UnionFind,
    pub functions: IndexMap<Symbol, Function>,
    rulesets: IndexMap<Symbol, Ruleset>,
    rule_last_run_timestamp: HashMap<Symbol, u32>,
    interactive_mode: bool,
    timestamp: u32,
    pub run_mode: RunMode,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
    type_info: TypeInfo,
    extract_report: Option<ExtractReport>,
    /// The run report for the most recent run of a schedule.
    recent_run_report: Option<RunReport>,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,
    /// Messages to be printed to the user. If this is `None`, then we are ignoring messages.
    msgs: Option<Vec<String>>,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut eg = Self {
            backend: Default::default(),
            parser: Default::default(),
            egraphs: vec![],
            unionfind: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            rule_last_run_timestamp: Default::default(),
            timestamp: 0,
            run_mode: RunMode::Normal,
            interactive_mode: false,
            fact_directory: None,
            seminaive: true,
            extract_report: None,
            recent_run_report: None,
            overall_run_report: Default::default(),
            msgs: Some(vec![]),
            type_info: Default::default(),
        };

        eg.add_sort(UnitSort, span!()).unwrap();
        eg.add_sort(StringSort, span!()).unwrap();
        eg.add_sort(BoolSort, span!()).unwrap();
        eg.add_sort(I64Sort, span!()).unwrap();
        eg.add_sort(F64Sort, span!()).unwrap();
        eg.add_sort(BigIntSort, span!()).unwrap();
        eg.add_sort(BigRatSort, span!()).unwrap();
        eg.type_info.add_presort::<MapSort>(span!()).unwrap();
        eg.type_info.add_presort::<SetSort>(span!()).unwrap();
        eg.type_info.add_presort::<VecSort>(span!()).unwrap();
        eg.type_info.add_presort::<FunctionSort>(span!()).unwrap();
        eg.type_info.add_presort::<MultiSetSort>(span!()).unwrap();

        add_primitive!(&mut eg, "!=" = |a: #, b: #| -?> () {
            (a != b).then_some(())
        });
        add_primitive!(&mut eg, "value-eq" = |a: #, b: #| -?> () {
            (a == b).then_some(())
        });
        add_primitive!(&mut eg, "ordering-min" = |a: #, b: #| -> # {
            if a < b { a } else { b }
        });
        add_primitive!(&mut eg, "ordering-max" = |a: #, b: #| -> # {
            if a > b { a } else { b }
        });

        eg.rulesets
            .insert("".into(), Ruleset::Rules("".into(), Default::default()));

        eg
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
    pub fn is_interactive_mode(&self) -> bool {
        self.interactive_mode
    }

    pub fn push(&mut self) {
        self.egraphs.push(self.clone());
    }

    /// Disable saving messages to be printed to the user and remove any saved messages.
    ///
    /// When messages are disabled the vec of messages returned by evaluating commands will always be empty.
    pub fn disable_messages(&mut self) {
        self.msgs = None;
    }

    /// Enable saving messages to be printed to the user.
    pub fn enable_messages(&mut self) {
        self.msgs = Some(vec![]);
    }

    /// Whether messages are enabled.
    pub fn messages_enabled(&self) -> bool {
        self.msgs.is_some()
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
            None => Err(Error::Pop(span!())),
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
            for (i, inputs, output) in function.nodes.iter_range(0..function.nodes.len(), true) {
                assert_eq!(inputs.len(), function.schema.input.len());
                for (input, sort) in inputs.iter().zip(&function.schema.input) {
                    assert_eq!(
                        input,
                        &self.find(sort, *input),
                        "[{i}] {name}({inputs:?}) = {output:?}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output.value,
                    self.find(&function.schema.output, output.value),
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
    pub fn find(&self, sort: &ArcSort, value: Value) -> Value {
        if sort.is_eq_sort() {
            Value {
                #[cfg(debug_assertions)]
                tag: value.tag,
                bits: self.unionfind.find(value.bits),
            }
        } else {
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
        let merge_prog = match &function.merge {
            MergeFn::Expr(e) => Some(e.clone()),
            MergeFn::AssertEq | MergeFn::Union => None,
        };

        for (inputs, old, new) in merges {
            if let Some(prog) = &merge_prog {
                // TODO: error handling?
                self.run_actions(&mut stack, &[*old, *new], prog).unwrap();
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
            Literal::Int(i) => i.store(&I64Sort),
            Literal::Float(f) => f.store(&F64Sort),
            Literal::String(s) => s.store(&StringSort),
            Literal::Unit => ().store(&UnitSort),
            Literal::Bool(b) => b.store(&BoolSort),
        }
    }

    pub fn function_to_dag(
        &self,
        sym: Symbol,
        n: usize,
    ) -> Result<(Vec<(Term, Term)>, TermDag), Error> {
        if true {
            todo!("function_to_dag")
        }

        let f = self
            .functions
            .get(&sym)
            .ok_or(TypeError::UnboundFunction(sym, span!()))?;
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
                    children.push(
                        a_type
                            .extract_term(self, a, &extractor, &mut termdag)
                            .unwrap()
                            .1,
                    )
                };
            }

            let out = if schema.output.is_eq_sort() {
                extractor
                    .find_best(out.value, &mut termdag, &schema.output)
                    .unwrap()
                    .1
            } else {
                schema
                    .output
                    .extract_term(self, out.value, &extractor, &mut termdag)
                    .unwrap()
                    .1
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
        let out_is_unit = f.schema.output.name() == UnitSort.name();

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
                .ok_or(TypeError::UnboundFunction(sym, span!()))?;
            log::info!("Function {} has size {}", sym, f.nodes.len());
            assert_eq!(f.nodes.len(), self.backend.table_size(f.new_backend_id));
            self.print_msg(f.nodes.len().to_string());
            Ok(())
        } else {
            // Print size of all functions
            let mut lens = self
                .functions
                .iter()
                .map(|(sym, f)| {
                    assert_eq!(f.nodes.len(), self.backend.table_size(f.new_backend_id));
                    (*sym, f.nodes.len())
                })
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

    /// Extract a value to a [`TermDag`] and [`Term`] in the [`TermDag`].
    /// Note that the `TermDag` may contain a superset of the nodes in the `Term`.
    /// See also `extract_value_to_string` for convenience.
    pub fn extract_value(&self, sort: &ArcSort, value: Value) -> Result<(TermDag, Term), Error> {
        let mut termdag = TermDag::default();
        let term = self.extract(value, &mut termdag, sort)?.1;
        Ok((termdag, term))
    }

    /// Extract a value to a string for printing.
    /// See also `extract_value` for more control.
    pub fn extract_value_to_string(&self, sort: &ArcSort, value: Value) -> Result<String, Error> {
        let (termdag, term) = self.extract_value(sort, value)?;
        Ok(termdag.to_string(&term))
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

                for (rule_name, (rule, _)) in copy_rules.iter() {
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
                    let (rule, _) = compiled_rules.get(&rule_name).unwrap();
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
                            self.run_actions(stack, &[], &rule.program)
                                .unwrap_or_else(|e| {
                                    panic!("error while running actions for {rule_name}: {e}")
                                });
                        }
                    } else {
                        for values in all_matches.chunks(num_vars) {
                            stack.clear();
                            self.run_actions(stack, values, &rule.program)
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
        let old_updated = self.did_change_tables() || n_unions_before != self.unionfind.n_unions();
        run_report.updated |= old_updated;

        {
            fn collect_rule_ids(
                ruleset: Symbol,
                rulesets: &IndexMap<Symbol, Ruleset>,
                ids: &mut Vec<egglog_bridge::RuleId>,
            ) {
                match &rulesets[&ruleset] {
                    Ruleset::Rules(_, rules) => {
                        for (_, id) in rules.values() {
                            ids.push(*id);
                        }
                    }
                    Ruleset::Combined(_, sub_rulesets) => {
                        for sub_ruleset in sub_rulesets {
                            collect_rule_ids(*sub_ruleset, rulesets, ids);
                        }
                    }
                }
            }

            let mut rule_ids = Vec::new();
            collect_rule_ids(ruleset, &self.rulesets, &mut rule_ids);
            let new_updated = self.backend.run_rules(&rule_ids).unwrap();
            assert_eq!(old_updated, new_updated);
        }

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
        name: Symbol,
        rule: ast::ResolvedRule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
        let core_rule =
            rule.to_canonicalized_core_rule(&self.type_info, &mut self.parser.symbol_gen)?;
        let (query, actions) = (core_rule.body, core_rule.head);

        let rule_id = {
            let mut translator = BackendRule::new(
                self.backend.new_rule(name.into(), self.seminaive),
                &self.functions,
                &self.type_info,
            );
            translator.query(&query, false);
            translator.actions(&actions)?;
            translator.build()
        };

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
                        indexmap::map::Entry::Vacant(e) => e.insert((compiled_rule, rule_id)),
                    };
                    Ok(name)
                }
                Ruleset::Combined(_, _) => Err(Error::CombinedRulesetError(ruleset, rule.span)),
            }
        } else {
            Err(Error::NoSuchRuleset(ruleset, rule.span))
        }
    }

    fn eval_actions(&mut self, actions: &ResolvedActions) -> Result<(), Error> {
        let (actions, _) = actions.to_core_actions(
            &self.type_info,
            &mut Default::default(),
            &mut self.parser.symbol_gen,
        )?;

        let new_result = {
            let mut translator = BackendRule::new(
                self.backend.new_rule("eval_actions", false),
                &self.functions,
                &self.type_info,
            );
            translator.actions(&actions)?;
            let id = translator.build();
            let result = self.backend.run_rules(&[id]);
            self.backend.free_rule(id);
            result
        };

        let program = self
            .compile_actions(&Default::default(), &actions)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        let old_result = self.run_actions(&mut stack, &[], &program);

        match (old_result, new_result) {
            (Ok(()), Ok(_)) => Ok(()),
            (Err(e), Err(_)) => Err(e),
            (old, new) => panic!("backends did not match:\nold={old:?}\nnew={new:?}"),
        }
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, core_relations::Value), Error> {
        let fresh_name = self.parser.symbol_gen.fresh(&"egraph_evalexpr".into());
        let command = Command::Action(Action::Let(expr.span(), fresh_name, expr.clone()));
        self.run_program(vec![command])?;
        // find the table with the same name as the fresh name
        let func = self.functions.get(&fresh_name).unwrap();
        let value_new_backend = self.backend.lookup_id(func.new_backend_id, &[]).unwrap();

        let value = func.nodes.get(&[]).unwrap().value;

        let sort = func.schema.output.clone();

        // to be removed once the old backend is gone
        if sort.name().as_str() == "i64" {
            let value_new_backend: &i64 = &self.backend.primitives().unwrap_ref(value_new_backend);
            assert_eq!(value_new_backend, &(value.bits as i64));
        } else if sort.name().as_str() == "String" {
            let symbol_new: &Symbol = &self.backend.primitives().unwrap_ref(value_new_backend);
            let string_sort = self.get_sort::<StringSort>();
            let symbol_old = &Symbol::load(&string_sort, &value);
            assert_eq!(symbol_new, symbol_old);
        }

        Ok((sort, value_new_backend))
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
            "interactive_mode" => {
                if let ResolvedExpr::Lit(_ann, Literal::Int(i)) = value {
                    self.interactive_mode = i != 0;
                } else {
                    panic!("interactive_mode must be an integer");
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
        let core_rule =
            rule.to_canonicalized_core_rule(&self.type_info, &mut self.parser.symbol_gen)?;
        let query = core_rule.body;

        let new_matched = {
            let ext_sc = egglog_bridge::SideChannel::default();
            let ext_sc_ref = ext_sc.clone();
            let ext_id = self
                .backend
                .register_external_func(core_relations::make_external_func(move |_, _| {
                    *ext_sc_ref.lock().unwrap() = Some(());
                    Some(core_relations::Value::new_const(0))
                }));

            let mut translator = BackendRule::new(
                self.backend.new_rule("check_facts", false),
                &self.functions,
                &self.type_info,
            );
            translator.query(&query, true);
            translator.rb.call_external_func(
                ext_id,
                &[],
                egglog_bridge::ColumnTy::Id,
                "this function will never panic",
            );
            let id = translator.build();
            let _ = self.backend.run_rules(&[id]).unwrap();
            self.backend.free_rule(id);

            self.backend.free_external_func(ext_id);

            let ext_sc_val = ext_sc.lock().unwrap().take();
            matches!(ext_sc_val, Some(()))
        };

        let ordering = &query.get_vars();
        let query = self.compile_gj_query(query, ordering);

        let mut matched = false;
        self.run_query(&query, 0, true, |values| {
            assert_eq!(values.len(), query.vars.len());
            matched = true;
            Err(())
        });

        assert_eq!(matched, new_matched);

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
                self.add_rule_with_name(name, rule, ruleset)?;
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
                    .append(true)
                    .create(true)
                    .open(&filename)
                    .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
    
                let unit_id = self.backend.primitives().get_ty::<()>();
                let unit_val = self.backend.primitives().get(());

                let results: Arc<Mutex<Vec<core_relations::Value>>> = Default::default();
                let results_ref = results.clone();
                let ext_id =
                    self.backend
                        .register_external_func(core_relations::make_external_func(
                            move |_es, vals| {
                                debug_assert!(vals.len() == 1);
                                results_ref.lock().unwrap().push(vals[0]);
                                Some(unit_val)
                            },
                        ));


                let mut translator = BackendRule::new(
                    self.backend.new_rule("outputs", false),
                    &self.functions,
                    &self.type_info,
                );
                let expr_types = exprs.iter().map(|e| e.output_type()).collect::<Vec<_>>();
                for expr in exprs {
                    let result_var = ResolvedVar {
                        name: self.parser.symbol_gen.fresh(&Symbol::from("__egglog_output")),
                        sort: expr.output_type(),
                        is_global_ref: false,
                    };
                    let actions = ResolvedActions::singleton(ResolvedAction::Let(
                        span.clone(),
                        result_var.clone(),
                        expr,
                    ));
                    let actions = actions
                        .to_core_actions(
                            &self.type_info,
                            &mut Default::default(),
                            &mut self.parser.symbol_gen,
                        )?
                        .0;
                    translator.actions(&actions)?;
                    let arg = translator.entry(&ResolvedAtomTerm::Var(span.clone(), result_var));
                    translator.rb.call_external_func(
                        ext_id,
                        &[arg],
                        egglog_bridge::ColumnTy::Primitive(unit_id),
                        "this function will never panic",
                    );
                }
                let id = translator.build();
                let _ = self.backend.run_rules(&[id]).unwrap();
                self.backend.free_rule(id);
                self.backend.free_external_func(ext_id);

                let results: Vec<_> = std::mem::take(results.lock().unwrap().as_mut());
                use std::io::Write;
                for (value, expr_type) in results.into_iter().zip(expr_types) {
                    // hack before extraction for the new backend gets merged:
                    if !expr_type.is_container_sort() && !expr_type.is_eq_sort() {
                        let primitive_id = self
                            .backend
                            .primitives()
                            .get_ty_by_id(expr_type.value_type().unwrap());
                        let formatted_val = PrimitivePrinter {
                            prim: self.backend.primitives(),
                            ty: primitive_id,
                            val: value,
                        };
                        writeln!(f, "{:?}", formatted_val)
                            .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?
                    } else {
                        todo!("handle the general case with extract")
                        // let term = self.extract(value, &mut termdag, &expr_type)?.1;
                        // use std::io::Write;
                        // writeln!(f, "{}", termdag.to_string(&term))
                        //     .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                    }
                }

                log::info!("Output to '{filename:?}'.")
            }
        };
        Ok(())
    }

    fn input_file(&mut self, func_name: Symbol, file: String) -> Result<(), Error> {
        let function_type = self
            .type_info
            .get_func_type(&func_name)
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

        if function_type.subtype != FunctionSubtype::Constructor {
            match func.schema.output.name().as_str() {
                "i64" | "String" | "Unit" => {}
                s => panic!("Unsupported type {} for input", s),
            }
        }

        log::info!("Opening file '{:?}'...", filename);
        let mut f = File::open(filename).unwrap();
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();

        let span: Span = span!();
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
                        Ok(f) => Expr::Lit(span.clone(), Literal::Float(f.into())),
                        Err(_) => Expr::Lit(span.clone(), Literal::String(s.into())),
                    },
                }
            };

            let mut exprs: Vec<Expr> = str_buf.iter().map(|&s| parse(s)).collect();

            actions.push(
                if function_type.subtype == FunctionSubtype::Constructor
                    || function_type.subtype == FunctionSubtype::Relation
                {
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
        let commands: Vec<_> = self.typecheck_program(&commands)?;
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
            !self.parser.symbol_gen.has_been_used(),
            "Reserved symbol must be set before any symbols are generated"
        );
        self.parser.symbol_gen = SymbolGen::new(sym.to_string());
    }

    fn process_command(&mut self, command: Command) -> Result<Vec<ResolvedNCommand>, Error> {
        let program = desugar::desugar_program(vec![command], &mut self.parser, self.seminaive)?;

        let program = self.typecheck_program(&program)?;

        let program = remove_globals(program, &mut self.parser.symbol_gen);

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

                let result = self.run_command(processed);

                if self.is_interactive_mode() {
                    self.print_msg(match result {
                        Ok(()) => "(done)".into(),
                        Err(_) => "(error)".into(),
                    });
                }

                result?
            }
        }
        log::logger().flush();

        Ok(self.flush_msgs())
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
        let parsed = self.parser.get_program_from_string(filename, input)?;
        self.run_program(parsed)
    }

    pub fn num_tuples(&self) -> usize {
        self.functions.values().map(|f| f.nodes.len()).sum()
    }

    /// Returns a sort based on the type.
    pub fn get_sort<S: Sort + Send + Sync>(&self) -> Arc<S> {
        self.type_info.get_sort()
    }

    /// Returns a sort that satisfies the type and predicate.
    pub fn get_sort_by<S: Sort + Send + Sync>(&self, f: impl Fn(&Arc<S>) -> bool) -> Arc<S> {
        self.type_info.get_sort_by(f)
    }

    /// Returns all sorts based on the type.
    pub fn get_sorts<S: Sort + Send + Sync>(&self) -> Vec<Arc<S>> {
        self.type_info.get_sorts()
    }

    /// Returns all sorts that satisfy the type and predicate.
    pub fn get_sorts_by<S: Sort + Send + Sync>(&self, f: impl Fn(&Arc<S>) -> bool) -> Vec<Arc<S>> {
        self.type_info.get_sorts_by(f)
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

    pub(crate) fn print_msg(&mut self, msg: String) {
        if let Some(ref mut msgs) = self.msgs {
            msgs.push(msg);
        }
    }

    fn flush_msgs(&mut self) -> Vec<String> {
        if let Some(ref mut msgs) = self.msgs {
            msgs.dedup_by(|a, b| a.is_empty() && b.is_empty());
            std::mem::take(msgs)
        } else {
            vec![]
        }
    }
}

struct BackendRule<'a> {
    pub rb: egglog_bridge::RuleBuilder<'a>,
    entries: HashMap<core::ResolvedAtomTerm, QueryEntry>,
    functions: &'a IndexMap<Symbol, Function>,
    type_info: &'a TypeInfo,
}

impl<'a> BackendRule<'a> {
    fn new(
        rb: egglog_bridge::RuleBuilder<'a>,
        functions: &'a IndexMap<Symbol, Function>,
        type_info: &'a TypeInfo,
    ) -> BackendRule<'a> {
        BackendRule {
            rb,
            functions,
            type_info,
            entries: Default::default(),
        }
    }

    fn entry(&mut self, x: &core::ResolvedAtomTerm) -> QueryEntry {
        self.entries
            .entry(x.clone())
            .or_insert_with(|| match x {
                core::GenericAtomTerm::Var(_, v) => self
                    .rb
                    .new_var_named(v.sort.column_ty(self.rb.egraph()), v.name.into()),
                core::GenericAtomTerm::Literal(_, l) => literal_to_entry(self.rb.egraph(), l),
                core::GenericAtomTerm::Global(..) => {
                    panic!("Globals should have been desugared")
                }
            })
            .clone()
    }

    fn func(&self, f: &typechecking::FuncType) -> egglog_bridge::FunctionId {
        self.functions[&f.name].new_backend_id
    }

    fn prim(
        &mut self,
        prim: &core::SpecializedPrimitive,
        args: &[core::ResolvedAtomTerm],
    ) -> (ExternalFunctionId, Vec<QueryEntry>, ColumnTy) {
        let mut qe_args = self.args(args);

        if prim.primitive.0.name() == "unstable-fn".into() {
            let core::ResolvedAtomTerm::Literal(_, Literal::String(name)) = args[0] else {
                panic!("expected string literal after `unstable-fn`")
            };
            let id = if let Some(f) = self.type_info.get_func_type(&name) {
                ResolvedFunctionId::Lookup(egglog_bridge::Lookup::new(
                    self.rb.egraph(),
                    self.func(f),
                ))
            } else if let Some(possible) = self.type_info.get_prims(&name) {
                let mut ps: Vec<_> = possible.iter().collect();
                ps.retain(|p| {
                    self.type_info
                        .get_sorts::<FunctionSort>()
                        .into_iter()
                        .any(|f| {
                            let types: Vec<_> = prim
                                .input
                                .iter()
                                .skip(1)
                                .chain(f.inputs())
                                .chain([&f.output()])
                                .cloned()
                                .collect();
                            p.accept(&types, self.type_info)
                        })
                });
                assert!(ps.len() == 1, "options for {name}: {ps:?}");
                ResolvedFunctionId::Prim(ps.into_iter().next().unwrap().1)
            } else {
                panic!("no callable for {name}");
            };
            let do_rebuild = prim
                .input
                .iter()
                .skip(1)
                .map(|s| s.is_eq_sort() || s.is_eq_container_sort())
                .collect();

            qe_args[0] = self
                .rb
                .egraph()
                .primitive_constant(ResolvedFunction { id, do_rebuild });
        }

        (
            prim.primitive.1,
            qe_args,
            prim.output.column_ty(self.rb.egraph()),
        )
    }

    fn args<'b>(
        &mut self,
        args: impl IntoIterator<Item = &'b core::ResolvedAtomTerm>,
    ) -> Vec<QueryEntry> {
        args.into_iter().map(|x| self.entry(x)).collect()
    }

    fn query(&mut self, query: &core::Query<ResolvedCall, ResolvedVar>, include_subsumed: bool) {
        for atom in &query.atoms {
            match &atom.head {
                ResolvedCall::Func(f) => {
                    let f = self.func(f);
                    let args = self.args(&atom.args);
                    let is_subsumed = match include_subsumed {
                        true => None,
                        false => Some(false),
                    };
                    self.rb.query_table(f, &args, is_subsumed).unwrap()
                }
                ResolvedCall::Primitive(p) => {
                    let (p, args, ty) = self.prim(p, &atom.args);
                    self.rb.query_prim(p, &args, ty).unwrap()
                }
            }
        }
    }

    fn actions(&mut self, actions: &core::ResolvedCoreActions) -> Result<(), Error> {
        for action in &actions.0 {
            match action {
                core::GenericCoreAction::Let(span, v, f, args) => {
                    let v = core::GenericAtomTerm::Var(span.clone(), v.clone());
                    let y = match f {
                        ResolvedCall::Func(f) => {
                            let name = f.name;
                            let f = self.func(f);
                            let args = self.args(args);
                            self.rb
                                .lookup(f, &args, || {
                                    format!("{span}: lookup of function {name} failed")
                                })
                                .into()
                        }
                        ResolvedCall::Primitive(p) => {
                            let name = p.primitive.0.name();
                            let (p, args, ty) = self.prim(p, args);
                            self.rb
                                .call_external_func(
                                    p,
                                    &args,
                                    ty,
                                    format!("{span}: call of primitive {name} failed").as_str(),
                                )
                                .into()
                        }
                    };
                    self.entries.insert(v, y);
                }
                core::GenericCoreAction::LetAtomTerm(span, v, x) => {
                    let v = core::GenericAtomTerm::Var(span.clone(), v.clone());
                    let x = self.entry(x);
                    self.entries.insert(v, x);
                }
                core::GenericCoreAction::Set(_, f, xs, y) => match f {
                    ResolvedCall::Primitive(..) => panic!("runtime primitive set!"),
                    ResolvedCall::Func(f) => {
                        let f = self.func(f);
                        let args = self.args(xs.iter().chain([y]));
                        self.rb.set(f, &args)
                    }
                },
                core::GenericCoreAction::Change(_, change, f, args) => match f {
                    ResolvedCall::Primitive(..) => panic!("runtime primitive change!"),
                    ResolvedCall::Func(f) => {
                        let name = f.name;
                        let can_subsume = self.functions[&f.name].can_subsume;
                        let f = self.func(f);
                        let args = self.args(args);
                        match change {
                            Change::Delete => self.rb.remove(f, &args),
                            Change::Subsume if can_subsume => self.rb.subsume(f, &args),
                            Change::Subsume => return Err(Error::SubsumeMergeError(name)),
                        }
                    }
                },
                core::GenericCoreAction::Union(_, x, y) => {
                    let x = self.entry(x);
                    let y = self.entry(y);
                    self.rb.union(x, y)
                }
                core::GenericCoreAction::Panic(_, message) => self.rb.panic(message.clone()),
                core::GenericCoreAction::Extract(_, _x, _n) => todo!("no extraction yet"),
            }
        }
        Ok(())
    }

    fn build(self) -> egglog_bridge::RuleId {
        self.rb.build()
    }
}

fn literal_to_entry(egraph: &egglog_bridge::EGraph, l: &Literal) -> QueryEntry {
    match l {
        Literal::Int(x) => egraph.primitive_constant::<i64>(*x),
        Literal::Float(x) => egraph.primitive_constant::<sort::F>(*x),
        Literal::String(x) => egraph.primitive_constant::<sort::S>(*x),
        Literal::Bool(x) => egraph.primitive_constant::<bool>(*x),
        Literal::Unit => egraph.primitive_constant::<()>(()),
    }
}

fn literal_to_value(egraph: &egglog_bridge::EGraph, l: &Literal) -> core_relations::Value {
    match l {
        Literal::Int(x) => egraph.primitives().get::<i64>(*x),
        Literal::Float(x) => egraph.primitives().get::<sort::F>(*x),
        Literal::String(x) => egraph.primitives().get::<sort::S>(*x),
        Literal::Bool(x) => egraph.primitives().get::<bool>(*x),
        Literal::Unit => egraph.primitives().get::<()>(()),
    }
}

// Currently, only the following errors can thrown without location information:
// * PrimitiveError
// * MergeError
// * SubsumeMergeError
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] ParseError),
    #[error(transparent)]
    NotFoundError(#[from] NotFoundError),
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error("Errors:\n{}", ListDisplay(.0, "\n"))]
    TypeErrors(Vec<TypeError>),
    #[error("{1}\nCheck failed: \n{}", ListDisplay(.0, "\n"))]
    CheckError(Vec<Fact>, Span),
    #[error("{1}\nNo such ruleset: {0}")]
    NoSuchRuleset(Symbol, Span),
    #[error("{1}\nAttempted to add a rule to combined ruleset {0}. Combined rulesets may only depend on other rulesets.")]
    CombinedRulesetError(Symbol, Span),
    #[error("Evaluating primitive {0:?} failed. ({0:?} {:?})", ListDebug(.1, " "))]
    PrimitiveError(Primitive, Vec<Value>),
    #[error("Illegal merge attempted for function {0}, {1:?} != {2:?}")]
    MergeError(Symbol, Value, Value),
    #[error("{0}\nTried to pop too much")]
    Pop(Span),
    #[error("{0}\nCommand should have failed.")]
    ExpectFail(Span),
    #[error("{2}\nIO error: {0}: {1}")]
    IoError(PathBuf, std::io::Error, Span),
    #[error("Cannot subsume function with merge: {0}")]
    SubsumeMergeError(Symbol),
    #[error("extraction failure: {:?}", .0)]
    ExtractError(Value),
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use lazy_static::lazy_static;

    use crate::constraint::SimpleTypeConstraint;
    use crate::sort::*;
    use crate::*;

    #[derive(Clone)]
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
            values: &[Value],
            _sorts: (&[ArcSort], &ArcSort),
            _egraph: Option<&mut EGraph>,
        ) -> Option<Value> {
            let mut sum = 0;
            let vec1 = VecContainer::load(&self.vec, &values[0]);
            let vec2 = VecContainer::load(&self.vec, &values[1]);
            assert_eq!(vec1.data.len(), vec2.data.len());
            for (a, b) in vec1.data.iter().zip(vec2.data.iter()) {
                let a = i64::load(&self.ele, a);
                let b = i64::load(&self.ele, b);
                sum += a * b;
            }
            Some(sum.store(&self.ele))
        }
    }

    impl ExternalFunction for InnerProduct {
        fn invoke(
            &self,
            exec_state: &mut core_relations::ExecutionState<'_>,
            args: &[core_relations::Value],
        ) -> Option<core_relations::Value> {
            let mut sum = 0;
            let vec1 = exec_state
                .containers()
                .get_val::<VecContainer<_>>(args[0])
                .unwrap();
            let vec2 = exec_state
                .containers()
                .get_val::<VecContainer<_>>(args[1])
                .unwrap();
            assert_eq!(vec1.data.len(), vec2.data.len());
            for (a, b) in vec1.data.iter().zip(vec2.data.iter()) {
                let a = exec_state.prims().unwrap::<i64>(*a);
                let b = exec_state.prims().unwrap::<i64>(*b);
                sum += a * b;
            }
            Some(exec_state.prims().get::<i64>(sum))
        }
    }

    #[test]
    fn test_user_defined_primitive() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(None, "(sort IntVec (Vec i64))")
            .unwrap();

        let int_vec_sort =
            egraph.get_sort_by(|s: &Arc<VecSort>| s.element().name() == I64Sort.name());

        egraph.add_primitive(InnerProduct {
            ele: I64Sort.into(),
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

    lazy_static! {
        pub static ref RT: Mutex<EGraph> = Mutex::new(EGraph::default());
    }
}
