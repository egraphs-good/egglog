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
pub mod ast;
pub mod constraint;
mod extract;
mod function;
mod gj;
mod serialize;
pub mod sort;
mod termdag;
// mod terms;
mod typecheck;
mod typechecking;
mod unionfind;
pub mod util;
mod value;

use ast::desugar::Desugar;
use extract::Extractor;
use hashbrown::hash_map::Entry;
use index::ColumnIndex;
use instant::{Duration, Instant};
pub use serialize::SerializeConfig;
use sort::*;
pub use termdag::{Term, TermDag, TermId};
// use terms::TermState;
use thiserror::Error;

use generic_symbolic_expressions::Sexp;

use ast::*;
pub use typechecking::{TypeInfo, UNIT_SYM};

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
use typecheck::{AtomTerm, Program, Query, ResolvedCall, ResolvedCoreRule};

pub type ArcSort = Arc<dyn Sort>;

pub use value::*;

use function::*;
use gj::*;
use unionfind::*;
use util::*;

use crate::constraint::Problem;
use crate::typechecking::TypeError;

pub type Subst = IndexMap<Symbol, Value>;

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    fn get_type_constraints(&self) -> Box<dyn TypeConstraint>;
    fn apply(&self, values: &[Value], egraph: &EGraph) -> Option<Value>;
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
    // TODO: this is temporary and should be removed once refactoring is done
    fn accept(&self, tys: &[Arc<dyn Sort>]) -> Option<Arc<dyn Sort>> {
        let mut constraints = vec![];
        // + 1 to account for the output type
        let lits: Vec<_> = (0..tys.len() + 1)
            .map(|i| AtomTerm::Literal(Literal::Int(i as i64)))
            .collect();
        for (lit, ty) in lits.iter().zip(tys.iter()) {
            constraints.push(Constraint::Assign(lit.clone(), ty.clone()))
        }
        constraints.extend(self.get_type_constraints().get(&lits));
        let output_type = AtomTerm::Literal(Literal::Int(tys.len() as i64));
        let problem = Problem {
            constraints,
            range: HashSet::from_iter([output_type.clone()]),
        };
        let assignment = problem.solve(|sort| sort.name()).ok()?;
        Some(assignment.get(&output_type).unwrap().clone())
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

    fn get_type_constraints(&self) -> Box<dyn TypeConstraint> {
        let sorts: Vec<_> = self
            .input
            .iter()
            .chain(once(&self.output as &ArcSort))
            .cloned()
            .collect();
        SimpleTypeConstraint::new(self.name(), sorts).into_box()
    }
    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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
    pub(crate) desugar: Desugar,
    functions: HashMap<Symbol, Function>,
    rulesets: HashMap<Symbol, HashMap<Symbol, Rule>>,
    ruleset_iteration: HashMap<Symbol, usize>,
    proofs_enabled: bool,
    terms_enabled: bool,
    interactive_mode: bool,
    timestamp: u32,
    pub(crate) term_header_added: bool,
    pub test_proofs: bool,
    pub match_limit: usize,
    pub node_limit: usize,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
    // sort, value, and timestamp
    pub global_bindings: HashMap<Symbol, (ArcSort, Value, u32)>,
    extract_report: Option<ExtractReport>,
    /// The run report for the most recent run of a schedule.
    recent_run_report: Option<RunReport>,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,
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
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            egraphs: vec![],
            unionfind: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            ruleset_iteration: Default::default(),
            desugar: Desugar::default(),
            global_bindings: Default::default(),
            match_limit: usize::MAX,
            node_limit: usize::MAX,
            timestamp: 0,
            proofs_enabled: false,
            terms_enabled: false,
            term_header_added: false,
            interactive_mode: false,
            test_proofs: false,
            fact_directory: None,
            seminaive: true,
            extract_report: None,
            recent_run_report: None,
            overall_run_report: Default::default(),
            msgs: Default::default(),
        };
        egraph.rulesets.insert("".into(), Default::default());
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(UnresolvedExpr);

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
        // we can't use find before something
        // is added to the parent table, so this
        // is disabled in terms mode
        if self.terms_enabled {
            return;
        }
        #[cfg(debug_assertions)]
        for (name, function) in self.functions.iter() {
            function.nodes.assert_sorted();
            for (i, inputs, output) in function.nodes.iter_range(0..function.nodes.len()) {
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
            let parent_name = self.desugar.parent_name(value.tag);
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

    pub fn print_size(&mut self, sym: Option<Symbol>) -> Result<(), Error> {
        if let Some(sym) = sym {
            let f = self.functions.get(&sym).ok_or(TypeError::Unbound(sym))?;
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
            Schedule::Run(config) => self.run_rules(config),
            Schedule::Repeat(limit, sched) => {
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
            Schedule::Saturate(sched) => {
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
            Schedule::Sequence(scheds) => {
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

    fn run_rules(&mut self, config: &ResolvedRunConfig) -> RunReport {
        let mut report: RunReport = Default::default();

        // first rebuild
        let rebuild_start = Instant::now();
        let updates = self.rebuild_nofail();
        log::debug!("database size: {}", self.num_tuples());
        log::debug!("Made {updates} updates");
        // add to the rebuild time for this ruleset
        *report
            .rebuild_time_per_ruleset
            .entry(config.ruleset)
            .or_default() += rebuild_start.elapsed();
        self.timestamp += 1;

        let RunConfig { ruleset, until } = config;

        if let Some(facts) = until {
            if self.check_facts(facts).is_ok() {
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
        // add to the ruleset searched time
        *report
            .search_time_per_ruleset
            .entry(ruleset)
            .or_insert(Duration::default()) += search_elapsed;

        let apply_start = Instant::now();
        for (name, all_values, search_time) in searched {
            let rule = rules.get_mut(name).unwrap();
            // add to the rule's search time
            *report
                .search_time_per_rule
                .entry(*name)
                .or_insert(Duration::default()) += search_time;
            let num_vars = rule.query.vars.len();

            // make sure the query requires matches
            if num_vars != 0 {
                *report.num_matches_per_rule.entry(*name).or_insert(0) +=
                    all_values.len() / num_vars;

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
                stack.clear();
                self.run_actions(stack, &[], &rule.program, true)
                    .unwrap_or_else(|e| panic!("error while running actions for {name}: {e}"));
            } else {
                for values in all_values.chunks(num_vars) {
                    rule.matches += 1;
                    stack.clear();
                    self.run_actions(stack, values, &rule.program, true)
                        .unwrap_or_else(|e| panic!("error while running actions for {name}: {e}"));
                }
            }

            // add to the rule's apply time
            *report
                .apply_time_per_rule
                .entry(*name)
                .or_insert(Duration::default()) += rule_apply_start.elapsed();
        }
        self.rulesets.insert(ruleset, rules);
        let apply_elapsed = apply_start.elapsed();
        // add to the apply time for the ruleset
        *report
            .apply_time_per_ruleset
            .entry(ruleset)
            .or_insert(Duration::default()) += apply_elapsed;
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
        // TODO: should probably use ResolvedRule here
        rule: ast::ResolvedRule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let mut compiler = typecheck::Context::new(self);
        // let core_rule = compiler.compile_rule(&rule).map_err(Error::TypeErrors)?;
        let core_rule: ResolvedCoreRule = todo!("get the resolved core rule");
        let (query, action) = (core_rule.body, core_rule.head);
        let query = todo!("unresolve variables {:?}", query);

        // TODO: We should refactor compile_actions later as well
        let action: Vec<NormAction> = action.0.clone();
        let types = todo!("get types from type inference");

        let query = self.compile_gj_query(query);
        let program = self
            .compile_actions(types, &action)
            .map_err(Error::TypeErrors)?;
        let compiled_rule = Rule {
            query,
            matches: 0,
            times_banned: 0,
            banned_until: 0,
            todo_timestamp: 0,
            program,
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

    pub(crate) fn add_rule(
        &mut self,
        rule: ast::ResolvedRule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
        let name = format!("{}", rule);
        self.add_rule_with_name(name, rule, ruleset)
    }

    fn eval_actions(&mut self, actions: &[ResolvedAction]) -> Result<(), Error> {
        let types = Default::default();
        let actions: Vec<NormAction> =
            todo!("need to first lower to norm action before compilation");
        let program = self
            .compile_actions(&types, &actions)
            .map_err(Error::TypeErrors)?;
        let mut stack = vec![];
        self.run_actions(&mut stack, &[], &program, true)?;
        Ok(())
    }

    // TODO make a public version of eval_expr that makes a command,
    // then returns the value at the end.
    fn eval_expr(
        &mut self,
        expr: &ResolvedExpr,
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

    fn set_option(&mut self, name: &str, value: ResolvedExpr) {
        match name {
            "enable_proofs" => {
                self.proofs_enabled = true;
            }
            "interactive_mode" => {
                if let Expr::Lit(_ann, Literal::Int(i)) = value {
                    self.interactive_mode = i != 0;
                } else {
                    panic!("interactive_mode must be an integer");
                }
            }
            "match_limit" => {
                if let Expr::Lit(_ann, Literal::Int(i)) = value {
                    self.match_limit = i as usize;
                } else {
                    panic!("match_limit must be an integer");
                }
            }
            "node_limit" => {
                if let Expr::Lit(_ann, Literal::Int(i)) = value {
                    self.node_limit = i as usize;
                } else {
                    panic!("node_limit must be an integer");
                }
            }
            _ => panic!("Unknown option '{}'", name),
        }
    }

    fn check_facts(&mut self, facts: &[ResolvedFact]) -> Result<(), Error> {
        let mut compiler = typecheck::Context::new(self);
        let rule = ast::ResolvedRule {
            head: vec![],
            body: facts.to_vec(),
        };
        // let rule = compiler.compile_rule(&rule).map_err(Error::TypeErrors)?;
        let rule: ResolvedCoreRule = todo!("compile rule");
        let query0 = rule.body;
        let query0: Query<ResolvedCall, Symbol> =
            todo!("strip away the annotation on symbols {:?}", query0);
        let query = self.compile_gj_query(query0);

        let mut matched = false;
        // TODO what timestamp to use?
        self.run_query(&query, 0, |values| {
            assert_eq!(values.len(), query.vars.len());
            matched = true;
            Err(())
        });
        if !matched {
            // TODO add useful info here
            Err(Error::CheckError(
                facts.iter().map(|f| f.to_unresolved()).collect(),
            ))
        } else {
            Ok(())
        }
    }

    fn run_command(&mut self, command: ResolvedNCommand, should_run: bool) -> Result<(), Error> {
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
                self.add_rule(rule, ruleset)?;
                log::info!("Declared rule {name}.")
            }
            NCommand::RunSchedule(sched) => {
                if should_run {
                    let report = self.run_schedule(&sched);
                    log::info!("Ran schedule {}.", sched);
                    log::info!("Report: {}", report);
                    self.overall_run_report = self.overall_run_report.union(&report);
                    self.recent_run_report = Some(report);
                } else {
                    log::warn!("Skipping schedule.")
                }
            }
            NCommand::PrintOverallStatistics => {
                log::info!("Overall statistics:\n{}", self.overall_run_report);
                self.print_msg(format!("Overall statistics:\n{}", self.overall_run_report));
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
                        Action::Let((), name, contents) => {
                            let (etype, value) = self.eval_expr(&contents, None, true)?;
                            let present = self
                                .global_bindings
                                .insert(name.name.clone(), (etype, value, self.timestamp));
                            if present.is_some() {
                                panic!("Variable {name} was already present in global bindings");
                            }
                        }
                        _ => {
                            self.eval_actions(std::slice::from_ref(&action))?;
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
                self.input_file(name, file)?;
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
                    let term = self.extract(value, &mut termdag, &t).1;
                    use std::io::Write;
                    writeln!(f, "{}", termdag.to_string(&term))
                        .map_err(|e| Error::IoError(filename.clone(), e))?;
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
                "i64" | "String" => {}
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

        let mut actions: Vec<UnresolvedAction> = vec![];
        let mut str_buf: Vec<&str> = vec![];
        for line in contents.lines() {
            str_buf.clear();
            str_buf.extend(line.split('\t').map(|s| s.trim()));
            if str_buf.is_empty() {
                continue;
            }

            let parse = |s: &str| -> UnresolvedExpr {
                if let Ok(i) = s.parse() {
                    Expr::Lit((), Literal::Int(i))
                } else {
                    Expr::Lit((), Literal::String(s.into()))
                }
            };

            let mut exprs: Vec<UnresolvedExpr> = str_buf.iter().map(|&s| parse(s)).collect();

            actions.push(
                if function_type.is_datatype || function_type.output.name() == UNIT_SYM.into() {
                    Action::Expr((), Expr::Call((), func_name, exprs))
                } else {
                    let out = exprs.pop().unwrap();
                    Action::Set((), func_name, exprs, out)
                },
            );
        }
        let actions: Vec<ResolvedAction> = todo!("action resolution needed {:?}", actions);
        self.eval_actions(&actions)?;
        log::info!(
            "Read {} facts into {func_name} from '{file}'.",
            actions.len()
        );
        Ok(())
    }

    pub fn clear(&mut self) {
        for f in self.functions.values_mut() {
            f.clear();
        }
    }

    pub(crate) fn process_commands(
        &mut self,
        program: Vec<UnresolvedCommand>,
        stop: CompilerPassStop,
    ) -> Result<Vec<ResolvedNCommand>, Error> {
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
        self.desugar.number_underscores = underscores;
    }

    fn process_command(
        &mut self,
        command: UnresolvedCommand,
        stop: CompilerPassStop,
    ) -> Result<Vec<ResolvedNCommand>, Error> {
        let mut program =
            self.desugar
                .desugar_program(vec![command], self.test_proofs, self.seminaive)?;
        // TODO: a desugared program is not resolved yet
        // if stop == CompilerPassStop::Desugar {
        //     return Ok(program);
        // }

        // let type_info_before = self.type_info().clone();

        let program = self.desugar.type_info.typecheck_program(&program)?;
        // if stop == CompilerPassStop::TypecheckDesugared {
        //     return Ok(program);
        // }

        // now add term encoding
        // if self.terms_enabled {
        //     let program_terms = TermState::add_term_encoding(self, program);
        //     program = self.desugar.desugar_program(program_terms, false, false)?;
        // }

        // TODO: a term-encoded program is not resolved yet
        // if stop == CompilerPassStop::TermEncoding {
        //     return Ok(program);
        // }

        // reset type info
        // We don't need this anymore as type_info is pure
        // self.desugar.type_info = type_info_before;
        // let program = self.desugar.type_info.typecheck_program(&program)?;
        // if stop == CompilerPassStop::TypecheckTermEncoding {
        //     return Ok(program);
        // }

        Ok(program)
    }

    pub fn run_program(&mut self, program: Vec<UnresolvedCommand>) -> Result<Vec<String>, Error> {
        let should_run = true;

        for command in program {
            // Important to process each command individually
            // because push and pop create new scopes
            for processed in self.process_command(command, CompilerPassStop::All)? {
                self.run_command(processed, should_run)?;
            }
        }
        log::logger().flush();

        // remove consecutive empty lines
        Ok(self.flush_msgs())
    }

    pub fn parse_program(&self, input: &str) -> Result<Vec<UnresolvedCommand>, Error> {
        self.desugar.parse_program(input)
    }

    pub fn parse_and_run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let parsed = self.desugar.parse_program(input)?;
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
        self.desugar.type_info.add_arcsort(arcsort)
    }

    /// Add a user-defined primitive
    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        self.desugar.type_info.add_primitive(prim)
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
    pub fn serialize_for_graphviz(
        &self,
        split_primitive_outputs: bool,
    ) -> egraph_serialize::EGraph {
        let config = SerializeConfig {
            split_primitive_outputs,
            ..Default::default()
        };
        let mut serialized = self.serialize(config);
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

    pub(crate) fn type_info(&self) -> &TypeInfo {
        &self.desugar.type_info
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
    CheckError(Vec<UnresolvedFact>),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        constraint::SimpleTypeConstraint,
        sort::{FromSort, I64Sort, IntoSort, Sort, VecSort},
        EGraph, PrimitiveLike, Value,
    };

    struct InnerProduct {
        ele: Arc<I64Sort>,
        vec: Arc<VecSort>,
    }

    impl PrimitiveLike for InnerProduct {
        fn name(&self) -> symbol_table::GlobalSymbol {
            "inner-product".into()
        }

        fn get_type_constraints(&self) -> Box<dyn crate::constraint::TypeConstraint> {
            SimpleTypeConstraint::new(
                self.name(),
                vec![self.vec.clone(), self.vec.clone(), self.ele.clone()],
            )
            .into_box()
        }

        fn apply(&self, values: &[crate::Value], _egraph: &EGraph) -> Option<crate::Value> {
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
                "
                (let a (vec-of 1 2 3 4 5 6))
                (let b (vec-of 6 5 4 3 2 1))
                (check (= (inner-product a b) 56))
            ",
            )
            .unwrap();
    }
}
