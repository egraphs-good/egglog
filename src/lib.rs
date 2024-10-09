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
mod scheduler;
mod serialize;
pub mod sort;
mod termdag;
mod typechecking;
mod unionfind;
pub mod util;
mod value;

use ast::remove_globals::remove_globals;
use extract::Extractor;
use index::ColumnIndex;
use indexmap::map::Entry;
use instant::Instant;
use scheduler::RunReport;
pub use serialize::SerializeConfig;
pub use serialize::SerializedNode;
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
use scheduler::Scheduler;
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
    symbol_gen: SymbolGen,
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
    msgs: Vec<String>,
    scheduler_constructors: HashMap<Symbol, Rc<dyn Fn(Vec<Value>) -> Box<dyn Scheduler>>>,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            symbol_gen: SymbolGen::new("$".to_string()),
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
            msgs: Default::default(),
            type_info: Default::default(),
            scheduler_constructors: Default::default(),
        };
        egraph
            .rulesets
            .insert("".into(), Ruleset::Rules("".into(), Default::default()));
        egraph.scheduler_constructors.insert(
            "simple".into(),
            Rc::new(|_| Box::new(scheduler::SimpleScheduler)),
        );
        egraph.scheduler_constructors.insert(
            "backoff".into(),
            Rc::new(|_| Box::<scheduler::BackoffScheduler>::default()),
        );
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(String);

impl EGraph {
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
        if let Some(sort) = self.get_sort_from_value(&value) {
            if sort.is_eq_sort() {
                return Value {
                    tag: value.tag,
                    bits: usize::from(self.unionfind.find(Id::from(value.bits as usize))) as u64,
                };
            }
        }
        value
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
        let mut function = self.functions.get_mut(&func).unwrap();
        let n_unions = self.unionfind.n_unions();
        let merge_prog = match &function.merge.merge_vals {
            MergeFn::Expr(e) => Some(e.clone()),
            MergeFn::AssertEq | MergeFn::Union => None,
        };

        for (inputs, old, new) in merges {
            if let Some(prog) = function.merge.on_merge.clone() {
                self.run_actions(&[*old, *new], &prog).unwrap();
                function = self.functions.get_mut(&func).unwrap();
            }
            if let Some(prog) = &merge_prog {
                // TODO: error handling?
                let mut stack = self.run_actions(&[*old, *new], prog).unwrap();
                let merged = stack.pop().expect("merges should produce a value");
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
            Literal::Int(i) => i.store(&self.type_info.get_sort_nofail()).unwrap(),
            Literal::F64(f) => f.store(&self.type_info.get_sort_nofail()).unwrap(),
            Literal::String(s) => s.store(&self.type_info.get_sort_nofail()).unwrap(),
            Literal::Unit => ().store(&self.type_info.get_sort_nofail()).unwrap(),
            Literal::Bool(b) => b.store(&self.type_info.get_sort_nofail()).unwrap(),
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

    /// Run a schedule of commands.
    fn run_schedule(&mut self, sched: &ResolvedSchedule) -> Result<RunReport, Error> {
        scheduler::SimpleScheduler.run_schedule(self, sched)
    }

    /// Extract a value to a [`TermDag`] and [`Term`]
    /// in the [`TermDag`].
    /// See also extract_value_to_string for convenience.
    pub fn extract_value(&self, value: Value) -> (TermDag, Term) {
        let mut termdag = TermDag::default();
        let sort = self.type_info.sorts.get(&value.tag).unwrap();
        let term = self.extract(value, &mut termdag, sort).1;
        (termdag, term)
    }

    /// Extract a value to a string for printing.
    /// See also extract_value for more control.
    pub fn extract_value_to_string(&self, value: Value) -> String {
        let (termdag, term) = self.extract_value(value);
        termdag.to_string(&term)
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
        let core_rule = rule.to_canonicalized_core_rule(&self.type_info, &mut self.symbol_gen)?;
        let (query, actions) = (core_rule.body, core_rule.head);

        let vars = query.get_vars();
        let query = self.compile_gj_query(query, &vars);

        let program = self
            .compile_actions(&vars, &actions)
            .map_err(Error::TypeErrors)?;
        let compiled_rule = CompiledRule {
            props: rule.props,
            query,
            program,
        };
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
        name: Symbol,
    ) -> Result<Symbol, Error> {
        self.add_rule_with_name(name, rule, ruleset)
    }

    fn eval_actions(&mut self, actions: &ResolvedActions) -> Result<(), Error> {
        let (actions, _) = actions.to_core_actions(
            &self.type_info,
            &mut Default::default(),
            &mut self.symbol_gen,
        )?;
        let program = self
            .compile_actions(&Default::default(), &actions)
            .map_err(Error::TypeErrors)?;
        self.run_actions(&[], &program)?;
        Ok(())
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, Value), Error> {
        let fresh_name = self.symbol_gen.fresh(&"egraph_evalexpr".into());
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
    fn eval_resolved_expr(&mut self, expr: &ResolvedExpr) -> Result<Value, Error> {
        let (actions, mapped_expr) = expr.to_core_actions(
            &self.type_info,
            &mut Default::default(),
            &mut self.symbol_gen,
        )?;
        let target = mapped_expr.get_corresponding_var_or_lit(&self.type_info);
        let program = self
            .compile_expr(&Default::default(), &actions, &target)
            .map_err(Error::TypeErrors)?;
        let mut stack = self.run_actions(&[], &program)?;
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
            props: Default::default(),
            head: ResolvedActions::default(),
            body: facts.to_vec(),
        };
        let core_rule = rule.to_canonicalized_core_rule(&self.type_info, &mut self.symbol_gen)?;
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
                self.add_rule(rule, ruleset, name)?;
                log::info!("Declared rule {name}.")
            }
            ResolvedNCommand::RunSchedule(sched) => {
                let report = self.run_schedule(&sched)?;
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
                log::info!("Checked fact {}.", Facts(facts));
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
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(&filename)
                    .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                let mut termdag = TermDag::default();
                for expr in exprs {
                    let value = self.eval_resolved_expr(&expr)?;
                    let expr_type = expr.output_type(&self.type_info);
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
            .type_info
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
        let commands: Vec<_> = self
            .type_info
            .typecheck_program(&mut self.symbol_gen, &commands)?;
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
            !self.symbol_gen.has_been_used(),
            "Reserved symbol must be set before any symbols are generated"
        );
        self.symbol_gen = SymbolGen::new(sym.to_string());
    }

    fn process_command(&mut self, command: Command) -> Result<Vec<ResolvedNCommand>, Error> {
        let program =
            desugar::desugar_program(vec![command], &mut self.symbol_gen, self.seminaive)?;

        let program = self
            .type_info
            .typecheck_program(&mut self.symbol_gen, &program)?;

        let program = remove_globals(&self.type_info, program, &mut self.symbol_gen);

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
        let parsed = parse_program(filename, input)?;
        self.run_program(parsed)
    }

    pub fn num_tuples(&self) -> usize {
        self.functions.values().map(|f| f.nodes.len()).sum()
    }

    pub(crate) fn get_sort_from_value(&self, value: &Value) -> Option<&ArcSort> {
        self.type_info.sorts.get(&value.tag)
    }

    /// Returns the first sort that satisfies the type and predicate if there's one.
    /// Otherwise returns none.
    pub fn get_sort_by<S: Sort + Send + Sync>(
        &self,
        pred: impl Fn(&Arc<S>) -> bool,
    ) -> Option<Arc<S>> {
        self.type_info.get_sort_by(pred)
    }

    /// Returns a sort based on the type
    pub fn get_sort<S: Sort + Send + Sync>(&self) -> Option<Arc<S>> {
        self.type_info.get_sort_by(|_| true)
    }

    /// Add a user-defined sort
    pub fn add_arcsort(&mut self, arcsort: ArcSort) -> Result<(), TypeError> {
        self.type_info.add_arcsort(arcsort, DUMMY_SPAN.clone())
    }

    /// Add a user-defined primitive
    pub fn add_primitive(&mut self, prim: impl Into<Primitive>) {
        self.type_info.add_primitive(prim)
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
        self.msgs.push(msg);
    }

    fn flush_msgs(&mut self) -> Vec<String> {
        self.msgs.dedup_by(|a, b| a.is_empty() && b.is_empty());
        std::mem::take(&mut self.msgs)
    }

    /// Get the timestamp when the rule was last run.
    pub fn get_rule_timestamp(&self, rule: Symbol) -> u32 {
        *self.rule_last_run_timestamp.get(&rule).unwrap_or(&0)
    }

    /// Set the timestamp of a rule to the current timestamp.
    ///
    /// Note that the user is responsible for making sure the rule is indeed run
    /// over the database; the ruleset may not be fired for tuples created during
    /// `egraph.get_rule_timestamp(ruleset)`--`self.get_timestamp()` time.
    pub fn update_rule_timestamp(&mut self, ruleset: Symbol) {
        self.rule_last_run_timestamp.insert(ruleset, self.timestamp);
    }

    /// Get the current timestamp
    pub fn get_timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Bump the timestamp by 1
    pub fn bump_timestamp(&mut self) {
        self.timestamp += 1;
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
    #[error("{}\nScheduler not found: {0}", .1.get_quote())]
    SchedulerNotFound(String, Span),
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
