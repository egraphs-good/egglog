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
mod cli;
pub mod constraint;
mod core;
pub mod extract;
pub mod prelude;
pub mod scheduler;
mod serialize;
pub mod sort;
mod termdag;
mod typechecking;
pub mod util;

// This is used to allow the `add_primitive` macro to work in
// both this crate and other crates by referring to `::egglog`.
extern crate self as egglog;
pub use add_primitive::add_primitive;
use ast::*;
#[cfg(feature = "bin")]
pub use cli::bin::*;
use constraint::{Constraint, Problem, SimpleTypeConstraint, TypeConstraint};
use core::{AtomTerm, ResolvedAtomTerm, ResolvedCall};
use core_relations::{make_external_func, ExternalFunctionId};
pub use core_relations::{ExecutionState, Value};
use egglog_bridge::{ColumnTy, IterationReport, QueryEntry};
use extract::{Extractor, TreeAdditiveCostModel};
use indexmap::map::Entry;
use numeric_id::DenseIdMap;
use prelude::*;
use scheduler::{SchedulerId, SchedulerRecord};
pub use serialize::{SerializeConfig, SerializedNode};
use sort::*;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::iter::once;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
pub use termdag::{Term, TermDag, TermId};
use thiserror::Error;
pub use typechecking::TypeError;
use typechecking::TypeInfo;
use util::*;
use web_time::Duration;

pub type ArcSort = Arc<dyn Sort>;

pub trait Primitive {
    fn name(&self) -> Symbol;
    /// Constructs a type constraint for the primitive that uses the span information
    /// for error localization.
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint>;
    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value>;
}

/// Running a schedule produces a report of the results.
/// This includes rough timing information and whether
/// the database was updated.
/// Calling `union` on two run reports adds the timing
/// information together.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// If any changes were made to the database.
    pub updated: bool,
    pub search_and_apply_time_per_rule: HashMap<Symbol, Duration>,
    pub num_matches_per_rule: HashMap<Symbol, usize>,
    pub search_and_apply_time_per_ruleset: HashMap<Symbol, Duration>,
    pub merge_time_per_ruleset: HashMap<Symbol, Duration>,
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
        let mut rule_times_vec: Vec<_> = self.search_and_apply_time_per_rule.iter().collect();
        rule_times_vec.sort_by_key(|(_, time)| **time);

        for (rule, time) in rule_times_vec {
            let name = Self::truncate_rule_name(*rule);
            let time = time.as_secs_f64();
            let num_matches = self.num_matches_per_rule.get(rule).copied().unwrap_or(0);
            writeln!(
                f,
                "Rule {name}: search and apply {time:.3}s, num matches {num_matches}",
            )?;
        }

        let rulesets = self
            .search_and_apply_time_per_ruleset
            .keys()
            .chain(self.merge_time_per_ruleset.keys())
            .chain(self.rebuild_time_per_ruleset.keys())
            .collect::<IndexSet<_>>();

        for ruleset in rulesets {
            // print out the search and apply time for rule
            let search_and_apply_time = self
                .search_and_apply_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            let merge_time = self
                .merge_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            let rebuild_time = self
                .rebuild_time_per_ruleset
                .get(ruleset)
                .cloned()
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            writeln!(
                f,
                "Ruleset {ruleset}: search {search_and_apply_time:.3}s, merge {merge_time:.3}s, rebuild {rebuild_time:.3}s",
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
    fn union_times(times: &mut HashMap<Symbol, Duration>, other_times: &HashMap<Symbol, Duration>) {
        for (k, v) in other_times {
            let entry = times.entry(*k).or_default();
            *entry += *v;
        }
    }

    fn union_counts(counts: &mut HashMap<Symbol, usize>, other_counts: &HashMap<Symbol, usize>) {
        for (k, v) in other_counts {
            let entry = counts.entry(*k).or_default();
            *entry += *v;
        }
    }

    pub fn union(&mut self, other: &Self) {
        self.updated |= other.updated;
        RunReport::union_times(
            &mut self.search_and_apply_time_per_rule,
            &other.search_and_apply_time_per_rule,
        );
        RunReport::union_counts(&mut self.num_matches_per_rule, &other.num_matches_per_rule);
        RunReport::union_times(
            &mut self.search_and_apply_time_per_ruleset,
            &other.search_and_apply_time_per_ruleset,
        );
        RunReport::union_times(
            &mut self.merge_time_per_ruleset,
            &other.merge_time_per_ruleset,
        );
        RunReport::union_times(
            &mut self.rebuild_time_per_ruleset,
            &other.rebuild_time_per_ruleset,
        );
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
    backend: egglog_bridge::EGraph,
    pub parser: Parser,
    names: check_shadowing::Names,
    /// pushed_egraph forms a linked list of pushed egraphs.
    /// Pop reverts the egraph to the last pushed egraph.
    pushed_egraph: Option<Box<Self>>,
    functions: IndexMap<Symbol, Function>,
    rulesets: IndexMap<Symbol, Ruleset>,
    interactive_mode: bool,
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
    schedulers: DenseIdMap<SchedulerId, SchedulerRecord>,
    commands: IndexMap<Symbol, Arc<dyn UserDefinedCommand>>,
}

/// A user-defined command allows users to inject custom command that can be called
/// in an egglog program.
///
/// Compared to an external function, a user-defined command is more powerful because
/// it has an exclusive access to the e-graph.
pub trait UserDefinedCommand: Send + Sync {
    /// Run the command with the given arguments.
    fn update(&self, egraph: &mut EGraph, args: &[Expr]) -> Result<(), Error>;
}

#[derive(Clone)]
pub struct Function {
    decl: ResolvedFunctionDecl,
    pub schema: ResolvedSchema,
    pub can_subsume: bool,
    backend_id: egglog_bridge::FunctionId,
}

#[derive(Clone, Debug)]
pub struct ResolvedSchema {
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

impl ResolvedSchema {
    pub fn get_by_pos(&self, index: usize) -> Option<&ArcSort> {
        if self.input.len() == index {
            Some(&self.output)
        } else {
            self.input.get(index)
        }
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("decl", &self.decl)
            .field("schema", &self.schema)
            .finish()
    }
}

impl Default for EGraph {
    fn default() -> Self {
        let mut eg = Self {
            backend: Default::default(),
            parser: Default::default(),
            names: Default::default(),
            pushed_egraph: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            run_mode: RunMode::Normal,
            interactive_mode: false,
            fact_directory: None,
            seminaive: true,
            extract_report: None,
            recent_run_report: None,
            overall_run_report: Default::default(),
            msgs: Some(vec![]),
            type_info: Default::default(),
            schedulers: Default::default(),
            commands: Default::default(),
        };

        add_leaf_sort(&mut eg, UnitSort, span!()).unwrap();
        add_leaf_sort(&mut eg, StringSort, span!()).unwrap();
        add_leaf_sort(&mut eg, BoolSort, span!()).unwrap();
        add_leaf_sort(&mut eg, I64Sort, span!()).unwrap();
        add_leaf_sort(&mut eg, F64Sort, span!()).unwrap();
        add_leaf_sort(&mut eg, BigIntSort, span!()).unwrap();
        add_leaf_sort(&mut eg, BigRatSort, span!()).unwrap();
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
            .insert("".into(), Ruleset::Rules(Default::default()));

        eg
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(String);

impl EGraph {
    pub fn add_command(
        &mut self,
        name: Symbol,
        command: Arc<dyn UserDefinedCommand>,
    ) -> Result<(), Error> {
        if self.commands.contains_key(&name)
            || self.functions.contains_key(&name)
            || self.type_info.get_prims(&name).is_some()
        {
            return Err(Error::CommandAlreadyExists(name, span!()));
        }
        self.commands.insert(name, command);
        self.parser.add_user_defined(name)?;
        Ok(())
    }

    pub fn is_interactive_mode(&self) -> bool {
        self.interactive_mode
    }

    pub fn push(&mut self) {
        let prev_prev: Option<Box<Self>> = self.pushed_egraph.take();
        let mut prev = self.clone();
        prev.pushed_egraph = prev_prev;
        self.pushed_egraph = Some(Box::new(prev));
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
        match self.pushed_egraph.take() {
            Some(e) => {
                // Copy the reports and messages from the popped egraph
                let extract_report = self.extract_report.clone();
                let recent_run_report = self.recent_run_report.clone();
                let overall_run_report = self.overall_run_report.clone();
                let messages = self.msgs.clone();

                *self = *e;
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

    fn translate_expr_to_mergefn(
        &self,
        expr: &ResolvedExpr,
    ) -> Result<egglog_bridge::MergeFn, Error> {
        match expr {
            GenericExpr::Lit(_, literal) => {
                let val = literal_to_value(&self.backend, literal);
                Ok(egglog_bridge::MergeFn::Const(val))
            }
            GenericExpr::Var(span, resolved_var) => match resolved_var.name.as_str() {
                "old" => Ok(egglog_bridge::MergeFn::Old),
                "new" => Ok(egglog_bridge::MergeFn::New),
                // NB: type-checking should already catch unbound variables here.
                _ => Err(TypeError::Unbound(resolved_var.name, span.clone()).into()),
            },
            GenericExpr::Call(_, ResolvedCall::Func(f), args) => {
                let translated_args = args
                    .iter()
                    .map(|arg| self.translate_expr_to_mergefn(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(egglog_bridge::MergeFn::Function(
                    self.functions[&f.name].backend_id,
                    translated_args,
                ))
            }
            GenericExpr::Call(_, ResolvedCall::Primitive(p), args) => {
                let translated_args = args
                    .iter()
                    .map(|arg| self.translate_expr_to_mergefn(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(egglog_bridge::MergeFn::Primitive(
                    p.primitive.1,
                    translated_args,
                ))
            }
        }
    }

    fn declare_function(&mut self, decl: &ResolvedFunctionDecl) -> Result<(), Error> {
        let get_sort = |name: &Symbol| match self.type_info.get_sort_by_name(name) {
            Some(sort) => Ok(sort.clone()),
            None => Err(Error::TypeError(TypeError::UndefinedSort(
                *name,
                decl.span.clone(),
            ))),
        };

        let input = decl
            .schema
            .input
            .iter()
            .map(get_sort)
            .collect::<Result<Vec<_>, _>>()?;
        let output = get_sort(&decl.schema.output)?;

        let can_subsume = match decl.subtype {
            FunctionSubtype::Constructor => true,
            FunctionSubtype::Relation => true,
            FunctionSubtype::Custom => false,
        };

        use egglog_bridge::{DefaultVal, MergeFn};
        let backend_id = self.backend.add_table(egglog_bridge::FunctionConfig {
            schema: input
                .iter()
                .chain([&output])
                .map(|sort| sort.column_ty(&self.backend))
                .collect(),
            default: match decl.subtype {
                FunctionSubtype::Constructor => DefaultVal::FreshId,
                FunctionSubtype::Custom => DefaultVal::Fail,
                FunctionSubtype::Relation => DefaultVal::Const(self.backend.primitives().get(())),
            },
            merge: match decl.subtype {
                FunctionSubtype::Constructor => MergeFn::UnionId,
                FunctionSubtype::Relation => MergeFn::AssertEq,
                FunctionSubtype::Custom => match &decl.merge {
                    None => MergeFn::AssertEq,
                    Some(expr) => self.translate_expr_to_mergefn(expr)?,
                },
            },
            name: decl.name.to_string(),
            can_subsume,
        });

        let function = Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            can_subsume,
            backend_id,
        };

        let old = self.functions.insert(decl.name, function);
        if old.is_some() {
            panic!(
                "Typechecking should have caught function already bound: {}",
                decl.name
            );
        }

        Ok(())
    }

    /// Extract rows of a table using the default cost model with name sym
    /// The `include_output` parameter controls whether the output column is always extracted
    /// For functions, the output column is usually useful
    /// For constructors and relations, the output column can be ignored
    pub fn function_to_dag(
        &self,
        sym: Symbol,
        n: usize,
        include_output: bool,
    ) -> Result<(Vec<Term>, Option<Vec<Term>>, TermDag), Error> {
        let func = self
            .functions
            .get(&sym)
            .ok_or(TypeError::UnboundFunction(sym, span!()))?;
        let mut rootsorts = func.schema.input.clone();
        if include_output {
            rootsorts.push(func.schema.output.clone());
        }
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(rootsorts),
            self,
            TreeAdditiveCostModel::default(),
        );

        let mut termdag = TermDag::default();
        let mut inputs: Vec<Term> = Vec::new();
        let mut output: Option<Vec<Term>> = if include_output {
            Some(Vec::new())
        } else {
            None
        };

        let extract_row = |row: egglog_bridge::FunctionRow| {
            if inputs.len() < n {
                // include subsumed rows
                let mut children: Vec<Term> = Vec::new();
                for (value, sort) in row.vals.iter().zip(&func.schema.input) {
                    let (_, term) = extractor
                        .extract_best_with_sort(self, &mut termdag, *value, sort.clone())
                        .unwrap_or_else(|| (0, termdag.var("Unextractable".into())));
                    children.push(term);
                }
                inputs.push(termdag.app(sym, children));
                if include_output {
                    let value = row.vals[func.schema.input.len()];
                    let sort = &func.schema.output;
                    let (_, term) = extractor
                        .extract_best_with_sort(self, &mut termdag, value, sort.clone())
                        .unwrap_or_else(|| (0, termdag.var("Unextractable".into())));
                    output.as_mut().unwrap().push(term);
                }
                true
            } else {
                false
            }
        };

        self.backend.for_each_while(func.backend_id, extract_row);

        Ok((inputs, output, termdag))
    }

    pub fn print_function(&mut self, sym: Symbol, n: usize) -> Result<(), Error> {
        log::info!("Printing up to {n} tuples of table {sym}: ");
        let (terms, outputs, termdag) = self.function_to_dag(sym, n, true)?;
        let f = self
            .functions
            .get(&sym)
            // function_to_dag should have checked this
            .unwrap();
        let out_is_unit = f.schema.output.name() == UnitSort.name().into();

        let mut buf = String::new();
        let s = &mut buf;
        s.push_str("(\n");
        if terms.is_empty() {
            log::info!("   (none)");
        }
        for (term, output) in terms.iter().zip(&outputs.unwrap()) {
            let tuple_str = format!(
                "   {}{}",
                termdag.to_string(term),
                if !out_is_unit {
                    format!(" -> {}", termdag.to_string(output))
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
            let size = self.backend.table_size(f.backend_id);
            log::info!("Function {} has size {}", sym, size);
            self.print_msg(size.to_string());
            Ok(())
        } else {
            // Print size of all functions
            let mut lens = self
                .functions
                .iter()
                .map(|(sym, f)| (*sym, self.backend.table_size(f.backend_id)))
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
                    report.union(&rec);
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
                    report.union(&rec);
                    if !rec.updated {
                        break;
                    }
                }
                report
            }
            ResolvedSchedule::Sequence(_span, scheds) => {
                let mut report = RunReport::default();
                for sched in scheds {
                    report.union(&self.run_schedule(sched));
                }
                report
            }
        }
    }

    /// Extract a value to a [`TermDag`] and [`Term`] in the [`TermDag`].
    /// Note that the `TermDag` may contain a superset of the nodes in the `Term`.
    /// See also `extract_value_to_string` for convenience.
    pub fn extract_value(&self, sort: &ArcSort, value: Value) -> Result<(TermDag, Term), Error> {
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(vec![sort.clone()]),
            self,
            TreeAdditiveCostModel::default(),
        );
        let mut termdag = TermDag::default();
        let (_, term) = extractor.extract_best(self, &mut termdag, value).unwrap();
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
        report.union(&subreport);

        log::debug!("database size: {}", self.num_tuples());

        report
    }

    pub fn step_rules(&mut self, ruleset: Symbol) -> RunReport {
        fn collect_rule_ids(
            ruleset: Symbol,
            rulesets: &IndexMap<Symbol, Ruleset>,
            ids: &mut Vec<egglog_bridge::RuleId>,
        ) {
            match &rulesets[&ruleset] {
                Ruleset::Rules(rules) => {
                    for (_, id) in rules.values() {
                        ids.push(*id);
                    }
                }
                Ruleset::Combined(sub_rulesets) => {
                    for sub_ruleset in sub_rulesets {
                        collect_rule_ids(*sub_ruleset, rulesets, ids);
                    }
                }
            }
        }

        let mut rule_ids = Vec::new();
        collect_rule_ids(ruleset, &self.rulesets, &mut rule_ids);
        let iteration_report = self.backend.run_rules(&rule_ids).unwrap();
        let IterationReport {
            changed: updated,
            rule_reports,
            search_and_apply_time,
            merge_time,
            rebuild_time,
        } = iteration_report;

        let (search_and_apply_time_per_rule, num_matches_per_rule) = rule_reports
            .into_iter()
            .map(|(rule, report)| {
                (
                    (rule.as_str().into(), report.search_and_apply_time),
                    (rule.as_str().into(), report.num_matches),
                )
            })
            .unzip();

        let per_ruleset = |x| [(ruleset, x)].into_iter().collect();

        RunReport {
            updated,
            search_and_apply_time_per_rule,
            num_matches_per_rule,
            search_and_apply_time_per_ruleset: per_ruleset(search_and_apply_time),
            merge_time_per_ruleset: per_ruleset(merge_time),
            rebuild_time_per_ruleset: per_ruleset(rebuild_time),
        }
    }

    fn add_rule_with_name(
        &mut self,
        name: Symbol,
        rule: ast::ResolvedRule,
        ruleset: Symbol,
    ) -> Result<Symbol, Error> {
        let core_rule =
            rule.to_canonicalized_core_rule(&self.type_info, &mut self.parser.symbol_gen)?;
        let (query, actions) = (&core_rule.body, &core_rule.head);

        let rule_id = {
            let mut translator = BackendRule::new(
                self.backend.new_rule(name.into(), self.seminaive),
                &self.functions,
                &self.type_info,
            );
            translator.query(query, false);
            translator.actions(actions)?;
            translator.build()
        };

        if let Some(rules) = self.rulesets.get_mut(&ruleset) {
            match rules {
                Ruleset::Rules(rules) => {
                    match rules.entry(name) {
                        indexmap::map::Entry::Occupied(_) => {
                            panic!("Rule '{name}' was already present")
                        }
                        indexmap::map::Entry::Vacant(e) => e.insert((core_rule, rule_id)),
                    };
                    Ok(name)
                }
                Ruleset::Combined(_) => Err(Error::CombinedRulesetError(ruleset, rule.span)),
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

        let mut translator = BackendRule::new(
            self.backend.new_rule("eval_actions", false),
            &self.functions,
            &self.type_info,
        );
        translator.actions(&actions)?;
        let id = translator.build();
        let result = self.backend.run_rules(&[id]);
        self.backend.free_rule(id);

        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(Error::BackendError(e.to_string())),
        }
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, Value), Error> {
        let span = expr.span();
        let command = Command::Action(Action::Expr(span.clone(), expr.clone()));
        let resolved_commands = self.process_command(command)?;
        assert_eq!(resolved_commands.len(), 1);
        let resolved_command = resolved_commands.into_iter().next().unwrap();
        let resolved_expr = match resolved_command {
            ResolvedNCommand::CoreAction(ResolvedAction::Expr(_, resolved_expr)) => resolved_expr,
            _ => unreachable!(),
        };
        let sort = resolved_expr.output_type();
        let value = self.eval_resolved_expr(span, &resolved_expr)?;
        Ok((sort, value))
    }

    fn eval_resolved_expr(&mut self, span: Span, expr: &ResolvedExpr) -> Result<Value, Error> {
        let unit_id = self.backend.primitives().get_ty::<()>();
        let unit_val = self.backend.primitives().get(());

        let result: egglog_bridge::SideChannel<Value> = Default::default();
        let result_ref = result.clone();
        let ext_id = self
            .backend
            .register_external_func(make_external_func(move |_es, vals| {
                debug_assert!(vals.len() == 1);
                *result_ref.lock().unwrap() = Some(vals[0]);
                Some(unit_val)
            }));

        let mut translator = BackendRule::new(
            self.backend.new_rule("eval_resolved_expr", false),
            &self.functions,
            &self.type_info,
        );

        let result_var = ResolvedVar {
            name: self
                .parser
                .symbol_gen
                .fresh(&Symbol::from("eval_resolved_expr")),
            sort: expr.output_type(),
            is_global_ref: false,
        };
        let actions = ResolvedActions::singleton(ResolvedAction::Let(
            span.clone(),
            result_var.clone(),
            expr.clone(),
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
            || "this function will never panic".to_string(),
        );

        let id = translator.build();
        let _ = self.backend.run_rules(&[id]).unwrap();
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);

        let result = result.lock().unwrap().unwrap();
        Ok(result)
    }

    fn add_combined_ruleset(&mut self, name: Symbol, rulesets: Vec<Symbol>) {
        match self.rulesets.entry(name) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Combined(rulesets)),
        };
    }

    fn add_ruleset(&mut self, name: Symbol) {
        match self.rulesets.entry(name) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Rules(Default::default())),
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

        let ext_sc = egglog_bridge::SideChannel::default();
        let ext_sc_ref = ext_sc.clone();
        let ext_id = self
            .backend
            .register_external_func(make_external_func(move |_, _| {
                *ext_sc_ref.lock().unwrap() = Some(());
                Some(Value::new_const(0))
            }));

        let mut translator = BackendRule::new(
            self.backend.new_rule("check_facts", false),
            &self.functions,
            &self.type_info,
        );
        translator.query(&query, true);
        translator
            .rb
            .call_external_func(ext_id, &[], egglog_bridge::ColumnTy::Id, || {
                "this function will never panic".to_string()
            });
        let id = translator.build();
        let _ = self.backend.run_rules(&[id]).unwrap();
        self.backend.free_rule(id);

        self.backend.free_external_func(ext_id);

        let ext_sc_val = ext_sc.lock().unwrap().take();
        let matched = matches!(ext_sc_val, Some(()));

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
            ResolvedNCommand::AddRuleset(_span, name) => {
                self.add_ruleset(name);
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::UnstableCombinedRuleset(_span, name, others) => {
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
                self.overall_run_report.union(&report);
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
            ResolvedNCommand::Extract(span, expr, variants) => {
                let sort = expr.output_type();

                let x = self.eval_resolved_expr(span.clone(), &expr)?;
                let n = self.eval_resolved_expr(span, &variants)?;
                let n: i64 = self.backend.primitives().unwrap(n);

                let mut termdag = TermDag::default();

                let extractor = Extractor::compute_costs_from_rootsorts(
                    Some(vec![sort]),
                    self,
                    TreeAdditiveCostModel::default(),
                );
                if n == 0 {
                    if let Some((cost, term)) = extractor.extract_best(self, &mut termdag, x) {
                        // dont turn termdag into a string if we have messages disabled for performance reasons
                        if self.messages_enabled() {
                            let extracted = termdag.to_string(&term);
                            log::info!("extracted with cost {cost}: {extracted}");
                            self.print_msg(extracted);
                        }
                        self.extract_report = Some(ExtractReport::Best {
                            termdag,
                            cost,
                            term,
                        });
                    } else {
                        return Err(Error::ExtractError(
                            "Unable to find any valid extraction (likely due to subsume or delete)"
                                .to_string(),
                        ));
                    }
                } else {
                    if n < 0 {
                        panic!("Cannot extract negative number of variants");
                    }
                    let terms: Vec<Term> = extractor
                        .extract_variants(self, &mut termdag, x, n as usize)
                        .iter()
                        .map(|e| e.1.clone())
                        .collect();
                    // Same as above, avoid turning termdag into a string if we have messages disabled for performance
                    if self.messages_enabled() {
                        log::info!("extracted variants:");
                        let mut msg = String::default();
                        msg += "(\n";
                        assert!(!terms.is_empty());
                        for expr in &terms {
                            let str = termdag.to_string(expr);
                            log::info!("   {str}");
                            msg += &format!("   {str}\n");
                        }
                        msg += ")";
                        self.print_msg(msg);
                    }
                    self.extract_report = Some(ExtractReport::Variants { termdag, terms });
                }
            }
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

                let extractor = Extractor::compute_costs_from_rootsorts(
                    None,
                    self,
                    TreeAdditiveCostModel::default(),
                );
                let mut termdag: TermDag = Default::default();

                use std::io::Write;
                for expr in exprs {
                    let value = self.eval_resolved_expr(span.clone(), &expr)?;
                    let expr_type = expr.output_type();

                    let term = extractor
                        .extract_best_with_sort(self, &mut termdag, value, expr_type)
                        .unwrap()
                        .1;
                    writeln!(f, "{}", termdag.to_string(&term))
                        .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                }

                log::info!("Output to '{filename:?}'.")
            }
            ResolvedNCommand::UserDefined(_span, name, exprs) => {
                let command = self.commands.swap_remove(&name).unwrap_or_else(|| {
                    panic!("Unrecognized user-defined command: {}", name);
                });
                command.update(self, &exprs)?;
                self.commands.insert(name, command);
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

        // Can also do a row-major Vec<Value>
        let mut parsed_contents: Vec<Vec<Value>> = Vec::with_capacity(contents.lines().count());

        let mut row_schema = func.schema.input.clone();
        if function_type.subtype == FunctionSubtype::Custom {
            row_schema.push(func.schema.output.clone());
        }

        log::debug!("{:?}", row_schema);

        let unit_val = self.backend.primitives().get(());

        for line in contents.lines() {
            let mut it = line.split('\t').map(|s| s.trim());

            let mut row: Vec<Value> = Vec::with_capacity(row_schema.len());

            for sort in row_schema.iter() {
                if let Some (raw) = it.next() {
                    let val = match sort.name().as_str() {
                        "i64" => {
                            if let Ok(i) = raw.parse::<i64>() {
                                self.backend.primitives().get(i)
                            } else {
                                return Err(Error::InputFileFormatError(file));
                            }
                        }
                        "f64" => {
                            if let Ok(f) = raw.parse::<f64>() {
                                self.backend
                                    .primitives()
                                    .get::<F>(core_relations::Boxed::new(f.into()))
                            } else {
                                return Err(Error::InputFileFormatError(file));
                            }
                        }
                        "String" => self
                            .backend
                            .primitives()
                            .get::<S>(SymbolWrapper::new(raw.to_string().into())),
                        "Unit" => unit_val,
                        _ => panic!("Unreachable"),
                    };
                    row.push(val);
                } else {
                    break;
                }
            }

            if row.is_empty() {
                continue;
            }

            if row.len() != row_schema.len() || it.next().is_some() {
                return Err(Error::InputFileFormatError(file));
            }

            parsed_contents.push(row);
        }

        log::debug!("Successfully loaded file.");

        let num_facts = parsed_contents.len();

        let table_action = egglog_bridge::TableAction::new(&self.backend, func.backend_id);

        let unit_id = self.backend.primitives().get_ty::<()>();
        let use_insert = function_type.subtype != FunctionSubtype::Constructor;

        let ext_id = self
            .backend
            .register_external_func(make_external_func(move |es, _| {
                for row in parsed_contents.iter() {
                    if use_insert {
                        table_action.insert(es, row.to_vec());
                    } else {
                        table_action.lookup(es, row);
                    }
                }
                Some(unit_val)
            }));

        let mut translator = BackendRule::new(
            self.backend.new_rule("input_file", false),
            &self.functions,
            &self.type_info,
        );

        translator.rb.call_external_func(
            ext_id,
            &[],
            egglog_bridge::ColumnTy::Primitive(unit_id),
            || "this function will never panic".to_string(),
        );

        let id = translator.build();
        let _ = self.backend.run_rules(&[id]).unwrap();
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);

        log::info!("Read {num_facts} facts into {func_name} from '{file}'.");
        Ok(())
    }

    fn process_command(&mut self, command: Command) -> Result<Vec<ResolvedNCommand>, Error> {
        let program = desugar::desugar_program(vec![command], &mut self.parser, self.seminaive)?;

        let program = self.typecheck_program(&program)?;

        let program = remove_globals::remove_globals(program, &mut self.parser.symbol_gen);

        for command in &program {
            self.names.check_shadowing(command)?;
        }

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
        self.functions
            .values()
            .map(|f| self.backend.table_size(f.backend_id))
            .sum()
    }

    /// Returns a sort based on the type.
    pub fn get_sort<S: Sort>(&self) -> Arc<S> {
        self.type_info.get_sort()
    }

    /// Returns a sort that satisfies the type and predicate.
    pub fn get_sort_by<S: Sort>(&self, f: impl Fn(&Arc<S>) -> bool) -> Arc<S> {
        self.type_info.get_sort_by(f)
    }

    /// Returns all sorts based on the type.
    pub fn get_sorts<S: Sort>(&self) -> Vec<Arc<S>> {
        self.type_info.get_sorts()
    }

    /// Returns all sorts that satisfy the type and predicate.
    pub fn get_sorts_by<S: Sort>(&self, f: impl Fn(&Arc<S>) -> bool) -> Vec<Arc<S>> {
        self.type_info.get_sorts_by(f)
    }

    /// Returns a sort based on the predicate.
    pub fn get_arcsort_by(&self, f: impl Fn(&ArcSort) -> bool) -> ArcSort {
        self.type_info.get_arcsort_by(f)
    }

    /// Returns all sorts that satisfy the predicate.
    pub fn get_arcsorts_by(&self, f: impl Fn(&ArcSort) -> bool) -> Vec<ArcSort> {
        self.type_info.get_arcsorts_by(f)
    }

    /// Returns the sort with the given name if it exists.
    pub fn get_sort_by_name(&self, sym: &Symbol) -> Option<&ArcSort> {
        self.type_info.get_sort_by_name(sym)
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

    /// Convert from an egglog value to a Rust type.
    pub fn value_to_rust<T: core_relations::Primitive>(&self, x: Value) -> T {
        self.backend.primitives().unwrap::<T>(x)
    }

    /// Convert from a Rust type to an egglog value.
    pub fn rust_to_value<T: core_relations::Primitive>(&self, x: T) -> Value {
        self.backend.primitives().get::<T>(x)
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

    pub fn get_size(&self, function_id: egglog_bridge::FunctionId) -> usize {
        self.backend.table_size(function_id)
    }
}

struct BackendRule<'a> {
    rb: egglog_bridge::RuleBuilder<'a>,
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
        self.functions[&f.name].backend_id
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
                ResolvedFunctionId::Lookup(egglog_bridge::TableAction::new(
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

            qe_args[0] = self.rb.egraph().primitive_constant(ResolvedFunction {
                id,
                do_rebuild,
                name,
            });
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
                            let span = span.clone();
                            self.rb
                                .lookup(f, &args, move || {
                                    format!("{span}: lookup of function {name} failed")
                                })
                                .into()
                        }
                        ResolvedCall::Primitive(p) => {
                            let name = p.primitive.0.name();
                            let (p, args, ty) = self.prim(p, args);
                            let span = span.clone();
                            self.rb
                                .call_external_func(p, &args, ty, move || {
                                    format!("{span}: call of primitive {name} failed")
                                })
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
        Literal::Float(x) => egraph.primitive_constant::<sort::F>(x.into()),
        Literal::String(x) => egraph.primitive_constant::<sort::S>(sort::S::new(*x)),
        Literal::Bool(x) => egraph.primitive_constant::<bool>(*x),
        Literal::Unit => egraph.primitive_constant::<()>(()),
    }
}

fn literal_to_value(egraph: &egglog_bridge::EGraph, l: &Literal) -> Value {
    match l {
        Literal::Int(x) => egraph.primitives().get::<i64>(*x),
        Literal::Float(x) => egraph.primitives().get::<sort::F>(x.into()),
        Literal::String(x) => egraph.primitives().get::<sort::S>(sort::S::new(*x)),
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
    #[error("{0}")]
    BackendError(String),
    #[error("{0}\nTried to pop too much")]
    Pop(Span),
    #[error("{0}\nCommand should have failed.")]
    ExpectFail(Span),
    #[error("{2}\nIO error: {0}: {1}")]
    IoError(PathBuf, std::io::Error, Span),
    #[error("Cannot subsume function with merge: {0}")]
    SubsumeMergeError(Symbol),
    #[error("extraction failure: {:?}", .0)]
    ExtractError(String),
    #[error("{1}\n{2}\nShadowing is not allowed, but found {0}")]
    Shadowing(Symbol, Span, Span),
    #[error("{1}\nCommand already exists: {0}")]
    CommandAlreadyExists(Symbol, Span),
    #[error("Incorrect format in file '{0}'.")]
    InputFileFormatError(String),
}

#[cfg(test)]
mod tests {
    use lazy_static::lazy_static;
    use std::sync::Mutex;

    use crate::constraint::SimpleTypeConstraint;
    use crate::sort::*;
    use crate::*;

    #[derive(Clone)]
    struct InnerProduct {
        vec: ArcSort,
    }

    impl Primitive for InnerProduct {
        fn name(&self) -> symbol_table::GlobalSymbol {
            "inner-product".into()
        }

        fn get_type_constraints(&self, span: &Span) -> Box<dyn crate::constraint::TypeConstraint> {
            SimpleTypeConstraint::new(
                self.name(),
                vec![self.vec.clone(), self.vec.clone(), I64Sort.to_arcsort()],
                span.clone(),
            )
            .into_box()
        }

        fn apply(&self, exec_state: &mut ExecutionState<'_>, args: &[Value]) -> Option<Value> {
            let mut sum = 0;
            let vec1 = exec_state
                .containers()
                .get_val::<VecContainer>(args[0])
                .unwrap();
            let vec2 = exec_state
                .containers()
                .get_val::<VecContainer>(args[1])
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

        let int_vec_sort = egraph.get_arcsort_by(|s| {
            s.value_type() == Some(std::any::TypeId::of::<VecContainer>())
                && s.inner_sorts()[0].name() == I64Sort.name().into()
        });

        egraph.add_primitive(InnerProduct { vec: int_vec_sort });

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

    // Test that `EGraph` is `Send` and `Sync`
    lazy_static! {
        pub static ref RT: Mutex<EGraph> = Mutex::new(EGraph::default());
    }

    fn get_function(egraph: &EGraph, name: &str) -> Function {
        egraph.functions.get(&Symbol::from(name)).unwrap().clone()
    }

    fn get_value(egraph: &EGraph, name: &str) -> Value {
        let mut out = None;
        let id = get_function(egraph, name).backend_id;
        egraph.backend.for_each(id, |row| out = Some(row.vals[0]));
        out.unwrap()
    }

    #[test]
    fn test_subsumed_unextractable_rebuild_arg() {
        // Tests that a term stays unextractable even after a rebuild after a union would change the value of one of its args
        let mut egraph = EGraph::default();

        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype Math)
                (constructor container (Math) Math)
                (constructor exp () Math :cost 100)
                (constructor cheap () Math)
                (constructor cheap-1 () Math)
                ; we make the container cheap so that it will be extracted if possible, but then we mark it as subsumed
                ; so the (exp) expr should be extracted instead
                (let res (container (cheap)))
                (union res (exp))
                (cheap)
                (cheap-1)
                (subsume (container (cheap)))
                "#,
            ).unwrap();
        // At this point (cheap) and (cheap-1) should have different values, because they aren't unioned
        let orig_cheap_value = get_value(&egraph, "cheap");
        let orig_cheap_1_value = get_value(&egraph, "cheap-1");
        assert_ne!(orig_cheap_value, orig_cheap_1_value);
        // Then we can union them
        egraph
            .parse_and_run_program(
                None,
                r#"
                (union (cheap-1) (cheap))
                "#,
            )
            .unwrap();
        // And verify that their values are now the same and different from the original (cheap) value.
        let new_cheap_value = get_value(&egraph, "cheap");
        let new_cheap_1_value = get_value(&egraph, "cheap-1");
        assert_eq!(new_cheap_value, new_cheap_1_value);
        assert!(new_cheap_value != orig_cheap_value || new_cheap_1_value != orig_cheap_1_value);
        // Now verify that if we extract, it still respects the unextractable, even though it's a different values now
        egraph
            .parse_and_run_program(
                None,
                r#"
                (extract res)
                "#,
            )
            .unwrap();
        let report = egraph.get_extract_report().clone().unwrap();
        let ExtractReport::Best { term, termdag, .. } = report else {
            panic!();
        };
        let span = span!();
        let expr = termdag.term_to_expr(&term, span.clone());
        assert_eq!(expr, Expr::Call(span, Symbol::from("exp"), vec![]));
    }

    #[test]
    fn test_subsumed_unextractable_rebuild_self() {
        // Tests that a term stays unextractable even after a rebuild after a union change its output value.
        let mut egraph = EGraph::default();

        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype Math)
                (constructor container (Math) Math)
                (constructor exp () Math :cost 100)
                (constructor cheap () Math)
                (exp)
                (let x (cheap))
                (subsume (cheap))
                "#,
            )
            .unwrap();

        let orig_cheap_value = get_value(&egraph, "cheap");
        // Then we can union them
        egraph
            .parse_and_run_program(
                None,
                r#"
                (union (exp) x)
                "#,
            )
            .unwrap();
        // And verify that the cheap value is now different
        let new_cheap_value = get_value(&egraph, "cheap");
        assert_ne!(new_cheap_value, orig_cheap_value);

        // Now verify that if we extract, it still respects the subsumption, even though it's a different values now
        egraph
            .parse_and_run_program(
                None,
                r#"
                (extract x)
                "#,
            )
            .unwrap();
        let report = egraph.get_extract_report().clone().unwrap();
        let ExtractReport::Best { term, termdag, .. } = report else {
            panic!();
        };
        let span = span!();
        let expr = termdag.term_to_expr(&term, span.clone());
        assert_eq!(expr, Expr::Call(span, Symbol::from("exp"), vec![]));
    }
}
