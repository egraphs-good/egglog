//! # egglog
//! egglog is a language specialized for writing equality saturation
//! applications. It is the successor to the rust library [egg](https://github.com/egraphs-good/egg).
//! egglog is faster and more general than egg.
//!
//! # Documentation
//! Documentation for the egglog language can be found here: [`Command`].
//!
//! # Tutorial
//! We have a [text tutorial](https://egraphs-good.github.io/egglog-tutorial/01-basics.html) on egglog and how to use it.
//! We also have a slightly outdated [video tutorial](https://www.youtube.com/watch?v=N2RDQGRBrSY).
//!
//!
//!
pub mod ast;
#[cfg(feature = "bin")]
mod cli;
mod command_macro;
pub mod constraint;
mod core;
pub mod egraph_operations;
pub mod extract;
pub mod prelude;
pub mod scheduler;
mod serialize;
pub mod sort;
mod typechecking;
pub mod util;
pub use command_macro::{CommandMacro, CommandMacroRegistry};

// This is used to allow the `add_primitive` macro to work in
// both this crate and other crates by referring to `::egglog`.
extern crate self as egglog;
use ast::CanonicalizedVar;
use ast::*;
pub use ast::{ResolvedExpr, ResolvedFact, ResolvedVar};
#[cfg(feature = "bin")]
pub use cli::*;
use constraint::{Constraint, Problem, SimpleTypeConstraint, TypeConstraint};
pub use core::{Atom, AtomTerm, ResolvedCall, SpecializedPrimitive};
pub use core_relations::{BaseValue, ContainerValue, ExecutionState, Value};
use core_relations::{ExternalFunctionId, make_external_func};
use csv::Writer;
pub use egglog_add_primitive::add_primitive;
use egglog_ast::generic_ast::{Change, GenericExpr, Literal};
use egglog_ast::span::Span;
use egglog_ast::util::ListDisplay;
pub use egglog_bridge::FunctionRow;
pub use egglog_bridge::match_term_app;
pub use egglog_bridge::proof_format::{EqProofId, ProofStore, TermProofId};
use egglog_bridge::syntax::SyntaxId;
pub use egglog_bridge::termdag::{Term, TermDag, TermId};
use egglog_bridge::{ColumnTy, QueryEntry, SourceExpr, SourceSyntax, TopLevelLhsExpr};
use egglog_core_relations as core_relations;
use egglog_numeric_id as numeric_id;
use egglog_reports::{ReportLevel, RunReport};
use extract::{CostModel, DefaultCost, Extractor, TreeAdditiveCostModel};
use indexmap::IndexSet;
use indexmap::map::Entry;
use log::{Level, log_enabled};
use numeric_id::DenseIdMap;
use prelude::*;
use scheduler::{SchedulerId, SchedulerRecord};
pub use serialize::{SerializeConfig, SerializeOutput, SerializedNode};
use sort::*;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io::{Read, Write as _};
use std::iter::once;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use thiserror::Error;
pub use typechecking::TypeError;
pub use typechecking::TypeInfo;
use util::*;

use crate::ast::desugar::desugar_command;
use crate::ast::{CorrespondingVar, MappedExpr, MappedFact};
use crate::core::{GenericActionsExt, ResolvedRuleExt};

pub const GLOBAL_NAME_PREFIX: &str = "$";

pub type ArcSort = Arc<dyn Sort>;

/// A trait for implementing custom primitive operations in egglog.
///
/// Primitives are built-in functions that can be called in both rule queries and actions.
pub trait Primitive {
    /// Returns the name of this primitive operation.
    fn name(&self) -> &str;

    /// Constructs a type constraint for the primitive that uses the span information
    /// for error localization.
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint>;

    /// Applies the primitive operation to the given arguments.
    ///
    /// Returns `Some(value)` if the operation succeeds, or `None` if it fails.
    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value>;
}

/// A user-defined command output trait.
pub trait UserDefinedCommandOutput: Debug + std::fmt::Display + Send + Sync {}
impl<T> UserDefinedCommandOutput for T where T: Debug + std::fmt::Display + Send + Sync {}

/// Output from a command.
#[derive(Clone, Debug)]
pub enum CommandOutput {
    /// The size of a function
    PrintFunctionSize(usize),
    /// The name of all functions and their sizes
    PrintAllFunctionsSize(Vec<(String, usize)>),
    /// The best function found after extracting
    ExtractBest(TermDag, DefaultCost, Term),
    /// The variants of a function found after extracting
    ExtractVariants(TermDag, Vec<Term>),
    /// The report from all runs
    OverallStatistics(RunReport),
    /// A printed function and all its values
    PrintFunction(Function, TermDag, Vec<(Term, Term)>, PrintFunctionMode),
    /// The report from a single run
    RunSchedule(RunReport),
    /// A user defined output
    UserDefined(Arc<dyn UserDefinedCommandOutput>),
}

impl std::fmt::Display for CommandOutput {
    /// Format the command output for display, ending with a newline.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandOutput::PrintFunctionSize(size) => writeln!(f, "{}", size),
            CommandOutput::PrintAllFunctionsSize(names_and_sizes) => {
                for name in names_and_sizes {
                    writeln!(f, "{}: {}", name.0, name.1)?;
                }
                Ok(())
            }
            CommandOutput::ExtractBest(termdag, _cost, term) => {
                writeln!(f, "{}", termdag.to_string(term))
            }
            CommandOutput::ExtractVariants(termdag, terms) => {
                writeln!(f, "(")?;
                for expr in terms {
                    writeln!(f, "   {}", termdag.to_string(expr))?;
                }
                writeln!(f, ")")
            }
            CommandOutput::OverallStatistics(run_report) => {
                write!(f, "Overall statistics:\n{}", run_report)
            }
            CommandOutput::PrintFunction(function, termdag, terms_and_outputs, mode) => {
                let out_is_unit = function.schema.output.name() == UnitSort.name();
                if *mode == PrintFunctionMode::CSV {
                    let mut wtr = Writer::from_writer(vec![]);
                    for (term, output) in terms_and_outputs {
                        match term {
                            Term::App(name, children) => {
                                let mut values = vec![name.clone()];
                                for child_id in children {
                                    values.push(termdag.to_string(termdag.get(*child_id)));
                                }

                                if !out_is_unit {
                                    values.push(termdag.to_string(output));
                                }
                                wtr.write_record(&values).map_err(|_| std::fmt::Error)?;
                            }
                            _ => panic!("Expect function_to_dag to return a list of apps."),
                        }
                    }
                    let csv_bytes = wtr.into_inner().map_err(|_| std::fmt::Error)?;
                    f.write_str(&String::from_utf8(csv_bytes).map_err(|_| std::fmt::Error)?)
                } else {
                    writeln!(f, "(")?;
                    for (term, output) in terms_and_outputs.iter() {
                        write!(f, "   {}", termdag.to_string(term))?;
                        if !out_is_unit {
                            write!(f, " -> {}", termdag.to_string(output))?;
                        }
                        writeln!(f)?;
                    }
                    writeln!(f, ")")
                }
            }
            CommandOutput::RunSchedule(_report) => Ok(()),
            CommandOutput::UserDefined(output) => {
                write!(f, "{}", *output)
            }
        }
    }
}

/// The main interface for an e-graph in egglog.
///
/// An [`EGraph`] maintains a collection of equivalence classes of terms and provides
/// operations for adding facts, running rules, and extracting optimal terms.
///
/// # Examples
///
/// ```
/// use egglog::*;
///
/// let mut egraph = EGraph::default();
/// egraph.parse_and_run_program(None, "(datatype Math (Num i64) (Add Math Math))").unwrap();
/// ```
#[derive(Clone)]
pub struct EGraph {
    backend: egglog_bridge::EGraph,
    pub parser: Parser,
    names: check_shadowing::Names,
    /// pushed_egraph forms a linked list of pushed egraphs.
    /// Pop reverts the egraph to the last pushed egraph.
    pushed_egraph: Option<Box<Self>>,
    functions: IndexMap<String, Function>,
    rulesets: IndexMap<String, Ruleset>,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
    type_info: TypeInfo,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,
    schedulers: DenseIdMap<SchedulerId, SchedulerRecord>,
    commands: IndexMap<String, Arc<dyn UserDefinedCommand>>,
    strict_mode: bool,
    warned_about_missing_global_prefix: bool,
    /// Registry for command-level macros
    command_macros: CommandMacroRegistry,
}

/// A user-defined command allows users to inject custom command that can be called
/// in an egglog program.
///
/// Compared to an external function, a user-defined command is more powerful because
/// it has an exclusive access to the e-graph.
pub trait UserDefinedCommand: Send + Sync {
    /// Run the command with the given arguments.
    fn update(&self, egraph: &mut EGraph, args: &[Expr]) -> Result<Option<CommandOutput>, Error>;
}

/// A function in the e-graph.
///
/// This contains the schema information of the function and
/// the backend id of the function in the e-graph.
#[derive(Clone)]
pub struct Function {
    decl: ResolvedFunctionDecl,
    schema: ResolvedSchema,
    can_subsume: bool,
    backend_id: egglog_bridge::FunctionId,
}

impl Function {
    /// Get the name of the function.
    pub fn name(&self) -> &str {
        &self.decl.name
    }

    /// Get the schema of the function.
    pub fn schema(&self) -> &ResolvedSchema {
        &self.schema
    }

    /// Whether this function supports subsumption.
    pub fn can_subsume(&self) -> bool {
        self.can_subsume
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedSchema {
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

impl ResolvedSchema {
    /// Get the type at position `index`, counting the `output` sort as at position `input.len()`.
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

impl EGraph {
    fn new_from_backend(backend: egglog_bridge::EGraph) -> Self {
        let mut eg = Self {
            backend,
            parser: Default::default(),
            names: Default::default(),
            pushed_egraph: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            fact_directory: None,
            seminaive: true,
            overall_run_report: Default::default(),
            type_info: Default::default(),
            schedulers: Default::default(),
            commands: Default::default(),
            strict_mode: false,
            warned_about_missing_global_prefix: false,
            command_macros: Default::default(),
        };

        add_base_sort(&mut eg, UnitSort, span!()).unwrap();
        add_base_sort(&mut eg, StringSort, span!()).unwrap();
        add_base_sort(&mut eg, BoolSort, span!()).unwrap();
        add_base_sort(&mut eg, I64Sort, span!()).unwrap();
        add_base_sort(&mut eg, F64Sort, span!()).unwrap();
        add_base_sort(&mut eg, BigIntSort, span!()).unwrap();
        add_base_sort(&mut eg, BigRatSort, span!()).unwrap();
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

    /// Create a fresh e-graph with proof tracing enabled.
    ///
    /// Proofs are disabled by default. Use this constructor to enable proofs and provenance
    /// tracking.
    pub fn with_proofs() -> Self {
        Self::new_from_backend(egglog_bridge::EGraph::with_tracing())
    }

    /// Returns `true` if this e-graph was constructed with proofs enabled.
    pub fn proofs_enabled(&self) -> bool {
        self.backend.proofs_enabled()
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new_from_backend(Default::default())
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(String);

impl EGraph {
    /// Add a user-defined command to the e-graph
    /// Get the type information for this e-graph
    pub fn type_info(&mut self) -> &mut TypeInfo {
        &mut self.type_info
    }

    /// Get read-only access to the command macro registry
    pub fn command_macros(&self) -> &CommandMacroRegistry {
        &self.command_macros
    }

    /// Get mutable access to the command macro registry
    pub fn command_macros_mut(&mut self) -> &mut CommandMacroRegistry {
        &mut self.command_macros
    }

    pub fn add_command(
        &mut self,
        name: String,
        command: Arc<dyn UserDefinedCommand>,
    ) -> Result<(), Error> {
        if self.commands.contains_key(&name)
            || self.functions.contains_key(&name)
            || self.type_info.get_prims(&name).is_some()
        {
            return Err(Error::CommandAlreadyExists(name, span!()));
        }
        self.commands.insert(name.clone(), command);
        self.parser.add_user_defined(name)?;
        Ok(())
    }

    /// Configure whether globals missing the required `$` prefix are treated as errors.
    pub fn set_strict_mode(&mut self, strict_mode: bool) {
        self.strict_mode = strict_mode;
    }

    /// Returns `true` when missing `$` prefixes on globals are treated as errors.
    pub fn strict_mode(&self) -> bool {
        self.strict_mode
    }

    fn ensure_global_name_prefix(&mut self, span: &Span, name: &str) -> Result<(), TypeError> {
        if name.starts_with(GLOBAL_NAME_PREFIX) {
            return Ok(());
        }
        if self.strict_mode {
            Err(TypeError::GlobalMissingPrefix {
                name: name.to_owned(),
                span: span.clone(),
            })
        } else {
            self.warn_missing_global_prefix(span, name)?;
            Ok(())
        }
    }

    fn warn_missing_global_prefix(
        &mut self,
        span: &Span,
        canonical_name: &str,
    ) -> Result<(), TypeError> {
        if self.strict_mode {
            return Err(TypeError::NonGlobalPrefixed {
                name: format!("{}{}", GLOBAL_NAME_PREFIX, canonical_name),
                span: span.clone(),
            });
        }
        if self.warned_about_missing_global_prefix {
            return Ok(());
        }
        self.warned_about_missing_global_prefix = true;
        log::warn!(
            "{}\nGlobal `{}` should start with `{}`. Enable `--strict-mode` to turn this warning into an error. Suppressing additional warnings of this type.",
            span,
            canonical_name,
            GLOBAL_NAME_PREFIX
        );
        Ok(())
    }

    /// Push a snapshot of the e-graph into the stack.
    ///
    /// See [`EGraph::pop`].
    pub fn push(&mut self) {
        let prev_prev: Option<Box<Self>> = self.pushed_egraph.take();
        let mut prev = self.clone();
        prev.pushed_egraph = prev_prev;
        self.pushed_egraph = Some(Box::new(prev));
    }

    /// Pop the current egraph off the stack, replacing
    /// it with the previously pushed egraph.
    /// It preserves the run report and messages from the popped
    /// egraph.
    pub fn pop(&mut self) -> Result<(), Error> {
        match self.pushed_egraph.take() {
            Some(e) => {
                // Copy the overall report from the popped egraph
                let overall_run_report = self.overall_run_report.clone();
                *self = *e;
                self.overall_run_report = overall_run_report;
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
                let val = self.backend.literal_to_value(literal);
                Ok(egglog_bridge::MergeFn::Const(val))
            }
            GenericExpr::Var(span, resolved_var) => match resolved_var.name.as_str() {
                "old" => Ok(egglog_bridge::MergeFn::Old),
                "new" => Ok(egglog_bridge::MergeFn::New),
                // NB: type-checking should already catch unbound variables here.
                _ => Err(TypeError::Unbound(resolved_var.name.clone(), span.clone()).into()),
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
                    p.external_id(),
                    translated_args,
                ))
            }
        }
    }

    fn declare_function(&mut self, decl: &ResolvedFunctionDecl) -> Result<(), Error> {
        let get_sort = |name: &String| match self.type_info.get_sort_by_name(name) {
            Some(sort) => Ok(sort.clone()),
            None => Err(Error::TypeError(TypeError::UndefinedSort(
                name.to_owned(),
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
                FunctionSubtype::Relation => DefaultVal::Const(self.backend.base_values().get(())),
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
            fiat_reason_only: false,
        });

        let function = Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            can_subsume,
            backend_id,
        };

        let old = self.functions.insert(decl.name.clone(), function);
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
        sym: &str,
        n: usize,
        include_output: bool,
    ) -> Result<(Vec<Term>, Option<Vec<Term>>, TermDag), Error> {
        let func = self
            .functions
            .get(sym)
            .ok_or(TypeError::UnboundFunction(sym.to_owned(), span!()))?;
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
                inputs.push(termdag.app(sym.to_owned(), children));
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

    /// Print up to `n` the tuples in a given function.
    /// Print all tuples if `n` is not provided.
    pub fn print_function(
        &mut self,
        sym: &str,
        n: Option<usize>,
        file: Option<File>,
        mode: PrintFunctionMode,
    ) -> Result<Option<CommandOutput>, Error> {
        let n = match n {
            Some(n) => {
                log::info!("Printing up to {n} tuples of function {sym} as {mode}");
                n
            }
            None => {
                log::info!("Printing all tuples of function {sym} as {mode}");
                usize::MAX
            }
        };

        let (terms, outputs, termdag) = self.function_to_dag(sym, n, true)?;
        let f = self
            .functions
            .get(sym)
            // function_to_dag should have checked this
            .unwrap();
        let terms_and_outputs: Vec<_> = terms.into_iter().zip(outputs.unwrap()).collect();
        let output = CommandOutput::PrintFunction(f.clone(), termdag, terms_and_outputs, mode);
        match file {
            Some(mut file) => {
                log::info!("Writing output to file");
                file.write_all(output.to_string().as_bytes())
                    .expect("Error writing to file");
                Ok(None)
            }
            None => Ok(Some(output)),
        }
    }

    /// Print the size of a function. If no function name is provided,
    /// print the size of all functions in "name: len" pairs.
    pub fn print_size(&self, sym: Option<&str>) -> Result<CommandOutput, Error> {
        if let Some(sym) = sym {
            let f = self
                .functions
                .get(sym)
                .ok_or(TypeError::UnboundFunction(sym.to_owned(), span!()))?;
            let size = self.backend.table_size(f.backend_id);
            log::info!("Function {} has size {}", sym, size);
            Ok(CommandOutput::PrintFunctionSize(size))
        } else {
            // Print size of all functions
            let mut lens = self
                .functions
                .iter()
                .map(|(sym, f)| (sym.clone(), self.backend.table_size(f.backend_id)))
                .collect::<Vec<_>>();

            // Function name's alphabetical order
            lens.sort_by_key(|(name, _)| name.clone());
            if log_enabled!(Level::Info) {
                for (sym, len) in &lens {
                    log::info!("Function {} has size {}", sym, len);
                }
            }
            Ok(CommandOutput::PrintAllFunctionsSize(lens))
        }
    }

    // returns whether the egraph was updated
    fn run_schedule(&mut self, sched: &ResolvedSchedule) -> Result<RunReport, Error> {
        match sched {
            ResolvedSchedule::Run(span, config) => self.run_rules(span, config),
            ResolvedSchedule::Repeat(_span, limit, sched) => {
                let mut report = RunReport::default();
                for _i in 0..*limit {
                    let rec = self.run_schedule(sched)?;
                    let updated = rec.updated;
                    report.union(rec);
                    if !updated {
                        break;
                    }
                }
                Ok(report)
            }
            ResolvedSchedule::Saturate(_span, sched) => {
                let mut report = RunReport::default();
                loop {
                    let rec = self.run_schedule(sched)?;
                    let updated = rec.updated;
                    report.union(rec);
                    if !updated {
                        break;
                    }
                }
                Ok(report)
            }
            ResolvedSchedule::Sequence(_span, scheds) => {
                let mut report = RunReport::default();
                for sched in scheds {
                    report.union(self.run_schedule(sched)?);
                }
                Ok(report)
            }
        }
    }

    /// Extract a value to a [`TermDag`] and [`Term`] in the [`TermDag`] using the default cost model.
    /// See also [`EGraph::extract_value_with_cost_model`] for more control.
    pub fn extract_value(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<(TermDag, Term, DefaultCost), Error> {
        self.extract_value_with_cost_model(sort, value, TreeAdditiveCostModel::default())
    }

    /// Extract a value to a [`TermDag`] and [`Term`] in the [`TermDag`].
    /// Note that the `TermDag` may contain a superset of the nodes in the `Term`.
    /// See also [`EGraph::extract_value_to_string`] for convenience.
    pub fn extract_value_with_cost_model<CM: CostModel<DefaultCost> + 'static>(
        &self,
        sort: &ArcSort,
        value: Value,
        cost_model: CM,
    ) -> Result<(TermDag, Term, DefaultCost), Error> {
        let extractor =
            Extractor::compute_costs_from_rootsorts(Some(vec![sort.clone()]), self, cost_model);
        let mut termdag = TermDag::default();
        let (cost, term) = extractor.extract_best(self, &mut termdag, value).unwrap();
        Ok((termdag, term, cost))
    }

    /// Extract a value to a string for printing.
    /// See also [`EGraph::extract_value`] for more control.
    pub fn extract_value_to_string(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<(String, DefaultCost), Error> {
        let (termdag, term, cost) = self.extract_value(sort, value)?;
        Ok((termdag.to_string(&term), cost))
    }

    fn run_rules(&mut self, span: &Span, config: &ResolvedRunConfig) -> Result<RunReport, Error> {
        let mut report: RunReport = Default::default();

        let GenericRunConfig { ruleset, until } = config;

        if let Some(facts) = until {
            if self.check_facts(span, facts).is_ok() {
                log::info!(
                    "Breaking early because of facts:\n {}!",
                    ListDisplay(facts, "\n")
                );
                return Ok(report);
            }
        }

        let subreport = self.step_rules(ruleset)?;
        report.union(subreport);

        if log_enabled!(Level::Debug) {
            log::debug!("database size: {}", self.num_tuples());
        }

        Ok(report)
    }

    /// Runs a ruleset for an iteration.
    ///
    /// This applies every match it finds (under semi-naive).
    /// See [`EGraph::step_rules_with_scheduler`] for more fine-grained control.
    ///
    /// This will return an error if an egglog primitive returns None in an action.
    pub fn step_rules(&mut self, ruleset: &str) -> Result<RunReport, Error> {
        fn collect_rule_ids(
            ruleset: &str,
            rulesets: &IndexMap<String, Ruleset>,
            ids: &mut Vec<egglog_bridge::RuleId>,
        ) {
            match &rulesets[ruleset] {
                Ruleset::Rules(rules) => {
                    for (_, id) in rules.values() {
                        ids.push(*id);
                    }
                }
                Ruleset::Combined(sub_rulesets) => {
                    for sub_ruleset in sub_rulesets {
                        collect_rule_ids(sub_ruleset, rulesets, ids);
                    }
                }
            }
        }

        let mut rule_ids = Vec::new();
        collect_rule_ids(ruleset, &self.rulesets, &mut rule_ids);

        let iteration_report = self
            .backend
            .run_rules(&rule_ids)
            .map_err(|e| Error::BackendError(e.to_string()))?;

        Ok(RunReport::singleton(ruleset, iteration_report))
    }

    fn add_rule(&mut self, rule: ast::ResolvedRule) -> Result<String, Error> {
        let canonical =
            rule.to_canonicalized_core_rule(&self.type_info, &mut self.parser.symbol_gen)?;
        let (query, actions) = (&canonical.rule.body, &canonical.rule.head);

        let rule_id = {
            let mut translator = BackendRule::new(
                self.backend.new_rule(&rule.name, self.seminaive),
                &self.functions,
                &self.type_info,
                canonical.mapped_facts.clone(),
            );
            translator.query(query, false);
            translator.actions(actions)?;
            translator.build()
        };

        if let Some(rules) = self.rulesets.get_mut(&rule.ruleset) {
            match rules {
                Ruleset::Rules(rules) => {
                    match rules.entry(rule.name.clone()) {
                        indexmap::map::Entry::Occupied(_) => {
                            let name = rule.name;
                            panic!("Rule '{name}' was already present")
                        }
                        indexmap::map::Entry::Vacant(e) => e.insert((canonical.rule, rule_id)),
                    };
                    Ok(rule.name)
                }
                Ruleset::Combined(_) => Err(Error::CombinedRulesetError(rule.ruleset, rule.span)),
            }
        } else {
            Err(Error::NoSuchRuleset(rule.ruleset, rule.span))
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
            Vec::new(),
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

    /// Evaluates an expression, returns the sort of the expression and the evaluation result.
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, Value), Error> {
        let span = expr.span();
        let command = Command::Action(Action::Expr(span.clone(), expr.clone()));
        let resolved_commands = self.resolve_command(command)?;
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
        let unit_id = self.backend.base_values().get_ty::<()>();
        let unit_val = self.backend.base_values().get(());

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
            Vec::new(),
        );

        let result_var = ResolvedVar {
            name: self.parser.symbol_gen.fresh("eval_resolved_expr"),
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

        let arg = translator.entry(&core::CanonicalizedResolvedAtomTerm::Var(
            span.clone(),
            CanonicalizedVar::new_current(result_var),
        ));
        translator.rb.call_external_func(
            ext_id,
            &[arg],
            egglog_bridge::ColumnTy::Base(unit_id),
            || "this function will never panic".to_string(),
        );

        let id = translator.build();
        let rule_result = self.backend.run_rules(&[id]);
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);
        let _ = rule_result.map_err(|e| {
            Error::BackendError(format!("Failed to evaluate expression '{}': {}", expr, e))
        })?;

        let result = result.lock().unwrap().unwrap();
        Ok(result)
    }
}

impl EGraph {
    fn add_combined_ruleset(&mut self, name: String, rulesets: Vec<String>) {
        match self.rulesets.entry(name.clone()) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Combined(rulesets)),
        };
    }

    fn add_ruleset(&mut self, name: String) {
        match self.rulesets.entry(name.clone()) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Rules(Default::default())),
        };
    }

    fn check_facts(&mut self, span: &Span, facts: &[ResolvedFact]) -> Result<(), Error> {
        let fresh_name = self.parser.symbol_gen.fresh("check_facts");
        let fresh_ruleset = self.parser.symbol_gen.fresh("check_facts_ruleset");
        let rule = ast::ResolvedRule {
            span: span.clone(),
            head: ResolvedActions::default(),
            body: facts.to_vec(),
            name: fresh_name.clone(),
            ruleset: fresh_ruleset.clone(),
        };
        let canonical_rule =
            rule.to_canonicalized_core_rule(&self.type_info, &mut self.parser.symbol_gen)?;
        let query = canonical_rule.rule.body.clone();

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
            canonical_rule.mapped_facts,
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

    fn run_command(&mut self, command: ResolvedNCommand) -> Result<Option<CommandOutput>, Error> {
        match command {
            // Sorts are already declared during typechecking
            ResolvedNCommand::Sort(_span, name, _presort_and_args) => {
                log::info!("Declared sort {}.", name)
            }
            ResolvedNCommand::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                log::info!("Declared {} {}.", fdecl.subtype, fdecl.name)
            }
            ResolvedNCommand::AddRuleset(_span, name) => {
                self.add_ruleset(name.clone());
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::UnstableCombinedRuleset(_span, name, others) => {
                self.add_combined_ruleset(name.clone(), others);
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::NormRule { rule } => {
                let name = rule.name.clone();
                self.add_rule(rule)?;
                log::info!("Declared rule {name}.")
            }
            ResolvedNCommand::RunSchedule(sched) => {
                let report = self.run_schedule(&sched)?;
                log::info!("Ran schedule {}.", sched);
                log::info!("Report: {}", report);
                self.overall_run_report.union(report.clone());
                return Ok(Some(CommandOutput::RunSchedule(report)));
            }
            ResolvedNCommand::PrintOverallStatistics(span, file) => match file {
                None => {
                    log::info!("Printed overall statistics");
                    return Ok(Some(CommandOutput::OverallStatistics(
                        self.overall_run_report.clone(),
                    )));
                }
                Some(path) => {
                    let mut file = std::fs::File::create(&path)
                        .map_err(|e| Error::IoError(path.clone().into(), e, span.clone()))?;
                    log::info!("Printed overall statistics to json file {}", path);

                    serde_json::to_writer(&mut file, &self.overall_run_report)
                        .expect("error serializing to json");
                }
            },
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
                let n: i64 = self.backend.base_values().unwrap(n);

                let mut termdag = TermDag::default();

                let extractor = Extractor::compute_costs_from_rootsorts(
                    Some(vec![sort]),
                    self,
                    TreeAdditiveCostModel::default(),
                );
                return if n == 0 {
                    if let Some((cost, term)) = extractor.extract_best(self, &mut termdag, x) {
                        // dont turn termdag into a string if we have messages disabled for performance reasons
                        if log_enabled!(Level::Info) {
                            log::info!("extracted with cost {cost}: {}", termdag.to_string(&term));
                        }
                        Ok(Some(CommandOutput::ExtractBest(termdag, cost, term)))
                    } else {
                        Err(Error::ExtractError(
                            "Unable to find any valid extraction (likely due to subsume or delete)"
                                .to_string(),
                        ))
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
                    if log_enabled!(Level::Info) {
                        let expr_str = expr.to_string();
                        log::info!("extracted {} variants for {expr_str}", terms.len());
                    }
                    Ok(Some(CommandOutput::ExtractVariants(termdag, terms)))
                };
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
            ResolvedNCommand::PrintFunction(span, f, n, file, mode) => {
                let file = file
                    .map(|file| {
                        std::fs::File::create(&file)
                            .map_err(|e| Error::IoError(file.into(), e, span.clone()))
                    })
                    .transpose()?;
                return self.print_function(&f, n, file, mode).map_err(|e| match e {
                    Error::TypeError(TypeError::UnboundFunction(f, _)) => {
                        Error::TypeError(TypeError::UnboundFunction(f, span.clone()))
                    }
                    // This case is currently impossible
                    _ => e,
                });
            }
            ResolvedNCommand::PrintSize(span, f) => {
                let res = self.print_size(f.as_deref()).map_err(|e| match e {
                    Error::TypeError(TypeError::UnboundFunction(f, _)) => {
                        Error::TypeError(TypeError::UnboundFunction(f, span.clone()))
                    }
                    // This case is currently impossible
                    _ => e,
                })?;
                return Ok(Some(res));
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
                self.input_file(&name, file)?;
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
                let res = command.update(self, &exprs);
                self.commands.insert(name, command);
                return res;
            }
        };

        Ok(None)
    }

    fn input_file(&mut self, func_name: &str, file: String) -> Result<(), Error> {
        let function_type = self
            .type_info
            .get_func_type(func_name)
            .unwrap_or_else(|| panic!("Unrecognized function name {}", func_name));
        let func = self.functions.get_mut(func_name).unwrap();

        let mut filename = self.fact_directory.clone().unwrap_or_default();
        filename.push(file.as_str());

        // check that the function uses supported types

        for t in &func.schema.input {
            match t.name() {
                "i64" | "f64" | "String" => {}
                s => panic!("Unsupported type {} for input", s),
            }
        }

        if function_type.subtype != FunctionSubtype::Constructor {
            match func.schema.output.name() {
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

        let unit_val = self.backend.base_values().get(());

        for line in contents.lines() {
            let mut it = line.split('\t').map(|s| s.trim());

            let mut row: Vec<Value> = Vec::with_capacity(row_schema.len());

            for sort in row_schema.iter() {
                if let Some(raw) = it.next() {
                    let val = match sort.name() {
                        "i64" => {
                            if let Ok(i) = raw.parse::<i64>() {
                                self.backend.base_values().get(i)
                            } else {
                                return Err(Error::InputFileFormatError(file));
                            }
                        }
                        "f64" => {
                            if let Ok(f) = raw.parse::<f64>() {
                                self.backend
                                    .base_values()
                                    .get::<F>(core_relations::Boxed::new(f.into()))
                            } else {
                                return Err(Error::InputFileFormatError(file));
                            }
                        }
                        "String" => self.backend.base_values().get::<S>(raw.to_string().into()),
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

        let mut table_action = egglog_bridge::TableAction::new(&self.backend, func.backend_id);

        if function_type.subtype != FunctionSubtype::Constructor {
            self.backend.with_execution_state(|es| {
                for row in parsed_contents.iter() {
                    table_action.insert(es, row.iter().copied());
                }
                Some(unit_val)
            });
        } else {
            self.backend.with_execution_state(|es| {
                for row in parsed_contents.iter() {
                    table_action.lookup(es, row);
                }
                Some(unit_val)
            });
        }

        self.backend.flush_updates();

        log::info!("Read {num_facts} facts into {func_name} from '{file}'.");
        Ok(())
    }

    /// Desugars, typechecks, and removes globals from a single [`Command`].
    /// Leverages previous type information in the [`EGraph`] to do so, adding new type information.
    fn resolve_command(&mut self, command: Command) -> Result<Vec<ResolvedNCommand>, Error> {
        let desugared = desugar_command(command, &mut self.parser)?;
        let mut typechecked = self.typecheck_program(&desugared)?;

        typechecked = remove_globals::remove_globals(typechecked, &mut self.parser.symbol_gen);
        for command in &typechecked {
            self.names.check_shadowing(command)?;
        }

        Ok(typechecked)
    }

    /// Run a program, returning the desugared outputs as well as the CommandOutputs.
    /// Can optionally not run the commands, just adding type information.
    fn process_program_internal(
        &mut self,
        program: Vec<Command>,
        run_commands: bool,
    ) -> Result<(Vec<CommandOutput>, Vec<ResolvedCommand>), Error> {
        let mut outputs = Vec::new();
        let mut desugared_commands = Vec::new();

        for before_expanded_command in program {
            // First do user-provided macro expansion for this command,
            // which may rely on type information from previous commands.
            let macro_expanded = self.command_macros.apply(
                before_expanded_command,
                &mut self.parser.symbol_gen,
                &self.type_info,
            )?;

            for command in macro_expanded {
                // handle include specially- we keep them as-is for desugaring
                if let Command::Include(span, file) = &command {
                    let s = std::fs::read_to_string(file)
                        .unwrap_or_else(|_| panic!("{span} Failed to read file {file}"));
                    let included_program = self
                        .parser
                        .get_program_from_string(Some(file.clone()), &s)?;
                    // run program internal on these include commands
                    let (included_outputs, _included_desugared) =
                        self.process_program_internal(included_program, run_commands)?;
                    outputs.extend(included_outputs);
                    desugared_commands.push(ResolvedCommand::Include(span.clone(), file.clone()));
                } else {
                    for processed in self.resolve_command(command)? {
                        desugared_commands.push(processed.to_command());

                        // even in desugar mode we still run push and pop
                        if run_commands
                            || matches!(
                                processed,
                                ResolvedNCommand::Push(_) | ResolvedNCommand::Pop(_, _)
                            )
                        {
                            let result = self.run_command(processed)?;
                            if let Some(output) = result {
                                outputs.push(output);
                            }
                        }
                    }
                }
            }
        }

        Ok((outputs, desugared_commands))
    }

    /// Run a program, represented as an AST.
    /// Return a list of messages.
    pub fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<CommandOutput>, Error> {
        let (outputs, _desugared_commands) = self.process_program_internal(program, true)?;
        Ok(outputs)
    }

    /// Desugars an egglog program by parsing and desugaring each command.
    /// Outputs a new egglog program without any syntactic sugar, either user provided ([`CommandMacro`]) or built-in (e.g., `rewrite` commands).
    pub fn desugar_program(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<ResolvedCommand>, Error> {
        let parsed = self.parser.get_program_from_string(filename, input)?;
        let (_outputs, desugared_commands) = self.process_program_internal(parsed, false)?;
        Ok(desugared_commands)
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
    ) -> Result<Vec<CommandOutput>, Error> {
        let parsed = self.parser.get_program_from_string(filename, input)?;
        self.run_program(parsed)
    }

    /// Get the number of tuples in the database.
    ///
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
    pub fn get_sort_by_name(&self, sym: &str) -> Option<&ArcSort> {
        self.type_info.get_sort_by_name(sym)
    }

    /// Gets the overall run report and returns it.
    pub fn get_overall_run_report(&self) -> &RunReport {
        &self.overall_run_report
    }

    /// Convert from an egglog value to a Rust type.
    pub fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.backend.base_values().unwrap::<T>(x)
    }

    /// Convert from a Rust type to an egglog value.
    pub fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.backend.base_values().get::<T>(x)
    }

    /// Convert from an egglog value to a reference of a Rust container type.
    ///
    /// Returns `None` if the value cannot be converted to the requested container type.
    ///
    /// Warning: The return type of this function may contain lock guards.
    /// Attempts to modify the contents of the containers database may deadlock if the given guard has not been dropped.
    pub fn value_to_container<T: ContainerValue>(
        &self,
        x: Value,
    ) -> Option<impl Deref<Target = T>> {
        self.backend.container_values().get_val::<T>(x)
    }

    /// Convert from a Rust container type to an egglog value.
    pub fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        self.backend.with_execution_state(|state| {
            self.backend.container_values().register_val::<T>(x, state)
        })
    }

    /// Get the size of a function in the e-graph.
    ///
    /// `panics` if the function does not exist.
    pub fn get_size(&self, func: &str) -> usize {
        let function_id = self.functions.get(func).unwrap().backend_id;
        self.backend.table_size(function_id)
    }

    /// Lookup a tuple in afunction in the e-graph.
    ///
    /// Returns `None` if the tuple does not exist.
    /// `panics` if the function does not exist.
    pub fn lookup_function(&self, name: &str, key: &[Value]) -> Option<Value> {
        let func = self.functions.get(name).unwrap().backend_id;
        self.backend.lookup_id(func, key)
    }

    /// Get a function by name.
    ///
    /// Returns `None` if the function does not exist.
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    pub fn set_report_level(&mut self, level: ReportLevel) {
        self.backend.set_report_level(level);
    }

    /// A basic method for dumping the state of the database to `log::info!`.
    ///
    /// For large tables, this is unlikely to give particularly useful output.
    pub fn dump_debug_info(&self) {
        self.backend.dump_debug_info();
    }

    /// Get the canonical representation for `val` based on type.
    pub fn get_canonical_value(&self, val: Value, sort: &ArcSort) -> Value {
        self.backend
            .get_canon_repr(val, sort.column_ty(&self.backend))
    }

    /// Generate a proof explaining how a term was constructed in the e-graph.
    ///
    /// This method requires that the e-graph was created with [`EGraph::with_proofs()`].
    /// The proof is stored in the provided `ProofStore` and can be printed or inspected.
    ///
    /// # Arguments
    /// * `id` - The value representing the term to explain
    /// * `store` - A mutable reference to a `ProofStore` where the proof will be stored
    ///
    /// # Returns
    /// A `TermProofId` that can be used to print or traverse the proof.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Proofs are not enabled (e-graph not created with `with_proofs()`)
    /// - The proof cannot be reconstructed
    ///
    /// # Example
    /// ```
    /// # use egglog::prelude::*;
    /// # use egglog::ProofStore;
    /// let mut egraph = EGraph::with_proofs();
    /// egraph.parse_and_run_program(None, "
    ///     (datatype Math (Num i64) (Add Math Math))
    ///     (let x (Add (Num 1) (Num 2)))
    /// ").unwrap();
    ///
    /// // Get the value for x
    /// let (_, x_value) = egraph.eval_expr(&expr!(x)).unwrap();
    ///
    /// // Generate a proof explaining how x was constructed
    /// let mut store = ProofStore::default();
    /// let proof_id = egraph.explain_term(x_value, &mut store).unwrap();
    ///
    /// // Print the proof
    /// store.print_term_proof(proof_id, &mut std::io::stdout()).unwrap();
    /// ```
    pub fn explain_term(
        &mut self,
        id: Value,
        store: &mut ProofStore,
    ) -> egglog_bridge::Result<TermProofId> {
        self.backend.explain_term(id, store)
    }

    /// Generate a proof explaining why two terms are equal in the e-graph.
    ///
    /// This method requires that the e-graph was created with [`EGraph::with_proofs()`].
    /// The proof shows the sequence of rewrites that establish the equality between the two terms.
    ///
    /// # Arguments
    /// * `id1` - The value representing the first term
    /// * `id2` - The value representing the second term
    /// * `store` - A mutable reference to a `ProofStore` where the proof will be stored
    ///
    /// # Returns
    /// An `EqProofId` that can be used to print or traverse the equality proof.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Proofs are not enabled (e-graph not created with `with_proofs()`)
    /// - The two terms are not actually equal in the e-graph
    /// - The proof cannot be reconstructed
    ///
    /// # Example
    /// ```
    /// # use egglog::prelude::*;
    /// # use egglog::ProofStore;
    /// let mut egraph = EGraph::with_proofs();
    /// egraph.parse_and_run_program(None, "
    ///     (datatype Math (Num i64) (Add Math Math))
    ///     (rule ((Add x y)) ((union (Add x y) (Add y x))))
    ///     (let a (Add (Num 1) (Num 2)))
    ///     (let b (Add (Num 2) (Num 1)))
    ///     (run 1)
    /// ").unwrap();
    ///
    /// // Get the values for a and b
    /// let (_, a_value) = egraph.eval_expr(&expr!(a)).unwrap();
    /// let (_, b_value) = egraph.eval_expr(&expr!(b)).unwrap();
    ///
    /// // Generate a proof that a and b are equal
    /// let mut store = ProofStore::default();
    /// let proof_id = egraph.explain_terms_equal(a_value, b_value, &mut store).unwrap();
    ///
    /// // Print the proof
    /// store.print_eq_proof(proof_id, &mut std::io::stdout()).unwrap();
    /// ```
    pub fn explain_terms_equal(
        &mut self,
        id1: Value,
        id2: Value,
        store: &mut ProofStore,
    ) -> egglog_bridge::Result<EqProofId> {
        self.backend.explain_terms_equal(id1, id2, store)
    }
}

struct BackendRule<'a> {
    rb: egglog_bridge::RuleBuilder<'a>,
    entries: HashMap<core::CanonicalizedResolvedAtomTerm, QueryEntry>,
    functions: &'a IndexMap<String, Function>,
    type_info: &'a TypeInfo,
    var_types: DenseIdMap<egglog_bridge::VariableId, ColumnTy>,
    atoms: DenseIdMap<egglog_bridge::VariableId, egglog_bridge::AtomId>,
    resolved_var_entries: HashMap<ResolvedVar, QueryEntry>,
    proof_state: Option<ProofState>,
}

#[derive(Debug)]
struct FunctionInfo {
    atom: egglog_bridge::AtomId,
    backend_id: egglog_bridge::FunctionId,
}

#[derive(Debug)]
struct PrimInfo {
    var: egglog_bridge::VariableId,
    ty: ColumnTy,
    func: ExternalFunctionId,
    name: Arc<str>,
}

struct ProofState {
    facts: Vec<MappedFact<ResolvedCall, ResolvedVar>>,
    function_info: HashMap<String, FunctionInfo>,
    prim_info: HashMap<String, PrimInfo>,
}

struct SyntaxBuilder<'a> {
    env: SourceSyntax,
    proof_state: &'a ProofState,
    egraph: &'a egglog_bridge::EGraph,
    var_map: &'a HashMap<core::CanonicalizedResolvedAtomTerm, QueryEntry>,
    var_types: &'a DenseIdMap<egglog_bridge::VariableId, ColumnTy>,
}

impl SyntaxBuilder<'_> {
    fn reconstruct_syntax(mut self) -> SourceSyntax {
        for fact in &self.proof_state.facts {
            // To use the handy "?". See the comment on `reconstruct_expr` for why this can fail.
            || -> Option<()> {
                match fact {
                    GenericFact::Eq(_, l, r) => {
                        let l_id = self.reconstruct_expr(l)?;
                        let r_id = self.reconstruct_expr(r)?;
                        self.env.add_toplevel_expr(TopLevelLhsExpr::Eq(l_id, r_id));
                    }
                    GenericFact::Fact(expr) => {
                        let id = self.reconstruct_expr(expr)?;
                        self.env.add_toplevel_expr(TopLevelLhsExpr::Exists(id));
                    }
                }
                Some(())
            }();
        }
        self.env
    }

    /// Generate the corresponding [`SyntaxId`] for an expression.
    ///
    /// This returns an Option<SyntaxId> because the source syntax can have expressions involving
    /// primitives where no actual Syntax needs to be passed down to proofs. For example, a
    /// primitive call asserting that `(= 2 (+ 1 1))` will have no corresponding substitution
    /// available, and even something like `(Num x) (= x (+ x x))` can easily be run at proof-check
    /// time, with no added auxiliary variables needed to be stored in the DB for the (= x (+ x x))
    /// atom.
    ///
    /// In cases like this `reconstruct_expr` returns None and the rest of the process
    /// short-circuits.
    fn reconstruct_expr(
        &mut self,
        expr: &GenericExpr<CorrespondingVar<ResolvedCall, ResolvedVar>, ResolvedVar>,
    ) -> Option<SyntaxId> {
        Some(match expr {
            GenericExpr::Var(span, var) => {
                let Some(qe) = self.var_map.get(&core::CanonicalizedResolvedAtomTerm::Var(
                    span.clone(),
                    CanonicalizedVar::new_current(var.clone()),
                )) else {
                    panic!("no mapping found for variable {var} [span={span}]")
                };
                let QueryEntry::Var(v) = qe else {
                    panic!(
                        "found a non-variable entry mapped from a variable {var} [span={span}], instead found {qe:?}"
                    );
                };
                let ty = self.var_types[v.id];
                self.env.add_expr(SourceExpr::Var {
                    id: v.id,
                    ty,
                    name: var.name().into(),
                })
            }
            GenericExpr::Lit(_, lit) => {
                let (val, ty) = self.egraph.literal_to_typed_constant(lit);
                self.env.add_expr(SourceExpr::Const { ty, val })
            }

            GenericExpr::Call(_, CorrespondingVar { head, to }, children) => {
                let mut any_failed = false;
                let args: Vec<_> = children
                    .iter()
                    .filter_map(|child| {
                        let res = self.reconstruct_expr(child);
                        any_failed |= res.is_none();
                        res
                    })
                    .collect();
                if any_failed {
                    return None;
                }
                match head {
                    ResolvedCall::Func(_) => {
                        let FunctionInfo { atom, backend_id } =
                            self.proof_state.function_info[to.name()];
                        self.env.add_expr(SourceExpr::FunctionCall {
                            func: backend_id,
                            atom,
                            args,
                        })
                    }
                    ResolvedCall::Primitive(_) => {
                        let PrimInfo {
                            var,
                            ty,
                            func,
                            name,
                        } = self.proof_state.prim_info.get(to.name())?;
                        self.env.add_expr(SourceExpr::ExternalCall {
                            var: *var,
                            ty: *ty,
                            func: *func,
                            name: name.clone(),
                            args,
                        })
                    }
                }
            }
        })
    }
}

impl ProofState {
    fn new(facts: Vec<MappedFact<ResolvedCall, ResolvedVar>>) -> ProofState {
        ProofState {
            facts,
            function_info: Default::default(),
            prim_info: Default::default(),
        }
    }

    fn record_call(&mut self, res_var: String, func: FunctionInfo) {
        self.function_info.insert(res_var, func);
    }

    fn record_prim(&mut self, res_var: String, prim: PrimInfo) {
        self.prim_info.insert(res_var, prim);
    }
}

impl<'a> BackendRule<'a> {
    fn new(
        rb: egglog_bridge::RuleBuilder<'a>,
        functions: &'a IndexMap<String, Function>,
        type_info: &'a TypeInfo,
        mapped_facts: Vec<MappedFact<ResolvedCall, ResolvedVar>>,
    ) -> BackendRule<'a> {
        let proofs_enabled = rb.egraph().proofs_enabled();

        BackendRule {
            rb,
            functions,
            type_info,
            entries: Default::default(),
            var_types: Default::default(),
            atoms: Default::default(),
            resolved_var_entries: Default::default(),
            proof_state: proofs_enabled.then(move || ProofState::new(mapped_facts)),
        }
    }

    fn entry(&mut self, x: &core::CanonicalizedResolvedAtomTerm) -> QueryEntry {
        self.entries
            .entry(x.clone())
            .or_insert_with(|| match x {
                core::GenericAtomTerm::Var(_, v) => {
                    let ty = v.var.sort.column_ty(self.rb.egraph());
                    let entry = self.rb.new_var_named(ty, &v.var.name);
                    if let QueryEntry::Var(var) = &entry {
                        self.var_types.insert(var.id, ty);
                        self.resolved_var_entries
                            .insert(v.var.clone(), entry.clone());
                    }
                    entry
                }
                core::GenericAtomTerm::Literal(_, l) => self.rb.egraph().literal_to_entry(l),
                core::GenericAtomTerm::Global(_, _) => {
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
        args: &[core::CanonicalizedResolvedAtomTerm],
    ) -> (ExternalFunctionId, Vec<QueryEntry>, ColumnTy) {
        let mut qe_args = self.args(args);

        if prim.name() == "unstable-fn" {
            let core::CanonicalizedResolvedAtomTerm::Literal(_, Literal::String(ref name)) =
                args[0]
            else {
                panic!("expected string literal after `unstable-fn`")
            };
            let id = if let Some(f) = self.type_info.get_func_type(name) {
                ResolvedFunctionId::Lookup(egglog_bridge::TableAction::new(
                    self.rb.egraph(),
                    self.func(f),
                ))
            } else if let Some(possible) = self.type_info.get_prims(name) {
                let mut ps: Vec<_> = possible.iter().collect();
                ps.retain(|p| {
                    self.type_info
                        .get_sorts::<FunctionSort>()
                        .into_iter()
                        .any(|f| {
                            let types: Vec<_> = prim
                                .input()
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
            let partial_arcsorts = prim.input().iter().skip(1).cloned().collect();

            qe_args[0] = self.rb.egraph().base_value_constant(ResolvedFunction {
                id,
                partial_arcsorts,
                name: name.clone(),
            });
        }

        (
            prim.external_id(),
            qe_args,
            prim.output().column_ty(self.rb.egraph()),
        )
    }

    fn args<'b>(
        &mut self,
        args: impl IntoIterator<Item = &'b core::CanonicalizedResolvedAtomTerm>,
    ) -> Vec<QueryEntry> {
        args.into_iter().map(|x| self.entry(x)).collect()
    }

    fn canon_term(&self, term: &core::ResolvedAtomTerm) -> core::CanonicalizedResolvedAtomTerm {
        match term {
            core::GenericAtomTerm::Var(span, v) => {
                core::GenericAtomTerm::Var(span.clone(), CanonicalizedVar::new_current(v.clone()))
            }
            core::GenericAtomTerm::Literal(span, lit) => {
                core::GenericAtomTerm::Literal(span.clone(), lit.clone())
            }
            core::GenericAtomTerm::Global(span, v) => core::GenericAtomTerm::Global(
                span.clone(),
                CanonicalizedVar::new_current(v.clone()),
            ),
        }
    }

    fn canon_args<'b>(
        &self,
        args: impl IntoIterator<Item = &'b core::ResolvedAtomTerm>,
    ) -> Vec<core::CanonicalizedResolvedAtomTerm> {
        args.into_iter().map(|a| self.canon_term(a)).collect()
    }

    fn query(
        &mut self,
        query: &core::Query<ResolvedCall, CanonicalizedVar<ResolvedVar>>,
        include_subsumed: bool,
    ) {
        for atom in &query.atoms {
            match &atom.head {
                ResolvedCall::Func(f) => {
                    let f_id = self.func(f);
                    let args = self.args(&atom.args);
                    let is_subsumed = match include_subsumed {
                        true => None,
                        false => Some(false),
                    };
                    let atom_id = self.rb.query_table(f_id, &args, is_subsumed).unwrap();
                    self.proof_state.as_mut().map(|ps| -> Option<()> {
                        let last = atom.args.last()?;
                        let core::CanonicalizedResolvedAtomTerm::Var(_span, var) = last else {
                            panic!("expected unique variable as last argument to a query, instead got {atom:?}");
                        };
                        ps.record_call(
                            var.orig.name().into(),
                            FunctionInfo { atom: atom_id, backend_id:f_id  },
                        );
                        Some(())
                    });
                    if let Some(QueryEntry::Var(var)) = args.last() {
                        self.atoms.insert(var.id, atom_id);
                    }
                }
                ResolvedCall::Primitive(p) => {
                    let (ext_id, args, ty) = self.prim(p, &atom.args);
                    self.proof_state.as_mut().map(|ps| -> Option<()> {
                        let Some(QueryEntry::Var(var)) = args.last() else {
                            // This is an assertion about the output of some primitive. The
                            // primitive itself is not returning a useful variable for other atoms
                            // in the query. The proof checker as a result does not need to know
                            // anything more about this call.
                            return None;
                        };
                        let core::CanonicalizedResolvedAtomTerm::Var(_span, res_var) =
                            atom.args.last()?
                        else {
                            panic!("expected unique variable as last argument to a prim query, instead got {atom:?}");
                        };
                        ps.record_prim(
                            res_var.orig.name().into(),
                            PrimInfo {
                                var: var.id,
                                ty,
                                func: ext_id,
                                name: p.name().into(),
                            },
                        );
                        Some(())
                    });
                    self.rb.query_prim(ext_id, &args, ty).unwrap();
                }
            }
        }
    }

    fn actions(&mut self, actions: &core::ResolvedCoreActions) -> Result<(), Error> {
        for action in &actions.0 {
            match action {
                core::GenericCoreAction::Let(span, v, f, args) => {
                    let canon_v = CanonicalizedVar::new_current(v.clone());
                    let v = core::GenericAtomTerm::Var(span.clone(), canon_v.clone());
                    let canon_args = self.canon_args(args);
                    let y = match f {
                        ResolvedCall::Func(f) => {
                            let name = f.name.clone();
                            let f = self.func(f);
                            let args = self.args(canon_args.iter());
                            let span = span.clone();
                            self.rb.lookup(f, &args, move || {
                                format!("{span}: lookup of function {name} failed")
                            })
                        }
                        ResolvedCall::Primitive(p) => {
                            let name = p.name().to_owned();
                            let (p, args, ty) = self.prim(p, &canon_args);
                            let span = span.clone();
                            self.rb.call_external_func(p, &args, ty, move || {
                                format!("{span}: call of primitive {name} failed")
                            })
                        }
                    };
                    self.entries.insert(v, y.into());
                }
                core::GenericCoreAction::LetAtomTerm(span, v, x) => {
                    let v = core::GenericAtomTerm::Var(
                        span.clone(),
                        CanonicalizedVar::new_current(v.clone()),
                    );
                    let x = self.entry(&self.canon_term(x));
                    self.entries.insert(v, x);
                }
                core::GenericCoreAction::Set(_, f, xs, y) => match f {
                    ResolvedCall::Primitive(..) => panic!("runtime primitive set!"),
                    ResolvedCall::Func(f) => {
                        let f = self.func(f);
                        let canon_args = self.canon_args(xs.iter().chain([y]));
                        let args = self.args(canon_args.iter());
                        self.rb.set(f, &args)
                    }
                },
                core::GenericCoreAction::Change(span, change, f, args) => match f {
                    ResolvedCall::Primitive(..) => panic!("runtime primitive change!"),
                    ResolvedCall::Func(f) => {
                        let name = f.name.clone();
                        let can_subsume = self.functions[&f.name].can_subsume;
                        let f = self.func(f);
                        let canon_args = self.canon_args(args);
                        let args = self.args(canon_args.iter());
                        match change {
                            Change::Delete => self.rb.remove(f, &args),
                            Change::Subsume if can_subsume => self.rb.subsume(f, &args),
                            Change::Subsume => {
                                return Err(Error::SubsumeMergeError(name, span.clone()));
                            }
                        }
                    }
                },
                core::GenericCoreAction::Union(_, x, y) => {
                    let x = self.entry(&self.canon_term(x));
                    let y = self.entry(&self.canon_term(y));
                    self.rb.union(x, y)
                }
                core::GenericCoreAction::Panic(_, message) => self.rb.panic(message.clone()),
            }
        }
        Ok(())
    }

    fn build(self) -> egglog_bridge::RuleId {
        if let Some(proof_state) = self.proof_state.as_ref() {
            let syntax = SyntaxBuilder {
                proof_state,
                egraph: self.rb.egraph(),
                var_map: &self.entries,
                var_types: &self.var_types,
                env: Default::default(),
            }
            .reconstruct_syntax();
            self.rb.build_with_syntax(syntax)
        } else {
            self.rb.build()
        }
    }
}

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
    #[error("{}\nCheck failed: \n{}", .1, ListDisplay(.0, "\n"))]
    CheckError(Vec<Fact>, Span),
    #[error("{1}\nNo such ruleset: {0}")]
    NoSuchRuleset(String, Span),
    #[error(
        "{1}\nAttempted to add a rule to combined ruleset {0}. Combined rulesets may only depend on other rulesets."
    )]
    CombinedRulesetError(String, Span),
    #[error("{0}")]
    BackendError(String),
    #[error("{0}\nTried to pop too much")]
    Pop(Span),
    #[error("{0}\nCommand should have failed.")]
    ExpectFail(Span),
    #[error("{2}\nIO error: {0}: {1}")]
    IoError(PathBuf, std::io::Error, Span),
    #[error("{1}\nCannot subsume function with merge: {0}")]
    SubsumeMergeError(String, Span),
    #[error("extraction failure: {:?}", .0)]
    ExtractError(String),
    #[error("{1}\n{2}\nShadowing is not allowed, but found {0}")]
    Shadowing(String, Span, Span),
    #[error("{1}\nCommand already exists: {0}")]
    CommandAlreadyExists(String, Span),
    #[error("Incorrect format in file '{0}'.")]
    InputFileFormatError(String),
}

#[cfg(test)]
mod tests {
    use crate::constraint::SimpleTypeConstraint;
    use crate::sort::*;
    use crate::*;

    #[derive(Clone)]
    struct InnerProduct {
        vec: ArcSort,
    }

    impl Primitive for InnerProduct {
        fn name(&self) -> &str {
            "inner-product"
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
                .container_values()
                .get_val::<VecContainer>(args[0])
                .unwrap();
            let vec2 = exec_state
                .container_values()
                .get_val::<VecContainer>(args[1])
                .unwrap();
            assert_eq!(vec1.data.len(), vec2.data.len());
            for (a, b) in vec1.data.iter().zip(vec2.data.iter()) {
                let a = exec_state.base_values().unwrap::<i64>(*a);
                let b = exec_state.base_values().unwrap::<i64>(*b);
                sum += a * b;
            }
            Some(exec_state.base_values().get::<i64>(sum))
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
                && s.inner_sorts()[0].name() == I64Sort.name()
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

    // Test that an `EGraph` is `Send` & `Sync`
    #[test]
    fn test_egraph_send_sync() {
        fn is_send<T: Send>(_t: &T) -> bool {
            true
        }
        fn is_sync<T: Sync>(_t: &T) -> bool {
            true
        }
        let egraph = EGraph::default();
        assert!(is_send(&egraph) && is_sync(&egraph));
    }

    fn get_function(egraph: &EGraph, name: &str) -> Function {
        egraph.functions.get(name).unwrap().clone()
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
        let outputs = egraph
            .parse_and_run_program(
                None,
                r#"
                (extract res)
                "#,
            )
            .unwrap();
        assert_eq!(outputs[0].to_string(), "(exp)\n");
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
        let res = egraph
            .parse_and_run_program(
                None,
                r#"
                (extract x)
                "#,
            )
            .unwrap();
        assert_eq!(res[0].to_string(), "(exp)\n");
    }
}
