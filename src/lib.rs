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
pub mod duckdb_config;
pub use duckdb_config::DuckBackendConfig;
#[cfg(feature = "bin")]
mod cli;
mod command_macro;
pub mod constraint;
mod core;
pub mod extract;
pub mod prelude;
mod proofs;

pub mod scheduler;
mod serialize;
pub mod sort;
mod termdag;
mod typechecking;
pub mod util;
pub use command_macro::{CommandMacro, CommandMacroRegistry};

// This is used to allow the `add_primitive` macro to work in
// both this crate and other crates by referring to `::egglog`.
extern crate self as egglog;
pub use ast::{ResolvedExpr, ResolvedFact, ResolvedVar};
#[cfg(feature = "bin")]
pub use cli::*;
use constraint::{Constraint, Problem, SimpleTypeConstraint, TypeConstraint};
use core::CoreActionContext;
use core::ResolvedAtomTerm;
pub use core::{Atom, AtomTerm};
pub use core::{ResolvedCall, SpecializedPrimitive};
pub use core_relations::{BaseValue, ContainerValue, ExecutionState, Value};
use core_relations::{ExternalFunctionId, make_external_func};
use csv::Writer;
pub use egglog_add_primitive::add_literal_prim;
pub use egglog_add_primitive::add_primitive;
pub use egglog_add_primitive::add_primitive_with_validator;
use egglog_ast::generic_ast::{Change, GenericExpr, Literal};
use egglog_ast::span::Span;
use egglog_ast::util::ListDisplay;
pub use egglog_bridge::FunctionRow;
use egglog_bridge::{ColumnTy, QueryEntry, UnionAction};
use egglog_core_relations as core_relations;
use egglog_numeric_id as numeric_id;
use egglog_reports::{ReportLevel, RunReport};
use extract::{DefaultCost, Extractor, TreeAdditiveCostModel};
use indexmap::map::Entry;
use log::{Level, log_enabled};
use numeric_id::DenseIdMap;
use prelude::*;
pub use proofs::proof_encoding_helpers::{file_supports_proofs, program_supports_proofs};
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
pub use termdag::{Term, TermDag, TermId};
use thiserror::Error;
pub use typechecking::PrimitiveValidator;
pub use typechecking::TypeError;
pub use typechecking::TypeInfo;
use util::*;

use crate::ast::desugar::desugar_command;
use crate::ast::*;
use crate::core::{GenericActionsExt, ResolvedRuleExt};
use crate::proofs::proof_encoding::{EncodingState, ProofInstrumentor};
use crate::proofs::proof_encoding_helpers::{
    ProofEncodingUnsupportedReason, command_supports_proof_encoding,
};
use crate::proofs::proof_extraction::ProveExistsError;
use crate::proofs::proof_format::{ProofId, ProofStore};
use crate::proofs::proof_normal_form::proof_form;

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
#[allow(clippy::large_enum_variant)]
pub enum CommandOutput {
    /// The size of a function
    PrintFunctionSize(usize),
    /// The name of all functions and their sizes
    PrintAllFunctionsSize(Vec<(String, usize)>),
    /// The best term found after extracting
    ExtractBest(TermDag, DefaultCost, TermId),
    /// The variants of a function found after extracting. Like normal extraction, but has to choose one extraction per e-node in the e-class.
    ExtractVariants(TermDag, Vec<TermId>),
    /// A high-level proof witnessing constructor existence
    ProveExists {
        proof_store: ProofStore,
        proof_id: ProofId,
    },
    /// The report from all runs
    OverallStatistics(RunReport),
    /// A printed function and all its values
    PrintFunction(Function, TermDag, Vec<(TermId, TermId)>, PrintFunctionMode),
    /// The report from a single run
    RunSchedule(RunReport),
    /// A user defined output
    UserDefined(Arc<dyn UserDefinedCommandOutput>),
}

impl std::fmt::Display for CommandOutput {
    /// Format the command output for display, ending with a newline.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandOutput::PrintFunctionSize(size) => writeln!(f, "{size}"),
            CommandOutput::PrintAllFunctionsSize(names_and_sizes) => {
                write!(f, "(")?;
                for (i, (name, size)) in names_and_sizes.iter().enumerate() {
                    // indent except for the first line
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    // write the pair of funciton symbol and size
                    write!(f, "({name} {size})")?;
                    // add a newline except at the end
                    if i < names_and_sizes.len() - 1 {
                        writeln!(f)?;
                    }
                }
                writeln!(f, ")")
            }
            CommandOutput::ExtractBest(termdag, _cost, term) => {
                writeln!(f, "{}", termdag.to_string(*term))
            }
            CommandOutput::ExtractVariants(termdag, terms) => {
                writeln!(f, "(")?;
                for expr in terms {
                    writeln!(f, "   {}", termdag.to_string(*expr))?;
                }
                writeln!(f, ")")
            }
            CommandOutput::ProveExists {
                proof_store,
                proof_id,
            } => writeln!(f, "{}", proof_store.proof_to_string(*proof_id)),
            CommandOutput::OverallStatistics(run_report) => {
                write!(f, "Overall statistics:\n{run_report}")
            }
            CommandOutput::PrintFunction(function, termdag, terms_and_outputs, mode) => {
                let out_is_unit = function.schema.output.name() == UnitSort.name();
                if *mode == PrintFunctionMode::CSV {
                    let mut wtr = Writer::from_writer(vec![]);
                    for (term_id, output) in terms_and_outputs {
                        let term = termdag.get(*term_id);
                        match term {
                            Term::App(name, children) => {
                                let mut values = vec![name.clone()];
                                for child_id in children {
                                    values.push(termdag.to_string(*child_id));
                                }

                                if !out_is_unit {
                                    values.push(termdag.to_string(*output));
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
                        write!(f, "   {}", termdag.to_string(*term))?;
                        if !out_is_unit {
                            write!(f, " -> {}", termdag.to_string(*output))?;
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
    backend: Box<dyn egglog_backend_trait::Backend>,
    pub parser: Parser,
    names: check_shadowing::Names,
    /// pushed_egraph forms a linked list of pushed egraphs.
    /// Pop reverts the egraph to the last pushed egraph.
    pushed_egraph: Option<Box<Self>>,
    functions: IndexMap<String, Function>,
    rulesets: IndexMap<String, Ruleset>,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,
    /// Enable the seminaive-as-encoding pass. See
    /// `seminaive-encoding-experiment.md` at the repo root.
    pub seminaive_encoding_enabled: bool,
    /// Whether the seminaive-encoding pass has already emitted its
    /// `next_ts` header. The pass is invoked once per command, but
    /// `next_ts` should only be declared once per session.
    pub(crate) seminaive_encoding_header_emitted: bool,
    /// Functions/constructors/relations seen so far by the
    /// seminaive-encoding pass, accumulated across per-command
    /// invocations. Needed because a rule in command N can reference
    /// a function declared in command M < N.
    pub(crate) seminaive_tracked: HashSet<String>,
    type_info: TypeInfo,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,
    schedulers: DenseIdMap<SchedulerId, SchedulerRecord>,
    commands: IndexMap<String, Arc<dyn UserDefinedCommand>>,
    strict_mode: bool,
    warned_about_global_prefix: bool,
    /// Registry for command-level macros
    command_macros: CommandMacroRegistry,
    proof_state: EncodingState,
    /// In proof mode, this is the program before proof instrumentation and the version we use for proof checking.
    proof_check_program: Vec<ResolvedNCommand>,
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

    /// Whether this is a let binding
    pub fn is_let_binding(&self) -> bool {
        self.decl.internal_let
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

impl Default for EGraph {
    fn default() -> Self {
        Self::with_backend(Box::new(egglog_bridge::EGraph::default()))
    }
}

impl EGraph {
    /// Build a frontend `EGraph` around the given backend trait object.
    /// This is the shared backbone of [`EGraph::default`] and
    /// [`EGraph::with_duckdb_backend`].
    fn with_backend(backend: Box<dyn egglog_backend_trait::Backend>) -> Self {
        let mut parser = Parser::default();
        let proof_state = EncodingState::new(&mut parser.symbol_gen);
        let mut eg = Self {
            backend,
            parser,
            names: Default::default(),
            pushed_egraph: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            fact_directory: None,
            seminaive: true,
            seminaive_encoding_enabled: false,
            seminaive_encoding_header_emitted: false,
            seminaive_tracked: HashSet::default(),
            overall_run_report: Default::default(),
            type_info: Default::default(),
            schedulers: Default::default(),
            commands: Default::default(),
            strict_mode: false,
            warned_about_global_prefix: false,
            command_macros: Default::default(),
            proof_state,
            proof_check_program: vec![],
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
        eg.type_info.add_presort::<PairSort>(span!()).unwrap();

        // Add != with a validator that computes inequality result
        let neq_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            if args.len() == 2 && args[0] != args[1] {
                // Return unit literal for successful inequality
                Some(termdag.lit(Literal::Unit))
            } else {
                None
            }
        };
        add_primitive_with_validator!(
            &mut eg,
            "!=" = |a: #, b: #| -?> () {
                (a != b).then_some(())
            },
            neq_validator
        );

        add_primitive_with_validator!(
            &mut eg,
            "bool-!=" = |a: #, b: #| -> bool {
                (a != b)
            },
            |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
                if args.len() == 2 {
                    Some(termdag.lit(Literal::Bool(args[0] != args[1])))
                } else {
                    None
                }
            }
        );

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

    /// Construct an `EGraph` whose backend is the DuckDB-backed
    /// [`egglog_bridge_duckdb::EGraph`] driven through the
    /// [`egglog_backend_trait::Backend`] trait.
    ///
    /// This is the entry point exercised by `tests/files.rs`'s duckdb
    /// path after Phase 2 Commit 14. It performs the same sort /
    /// primitive registration as [`EGraph::default`] but the underlying
    /// storage and rule execution route through DuckDB.
    ///
    /// Term encoding (or proof-tracking term encoding) is enabled
    /// unconditionally: the DuckDB backend is term-encoding only.
    pub fn with_duckdb_backend(config: DuckBackendConfig) -> anyhow::Result<Self> {
        let mut db = egglog_bridge_duckdb::EGraph::new()?;
        if config.native_uf {
            db.enable_native_uf();
        }
        let backend: Box<dyn egglog_backend_trait::Backend> = Box::new(db);
        let mut eg = Self::with_backend(backend);

        // Term encoding requires a separate typechecker EGraph for
        // re-typechecking after the encoder runs. The duckdb backend
        // cannot be cloned (its `Connection` is not trivially
        // cloneable), so we instead build a bridge-backed typechecker
        // that mirrors the same sort + primitive registration done
        // above.
        let typechecker = if config.proofs {
            EGraph::new_with_proofs()
        } else {
            EGraph::default().with_term_encoding_enabled()
        };
        eg.proof_state.original_typechecking = Some(Box::new(typechecker));
        if config.proofs {
            eg.proof_state.proofs_enabled = true;
        }
        Ok(eg)
    }
}

struct ResolvedNCommands {
    desugared: Vec<ResolvedNCommand>,
    /// In proof mode, populated with the desugared program before instrumented with proofs
    desugared_before_proofs: Vec<ResolvedNCommand>,
}

struct ResolvedNCommandsWithOutput {
    outputs: Vec<CommandOutput>,
    resolved: Vec<ResolvedNCommand>,
    /// In proof mode, populated with the desugared program before instrumented with proofs
    resolved_before_proofs: Vec<ResolvedNCommand>,
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(String);

impl EGraph {
    /// Downcast `self.backend` to the concrete bridge `EGraph`. Use this
    /// when invoking bridge-inherent methods that are not lifted onto the
    /// [`egglog_backend_trait::Backend`] trait (typed generics, `TableAction`
    /// / `UnionAction` constructors, `base_values()` / `container_values()`
    /// accessors, etc.). Panics if `self.backend` is not a bridge `EGraph`
    /// — these call sites are bridge-only by design.
    pub(crate) fn bridge(&self) -> &egglog_bridge::EGraph {
        self.backend
            .as_any()
            .downcast_ref::<egglog_bridge::EGraph>()
            .expect("this code path is bridge-only")
    }

    /// Mutable counterpart of [`EGraph::bridge`].
    pub(crate) fn bridge_mut(&mut self) -> &mut egglog_bridge::EGraph {
        self.backend
            .as_any_mut()
            .downcast_mut::<egglog_bridge::EGraph>()
            .expect("this code path is bridge-only")
    }

    /// Create a new e-graph with the term-encoding pipeline enabled.
    ///
    /// In term-encoding mode the e-graph eagerly instruments every constructor
    /// and function with auxiliary term tables, view tables, and per-sort
    /// union-finds so that canonical representatives and their justifications are
    /// materialized explicitly.  This makes it possible to record and emit
    /// equality proofs while preserving the observable behaviour of supported
    /// commands.
    pub fn new_with_term_encoding() -> Self {
        let mut egraph = EGraph::default();
        egraph.proof_state.original_typechecking = Some(Box::new(egraph.clone()));
        egraph
    }

    /// Create a new e-graph with proof generation enabled.
    pub fn new_with_proofs() -> Self {
        let mut egraph = EGraph::new_with_term_encoding();
        egraph.proof_state.proofs_enabled = true;
        egraph
    }

    /// Enable the term-encoding pipeline on an existing `EGraph`.
    ///
    /// This method is to support the current CLI implementation with egglog-experimental (https://github.com/egraphs-good/egglog/issues/768)
    pub(crate) fn with_term_encoding_enabled(mut self) -> Self {
        self.proof_state.original_typechecking = Some(Box::new(self.clone()));
        self
    }

    /// Enable proof generation on this e-graph.
    /// TODO proofs should be turned on during creation of the e-graph, not afterwards.
    /// This method is to support the current CLI implementation with egglog-experimental (https://github.com/egraphs-good/egglog/issues/768)
    pub(crate) fn with_proofs_enabled(mut self) -> Self {
        self = self.with_term_encoding_enabled();
        self.proof_state.proofs_enabled = true;
        self
    }

    /// Enable testing of getting proofs for all `check` commands.
    pub fn with_proof_testing(mut self) -> Self {
        self.proof_state.proof_testing = true;
        self
    }

    /// Enable the seminaive-as-encoding experiment. Adds a pass after
    /// term encoding that lifts seminaive evaluation into the IR via
    /// per-rule timestamp predicates. Requires term encoding to also
    /// be enabled. See `seminaive-encoding-experiment.md` at the repo
    /// root.
    pub fn with_seminaive_encoding_enabled(mut self) -> Self {
        self.seminaive_encoding_enabled = true;
        self
    }

    /// Set the number of threads used for parallel operations.
    ///
    /// This is a helper that simply configures the global rayon thread pool. It can only be called
    /// once per process; subsequent calls will be ignored.
    ///
    /// # Panics
    ///
    /// Panics on wasm if `num_threads > 1`.
    pub fn set_num_threads(num_threads: usize) {
        #[cfg(target_family = "wasm")]
        if num_threads > 1 {
            panic!("cannot use more than 1 thread on wasm");
        }
        #[cfg(not(target_family = "wasm"))]
        {
            // This will fail silently if the global pool has already been configured.
            let err = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global();
            // print log if successful
            if matches!(err, Ok(())) {
                log::info!("Initialize global thread pool with  {num_threads} threads");
            } else {
                log::warn!(
                    "Failed to initialize global thread pool with {num_threads} threads. This may be because the thread pool was already initialized with a different number of threads. Error: {err:?}"
                );
            }
        }
    }

    /// Return the number of threads in the rayon thread pool.
    pub fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }

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

    /// Configure whether the internal reserved symbol (@) is allowed in user-defined names.
    /// WARNING: do not use, this is for testing running egglog after desugaring.
    /// Public so files.rs can use it, hidden from documentation because it is not intended for general use.
    #[doc(hidden)]
    pub fn ensure_no_reserved_symbols(&mut self, should_ensure: bool) {
        self.parser.ensure_no_reserved_symbols = should_ensure;
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
            return Err(TypeError::GlobalMissingPrefix {
                name: format!("{GLOBAL_NAME_PREFIX}{canonical_name}"),
                span: span.clone(),
            });
        }
        if self.warned_about_global_prefix {
            return Ok(());
        }
        self.warned_about_global_prefix = true;
        log::warn!(
            "{span}\nGlobal `{canonical_name}` should start with `{GLOBAL_NAME_PREFIX}`. Enable `--strict-mode` to turn this warning into an error. Suppressing additional warnings of this type."
        );
        Ok(())
    }

    fn warn_prefixed_non_globals(
        &mut self,
        span: &Span,
        canonical_name: &str,
    ) -> Result<(), TypeError> {
        if self.strict_mode {
            return Err(TypeError::NonGlobalPrefixed {
                name: canonical_name.to_string(),
                span: span.clone(),
            });
        }
        if self.warned_about_global_prefix {
            return Ok(());
        }
        self.warned_about_global_prefix = true;
        log::warn!(
            "{span}\nNon-global `{canonical_name}` should not start with `{GLOBAL_NAME_PREFIX}`. Enable `--strict-mode` to turn this warning into an error. Suppressing additional warnings of this type."
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
            Some(mut e) => {
                // Preserve the overall report from the popped egraph
                std::mem::swap(&mut self.overall_run_report, &mut e.overall_run_report);
                // Preserve the symbol generator so that fresh symbols
                // generated after pop don't collide with ones generated before pop.
                std::mem::swap(&mut self.parser.symbol_gen, &mut e.parser.symbol_gen);
                *self = *e;
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
                let val = literal_to_value(self.bridge(), literal);
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
            // View tables (functions with term_constructor) need subsumption support
            FunctionSubtype::Custom => decl.term_constructor.is_some(),
        };

        use egglog_bridge::{DefaultVal, MergeFn};
        let backend_id = self.backend.add_table(egglog_bridge::FunctionConfig {
            schema: input
                .iter()
                .chain([&output])
                .map(|sort| sort.column_ty(&*self.backend))
                .collect(),
            default: match decl.subtype {
                FunctionSubtype::Constructor => DefaultVal::FreshId,
                FunctionSubtype::Custom => DefaultVal::Fail,
            },
            merge: match decl.subtype {
                FunctionSubtype::Constructor => MergeFn::UnionId,
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

    /// Provide a program for use in proof checking.
    /// This enables testing of a desugared egglog proof program outside of proof mode.
    /// When proof_testing is true, turns all the `check` commands into `prove` commands.
    /// Not intended for general use but needed in files.rs, so public but hidden.
    #[doc(hidden)]
    pub fn set_proof_checking_program(
        &mut self,
        prog: Vec<Command>,
        proof_testing: bool,
    ) -> Result<(), Error> {
        // make a new e-graph, desugar the program in proof mode
        let mut proof_check_eg = EGraph::new_with_proofs();
        if proof_testing {
            proof_check_eg = proof_check_eg.with_proof_testing();
        }
        let resolved = proof_check_eg.process_program_internal(prog, false)?;

        self.proof_check_program = resolved.resolved_before_proofs;
        Ok(())
    }

    /// Print the size of a function. If no function name is provided,
    /// print the size of all non-hidden functions as an s-expression list of
    /// `(name size)` pairs, e.g. `((name size) ...)`.
    pub fn print_size(&self, sym: Option<&str>) -> Result<CommandOutput, Error> {
        if let Some(sym) = sym {
            // In proof mode, we have view tables instead of term tables.
            // So we do a linear scan to find the view table first, falling back on the normal table otherwise.
            // (We don't check the proof mode flag so that this still works after desugaring)
            let f = self
                .functions
                .values()
                .find(|f| f.decl.term_constructor.as_deref() == Some(sym))
                .or_else(|| self.functions.get(sym))
                .ok_or(TypeError::UnboundFunction(sym.to_owned(), span!()))?;
            // Skip hidden and let_binding functions
            if f.decl.internal_hidden || f.decl.internal_let {
                return Err(TypeError::UnboundFunction(sym.to_owned(), span!()).into());
            }
            let size = self.backend.table_size(f.backend_id);
            log::info!("Function {sym} has size {size}");
            Ok(CommandOutput::PrintFunctionSize(size))
        } else {
            // Print size of all non-hidden, non-let_binding functions
            // For view tables, use the term_constructor name instead
            let mut lens = self
                .functions
                .iter()
                .filter(|(_, f)| !f.decl.internal_hidden && !f.decl.internal_let)
                .map(|(sym, f)| {
                    let name = f
                        .decl
                        .term_constructor
                        .clone()
                        .unwrap_or_else(|| sym.clone());
                    (name, self.backend.table_size(f.backend_id))
                })
                .collect::<Vec<_>>();

            // Function name's alphabetical order
            lens.sort_by_key(|(name, _)| name.clone());
            if log_enabled!(Level::Info) {
                for (sym, len) in &lens {
                    log::info!("Function {sym} has size {len}");
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

    fn run_rules(&mut self, span: &Span, config: &ResolvedRunConfig) -> Result<RunReport, Error> {
        log::debug!("Running ruleset: {}", config.ruleset);
        let mut report: RunReport = Default::default();

        let GenericRunConfig { ruleset, until } = config;

        if let Some(facts) = until
            && self.check_facts(span, facts).is_ok()
        {
            log::info!(
                "Breaking early because of facts:\n {}!",
                ListDisplay(facts, "\n")
            );
            return Ok(report);
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
        fn collect(
            ruleset: &str,
            rulesets: &IndexMap<String, Ruleset>,
            ids: &mut Vec<egglog_bridge::RuleId>,
            names: &mut Vec<String>,
        ) {
            match &rulesets[ruleset] {
                Ruleset::Rules(rules) => {
                    for (n, (_, id)) in rules.iter() {
                        ids.push(*id);
                        names.push(n.clone());
                    }
                }
                Ruleset::Combined(sub_rulesets) => {
                    for sub_ruleset in sub_rulesets {
                        collect(sub_ruleset, rulesets, ids, names);
                    }
                }
            }
        }

        let mut rule_ids = Vec::new();
        let mut rule_names = Vec::new();
        collect(ruleset, &self.rulesets, &mut rule_ids, &mut rule_names);

        // Seminaive-encoding hook: bump next_ts before the iteration if the
        // program defines it. No-op for programs that don't use the encoding.
        let new_ts = self.bump_next_ts_global();

        let iteration_report = self
            .backend
            .run_rules(&rule_ids)
            .map_err(|e| Error::BackendError(e.to_string()))?;

        // After the iteration, set last_run_at_<src> to the bumped value for
        // every source rule that ran. The source name is the rule name with
        // any `@<digits>` variant suffix stripped.
        if let Some(ts) = new_ts {
            self.update_last_run_at_globals(&rule_names, ts);
        }

        Ok(RunReport::singleton(ruleset, iteration_report))
    }

    /// If the program defines a nullary i64 function `next_ts`, increment
    /// it by 1 and return the new value. Returns `None` otherwise.
    fn bump_next_ts_global(&mut self) -> Option<i64> {
        let func = self.functions.get("next_ts")?;
        let id = func.backend_id;
        let cur_val = self.backend.lookup_id(id, &[])?;
        let cur: i64 = self.bridge().base_values().unwrap::<i64>(cur_val);
        let new = cur + 1;
        let new_val = self.bridge().base_values().get::<i64>(new);
        self.backend
            .add_values(Box::new([(id, vec![new_val])].into_iter()));
        Some(new)
    }

    /// For every rule name in `rule_names`, if a nullary i64 function
    /// `last_run_at_<source>` exists (where `<source>` is the rule name
    /// minus any `@<digits>` variant suffix), set it to `ts`.
    fn update_last_run_at_globals(&mut self, rule_names: &[String], ts: i64) {
        let mut sources: HashSet<&str> = HashSet::default();
        for n in rule_names {
            sources.insert(n.split('@').next().unwrap_or(n.as_str()));
        }
        let ts_val = self.bridge().base_values().get::<i64>(ts);
        let updates: Vec<_> = sources
            .into_iter()
            .filter_map(|src| {
                let global = format!("last_run_at_{src}");
                self.functions
                    .get(&global)
                    .map(|f| (f.backend_id, vec![ts_val]))
            })
            .collect();
        if !updates.is_empty() {
            self.backend.add_values(Box::new(updates.into_iter()));
        }
    }

    fn add_rule(&mut self, rule: ast::ResolvedRule) -> Result<String, Error> {
        // Disable union_to_set optimization in proof or term encoding mode, since
        // it expects only `union` on constructors (not set).
        let core_rule = rule.to_canonicalized_core_rule(
            &self.type_info,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        )?;
        let (query, actions) = (&core_rule.body, &core_rule.head);

        let rule_id = {
            // Construct the trait-routed rule builder. The lifetime
            // separately borrows `&mut self.backend` for the builder
            // and `&self.backend` for column-ty / pool reads — these
            // overlap, so we route the read-only borrow through a raw
            // pointer reborrow inside an unsafe block. Both pointers
            // are derived from the same `Box<dyn Backend>` and are
            // never aliased mutably while the builder is alive: the
            // builder only borrows `&mut Self`'s internal state, never
            // `BaseValuePool` mutably.
            let backend_ptr: *const dyn egglog_backend_trait::Backend = &*self.backend;
            let rb = self.backend.new_rule(&rule.name, self.seminaive);
            let mut translator = BackendRule::new(
                rb,
                // SAFETY: the rule builder's internal `&mut Backend`
                // borrow does not touch the `BaseValuePool` / column-
                // ty lookups, so the disjoint reborrow is sound for
                // the duration of `query`/`actions`/`build`.
                unsafe { &*backend_ptr },
                &self.functions,
                &self.type_info,
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
                        indexmap::map::Entry::Vacant(e) => e.insert((core_rule, rule_id)),
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
        let mut binding = IndexSet::default();
        let mut ctx = CoreActionContext::new(
            &self.type_info,
            &mut binding,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        );
        let (actions, _) = actions.to_core_actions(&mut ctx)?;

        let id = {
            let backend_ptr: *const dyn egglog_backend_trait::Backend = &*self.backend;
            let rb = self.backend.new_rule("eval_actions", false);
            let mut translator = BackendRule::new(
                rb,
                // SAFETY: see `add_rule` for the disjoint-reborrow rationale.
                unsafe { &*backend_ptr },
                &self.functions,
                &self.type_info,
            );
            translator.actions(&actions)?;
            translator.build()
        };
        let result = self.backend.run_rules(&[id]);
        self.backend.free_rule(id);

        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(Error::BackendError(e.to_string())),
        }
    }

    /// Get the list of all functions in the e-graph.
    pub fn get_function_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }

    /// Read the contents of the given function.
    /// The callback f is called with each row and its subsumption status.
    ///
    /// Raises an error if the function does not exist.
    pub fn function_for_each(
        &self,
        func_name: &str,
        mut f: impl FnMut(FunctionRow<'_>),
    ) -> Result<(), Error> {
        let func = self
            .functions
            .get(func_name)
            .ok_or_else(|| TypeError::UnboundFunction(func_name.to_string(), span!()))?;
        self.backend.for_each(func.backend_id, &mut |row| f(row));
        Ok(())
    }

    /// Evaluates an expression, returns the sort of the expression and the evaluation result.
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, Value), Error> {
        let span = expr.span();
        let command = Command::Action(Action::Expr(span.clone(), expr.clone()));
        // Resolve against the *pre-term-encoding* typechecker. Term
        // encoding lifts `Action::Expr(...)` into UF/View table
        // updates and drops the return value (literals become a bare
        // string that's never re-parsed), so routing eval_expr
        // through it loses the very thing we want to evaluate.
        // resolve_command_before_proofs typechecks against
        // original_typechecking (or self.type_info if it's absent)
        // and preserves the Expr action verbatim — and
        // eval_resolved_expr below evaluates it against the live
        // backend, which still carries the user-facing function
        // names (term encoding adds @ViewN tables alongside them,
        // not in place of them).
        let resolved_before_proofs = self.resolve_command_before_proofs(command)?;
        if self.are_proofs_enabled() {
            self.proof_check_program
                .extend(resolved_before_proofs.clone());
        }
        let mut expr_action = None;
        for cmd in resolved_before_proofs {
            if let ResolvedNCommand::CoreAction(ResolvedAction::Expr(_, e)) = cmd {
                expr_action = Some(e);
                break;
            }
        }
        let resolved_expr = expr_action.ok_or_else(|| {
            Error::BackendError(
                "eval_expr: resolver did not produce an expression action".into(),
            )
        })?;
        let sort = resolved_expr.output_type();
        let value = self.eval_resolved_expr(span, &resolved_expr)?;
        Ok((sort, value))
    }

    fn eval_resolved_expr(&mut self, span: Span, expr: &ResolvedExpr) -> Result<Value, Error> {
        let unit_id = self.bridge().base_values().get_ty::<()>();
        let unit_val = self.bridge().base_values().get(());

        let result: egglog_bridge::SideChannel<Value> = Default::default();
        let result_ref = result.clone();
        let ext_id = self
            .backend
            .register_external_func(Box::new(make_external_func(move |_es, vals| {
                debug_assert!(vals.len() == 1);
                *result_ref.lock().unwrap() = Some(vals[0]);
                Some(unit_val)
            })));
        // Side-channel primitive — see `check_facts` for context.
        if let Some(duck) = self
            .backend
            .as_any_mut()
            .downcast_mut::<egglog_bridge_duckdb::EGraph>()
        {
            duck.set_external_func_name(ext_id, "__eval_resolved_sentinel".to_string());
        }

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
        let mut binding = IndexSet::default();
        let mut ctx = CoreActionContext::new(
            &self.type_info,
            &mut binding,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        );
        let actions = actions.to_core_actions(&mut ctx)?.0;

        let id = {
            let backend_ptr: *const dyn egglog_backend_trait::Backend = &*self.backend;
            let rb = self.backend.new_rule("eval_resolved_expr", false);
            let mut translator = BackendRule::new(
                rb,
                // SAFETY: see `add_rule` for the disjoint-reborrow rationale.
                unsafe { &*backend_ptr },
                &self.functions,
                &self.type_info,
            );
            translator.actions(&actions)?;

            let arg = translator.entry(&ResolvedAtomTerm::Var(span.clone(), result_var));
            translator.rb.call_external_func(
                ext_id,
                &[arg],
                egglog_bridge::ColumnTy::Base(unit_id),
                "this function will never panic".to_string(),
            );

            translator.build()
        };
        let rule_result = self.backend.run_rules(&[id]);
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);
        let _ = rule_result.map_err(|e| {
            Error::BackendError(format!("Failed to evaluate expression '{expr}': {e}"))
        })?;

        let result = result.lock().unwrap().unwrap();
        Ok(result)
    }

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
        let core_rule = rule.to_canonicalized_core_rule(
            &self.type_info,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        )?;
        let query = core_rule.body;

        // Backend-agnostic check: build the body via the trait, then
        // ask the rule builder whether it matches. The DuckDB impl
        // runs `SELECT … LIMIT 1` against the compiled body; the
        // bridge backend falls back to the side-channel sentinel
        // pattern (see below).
        let is_duck = self
            .backend
            .as_any()
            .downcast_ref::<egglog_bridge_duckdb::EGraph>()
            .is_some();

        if is_duck {
            let backend_ptr: *const dyn egglog_backend_trait::Backend = &*self.backend;
            let rb = self.backend.new_rule("check_facts", false);
            let mut translator = BackendRule::new(
                rb,
                // SAFETY: see `add_rule` for the disjoint-reborrow rationale.
                unsafe { &*backend_ptr },
                &self.functions,
                &self.type_info,
            );
            translator.query(&query, true);
            let matched = translator
                .into_rb()
                .build_check()
                .map_err(|e| Error::BackendError(format!("check_facts: {e}")))?;
            if !matched {
                return Err(Error::CheckError(
                    facts.iter().map(|f| f.clone().make_unresolved()).collect(),
                    span.clone(),
                ));
            }
            return Ok(());
        }

        let ext_sc = egglog_bridge::SideChannel::default();
        let ext_sc_ref = ext_sc.clone();
        let ext_id = self
            .backend
            .register_external_func(Box::new(make_external_func(move |_, _| {
                *ext_sc_ref.lock().unwrap() = Some(());
                Some(Value::new_const(0))
            })));

        let id = {
            let backend_ptr: *const dyn egglog_backend_trait::Backend = &*self.backend;
            let rb = self.backend.new_rule("check_facts", false);
            let mut translator = BackendRule::new(
                rb,
                // SAFETY: see `add_rule` for the disjoint-reborrow rationale.
                unsafe { &*backend_ptr },
                &self.functions,
                &self.type_info,
            );
            translator.query(&query, true);
            translator.rb.call_external_func(
                ext_id,
                &[],
                egglog_bridge::ColumnTy::Id,
                "this function will never panic".to_string(),
            );
            translator.build()
        };
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
            ResolvedNCommand::Sort {
                name,
                uf,
                proof_func,
                ..
            } => {
                // If the sort has a :internal-uf field, store the mapping for extraction
                if let Some(uf_name) = uf {
                    self.proof_state.uf_parent.insert(name.clone(), uf_name);
                }
                // If the sort has a :internal-proof-func field, store the mapping for proof lookup.
                // This annotation is set by proof instrumentation and consumed here.
                if let Some(proof_func_name) = proof_func {
                    self.proof_state
                        .proof_func_parent
                        .insert(name.clone(), proof_func_name);
                }
                log::info!("Declared sort {name}.")
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
                log::info!("Ran schedule {sched}.");
                log::info!("Report: {report}");
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
                    log::info!("Printed overall statistics to json file {path}");

                    serde_json::to_writer(&mut file, &self.overall_run_report)
                        .expect("error serializing to json");
                }
            },
            ResolvedNCommand::Check(span, facts) => {
                self.check_facts(&span, &facts)?;
                log::info!("Checked fact {facts:?}.");
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
                // Extraction relies on bridge-specific facilities
                // (`Extractor::compute_costs_from_rootsorts` walks the
                // bridge's typed tables). The duck backend silently
                // skips extract commands — matches the legacy
                // duck pipeline's behavior (see
                // `backend_duckdb.rs::dispatch`, where
                // `GenericNCommand::Extract` is a no-op).
                if self
                    .backend
                    .as_any()
                    .downcast_ref::<egglog_bridge_duckdb::EGraph>()
                    .is_some()
                {
                    return Ok(None);
                }
                let sort = expr.output_type();

                let x = self.eval_resolved_expr(span.clone(), &expr)?;
                let n = self.eval_resolved_expr(span, &variants)?;
                let n: i64 = self.bridge().base_values().unwrap(n);

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
                            log::info!("extracted with cost {cost}: {}", termdag.to_string(term));
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
                    let terms: Vec<TermId> = extractor
                        .extract_variants(self, &mut termdag, x, n as usize)
                        .iter()
                        .map(|e| e.1)
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
                    writeln!(f, "{}", termdag.to_string(term))
                        .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                }

                log::info!("Output to '{filename:?}'.")
            }
            ResolvedNCommand::UserDefined(_span, name, exprs) => {
                let command = self.commands.swap_remove(&name).unwrap_or_else(|| {
                    panic!("Unrecognized user-defined command: {name}");
                });
                let res = command.update(self, &exprs);
                self.commands.insert(name, command);
                return res;
            }
            ResolvedNCommand::ProveExists(span, resolved_call) => {
                let mut instrument = ProofInstrumentor { egraph: self };
                let (proof_store, proof_id) =
                    instrument
                        .prove_exists(&resolved_call)
                        .map_err(|error| Error::ProofError {
                            span: span.clone(),
                            error,
                        })?;
                return Ok(Some(CommandOutput::ProveExists {
                    proof_store,
                    proof_id,
                }));
            }
        };

        Ok(None)
    }

    fn input_file(&mut self, func_name: &str, file: String) -> Result<(), Error> {
        let function_type = self
            .type_info
            .get_func_type(func_name)
            .unwrap_or_else(|| panic!("Unrecognized function name {func_name}"));
        let (func_backend_id, func_schema_input, func_schema_output) = {
            let func = self.functions.get(func_name).unwrap();
            (
                func.backend_id,
                func.schema.input.clone(),
                func.schema.output.clone(),
            )
        };

        let mut filename = self.fact_directory.clone().unwrap_or_default();
        filename.push(file.as_str());

        // check that the function uses supported types

        for t in &func_schema_input {
            match t.name() {
                "i64" | "f64" | "String" => {}
                s => panic!("Unsupported type {s} for input"),
            }
        }

        if function_type.subtype != FunctionSubtype::Constructor {
            match func_schema_output.name() {
                "i64" | "String" | "Unit" => {}
                s => panic!("Unsupported type {s} for input"),
            }
        }

        log::info!("Opening file '{filename:?}'...");
        let mut f = File::open(filename).unwrap();
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();

        // Can also do a row-major Vec<Value>
        let mut parsed_contents: Vec<Vec<Value>> = Vec::with_capacity(contents.lines().count());

        let mut row_schema = func_schema_input;
        if function_type.subtype == FunctionSubtype::Custom {
            row_schema.push(func_schema_output);
        }

        log::debug!("{row_schema:?}");

        // Use the trait-level base value pool so this works on both
        // the bridge backend and the DuckDB backend. `pool_get<T>`
        // takes a `&dyn BaseValuePool` and converts a typed value
        // into the runtime `Value` representation; both backends
        // implement the trait, so neither needs a downcast.
        use egglog_backend_trait::pool_get;
        let pool = self.backend.base_value_pool();
        let unit_val = pool_get::<()>(pool, ());

        for line in contents.lines() {
            let mut it = line.split('\t').map(|s| s.trim());

            let mut row: Vec<Value> = Vec::with_capacity(row_schema.len());

            for sort in row_schema.iter() {
                if let Some(raw) = it.next() {
                    let val = match sort.name() {
                        "i64" => {
                            if let Ok(i) = raw.parse::<i64>() {
                                pool_get::<i64>(self.backend.base_value_pool(), i)
                            } else {
                                return Err(Error::InputFileFormatError(file));
                            }
                        }
                        "f64" => {
                            if let Ok(f) = raw.parse::<f64>() {
                                pool_get::<F>(
                                    self.backend.base_value_pool(),
                                    core_relations::Boxed::new(f.into()),
                                )
                            } else {
                                return Err(Error::InputFileFormatError(file));
                            }
                        }
                        "String" => pool_get::<S>(
                            self.backend.base_value_pool(),
                            raw.to_string().into(),
                        ),
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

        if function_type.subtype != FunctionSubtype::Constructor {
            self.backend.insert_rows(func_backend_id, &parsed_contents);
        } else {
            self.backend
                .lookup_constructor_rows(func_backend_id, &parsed_contents);
        }

        self.backend.flush_updates();

        log::info!("Read {num_facts} facts into {func_name} from '{file}'.");
        Ok(())
    }

    /// Returns true if proofs are enabled.
    pub fn are_proofs_enabled(&self) -> bool {
        self.proof_state.proofs_enabled
    }

    /// True iff this egraph's backend is the duckdb-backed one (vs.
    /// the in-process bridge backend). Used by `cli.rs` to decide
    /// whether the `--duckdb` CLI flag should rebuild the egraph
    /// from scratch or use the caller-supplied one as-is — relevant
    /// for downstream crates (e.g. `egglog-experimental`) that
    /// register commands / primitives up front and need them to
    /// survive into the run.
    pub fn has_duckdb_backend(&self) -> bool {
        self.backend
            .as_any()
            .is::<egglog_bridge_duckdb::EGraph>()
    }

    fn resolve_command_before_proofs(
        &mut self,
        command: Command,
    ) -> Result<Vec<ResolvedNCommand>, Error> {
        let desugared = desugar_command(command, &mut self.parser, self.proof_state.proof_testing)?;
        if let Some(original_typechecking) = self.proof_state.original_typechecking.as_mut() {
            // Typecheck using the original egraph
            // TODO this is ugly- we don't need an entire e-graph just for type information.
            let typechecked = original_typechecking.typecheck_program(&desugared)?;

            for command in &typechecked {
                if let Err(reason) = command_supports_proof_encoding(
                    &command.to_command(),
                    &self.type_info,
                    self.proof_state.proofs_enabled,
                ) {
                    let command_text = format!("{}", command.to_command());
                    return Err(Error::UnsupportedProofCommand {
                        command: command_text,
                        reason,
                    });
                }
            }

            Ok(proof_form(typechecked, &mut self.parser.symbol_gen))
        } else {
            let mut typechecked = self.typecheck_program(&desugared)?;

            typechecked = remove_globals::remove_globals(typechecked, &mut self.parser.symbol_gen);
            for command in &typechecked {
                self.names.check_shadowing(command)?;
            }
            Ok(typechecked)
        }
    }

    /// Desugars, typechecks, and removes globals from a single [`Command`].
    /// Leverages previous type information in the [`EGraph`] to do so, adding new type information.
    /// When will_run is true, adds to `desugared_commands_run_so_far`, which is used for proof checking.
    fn resolve_command(&mut self, command: Command) -> Result<ResolvedNCommands, Error> {
        let resolved_before_proofs = self.resolve_command_before_proofs(command)?;

        // Add term encoding when it is enabled
        if self.proof_state.original_typechecking.is_none() {
            Ok(ResolvedNCommands {
                desugared: resolved_before_proofs,
                desugared_before_proofs: vec![],
            })
        } else {
            // Now remove globals for actual execution (but NOT from desugared_commands)
            let typechecked_no_globals = proof_global_remover::remove_globals(
                resolved_before_proofs.clone(),
                &mut self.parser.symbol_gen,
            );
            for command in &typechecked_no_globals {
                self.names.check_shadowing(command)?;
            }

            let mut term_encoding_added =
                ProofInstrumentor::add_term_encoding(self, typechecked_no_globals);
            if self.seminaive_encoding_enabled {
                let emit_header = !self.seminaive_encoding_header_emitted;
                term_encoding_added = proofs::seminaive_encoding::add_seminaive_encoding(
                    term_encoding_added,
                    &mut self.parser,
                    &mut self.seminaive_tracked,
                    emit_header,
                );
                self.seminaive_encoding_header_emitted = true;
            }
            let mut new_typechecked = vec![];
            for new_cmd in term_encoding_added {
                let desugared =
                    desugar_command(new_cmd, &mut self.parser, self.proof_state.proof_testing)?;
                for cmd in &desugared {
                    log::debug!("Desugared term encoding: {}", cmd.to_command());
                }

                // Now typecheck using self, adding term type information.
                let desugared_typechecked = self.typecheck_program(&desugared)?;
                // remove globals again, but this time allow primitive globals
                let desugared_typechecked = remove_globals::remove_globals(
                    desugared_typechecked,
                    &mut self.parser.symbol_gen,
                );

                new_typechecked.extend(desugared_typechecked);
            }
            Ok(ResolvedNCommands {
                desugared: new_typechecked,
                desugared_before_proofs: resolved_before_proofs,
            })
        }
    }

    /// Run a program, returning the desugared outputs as well as the CommandOutputs.
    /// Can optionally not run the commands, just adding type information.
    /// Parse + resolve a program into the normalized IR
    /// (`ResolvedNCommand`s) without executing it. Used by alternative
    /// backends (e.g. DuckDB) that consume the resolved IR directly.
    pub(crate) fn resolve_program_to_ncommands(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<ResolvedNCommand>, Error> {
        let parsed = self.parser.get_program_from_string(filename, input)?;
        let res = self.process_program_internal(parsed, false)?;
        Ok(res.resolved)
    }

    fn process_program_internal(
        &mut self,
        program: Vec<Command>,
        run_commands: bool,
    ) -> Result<ResolvedNCommandsWithOutput, Error> {
        let mut outputs = Vec::new();
        let mut desugared_before_proofs = Vec::new();
        let mut desugared = Vec::new();

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
                    let resolved = self.process_program_internal(included_program, run_commands)?;
                    outputs.extend(resolved.outputs);
                    desugared.extend(resolved.resolved);
                    desugared_before_proofs.extend(resolved.resolved_before_proofs);
                } else {
                    let resolved = self.resolve_command(command)?;
                    if run_commands && self.are_proofs_enabled() {
                        self.proof_check_program
                            .extend(resolved.desugared_before_proofs.clone());
                    }

                    desugared_before_proofs.extend(resolved.desugared_before_proofs);
                    desugared.extend(resolved.desugared.clone());

                    for processed in resolved.desugared {
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

        Ok(ResolvedNCommandsWithOutput {
            outputs,
            resolved_before_proofs: desugared_before_proofs,
            resolved: desugared,
        })
    }

    /// Run a program, represented as an AST.
    /// Return a list of messages.
    pub fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<CommandOutput>, Error> {
        let res = self.process_program_internal(program, true)?;
        Ok(res.outputs)
    }

    /// Resolves an egglog program by parsing, typechecking, and desugaring each command.
    /// Outputs a new egglog program without any syntactic sugar, either user provided ([`CommandMacro`]) or built-in (e.g., `rewrite` commands).
    /// Also removes globals from the program by replacing with new constructors.
    pub fn resolve_program(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<ResolvedCommand>, Error> {
        let parsed = self.parser.get_program_from_string(filename, input)?;
        let res = self.process_program_internal(parsed, false)?;
        Ok(res.resolved.into_iter().map(|c| c.to_command()).collect())
    }

    /// Takes a source program `input` and parses it into a list of [`Command`]s.
    pub fn parse_program(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<Command>, Error> {
        let parsed = self.parser.get_program_from_string(filename, input)?;
        Ok(parsed)
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
        self.bridge().base_values().unwrap::<T>(x)
    }

    /// Convert from a Rust type to an egglog value.
    pub fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.bridge().base_values().get::<T>(x)
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
        self.bridge().container_values().get_val::<T>(x)
    }

    /// Convert from a Rust container type to an egglog value.
    pub fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        // The bridge's `get_container_value` already wraps the
        // `with_execution_state` + `register_val::<T>` sequence (it also
        // idempotently registers the type). Since
        // `container_register_val<C>` on the trait routes through
        // `register_val_dyn`, which is `unimplemented!()` on the bridge under
        // the current Phase 2 state, we call the bridge's concrete method
        // directly here. When `EGraph::backend` becomes `Box<dyn Backend>`
        // in Commit 8, this site will need a typed container-registration
        // path on the trait.
        self.bridge_mut().get_container_value::<T>(x)
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
        let func = self
            .functions
            .get(name)
            .unwrap_or_else(|| panic!("Could not find function {name}"))
            .backend_id;
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
            .get_canon_repr(val, sort.column_ty(&*self.backend))
    }

    /// Create a new union action that can be used to union two values.
    pub fn new_union_action(&self) -> egglog_bridge::UnionAction {
        UnionAction::new(self.bridge())
    }
}

struct BackendRule<'a> {
    rb: Box<dyn egglog_backend_trait::RuleBuilderOps + 'a>,
    backend: &'a dyn egglog_backend_trait::Backend,
    entries: HashMap<core::ResolvedAtomTerm, QueryEntry>,
    functions: &'a IndexMap<String, Function>,
    type_info: &'a TypeInfo,
}

impl<'a> BackendRule<'a> {
    fn new(
        rb: Box<dyn egglog_backend_trait::RuleBuilderOps + 'a>,
        backend: &'a dyn egglog_backend_trait::Backend,
        functions: &'a IndexMap<String, Function>,
        type_info: &'a TypeInfo,
    ) -> BackendRule<'a> {
        BackendRule {
            rb,
            backend,
            functions,
            type_info,
            entries: Default::default(),
        }
    }

    fn entry(&mut self, x: &core::ResolvedAtomTerm) -> QueryEntry {
        self.entries
            .entry(x.clone())
            .or_insert_with(|| match x {
                core::GenericAtomTerm::Var(_, v) => {
                    let ty = v.sort.column_ty(self.backend);
                    self.rb.new_var_named(ty, &v.name)
                }
                core::GenericAtomTerm::Literal(_, l) => literal_to_entry_dyn(self.backend, l),
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

        if prim.name() == "unstable-fn" {
            // `unstable-fn` is a bridge-only feature: it lives on the
            // ResolvedFunction container and requires a `TableAction`
            // that the trait surface doesn't expose. DuckDB programs
            // never reach this branch because `FunctionSort`
            // containers are gated out at upstream typechecking.
            let bridge = self
                .backend
                .as_any()
                .downcast_ref::<egglog_bridge::EGraph>()
                .expect("unstable-fn requires the bridge backend");
            let core::ResolvedAtomTerm::Literal(_, Literal::String(ref name)) = args[0] else {
                panic!("expected string literal after `unstable-fn`")
            };
            let id = if let Some(f) = self.type_info.get_func_type(name) {
                ResolvedFunctionId::Lookup(egglog_bridge::TableAction::new(
                    bridge,
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
                ResolvedFunctionId::Prim(ps.into_iter().next().unwrap().id)
            } else {
                panic!("no callable for {name}");
            };
            let partial_arcsorts = prim.input().iter().skip(1).cloned().collect();

            qe_args[0] = bridge.base_value_constant(ResolvedFunction {
                id,
                partial_arcsorts,
                name: name.clone(),
            });
        }

        // Type-disambiguate built-in primitive names for the duckdb
        // backend. Several egglog primitives share a name across
        // sorts (`^` is XOR for i64 / POWER for f64; `/` is integer-
        // div for i64 / float-div for f64; `+` is string concat for
        // String / numeric add for i64/f64). DuckDB's SQL operators
        // pick one meaning per symbol, and the duck rule-builder
        // dispatches by name, so without an override the wrong SQL
        // gets emitted. Each `SpecializedPrimitive` carries a unique
        // `external_id`; we rename per-id so the duck side maps each
        // id to the SQL form for its specific type.
        //
        // Hardcoded for now — see also `compile.rs::prim_sql`. A more
        // principled design (per-primitive SQL emitter registered via
        // the macro) is deferred.
        let pname = prim.name();
        let pout = prim.output().name();
        let in0 = prim.input().first().map(|s| s.name());
        // For sort-overloaded primitives, route the duckdb backend's
        // rule-builder to a sort-specific duck name. compile.rs's
        // `prim_sql` maps these to either native SQL ops or — for
        // BigRat operations that have no SQL equivalent — to per-op
        // UDFs registered on demand via
        // `EGraph::set_external_func_name` → `register_builtin_prim_udf`.
        let duck_name: Option<&str> = match (pname, &*pout, in0) {
            ("^", "i64", _) => Some("i64-xor"),
            ("/", "i64", _) => Some("int-div"),
            ("+", "String", _) => Some("string-concat"),
            // BigRat overloads: dispatch by op name and arg sort.
            // Comparisons return Unit; arithmetic returns BigRat;
            // numer/denom return BigInt; to-f64 returns f64. We use
            // the BigRat *argument* type to disambiguate from i64/f64
            // overloads of the same op name.
            ("+", "BigRat", _) => Some("bigrat-add"),
            ("-", "BigRat", _) => Some("bigrat-sub"),
            ("*", "BigRat", _) => Some("bigrat-mul"),
            ("/", "BigRat", _) => Some("bigrat-div"),
            ("min", "BigRat", _) => Some("bigrat-min"),
            ("max", "BigRat", _) => Some("bigrat-max"),
            ("pow", "BigRat", _) => Some("bigrat-pow"),
            ("neg", "BigRat", _) => Some("bigrat-neg"),
            ("abs", "BigRat", _) => Some("bigrat-abs"),
            ("floor", "BigRat", _) => Some("bigrat-floor"),
            ("ceil", "BigRat", _) => Some("bigrat-ceil"),
            ("round", "BigRat", _) => Some("bigrat-round"),
            ("sqrt", "BigRat", _) => Some("bigrat-sqrt"),
            ("log", "BigRat", _) => Some("bigrat-log"),
            ("cbrt", "BigRat", _) => Some("bigrat-cbrt"),
            ("<", "Unit", Some("BigRat")) => Some("bigrat-lt"),
            (">", "Unit", Some("BigRat")) => Some("bigrat-gt"),
            ("<=", "Unit", Some("BigRat")) => Some("bigrat-le"),
            (">=", "Unit", Some("BigRat")) => Some("bigrat-ge"),
            ("numer", _, Some("BigRat")) => Some("bigrat-numer"),
            ("denom", _, Some("BigRat")) => Some("bigrat-denom"),
            ("to-f64", _, Some("BigRat")) => Some("bigrat-to-f64"),
            _ => None,
        };
        if let Some(name) = duck_name {
            self.rb.rename_prim(prim.external_id(), name.to_owned());
        }

        (
            prim.external_id(),
            qe_args,
            prim.output().column_ty(self.backend),
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
                    self.rb.query_table(f, &args, is_subsumed).unwrap();
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
                            let name = f.name.clone();
                            let f = self.func(f);
                            let args = self.args(args);
                            let span = span.clone();
                            self.rb.lookup(
                                f,
                                &args,
                                format!("{span}: lookup of function {name} failed"),
                            )
                        }
                        ResolvedCall::Primitive(p) => {
                            let name = p.name().to_owned();
                            let (p, args, ty) = self.prim(p, args);
                            let span = span.clone();
                            self.rb.call_external_func(
                                p,
                                &args,
                                ty,
                                format!("{span}: call of primitive {name} failed"),
                            )
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
                core::GenericCoreAction::Change(span, change, f, args) => match f {
                    ResolvedCall::Primitive(..) => panic!("runtime primitive change!"),
                    ResolvedCall::Func(f) => {
                        let name = f.name.clone();
                        let can_subsume = self.functions[&f.name].can_subsume;
                        let f = self.func(f);
                        let args = self.args(args);
                        match change {
                            Change::Delete => self.rb.remove(f, &args),
                            Change::Subsume if can_subsume => {
                                self.rb
                                    .subsume(f, &args)
                                    .map_err(|e| Error::BackendError(format!("subsume failed: {e}")))?;
                            }
                            Change::Subsume => {
                                return Err(Error::SubsumeMergeError(name, span.clone()));
                            }
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
        self.rb.build().expect("rule build failed")
    }

    fn into_rb(self) -> Box<dyn egglog_backend_trait::RuleBuilderOps + 'a> {
        self.rb
    }
}

/// Trait-routed literal-to-entry helper. Constructs a typed
/// `QueryEntry::Const` via the backend's `BaseValuePool` so the
/// caller doesn't need to downcast to the concrete bridge.
fn literal_to_entry_dyn(
    backend: &dyn egglog_backend_trait::Backend,
    l: &Literal,
) -> QueryEntry {
    use egglog_backend_trait::{pool_get, pool_get_ty};
    let pool = backend.base_value_pool();
    match l {
        Literal::Int(x) => {
            let val = pool_get::<i64>(pool, *x);
            backend.base_value_constant_dyn(val, pool_get_ty::<i64>(pool))
        }
        Literal::Float(x) => {
            let val = pool_get::<sort::F>(pool, x.into());
            backend.base_value_constant_dyn(val, pool_get_ty::<sort::F>(pool))
        }
        Literal::String(x) => {
            let val = pool_get::<sort::S>(pool, sort::S::new(x.clone()));
            backend.base_value_constant_dyn(val, pool_get_ty::<sort::S>(pool))
        }
        Literal::Bool(x) => {
            let val = pool_get::<bool>(pool, *x);
            backend.base_value_constant_dyn(val, pool_get_ty::<bool>(pool))
        }
        Literal::Unit => {
            let val = pool_get::<()>(pool, ());
            backend.base_value_constant_dyn(val, pool_get_ty::<()>(pool))
        }
    }
}

fn literal_to_value(egraph: &egglog_bridge::EGraph, l: &Literal) -> Value {
    match l {
        Literal::Int(x) => egraph.base_values().get::<i64>(*x),
        Literal::Float(x) => egraph.base_values().get::<sort::F>(x.into()),
        Literal::String(x) => egraph.base_values().get::<sort::S>(sort::S::new(x.clone())),
        Literal::Bool(x) => egraph.base_values().get::<bool>(*x),
        Literal::Unit => egraph.base_values().get::<()>(()),
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
    #[error("{span}\n{error}")]
    ProofError {
        span: Span,
        #[source]
        error: ProveExistsError,
    },
    #[error("{1}\n{2}\nShadowing is not allowed, but found {0}")]
    Shadowing(String, Span, Span),
    #[error("{1}\nCommand already exists: {0}")]
    CommandAlreadyExists(String, Span),
    #[error("Incorrect format in file '{0}'.")]
    InputFileFormatError(String),
    #[error(
        "Command is not supported by the current proof term encoding implementation.\n\
         Reason: {reason}\n\
         This typically means the command uses constructs that cannot yet be represented as proof terms.\n\
         Consider disabling proof term encoding for this run or rewriting the command to avoid unsupported features.\n\
         Offending command: {command}"
    )]
    UnsupportedProofCommand {
        command: String,
        reason: ProofEncodingUnsupportedReason,
    },
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
        egraph
            .backend
            .for_each(id, &mut |row| out = Some(row.vals[0]));
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
