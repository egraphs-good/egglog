#![doc = include_str!("lib.md")]
pub mod api;
pub mod ast;
pub mod builtin;
#[cfg(feature = "bin")]
mod cli;
mod command_macro;
pub mod constraint;
mod core;
mod exec_state;
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
pub use core_relations::{BaseValue, ContainerValue, Value};
use core_relations::{ExecutionState, ExternalFunctionId, make_external_func};
use csv::Writer;
pub use egglog_add_primitive::add_literal_prim;
pub use egglog_add_primitive::add_primitive;
pub use egglog_add_primitive::add_primitive_with_validator;
use egglog_ast::generic_ast::{Change, GenericExpr, Literal};
use egglog_ast::span::Span;
use egglog_ast::util::ListDisplay;
use egglog_bridge::{ColumnTy, QueryEntry};
use egglog_core_relations as core_relations;
use egglog_numeric_id as numeric_id;
use egglog_reports::{ReportLevel, RunReport};
pub use exec_state::{
    Context, Core, Enode, FullState, FunctionEntry, PureState, Read, ReadState, Write, WriteState,
};
use extract::DefaultCost;
use indexmap::map::Entry;
use log::{Level, log_enabled};
use numeric_id::DenseIdMap;
use prelude::*;
pub use proofs::proof_encoding_helpers::{file_supports_proofs, program_supports_proofs};

/// Read-only proof reconstruction API.
pub mod proof {
    pub use crate::proofs::proof_extraction::ProveExistsError;
    pub use crate::proofs::proof_format::{Justification, Proof, ProofId, ProofStore, Proposition};
}
use scheduler::{SchedulerId, SchedulerRecord};
pub use serialize::{SerializeConfig, SerializeOutput, SerializedNode};
use sort::*;
use std::any::{Any, TypeId};
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io::{Read as _, Write as _};
use std::iter::once;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
pub use termdag::{OrdTerm, Term, TermDag, TermId};
use thiserror::Error;
pub use typechecking::FuncType;
pub use typechecking::PrimitiveValidator;
pub use typechecking::PrimitiveWithId;
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

/// Whether native work belongs to a requested command or compiler-generated work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionOrigin {
    Requested,
    Internal,
}

/// Location in the schedule before proof instrumentation. Iterations address
/// its enclosing loops, not any generated maintenance loops.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleLocation<'a> {
    pub path: &'a [usize],
    pub iterations: &'a [usize],
}

/// The actual branch which completed a native loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeLoopStop {
    CountReached,
    CanStop,
    Unchanged,
}

/// Whether a bounded native Repeat may finish before its count is reached.
/// This policy applies only to requested schedule locations; compiler-generated
/// maintenance loops retain the native early-stop behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepeatContinuation {
    CanStop,
    ExactCount,
}

/// Synchronous observations from the existing native interpreter.
///
/// Observers can aggregate leaf reports without retaining an event history. A
/// failed run has no measured report; its entry must not be mistaken for zero
/// work. Schedule outputs include their inline maintenance, so adding their
/// timings to the leaf reports would count that work twice.
#[derive(Debug)]
pub enum ExecutionEvent<'a> {
    /// One input command has been desugared and typechecked, before
    /// global removal, proof admission or instrumentation. The borrowed batch
    /// retains original globals and source-level definitions. This is not an
    /// installation/completion event: later admission or execution may fail.
    CommandResolved {
        commands: &'a [(ResolvedNCommand, ExecutionOrigin)],
    },
    /// One actual action block completed or failed after its ordinary recovery.
    /// `changed` includes committed prefixes on error, not attempted writes.
    /// This is not a Run and adds no schedule output, rule report or timing.
    ActionsFinished {
        origin: ExecutionOrigin,
        changed: bool,
        result: Result<(), &'a Error>,
    },
    /// A command in the resolved execution plan completed before any following
    /// maintenance. Definition installation can succeed before a helper fails.
    CommandCompleted {
        origin: ExecutionOrigin,
        command: &'a ResolvedNCommand,
    },
    /// A resolved command returned an execution error. Resolution errors and
    /// inner errors caught by `Fail` do not emit this event. A failure in later
    /// maintenance remains separate from an earlier requested completion.
    CommandFailed {
        origin: ExecutionOrigin,
        command: &'a ResolvedNCommand,
        error: &'a Error,
    },
    RunStarted {
        origin: ExecutionOrigin,
        schedule: Option<ScheduleLocation<'a>>,
    },
    RunFinished {
        origin: ExecutionOrigin,
        schedule: Option<ScheduleLocation<'a>>,
        result: Result<&'a RunReport, &'a Error>,
    },
    LoopFinished {
        origin: ExecutionOrigin,
        schedule: Option<ScheduleLocation<'a>>,
        completed: usize,
        stop: Option<NativeLoopStop>,
        result: Result<&'a RunReport, &'a Error>,
    },
}

/// A temporary native action compiler with typed local bindings. This is not a
/// stored program or a new command language: ordinary actions use the same
/// typechecker and backend lowering as top-level actions, in one execution
/// boundary. Fresh bindings and captures are positioned among those actions.
pub struct ScopedActionBuilder<'a> {
    translator: BackendRule<'a>,
    symbol_gen: &'a mut SymbolGen,
    bindings: IndexMap<String, (Span, ArcSort)>,
    captured: Arc<std::sync::Mutex<Vec<(String, Value, ColumnTy)>>>,
}

impl ScopedActionBuilder<'_> {
    /// Typecheck and compile an ordinary action without promoting local lets
    /// to global nullary tables. Global references still undergo native global
    /// removal, and all actions retain their authored order.
    pub fn action(&mut self, action: Action) -> Result<(), Error> {
        if let Action::Let(_, name, _) = &action
            && self.bindings.contains_key(name)
        {
            return Err(Error::BackendError(format!("local {name} already bound")));
        }
        let mut used = IndexSet::default();
        action.clone().visit_exprs(&mut |expr| {
            if let Expr::Var(_, name) = &expr {
                used.insert(name.clone());
            }
            expr
        });
        let bindings: IndexMap<_, _> = used
            .iter()
            .filter_map(|name| {
                self.bindings
                    .get(name)
                    .map(|binding| (name.as_str(), binding.clone()))
            })
            .collect();
        let resolved = self.translator.type_info.typecheck_standalone_actions(
            self.symbol_gen,
            &Actions::singleton(action),
            &bindings,
            Context::Full,
        )?;
        let resolved = ResolvedActions::new(
            resolved
                .0
                .into_iter()
                .map(|action| {
                    action.map_exprs(&mut |expr| remove_globals::remove_globals_expr(expr.clone()))
                })
                .collect(),
        );
        let mut bound = bindings
            .iter()
            .map(|(name, (_, sort))| ResolvedVar {
                name: (*name).to_owned(),
                sort: sort.clone(),
                is_global_ref: false,
            })
            .collect();
        let mut context = CoreActionContext::new(
            self.translator.type_info,
            &mut bound,
            self.symbol_gen,
            false,
        );
        let (core, _) = resolved.to_core_actions(&mut context)?;
        self.translator.actions(&core)?;
        for action in resolved.0 {
            if let ResolvedAction::Let(span, variable, _) = action {
                self.bindings.insert(variable.name, (span, variable.sort));
            }
        }
        Ok(())
    }

    /// Bind a fresh equality class at this position if execution reaches it.
    /// Merely compiling this operation cannot allocate an identity.
    pub fn fresh_eq_class(&mut self, name: String, sort: &str, span: Span) -> Result<(), Error> {
        if self.bindings.contains_key(&name) {
            return Err(Error::BackendError(format!("local {name} already bound")));
        }
        let sort = self
            .translator
            .type_info
            .sorts
            .get(sort)
            .filter(|sort| sort.is_eq_sort())
            .ok_or_else(|| Error::BackendError(format!("unknown equality sort {sort}")))?
            .clone();
        let value = self.translator.rb.fresh_eq_class(Arc::from(sort.name()));
        let variable = ResolvedVar {
            name: name.clone(),
            sort: sort.clone(),
            is_global_ref: false,
        };
        self.translator
            .entries
            .insert(ResolvedAtomTerm::Var(span.clone(), variable), value.into());
        self.bindings.insert(name, (span, sort));
        Ok(())
    }

    /// Retain an equality local once this position has completed. Captures
    /// preceding a failing later action are returned by `eval_scoped_actions`.
    pub fn capture_eq_class(&mut self, name: &str) -> Result<(), Error> {
        self.bindings
            .get(name)
            .filter(|(_, sort)| sort.is_eq_sort())
            .ok_or_else(|| Error::BackendError(format!("unbound equality local {name}")))?;
        self.capture_local(name).map(|_| ())
    }

    /// Capture a bound local and return its native sort. Equality values are
    /// canonicalized after recovery; base/container handles retain their type.
    pub fn capture_local(&mut self, name: &str) -> Result<ArcSort, Error> {
        let (span, sort) = self
            .bindings
            .get(name)
            .ok_or_else(|| Error::BackendError(format!("unbound local {name}")))?;
        let column = sort.column_ty(self.translator.rb.egraph());
        let argument = self.translator.entry(&ResolvedAtomTerm::Var(
            span.clone(),
            ResolvedVar {
                name: name.to_owned(),
                sort: sort.clone(),
                is_global_ref: false,
            },
        ));
        let captured = self.captured.clone();
        let name = name.to_owned();
        self.translator.rb.call_scoped_infallible(
            Box::new(make_external_func(move |_, args| {
                let [value] = args else {
                    unreachable!("capture has one argument")
                };
                captured
                    .lock()
                    .unwrap()
                    .push((name.clone(), *value, column));
                Some(*value)
            })),
            &[argument],
            column,
        );
        Ok(sort.clone())
    }
}

/// Methods shared by every kind-specific primitive trait.
///
/// `name` and `get_type_constraints` aren't capability-dependent, so
/// the four kind-specific traits ([`PurePrim`], [`WritePrim`],
/// [`ReadPrim`], [`FullPrim`]) share this supertrait.
pub trait Primitive: Send + Sync + Any {
    /// Returns the name of this primitive.
    fn name(&self) -> &str;

    /// Constructs a type constraint for this primitive.
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint>;

    /// The authoritative native scheme, when this registration has one.
    /// Missing descriptions remain explicit inventory gaps.
    fn builtin_description(&self) -> Option<&builtin::BuiltinDescription> {
        self.builtin_instance()
            .map(|instance| instance.description.as_ref())
    }

    /// Checked instantiation of a presort-owned definition, when applicable.
    fn builtin_instance(&self) -> Option<&builtin::BuiltinInstance> {
        None
    }

    /// Whether this registration requires an independent native description.
    /// Internal compiler helpers may opt out; ordinary providers retain gaps.
    fn requires_catalog_definition(&self) -> bool {
        true
    }
}

/// A primitive whose body sees a [`PureState`]. Register via
/// [`EGraph::add_pure_primitive`].
pub trait PurePrim: Primitive {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value>;
}

/// A primitive whose body sees a [`WriteState`]. Register via
/// [`EGraph::add_write_primitive`].
pub trait WritePrim: Primitive {
    fn apply<'a, 'db>(&self, state: WriteState<'a, 'db>, args: &[Value]) -> Option<Value>;
}

/// A primitive whose body sees a [`ReadState`]. Register via
/// [`EGraph::add_read_primitive`].
pub trait ReadPrim: Primitive {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value>;
}

/// A primitive whose body sees a [`FullState`]. Register via
/// [`EGraph::add_full_primitive`].
pub trait FullPrim: Primitive {
    fn apply<'a, 'db>(&self, state: FullState<'a, 'db>, args: &[Value]) -> Option<Value>;
}

/// A type-erased output from a user-defined command.
pub trait UserDefinedCommandOutput: Debug + std::fmt::Display + Send + Sync {
    /// Views a stored output as [`Any`] so consumers can safely downcast it.
    ///
    /// Call this on a `&dyn UserDefinedCommandOutput`; smart pointers that
    /// themselves satisfy this trait should first be dereferenced with
    /// `as_ref()`.
    ///
    /// [`CommandOutput::UserDefined`] owns its output, so values stored there
    /// satisfy the required `'static` bound. Keeping the bound on this method
    /// lets borrowed values continue to implement `UserDefinedCommandOutput`.
    fn as_any(&self) -> &dyn Any
    where
        Self: 'static;
}
impl<T> UserDefinedCommandOutput for T
where
    T: Debug + std::fmt::Display + Send + Sync,
{
    fn as_any(&self) -> &dyn Any
    where
        Self: 'static,
    {
        self
    }
}

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

impl CommandOutput {
    /// Render command outputs to a string that is identical whether the program
    /// ran normally or under proof encoding (`--proofs`). Drops outputs that
    /// legitimately differ or are non-deterministic (timing, `PrintFunction`
    /// per #793, extraction variants) and reduces `ExtractBest` to its cost.
    pub fn snapshot_stable_under_proof_encoding(outputs: &[CommandOutput]) -> String {
        outputs
            .iter()
            .filter_map(|output| match output {
                CommandOutput::OverallStatistics(_) => None,
                CommandOutput::PrintFunction(..) => None,
                CommandOutput::ExtractBest(_, cost, _) => {
                    Some(format!("(extraction-costs {cost})\n"))
                }
                CommandOutput::ExtractVariants(..) => None,
                other => Some(other.to_string()),
            })
            .collect::<Vec<_>>()
            .join("")
    }
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
                let out_is_unit = function.func_type.output.name() == UnitSort.name();
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
pub(crate) trait ExtensionStateValue: Any + dyn_clone::DynClone + Send + Sync {}

impl<T> ExtensionStateValue for T where T: Any + Clone + Send + Sync {}

dyn_clone::clone_trait_object!(ExtensionStateValue);

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
    pub no_decomp: bool,
    type_info: TypeInfo,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,
    collect_reports: bool,
    schedulers: DenseIdMap<SchedulerId, SchedulerRecord>,
    commands: IndexMap<String, Arc<dyn UserDefinedCommand>>,
    extension_state: HashMap<TypeId, Box<dyn ExtensionStateValue>>,
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
    fn update(&self, egraph: &mut EGraph, args: &[Expr]) -> Result<Vec<CommandOutput>, Error>;

    /// Opts a declaration command into compile-only preparation. Implementations
    /// may typecheck and register code, but must not evaluate bodies, captures,
    /// queries, actions or validators. Runtime IDs belong to the supplied graph.
    fn prepare(&self, _egraph: &mut EGraph, _args: &[Expr]) -> Result<(), Error> {
        Err(Error::BackendError(
            "compile-only validation does not execute user-defined commands".into(),
        ))
    }
}

/// A function in the e-graph.
///
/// This contains the schema information of the function and
/// the backend id of the function in the e-graph.
#[derive(Clone)]
pub struct Function {
    decl: ResolvedFunctionDecl,
    func_type: Arc<FuncType>,
    can_subsume: bool,
    backend_id: egglog_bridge::FunctionId,
}

impl Function {
    /// Get the name of the function.
    pub fn name(&self) -> &str {
        &self.decl.name
    }

    /// The function's resolved signature.
    pub fn func_type(&self) -> &FuncType {
        &self.func_type
    }

    /// Whether this function supports subsumption.
    pub fn can_subsume(&self) -> bool {
        self.can_subsume
    }

    /// Whether this is a let binding
    pub fn is_let_binding(&self) -> bool {
        self.decl.internal_let
    }

    /// Whether this function is internally hidden (e.g., compiler-generated
    /// helper tables that should not appear in user-facing listings).
    pub fn is_hidden(&self) -> bool {
        self.decl.internal_hidden
    }

    /// The term-constructor name associated with this function table, if
    /// any. Set on view tables created by the term/proof encoding to refer
    /// back to the user-visible constructor name.
    pub fn term_constructor(&self) -> Option<&str> {
        self.decl.term_constructor.as_deref()
    }

    /// Whether this constructor is excluded from ordinary extraction.
    pub fn is_unextractable(&self) -> bool {
        self.decl.unextractable
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("decl", &self.decl)
            .field("func_type", &self.func_type)
            .finish()
    }
}

impl Default for EGraph {
    fn default() -> Self {
        let mut parser = Parser::default();
        let proof_state = EncodingState::new(&mut parser.symbol_gen);
        let mut eg = Self {
            backend: Default::default(),
            parser,
            names: Default::default(),
            pushed_egraph: Default::default(),
            functions: Default::default(),
            rulesets: Default::default(),
            fact_directory: None,
            seminaive: true,
            no_decomp: false,
            overall_run_report: Default::default(),
            collect_reports: true,
            type_info: Default::default(),
            schedulers: Default::default(),
            commands: Default::default(),
            extension_state: Default::default(),
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
        eg.type_info.add_presort::<SetSort>(span!()).unwrap();
        eg.type_info.add_presort::<VecSort>(span!()).unwrap();
        eg.type_info.add_presort::<FunctionSort>(span!()).unwrap();
        eg.type_info.add_presort::<MultiSetSort>(span!()).unwrap();
        eg.type_info.add_presort::<PairSort>(span!()).unwrap();
        eg.type_info.add_presort::<MapSort>(span!()).unwrap();

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
            "!=" [id = "egglog.core.ne"] = |a: #, b: #| -?> () {
                (a != b).then_some(())
            },
            neq_validator
        );

        add_primitive_with_validator!(
            &mut eg,
            "bool-!=" [id = "egglog.core.bool_ne"] = |a: #, b: #| -> bool {
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

        add_primitive!(&mut eg, "value-eq" [id = "egglog.core.value_eq"] = |a: #, b: #| -?> () {
            (a == b).then_some(())
        });
        add_primitive!(&mut eg, "ordering-min" [catalog = false] = |a: #, b: #| -> # {
            if a < b { a } else { b }
        });
        add_primitive!(&mut eg, "ordering-max" [catalog = false] = |a: #, b: #| -> # {
            if a > b { a } else { b }
        });

        eg.rulesets.insert(
            "".into(),
            Ruleset::Rules(Default::default(), ExecutionOrigin::Requested),
        );

        eg
    }
}

#[derive(Clone)]
struct OriginatedCommand<C> {
    command: C,
    origin: ExecutionOrigin,
    schedule: Option<ScheduleProvenance>,
    // Only the compiler's common command postlude, never inline Run rebuilding.
    command_tail: bool,
}

/// Flat correspondence only: no executable schedule or retained event history.
#[derive(Clone, Default)]
struct ScheduleProvenance {
    nodes: HashMap<Vec<usize>, ScheduleSite>,
}

#[derive(Clone)]
struct ScheduleSite {
    path: Vec<usize>,
    internal: bool,
    // Checks that the remaining compiler passes preserve the instrumented
    // topology. This is a node tag, not another schedule representation.
    kind: u8,
}

impl ScheduleProvenance {
    fn validate(&self, command: &ResolvedNCommand) -> Result<(), Error> {
        let mut command = command;
        while let ResolvedNCommand::Fail(_, child) = command {
            command = child;
        }
        let ResolvedNCommand::RunSchedule(schedule) = command else {
            return Err(Error::BackendError(
                "schedule provenance lost its command".into(),
            ));
        };
        let mut pending = vec![(schedule, vec![])];
        let mut seen = 0;
        while let Some((schedule, path)) = pending.pop() {
            let kind = match schedule {
                ResolvedSchedule::Run(..) => 0,
                ResolvedSchedule::Repeat(..) => 1,
                ResolvedSchedule::Saturate(..) => 2,
                ResolvedSchedule::Sequence(..) => 3,
            };
            if self.nodes.get(&path).is_none_or(|site| site.kind != kind) {
                return Err(Error::BackendError(
                    "schedule provenance topology changed".into(),
                ));
            }
            seen += 1;
            let children = match schedule {
                ResolvedSchedule::Run(..) => &[][..],
                ResolvedSchedule::Repeat(_, _, child) | ResolvedSchedule::Saturate(_, child) => {
                    std::slice::from_ref(child.as_ref())
                }
                ResolvedSchedule::Sequence(_, children) => children.as_slice(),
            };
            for (i, child) in children.iter().enumerate() {
                let mut child_path = path.clone();
                child_path.push(i);
                pending.push((child, child_path));
            }
        }
        if seen != self.nodes.len() {
            return Err(Error::BackendError(
                "schedule provenance contains stale nodes".into(),
            ));
        }
        Ok(())
    }
}

struct ScheduleObservation<'a> {
    provenance: Option<&'a ScheduleProvenance>,
    repeat_policy: &'a dyn Fn(ScheduleLocation<'_>) -> RepeatContinuation,
    scheduler_selector: &'a dyn Fn(ScheduleLocation<'_>) -> Option<scheduler::SchedulerId>,
    // Uninstrumented requested schedules need no correspondence table.
    identity: bool,
    path: Vec<usize>,
    iterations: Vec<usize>,
}

#[derive(Clone, Copy)]
struct CommandPolicies<'a> {
    repeat: &'a dyn Fn(ScheduleLocation<'_>) -> RepeatContinuation,
    scheduler: &'a dyn Fn(ScheduleLocation<'_>) -> Option<scheduler::SchedulerId>,
    accumulate_schedule: bool,
}

impl Default for CommandPolicies<'_> {
    fn default() -> Self {
        Self {
            repeat: &|_| RepeatContinuation::CanStop,
            scheduler: &|_| None,
            accumulate_schedule: true,
        }
    }
}

impl ScheduleObservation<'_> {
    fn site(&self) -> Option<(ScheduleLocation<'_>, bool)> {
        let (path, internal) = if let Some(provenance) = self.provenance {
            let site = &provenance.nodes[&self.path];
            (site.path.as_slice(), site.internal)
        } else if self.identity {
            (self.path.as_slice(), false)
        } else {
            return None;
        };
        Some((
            ScheduleLocation {
                path,
                iterations: &self.iterations,
            },
            internal,
        ))
    }
}

struct ResolvedNCommands {
    desugared: Vec<OriginatedCommand<ResolvedNCommand>>,
    /// In proof mode, populated with the desugared program before instrumented with proofs
    desugared_before_proofs: Vec<ResolvedNCommand>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProcessingMode {
    Execute,
    ExecuteEmbedded,
    Resolve,
    Compile,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CommandPurpose {
    Authored,
    Condition,
}

fn expect_failure<T>(span: Span, result: Result<T, Error>) -> Result<(), Error> {
    if let Err(error) = result {
        log::info!("Command failed as expected: {error}");
        Ok(())
    } else {
        Err(Error::ExpectFail(span))
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(String);

impl EGraph {
    /// Create a new e-graph configured with `num_threads`.
    ///
    /// Passing `1` keeps execution serial. Passing `0` uses available
    /// parallelism.
    pub fn new(num_threads: usize) -> Self {
        EGraph::default().with_num_threads(num_threads)
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

    /// Enable the term-encoding pipeline on a fresh, provider-registered e-graph.
    ///
    /// Call this during initialization, after registering providers and before
    /// resolving or executing user programs. The original typechecking state is
    /// captured here: existing declarations and actions are not retroactively
    /// instrumented, and later provider registrations are not copied into it.
    pub fn with_term_encoding_enabled(mut self) -> Self {
        self.proof_state.original_typechecking = Some(Box::new(self.clone()));
        self
    }

    /// Enable proof generation on a fresh, provider-registered e-graph.
    ///
    /// This has the same initialization precondition as
    /// [`Self::with_term_encoding_enabled`]: register providers first and enable
    /// proofs before resolving or executing user programs.
    pub fn with_proofs_enabled(mut self) -> Self {
        self = self.with_term_encoding_enabled();
        self.proof_state.proofs_enabled = true;
        self
    }

    /// Enable testing of getting proofs for all `check` commands.
    pub fn with_proof_testing(mut self) -> Self {
        self.proof_state.proof_testing = true;
        self
    }

    /// Return a copy of this e-graph configured with `num_threads`.
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.set_num_threads(num_threads);
        self
    }

    /// Set the number of threads used by this e-graph.
    ///
    /// Passing `1` keeps execution serial. Passing `0` uses available
    /// parallelism.
    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.backend.set_num_threads(num_threads);
        if let Some(original) = &mut self.proof_state.original_typechecking {
            original.set_num_threads(num_threads);
        }
    }

    /// Return the number of threads configured for this e-graph.
    pub fn num_threads(&self) -> usize {
        self.backend.num_threads()
    }

    /// Return extension-owned state stored on this e-graph.
    ///
    /// Extension state is keyed by Rust type and follows the same lifecycle as
    /// the rest of the e-graph: cloning an [`EGraph`] clones the state, and
    /// `push`/`pop` snapshots and restores it.
    pub fn extension_state<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        let value = self.extension_state.get(&TypeId::of::<T>())?;
        (value.as_ref() as &dyn Any).downcast_ref()
    }

    /// Return mutable extension-owned state, inserting `T::default()` when absent.
    pub fn extension_state_or_default<T>(&mut self) -> &mut T
    where
        T: Default + Clone + Send + Sync + 'static,
    {
        let value = self
            .extension_state
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()));
        (value.as_mut() as &mut dyn Any)
            .downcast_mut()
            .expect("extension state entry must have the requested type")
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
    /// It preserves the cumulative run report and complete symbol generator
    /// from the popped egraph.
    pub fn pop(&mut self) -> Result<(), Error> {
        match self.pushed_egraph.take() {
            Some(saved) => {
                self.restore_from(*saved);
                Ok(())
            }
            None => Err(Error::Pop(span!())),
        }
    }

    /// Restore a saved graph with the same retained-state invariant as [`Self::pop`].
    ///
    /// The current cumulative report replaces the saved report (it already
    /// includes that prefix). Keep the complete current symbol generator so
    /// future fresh symbols cannot collide with names minted in the scope.
    /// All other state, including resources and the push stack, comes from saved.
    /// Callers owning external snapshots must validate their lifecycle pairing.
    pub fn restore_from(&mut self, mut saved: Self) {
        std::mem::swap(&mut self.overall_run_report, &mut saved.overall_run_report);
        std::mem::swap(&mut self.parser.symbol_gen, &mut saved.parser.symbol_gen);
        *self = saved;
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
                let mut translated_args = args
                    .iter()
                    .map(|arg| self.translate_expr_to_mergefn(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                if p.name() == "unstable-fn" {
                    let Some(GenericExpr::Lit(span, Literal::String(name))) = args.first() else {
                        return Err(Error::BackendError(
                            "expected string literal after `unstable-fn`".into(),
                        ));
                    };
                    let resolved = resolve_function_container_target_with_context(
                        &self.backend,
                        &self.functions,
                        &self.type_info,
                        name,
                        p,
                        self.backend
                            .action_registry()
                            .read()
                            .unwrap()
                            .default_panic_id(),
                        // Merge expressions evaluate in the `Write` context.
                        crate::Context::Write,
                        span,
                    )?;
                    translated_args[0] =
                        egglog_bridge::MergeFn::Const(self.backend.base_values().get(resolved));
                }
                Ok(egglog_bridge::MergeFn::Primitive(
                    p.external_id(crate::Context::Write),
                    translated_args,
                ))
            }
        }
    }

    fn declare_function(&mut self, decl: &ResolvedFunctionDecl) -> Result<(), Error> {
        let default_name = match &decl.table_name {
            Some(ast::TableName::Default(name)) => Some(name),
            None => Some(&decl.name),
            _ => None,
        };
        if let Some(name) = default_name
            && (self.type_info.table_names.values().any(|(other, _)| {
                matches!(other, ast::TableName::Default(other_name) if other_name == name)
            }) || matches!(decl.table_name, Some(ast::TableName::Default(_)))
                && self.functions.get(name).is_some_and(|function| function.func_type().table_name.is_none()))
        {
            return Err(TypeError::TableNameAlreadyBound(name.clone(), decl.span.clone()).into());
        }
        if let Some(ast::TableName::Public(name)) = &decl.table_name
            && self.type_info.table_names.values().any(|(other, _)| {
                matches!(other, ast::TableName::Public(other_name) if other_name == name)
            })
        {
            return Err(TypeError::TableNameAlreadyBound(name.clone(), decl.span.clone()).into());
        }
        // Typechecking records the signatures it resolves, so reuse that.
        // Desugaring also generates declarations (the functions global bindings
        // lower to, proof tables) without typechecking them; resolving those
        // here is what keeps every declared function reachable by name.
        let func_type =
            match self.type_info.get_func_type(&decl.name) {
                Some(func_type) => {
                    debug_assert!(
                        func_type.subtype == decl.subtype
                            && func_type.input.len() == decl.schema.input.len()
                            && func_type.input.iter().zip(&decl.schema.input).all(
                                |(sort, name)| {
                                    self.type_info
                                        .get_sort_by_name(name)
                                        .is_some_and(|declared| sort.name() == declared.name())
                                }
                            )
                            && self
                                .type_info
                                .get_sort_by_name(&decl.schema.output)
                                .is_some_and(|sort| func_type.output.name() == sort.name()),
                        "recorded signature for {} disagrees with its declaration",
                        decl.name
                    );
                    func_type.clone()
                }
                None => {
                    let get_sort = |name: &String| match self.type_info.get_sort_by_name(name) {
                        Some(sort) => Ok(sort.clone()),
                        None => Err(Error::TypeError(TypeError::UndefinedSort(
                            name.to_owned(),
                            decl.span.clone(),
                        ))),
                    };
                    let func_type = Arc::new(FuncType {
                        name: decl.name.clone(),
                        table_name: decl.table_name.clone(),
                        subtype: decl.subtype,
                        input: decl
                            .schema
                            .input
                            .iter()
                            .map(get_sort)
                            .collect::<Result<Vec<_>, _>>()?,
                        output: get_sort(&decl.schema.output)?,
                    });
                    self.type_info.declare_func_type(func_type.clone());
                    func_type
                }
            };

        let can_subsume = match decl.subtype {
            FunctionSubtype::Constructor => true,
            // View tables (functions with term_constructor) need subsumption support
            FunctionSubtype::Custom => decl.term_constructor.is_some(),
        };

        use egglog_bridge::{DefaultVal, MergeFn};
        let backend_id = self.backend.add_table(egglog_bridge::FunctionConfig {
            eq_sorts: func_type
                .input
                .iter()
                .chain([&func_type.output])
                .map(|sort| sort.is_eq_sort().then(|| Arc::<str>::from(sort.name())))
                .collect(),
            schema: func_type
                .input
                .iter()
                .chain([&func_type.output])
                .map(|sort| sort.column_ty(&self.backend))
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
            // Encoding creates ordinary let-binding tables for temporary
            // values. Preserve authored globals and their views using the
            // original namespace and the view's exact constructor backlink;
            // neither hidden/internal flags nor generated spellings suffice.
            compiler_binding: decl.internal_let
                && self
                    .proof_state
                    .original_typechecking
                    .as_ref()
                    .is_some_and(|original| {
                        !original.type_info.is_global(&decl.name)
                            && !decl.term_constructor.as_ref().is_some_and(|constructor| {
                                original.type_info.is_global(constructor)
                            })
                    }),
        });

        let function = Function {
            decl: decl.clone(),
            func_type,
            can_subsume,
            backend_id,
        };
        if function.is_hidden() {
            self.type_info
                .hidden_functions
                .insert(function.name().to_owned());
        }

        if let Some(name) = &decl.table_name {
            self.type_info.table_names.insert(
                decl.name.clone(),
                (
                    name.clone(),
                    egglog_bridge::TableAction::new(&self.backend, backend_id),
                ),
            );
        }
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
        file: Option<(File, PathBuf)>,
        span: Span,
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
            Some((mut file, path)) => {
                log::info!("Writing output to file");
                file.write_all(output.to_string().as_bytes())
                    .map_err(|e| Error::IoError(path, e, span.clone()))?;
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
        let resolved = proof_check_eg.process_program_internal(
            prog,
            ProcessingMode::Resolve,
            &mut |_, _, _| {},
            &mut |_, _| {},
            &|_, _| RepeatContinuation::CanStop,
            &|_, _| None,
            &mut |_, _| None,
        )?;

        self.proof_check_program = resolved.desugared_before_proofs;
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
    fn run_schedule(
        &mut self,
        sched: &ResolvedSchedule,
        origin: ExecutionOrigin,
        observation: &mut ScheduleObservation<'_>,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> Result<RunReport, Error> {
        match sched {
            ResolvedSchedule::Run(span, config) => {
                let origin = match self.rulesets.get(&config.ruleset) {
                    Some(
                        Ruleset::Rules(_, ExecutionOrigin::Internal)
                        | Ruleset::Combined(_, ExecutionOrigin::Internal),
                    ) => ExecutionOrigin::Internal,
                    _ if observation.site().is_some_and(|(_, internal)| internal) => {
                        ExecutionOrigin::Internal
                    }
                    _ => origin,
                };
                observer(ExecutionEvent::RunStarted {
                    origin,
                    schedule: observation.site().map(|(site, _)| site),
                });
                let result = self.run_rules(span, config, observer, &|| {
                    if origin != ExecutionOrigin::Requested {
                        return None;
                    }
                    observation.site().and_then(|(site, internal)| {
                        (!internal)
                            .then(|| (observation.scheduler_selector)(site))
                            .flatten()
                    })
                });
                observer(ExecutionEvent::RunFinished {
                    origin,
                    schedule: observation.site().map(|(site, _)| site),
                    result: result.as_ref(),
                });
                result
            }
            ResolvedSchedule::Repeat(_, _, child) | ResolvedSchedule::Saturate(_, child) => {
                let limit = match sched {
                    ResolvedSchedule::Repeat(_, count, _) => Some(*count),
                    _ => None,
                };
                let source_loop = observation.site().is_some_and(|(_, internal)| !internal);
                let exact_count = limit.is_some()
                    && observation.site().is_some_and(|(site, internal)| {
                        !internal
                            && (observation.repeat_policy)(site) == RepeatContinuation::ExactCount
                    });
                let mut completed = 0;
                let mut stop = NativeLoopStop::CountReached;
                let result = (|| {
                    let mut report = RunReport::default();
                    while limit.is_none_or(|limit| completed < limit) {
                        if limit.is_none() {
                            log::debug!(
                                "Saturate iteration {} start: {}",
                                completed + 1,
                                Self::schedule_for_log(child)
                            );
                        }
                        observation.path.push(0);
                        if source_loop {
                            observation.iterations.push(completed);
                        }
                        let rec = self.run_schedule(child, origin, observation, observer);
                        observation.path.pop();
                        if source_loop {
                            observation.iterations.pop();
                        }
                        let rec = rec?;
                        completed += 1;
                        let can_stop = rec.can_stop;
                        let updated = rec.updated;
                        if limit.is_none() {
                            log::debug!(
                                "Saturate iteration {completed} end: {}",
                                Self::run_report_debug_summary(&rec)
                            );
                        }
                        report.union(rec);
                        if limit.is_some() && can_stop && !exact_count {
                            stop = NativeLoopStop::CanStop;
                            break;
                        }
                        if limit.is_none() && !updated {
                            log::debug!("Saturate reached fixpoint after {completed} iteration(s)");
                            stop = NativeLoopStop::Unchanged;
                            break;
                        }
                    }
                    Ok(report)
                })();
                let site = observation.site();
                observer(ExecutionEvent::LoopFinished {
                    origin: if site.is_some_and(|(_, internal)| internal) {
                        ExecutionOrigin::Internal
                    } else {
                        origin
                    },
                    schedule: site.filter(|(_, internal)| !internal).map(|(site, _)| site),
                    completed,
                    stop: result.as_ref().ok().map(|_| stop),
                    result: result.as_ref(),
                });
                result
            }
            ResolvedSchedule::Sequence(_span, scheds) => {
                let mut report = RunReport::default();
                for (index, sched) in scheds.iter().enumerate() {
                    observation.path.push(index);
                    let result = self.run_schedule(sched, origin, observation, observer);
                    observation.path.pop();
                    report.union(result?);
                }
                Ok(report)
            }
        }
    }

    fn run_rules(
        &mut self,
        span: &Span,
        config: &ResolvedRunConfig,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
        scheduler_selector: &dyn Fn() -> Option<scheduler::SchedulerId>,
    ) -> Result<RunReport, Error> {
        log::debug!("Running ruleset: {}", config.ruleset);
        let mut report: RunReport = Default::default();

        let GenericRunConfig { ruleset, until } = config;

        if !self.rulesets.contains_key(ruleset) {
            return Err(Error::NoSuchRuleset(ruleset.clone(), span.clone()));
        }

        if let Some(facts) = until
            && self.check_facts(span, facts, observer).is_ok()
        {
            log::info!(
                "Breaking early because of facts:\n {}!",
                ListDisplay(facts, "\n")
            );
            return Ok(report);
        }

        let subreport = if let Some(scheduler) = scheduler_selector() {
            self.step_rules_with_scheduler(scheduler, ruleset)?
        } else {
            self.step_rules(ruleset)?
        };
        report.union(subreport);

        if log_enabled!(Level::Debug) {
            log::debug!(
                "Finished ruleset {ruleset}: database size {}, {}",
                self.num_tuples(),
                Self::run_report_debug_summary(&report)
            );
        }

        Ok(report)
    }

    fn run_report_debug_summary(report: &RunReport) -> String {
        let mut rules = report
            .num_matches_per_rule
            .iter()
            .filter(|(_, matches)| **matches > 0)
            .collect::<Vec<_>>();
        rules.sort_by(|(_, left), (_, right)| right.cmp(left));

        let top_rules = rules
            .into_iter()
            .take(5)
            .map(|(rule, matches)| {
                format!("{}={matches}", Self::truncate_for_log(rule.as_ref(), 80))
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "updated={}, can_stop={}, iterations={}, top_matches=[{}]",
            report.updated,
            report.can_stop,
            report.iterations.len(),
            top_rules
        )
    }

    fn schedule_for_log(sched: &ResolvedSchedule) -> String {
        Self::truncate_for_log(&sched.to_string(), 160)
    }

    fn truncate_for_log(s: &str, limit: usize) -> String {
        let mut s = s.replace('\n', " ");
        if s.len() > limit {
            s.truncate(limit);
            s.push_str("...");
        }
        s
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
                Ruleset::Rules(rules, _) => {
                    for rule in rules.values() {
                        ids.push(rule.backend_id);
                    }
                }
                Ruleset::Combined(sub_rulesets, _) => {
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
            .run_rules(&rule_ids, Some(&self.type_info))
            .map_err(|error| Error::from_backend(error.as_ref()))?;

        Ok(self.iteration_result(ruleset, iteration_report))
    }

    // Diagnostic collection must not allocate aggregate maps or an Arc history
    // for an otherwise control-only native invocation, including internal queries.
    fn iteration_result(
        &self,
        ruleset: &str,
        iteration: egglog_reports::IterationReport,
    ) -> RunReport {
        if self.collect_reports {
            RunReport::singleton(ruleset, iteration)
        } else {
            RunReport {
                updated: iteration.changed(),
                can_stop: !iteration.changed(),
                ..Default::default()
            }
        }
    }

    /// Inspect selected current rule queries without running or changing them.
    pub fn explain_rulesets(&self, names: &[String]) -> Result<serde_json::Value, Error> {
        fn collect(
            graph: &EGraph,
            name: &str,
            ids: &mut Vec<egglog_bridge::RuleId>,
        ) -> Result<(), Error> {
            match graph
                .rulesets
                .get(name)
                .ok_or_else(|| Error::NoSuchRuleset(name.into(), span!()))?
            {
                Ruleset::Rules(rules, _) => ids.extend(rules.values().map(|rule| rule.backend_id)),
                Ruleset::Combined(names, _) => {
                    for name in names {
                        collect(graph, name, ids)?;
                    }
                }
            }
            Ok(())
        }
        let mut ids = vec![];
        for name in names {
            collect(self, name, &mut ids)?;
        }
        let plans = self
            .backend
            .explain_rules(&ids)
            .map_err(|error| Error::from_backend(error.as_ref()))?;
        serde_json::to_value(plans).map_err(|error| Error::BackendError(error.to_string()))
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

        // The `:naive` rule option opts a single rule out of seminaive
        // evaluation. This widens primitive-context selection from
        // Pure/Write to Read/Full, so primitives that read or write the
        // database can run inside this rule.
        let seminaive = self.seminaive && !rule.eval_mode.is_naive();
        // The `:no-decomp` rule option (and the global `--no-decomp`
        // flag) skips tree-decomposition in query planning, forcing
        // the single-bag fast path.
        let no_decomp = self.no_decomp || rule.no_decomp;
        let requires_read_context = !seminaive
            || matches!(
                rule.eval_mode,
                RuleEvalMode::Naive | RuleEvalMode::UnsafeSeminaive
            );

        let rule_id = {
            let mut rb = self.backend.new_rule(&rule.name, seminaive);
            rb.set_no_decomp(no_decomp);
            let mut translator =
                BackendRule::new(rb, &self.functions, &self.type_info, requires_read_context);
            translator.rule = Some(rule.name.clone());
            translator.query(query, rule.include_subsumed)?;
            translator.actions(actions)?;
            translator.build()
        };

        if let Some(rules) = self.rulesets.get_mut(&rule.ruleset) {
            match rules {
                Ruleset::Rules(rules, _) => {
                    match rules.entry(rule.name.clone()) {
                        indexmap::map::Entry::Occupied(_) => {
                            return Err(Error::RuleAlreadyExists(rule.name, rule.span));
                        }
                        indexmap::map::Entry::Vacant(e) => e.insert(CompiledRule {
                            core: core_rule,
                            backend_id: rule_id,
                            seminaive,
                            requires_read_context,
                            no_decomp,
                            include_subsumed: rule.include_subsumed,
                        }),
                    };
                    Ok(rule.name)
                }
                Ruleset::Combined(..) => Err(Error::CombinedRulesetError(rule.ruleset, rule.span)),
            }
        } else {
            Err(Error::NoSuchRuleset(rule.ruleset, rule.span))
        }
    }

    /// Compile one table-free action block with request-owned equality locals.
    /// `execute = false` performs the identical lowering without invoking any
    /// action, fresh allocation, or capture. Callers should validate against a
    /// disposable compiler clone, as with `validate_program`.
    ///
    /// Captures completed before a runtime failure are returned alongside the
    /// error. This uses the same single rule/recovery boundary as `eval_actions`;
    /// it emits neither a schedule output nor an extra user-visible run report.
    /// Encoded modes must use their ordinary action/proof lowering instead.
    pub fn eval_scoped_actions(
        &mut self,
        inputs: &[(String, String, Value)],
        execute: bool,
        compile: impl FnOnce(&mut ScopedActionBuilder<'_>) -> Result<(), Error>,
    ) -> (Result<(), Error>, Vec<(String, Value)>) {
        self.eval_scoped_actions_observed(inputs, execute, compile, &mut |_| {})
    }

    /// Execute the same scoped action block, observing real committed progress
    /// on success or error. Compilation alone emits no execution event. The
    /// existing callback, capture and recovery lifetimes remain unchanged.
    pub fn eval_scoped_actions_observed(
        &mut self,
        inputs: &[(String, String, Value)],
        execute: bool,
        compile: impl FnOnce(&mut ScopedActionBuilder<'_>) -> Result<(), Error>,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> (Result<(), Error>, Vec<(String, Value)>) {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut changed = false;
        let mut compiled = false;
        let result = (|| {
            if self.proof_state.original_typechecking.is_some() {
                return Err(Error::BackendError(
                    "scoped equality actions require normal execution mode".into(),
                ));
            }
            let mut bindings = IndexMap::default();
            let mut entries = HashMap::default();
            for (name, sort, value) in inputs {
                let sort = self
                    .type_info
                    .sorts
                    .get(sort)
                    .filter(|sort| sort.is_eq_sort())
                    .ok_or_else(|| Error::BackendError(format!("unknown equality sort {sort}")))?
                    .clone();
                if bindings
                    .insert(name.clone(), (Span::Panic, sort.clone()))
                    .is_some()
                {
                    return Err(Error::BackendError(format!("local {name} already bound")));
                }
                let value = if execute {
                    self.backend.get_canon_repr(*value, ColumnTy::Id)
                } else {
                    *value
                };
                entries.insert(
                    ResolvedAtomTerm::Var(
                        Span::Panic,
                        ResolvedVar {
                            name: name.clone(),
                            sort,
                            is_global_ref: false,
                        },
                    ),
                    QueryEntry::Const {
                        val: value,
                        ty: ColumnTy::Id,
                    },
                );
            }
            let functions = &self.functions;
            let type_info = &self.type_info;
            let symbol_gen = &mut self.parser.symbol_gen;
            let (result, committed) = self.backend.observe_actions(|backend| {
                backend.run_scoped_rule("scoped_actions", execute, Some(type_info), |rule| {
                    let mut translator = BackendRule::new(rule, functions, type_info, true);
                    translator.entries = entries;
                    let mut builder = ScopedActionBuilder {
                        translator,
                        symbol_gen,
                        bindings,
                        captured: captured.clone(),
                    };
                    compile(&mut builder)?;
                    compiled = true;
                    Ok(builder.translator.build())
                })
            });
            changed = committed;
            result.map_err(|error| Error::from_backend(error.as_ref()))
        })();
        let mut captured = std::mem::take(&mut *captured.lock().unwrap());
        for (_, value, column) in &mut captured {
            if matches!(column, ColumnTy::Id) {
                *value = self.backend.get_canon_repr(*value, ColumnTy::Id);
            }
        }
        let captured = captured
            .into_iter()
            .map(|(name, value, _)| (name, value))
            .collect();
        if execute && compiled {
            observer(ExecutionEvent::ActionsFinished {
                origin: ExecutionOrigin::Requested,
                changed,
                result: result.as_ref().map(|_| ()),
            });
        }
        (result, captured)
    }

    fn eval_actions(
        &mut self,
        actions: &ResolvedActions,
        execute: bool,
        origin: ExecutionOrigin,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> Result<(), Error> {
        let mut binding = IndexSet::default();
        let mut ctx = CoreActionContext::new(
            &self.type_info,
            &mut binding,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        );
        let (actions, _) = actions.to_core_actions(&mut ctx)?;

        let functions = &self.functions;
        let type_info = &self.type_info;
        let mut compiled = false;
        let (result, changed) = self.backend.observe_actions(|backend| {
            backend.run_scoped_rule("eval_actions", execute, Some(type_info), |rule| {
                // Global actions use Read/Full contexts (may read the DB).
                let mut translator = BackendRule::new(rule, functions, type_info, true);
                translator.actions(&actions)?;
                let id = translator.build();
                compiled = true;
                Ok(id)
            })
        });
        let result = result.map_err(|error| {
            if compiled {
                Error::from_backend(error.as_ref())
            } else {
                // Translation errors used to return directly before running
                // the backend. RAII must not erase their variant or span.
                error
                    .downcast::<Error>()
                    .unwrap_or_else(|error| Error::from_backend(error.as_ref()))
            }
        });
        if execute && compiled {
            observer(ExecutionEvent::ActionsFinished {
                origin,
                changed,
                result: result.as_ref().map(|_| ()),
            });
        }
        result
    }

    /// Get the list of all functions in the e-graph.
    pub fn get_function_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }

    /// Iterate over every `(name, function)` pair registered in the
    /// e-graph, in registration order.
    pub fn functions_iter(&self) -> impl Iterator<Item = (&String, &Function)> {
        self.functions.iter()
    }

    /// Run `f` against a raw execution state, with this e-graph's declarations
    /// visible to any primitive it reaches.
    ///
    /// Every execution egglog starts goes through here or its `_tracked` twin,
    /// so no call site has to decide whether a primitive it cannot see might
    /// want to resolve a signature.
    fn with_execution_state<R>(&self, f: impl FnOnce(&mut ExecutionState<'_>) -> R) -> R {
        self.backend.with_execution_state(Some(&self.type_info), f)
    }

    /// [`EGraph::with_execution_state`], also reporting whether `f` staged a
    /// mutation.
    fn with_execution_state_tracked<R>(
        &self,
        f: impl FnOnce(&mut ExecutionState<'_>) -> R,
    ) -> (R, bool) {
        self.backend
            .with_execution_state_tracked(Some(&self.type_info), f)
    }

    /// Run a read-only closure against the e-graph. The closure receives
    /// a [`ReadState`], so it can read but not write. Because this
    /// borrows `&self`, the closure and its callbacks may also call other
    /// `&self` methods such as [`EGraph::value_to_base`].
    pub fn read<R>(&self, f: impl FnOnce(ReadState<'_, '_>) -> R) -> R {
        let registry = self.backend.action_registry().clone();
        let guard = registry.read().unwrap();
        self.with_execution_state_tracked(|es| f(ReadState::wrap(es, &guard, Context::Read)))
            .0
    }

    /// Call `f` on each [`FunctionEntry`] of a function table. Top-level
    /// form of [`Read::function_entries`]; errors if `name` is a
    /// constructor or unregistered.
    pub fn function_entries(
        &self,
        name: &str,
        f: impl FnMut(FunctionEntry<'_>),
    ) -> Result<(), Error> {
        self.read(|rs| rs.function_entries(name, f))
    }

    /// Like [`EGraph::function_entries`], but stops when `f` returns `false`.
    pub fn function_entries_while(
        &self,
        name: &str,
        f: impl FnMut(FunctionEntry<'_>) -> bool,
    ) -> Result<(), Error> {
        self.read(|rs| rs.function_entries_while(name, f))
    }

    /// Call `f` on each [`Enode`] of a constructor / relation table.
    /// Top-level form of [`Read::constructor_enodes`]; errors if `name`
    /// is a function or unregistered.
    pub fn constructor_enodes(&self, name: &str, f: impl FnMut(Enode<'_>)) -> Result<(), Error> {
        self.read(|rs| rs.constructor_enodes(name, f))
    }

    /// Like [`EGraph::constructor_enodes`], but stops when `f` returns `false`.
    pub fn constructor_enodes_while(
        &self,
        name: &str,
        f: impl FnMut(Enode<'_>) -> bool,
    ) -> Result<(), Error> {
        self.read(|rs| rs.constructor_enodes_while(name, f))
    }

    /// Remove every row from the named function in bulk.
    ///
    /// This is intended as a faster alternative to issuing a `(delete …)` for
    /// every row of the function: it drops the backing row storage in
    /// O(1)-in-row-count time, rather than O(n) per-row teardown. Any pending
    /// staged inserts/removes for this function are dropped as part of the
    /// clear, so callers that have staged updates they want to land first
    /// should arrange for those to be flushed beforehand.
    ///
    /// Cached indexes and subsets that reference this table are invalidated by
    /// a generation bump and are lazily rebuilt against the now-empty table on
    /// next access.
    ///
    /// Raises an error if the function does not exist.
    pub fn clear_function(&mut self, func_name: &str) -> Result<(), Error> {
        let backend_id = self
            .functions
            .get(func_name)
            .ok_or_else(|| TypeError::UnboundFunction(func_name.to_string(), span!()))?
            .backend_id;
        self.backend.clear_table(backend_id);
        Ok(())
    }

    /// Evaluates an expression, returns the sort of the expression and the evaluation result.
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<(ArcSort, Value), Error> {
        let span = expr.span();
        let command = Command::Action(Action::Expr(span.clone(), expr.clone()));
        let resolved = self.resolve_command(
            command,
            CommandPurpose::Authored,
            &mut |_| {},
            &mut |_, _| None,
        )?;
        if self.are_proofs_enabled() {
            self.proof_check_program
                .extend(resolved.desugared_before_proofs);
        }
        let resolved_commands = resolved.desugared;

        if resolved_commands.len() != 1 {
            return Err(Error::BackendError(
                "eval_expr expects a single resolved command".to_string(),
            ));
        }
        let Some(OriginatedCommand {
            command: resolved_command,
            ..
        }) = resolved_commands.into_iter().next()
        else {
            return Err(Error::BackendError(
                "eval_expr expects a single resolved command".to_string(),
            ));
        };
        let resolved_expr = match resolved_command {
            ResolvedNCommand::CoreAction(ResolvedAction::Expr(_, resolved_expr)) => resolved_expr,
            cmd => {
                return Err(Error::BackendError(format!(
                    "eval_expr: unexpected resolved command: {cmd:?}"
                )));
            }
        };
        let sort = resolved_expr.output_type();
        let value = self.eval_resolved_expr(span, &resolved_expr)?;
        Ok((sort, value))
    }

    /// Evaluate a closed expression with an expected output sort.
    ///
    /// The expected sort participates in the ordinary native overload solver,
    /// including output-only primitives and function-constructor capture types.
    /// Evaluation uses the same checked-expression lowering as [`Self::eval_expr`].
    /// Encoded modes require their ordinary command/proof instrumentation instead.
    pub fn eval_expr_with_output(
        &mut self,
        expr: &Expr,
        output_sort: ArcSort,
    ) -> Result<(ArcSort, Value), Error> {
        if self.proof_state.original_typechecking.is_some() {
            return Err(Error::BackendError(
                "expected-output expression evaluation requires normal execution mode".into(),
            ));
        }
        let resolved =
            self.typecheck_expr_with_bindings_and_output(expr, &[], output_sort, Context::Full)?;
        let sort = resolved.output_type();
        let value = self.eval_resolved_expr(expr.span(), &resolved)?;
        Ok((sort, value))
    }

    /// Typecheck an expression under explicit local bindings, an expected
    /// output sort, and a primitive call context.
    ///
    /// `bindings` contains the local variables in scope while resolving `expr`.
    /// Each tuple is `(name, span, sort)`, where `span` is used for diagnostics
    /// tied to that binding. `output_sort` constrains overload resolution and
    /// output-type inference for the expression. Global references are rewritten
    /// into the same zero-argument function calls used during command execution.
    /// `context` should match the runtime context where the resolved expression
    /// will be evaluated.
    pub fn typecheck_expr_with_bindings_and_output(
        &mut self,
        expr: &Expr,
        bindings: &[(String, Span, ArcSort)],
        output_sort: ArcSort,
        context: Context,
    ) -> Result<ResolvedExpr, TypeError> {
        let mut binding_map = IndexMap::default();
        binding_map.reserve(bindings.len());
        for (name, span, sort) in bindings {
            if binding_map
                .insert(name.as_str(), (span.clone(), sort.clone()))
                .is_some()
            {
                return Err(TypeError::AlreadyDefined(name.clone(), span.clone()));
            }
        }
        let resolved = self.type_info.typecheck_expr_with_output(
            &mut self.parser.symbol_gen,
            expr,
            &binding_map,
            output_sort,
            context,
        )?;
        Ok(remove_globals::remove_globals_expr(resolved))
    }

    /// Replace literal `(unstable-fn "...")` targets with hidden evaluator bindings.
    ///
    /// The returned expression must be evaluated with the returned bindings in
    /// scope before any caller-supplied local bindings. This lets direct
    /// execution-state evaluation use the same hidden `ResolvedFunction` value
    /// that backend rule lowering injects for `unstable-fn`.
    ///
    /// For example, a resolved body like `(unstable-app (unstable-fn "f") _0)`
    /// cannot evaluate the string literal `"f"` directly. This helper replaces
    /// the `(unstable-fn "f")` sub-expression with a fresh hidden variable and
    /// returns a binding from that variable to the prepared function value.
    pub fn prepare_unstable_fn_targets_for_eval(
        &mut self,
        expr: &ResolvedExpr,
    ) -> Result<(ResolvedExpr, Vec<(String, Value)>), Error> {
        let mut bindings = Vec::new();
        let expr = self.prepare_unstable_fn_targets_for_eval_inner(expr, &mut bindings)?;
        Ok((expr, bindings))
    }

    fn prepare_unstable_fn_targets_for_eval_inner(
        &mut self,
        expr: &ResolvedExpr,
        bindings: &mut Vec<(String, Value)>,
    ) -> Result<ResolvedExpr, Error> {
        match expr {
            ResolvedExpr::Lit(..) | ResolvedExpr::Var(..) => Ok(expr.clone()),
            ResolvedExpr::Call(span, resolved_call, children) => {
                if let ResolvedCall::Primitive(prim) = resolved_call
                    && prim.name() == "unstable-fn"
                {
                    let Some(ResolvedExpr::Lit(target_span, Literal::String(name))) =
                        children.first()
                    else {
                        return Err(Error::BackendError(format!(
                            "{}\nunstable-fn requires a literal string function name",
                            children
                                .first()
                                .map(ResolvedExpr::span)
                                .unwrap_or_else(|| Span::Panic)
                        )));
                    };
                    let panic_id = self.backend.new_panic(format!(
                        "unstable-fn over `{name}` was applied in a context where its wrapped \
                         function is not valid for this call site, if in a rule, add :naive."
                    ));
                    let resolved_function = resolve_function_container_target_with_context(
                        &self.backend,
                        &self.functions,
                        &self.type_info,
                        name,
                        prim,
                        panic_id,
                        // Top-level action evaluation runs in the `Full` context.
                        crate::Context::Full,
                        target_span,
                    )?;
                    let fn_value = self.backend.base_values().get(resolved_function);
                    let binding_name = self.parser.symbol_gen.fresh("unstable_fn_target");
                    bindings.push((binding_name.clone(), fn_value));
                    let mut prepared_children = Vec::with_capacity(children.len());
                    prepared_children.push(ResolvedExpr::Var(
                        target_span.clone(),
                        ResolvedVar {
                            name: binding_name,
                            sort: children[0].output_type(),
                            is_global_ref: false,
                        },
                    ));
                    for child in &children[1..] {
                        prepared_children.push(
                            self.prepare_unstable_fn_targets_for_eval_inner(child, bindings)?,
                        );
                    }
                    return Ok(ResolvedExpr::Call(
                        span.clone(),
                        resolved_call.clone(),
                        prepared_children,
                    ));
                }

                let prepared_children = children
                    .iter()
                    .map(|child| self.prepare_unstable_fn_targets_for_eval_inner(child, bindings))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(ResolvedExpr::Call(
                    span.clone(),
                    resolved_call.clone(),
                    prepared_children,
                ))
            }
        }
    }

    fn eval_resolved_expr(&mut self, span: Span, expr: &ResolvedExpr) -> Result<Value, Error> {
        let unit_id = self.backend.base_values().get_ty::<()>();
        let unit_val = self.backend.base_values().get(());

        let result: egglog_bridge::SideChannel<Value> = Default::default();
        let result_ref = result.clone();
        let ext_id = self
            .backend
            .register_external_func(Box::new(make_external_func(move |_es, vals| {
                debug_assert!(vals.len() == 1);
                *result_ref.lock().unwrap() = Some(vals[0]);
                Some(unit_val)
            })));

        let mut translator = BackendRule::new(
            self.backend.new_rule("eval_resolved_expr", false),
            &self.functions,
            &self.type_info,
            true, // global action: Read/Full contexts (may read the DB)
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
        let mut binding = IndexSet::default();
        let mut ctx = CoreActionContext::new(
            &self.type_info,
            &mut binding,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        );
        let actions = actions.to_core_actions(&mut ctx)?.0;
        translator.actions(&actions)?;

        let arg = translator.entry(&ResolvedAtomTerm::Var(span.clone(), result_var));
        translator.rb.call_external_func(
            ext_id,
            &[arg],
            egglog_bridge::ColumnTy::Base(unit_id),
            || "this function will never panic".to_string(),
        );

        let id = translator.build();
        let rule_result = self.backend.run_rules(&[id], Some(&self.type_info));
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);
        let _ = rule_result.map_err(|e| {
            Error::BackendError(format!("Failed to evaluate expression '{expr}': {e}"))
        })?;

        let result = result.lock().unwrap().unwrap();
        Ok(result)
    }

    fn add_combined_ruleset(
        &mut self,
        name: String,
        rulesets: Vec<String>,
        origin: ExecutionOrigin,
    ) {
        match self.rulesets.entry(name.clone()) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Combined(rulesets, origin)),
        };
    }

    fn add_ruleset(&mut self, name: String, origin: ExecutionOrigin) {
        match self.rulesets.entry(name.clone()) {
            Entry::Occupied(_) => panic!("Ruleset '{name}' was already present"),
            Entry::Vacant(e) => e.insert(Ruleset::Rules(Default::default(), origin)),
        };
    }

    fn check_facts(
        &mut self,
        span: &Span,
        facts: &[ResolvedFact],
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> Result<(), Error> {
        let fresh_name = self.parser.symbol_gen.fresh("check_facts");
        let fresh_ruleset = self.parser.symbol_gen.fresh("check_facts_ruleset");
        let rule = ast::ResolvedRule {
            span: span.clone(),
            head: ResolvedActions::default(),
            body: facts.to_vec(),
            name: fresh_name.clone(),
            ruleset: fresh_ruleset.clone(),
            eval_mode: RuleEvalMode::default(),
            no_decomp: false,
            include_subsumed: false,
        };
        let core_rule = rule.to_canonicalized_core_rule(
            &self.type_info,
            &mut self.parser.symbol_gen,
            self.proof_state.original_typechecking.is_none(),
        )?;
        let query = core_rule.body;

        let ext_sc = egglog_bridge::SideChannel::default();
        let ext_sc_ref = ext_sc.clone();
        let ext_id = self
            .backend
            .register_external_func(Box::new(make_external_func(move |exec_state, _| {
                *ext_sc_ref.lock().unwrap() = Some(());
                exec_state.trigger_early_stop();
                Some(Value::new_const(0))
            })));

        let mut translator = BackendRule::new(
            self.backend.new_rule("check_facts", false),
            &self.functions,
            &self.type_info,
            true, // global query: Read context (may read the DB)
        );
        translator.query(&query, true)?;
        translator
            .rb
            .call_external_func(ext_id, &[], egglog_bridge::ColumnTy::Id, || {
                "this function will never panic".to_string()
            });
        let id = translator.build();
        observer(ExecutionEvent::RunStarted {
            origin: ExecutionOrigin::Internal,
            schedule: None,
        });
        let run_result = self.backend.run_rules(&[id], Some(&self.type_info));
        self.backend.free_rule(id);
        self.backend.free_external_func(ext_id);
        let run_result = run_result
            .map(|report| self.iteration_result(&fresh_ruleset, report))
            .map_err(|error| Error::from_backend(error.as_ref()));
        observer(ExecutionEvent::RunFinished {
            origin: ExecutionOrigin::Internal,
            schedule: None,
            result: run_result.as_ref(),
        });
        run_result?;

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

    fn run_command(
        &mut self,
        command: ResolvedNCommand,
        origin: ExecutionOrigin,
        provenance: Option<&ScheduleProvenance>,
        policies: CommandPolicies<'_>,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> Result<Vec<CommandOutput>, Error> {
        match command {
            // Sorts are already declared during typechecking
            ResolvedNCommand::Sort {
                name,
                uf,
                proof_func,
                proof_constructors,
                ..
            } => {
                // Restore the sort's UF tables into proof_state: the constructor
                // for extraction's `find_canonical`, the index for container
                // rebuild's leader lookups.
                if let Some((uf_ctor, uf_index)) = uf {
                    self.proof_state.uf_parent.insert(name.clone(), uf_ctor);
                    if let Some(uf_index) = uf_index {
                        self.proof_state.uf_function.insert(name.clone(), uf_index);
                    }
                }
                // If the sort has a :internal-proof-func field, store the mapping for proof lookup.
                // This annotation is set by proof instrumentation and consumed here.
                if let Some(proof_func_name) = proof_func {
                    self.proof_state
                        .proof_func_parent
                        .insert(name.clone(), proof_func_name);
                }
                // The Proof sort's :internal-proof-names records the global proof
                // constructors; restore them so container rebuild can recover them.
                if let Some(pc) = proof_constructors {
                    let names = &mut self.proof_state.proof_names;
                    names.proof_datatype = name.clone();
                    names.congr_constructor = pc.congr;
                    names.eq_trans_constructor = pc.trans;
                    names.eq_sym_constructor = pc.sym;
                    names.container_normalize_constructor = pc.normalize;
                }
                log::info!("Declared sort {name}.")
            }
            ResolvedNCommand::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                log::info!("Declared {} {}.", fdecl.subtype, fdecl.name)
            }
            ResolvedNCommand::AddRuleset(_span, name) => {
                self.add_ruleset(name.clone(), origin);
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::UnstableCombinedRuleset(_span, name, others) => {
                self.add_combined_ruleset(name.clone(), others, origin);
                log::info!("Declared ruleset {name}.");
            }
            ResolvedNCommand::NormRule { rule } => {
                let name = rule.name.clone();
                self.add_rule(rule)?;
                log::info!("Declared rule {name}.")
            }
            ResolvedNCommand::RunSchedule(sched) => {
                let report = self.run_schedule(
                    &sched,
                    origin,
                    &mut ScheduleObservation {
                        provenance,
                        repeat_policy: policies.repeat,
                        scheduler_selector: policies.scheduler,
                        identity: provenance.is_none() && origin == ExecutionOrigin::Requested,
                        path: vec![],
                        iterations: vec![],
                    },
                    observer,
                )?;
                log::info!("Ran schedule {sched}.");
                log::info!("Report: {report}");
                if policies.accumulate_schedule && self.collect_reports {
                    self.overall_run_report.union(report.clone());
                }
                return Ok(vec![CommandOutput::RunSchedule(report)]);
            }
            ResolvedNCommand::PrintOverallStatistics(span, file) => match file {
                None => {
                    log::info!("Printed overall statistics");
                    return Ok(vec![CommandOutput::OverallStatistics(
                        self.overall_run_report.clone(),
                    )]);
                }
                Some(path) => {
                    let mut file = std::fs::File::create(&path)
                        .map_err(|e| Error::IoError(path.clone().into(), e, span.clone()))?;
                    log::info!("Printed overall statistics to json file {path}");

                    serde_json::to_writer(&mut file, &self.overall_run_report).map_err(|e| {
                        Error::BackendError(format!("failed writing statistics: {e}"))
                    })?;
                }
            },
            ResolvedNCommand::Check(span, facts) => {
                self.check_facts(&span, &facts, observer)?;
                log::info!("Checked fact {facts:?}.");
            }
            ResolvedNCommand::ActionScope(_, actions) => {
                self.eval_actions(&actions, true, origin, observer)?;
            }
            ResolvedNCommand::CoreAction(action) => match &action {
                ResolvedAction::Let(_, name, contents) => {
                    panic!("Globals should have been desugared away: {name} = {contents}")
                }
                _ => {
                    self.eval_actions(
                        &ResolvedActions::new(vec![action.clone()]),
                        true,
                        origin,
                        observer,
                    )?;
                }
            },
            ResolvedNCommand::Extract(span, expr, variants) => {
                let sort = expr.output_type();

                let x = self.eval_resolved_expr(span.clone(), &expr)?;
                let n = self.eval_resolved_expr(span, &variants)?;
                let n: i64 = self.backend.base_values().unwrap(n);

                return if n == 0 {
                    let (termdag, term, cost) = self.extract_value(&sort, x)?;
                    // dont turn termdag into a string if we have messages disabled for performance reasons
                    if log_enabled!(Level::Info) {
                        log::info!("extracted with cost {cost}: {}", termdag.to_string(term));
                    }
                    Ok(vec![CommandOutput::ExtractBest(termdag, cost, term)])
                } else {
                    if n < 0 {
                        return Err(Error::ExtractError(
                            "cannot extract a negative number of variants".to_string(),
                        ));
                    }
                    let extracted = self.extract_variants(vec![(sort, x)], n as usize)?;
                    let terms: Vec<TermId> = extracted
                        .variants
                        .into_iter()
                        .next()
                        .expect("extract_variants returns one result for one root")
                        .into_iter()
                        .map(|extracted| extracted.term)
                        .collect();
                    if log_enabled!(Level::Info) {
                        let expr_str = expr.to_string();
                        log::info!("extracted {} variants for {expr_str}", terms.len());
                    }
                    Ok(vec![CommandOutput::ExtractVariants(
                        extracted.termdag,
                        terms,
                    )])
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
                        let path: PathBuf = file.into();
                        match std::fs::File::create(&path) {
                            Ok(f) => Ok((f, path)),
                            Err(e) => Err(Error::IoError(path, e, span.clone())),
                        }
                    })
                    .transpose()?;
                return self
                    .print_function(&f, n, file, span.clone(), mode)
                    .map_err(|e| match e {
                        Error::TypeError(TypeError::UnboundFunction(f, _)) => {
                            Error::TypeError(TypeError::UnboundFunction(f, span.clone()))
                        }
                        // This case is currently impossible
                        _ => e,
                    })
                    .map(|opt| opt.into_iter().collect());
            }
            ResolvedNCommand::PrintSize(span, f) => {
                let res = self.print_size(f.as_deref()).map_err(|e| match e {
                    Error::TypeError(TypeError::UnboundFunction(f, _)) => {
                        Error::TypeError(TypeError::UnboundFunction(f, span.clone()))
                    }
                    // This case is currently impossible
                    _ => e,
                })?;
                return Ok(vec![res]);
            }
            ResolvedNCommand::Fail(span, c) => {
                let result = self.run_command(*c, origin, provenance, policies, observer);
                expect_failure(span, result)?;
            }
            ResolvedNCommand::Input { span, name, file } => {
                self.input_file(&name, file, span)?;
            }
            ResolvedNCommand::Output { span, file, exprs } => {
                let mut filename = self.fact_directory.clone().unwrap_or_default();
                filename.push(file.as_str());

                // Preserve the command's file-error ordering: reject an invalid
                // output path before evaluating expressions.
                use std::io::Write;
                let mut f = File::options()
                    .append(true)
                    .create(true)
                    .open(&filename)
                    .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;

                // Evaluation may mutate the e-graph, so finish it before
                // extraction takes a shared borrow for its prepared costs.
                let roots = exprs
                    .iter()
                    .map(|expr| {
                        let value = self.eval_resolved_expr(span.clone(), expr)?;
                        Ok((expr.output_type(), value))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let extracted = self.extract_best(roots)?;
                for root in extracted.terms {
                    let root = root.ok_or_else(|| {
                        Error::ExtractError(
                            "Unable to find any valid extraction (likely due to subsume or delete)"
                                .to_string(),
                        )
                    })?;
                    writeln!(f, "{}", extracted.termdag.to_string(root.term))
                        .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
                }

                log::info!("Output to '{filename:?}'.")
            }
            ResolvedNCommand::UserDefined(_span, name, exprs) => {
                let command = self
                    .commands
                    .get(&name)
                    .ok_or_else(|| {
                        NotFoundError(format!("Unrecognized user-defined command: {name}"))
                    })?
                    .clone();
                return command.update(self, &exprs);
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
                return Ok(vec![CommandOutput::ProveExists {
                    proof_store,
                    proof_id,
                }]);
            }
        };

        Ok(vec![])
    }

    fn input_file(&mut self, func_name: &str, file: String, span: Span) -> Result<(), Error> {
        let function_type = self.type_info.get_func_type(func_name).ok_or_else(|| {
            Error::TypeError(TypeError::UnboundFunction(
                func_name.to_string(),
                span.clone(),
            ))
        })?;
        let func = self.functions.get_mut(func_name).unwrap();

        let parsed_contents =
            parse_input_file(&func.func_type, &file, self.fact_directory.as_deref(), span)?
                .into_iter()
                .map(|row| {
                    row.iter()
                        .map(|literal| literal_to_value(&self.backend, literal))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

        let unit_val = self.backend.base_values().get(());
        let num_facts = parsed_contents.len();

        let table_action = egglog_bridge::TableAction::new(&self.backend, func.backend_id);

        if function_type.subtype != FunctionSubtype::Constructor {
            self.with_execution_state(|es| {
                for row in parsed_contents.iter() {
                    table_action.insert(es, row.iter().copied());
                }
                Some(unit_val)
            });
        } else {
            self.with_execution_state(|es| {
                for row in parsed_contents.iter() {
                    // Constructor semantics: mint a fresh eclass id for
                    // each missing key.
                    table_action.lookup_or_insert(es, row);
                }
                Some(unit_val)
            });
        }

        self.backend.flush_updates();

        log::info!("Read {num_facts} facts into {func_name} from '{file}'.");
        Ok(())
    }

    /// Returns true if proofs are enabled.
    pub fn are_proofs_enabled(&self) -> bool {
        self.proof_state.proofs_enabled
    }

    /// Desugars and typechecks one already macro-expanded command, preserving globals.
    ///
    /// This updates the authoring type environment but executes no actions.
    /// In ordinary mode, native name checks run on a discarded global-lowered
    /// copy; the returned commands retain globals and are not proof-instrumented.
    /// Frontends can export resolved sorts and global-reference annotations before
    /// lowering passes erase them. Built-in sugar (including relation/rewrite
    /// origins) is already eliminated; callers needing that provenance must
    /// retain it while resolving the command. Use a separate graph for export.
    pub fn resolve_command_preserving_globals(
        &mut self,
        command: Command,
    ) -> Result<Vec<ResolvedNCommand>, Error> {
        let typechecked: Vec<_> = self
            .desugar_and_typecheck_command(command, CommandPurpose::Authored, &mut |_, _| None)?
            .into_iter()
            .map(|(command, _)| command)
            .collect();
        if self.proof_state.original_typechecking.is_none() {
            let lowered = remove_globals::remove_globals(
                typechecked.clone(),
                &mut self.parser.symbol_gen.clone(),
            );
            for command in &lowered {
                self.names.check_shadowing(command)?;
            }
        }
        Ok(typechecked)
    }

    /// Shared compiler phase before global removal and proof instrumentation.
    fn desugar_and_typecheck_command(
        &mut self,
        command: Command,
        purpose: CommandPurpose,
        select_sort: &mut SortSelector<'_>,
    ) -> Result<Vec<(ResolvedNCommand, ExecutionOrigin)>, Error> {
        let desugared = desugar_command(
            command,
            &mut self.parser,
            self.proof_state.proof_testing && purpose == CommandPurpose::Authored,
        )?;
        let typechecking = self.proof_state.original_typechecking.as_deref_mut();
        Ok(match typechecking {
            Some(original) => original.typecheck_program(&desugared, select_sort)?,
            None => self.typecheck_program(&desugared, select_sort)?,
        })
    }

    /// Resolves one already macro-expanded command without executing it.
    ///
    /// This updates declaration/type information and removes globals, but does
    /// not instrument proof terms or execute actions. Adapters can call it on a
    /// clone to validate a complete decoded program before mutating the live
    /// graph. User-defined commands retain their own runtime validation.
    pub fn resolve_command_before_proofs(
        &mut self,
        command: Command,
    ) -> Result<Vec<ResolvedNCommand>, Error> {
        Ok(self
            .resolve_command_before_proofs_with_origins(
                command,
                CommandPurpose::Authored,
                &mut |_| {},
                &mut |_, _| None,
            )?
            .into_iter()
            .map(|(command, _)| command)
            .collect())
    }

    fn resolve_command_before_proofs_with_origins(
        &mut self,
        command: Command,
        purpose: CommandPurpose,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
        select_sort: &mut SortSelector<'_>,
    ) -> Result<Vec<(ResolvedNCommand, ExecutionOrigin)>, Error> {
        let typechecked = self.desugar_and_typecheck_command(command, purpose, select_sort)?;
        observer(ExecutionEvent::CommandResolved {
            commands: &typechecked,
        });
        if let Some(original_typechecking) = self.proof_state.original_typechecking.as_ref() {
            for (command, _) in &typechecked {
                if let Err(reason) = command_supports_proof_encoding(
                    &command.to_command(),
                    &original_typechecking.type_info,
                ) {
                    let command_text = format!("{}", command.to_command());
                    return Err(Error::UnsupportedProofCommand {
                        command: command_text,
                        reason,
                    });
                }
            }

            Ok(typechecked
                .into_iter()
                .flat_map(|(command, origin)| {
                    proof_form(vec![command], &mut self.parser.symbol_gen)
                        .into_iter()
                        .map(move |command| (command, origin))
                })
                .collect())
        } else {
            let typechecked: Vec<_> = typechecked
                .into_iter()
                .flat_map(|(command, origin)| {
                    remove_globals::remove_globals(vec![command], &mut self.parser.symbol_gen)
                        .into_iter()
                        .map(move |command| (command, origin))
                })
                .collect();
            for (command, _) in &typechecked {
                self.names.check_shadowing(command)?;
            }
            Ok(typechecked)
        }
    }

    /// Desugars, typechecks, and removes globals from a single [`Command`].
    /// Leverages previous type information in the [`EGraph`] to do so, adding new type information.
    /// When will_run is true, adds to `desugared_commands_run_so_far`, which is used for proof checking.
    fn resolve_command(
        &mut self,
        command: Command,
        purpose: CommandPurpose,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
        select_sort: &mut SortSelector<'_>,
    ) -> Result<ResolvedNCommands, Error> {
        let resolved_before_proofs = self.resolve_command_before_proofs_with_origins(
            command,
            purpose,
            observer,
            select_sort,
        )?;

        // Add term encoding when it is enabled
        if self.proof_state.original_typechecking.is_none() {
            Ok(ResolvedNCommands {
                desugared: resolved_before_proofs
                    .into_iter()
                    .map(|(command, origin)| OriginatedCommand {
                        command,
                        origin,
                        schedule: None,
                        command_tail: false,
                    })
                    .collect(),
                desugared_before_proofs: vec![],
            })
        } else {
            // An applied-sort alias has already selected the original
            // environment's registered representative. Bind the runtime
            // environment's own instance, then keep this actual Sort/Fail
            // anchor without re-declaring or instrumenting that instance.
            if let [(command, origin)] = resolved_before_proofs.as_slice() {
                let mut anchor = command;
                while let ResolvedNCommand::Fail(_, inner) = anchor {
                    anchor = inner;
                }
                if let ResolvedNCommand::Sort { name, span, .. } = anchor
                    && let Some(target) = self.authoring_type_info().get_sort_by_name(name)
                    && target.name() != name
                {
                    let target = target.name().to_owned();
                    self.complete_prepared_sort_alias(name, &target, span.clone(), command)?;
                    return Ok(ResolvedNCommands {
                        desugared: vec![OriginatedCommand {
                            command: command.clone(),
                            origin: *origin,
                            schedule: None,
                            command_tail: false,
                        }],
                        desugared_before_proofs: vec![command.clone()],
                    });
                }
            }
            // Now remove globals for actual execution (but NOT from desugared_commands)
            let typechecked_no_globals: Vec<_> = resolved_before_proofs
                .iter()
                .flat_map(|(command, origin)| {
                    proof_global_remover::remove_globals(
                        vec![command.clone()],
                        &mut self.parser.symbol_gen,
                    )
                    .into_iter()
                    .map(move |command| (command, *origin))
                })
                .collect();
            for (command, _) in &typechecked_no_globals {
                self.names.check_shadowing(command)?;
            }

            let term_encoding_added =
                ProofInstrumentor::add_term_encoding(self, typechecked_no_globals);
            let new_typechecked =
                self.resolve_instrumented_commands(term_encoding_added, purpose)?;
            Ok(ResolvedNCommands {
                desugared: new_typechecked,
                desugared_before_proofs: resolved_before_proofs
                    .into_iter()
                    .map(|(command, _)| command)
                    .collect(),
            })
        }
    }

    // Finish the existing physical compiler pipeline for instrumentation-owned
    // commands, preserving structural tail tags through desugaring.
    fn resolve_instrumented_commands(
        &mut self,
        commands: Vec<OriginatedCommand<Command>>,
        purpose: CommandPurpose,
    ) -> Result<Vec<OriginatedCommand<ResolvedNCommand>>, Error> {
        let mut new_typechecked = vec![];
        for OriginatedCommand {
            command: new_cmd,
            origin,
            schedule,
            command_tail,
        } in commands
        {
            let mut desugared = desugar_command(
                new_cmd,
                &mut self.parser,
                self.proof_state.proof_testing && purpose == CommandPurpose::Authored,
            )?;
            for (cmd, child_origin) in &mut desugared {
                if origin == ExecutionOrigin::Internal {
                    *child_origin = ExecutionOrigin::Internal;
                }
                log::trace!("Desugared term encoding: {}", cmd.to_command());
            }

            // Now typecheck using self, adding term type information.
            let desugared_typechecked = self.typecheck_program(&desugared, &mut |_, _| None)?;
            // remove globals again, but this time allow primitive globals
            let desugared_typechecked =
                desugared_typechecked
                    .into_iter()
                    .flat_map(|(command, origin)| {
                        remove_globals::remove_globals(vec![command], &mut self.parser.symbol_gen)
                            .into_iter()
                            .map(move |command| (command, origin))
                    });

            let mut schedule = schedule;
            let desugared_typechecked: Vec<_> = desugared_typechecked.collect();
            let last = desugared_typechecked.len().checked_sub(1);
            for (i, (command, origin)) in desugared_typechecked.into_iter().enumerate() {
                let mapping = if Some(i) == last {
                    schedule.take()
                } else {
                    None
                };
                if let Some(mapping) = &mapping {
                    mapping.validate(&command)?;
                }
                new_typechecked.push(OriginatedCommand {
                    command,
                    origin,
                    schedule: mapping,
                    command_tail,
                });
            }
            if schedule.is_some() {
                return Err(Error::BackendError(
                    "schedule provenance lost during desugaring".into(),
                ));
            }
        }
        Ok(new_typechecked)
    }

    /// Compile a resolved command without running actions, queries or schedules.
    /// Definitions use the same registration and BackendRule compiler as execution.
    fn compile_command(
        &mut self,
        command: ResolvedNCommand,
        origin: ExecutionOrigin,
    ) -> Result<(), Error> {
        match command {
            command @ (ResolvedNCommand::Sort { .. }
            | ResolvedNCommand::Function(_)
            | ResolvedNCommand::AddRuleset(..)
            | ResolvedNCommand::UnstableCombinedRuleset(..)
            | ResolvedNCommand::NormRule { .. }
            | ResolvedNCommand::Push(_)
            | ResolvedNCommand::Pop(..)) => {
                self.run_command(
                    command,
                    origin,
                    None,
                    CommandPolicies::default(),
                    &mut |_| {},
                )?;
            }
            ResolvedNCommand::ActionScope(_, actions) => {
                self.eval_actions(&actions, false, origin, &mut |_| {})?;
            }
            ResolvedNCommand::CoreAction(action) => {
                self.eval_actions(
                    &ResolvedActions::singleton(action),
                    false,
                    origin,
                    &mut |_| {},
                )?;
            }
            ResolvedNCommand::Check(span, facts) if origin == ExecutionOrigin::Internal => {
                let rule = ResolvedRule {
                    span,
                    head: ResolvedActions::default(),
                    body: facts,
                    name: String::new(),
                    ruleset: String::new(),
                    eval_mode: RuleEvalMode::default(),
                    no_decomp: false,
                    include_subsumed: true,
                }
                .to_canonicalized_core_rule(
                    &self.type_info,
                    &mut self.parser.symbol_gen,
                    self.proof_state.original_typechecking.is_none(),
                )?;
                let mut translator = BackendRule::new(
                    self.backend.new_rule("validate_query", false),
                    &self.functions,
                    &self.type_info,
                    true,
                );
                translator.query(&rule.body, true)?;
                let id = translator.build();
                self.backend.free_rule(id);
            }
            ResolvedNCommand::RunSchedule(schedule) => {
                let mut schedules = vec![schedule];
                while let Some(schedule) = schedules.pop() {
                    match schedule {
                        ResolvedSchedule::Run(span, config) => {
                            if !self.rulesets.contains_key(&config.ruleset) {
                                return Err(Error::NoSuchRuleset(config.ruleset, span));
                            }
                            if let Some(facts) = config.until {
                                self.compile_command(
                                    ResolvedNCommand::Check(span, facts),
                                    ExecutionOrigin::Internal,
                                )?;
                            }
                        }
                        ResolvedSchedule::Repeat(_, _, schedule)
                        | ResolvedSchedule::Saturate(_, schedule) => schedules.push(*schedule),
                        ResolvedSchedule::Sequence(_, children) => {
                            schedules.extend(children.into_iter().rev())
                        }
                    }
                }
            }
            // Runtime success/failure is deliberately unknown during validation.
            ResolvedNCommand::Fail(_, command) => self.compile_command(*command, origin)?,
            ResolvedNCommand::UserDefined(_, name, args) => {
                self.prepare_user_defined_command(&name, &args)?;
            }
            _ => {}
        }
        Ok(())
    }

    // Shared execution, completion and failure boundary for compiled commands.
    fn execute_originated(
        &mut self,
        command: &OriginatedCommand<ResolvedNCommand>,
        policies: CommandPolicies<'_>,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> Result<Vec<CommandOutput>, Error> {
        let result = self.run_command(
            command.command.clone(),
            command.origin,
            command.schedule.as_ref(),
            policies,
            observer,
        );
        match &result {
            Ok(_) => observer(ExecutionEvent::CommandCompleted {
                origin: command.origin,
                command: &command.command,
            }),
            Err(error) => observer(ExecutionEvent::CommandFailed {
                origin: command.origin,
                command: &command.command,
                error,
            }),
        }
        result
    }

    /// Expand one command against this compiler's selected authoring types,
    /// advancing its own symbol generator without cloning the type environment.
    pub fn expand_command_macros(&mut self, command: Command) -> Result<Vec<Command>, Error> {
        let macro_type_info = self
            .proof_state
            .original_typechecking
            .as_ref()
            .map(|graph| &graph.type_info)
            .unwrap_or(&self.type_info);
        self.command_macros
            .apply(command, &mut self.parser.symbol_gen, macro_type_info)
    }

    /// Process a program through the shared native compiler/interpreter.
    #[allow(clippy::too_many_arguments)]
    fn process_program_internal(
        &mut self,
        program: Vec<Command>,
        mode: ProcessingMode,
        outputs: &mut dyn FnMut(usize, ExecutionOrigin, CommandOutput),
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
        repeat_policy: &dyn Fn(usize, ScheduleLocation<'_>) -> RepeatContinuation,
        scheduler_selector: &dyn Fn(usize, ScheduleLocation<'_>) -> Option<scheduler::SchedulerId>,
        select_sort: &mut SortSelector<'_>,
    ) -> Result<ResolvedNCommands, Error> {
        let executing = matches!(
            mode,
            ProcessingMode::Execute | ProcessingMode::ExecuteEmbedded
        );
        let embedded = mode == ProcessingMode::ExecuteEmbedded;
        let mut desugared_before_proofs = Vec::new();
        let mut desugared = Vec::new();

        for (cause, before_expanded_command) in program.into_iter().enumerate() {
            // First do user-provided macro expansion for this command,
            // which may rely on type information from previous commands.
            let macro_expanded = self.expand_command_macros(before_expanded_command)?;

            for command in macro_expanded {
                // handle include specially- we keep them as-is for desugaring
                if let Command::Include(span, file) = &command {
                    let s = std::fs::read_to_string(file)
                        .map_err(|e| Error::IoError(file.clone().into(), e, span.clone()))?;
                    let included_program = self
                        .parser
                        .get_program_from_string(Some(file.clone()), &s)?;
                    // run program internal on these include commands
                    let resolved = self.process_program_internal(
                        included_program,
                        mode,
                        &mut |_, origin, output| outputs(cause, origin, output),
                        &mut |_, event| observer(cause, event),
                        &|_, site| repeat_policy(cause, site),
                        &|_, site| scheduler_selector(cause, site),
                        select_sort,
                    )?;
                    desugared.extend(resolved.desugared);
                    desugared_before_proofs.extend(resolved.desugared_before_proofs);
                } else {
                    let resolved = self.resolve_command(
                        command,
                        CommandPurpose::Authored,
                        &mut |event| observer(cause, event),
                        select_sort,
                    )?;
                    if executing && self.are_proofs_enabled() {
                        self.proof_check_program
                            .extend(resolved.desugared_before_proofs.clone());
                    }

                    desugared_before_proofs.extend(resolved.desugared_before_proofs);
                    desugared.extend(resolved.desugared.clone());
                    for (processed_index, processed) in resolved.desugared.iter().enumerate() {
                        if mode == ProcessingMode::Compile {
                            self.compile_command(processed.command.clone(), processed.origin)?;
                            continue;
                        }
                        if embedded && processed.command_tail {
                            continue;
                        }
                        // even in desugar mode we still run push and pop
                        if executing
                            || matches!(
                                processed.command,
                                ResolvedNCommand::Push(_) | ResolvedNCommand::Pop(_, _)
                            )
                        {
                            let mut anchor = &processed.command;
                            while let ResolvedNCommand::Fail(_, child) = anchor {
                                anchor = child;
                            }
                            let scoped = processed.origin == ExecutionOrigin::Requested
                                && matches!(anchor, ResolvedNCommand::ActionScope(..));
                            let mut scope_started = false;
                            let result = self.execute_originated(
                                processed,
                                CommandPolicies {
                                    repeat: &|site| repeat_policy(cause, site),
                                    scheduler: &|site| scheduler_selector(cause, site),
                                    accumulate_schedule: !embedded
                                        || processed.origin != ExecutionOrigin::Requested,
                                },
                                &mut |event| {
                                    scope_started |= scoped
                                        && matches!(
                                            event,
                                            ExecutionEvent::ActionsFinished {
                                                origin: ExecutionOrigin::Requested,
                                                ..
                                            }
                                        );
                                    observer(cause, event);
                                },
                            );
                            let result = match result {
                                Ok(result) => result,
                                Err(mut error) => {
                                    // The VM and native recovery have already
                                    // retained the executed scope prefix. Its
                                    // encoded UF/view indices still need the
                                    // existing internal maintenance tail.
                                    if scope_started {
                                        for tail in resolved
                                            .desugared
                                            .iter()
                                            .skip(processed_index + 1)
                                            .filter(|tail| tail.command_tail)
                                        {
                                            match self.execute_originated(
                                                tail,
                                                CommandPolicies {
                                                    repeat: &|site| repeat_policy(cause, site),
                                                    scheduler: &|site| {
                                                        scheduler_selector(cause, site)
                                                    },
                                                    accumulate_schedule: !embedded
                                                        || tail.origin
                                                            != ExecutionOrigin::Requested,
                                                },
                                                &mut |event| observer(cause, event),
                                            ) {
                                                Ok(result) => {
                                                    for output in result {
                                                        outputs(cause, tail.origin, output);
                                                    }
                                                }
                                                Err(recovery) => {
                                                    let message = format!(
                                                        "{error}; encoded recovery after the failed action scope also failed: {recovery}"
                                                    );
                                                    error = match error {
                                                        Error::Panic { span, rule, .. } => {
                                                            Error::Panic {
                                                                message,
                                                                span,
                                                                rule,
                                                            }
                                                        }
                                                        _ => Error::BackendError(message),
                                                    };
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    return Err(error);
                                }
                            };
                            for output in result {
                                outputs(cause, processed.origin, output);
                            }
                        }
                    }
                }
            }
        }

        Ok(ResolvedNCommands {
            desugared_before_proofs,
            desugared,
        })
    }

    /// Run a program, represented as an AST.
    /// Return a list of messages.
    ///
    /// Egglog errors, including `(panic ...)`, are not transactional: effects
    /// completed before an error remain. After a rule-action error, a successful
    /// recovery rebuild leaves the database canonical and reusable. Rust panics
    /// in extension code unwind normally instead of becoming [`enum@Error`] values.
    /// Outputs completed before an error are discarded; use
    /// [`Self::run_program_with_outputs`] to retain them.
    pub fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<CommandOutput>, Error> {
        let mut outputs = Vec::new();
        self.run_program_with_outputs(program, &mut outputs)?;
        Ok(outputs)
    }

    /// Run a program, appending outputs in native completion order.
    ///
    /// Each completed native command's outputs are appended to `outputs`, also
    /// through macro expansion, desugaring and nested includes. Existing entries
    /// are preserved. On error, previously appended outputs and completed effects
    /// remain; the failing command and later commands add no successful output.
    /// Proof helper and maintenance outputs are included without filtering.
    /// Error recovery and extension panics behave as in [`Self::run_program`].
    pub fn run_program_with_outputs(
        &mut self,
        program: Vec<Command>,
        outputs: &mut Vec<CommandOutput>,
    ) -> Result<(), Error> {
        self.process_program_internal(
            program,
            ProcessingMode::Execute,
            &mut |_, _, output| outputs.push(output),
            &mut |_, _| {},
            &|_, _| RepeatContinuation::CanStop,
            &|_, _| None,
            &mut |_, _| None,
        )?;
        Ok(())
    }

    /// Run through the native interpreter with explicit output and run provenance.
    ///
    /// Both callbacks' indices identify the submitted command, including work
    /// generated by macros, desugaring, proof instrumentation and nested includes.
    /// Requested outputs are published before trailing maintenance. `Fail` still
    /// suppresses its inner outputs, while its actual runs remain observable.
    /// Completion callbacks expose registration progress even if a later helper
    /// fails. Nothing is retained in the graph; callers should aggregate reports
    /// for their request. Error recovery and extension panics are unchanged.
    pub fn run_program_observed(
        &mut self,
        program: Vec<Command>,
        outputs: &mut dyn FnMut(usize, ExecutionOrigin, CommandOutput),
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
    ) -> Result<(), Error> {
        self.run_program_observed_with_repeat_policy(program, outputs, observer, &|_, _| {
            RepeatContinuation::CanStop
        })
    }

    /// Execute with call-local applied-sort representative selection. Providers
    /// prepare each sort once without mutation; selection precedes native type,
    /// backend, primitive and proof-helper installation. Returning a registered
    /// name requests a structurally compatible alias, validated by the compiler.
    /// The policy is never stored and ordinary execution/Fail boundaries remain.
    pub fn run_program_observed_with_sort_selection(
        &mut self,
        program: Vec<Command>,
        outputs: &mut dyn FnMut(usize, ExecutionOrigin, CommandOutput),
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
        select_sort: &mut SortSelector<'_>,
    ) -> Result<(), Error> {
        self.process_program_internal(
            program,
            ProcessingMode::Execute,
            outputs,
            observer,
            &|_, _| RepeatContinuation::CanStop,
            &|_, _| None,
            select_sort,
        )?;
        Ok(())
    }

    /// Run with a call-local continuation policy for bounded requested loops.
    ///
    /// The callback receives the submitted command index and its original
    /// schedule location, before proof instrumentation. It is queried once on
    /// entry to each requested Repeat. Internal/helper loops never consult it.
    /// Exact counts do not change failure, output, reporting or maintenance
    /// boundaries. Native callers keep early stopping through
    /// [`Self::run_program_observed`]. No policy is retained in the e-graph.
    pub fn run_program_observed_with_repeat_policy(
        &mut self,
        program: Vec<Command>,
        outputs: &mut dyn FnMut(usize, ExecutionOrigin, CommandOutput),
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
        repeat_policy: &dyn Fn(usize, ScheduleLocation<'_>) -> RepeatContinuation,
    ) -> Result<(), Error> {
        self.process_program_internal(
            program,
            ProcessingMode::Execute,
            outputs,
            observer,
            repeat_policy,
            &|_, _| None,
            &mut |_, _| None,
        )?;
        Ok(())
    }

    /// Run with call-local requested-loop and requested-Run policies.
    ///
    /// The scheduler selector receives original submitted-command/schedule
    /// coordinates, and is called immediately before an actual requested rule
    /// step (after any native until pre-check). Returning an existing scheduler
    /// id chooses `step_rules_with_scheduler`; `None` keeps ordinary stepping.
    /// Compiler/proof-maintenance runs never consult it. Policies are not stored;
    /// native observation, output, proof and recovery boundaries are unchanged.
    pub fn run_program_observed_with_policies(
        &mut self,
        program: Vec<Command>,
        outputs: &mut dyn FnMut(usize, ExecutionOrigin, CommandOutput),
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
        repeat_policy: &dyn Fn(usize, ScheduleLocation<'_>) -> RepeatContinuation,
        scheduler_selector: &dyn Fn(usize, ScheduleLocation<'_>) -> Option<scheduler::SchedulerId>,
    ) -> Result<(), Error> {
        self.process_program_internal(
            program,
            ProcessingMode::Execute,
            outputs,
            observer,
            repeat_policy,
            scheduler_selector,
            &mut |_, _| None,
        )?;
        Ok(())
    }

    /// Execute actual native commands inside a caller-owned schedule anchor.
    ///
    /// Uses the ordinary compiler, proof history, interpreter and observations.
    /// Only structurally tagged common postludes of requested commands are
    /// omitted: inline schedule rebuilding and prerequisite helper work remain.
    /// The caller must surround the whole schedule with [`Self::with_schedule_envelope`].
    pub fn run_program_embedded_observed(
        &mut self,
        program: Vec<Command>,
        outputs: &mut dyn FnMut(usize, ExecutionOrigin, CommandOutput),
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
        scheduler_selector: &dyn Fn(usize, ScheduleLocation<'_>) -> Option<scheduler::SchedulerId>,
    ) -> Result<(), Error> {
        self.process_program_internal(
            program,
            ProcessingMode::ExecuteEmbedded,
            outputs,
            observer,
            &|_, _| RepeatContinuation::CanStop,
            scheduler_selector,
            &mut |_, _| None,
        )?;
        Ok(())
    }

    /// Surround one opaque schedule callback with native command maintenance.
    ///
    /// Setup and the common postlude use the same builders and executor as
    /// ordinary native commands. No synthetic schedule or proof AST is recorded;
    /// embedded commands record their own actual pre-proof history. `fail_spans`
    /// lists native Fail wrappers outermost first. Only the callback result is
    /// caught, never setup/postlude errors or Rust panics. Completion is called
    /// before the postlude; `None` denotes a caught/suppressed anchor.
    pub fn with_schedule_envelope<C>(
        &mut self,
        fail_spans: &[Span],
        context: &mut C,
        anchor: impl FnOnce(&mut Self, &mut C) -> Result<RunReport, Error>,
        completed: impl FnOnce(&mut C, Option<&RunReport>),
        mut observer: impl FnMut(&mut C, ExecutionEvent<'_>),
    ) -> Result<(), Error> {
        let (before, after) = if self.proof_state.original_typechecking.is_some() {
            let (before, after) = ProofInstrumentor::schedule_envelope(self);
            (
                self.resolve_instrumented_commands(before, CommandPurpose::Authored)?,
                self.resolve_instrumented_commands(after, CommandPurpose::Authored)?,
            )
        } else {
            (vec![], vec![])
        };
        for command in &before {
            self.execute_originated(command, CommandPolicies::default(), &mut |event| {
                observer(context, event)
            })?;
        }
        let result = anchor(self, context);
        // Native RunSchedule publishes its aggregate only when the complete
        // anchor succeeds, before Fail may turn that success into ExpectFail.
        if let Ok(report) = &result
            && self.collect_reports
        {
            self.overall_run_report.union(report.clone());
        }
        let mut result = result.map(Some);
        for span in fail_spans.iter().rev() {
            result = expect_failure(span.clone(), result).map(|()| None);
        }
        if fail_spans.is_empty() {
            let result = result?;
            completed(context, result.as_ref());
        } else {
            // Suppression/caught inner locators end at the wrapped anchor,
            // including an ExpectFail raised by one of the wrappers.
            completed(context, None);
            result?;
        }
        for command in &after {
            self.execute_originated(command, CommandPolicies::default(), &mut |event| {
                observer(context, event)
            })?;
        }
        Ok(())
    }

    /// Validate an ordinary internal condition without executing it.
    ///
    /// Unlike an authored Check in proof-testing mode, a condition never expands
    /// into Prove/ProveExists. Planning and actual witness failure remain at the
    /// execution boundary, just as for an ordinary Check.
    pub fn validate_condition(&self, span: Span, facts: Vec<Fact>) -> Result<(), Error> {
        self.clone().compile_condition(span, facts)
    }

    /// Compile an internal condition in this disposable compiler context.
    ///
    /// This shares [`Self::validate_condition`]'s admission and proof behavior,
    /// but retains compiler metadata in place instead of cloning the graph.
    /// It does not execute the condition or invoke primitive bodies/validators.
    pub fn compile_condition(&mut self, span: Span, facts: Vec<Fact>) -> Result<(), Error> {
        let resolved = self.resolve_command(
            Command::Check(span, facts),
            CommandPurpose::Condition,
            &mut |_| {},
            &mut |_, _| None,
        )?;
        for command in resolved.desugared {
            self.compile_command(command.command, command.origin)?;
        }
        Ok(())
    }

    /// Execute a non-output-bearing ordinary condition within a schedule.
    ///
    /// The existing typechecker, proof fact instrumentation and `check_facts`
    /// interpreter are used, including actual internal query observations and
    /// cleanup. Only `Error::CheckError` means that the query did not match;
    /// callers must propagate all other errors. No proof-testing proof request
    /// or common command postlude is introduced for this internal pre-check.
    pub fn check_condition_observed(
        &mut self,
        span: Span,
        facts: Vec<Fact>,
        observer: &mut dyn FnMut(ExecutionEvent<'_>),
    ) -> Result<(), Error> {
        let resolved = self.resolve_command(
            Command::Check(span, facts),
            CommandPurpose::Condition,
            observer,
            &mut |_, _| None,
        )?;
        if self.are_proofs_enabled() {
            self.proof_check_program
                .extend(resolved.desugared_before_proofs);
        }
        for command in &resolved.desugared {
            if !command.command_tail {
                self.execute_originated(command, CommandPolicies::default(), observer)?;
            }
        }
        Ok(())
    }

    /// Validate a program on a disposable compiler clone, then discard it.
    ///
    /// This registers declarations and fully compiles rules (including native
    /// groundedness checks), actions and internal queries. Requested Checks are
    /// typechecked, but their query planning remains at the execution boundary,
    /// where native Fail can catch it. Proof/term expansion and internal helper
    /// validation are unchanged. It executes no actions, runs, primitive bodies,
    /// proof validators, maintenance or user-defined runtime commands. Extension
    /// declarations may opt into [`UserDefinedCommand::prepare`].
    /// No backend IDs or speculative definitions are transferred to this graph.
    /// Returns the disposable compiler clone for further compile-only validation.
    /// Runtime errors such as a missing witness are left to actual execution.
    pub fn validate_program(&self, program: Vec<Command>) -> Result<Self, Error> {
        let mut compiler = self.clone();
        compiler.compile_program_observed(program, &mut |_, _| {})?;
        Ok(compiler)
    }

    /// Register and compile a program in this compiler context, without running
    /// authored actions, schedules, checks or primitive bodies. Unlike
    /// [`Self::validate_program`], this does not clone the graph: successful
    /// registrations and partial compiler metadata remain in place on error.
    /// Observers receive original [`ExecutionEvent::CommandResolved`] batches,
    /// but no execution/completion events. `Fail` validates its inner command;
    /// it does not assert an expected execution error in this compile-only mode.
    pub fn compile_program_observed(
        &mut self,
        program: Vec<Command>,
        observer: &mut dyn FnMut(usize, ExecutionEvent<'_>),
    ) -> Result<(), Error> {
        self.process_program_internal(
            program,
            ProcessingMode::Compile,
            &mut |_, _, _| {},
            observer,
            &|_, _| RepeatContinuation::CanStop,
            &|_, _| None,
            &mut |_, _| None,
        )?;
        Ok(())
    }

    /// Prepares an explicitly opted-in extension declaration without invoking
    /// its runtime command. Unregistered and execution-only commands reject.
    pub fn prepare_user_defined_command(&mut self, name: &str, args: &[Expr]) -> Result<(), Error> {
        let command =
            self.commands.get(name).cloned().ok_or_else(|| {
                NotFoundError(format!("Unrecognized user-defined command: {name}"))
            })?;
        command.prepare(self, args)
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
        let res = self.process_program_internal(
            parsed,
            ProcessingMode::Resolve,
            &mut |_, _, _| {},
            &mut |_, _| {},
            &|_, _| RepeatContinuation::CanStop,
            &|_, _| None,
            &mut |_, _| None,
        )?;
        Ok(res
            .desugared
            .into_iter()
            .map(|c| c.command.to_command())
            .collect())
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

    /// Native typed class inventory, including empty and unreachable classes.
    /// Read only after native maintenance has settled. Encoded-mode helper sorts
    /// retain their native names; logical projection is the caller's responsibility.
    pub fn materialized_eq_classes(&self) -> std::collections::BTreeSet<(Arc<str>, Value)> {
        self.backend.materialized_eq_classes()
    }

    /// Allocate an empty native equality class without introducing a table row.
    pub fn fresh_eq_class(&mut self, sort: &str) -> Result<Value, Error> {
        let resolved = self
            .get_sort_by_name(sort)
            .ok_or_else(|| Error::BackendError(format!("unknown equality sort {sort}")))?;
        if !resolved.is_eq_sort() {
            return Err(Error::BackendError(format!(
                "{sort} is not an equality sort"
            )));
        }
        Ok(self.backend.fresh_eq_class(Arc::from(resolved.name())))
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

    /// Returns the unique sort whose runtime values have Rust type `T`.
    pub fn get_arcsort_for_value_type<T: 'static>(&self) -> ArcSort {
        self.type_info.get_arcsort_for_value_type::<T>()
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
    /// This method assumes `x` belongs to sort `T`.
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
        self.with_execution_state(|state| {
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

    /// The total number of rows across all non-hidden functions, including
    /// global `let` bindings.
    pub fn total_size(&self) -> usize {
        self.functions
            .values()
            .filter(|f| !f.is_hidden())
            .map(|f| self.backend.table_size(f.backend_id))
            .sum()
    }

    /// The number of e-nodes in the e-graph: the total number of rows across
    /// visible constructor tables for unionable eq-sorts, including their
    /// term/proof-encoding views. Analysis functions, global lets, hidden
    /// helpers, and `relation`s are excluded.
    pub fn num_nodes(&self) -> usize {
        self.functions
            .values()
            .filter(|f| {
                // Encoded views all carry `term_constructor`. A constructor's
                // hidden term table has only its original inputs, while an
                // analysis function's also stores its output, so only a
                // constructor view is one column wider than its term table.
                let is_constructor_view = f.term_constructor().is_some_and(|name| {
                    self.functions.get(name).is_some_and(|term_table| {
                        term_table.func_type().input.len() + 1 == f.func_type().input.len()
                    })
                });
                !f.is_hidden()
                    && !f.is_let_binding()
                    && (f.func_type().subtype == FunctionSubtype::Constructor
                        || is_constructor_view)
                    && self.type_info.is_sort_unionable(f.extraction_output_sort())
            })
            .map(|f| self.backend.table_size(f.backend_id))
            .sum()
    }

    /// Get a function by name.
    ///
    /// Returns `None` if the function does not exist.
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    /// Returns the typed child values stored inside a container-sort value.
    ///
    /// `sort` must be a container sort and `value` must belong to that sort.
    pub fn container_inner_values(&self, sort: &ArcSort, value: Value) -> Vec<(ArcSort, Value)> {
        sort.inner_values(self.backend.container_values(), value)
    }

    /// Reconstructs a base-sort value into a [`TermDag`].
    ///
    /// `sort` must be a base sort and `value` must belong to that sort in this
    /// e-graph.
    pub fn reconstruct_base_value(
        &self,
        sort: &ArcSort,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        sort.reconstruct_termdag_base(self.backend.base_values(), value, termdag)
    }

    /// Reconstructs a container-sort value from its extracted element terms.
    ///
    /// `sort` and `value` must identify a container in this e-graph, and
    /// `element_terms` must correspond to [`EGraph::container_inner_values`] in
    /// the same order.
    pub fn reconstruct_container_value(
        &self,
        sort: &ArcSort,
        value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        sort.reconstruct_termdag_container(
            self.backend.container_values(),
            value,
            termdag,
            element_terms,
        )
    }

    /// Returns `true` if a user-defined command with the given name is
    /// registered in this e-graph.
    pub fn has_command(&self, name: &str) -> bool {
        self.commands.contains_key(name)
    }

    /// Invoke a registered user-defined command by name, passing the given
    /// unresolved expression arguments.
    ///
    /// This is equivalent to writing `(name args...)` at the top level, but
    /// callable directly from Rust.  Returns an error if no command with the given
    /// name is registered.
    pub fn run_user_defined_command(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<Vec<CommandOutput>, Error> {
        self.run_command(
            ResolvedNCommand::UserDefined(span!(), name.to_string(), args.to_vec()),
            ExecutionOrigin::Requested,
            None,
            CommandPolicies::default(),
            &mut |_| {},
        )
    }

    /// Set the report verbosity level for rule execution output.
    pub fn set_report_level(&mut self, level: ReportLevel) {
        self.collect_reports = level != ReportLevel::Off;
        self.backend.set_report_level(level);
    }

    /// A basic method for dumping the state of the database to `log::info!`.
    ///
    /// For large tables, this is unlikely to give particularly useful output.
    pub fn dump_debug_info(&self) {
        self.backend.dump_debug_info();
    }

    /// Run `f` with a [`FullState`] handle on this EGraph's database
    /// — the same handle a `:naive` rule's `add_rust_rule_full`
    /// callback receives. Use to drive name-indexed reads / writes
    /// (`fs.set`, `fs.add`, `fs.lookup`, `fs.eclass_of`,
    /// `fs.contains`, `fs.remove`, …) from outside a rule.
    ///
    /// # Flush semantics
    ///
    /// Pending writes flush once, **after** `f` returns. Two
    /// consequences:
    ///
    /// 1. A `set` / `add` / `remove` inside the closure is *not*
    ///    visible to a subsequent `lookup` / `contains` / `eclass_of`
    ///    in the **same** closure. Split write-then-read into separate
    ///    `update` calls.
    /// 2. Conversely, batching multiple writes in one closure is the
    ///    fast path — only one flush + rebuild happens, regardless of
    ///    how many writes occurred.
    /// 3. A closure that only reads (e.g. `lookup`, `constructor_enodes`)
    ///    stages nothing, so the flush is skipped entirely — a read
    ///    costs no more than a direct backend scan.
    ///
    /// # Example
    /// ```
    /// use egglog::prelude::*;
    /// let mut eg = EGraph::default();
    /// eg.parse_and_run_program(None, "(function f (i64) i64 :no-merge)")?;
    /// eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    /// let got = eg.update(|fs| fs.lookup("f", 1_i64))?;
    /// let got: Option<i64> = got.map(|v| eg.value_to_base::<i64>(v));
    /// assert_eq!(got, Some(42));
    /// # Ok::<(), egglog::Error>(())
    /// ```
    pub fn update<R>(
        &mut self,
        f: impl FnOnce(FullState<'_, '_>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        if self.are_proofs_enabled() {
            return Err(Error::ProofsIncompatibleApi {
                api: "EGraph::update",
                reason: "writes inside the closure bypass the proof-encoding pipeline,\n\
                         so any rule derivations resting on them would be unverifiable.",
            });
        }
        self.update_unchecked(f)
    }

    /// Internal version of [`EGraph::update`] without the proofs
    /// check. Used by proof-system internals that need to read the
    /// e-graph while the proof system itself is enabled.
    pub(crate) fn update_unchecked<R>(
        &mut self,
        f: impl FnOnce(FullState<'_, '_>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let registry = self.backend.action_registry().clone();
        let guard = registry.read().unwrap();
        let (result, changed) =
            self.with_execution_state_tracked(|es| f(FullState::wrap(es, &guard, Context::Full)));
        drop(guard);
        // A read-only closure stages nothing, so `flush_updates` would only do
        // a no-op merge plus a spurious timestamp bump and rebuild check. Skip
        // it unless the closure actually wrote, keeping reads as cheap as a
        // direct backend scan.
        if changed {
            self.backend.flush_updates();
        }
        result
    }

    /// Run a pattern query: bind the variables in `vars` against
    /// `facts` and return one [`HashMap`] per match, keyed by variable
    /// name. Values stay raw — convert via [`EGraph::value_to_base`].
    ///
    /// With zero vars, returns at most one empty map (so `.len()` is 1
    /// if the body matched, 0 if it didn't).
    pub fn query(
        &mut self,
        vars: &[(&str, ArcSort)],
        facts: ast::Facts<String, String>,
    ) -> Result<Vec<HashMap<String, Value>>, Error> {
        // Fail fast under proofs — otherwise the failure would
        // surface through `rust_rule`'s check below with a misleading
        // api: "rust_rule" in the error message.
        if self.are_proofs_enabled() {
            return Err(Error::ProofsIncompatibleApi {
                api: "EGraph::query",
                reason: "the underlying rust_rule callback has no proof-encoding validator,\n\
                         so query matches cannot be verified.",
            });
        }
        use std::sync::{Arc, Mutex};
        let names: Arc<[String]> = vars.iter().map(|(n, _)| (*n).to_owned()).collect();
        let results: Arc<Mutex<Vec<HashMap<String, Value>>>> = Arc::new(Mutex::new(Vec::new()));
        let results_weak = Arc::downgrade(&results);
        let names_for_cb = names.clone();

        let ruleset = self.parser.symbol_gen.fresh("query_ruleset");
        prelude::add_ruleset(self, &ruleset)?;
        // From here on, we OWN the ruleset and the rule and have to
        // clean them up on every exit path. Run the rest in a closure
        // and tear down before propagating.
        let outcome = (|| -> Result<_, Error> {
            prelude::rust_rule(self, "query", &ruleset, vars, facts, move |_, values| {
                let arc = results_weak.upgrade().unwrap();
                let mut results = arc.lock().unwrap();
                let map: HashMap<String, Value> = names_for_cb
                    .iter()
                    .zip(values.iter().copied())
                    .map(|(n, v)| (n.clone(), v))
                    .collect();
                results.push(map);
                Some(())
            })?;
            prelude::run_ruleset(self, &ruleset)?;
            Ok(())
        })();

        // Tear the temporary rule + ruleset down whether the body
        // succeeded or not.
        if let Some(Ruleset::Rules(rules, _)) = self.rulesets.swap_remove(&ruleset) {
            for (_, rule) in rules {
                self.backend.free_rule(rule.backend_id);
            }
        }
        outcome?;

        let Some(mutex) = Arc::into_inner(results) else {
            panic!("`results_weak` outlived the callback");
        };
        Ok(mutex.into_inner().unwrap())
    }
}

pub use crate::api::{ApiError, FromValue, FromValues, IntoValue, IntoValues, RawValues};

/// Build the runtime value backing a resolved `(unstable-fn name)` target.
///
/// For table-backed functions, this captures the table action that
/// `unstable-app` will call later. For primitive targets, this bakes one
/// dispatch id per runtime context so application can choose the entrypoint
/// matching the primitive body's current call-site context.
#[allow(clippy::too_many_arguments)]
fn resolve_function_container_target_with_context(
    backend: &egglog_bridge::EGraph,
    functions: &IndexMap<String, Function>,
    type_info: &TypeInfo,
    name: &str,
    primitive: &core::SpecializedPrimitive,
    panic_id: ExternalFunctionId,
    ctx: crate::Context,
    span: &Span,
) -> Result<ResolvedFunction, Error> {
    let Some(target_function) = type_info
        .get_sorts::<FunctionSort>()
        .into_iter()
        .find(|function| function.name() == primitive.output().name())
    else {
        return Err(Error::BackendError(format!(
            "`unstable-fn` output sort `{}` is not a function sort",
            primitive.output().name()
        )));
    };

    let partial_arcsorts: Vec<_> = primitive.input().iter().skip(1).cloned().collect();
    let remaining_inputs = target_function.inputs();
    let output = target_function.output();

    let id = if let Some(func) = functions.get(name) {
        let func_type = type_info.get_func_type(name).ok_or_else(|| {
            Error::BackendError(format!(
                "`unstable-fn` references `{name}`, which has no resolved type"
            ))
        })?;
        let expected_inputs = partial_arcsorts
            .iter()
            .chain(remaining_inputs)
            .collect::<Vec<_>>();
        let inputs_match = func_type.input.len() == expected_inputs.len()
            && func_type
                .input
                .iter()
                .zip(&expected_inputs)
                .all(|(actual, expected)| actual.name() == expected.name());
        if !inputs_match || func_type.output.name() != output.name() {
            let expected_input_names = expected_inputs
                .iter()
                .map(|sort| sort.name())
                .collect::<Vec<_>>()
                .join(", ");
            let actual_input_names = func_type
                .input
                .iter()
                .map(|sort| sort.name())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::BackendError(format!(
                "`unstable-fn` reference `{name}` expected ({}) -> {}, found ({}) -> {}",
                expected_input_names,
                output.name(),
                actual_input_names,
                func_type.output.name(),
            )));
        }

        let action = egglog_bridge::TableAction::new(backend, func.backend_id);
        match func_type.subtype {
            ast::FunctionSubtype::Constructor => ResolvedFunctionId::Constructor(action),
            ast::FunctionSubtype::Custom => ResolvedFunctionId::Function(action),
        }
    } else if let Some(primitives) = type_info.get_prims(name) {
        let signature: Vec<_> = partial_arcsorts
            .iter()
            .chain(remaining_inputs)
            .chain(once(&output))
            .cloned()
            .collect();
        let candidates: Vec<_> = primitives
            .iter()
            .filter(|primitive| primitive.accept(&signature, type_info))
            .collect();
        let mut ambiguous_ctx = None;
        let context_ids = enum_map::EnumMap::from_fn(|runtime_ctx| {
            let mut ids = candidates
                .iter()
                .filter_map(|primitive| primitive.context_ids[runtime_ctx]);
            // The first `next` finds the candidate for this runtime context;
            // the second detects whether there is more than one such candidate.
            match (ids.next(), ids.next()) {
                (None, _) => None,
                (Some(id), None) => Some(id),
                (Some(_), Some(_)) => {
                    ambiguous_ctx = Some(runtime_ctx);
                    None
                }
            }
        });
        if let Some(runtime_ctx) = ambiguous_ctx {
            return Err(TypeError::AmbiguousPrimitive {
                name: name.to_owned(),
                ctx: runtime_ctx,
                span: span.clone(),
            }
            .into());
        }
        if !context_ids.iter().any(|(_, id)| id.is_some()) {
            return Err(TypeError::UnresolvedPrimitive {
                name: name.to_owned(),
                ctx,
                span: span.clone(),
            }
            .into());
        }
        ResolvedFunctionId::Primitive { context_ids }
    } else {
        return Err(TypeError::UnresolvedPrimitive {
            name: name.to_owned(),
            ctx,
            span: span.clone(),
        }
        .into());
    };

    Ok(ResolvedFunction {
        id,
        partial_arcsorts,
        name: name.to_owned(),
        panic_id,
    })
}

struct BackendRule<'a> {
    rb: egglog_bridge::RuleBuilder<'a>,
    entries: HashMap<core::ResolvedAtomTerm, QueryEntry>,
    functions: &'a IndexMap<String, Function>,
    type_info: &'a TypeInfo,
    /// Whether primitives may read the database. When true the per-phase
    /// [`crate::Context`] widens from `Pure`/`Write` to `Read`/`Full` (query
    /// gains reads, action gains reads on top of writes). True for `:naive` /
    /// `:unsafe-seminaive` rules and a non-seminaive EGraph.
    requires_read_context: bool,
    // Exact native identity for compiled rules; temporary helpers have none.
    rule: Option<String>,
}

impl<'a> BackendRule<'a> {
    fn new(
        rb: egglog_bridge::RuleBuilder<'a>,
        functions: &'a IndexMap<String, Function>,
        type_info: &'a TypeInfo,
        requires_read_context: bool,
    ) -> BackendRule<'a> {
        BackendRule {
            rb,
            functions,
            type_info,
            requires_read_context,
            rule: None,
            entries: Default::default(),
        }
    }

    /// The [`crate::Context`] that applies when compiling
    /// primitives on the query side (LHS) of this rule. Under
    /// seminaive evaluation, queries are pure (no DB reads or
    /// writes); a `:naive` rule (or `eg.seminaive = false`) widens
    /// this to [`Context::Read`] so reads from primitives are
    /// admissible.
    fn query_context(&self) -> crate::Context {
        if self.requires_read_context {
            crate::Context::Read
        } else {
            crate::Context::Pure
        }
    }

    /// The [`crate::Context`] that applies when compiling
    /// primitives on the action side (RHS) of this rule. Under
    /// seminaive, actions may write but not read; a `:naive` rule
    /// widens to [`Context::Full`] so writes and reads are both
    /// admissible.
    fn action_context(&self) -> crate::Context {
        if self.requires_read_context {
            crate::Context::Full
        } else {
            crate::Context::Write
        }
    }

    fn entry(&mut self, x: &core::ResolvedAtomTerm) -> QueryEntry {
        self.entries
            .entry(x.clone())
            .or_insert_with(|| match x {
                core::GenericAtomTerm::Var(_, v) => self
                    .rb
                    .new_var_named(v.sort.column_ty(self.rb.egraph()), &v.name),
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
        ctx: crate::Context,
    ) -> Result<(ExternalFunctionId, Vec<QueryEntry>, ColumnTy), Error> {
        // The typechecker has already checked that this primitive is
        // valid in `ctx`; pick the runtime id that stamps the same ctx
        // onto the state wrapper when invoked.
        let resolved_id = prim.external_id(ctx);

        let mut qe_args = self.args(args);

        if prim.name() == "unstable-fn" {
            let target = match &args[0] {
                core::ResolvedAtomTerm::Literal(span, literal) => match literal.as_ref() {
                    Literal::String(name) => Some((span, name)),
                    _ => None,
                },
                _ => None,
            };
            let Some((span, name)) = target else {
                return Err(Error::BackendError(
                    "expected a string literal as the first argument to `unstable-fn`".to_string(),
                ));
            };
            // Pre-register a panic id used by `FunctionContainer::apply`
            // when the wrapped function is applied in a context that
            // doesn't admit it. Triggered at runtime via the egglog
            // panic side channel so misuse surfaces as an `Err` from
            // `run_rules` rather than a thread unwind.
            let panic_id = self.rb.new_panic(format!(
                "unstable-fn over `{name}` was applied in a context where its wrapped \
                 function is not valid for this call site, if in a rule, add :naive."
            ));
            let resolved = resolve_function_container_target_with_context(
                self.rb.egraph(),
                self.functions,
                self.type_info,
                name,
                prim,
                panic_id,
                ctx,
                span,
            )?;

            qe_args[0] = self.rb.egraph().base_value_constant(resolved);
        }

        Ok((
            resolved_id,
            qe_args,
            prim.output().column_ty(self.rb.egraph()),
        ))
    }

    fn args<'b>(
        &mut self,
        args: impl IntoIterator<Item = &'b core::ResolvedAtomTerm>,
    ) -> Vec<QueryEntry> {
        args.into_iter().map(|x| self.entry(x)).collect()
    }

    fn query(
        &mut self,
        query: &core::Query<ResolvedCall, ResolvedVar>,
        include_subsumed: bool,
    ) -> Result<(), Error> {
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
                    let ctx = self.query_context();
                    let (p, args, ty) = self.prim(p, &atom.args, ctx)?;
                    self.rb.query_prim(p, &args, ty).unwrap()
                }
            }
        }
        Ok(())
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
                            self.rb.lookup(f, &args, move || {
                                format!("{span}: lookup of function {name} failed")
                            })
                        }
                        ResolvedCall::Primitive(p) => {
                            let name = p.name().to_owned();
                            let ctx = self.action_context();
                            let (p, args, ty) = self.prim(p, args, ctx)?;
                            let span = span.clone();
                            self.rb.call_external_func(p, &args, ty, move || {
                                format!("{span}: call of primitive {name} failed")
                            })
                        }
                    };
                    self.entries.insert(v, y.into());
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
                            Change::Subsume if can_subsume => self.rb.subsume(f, &args),
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
                core::GenericCoreAction::Panic(span, message) => self.rb.panic_with_source(
                    message.clone(),
                    PanicOrigin {
                        span: span.clone(),
                        rule: self.rule.clone(),
                    },
                ),
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
        Literal::Int(x) => egraph.base_value_constant::<i64>(*x),
        Literal::Float(x) => egraph.base_value_constant::<sort::F>(x.into()),
        Literal::String(x) => egraph.base_value_constant::<sort::S>(sort::S::new(x.clone())),
        Literal::Bool(x) => egraph.base_value_constant::<bool>(*x),
        Literal::Unit => egraph.base_value_constant::<()>(()),
    }
}

/// Decode a native Input file into ordered literal rows without inserting them.
///
/// The resolved signature can come from a type-only authoring environment; no
/// backend table is required. Custom functions include their output column,
/// while constructors contain only keys. This uses Input's supported-sort and
/// tab-separated cell rules, validating every row before returning any of them.
/// Relative paths are resolved against `fact_directory`, as for native Input.
pub fn parse_input_file(
    function_type: &FuncType,
    file: &str,
    fact_directory: Option<&std::path::Path>,
    span: Span,
) -> Result<Vec<Vec<Literal>>, Error> {
    let mut filename = fact_directory
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    filename.push(file);

    // check that the function uses supported types

    for t in &function_type.input {
        match t.name() {
            "i64" | "f64" | "String" => {}
            s => return Err(Error::UnsupportedInputType(s.to_string(), span.clone())),
        }
    }

    if function_type.subtype != FunctionSubtype::Constructor {
        match function_type.output.name() {
            "i64" | "String" | "Unit" => {}
            s => return Err(Error::UnsupportedInputType(s.to_string(), span.clone())),
        }
    }

    log::info!("Opening file '{filename:?}'...");
    let mut f =
        File::open(&filename).map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;
    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .map_err(|e| Error::IoError(filename.clone(), e, span.clone()))?;

    let mut parsed_contents: Vec<Vec<Literal>> = Vec::with_capacity(contents.lines().count());

    let mut row_schema = function_type.input.clone();
    if function_type.subtype == FunctionSubtype::Custom {
        row_schema.push(function_type.output.clone());
    }

    log::debug!("{row_schema:?}");

    for line in contents.lines() {
        let mut it = line.split('\t').map(|s| s.trim());

        let mut row: Vec<Literal> = Vec::with_capacity(row_schema.len());

        for sort in row_schema.iter() {
            if let Some(raw) = it.next() {
                let val = match sort.name() {
                    "i64" => {
                        if let Ok(i) = raw.parse::<i64>() {
                            Literal::Int(i)
                        } else {
                            return Err(Error::InputFileFormatError(file.to_string()));
                        }
                    }
                    "f64" => {
                        if let Ok(f) = raw.parse::<f64>() {
                            Literal::Float(f.into())
                        } else {
                            return Err(Error::InputFileFormatError(file.to_string()));
                        }
                    }
                    "String" => Literal::String(raw.to_string()),
                    "Unit" => Literal::Unit,
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
            return Err(Error::InputFileFormatError(file.to_string()));
        }

        parsed_contents.push(row);
    }

    log::debug!("Successfully loaded file.");
    Ok(parsed_contents)
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
#[error("authored panic source")]
struct PanicOrigin {
    span: Span,
    rule: Option<String>,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] ParseError),
    #[error(transparent)]
    NotFoundError(#[from] NotFoundError),
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error(transparent)]
    ApiError(#[from] crate::api::ApiError),
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
    /// An authored panic, with its original diagnostic and source metadata.
    #[error("{message}")]
    Panic {
        /// Existing native diagnostic, including any recovery failure.
        message: String,
        /// Exact authored action location, never generated helper text.
        span: Span,
        /// Exact compiled native rule identity, absent for direct actions.
        rule: Option<String>,
    },
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
    #[error("{1}\nRule already exists: {0}")]
    RuleAlreadyExists(String, Span),
    #[error("{1}\nUnsupported type {0} for input")]
    UnsupportedInputType(String, Span),
    #[error("{0}\n{1}")]
    DesugarError(Span, String),
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
    #[error(
        "`{api}` is incompatible with proof mode: {reason} \
         Disable proofs or make the operation a command in the syntax of the egglog language and use `EGraph::parse_and_run`."
    )]
    ProofsIncompatibleApi {
        api: &'static str,
        reason: &'static str,
    },
}

impl Error {
    fn from_backend(error: &(dyn std::error::Error + 'static)) -> Self {
        if let Some(origin) = std::iter::successors(Some(error), |cause| cause.source())
            .find_map(|cause| cause.downcast_ref::<PanicOrigin>())
        {
            Self::Panic {
                message: error.to_string(),
                span: origin.span.clone(),
                rule: origin.rule.clone(),
            }
        } else {
            Self::BackendError(error.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint::SimpleTypeConstraint;
    use crate::*;

    use crate::PureState;

    #[test]
    fn input_file_parser_uses_type_environment_without_tables() {
        let mut graph = EGraph::default();
        let command = graph
            .parse_program(None, "(function parsed () String :no-merge)")
            .unwrap()
            .pop()
            .unwrap();
        graph.resolve_command_preserving_globals(command).unwrap();
        assert!(graph.get_function("parsed").is_none());
        let function_type = graph.type_info().get_func_type("parsed").unwrap();
        let rows = parse_input_file(
            function_type,
            "string_quotes.csv",
            Some(std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests"
            ))),
            Span::Panic,
        )
        .unwrap();
        assert_eq!(rows, [vec![Literal::String("abc".to_string())]]);
        assert!(graph.get_function("parsed").is_none());
    }

    #[test]
    fn input_file_parser_preserves_schema_io_and_format_error_order() {
        let mut graph = EGraph::default();
        for command in graph
            .parse_program(
                None,
                "(function unsupported (bool) String :no-merge) (function parsed () i64 :no-merge)",
            )
            .unwrap()
        {
            graph.resolve_command_preserving_globals(command).unwrap();
        }
        let directory = Some(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests"
        )));
        let unsupported = graph.type_info().get_func_type("unsupported").unwrap();
        assert!(matches!(
            parse_input_file(unsupported, "string_quotes.csv/missing", directory, Span::Panic),
            Err(Error::UnsupportedInputType(name, _)) if name == "bool"
        ));
        let supported = graph.type_info().get_func_type("parsed").unwrap();
        assert!(matches!(
            parse_input_file(
                supported,
                "string_quotes.csv/missing",
                directory,
                Span::Panic
            ),
            Err(Error::IoError(_, _, _))
        ));
        assert!(matches!(
            parse_input_file(supported, "string_quotes.csv", directory, Span::Panic),
            Err(Error::InputFileFormatError(file)) if file == "string_quotes.csv"
        ));
    }

    #[test]
    fn authored_panic_preserves_exact_span_and_rule_in_every_mode() {
        let text = "; λ🌿 byte offsets\n(relation Trigger ()) (Trigger)\n(rule ((Trigger)) ((panic \"stop π\") (panic \"unreachable π\")) :name \"exact rule\")\n(run 1)";
        for mode in 0..4 {
            let mut graph = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            let error = graph
                .parse_and_run_program(Some("panic-origin.egg".into()), text)
                .unwrap_err();
            let Error::Panic {
                message,
                span,
                rule,
            } = error
            else {
                panic!("typed authored panic: {error}")
            };
            assert_eq!(message, "Panic: stop π");
            assert_eq!(rule.as_deref(), Some("exact rule"));
            let Span::Egglog(span) = span else {
                panic!("authored span")
            };
            assert_eq!(span.file.name.as_deref(), Some("panic-origin.egg"));
            assert_eq!(span.file.contents, text);
            assert_eq!(span.i, text.find("(panic \"stop π\")").unwrap());
            assert_eq!(&text[span.i..span.j], "(panic \"stop π\")");
            let Ruleset::Rules(rules, _) = graph.rulesets.get("").unwrap() else {
                panic!("plain ruleset")
            };
            let spans: Vec<_> = rules["exact rule"]
                .core
                .head
                .0
                .iter()
                .filter_map(|action| match action {
                    core::GenericCoreAction::Panic(Span::Egglog(span), _) => Some((span.i, span.j)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                spans,
                [
                    (
                        text.find("(panic \"stop π\")").unwrap(),
                        text.find("(panic \"stop π\")").unwrap() + "(panic \"stop π\")".len()
                    ),
                    (
                        text.find("(panic \"unreachable π\")").unwrap(),
                        text.find("(panic \"unreachable π\")").unwrap()
                            + "(panic \"unreachable π\")".len()
                    ),
                ]
            );
            let direct = "; λ🌿\n(panic \"direct π\")";
            let error = graph
                .parse_and_run_program(Some("direct-origin.egg".into()), direct)
                .unwrap_err();
            let Error::Panic {
                message,
                span: Span::Egglog(span),
                rule,
            } = error
            else {
                panic!("typed direct panic: {error}")
            };
            assert_eq!(message, "Panic: direct π");
            assert!(rule.is_none());
            assert_eq!(&span.file.contents[span.i..span.j], "(panic \"direct π\")");
            assert_eq!(span.i, direct.find("(panic \"direct π\")").unwrap());
            assert_eq!(span.file.name.as_deref(), Some("direct-origin.egg"));
        }
    }

    #[test]
    fn callback_schedule_envelope_matches_native_zero_one_many_and_fail_boundaries() {
        for mode in 0..4 {
            let mut baseline = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            baseline
                .parse_and_run_program(None, "(datatype T (A)) (ruleset empty)")
                .unwrap();
            for count in [0, 1, 3] {
                for wrappers in 0..3 {
                    let leaf = Schedule::Run(
                        span!(),
                        RunConfig {
                            ruleset: "empty".into(),
                            until: None,
                        },
                    );
                    let mut command = Command::RunSchedule(Schedule::Repeat(
                        span!(),
                        count,
                        Box::new(leaf.clone()),
                    ));
                    let spans = vec![span!(); wrappers];
                    for span in &spans {
                        command = Command::Fail(span.clone(), Box::new(command));
                    }
                    let mut native = baseline.clone();
                    let mut native_events = vec![];
                    let mut native_outputs = 0;
                    let native_result = native.run_program_observed_with_repeat_policy(
                        vec![command],
                        &mut |_, origin, output| {
                            if origin == ExecutionOrigin::Requested
                                && matches!(output, CommandOutput::RunSchedule(_))
                            {
                                native_outputs += 1;
                            }
                        },
                        &mut |_, event| {
                            if let ExecutionEvent::RunStarted { origin, schedule } = event {
                                native_events.push((origin, schedule.is_some()));
                            }
                        },
                        &|_, _| RepeatContinuation::ExactCount,
                    );
                    let mut graph = baseline.clone();
                    let mut actual = (vec![], 0usize);
                    let callback_result = graph.with_schedule_envelope(
                        &spans,
                        &mut actual,
                        |graph, (events, _)| {
                            let mut total = RunReport::default();
                            for _ in 0..count {
                                let mut report = None;
                                graph.run_program_embedded_observed(
                                    vec![Command::RunSchedule(leaf.clone())],
                                    &mut |_, origin, output| {
                                        if origin == ExecutionOrigin::Requested
                                            && let CommandOutput::RunSchedule(next) = output
                                        {
                                            report = Some(next);
                                        }
                                    },
                                    &mut |_, event| {
                                        if let ExecutionEvent::RunStarted { origin, schedule } =
                                            event
                                        {
                                            events.push((origin, schedule.is_some()));
                                        }
                                    },
                                    &|_, _| None,
                                )?;
                                total.union(report.expect("actual completed leaf"));
                            }
                            Ok(total)
                        },
                        |(_, outputs), report| {
                            *outputs += usize::from(report.is_some());
                        },
                        |(events, _), event| {
                            if let ExecutionEvent::RunStarted { origin, schedule } = event {
                                events.push((origin, schedule.is_some()));
                            }
                        },
                    );
                    assert_eq!(
                        callback_result.is_ok(),
                        native_result.is_ok(),
                        "mode {mode}, count {count}, Fail {wrappers}"
                    );
                    assert_eq!(
                        actual.0, native_events,
                        "same real inline and outer helpers: mode {mode}, count {count}, Fail {wrappers}"
                    );
                    assert_eq!(actual.1, native_outputs, "same anchor suppression");
                    assert_eq!(
                        graph.get_overall_run_report().iterations.len(),
                        native.get_overall_run_report().iterations.len(),
                        "successful anchor statistics precede Fail: mode {mode}, count {count}, Fail {wrappers}"
                    );
                    assert_eq!(
                        graph.get_overall_run_report().num_matches_per_rule,
                        native.get_overall_run_report().num_matches_per_rule
                    );
                }
            }
        }
    }

    #[test]
    fn ordinary_condition_keeps_check_error_in_every_mode() {
        for mode in 0..4 {
            let mut graph = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            graph
                .parse_and_run_program(None, "(relation R (i64)) (R 1)")
                .unwrap();
            for (text, matched) in [("(check (R 1) (= 1 1))", true), ("(check (R 2))", false)] {
                let [Command::Check(span, facts)]: [Command; 1] =
                    graph.parse_program(None, text).unwrap().try_into().unwrap()
                else {
                    unreachable!()
                };
                let rows = graph.total_size();
                graph
                    .compile_condition(span.clone(), facts.clone())
                    .unwrap();
                assert_eq!(graph.total_size(), rows);
                let mut runs = 0;
                let result = graph.check_condition_observed(span, facts, &mut |event| {
                    if let ExecutionEvent::RunStarted { origin, schedule } = event {
                        assert_eq!(origin, ExecutionOrigin::Internal);
                        assert!(schedule.is_none());
                        runs += 1;
                    }
                });
                if matched {
                    assert!(result.is_ok(), "mode {mode}: {result:?}");
                } else {
                    assert!(
                        matches!(result, Err(Error::CheckError(..))),
                        "mode {mode}: {result:?}"
                    );
                }
                assert_eq!(runs, 1, "ordinary query, not proof-testing expansion");
            }
            if mode == 3 {
                let commands = graph.parse_program(None, "(check (R 1))").unwrap();
                let mut outputs = vec![];
                let mut helper_runs = 0;
                graph
                    .run_program_observed(
                        commands,
                        &mut |_, origin, output| {
                            if origin == ExecutionOrigin::Requested {
                                outputs.push(output);
                            } else if matches!(output, CommandOutput::RunSchedule(_)) {
                                helper_runs += 1;
                            }
                        },
                        &mut |_, _| {},
                    )
                    .unwrap();
                assert!(
                    matches!(outputs.as_slice(), [CommandOutput::ProveExists { .. }]),
                    "authored proof-testing Check remains a real proof request: {outputs:?}"
                );
                assert!(helper_runs > 0, "native proof helpers remain observable");
            }
        }
    }

    #[test]
    fn requested_run_scheduler_selection_excludes_until_and_maintenance() {
        #[derive(Clone)]
        struct Delay;
        impl scheduler::Scheduler for Delay {
            fn filter_matches(
                &mut self,
                _: &scheduler::SchedulerContext<'_>,
                _: &str,
                _: &str,
                _: &mut scheduler::Matches,
            ) -> bool {
                true
            }
        }
        for mode in 0..4 {
            let mut graph = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            graph.parse_and_run_program(None, "(relation R (i64)) (relation S (i64)) (ruleset copy) (rule ((R x)) ((S x)) :ruleset copy) (R 1)").unwrap();
            let scheduler = graph.add_scheduler(Box::new(Delay));
            let commands = graph
                .parser
                .get_program_from_string(None, "(run copy 1) (run copy 1 :until (R 1))")
                .unwrap();
            let selections = std::cell::RefCell::new(Vec::new());
            let mut requested = 0;
            graph
                .run_program_observed_with_policies(
                    commands,
                    &mut |_, _, _| (),
                    &mut |_, event| {
                        if matches!(
                            event,
                            ExecutionEvent::RunStarted {
                                origin: ExecutionOrigin::Requested,
                                ..
                            }
                        ) {
                            requested += 1;
                        }
                    },
                    &|_, _| RepeatContinuation::CanStop,
                    &|cause, site| {
                        selections.borrow_mut().push((cause, site.path.to_vec()));
                        Some(scheduler)
                    },
                )
                .unwrap();
            assert_eq!(*selections.borrow(), [(0, vec![0])], "mode {mode}");
            assert_eq!(
                requested, 2,
                "until retains its native observation boundary"
            );
            graph
                .parse_and_run_program(None, "(fail (check (S 1))) (run copy 1) (check (S 1))")
                .unwrap();
        }
    }

    #[test]
    fn repeat_continuation_is_selected_by_original_requested_loop_path() {
        for mode in 0..4 {
            let mut baseline = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            baseline
                .parse_and_run_program(None, "(ruleset empty)")
                .unwrap();
            let leaf = Schedule::Run(
                span!(),
                RunConfig {
                    ruleset: "empty".into(),
                    until: None,
                },
            );
            let schedule = Schedule::Repeat(
                span!(),
                3,
                Box::new(Schedule::Sequence(
                    span!(),
                    vec![leaf.clone(), Schedule::Repeat(span!(), 4, Box::new(leaf))],
                )),
            );
            for selection in 0..3 {
                let mut graph = baseline.clone();
                let mut invocations = HashMap::<Vec<usize>, usize>::default();
                let mut loops = Vec::new();
                let mut outputs = Vec::new();
                let mut observe = |_: usize, event: ExecutionEvent<'_>| match event {
                    ExecutionEvent::RunStarted {
                        origin: ExecutionOrigin::Requested,
                        schedule: Some(site),
                    } => *invocations.entry(site.path.to_vec()).or_default() += 1,
                    ExecutionEvent::LoopFinished {
                        origin: ExecutionOrigin::Requested,
                        schedule: Some(site),
                        completed,
                        stop,
                        ..
                    } => loops.push((site.path.to_vec(), completed, stop)),
                    _ => (),
                };
                let mut output = |_: usize, origin, value| {
                    if origin == ExecutionOrigin::Requested {
                        outputs.push(value);
                    }
                };
                let commands = vec![Command::RunSchedule(schedule.clone())];
                if selection == 0 {
                    graph
                        .run_program_observed(commands, &mut output, &mut observe)
                        .unwrap();
                } else {
                    graph
                        .run_program_observed_with_repeat_policy(
                            commands,
                            &mut output,
                            &mut observe,
                            &|cause, site| {
                                assert_eq!(cause, 0);
                                assert!(site.path.is_empty() || site.path == [0, 1]);
                                if selection == 2 || site.path.is_empty() {
                                    RepeatContinuation::ExactCount
                                } else {
                                    RepeatContinuation::CanStop
                                }
                            },
                        )
                        .unwrap();
                }
                let outer = if selection == 0 { 1 } else { 3 };
                let inner = if selection == 2 { 4 } else { 1 };
                assert_eq!(invocations[&vec![0, 0]], outer, "mode {mode}");
                assert_eq!(invocations[&vec![0, 1, 0]], outer * inner, "mode {mode}");
                assert_eq!(loops.len(), outer + 1, "no maintenance loop attribution");
                assert_eq!(
                    loops.last().unwrap(),
                    &(
                        vec![],
                        outer,
                        Some(if selection == 0 {
                            NativeLoopStop::CanStop
                        } else {
                            NativeLoopStop::CountReached
                        })
                    )
                );
                assert!(matches!(
                    outputs.as_slice(),
                    [CommandOutput::RunSchedule(_)]
                ));
            }
        }
    }

    #[test]
    fn observed_actions_report_committed_changes_not_attempts() {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(
                None,
                "(datatype A (N i64)) (function f (i64) i64 :merge new)",
            )
            .unwrap();
        let before_report = serde_json::to_value(&graph.overall_run_report).unwrap();
        for (source, expected) in [
            ("(set (f 0) 1)", true),
            ("(set (f 0) 1)", false),
            ("(set (f 0) 2)", true),
            ("(delete (f 0))", true),
            ("(delete (f 0))", false),
            ("(N 0)", true),
            ("(N 0)", false),
            ("(subsume (N 0))", true),
            ("(subsume (N 0))", false),
            ("(N 1) (N 2)", true),
            ("(union (N 1) (N 2))", true),
            ("(union (N 1) (N 2))", false),
            ("(set (f 3) 4) (delete (f 3))", true),
        ] {
            let commands = graph.parse_program(None, source).unwrap();
            let mut changes = vec![];
            graph
                .run_program_embedded_observed(
                    commands,
                    &mut |_, _, _| {},
                    &mut |_, event| {
                        if let ExecutionEvent::ActionsFinished {
                            changed, result, ..
                        } = event
                        {
                            assert!(result.is_ok());
                            changes.push(changed);
                        }
                    },
                    &|_, _| None,
                )
                .unwrap();
            assert!(!changes.is_empty(), "{source}");
            assert_eq!(
                changes.into_iter().any(|changed| changed),
                expected,
                "{source}"
            );
        }
        assert_eq!(
            serde_json::to_value(&graph.overall_run_report).unwrap(),
            before_report
        );
    }

    #[test]
    fn scoped_action_progress_covers_fresh_classes_compile_and_error_prefix() {
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, "(sort A)").unwrap();
        for execute in [false, true] {
            let mut changes = vec![];
            let (result, captures) = graph.eval_scoped_actions_observed(
                &[],
                execute,
                |builder| {
                    builder.fresh_eq_class("fresh".into(), "A", span!())?;
                    builder.capture_eq_class("fresh")?;
                    builder.action(Action::Panic(span!(), "after fresh class".into()))
                },
                &mut |event| {
                    if let ExecutionEvent::ActionsFinished {
                        changed, result, ..
                    } = event
                    {
                        changes.push((changed, result.is_err()));
                    }
                },
            );
            if execute {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("after fresh class")
                );
                assert_eq!(changes, [(true, true)]);
                assert_eq!(captures.len(), 1);
                assert_eq!(graph.materialized_eq_classes().len(), 1);
            } else {
                result.unwrap();
                assert!(changes.is_empty());
                assert!(captures.is_empty());
                assert!(graph.materialized_eq_classes().is_empty());
            }
        }
    }

    #[test]
    fn scoped_action_progress_retains_row_and_union_error_prefixes() {
        for union in [false, true] {
            for duplicate in [false, true] {
                let mut graph = EGraph::default();
                graph
                    .parse_and_run_program(
                        None,
                        "(datatype A (N i64)) (function f (i64) i64 :merge old) (N 0) (N 1)",
                    )
                    .unwrap();
                let source = if union {
                    "(union (N 0) (N 1))"
                } else {
                    "(set (f 0) 7)"
                };
                if duplicate {
                    graph.parse_and_run_program(None, source).unwrap();
                }
                let Command::Action(action) = graph.parse_program(None, source).unwrap().remove(0)
                else {
                    unreachable!()
                };
                let mut changes = vec![];
                let (result, captures) = graph.eval_scoped_actions_observed(
                    &[],
                    true,
                    |builder| {
                        builder.action(action)?;
                        builder.action(Action::Let(
                            span!(),
                            "u".into(),
                            Expr::Call(
                                span!(),
                                "N".into(),
                                vec![Expr::Lit(span!(), Literal::Int(0))],
                            ),
                        ))?;
                        builder.capture_eq_class("u")?;
                        builder.action(Action::Panic(span!(), "after same-action prefix".into()))
                    },
                    &mut |event| {
                        if let ExecutionEvent::ActionsFinished {
                            changed, result, ..
                        } = event
                        {
                            changes.push((changed, result.is_err()));
                        }
                    },
                );
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("after same-action prefix")
                );
                assert_eq!(
                    changes,
                    [(!duplicate, true)],
                    "union={union}, duplicate={duplicate}"
                );
                let zero = graph.backend.base_values_mut().get(0i64);
                let expected = graph
                    .backend
                    .lookup_id(graph.functions["N"].backend_id, &[zero])
                    .unwrap();
                assert_eq!(
                    captures,
                    [(
                        "u".into(),
                        graph.backend.get_canon_repr(expected, ColumnTy::Id)
                    )]
                );
                graph
                    .parse_and_run_program(
                        None,
                        if union {
                            "(check (= (N 0) (N 1)))"
                        } else {
                            "(check (= (f 0) 7))"
                        },
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn observed_actions_union_matches_real_data_and_uf_changes_in_all_modes() {
        for mode in 0..4 {
            let mut graph = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            graph
                .parse_and_run_program(None, "(datatype A (N i64)) (N 8) (N 9)")
                .unwrap();
            // This fixture has no authored globals: every let table is private
            // compiler storage. Keep all other term/view/proof data, plus real
            // native UF leaders, in an independent physical-boundary snapshot.
            let snapshot = |graph: &EGraph| {
                let mut rows = HashSet::default();
                for function in graph
                    .functions
                    .values()
                    .filter(|function| !function.is_let_binding())
                {
                    graph.backend.for_each(function.backend_id, |row| {
                        rows.insert((function.name().to_owned(), row.vals.to_vec(), row.subsumed));
                    });
                }
                let leaders = graph
                    .materialized_eq_classes()
                    .into_iter()
                    .map(|(sort, value)| {
                        (
                            sort,
                            value,
                            graph.backend.get_canon_repr(value, ColumnTy::Id),
                        )
                    })
                    .collect::<HashSet<_>>();
                (rows, leaders)
            };
            for iteration in 0..3 {
                let before = snapshot(&graph);
                let mut previous = before.clone();
                let mut commands = graph.parse_program(None, "(union (N 8) (N 9))").unwrap();
                assert_eq!(commands.len(), 1);
                let command = commands.remove(0);
                assert!(matches!(command, Command::Action(Action::Union(..))));
                // This built-in action needs no macro expansion. Use the same
                // resolver, pre-proof history and executor as the interpreter,
                // but snapshot after each physical command: helpers may commit
                // rows which trailing maintenance removes before the endpoint.
                let resolved = graph
                    .resolve_command(
                        command,
                        CommandPurpose::Authored,
                        &mut |_| {},
                        &mut |_, _| None,
                    )
                    .unwrap();
                if graph.are_proofs_enabled() {
                    graph
                        .proof_check_program
                        .extend(resolved.desugared_before_proofs.clone());
                }
                let mut changed = false;
                let mut committed = false;
                let mut requested = vec![];
                let mut internal_additions = HashSet::default();
                let mut tail_removals = HashSet::default();
                for processed in &resolved.desugared {
                    let mut actions = vec![];
                    let mut completed = false;
                    graph
                        .execute_originated(processed, CommandPolicies::default(), &mut |event| {
                            match event {
                                ExecutionEvent::ActionsFinished {
                                    origin,
                                    changed: next,
                                    result,
                                } => {
                                    assert!(result.is_ok());
                                    actions.push((origin, next));
                                    changed |= next;
                                }
                                ExecutionEvent::CommandCompleted { origin, .. } => {
                                    assert_eq!(origin, processed.origin);
                                    completed = true;
                                }
                                _ => {}
                            }
                        })
                        .unwrap();
                    assert!(completed);
                    // There is no mutation between these successful action and
                    // completion events and execute_originated's return.
                    let current = snapshot(&graph);
                    if !actions.is_empty() {
                        let delta = current != previous;
                        assert_eq!(
                            actions,
                            [(processed.origin, delta)],
                            "mode {mode}, iteration {iteration}: {}",
                            processed.command.to_command()
                        );
                        committed |= delta;
                        if processed.origin == ExecutionOrigin::Requested {
                            requested.extend(actions.iter().map(|(_, next)| *next));
                        } else {
                            internal_additions.extend(current.0.difference(&previous.0).cloned());
                        }
                    }
                    if processed.command_tail {
                        tail_removals.extend(previous.0.difference(&current.0).cloned());
                    }
                    if iteration != 0 {
                        assert_eq!(
                            current
                                .1
                                .iter()
                                .filter(|(sort, ..)| sort.as_ref() == "A")
                                .collect::<HashSet<_>>(),
                            before
                                .1
                                .iter()
                                .filter(|(sort, ..)| sort.as_ref() == "A")
                                .collect::<HashSet<_>>(),
                            "duplicate Union preserves actual native A leaders"
                        );
                    }
                    previous = current;
                }
                assert_eq!(
                    changed, committed,
                    "mode {mode}, union iteration {iteration}"
                );
                assert_eq!(requested, [iteration == 0]);
                if iteration == 0 {
                    assert!(
                        changed,
                        "the first Union changes the native equivalence relation"
                    );
                } else if mode == 0 {
                    assert!(!changed, "ordinary duplicate Union is inert");
                    assert_eq!(previous, before);
                } else if mode == 1 {
                    // Encoded duplicate helpers really commit a raw view and
                    // reflexive parent edge, despite an equal command endpoint.
                    let nine = graph.backend.base_values_mut().get(9i64);
                    let term = graph
                        .backend
                        .lookup_id(graph.functions["N"].backend_id, &[nine])
                        .unwrap();
                    let view = graph
                        .functions
                        .values()
                        .find(|function| function.term_constructor() == Some("N"))
                        .unwrap()
                        .name();
                    let parent = &graph.proof_state.uf_parent["A"];
                    for row in internal_additions.iter().filter(|(name, values, _)| {
                        (name == view && values[0] == nine && values[1] == term)
                            || (name == parent && values[0] == term && values[1] == term)
                    }) {
                        assert!(tail_removals.contains(row));
                    }
                    assert!(
                        internal_additions
                            .iter()
                            .any(|(name, values, _)| name == view
                                && values[0] == nine
                                && values[1] == term)
                    );
                    assert!(
                        internal_additions
                            .iter()
                            .any(|(name, values, _)| name == parent
                                && values[0] == term
                                && values[1] == term)
                    );
                    assert_eq!(previous, before, "tail maintenance removes transient rows");
                }
            }
            graph
                .parse_and_run_program(None, "(check (= (N 8) (N 9)))")
                .unwrap();
        }
    }

    #[test]
    fn observed_action_progress_preserves_modes_scopes_and_caught_prefixes() {
        for mode in 0..4 {
            let mut graph = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            graph
                .parse_and_run_program(None, "(datatype A (N i64))")
                .unwrap();
            let mut actions = vec![
                ("(N 1)", true),
                ("(N 1)", false),
                ("(let $saved (N 1))", true),
                ("(N 1)", false),
                ("(+ 1 2)", false),
            ];
            // Non-Eq globals remain unsupported by term/proof encoding.
            if mode == 0 {
                actions.push(("(let $scalar (+ 1 2))", true));
            }
            for (source, expected) in actions {
                let binding_rows: usize = graph
                    .functions
                    .values()
                    .filter(|function| function.is_let_binding())
                    .map(|function| graph.backend.table_size(function.backend_id))
                    .sum();
                let commands = graph.parse_program(None, source).unwrap();
                let mut changed = false;
                graph
                    .run_program_observed(commands, &mut |_, _, _| {}, &mut |_, event| {
                        if let ExecutionEvent::ActionsFinished { changed: next, .. } = event {
                            changed |= next;
                        }
                    })
                    .unwrap();
                assert_eq!(changed, expected, "mode {mode}: {source}");
                if mode != 0 && !expected {
                    let after: usize = graph
                        .functions
                        .values()
                        .filter(|function| function.is_let_binding())
                        .map(|function| graph.backend.table_size(function.backend_id))
                        .sum();
                    assert!(
                        after > binding_rows,
                        "compiler storage remains materialized"
                    );
                }
            }
            assert_eq!(
                graph
                    .backend
                    .table_size(graph.functions["$saved"].backend_id),
                1
            );
            if mode != 0 {
                assert!(graph.functions.values().any(|function| {
                    function.is_let_binding()
                        && function.term_constructor() == Some("$saved")
                        && graph.backend.table_size(function.backend_id) != 0
                }));
            }
            // ProofTesting must actually check an authored global, not merely
            // construct an encoded graph with the testing flag set.
            graph
                .parse_and_run_program(None, "(check (= $saved (N 1)))")
                .unwrap();
            if mode == 0 {
                graph
                    .parse_and_run_program(None, "(check (= $scalar 3))")
                    .unwrap();
            }
            let rows = graph.num_tuples();
            let (fresh, changed) = graph
                .backend
                .observe_actions(|backend| backend.fresh_eq_class("A".into()));
            assert!(changed, "mode {mode}: executed allocation has no row");
            assert_eq!(graph.num_tuples(), rows);
            let (classes, changed) = graph
                .backend
                .observe_actions(|backend| backend.materialized_eq_classes());
            assert!(classes.contains(&("A".into(), fresh)));
            assert!(!changed, "mode {mode}: inventory is read-only");
            let mut clone = graph.clone();
            clone.parse_and_run_program(None, "(N 2)").unwrap();
            graph
                .parse_and_run_program(None, "(push) (N 3) (pop)")
                .unwrap();
            assert!(graph.parse_and_run_program(None, "(check (N 2))").is_err());
            assert!(graph.parse_and_run_program(None, "(check (N 3))").is_err());
            let commands = graph
                .parse_program(None, "(N 4) (fail (panic \"caught\"))")
                .unwrap();
            let mut changes = vec![];
            graph
                .run_program_observed(commands, &mut |_, _, _| {}, &mut |_, event| {
                    if let ExecutionEvent::ActionsFinished {
                        changed, result, ..
                    } = event
                    {
                        changes.push((changed, result.is_err()));
                    }
                })
                .unwrap();
            assert!(changes.iter().any(|(changed, _)| *changed));
            assert!(changes.iter().any(|(changed, failed)| !changed && *failed));
            graph.parse_and_run_program(None, "(check (N 4))").unwrap();
        }
    }

    #[test]
    fn action_scope_preserves_failed_set_prefix_after_key_union() {
        let mut seed = EGraph::default();
        seed.parse_and_run_program(None, "(datatype A (N i64) (Wrap A)) (function f (A) A :no-merge) (N 0) (N 1) (set (f (N 1)) (N 2))").unwrap();
        let mut key_before = None;
        seed.backend
            .for_each(seed.functions["f"].backend_id, |row| {
                key_before = Some(row.vals[0])
            });
        seed.parse_and_run_program(None, "(union (N 0) (N 1))")
            .unwrap();
        let mut key_after = None;
        seed.backend
            .for_each(seed.functions["f"].backend_id, |row| {
                key_after = Some(row.vals[0])
            });
        assert_ne!(
            key_before, key_after,
            "the pre-existing row key really rebuilt"
        );
        assert_eq!(
            key_after,
            Some(
                seed.backend
                    .get_canon_repr(key_before.unwrap(), ColumnTy::Id)
            )
        );
        let snapshot = |graph: &EGraph| {
            let mut rows = HashSet::default();
            for name in ["N", "Wrap", "f"] {
                graph
                    .backend
                    .for_each(graph.functions[name].backend_id, |row| {
                        rows.insert((name, row.vals.to_vec(), row.subsumed));
                    });
            }
            let classes = graph
                .materialized_eq_classes()
                .into_iter()
                .map(|(sort, value)| {
                    (
                        sort,
                        value,
                        graph.backend.get_canon_repr(value, ColumnTy::Id),
                    )
                })
                .collect::<HashSet<_>>();
            (rows, classes)
        };
        for caught in [false, true] {
            let mut outcomes = vec![];
            for source in [
                "(set (f (N 1)) (Wrap (N 3)))",
                "(action-scope ((let key (N 1)) (let child (N 3)) (let value (Wrap child)) (set (f key) value)))",
            ] {
                let mut graph = seed.clone();
                let before = snapshot(&graph);
                let source = if caught {
                    format!("(fail {source})")
                } else {
                    source.into()
                };
                let commands = graph.parse_program(None, &source).unwrap();
                let mut events = vec![];
                let mut completed = vec![];
                let mut failed = vec![];
                let mut outputs = vec![];
                let result = graph.run_program_observed(
                    commands,
                    &mut |_, _, output| outputs.push(output),
                    &mut |_, event| match event {
                        ExecutionEvent::ActionsFinished {
                            origin,
                            changed,
                            result,
                        } => {
                            events.push((origin, changed, result.err().map(ToString::to_string)));
                        }
                        ExecutionEvent::CommandCompleted { origin, .. } => completed.push(origin),
                        ExecutionEvent::CommandFailed { origin, .. } => failed.push(origin),
                        _ => (),
                    },
                );
                assert_eq!(result.is_ok(), caught);
                assert!(outputs.is_empty());
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].0, ExecutionOrigin::Requested);
                assert!(
                    events[0].1,
                    "constructor prefix committed before the merge error"
                );
                assert!(events[0].2.is_some());
                assert_eq!(
                    completed,
                    if caught {
                        vec![ExecutionOrigin::Requested]
                    } else {
                        vec![]
                    }
                );
                assert_eq!(
                    failed,
                    if caught {
                        vec![]
                    } else {
                        vec![ExecutionOrigin::Requested]
                    }
                );
                assert_ne!(snapshot(&graph), before);
                graph
                    .parse_and_run_program(
                        None,
                        "(check (Wrap (N 3))) (check (= (f (N 1)) (N 2))) (check (= (N 0) (N 1)))",
                    )
                    .unwrap();
                graph
                    .parse_and_run_program(
                        None,
                        "(set (f (N 4)) (N 5)) (check (= (f (N 4)) (N 5)))",
                    )
                    .unwrap();
                outcomes.push((
                    snapshot(&graph),
                    events,
                    result.err().map(|error| error.to_string()),
                ));
            }
            assert_eq!(outcomes[0], outcomes[1], "caught={caught}");
        }
    }

    #[test]
    fn action_scope_panic_keeps_span_and_real_prefix_progress_through_fail() {
        let mut seed = EGraph::default();
        seed.parse_and_run_program(None, "(datatype A (N i64))")
            .unwrap();
        let command = seed
            .parse_program(
                None,
                "(action-scope ((let local (N 7)) (panic \"terminal\")))",
            )
            .unwrap()
            .remove(0);
        let Command::ActionScope(_, actions) = &command else {
            unreachable!()
        };
        let Action::Panic(expected_span, _) = actions.0.last().unwrap() else {
            unreachable!()
        };
        for wrappers in 0..3 {
            let mut graph = seed.clone();
            let mut command = command.clone();
            for _ in 0..wrappers {
                command = Command::Fail(span!(), Box::new(command));
            }
            for duplicate in [false, true] {
                let mut changes = vec![];
                let mut completed = 0;
                let mut failed = 0;
                let result = graph.run_program_observed(vec![command.clone()], &mut |_, _, _| panic!("scope output"), &mut |_, event| match event {
                    ExecutionEvent::ActionsFinished { origin, changed, result } => {
                        assert_eq!(origin, ExecutionOrigin::Requested);
                        assert!(matches!(result, Err(Error::Panic { span, .. }) if span == expected_span));
                        changes.push(changed);
                    }
                    ExecutionEvent::CommandCompleted { .. } => completed += 1,
                    ExecutionEvent::CommandFailed { .. } => failed += 1,
                    _ => (),
                });
                assert_eq!(result.is_ok(), wrappers == 1);
                assert_eq!(changes, [!duplicate]);
                assert_eq!(completed, usize::from(wrappers == 1));
                assert_eq!(failed, usize::from(wrappers != 1));
                graph.parse_and_run_program(None, "(check (N 7))").unwrap();
                assert_eq!(graph.get_function_names(), seed.get_function_names());
            }
        }
    }

    #[test]
    fn action_scope_parser_visitors_and_shape() {
        let mut parser = Parser::default();
        for source in [
            "(action-scope ((let x (N 1)) (set (f) x)))",
            "(action-scope ((let x (N 1)) x))",
            "(action-scope ((let x (panic \"call, not action\")) x))",
            "(action-scope (1))",
            "(action-scope ((let x 1) x 2 (panic \"stop\") 3))",
            "(action-scope (()))",
            "(fail (fail (action-scope ((panic \"stop\")))))",
        ] {
            let parsed = parser.get_program_from_string(None, source).unwrap();
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].to_string(), source);
            let reparsed = parser
                .get_program_from_string(None, &parsed[0].to_string())
                .unwrap();
            assert_eq!(reparsed[0].to_string(), source);
        }
        assert!(
            parser
                .get_program_from_string(None, "(action-scope ())")
                .is_err()
        );
        parser.add_action_macro(Arc::new(SimpleMacro::new("finish", |_, span, _| {
            Ok(vec![
                Action::Let(
                    span.clone(),
                    "made".into(),
                    Expr::Lit(span.clone(), Literal::Int(1)),
                ),
                Action::Expr(span.clone(), Expr::Var(span, "made".into())),
            ])
        })));
        let expanded = parser
            .get_program_from_string(None, "(action-scope ((finish)))")
            .unwrap()
            .remove(0);
        assert_eq!(expanded.to_string(), "(action-scope ((let made 1) made))");
        parser.add_action_macro(Arc::new(SimpleMacro::new("two-finals", |_, span, _| {
            Ok(vec![
                Action::Expr(span.clone(), Expr::Lit(span.clone(), Literal::Int(1))),
                Action::Expr(span.clone(), Expr::Lit(span, Literal::Int(2))),
            ])
        })));
        let ordered = parser
            .get_program_from_string(None, "(action-scope ((two-finals)))")
            .unwrap();
        assert_eq!(ordered[0].to_string(), "(action-scope (1 2))");
        assert_eq!(
            parser
                .get_program_from_string(None, &ordered[0].to_string())
                .unwrap()[0]
                .to_string(),
            ordered[0].to_string()
        );
        let mapped = expanded
            .map_symbols(&mut |head| head, &mut |leaf| format!("local_{leaf}"))
            .visit_exprs(&mut |expr| match expr {
                Expr::Lit(span, Literal::Int(1)) => Expr::Lit(span, Literal::Int(2)),
                other => other,
            })
            .map_string_symbols(&mut |_| panic!("no string-valued command symbols"));
        let mut visited = 0;
        let mapped = mapped.visit_actions(&mut |action| {
            visited += 1;
            action
        });
        assert_eq!(visited, 2);
        assert_eq!(
            mapped.to_string(),
            "(action-scope ((let local_made 2) local_made))"
        );
        let remapped = parser
            .get_program_from_string(None, "(action-scope ((let x (N 1)) (set (f) x)))")
            .unwrap()
            .remove(0)
            .map_symbols(&mut |head| format!("call_{head}"), &mut |leaf| {
                format!("var_{leaf}")
            });
        assert_eq!(
            remapped.to_string(),
            "(action-scope ((let var_x (call_N 1)) (set (call_f) var_x)))"
        );
        let mut graph = EGraph::default();
        assert!(matches!(
            graph.validate_program(vec![Command::ActionScope(span!(), Actions::new(vec![]))]),
            Err(Error::TypeError(TypeError::InvalidActionScope(_)))
        ));
        let resolved = graph
            .resolve_command_preserving_globals(mapped)
            .unwrap()
            .remove(0);
        let resolved = resolved.visit_queries(&mut |_| panic!("scope contains no query"));
        assert_eq!(
            resolved.to_command().to_string(),
            "(action-scope ((let local_made 2) local_made))"
        );
    }

    #[test]
    fn action_scope_locals_do_not_escape_or_shadow_globals() {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(
                None,
                "(datatype A (N i64)) (function f () A :no-merge) (let $global (N 9))",
            )
            .unwrap();
        let functions = graph.get_function_names();
        let globals = graph
            .type_info
            .global_sorts
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        graph
            .parse_and_run_program(
                None,
                "(action-scope ((let x (N 1)) (set (f) x))) (action-scope ((let x (N 2)) x))",
            )
            .unwrap();
        assert_eq!(graph.get_function_names(), functions);
        assert_eq!(
            graph
                .type_info
                .global_sorts
                .keys()
                .cloned()
                .collect::<HashSet<_>>(),
            globals
        );
        for source in [
            "(set (f) x)",
            "(action-scope ((let global (N 3)) global))",
            "(action-scope ((let x y) (let y (N 3)) x))",
            "(action-scope ((let x (N 3)) (let x (N 4)) x))",
        ] {
            let before = graph.num_tuples();
            assert!(
                graph.parse_and_run_program(None, source).is_err(),
                "{source}"
            );
            assert_eq!(graph.num_tuples(), before);
        }
        // Match ordinary local naming: default mode accepts the prefixed
        // spelling with a warning, but it never becomes a global binding.
        graph
            .parse_and_run_program(None, "(action-scope ((let $local (N 3)) $local))")
            .unwrap();
        assert!(!graph.type_info.is_global("$local"));
        assert!(
            graph
                .parse_and_run_program(None, "(set (f) $local)")
                .is_err()
        );
        let mut strict = graph.clone();
        strict.set_strict_mode(true);
        assert!(
            matches!(strict.parse_and_run_program(None, "(action-scope ((let $other (N 4)) $other))"), Err(Error::TypeError(TypeError::NonGlobalPrefixed { name, .. })) if name == "$other")
        );
        let mut clone = graph.clone();
        clone
            .parse_and_run_program(None, "(action-scope ((let x (N 5)) x))")
            .unwrap();
        graph
            .parse_and_run_program(None, "(push) (action-scope ((let x (N 4)) x)) (pop)")
            .unwrap();
        assert!(graph.parse_and_run_program(None, "(check (N 5))").is_err());
        assert!(graph.parse_and_run_program(None, "(check (N 4))").is_err());
        graph
            .parse_and_run_program(None, "(check (= (f) (N 1)))")
            .unwrap();
    }

    #[test]
    fn action_scope_whole_sequence_infers_exact_contextual_heads() {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(
                None,
                "(sort VI (Vec i64)) (sort VS (Vec String)) (function sink () VI :no-merge)",
            )
            .unwrap();
        let commands = graph
            .parse_program(
                None,
                "(action-scope ((let xs (vec-empty)) (set (sink) xs)))",
            )
            .unwrap();
        let resolved = graph
            .resolve_command_preserving_globals(commands[0].clone())
            .unwrap()
            .remove(0);
        let ResolvedNCommand::ActionScope(_, actions) = &resolved else {
            panic!("scope lost during resolution")
        };
        let ResolvedAction::Let(
            _,
            variable,
            ResolvedExpr::Call(_, ResolvedCall::Primitive(primitive), _),
        ) = &actions.0[0]
        else {
            panic!("selected primitive lost")
        };
        assert_eq!(variable.sort.name(), "VI");
        assert_eq!(primitive.output().name(), "VI");
        let ResolvedAction::Set(_, ResolvedCall::Func(function), _, _) = &actions.0[1] else {
            panic!("selected function lost")
        };
        assert_eq!(function.name, "sink");
        assert_eq!(function.output.name(), "VI");
        graph
            .eval_actions(actions, false, ExecutionOrigin::Internal, &mut |_| {
                panic!("compile-only action event")
            })
            .unwrap();
        graph
            .compile_command(resolved, ExecutionOrigin::Internal)
            .unwrap();
        assert_eq!(graph.num_tuples(), 0);
        graph.run_program(commands).unwrap();
        graph
            .parse_and_run_program(None, "(check (= (vec-length (sink)) 0))")
            .unwrap();
    }

    #[test]
    fn action_scope_full_context_is_inert_until_one_ordered_execution() {
        #[derive(Clone)]
        struct Mark(Arc<std::sync::Mutex<Vec<i64>>>);
        impl Primitive for Mark {
            fn name(&self) -> &str {
                "scope-mark"
            }
            fn get_type_constraints(
                &self,
                span: &Span,
            ) -> Box<dyn crate::constraint::TypeConstraint> {
                SimpleTypeConstraint::new(
                    self.name(),
                    vec![I64Sort.to_arcsort(), I64Sort.to_arcsort()],
                    span.clone(),
                )
                .into_box()
            }
        }
        impl FullPrim for Mark {
            fn apply<'a, 'db>(&self, state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
                self.0
                    .lock()
                    .unwrap()
                    .push(state.base_values().unwrap::<i64>(args[0]));
                Some(args[0])
            }
        }
        let calls = Arc::new(std::sync::Mutex::new(vec![]));
        let mut graph = EGraph::default();
        graph.add_full_primitive(Mark(calls.clone()), None);
        graph
            .parse_and_run_program(None, "(function out (i64) i64 :no-merge)")
            .unwrap();
        let before_report = serde_json::to_value(&graph.overall_run_report).unwrap();
        let commands = graph.parse_program(None, "(action-scope ((let key (scope-mark 1)) (let value (scope-mark 2)) (set (out key) value)))").unwrap();
        let compiler = graph.validate_program(commands.clone()).unwrap();
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(compiler.num_tuples(), 0);
        assert_eq!(graph.num_tuples(), 0);
        for expected in [true, false] {
            let mut events = vec![];
            graph
                .run_program_observed(
                    commands.clone(),
                    &mut |_, _, _| panic!("action output"),
                    &mut |_, event| {
                        if let ExecutionEvent::ActionsFinished {
                            origin,
                            changed,
                            result,
                        } = event
                        {
                            assert!(result.is_ok());
                            events.push((origin, changed));
                        }
                    },
                )
                .unwrap();
            assert_eq!(events, [(ExecutionOrigin::Requested, expected)]);
        }
        assert_eq!(*calls.lock().unwrap(), [1, 2, 1, 2]);
        assert_eq!(
            serde_json::to_value(&graph.overall_run_report).unwrap(),
            before_report
        );
        graph
            .parse_and_run_program(None, "(check (= (out 1) 2))")
            .unwrap();
    }

    #[test]
    fn action_scope_compile_errors_keep_variants_and_emit_no_action_event() {
        use egglog_numeric_id::NumericId;
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(
                None,
                "(datatype A (N i64)) (function f (i64) i64 :merge old)",
            )
            .unwrap();
        let control = graph.backend.new_rule("control", false).build();
        for source in [
            "(subsume (f 0))",
            "(action-scope ((let unused (N 7)) (let key 0) (subsume (f key))))",
        ] {
            let commands = graph.parse_program(None, source).unwrap();
            assert!(
                matches!(graph.validate_program(commands.clone()), Err(Error::SubsumeMergeError(name, _)) if name == "f")
            );
            let error = graph
                .run_program_observed(
                    commands.clone(),
                    &mut |_, _, _| panic!("compile output"),
                    &mut |_, event| {
                        assert!(!matches!(event, ExecutionEvent::ActionsFinished { .. }));
                    },
                )
                .unwrap_err();
            assert!(matches!(error, Error::SubsumeMergeError(name, _) if name == "f"));
            assert_eq!(graph.num_tuples(), 0);
            assert!(graph.materialized_eq_classes().is_empty());
            // Direct native Fail already catches backend translation errors;
            // validation/type errors still reject before that runtime wrapper.
            graph
                .run_program(vec![Command::Fail(span!(), Box::new(commands[0].clone()))])
                .unwrap();
            let next = graph.backend.new_rule("reused", false).build();
            assert_eq!(next.index(), control.index() + 1);
            graph.backend.free_rule(next);
        }
        graph.backend.free_rule(control);
        graph
            .parse_and_run_program(None, "(action-scope ((let node (N 7)) node)) (check (N 7))")
            .unwrap();
    }

    #[test]
    fn action_scope_encoded_admission_and_ordered_prefix_proofs() {
        let mut normal = EGraph::default();
        normal.add_full_primitive(FullOnly, None);
        let command = normal
            .parse_program(None, "(action-scope ((let n (full-only)) n))")
            .unwrap()
            .remove(0);
        let resolved = normal
            .resolve_command_preserving_globals(command)
            .unwrap()
            .remove(0);
        let mut command = resolved.to_command();
        for _ in 0..3 {
            assert!(matches!(
                command_supports_proof_encoding(&command, &normal.type_info),
                Err(ProofEncodingUnsupportedReason::PrimitiveWithoutValidator)
            ));
            command = ResolvedCommand::Fail(span!(), Box::new(command));
        }
        for mode in 0..4 {
            let mut graph = match mode {
                0 => EGraph::default(),
                1 => EGraph::new_with_term_encoding(),
                2 => EGraph::new_with_proofs(),
                _ => EGraph::new_with_proofs().with_proof_testing(),
            };
            graph
                .parse_and_run_program(
                    None,
                    "(datatype E (N i64) (Pair E E)) (function keep () E :merge new)",
                )
                .unwrap();
            graph.parse_and_run_program(None, "(action-scope ((let i (+ 1 2)) (let n (N i)) (set (keep) n))) (check (= (keep) (N 3)))").unwrap();
            let functions = graph.get_function_names();
            let mut source = "(action-scope ((let n (N 7)) (let p (Pair n n)) (set (keep) p) (union p (N 9)) (panic \"stop\") (union (N 10) (N 11))))".to_owned();
            for _ in 0..3 {
                let commands = graph.parse_program(None, &source).unwrap();
                graph.validate_program(commands.clone()).unwrap();
                let before = graph.num_tuples();
                graph.validate_program(commands.clone()).unwrap();
                assert_eq!(graph.num_tuples(), before);
                let mut boundaries = 0;
                let result = graph.run_program_observed(
                    commands,
                    &mut |_, origin, output| {
                        assert_eq!(origin, ExecutionOrigin::Internal, "mode={mode}: {output:?}")
                    },
                    &mut |_, event| {
                        if matches!(
                            event,
                            ExecutionEvent::ActionsFinished {
                                origin: ExecutionOrigin::Requested,
                                ..
                            }
                        ) {
                            boundaries += 1;
                        }
                    },
                );
                assert_eq!(
                    result.is_ok(),
                    source.starts_with("(fail (action-scope"),
                    "mode={mode}: {result:?}"
                );
                assert_eq!(boundaries, 1, "mode={mode}");
                source = format!("(fail {source})");
            }
            assert_eq!(graph.get_function_names(), functions);
            graph
                .parse_and_run_program(None, "(check (= (keep) (N 9))) (fail (check (N 10)))")
                .unwrap();
            if mode >= 2 {
                graph
                    .parse_and_run_program(None, "(prove (= (keep) (N 9)))")
                    .unwrap();
            }
            graph.parse_and_run_program(None, "(fail (action-scope ((let n (N 12)) (set (keep) n) (let bad (/ 1 0)) (union (N 20) (N 21))))) (check (= (keep) (N 12))) (fail (check (N 20)))").unwrap();
            if mode >= 2 {
                graph
                    .parse_and_run_program(None, "(prove (= (keep) (N 12)))")
                    .unwrap();
                for wrappers in 1..=3 {
                    let mut isolated = EGraph::new_with_proofs();
                    isolated
                        .parse_and_run_program(None, "(datatype E (N i64))")
                        .unwrap();
                    let mut command =
                        "(action-scope ((union (N 1) (N 2)) (panic \"stop\")))".to_owned();
                    for _ in 0..wrappers {
                        command = format!("(fail {command})");
                    }
                    assert_eq!(
                        isolated.parse_and_run_program(None, &command).is_ok(),
                        wrappers % 2 == 1
                    );
                    isolated
                        .parse_and_run_program(None, "(prove (= (N 1) (N 2)))")
                        .unwrap();
                }
                let mut partial = EGraph::new_with_proofs();
                partial
                    .parse_and_run_program(None, "(datatype E (N i64))")
                    .unwrap();
                assert!(
                    partial
                        .parse_and_run_program(None, "(action-scope ((union (N 31) (N (/ 1 0)))))")
                        .is_err()
                );
                partial
                    .parse_and_run_program(None, "(prove (= (N 31) (N 31)))")
                    .unwrap();
            }
            assert!(graph.parse_and_run_program(None, "(set (keep) n)").is_err());
            graph
                .parse_and_run_program(None, "(function merge-fails () i64 :merge (/ 1 0))")
                .unwrap();
            assert!(graph.parse_and_run_program(None, "(action-scope ((set (merge-fails) 1) (set (merge-fails) 2) (union (N 41) (N 42))))").is_err());
            // Set stages writes. Merge failure is observed at the flush after
            // the VM has evaluated this tail, unlike a failing operand/Panic.
            graph
                .parse_and_run_program(None, "(check (= (N 41) (N 42)))")
                .unwrap();
            if mode >= 2 {
                graph
                    .parse_and_run_program(None, "(prove (= (N 41) (N 42)))")
                    .unwrap();
            }
        }
    }

    #[test]
    fn scoped_actions_compile_inert_prefix_and_canonical_inputs() {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(None, "(datatype A (N i64)) (function keep () A :merge old)")
            .unwrap();
        let before_report = serde_json::to_value(&graph.overall_run_report).unwrap();
        let compile = |builder: &mut ScopedActionBuilder<'_>| {
            builder.fresh_eq_class("u".into(), "A", span!())?;
            builder.capture_eq_class("u")?;
            builder.action(Action::Set(
                span!(),
                "keep".into(),
                vec![],
                Expr::Var(span!(), "u".into()),
            ))?;
            builder.action(Action::Panic(span!(), "stop after capture".into()))
        };
        let (result, captured) = graph.eval_scoped_actions(&[], false, compile);
        result.unwrap();
        assert!(captured.is_empty());
        assert!(graph.materialized_eq_classes().is_empty());
        assert_eq!(graph.num_tuples(), 0);
        let (result, captured) = graph.eval_scoped_actions(&[], true, compile);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("stop after capture")
        );
        let [(name, value)] = captured.as_slice() else {
            panic!("missing prefix capture")
        };
        assert_eq!(name, "u");
        assert_eq!(graph.materialized_eq_classes().len(), 1);
        assert_eq!(graph.num_tuples(), 1);
        let inputs = vec![(name.clone(), "A".into(), *value)];
        let (result, captured) = graph.eval_scoped_actions(&inputs, true, |builder| {
            builder.action(Action::Let(
                span!(),
                "member".into(),
                Expr::Call(
                    span!(),
                    "N".into(),
                    vec![Expr::Lit(span!(), Literal::Int(7))],
                ),
            ))?;
            builder.action(Action::Union(
                span!(),
                Expr::Var(span!(), "u".into()),
                Expr::Var(span!(), "member".into()),
            ))?;
            builder.capture_eq_class("u")
        });
        result.unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(graph.materialized_eq_classes().len(), 1);
        graph
            .parse_and_run_program(None, "(check (= (keep) (N 7)))")
            .unwrap();
        assert_eq!(
            serde_json::to_value(&graph.overall_run_report).unwrap(),
            before_report
        );
        let prefix = graph.materialized_eq_classes();
        let (result, captured) = graph.eval_scoped_actions(&[], true, |builder| {
            builder.fresh_eq_class("unused".into(), "A", span!())?;
            builder.action(Action::Expr(
                span!(),
                Expr::Call(span!(), "missing".into(), vec![]),
            ))
        });
        assert!(result.is_err());
        assert!(captured.is_empty());
        assert_eq!(graph.materialized_eq_classes(), prefix);
    }

    #[test]
    fn scoped_typed_captures_preserve_base_values_and_equality_guards() {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(None, "(datatype A (N i64))")
            .unwrap();
        let functions = graph.get_function_names();
        let compile = |builder: &mut ScopedActionBuilder<'_>| {
            builder.fresh_eq_class("a".into(), "A", span!())?;
            builder.fresh_eq_class("b".into(), "A", span!())?;
            builder.action(Action::Union(
                span!(),
                Expr::Var(span!(), "a".into()),
                Expr::Var(span!(), "b".into()),
            ))?;
            builder.capture_eq_class("b")?;
            for (name, literal, sort) in [
                ("integer", Literal::Int(1), "i64"),
                ("text", Literal::String("typed capture".into()), "String"),
            ] {
                builder.action(Action::Let(
                    span!(),
                    name.into(),
                    Expr::Lit(span!(), literal),
                ))?;
                assert_eq!(builder.capture_local(name)?.name(), sort);
            }
            Ok(())
        };
        let (result, captured) = graph.eval_scoped_actions(&[], false, compile);
        result.unwrap();
        assert!(captured.is_empty());
        assert!(graph.materialized_eq_classes().is_empty());
        for _ in 0..3 {
            let (result, captured) = graph.eval_scoped_actions(&[], true, compile);
            result.unwrap();
            let [(name, equality), (_, integer), (_, text)] = captured.as_slice() else {
                panic!("three typed captures")
            };
            assert_eq!(name, "b");
            assert!(
                graph
                    .materialized_eq_classes()
                    .iter()
                    .any(|(_, value)| value == equality)
            );
            assert_eq!(graph.value_to_base::<i64>(*integer), 1);
            assert_eq!(
                graph.value_to_base::<sort::S>(*text).as_str(),
                "typed capture"
            );
        }
        let (result, captured) = graph.eval_scoped_actions(&[], false, |builder| {
            builder.action(Action::Let(
                span!(),
                "integer".into(),
                Expr::Lit(span!(), Literal::Int(1)),
            ))?;
            builder.capture_eq_class("integer")
        });
        assert!(result.is_err());
        assert!(captured.is_empty());
        assert_eq!(graph.get_function_names(), functions);
        assert_eq!(graph.num_tuples(), 0);
    }

    #[test]
    fn scoped_actions_preserve_escaping_function_value_callbacks() {
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, "(datatype A (N i64)) (sort F (UnstableFn (i64) A)) (function saved () F :merge old)").unwrap();
        let commands = graph
            .parse_program(None, "(set (saved) (unstable-fn \"N\"))")
            .unwrap();
        let mut baseline = graph.clone();
        baseline.run_program(commands.clone()).unwrap();
        // A query must not construct its missing witness, even natively.
        assert!(matches!(
            baseline.parse_and_run_program(None, "(check (= (unstable-app (saved) 7) (N 7)))"),
            Err(Error::CheckError(..))
        ));
        let (result, captures) = graph.eval_scoped_actions(&[], true, |builder| {
            for command in commands {
                let Command::Action(action) = command else {
                    unreachable!()
                };
                builder.action(action)?;
            }
            Ok(())
        });
        result.unwrap();
        assert!(captures.is_empty());
        baseline
            .parse_and_run_program(None, "(unstable-app (saved) 7)")
            .unwrap();
        graph
            .parse_and_run_program(None, "(unstable-app (saved) 7)")
            .unwrap();
        assert_eq!(graph.materialized_eq_classes().len(), 1);
        assert_eq!(graph.num_tuples(), baseline.num_tuples());
        // Constructor function values reject a read-only query application.
        // This deliberately invokes the cached panic callback after the scoped
        // rule has been released, and must retain the ordinary native error.
        let query = "(check (= (unstable-app (saved) 7) (N 7)))";
        let native_error = baseline.parse_and_run_program(None, query).unwrap_err();
        let scoped_error = graph.parse_and_run_program(None, query).unwrap_err();
        assert!(
            native_error
                .to_string()
                .contains("not valid for this call site")
        );
        assert_eq!(scoped_error.to_string(), native_error.to_string());
    }

    #[derive(Clone)]
    struct InnerProduct {
        vec: ArcSort,
    }

    // `InnerProduct` is pure, so it declares
    // `State = PureState` and is usable in all
    // contexts. The Rust type checker enforces that the body only uses
    // methods available on `PureState`.
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
    }

    impl PurePrim for InnerProduct {
        fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
            let mut sum = 0;
            let vec1 = state
                .container_values()
                .get_val::<VecContainer>(args[0])
                .unwrap();
            let vec2 = state
                .container_values()
                .get_val::<VecContainer>(args[1])
                .unwrap();
            assert_eq!(vec1.data.len(), vec2.data.len());
            for (a, b) in vec1.data.iter().zip(vec2.data.iter()) {
                let a = state.base_values().unwrap::<i64>(*a);
                let b = state.base_values().unwrap::<i64>(*b);
                sum += a * b;
            }
            Some(state.base_values().get::<i64>(sum))
        }
    }

    #[derive(Clone)]
    struct FullOnly;

    impl Primitive for FullOnly {
        fn name(&self) -> &str {
            "full-only"
        }

        fn get_type_constraints(&self, span: &Span) -> Box<dyn crate::constraint::TypeConstraint> {
            SimpleTypeConstraint::new(self.name(), vec![I64Sort.to_arcsort()], span.clone())
                .into_box()
        }
    }

    impl FullPrim for FullOnly {
        fn apply<'a, 'db>(&self, state: FullState<'a, 'db>, _args: &[Value]) -> Option<Value> {
            Some(state.base_values().get::<i64>(1))
        }
    }

    #[test]
    fn compile_program_observes_original_commands_once_without_execution() {
        for mode in 0..4 {
            let mut graph = EGraph::default();
            graph.add_full_primitive(FullOnly, None);
            let mut graph = match mode {
                0 => graph,
                1 => graph.with_term_encoding_enabled(),
                2 => graph.with_proofs_enabled(),
                _ => graph.with_proofs_enabled().with_proof_testing(),
            };
            let program = graph
                .parse_program(
                    None,
                    "(datatype E (N i64)) (let $held (N 1))
                 (rule ((N x)) ((N (+ x 1))) :name \"compiled\")",
                )
                .unwrap();
            let mut batches = Vec::new();
            graph
                .compile_program_observed(program, &mut |cause, event| match event {
                    ExecutionEvent::CommandResolved { commands } => {
                        batches.push((
                            cause,
                            commands
                                .iter()
                                .map(|(command, _)| command.clone())
                                .collect::<Vec<_>>(),
                        ));
                    }
                    other => panic!("compile emitted an execution event: {other:?}"),
                })
                .unwrap();
            assert_eq!(
                batches.iter().map(|(cause, _)| *cause).collect::<Vec<_>>(),
                [0, 1, 2]
            );
            assert!(
                matches!(batches[1].1.as_slice(), [ResolvedNCommand::CoreAction(ResolvedAction::Let(_, variable, ResolvedExpr::Call(_, _, _)))] if variable.name == "$held")
            );
            assert!(
                matches!(batches[2].1.as_slice(), [ResolvedNCommand::NormRule { rule }] if rule.name == "compiled")
            );
            assert_eq!(
                graph.num_tuples(),
                0,
                "compile executed an initializer/helper"
            );

            let invalid = graph
                .parse_program(None, "(action-scope ((missing)))")
                .unwrap();
            let mut resolved = 0;
            assert!(
                graph
                    .compile_program_observed(invalid, &mut |_, _| resolved += 1)
                    .is_err()
            );
            assert_eq!(resolved, 0, "typechecking failure emitted a resolved batch");

            let guarded = graph
                .parse_program(None, "(action-scope ((full-only)))")
                .unwrap();
            let mut resolved = 0;
            let result = graph.compile_program_observed(guarded, &mut |_, event| {
                assert!(matches!(event, ExecutionEvent::CommandResolved { .. }));
                resolved += 1;
            });
            assert_eq!(
                resolved, 1,
                "admission must follow the original resolved batch"
            );
            assert_eq!(result.is_ok(), mode == 0);
            assert_eq!(graph.num_tuples(), 0);
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

        egraph.add_pure_primitive(InnerProduct { vec: int_vec_sort }, None);

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

    #[test]
    fn flat_scalar_container_queries_are_checked_by_evaluation() {
        for testing in [false, true] {
            let mut graph = EGraph::new_with_proofs();
            if testing {
                graph = graph.with_proof_testing();
            }
            let outputs = graph
                .parse_and_run_program(
                    None,
                    r#"
                (datatype Item (A) (B))
                (sort Items (Vec Item))
                (sort Ints (Vec i64))
                (A) (B)
                (check (!= (vec-of 1) (vec-of 2)))
                (check (vec-of 0))
                (check (= (vec-of 0) xs))
                (check (= 1 q0) (= 2 q1)
                       (= (vec-of q0) v0) (= (vec-of q1) v1) (!= v0 v1))
                (check (= (A) a) (= (B) b)
                       (= (vec-of a) v0) (= (vec-of b) v1) (!= v0 v1))
                (fail (check (= (vec-of 1) v0) (= (vec-of 2) v1) (= v0 v1)))
                (union (A) (B))
                (check (= (A) (B)) (= (vec-of (A)) (vec-of (B))))
                (check (= (A) a) (= (B) b)
                       (= (vec-of a) v0) (= (vec-of b) v1) (= v0 v1))
                (datatype Witness (Wrap))
                (rule ((= (vec-of 1) v0) (= (vec-of 2) v1) (!= v0 v1)) ((Wrap)))
                (run 1)
                (prove-exists Wrap)
                (datatype PlainWitness (Plain))
                (rule ((!= 1 2)) ((Plain)))
                (run 1)
                (prove-exists Plain)
            "#,
                )
                .unwrap();
            assert!(
                outputs
                    .iter()
                    .any(|output| matches!(output, CommandOutput::ProveExists { .. }))
            );
            let (proof_store, proof_id) = outputs
                .iter()
                .rev()
                .find_map(|output| match output {
                    CommandOutput::ProveExists {
                        proof_store,
                        proof_id,
                    } => Some((proof_store, proof_id)),
                    _ => None,
                })
                .unwrap();
            assert!(matches!(
                proof_store.get(*proof_id).justification(),
                proofs::proof_format::Justification::Fiat
            ));
        }
    }

    #[test]
    fn proof_support_accepts_container_sort_declarations() {
        let mut egraph = EGraph::default();
        let resolved = egraph
            .resolve_program(None, "(datatype X (x))\n(sort XPair (Pair X i64))")
            .unwrap();
        assert!(program_supports_proofs(&resolved, &egraph.type_info));

        let mut egraph = EGraph::default();
        let resolved = egraph
            .resolve_program(None, "(datatype X (x))\n(sort XFn (UnstableFn (X) X))")
            .unwrap();
        assert!(program_supports_proofs(&resolved, &egraph.type_info));
    }

    #[test]
    fn proof_support_rejects_unstable_fn_primitives_without_validators() {
        let mut egraph = EGraph::default();
        let resolved = egraph
            .resolve_program(
                None,
                r#"
                (datatype X (x))
                (sort XFn (UnstableFn (X) X))
                (function id (X) X :merge old)
                (let f (unstable-fn "id"))
                "#,
            )
            .unwrap();
        assert!(!program_supports_proofs(&resolved, &egraph.type_info));
    }

    #[test]
    fn proof_support_accepts_set_primitive_validators() {
        let mut egraph = EGraph::default();
        let resolved = egraph
            .resolve_program(
                None,
                r#"
                (sort ISet (Set i64))
                (function Shared () ISet :merge (set-intersect old new))

                (check (= (set-insert (set-empty) 1) (set-of 1)))
                (check (= (set-remove (set-of 1 2) 2) (set-of 1)))
                (check (= (set-length (set-of 1 2)) 2))
                (check (set-contains (set-of 1 2) 1))
                (check (set-not-contains (set-of 1 2) 3))
                (check (= (set-union (set-of 1) (set-of 2)) (set-of 1 2)))
                (check (= (set-diff (set-of 1 2) (set-of 2)) (set-of 1)))
                (check (= (set-intersect (set-of 1 2) (set-of 2 3)) (set-of 2)))
                "#,
            )
            .unwrap();

        assert!(program_supports_proofs(&resolved, &egraph.type_info));
    }

    /// `set-get` indexes the runtime value order, which the proof checker
    /// cannot reproduce from terms, so it has no validator.
    #[test]
    fn proof_support_rejects_set_get() {
        let mut egraph = EGraph::default();
        let resolved = egraph
            .resolve_program(
                None,
                r#"
                (sort ISet (Set i64))
                (check (= (set-get (set-of 1 2) 0) 1))
                "#,
            )
            .unwrap();

        assert!(!program_supports_proofs(&resolved, &egraph.type_info));
    }

    #[test]
    fn test_typecheck_expr_with_bindings_and_output_rejects_mismatch() {
        let mut egraph = EGraph::default();
        let mut parser = crate::ast::Parser::default();
        let expr = parser.get_expr_from_string(None, "(+ 1 2)").unwrap();

        let resolved = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &[],
                I64Sort.to_arcsort(),
                Context::Pure,
            )
            .unwrap();
        assert_eq!(resolved.output_type().name(), I64Sort.name());

        let err = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &[],
                BoolSort.to_arcsort(),
                Context::Pure,
            )
            .unwrap_err();
        match err {
            TypeError::Mismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected.name(), BoolSort.name());
                assert_eq!(actual.name(), I64Sort.name());
            }
            other => panic!("expected mismatch, got {other:?}"),
        }

        let literal = parser.get_expr_from_string(None, "1").unwrap();
        let err = egraph
            .typecheck_expr_with_bindings_and_output(
                &literal,
                &[],
                BoolSort.to_arcsort(),
                Context::Pure,
            )
            .unwrap_err();
        match err {
            TypeError::Mismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected.name(), BoolSort.name());
                assert_eq!(actual.name(), I64Sort.name());
            }
            other => panic!("expected literal mismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_typecheck_expr_with_bindings_and_output_uses_explicit_bindings() {
        let mut egraph = EGraph::default();
        let mut parser = crate::ast::Parser::default();
        let expr = parser.get_expr_from_string(None, "(+ x 2)").unwrap();
        let bindings = vec![("x".to_string(), span!(), I64Sort.to_arcsort())];

        let resolved = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &bindings,
                I64Sort.to_arcsort(),
                Context::Pure,
            )
            .unwrap();

        assert_eq!(resolved.output_type().name(), I64Sort.name());
    }

    #[test]
    fn test_typecheck_expr_with_bindings_and_output_uses_context() {
        let mut egraph = EGraph::default();
        egraph.add_full_primitive(FullOnly, None);
        let mut parser = crate::ast::Parser::default();
        let expr = parser.get_expr_from_string(None, "(full-only)").unwrap();

        let resolved = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &[],
                I64Sort.to_arcsort(),
                Context::Full,
            )
            .unwrap();
        assert_eq!(resolved.output_type().name(), I64Sort.name());

        let err = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &[],
                I64Sort.to_arcsort(),
                Context::Pure,
            )
            .unwrap_err();
        match err {
            TypeError::UnboundFunction(name, _) => assert_eq!(name, "full-only"),
            other => panic!("expected unbound function, got {other:?}"),
        }
    }

    #[test]
    fn test_typecheck_expr_with_bindings_and_output_rejects_duplicate_bindings() {
        let mut egraph = EGraph::default();
        let mut parser = crate::ast::Parser::default();
        let expr = parser.get_expr_from_string(None, "x").unwrap();
        let bindings = vec![
            ("x".to_string(), span!(), I64Sort.to_arcsort()),
            ("x".to_string(), span!(), BoolSort.to_arcsort()),
        ];

        let err = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &bindings,
                I64Sort.to_arcsort(),
                Context::Pure,
            )
            .unwrap_err();

        match err {
            TypeError::AlreadyDefined(name, _) => assert_eq!(name, "x"),
            other => panic!("expected duplicate binding, got {other:?}"),
        }
    }

    #[test]
    fn test_typecheck_expr_with_bindings_and_output_rewrites_globals() {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(None, "(let $x 1)").unwrap();
        let mut parser = crate::ast::Parser::default();
        let expr = parser.get_expr_from_string(None, "$x").unwrap();

        let resolved = egraph
            .typecheck_expr_with_bindings_and_output(
                &expr,
                &[],
                I64Sort.to_arcsort(),
                Context::Read,
            )
            .unwrap();

        match resolved {
            ResolvedExpr::Call(_, ResolvedCall::Func(func), children) => {
                assert_eq!(func.name, "$x");
                assert!(children.is_empty());
                assert_eq!(func.output.name(), I64Sort.name());
            }
            other => panic!("expected global function call rewrite, got {other:?}"),
        }
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

    #[test]
    fn typed_class_inventory_native_modes_and_logical_projection() {
        for mode in 0..4 {
            let graph = EGraph::default();
            let mut graph = match mode {
                0 => graph,
                1 => graph.with_term_encoding_enabled(),
                2 => graph.with_proofs_enabled(),
                _ => graph.with_proofs_enabled().with_proof_testing(),
            };
            graph
                .parse_and_run_program(
                    None,
                    "(datatype A (AItem i64)) (datatype B (BItem i64)) (sort Unused)
                 (AItem 11) (AItem 22) (BItem 11)
                 (union (AItem 11) (AItem 22))
                 (check (= (AItem 11) (AItem 22)))",
                )
                .unwrap();
            graph
                .parse_and_run_program(
                    None,
                    "(delete (AItem 11)) (delete (AItem 22)) (delete (BItem 11)) (run 1)",
                )
                .unwrap();
            for (name, retained_terms) in [("AItem", 2), ("BItem", 1)] {
                if mode != 0 {
                    let views: Vec<_> = graph
                        .functions_iter()
                        .filter(|(_, function)| {
                            function.term_constructor().is_some_and(|term| term == name)
                        })
                        .map(|(name, _)| name.clone())
                        .collect();
                    assert_eq!(views.len(), 1);
                    assert_eq!(graph.get_size(&views[0]), 0, "mode {mode}: deleted view");
                    // Encoded Delete intentionally retains immutable term tables.
                    assert_eq!(graph.get_size(name), retained_terms);
                    graph.clear_function(name).unwrap();
                }
                assert_eq!(graph.get_size(name), 0, "mode {mode}: no producer rows");
            }
            let classes = graph.materialized_eq_classes();
            for (sort, _) in &classes {
                assert!(graph.get_sort_by_name(sort).unwrap().is_eq_sort());
            }
            assert!(!classes.iter().any(|(sort, _)| sort.as_ref() == "Unused"));
            for sort in ["A", "B"] {
                let mut logical = HashSet::default();
                for (_, value) in classes.iter().filter(|(name, _)| name.as_ref() == sort) {
                    let mut leader = *value;
                    // Existing encoding metadata, not helper-name reconstruction.
                    if let Some(uf) = graph.proof_state.uf_parent.get(sort) {
                        graph
                            .backend
                            .for_each(graph.functions[uf].backend_id, |row| {
                                if row.vals[0] == *value {
                                    leader = row.vals[1];
                                }
                            });
                    }
                    logical.insert(leader);
                }
                assert_eq!(logical.len(), 1, "mode {mode}, sort {sort}");
            }
            let rows = graph.num_tuples();
            let report = serde_json::to_value(&graph.overall_run_report).unwrap();
            assert_eq!(graph.materialized_eq_classes(), classes);
            assert_eq!(graph.num_tuples(), rows);
            assert_eq!(
                serde_json::to_value(&graph.overall_run_report).unwrap(),
                report
            );
            let bare = graph.fresh_eq_class("Unused").unwrap();
            assert!(
                graph
                    .materialized_eq_classes()
                    .contains(&("Unused".into(), bare))
            );
            assert_eq!(graph.num_tuples(), rows);
        }
    }

    #[test]
    fn typed_class_inventory_scopes_bare_classes_and_empty_native_name() {
        for use_pop in [false, true] {
            let mut graph = EGraph::default();
            graph
                .add_sort(
                    EqSort {
                        name: String::new(),
                    },
                    span!(),
                )
                .unwrap();
            let unnamed = graph.fresh_eq_class("").unwrap();
            assert!(graph.fresh_eq_class("i64").is_err());
            assert!(graph.fresh_eq_class("unknown").is_err());
            graph
                .parse_and_run_program(None, "(datatype A (AItem i64)) (AItem 1)")
                .unwrap();
            let prefix = graph.materialized_eq_classes();
            assert!(prefix.contains(&(Arc::<str>::from(""), unnamed)));
            let saved = graph.clone();
            if use_pop {
                graph.push();
            }
            graph.parse_and_run_program(None, "(AItem 2)").unwrap();
            let bare = graph.fresh_eq_class("A").unwrap();
            assert_eq!(graph.materialized_eq_classes().len(), prefix.len() + 2);
            if use_pop {
                graph.pop().unwrap();
            } else {
                graph.restore_from(saved);
            }
            assert_eq!(graph.materialized_eq_classes(), prefix);
            graph.parse_and_run_program(None, "(AItem 3)").unwrap();
            assert_eq!(graph.fresh_eq_class("A").unwrap(), bare);
            graph.clear_function("AItem").unwrap();
            assert_eq!(graph.num_tuples(), 0);
            assert_eq!(graph.materialized_eq_classes().len(), prefix.len() + 2);
        }
    }

    #[test]
    fn no_decomp_global_policy_preserves_installed_choices_and_execution() {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(
                None,
                "(relation L (i64 i64)) (relation M (i64 i64)) (relation N (i64 i64))
                 (relation OutA (i64)) (relation OutB (i64))
                 (relation OutC (i64)) (relation OutD (i64))
                 (L 1 2) (L 10 20) (M 2 3) (M 20 30) (N 3 4) (N 30 40)
                 (ruleset a) (ruleset b) (ruleset c) (ruleset d)
                 (rule ((L x y) (M y z) (N z w)) ((OutA x)) :ruleset a :name \"a\")",
            )
            .unwrap();
        graph.no_decomp = true;
        graph
            .parse_and_run_program(
                None,
                "(rule ((L x y) (M y z) (N z w)) ((OutB x)) :ruleset b :name \"b\")",
            )
            .unwrap();
        graph.no_decomp = false;
        graph
            .parse_and_run_program(
                None,
                "(rule ((L x y) (M y z) (N z w)) ((OutC x)) :ruleset c :name \"c\")
                 (rule ((L x y) (M y z) (N z w)) ((OutD x)) :ruleset d :name \"d\" :no-decomp)",
            )
            .unwrap();
        let choices = ["a", "b", "c", "d"].map(|name| {
            let ast::Ruleset::Rules(rules, _) = &graph.rulesets[name] else {
                panic!("expected an ordinary compiled ruleset");
            };
            let compiled = &rules[name];
            assert_eq!(compiled.no_decomp, matches!(name, "b" | "d"));
            (compiled.backend_id, compiled.no_decomp)
        });
        for (index, (id, _)) in choices.iter().enumerate() {
            assert!(choices[..index].iter().all(|(previous, _)| previous != id));
        }
        // Reports for decomposed plans are not implemented. TimeOnly still
        // executes the actual chain query; private compiled bits/IDs witness
        // old-plan retention without a new inspection API or report fallback.
        graph.set_report_level(ReportLevel::TimeOnly);
        graph.no_decomp = true;
        for (index, name) in ["a", "b", "c", "d"].into_iter().enumerate() {
            let outputs = graph
                .parse_and_run_program(None, &format!("(run {name} 1)"))
                .unwrap();
            let CommandOutput::RunSchedule(report) = outputs.last().unwrap() else {
                panic!("expected native execution report");
            };
            assert_eq!(report.num_matches_per_rule.get(name), Some(&2));
            let ast::Ruleset::Rules(rules, _) = &graph.rulesets[name] else {
                panic!("expected retained compiled ruleset");
            };
            assert_eq!(
                (rules[name].backend_id, rules[name].no_decomp),
                choices[index]
            );
        }
        graph
            .parse_and_run_program(
                None,
                "(check (OutA 1) (OutA 10) (OutB 1) (OutB 10)
                        (OutC 1) (OutC 10) (OutD 1) (OutD 10))
                 (fail (check (OutA 100)))",
            )
            .unwrap();
        graph.push();
        graph.no_decomp = false;
        graph.pop().unwrap();
        assert!(graph.no_decomp);
        let cloned = graph.clone();
        for (index, name) in ["a", "b", "c", "d"].into_iter().enumerate() {
            let ast::Ruleset::Rules(rules, _) = &cloned.rulesets[name] else {
                panic!("expected cloned compiled ruleset");
            };
            assert_eq!(
                (rules[name].backend_id, rules[name].no_decomp),
                choices[index]
            );
        }
    }

    #[test]
    fn test_extension_state_clones_and_restores_with_egraph() {
        let mut egraph = EGraph::default();
        assert_eq!(egraph.extension_state::<usize>(), None);
        assert_eq!(egraph.clone().extension_state::<usize>(), None);

        *egraph.extension_state_or_default::<usize>() = 1;

        let mut cloned = egraph.clone();
        assert_eq!(cloned.extension_state::<usize>(), Some(&1));
        *cloned.extension_state_or_default::<usize>() = 2;
        assert_eq!(egraph.extension_state::<usize>(), Some(&1));

        egraph.push();
        *egraph.extension_state_or_default::<usize>() = 3;
        egraph.pop().unwrap();

        assert_eq!(egraph.extension_state::<usize>(), Some(&1));
    }

    #[test]
    fn scope_restore_preserves_native_pop_fields_and_saved_state() {
        for use_pop in [false, true] {
            let mut graph = EGraph::default();
            let saved_threads = if cfg!(target_family = "wasm") { 1 } else { 2 };
            graph.set_num_threads(saved_threads);
            graph.set_report_level(ReportLevel::WithPlan);
            *graph.extension_state_or_default::<usize>() = 1;
            graph.parse_and_run_program(None, "(datatype M (N i64) (Box M)) (ruleset r) (rewrite (Box x) x :ruleset r) (Box (N 1)) (run r 1)").unwrap();
            assert!(!graph.overall_run_report.iterations.is_empty());
            let saved = graph.clone();
            if use_pop {
                graph.push();
            }
            graph.set_num_threads(1);
            graph.set_report_level(ReportLevel::TimeOnly);
            *graph.extension_state_or_default::<usize>() = 2;
            graph.parser.ensure_no_reserved_symbols = false;
            graph.seminaive = !saved.seminaive;
            graph
                .parse_and_run_program(None, "(Box (N 2)) (run r 1)")
                .unwrap();
            graph.parser.symbol_gen.include_zero(true);
            let minted = graph.parser.symbol_gen.fresh("scope");
            let symbols = graph.parser.symbol_gen.clone();
            let report = serde_json::to_value(&graph.overall_run_report).unwrap();
            assert!(
                graph.overall_run_report.iterations.len()
                    > saved.overall_run_report.iterations.len()
            );
            let seminaive = saved.seminaive;
            if use_pop {
                graph.pop().unwrap();
            } else {
                graph.restore_from(saved);
            }
            assert_eq!(
                serde_json::to_value(&graph.overall_run_report).unwrap(),
                report
            );
            assert_eq!(graph.parser.symbol_gen, symbols);
            assert_ne!(graph.parser.symbol_gen.fresh("scope"), minted);
            assert_eq!(graph.extension_state::<usize>(), Some(&1));
            assert_eq!(graph.num_threads(), saved_threads);
            assert!(graph.parser.ensure_no_reserved_symbols);
            assert_eq!(graph.seminaive, seminaive);
            graph
                .parse_and_run_program(None, "(fail (check (N 2)))")
                .unwrap();
            let outputs = graph
                .parse_and_run_program(None, "(Box (N 3)) (run r 1)")
                .unwrap();
            let CommandOutput::RunSchedule(report) = outputs.last().unwrap() else {
                panic!("missing run")
            };
            let rules: Vec<_> = report
                .iterations
                .iter()
                .flat_map(|i| i.rule_set_report.rule_reports.values())
                .flatten()
                .collect();
            assert!(!rules.is_empty());
            assert!(rules.iter().all(|rule| rule.plan.is_some()));
        }
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
                (constructor expensive () Math :cost 100)
                (constructor cheap () Math)
                (constructor cheap-1 () Math)
                ; we make the container cheap so that it will be extracted if possible, but then we mark it as subsumed
                ; so the (expensive) expr should be extracted instead
                (let res (container (cheap)))
                (union res (expensive))
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
        assert_eq!(outputs[0].to_string(), "(expensive)\n");
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
                (constructor expensive () Math :cost 100)
                (constructor cheap () Math)
                (expensive)
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
                (union (expensive) x)
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
        assert_eq!(res[0].to_string(), "(expensive)\n");
    }

    #[test]
    fn test_run_undefined_ruleset_errors() {
        let mut egraph = EGraph::default();
        let err = egraph
            .parse_and_run_program(None, "(ruleset test)\n(run test2 1)")
            .unwrap_err();
        assert!(matches!(err, Error::NoSuchRuleset(name, _) if name == "test2"));
    }

    #[test]
    fn test_duplicate_rule_name_errors() {
        let err = EGraph::default()
            .parse_and_run_program(
                None,
                "(relation foo (i64))
                 (rule ((foo x)) ((foo (+ x 1))) :name \"r\")
                 (rule ((foo x)) ((foo (+ x 2))) :name \"r\")",
            )
            .unwrap_err();
        assert!(matches!(err, Error::RuleAlreadyExists(..)));
    }

    #[test]
    fn test_extract_negative_variants_errors() {
        let err = EGraph::default()
            .parse_and_run_program(
                None,
                "(sort Math)(constructor Num (i64) Math)(let x (Num 5))(extract x -1)",
            )
            .unwrap_err();
        assert!(matches!(err, Error::ExtractError(..)));
    }

    #[test]
    fn test_input_missing_file_errors() {
        let err = EGraph::default()
            .parse_and_run_program(
                None,
                "(function edge (i64) i64 :merge old)(input edge \"/no/such/file_xyz\")",
            )
            .unwrap_err();
        assert!(matches!(err, Error::IoError(..)));
    }

    /// Regression test: `SymbolGen::fresh` minted colliding names for
    /// different hints (e.g. hint "x" at count 11 and hint "x1" at count 1
    /// both produce "@x11"). Two distinct globals sharing one minted name
    /// were then conflated while compiling an action, silently binding a
    /// global to the wrong term.
    #[test]
    fn test_fresh_name_collision_globals() {
        let mut egraph = EGraph::default();
        // Each `(or2 x y)` command consumes one `fresh("x")`. After the ten
        // `t` commands (plus the definition of `x` itself), the minted name
        // for a fresh "x" is "@x11". Global `x1` has only been minted once
        // (by its own definition), so minting for it also produces "@x11".
        // The final command references both globals, conflating them.
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort B)
                (constructor var (String) B)
                (constructor and2 (B B) B)
                (constructor or2 (B B) B)
                (let x (var "x"))
                (let y (var "y"))
                (let t1 (or2 x y))
                (let t2 (or2 x y))
                (let t3 (or2 x y))
                (let t4 (or2 x y))
                (let t5 (or2 x y))
                (let t6 (or2 x y))
                (let t7 (or2 x y))
                (let t8 (or2 x y))
                (let t9 (or2 x y))
                (let t10 (or2 x y))
                (let x1 (var "x1"))
                (let out (and2 x x1))
                "#,
            )
            .unwrap();

        // `out` must equal the term it was bound to. With the name
        // collision, the last action was miscompiled as
        // `(set (out) (and2 (x1) (x1)))`, so this check failed.
        egraph
            .parse_and_run_program(
                None,
                r#"
                (check (= out (and2 (var "x") x1)))
                "#,
            )
            .unwrap();
    }
}
