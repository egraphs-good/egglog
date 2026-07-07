//! Rust entry point for egglog.
//!
//! ```
//! use egglog::prelude::*;
//! ```
//!
//! Most workflows are best expressed as an egglog program parsed and
//! run via [`crate::EGraph::parse_and_run_program`]: declare your
//! sorts, functions, and rules once in egglog text, then call into
//! the database from Rust around it.
//!
//! ```
//! use egglog::prelude::*;
//! let mut eg = EGraph::default();
//! eg.parse_and_run_program(
//!     None,
//!     "(datatype Math (Num i64) (Add Math Math))
//!      (Add (Num 1) (Num 2))",
//! )?;
//! # Ok::<(), egglog::Error>(())
//! ```
//!
//! Three cases need a Rust escape from the language:
//!
//! 1. **Driving the e-graph database directly** — building rows,
//!    looking them up, iterating tables, running ad-hoc queries from
//!    Rust between (or instead of) running egglog rules.
//! 2. **Custom rules whose right-hand side runs Rust** — a
//!    [`rust_rule`] / [`rust_rule_full`] callback fires on every
//!    match and gets a state handle to read/write the database.
//!    Useful when the rule body needs arithmetic, control flow, or
//!    data conversion that's awkward in egglog itself.
//! 3. **Custom primitives callable from egglog expressions** — new
//!    functions (e.g. arithmetic on a custom Rust type, an FFI call,
//!    a cost computation) that you want to invoke from egglog code as
//!    if they were built-ins. See "Extending egglog" below.
//!
//! Cases 1 and 2 share the same surface: the [`crate::Read`] and
//! [`crate::Write`] capability traits, implemented on the
//! [`crate::PureState`] / [`crate::ReadState`] / [`crate::WriteState`]
//! / [`crate::FullState`] wrappers. Inside a rule callback you
//! receive one of those wrappers directly; outside a rule you get a
//! [`crate::FullState`] from [`crate::EGraph::update`]:
//!
//! ```
//! use egglog::prelude::*;
//! let mut eg = EGraph::default();
//! eg.parse_and_run_program(None, "(function f (i64) i64 :no-merge)")?;
//!
//! // Stage writes; flush once when the closure returns.
//! eg.update(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
//!
//! // Read in a separate `update` call — writes inside a closure
//! // aren't visible to reads in the same closure.
//! let v = eg.update(|fs| fs.lookup("f", 1_i64))?;
//! assert_eq!(v.map(|v| eg.value_to_base::<i64>(v)), Some(42));
//! # Ok::<(), egglog::Error>(())
//! ```
//!
//! ## Adding and reading facts
//!
//! Methods on [`crate::Read`] and [`crate::Write`], available via the
//! state wrappers:
//!
//! - [`crate::Write::set`] — write a function-table cell `(set (f k) v)`.
//! - [`crate::Write::add`] — mint or look up a constructor / relation eclass.
//! - [`crate::Write::remove`] — remove a row from any subtype.
//! - [`crate::Write::union`] — union two eclass `Value`s in the union-find.
//! - [`crate::Read::lookup`] — read a function's output value.
//! - [`crate::Read::eclass_of`] — read a constructor's eclass without minting.
//! - [`crate::Read::contains`] — row presence on any subtype.
//!
//! ## Iterating and querying
//!
//! - [`crate::Read::function_entries`] visits each entry of a function
//!   table via a callback, exposing `inputs` / `output` as raw `Value`s.
//! - [`crate::Read::constructor_enodes`] visits each enode of a
//!   constructor / relation table, exposing `children` / `eclass`.
//! - [`crate::EGraph::query`] runs a one-shot pattern query and
//!   returns one `HashMap<String, Value>` per match, keyed by
//!   variable name. Useful for extracting bindings without compiling
//!   a persistent rule.
//!
//! ## Extracting terms
//!
//! Pulling a Rust-side term out of an eclass:
//!
//! - [`crate::EGraph::extract_value`] — picks the lowest-cost
//!   representative under the default tree-additive cost model and
//!   returns it as a [`crate::TermId`] in a [`crate::TermDag`].
//! - [`crate::EGraph::extract_value_with_cost_model`] — same but with
//!   a user-supplied cost model, an impl of
//!   [`crate::extract::CostModel`].
//! - [`crate::EGraph::extract_value_to_string`] — convenience: prints
//!   the extracted term back as egglog-syntax text.
//!
//! See the [`crate::extract`] module for the full API
//! ([`crate::extract::Extractor`], variant extraction,
//! sort-restricted extraction, custom cost types).
//!
//! To get the `(sort, Value)` pair an `extract_value` call needs in
//! the first place, the easiest path is to let-bind a global name in
//! egglog and then resolve it with [`crate::EGraph::eval_expr`]:
//!
//! ```
//! use egglog::prelude::*;
//! let mut eg = EGraph::default();
//! eg.parse_and_run_program(
//!     None,
//!     "(datatype Math (Num i64) (Add Math Math))
//!      (let $root (Add (Num 1) (Num 2)))",
//! )?;
//! let (sort, value) = eg.eval_expr(&exprs::var("$root"))?;
//! let (_dag, _term, _cost) = eg.extract_value(&sort, value)?;
//! # Ok::<(), egglog::Error>(())
//! ```
//!
//! ## Rules
//!
//! - [`rule`] — add a rule whose RHS is egglog code.
//! - [`rust_rule`] — add a rule whose RHS is a Rust closure
//!   `Fn(WriteState, &[Value]) -> Option<()>`. Seminaive-safe.
//! - [`rust_rule_full`] — same but the closure receives a
//!   [`crate::FullState`] (can read the database in the callback).
//!   Runs `:naive` since the body sees live state.
//! - [`run_ruleset`] / [`add_ruleset`] step rules forward.
//!
//! ## Extending egglog
//!
//! - **Custom sort types:** [`BaseSort`] / [`ContainerSort`].
//! - **Custom primitives:** implement [`crate::Primitive`] plus one
//!   of [`crate::PurePrim`] / [`crate::ReadPrim`] /
//!   [`crate::WritePrim`] / [`crate::FullPrim`] (pick by what the
//!   body needs to do — pure, read-only, write-only, full). Register
//!   via the matching `EGraph::add_*_primitive`. The state wrapper
//!   the body sees enforces what the body can do; see issue #772 for
//!   the seminaive-safety reasoning.
//! - **Simple pure primitives:** the [`add_primitive!`] /
//!   [`add_primitive_with_validator!`] / [`add_literal_prim!`] macros
//!   cover the "pure native function" case without writing out the
//!   full trait impl.
//!
//! ## Caveat: type unsafety at the column level
//!
//! Eclass identifiers flow through the API as bare [`Value`]s. A
//! `Value` returned from a `Math` constructor and a `Value` returned
//! from a `List` constructor are indistinguishable at compile time
//! *and* at runtime — callers track sort identity themselves.
//! [`crate::Write::union`] does not check that its arguments are the
//! same eq-sort; [`crate::Write::set`] and [`crate::Write::add`] do
//! not check that their column values match the table's declared
//! column sorts. Arity (column count) and subtype (function vs.
//! constructor) *are* checked at runtime via [`crate::ApiError`].

use crate::*;
use std::any::{Any, TypeId};

// Re-exports in `prelude` for convenience.
pub use egglog::ast::{Action, Fact, Facts, GenericActions, RustSpan, Span};
pub use egglog::sort::{BigIntSort, BigRatSort, BoolSort, F64Sort, I64Sort, StringSort, UnitSort};
pub use egglog::{CommandMacro, CommandMacroRegistry};
pub use egglog::{Core, FullState, PureState, Read, ReadState, Write, WriteState};
pub use egglog::{EGraph, span};
pub use egglog::{action, actions, datatype, expr, fact, facts, sort, vars};

/// Trait for types that can be converted to/from Literal for use in validated primitives.
/// This enables automatic validator generation for literal primitives.
pub trait LiteralConvertible: Sized {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal;
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self>;
}

impl LiteralConvertible for i64 {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal {
        egglog_ast::generic_ast::Literal::Int(self)
    }
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self> {
        match lit {
            egglog_ast::generic_ast::Literal::Int(i) => Some(*i),
            _ => None,
        }
    }
}

impl LiteralConvertible for bool {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal {
        egglog_ast::generic_ast::Literal::Bool(self)
    }
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self> {
        match lit {
            egglog_ast::generic_ast::Literal::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl LiteralConvertible for ordered_float::OrderedFloat<f64> {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal {
        egglog_ast::generic_ast::Literal::Float(self)
    }
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self> {
        match lit {
            egglog_ast::generic_ast::Literal::Float(f) => Some(*f),
            _ => None,
        }
    }
}

impl LiteralConvertible for egglog::sort::F {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal {
        egglog_ast::generic_ast::Literal::Float(self.0)
    }
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self> {
        match lit {
            egglog_ast::generic_ast::Literal::Float(f) => Some(egglog::sort::F::from(*f)),
            _ => None,
        }
    }
}

impl LiteralConvertible for egglog::sort::S {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal {
        egglog_ast::generic_ast::Literal::String(self.0)
    }
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self> {
        match lit {
            egglog_ast::generic_ast::Literal::String(s) => Some(egglog::sort::S::new(s.clone())),
            _ => None,
        }
    }
}

impl LiteralConvertible for () {
    fn to_literal(self) -> egglog_ast::generic_ast::Literal {
        egglog_ast::generic_ast::Literal::Unit
    }
    fn from_literal(lit: &egglog_ast::generic_ast::Literal) -> Option<Self> {
        match lit {
            egglog_ast::generic_ast::Literal::Unit => Some(()),
            _ => None,
        }
    }
}

pub mod exprs {
    use super::*;

    /// Creates a variable expression.
    pub fn var(name: &str) -> Expr {
        Expr::Var(span!(), name.to_owned())
    }

    /// Creates an integer literal expression.
    pub fn int(value: i64) -> Expr {
        Expr::Lit(span!(), Literal::Int(value))
    }

    /// Creates a float literal expression.
    pub fn float(value: f64) -> Expr {
        Expr::Lit(span!(), Literal::Float(value.into()))
    }

    /// Creates a string literal expression.
    pub fn string(value: &str) -> Expr {
        Expr::Lit(span!(), Literal::String(value.to_owned()))
    }

    /// Creates a unit literal expression.
    pub fn unit() -> Expr {
        Expr::Lit(span!(), Literal::Unit)
    }

    /// Creates a boolean literal expression.
    pub fn bool(value: bool) -> Expr {
        Expr::Lit(span!(), Literal::Bool(value))
    }

    /// Creates a function call expression.
    pub fn call(f: &str, xs: Vec<Expr>) -> Expr {
        Expr::Call(span!(), f.to_owned(), xs)
    }
}

/// Create a new ruleset.
pub fn add_ruleset(egraph: &mut EGraph, ruleset: &str) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::AddRuleset(span!(), ruleset.to_owned())])
}

/// Run one iteration of a ruleset.
pub fn run_ruleset(egraph: &mut EGraph, ruleset: &str) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::RunSchedule(Schedule::Run(
        span!(),
        RunConfig {
            ruleset: ruleset.to_owned(),
            until: None,
        },
    ))])
}

#[macro_export]
macro_rules! sort {
    (BigInt) => {
        BigIntSort.to_arcsort()
    };
    (BigRat) => {
        BigRatSort.to_arcsort()
    };
    (bool) => {
        BoolSort.to_arcsort()
    };
    (f64) => {
        F64Sort.to_arcsort()
    };
    (i64) => {
        I64Sort.to_arcsort()
    };
    (String) => {
        StringSort.to_arcsort()
    };
    (Unit) => {
        UnitSort.to_arcsort()
    };
    ($t:expr) => {
        $t
    };
}

#[macro_export]
macro_rules! vars {
    [$($x:ident : $t:tt),* $(,)?] => {
        &[$((stringify!($x), sort!($t))),*]
    };
}

#[macro_export]
macro_rules! expr {
    ((unquote $unquoted:expr)) => { $unquoted };
    (($func:tt $($arg:tt)*)) => { exprs::call(stringify!($func), vec![$(expr!($arg)),*]) };
    ($value:literal) => { exprs::int($value) };
    ($quoted:tt) => { exprs::var(stringify!($quoted)) };
}

#[macro_export]
macro_rules! fact {
    ((= $($arg:tt)*)) => { Fact::Eq(span!(), $(expr!($arg)),*) };
    ($a:tt) => { Fact::Fact(expr!($a)) };
}

#[macro_export]
macro_rules! facts {
    ($($tree:tt)*) => { Facts(vec![$(fact!($tree)),*]) };
}

#[macro_export]
macro_rules! action {
    ((let $name:ident $value:tt)) => {
        Action::Let(span!(), String::from(stringify!($name)), expr!($value))
    };
    ((set ($f:ident $($x:tt)*) $value:tt)) => {
        Action::Set(span!(), String::from(stringify!($f)), vec![$(expr!($x)),*], expr!($value))
    };
    ((delete ($f:ident $($x:tt)*))) => {
        Action::Change(span!(), Change::Delete, String::from(stringify!($f)), vec![$(expr!($x)),*])
    };
    ((subsume ($f:ident $($x:tt)*))) => {
        Action::Change(span!(), Change::Subsume, String::from(stringify!($f)), vec![$(expr!($x)),*])
    };
    ((union $x:tt $y:tt)) => {
        Action::Union(span!(), expr!($x), expr!($y))
    };
    ((panic $message:literal)) => {
        Action::Panic(span!(), $message.to_owned())
    };
    ($x:tt) => {
        Action::Expr(span!(), expr!($x))
    };
}

#[macro_export]
macro_rules! actions {
    ($($tree:tt)*) => { GenericActions(vec![$(action!($tree)),*]) };
}

/// Add a rule to the e-graph whose right-hand side is made up of actions.
/// ```
/// use egglog::prelude::*;
///
/// let mut egraph = EGraph::default();
/// egraph.parse_and_run_program(
///     None,
///     "
/// (function fib (i64) i64 :no-merge)
/// (set (fib 0) 0)
/// (set (fib 1) 1)
/// (rule (
///     (= f0 (fib x))
///     (= f1 (fib (+ x 1)))
/// ) (
///     (set (fib (+ x 2)) (+ f0 f1))
/// ))
/// (run 10)
///     ",
/// )?;
///
/// let big_number = 20;
///
/// // check that `(fib 20)` is not in the e-graph
/// let results = egraph.query(
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// assert!(results.is_empty());
///
/// let ruleset = "custom_ruleset";
/// add_ruleset(&mut egraph, ruleset)?;
///
/// // add the rule from `build_test_database` to the egraph
/// rule(
///     &mut egraph,
///     ruleset,
///     facts![
///         (= f0 (fib x))
///         (= f1 (fib (+ x 1)))
///     ],
///     actions![
///         (set (fib (+ x 2)) (+ f0 f1))
///     ],
/// )?;
///
/// // run that rule 10 times
/// for _ in 0..10 {
///     run_ruleset(&mut egraph, ruleset)?;
/// }
///
/// // check that `(fib 20)` is now in the e-graph
/// let results = egraph.query(
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// let f: Vec<i64> = results.iter().map(|m| egraph.value_to_base::<i64>(m["f"])).collect();
/// assert_eq!(f, [6765]);
///
/// # Ok::<(), egglog::Error>(())
/// ```
pub fn rule(
    egraph: &mut EGraph,
    ruleset: &str,
    facts: Facts<String, String>,
    actions: Actions,
) -> Result<Vec<CommandOutput>, Error> {
    let rule = Rule {
        span: span!(),
        head: actions,
        body: facts.0,
        name: "".into(),
        ruleset: ruleset.into(),
        eval_mode: RuleEvalMode::Seminaive,
        no_decomp: false,
        include_subsumed: false,
    };

    egraph.run_program(vec![Command::Rule { rule }])
}

#[derive(Clone)]
struct RustRuleRhs<F>
where
    F: for<'a, 'db> Fn(crate::WriteState<'a, 'db>, &[Value]) -> Option<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    name: String,
    inputs: Vec<ArcSort>,
    func: F,
}

impl<F> Primitive for RustRuleRhs<F>
where
    F: for<'a, 'db> Fn(crate::WriteState<'a, 'db>, &[Value]) -> Option<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let sorts: Vec<_> = self
            .inputs
            .iter()
            .chain(once(&UnitSort.to_arcsort()))
            .cloned()
            .collect();
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
}

impl<F> WritePrim for RustRuleRhs<F>
where
    F: for<'a, 'db> Fn(crate::WriteState<'a, 'db>, &[Value]) -> Option<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    fn apply<'a, 'db>(&self, state: crate::WriteState<'a, 'db>, values: &[Value]) -> Option<Value> {
        let unit = state.base_values().get(());
        (self.func)(state, values)?;
        Some(unit)
    }
}

/// Add a rule to the e-graph whose right-hand side is a Rust callback.
/// ```
/// use egglog::prelude::*;
///
/// let mut egraph = EGraph::default();
/// egraph.parse_and_run_program(
///     None,
///     "
/// (function fib (i64) i64 :no-merge)
/// (set (fib 0) 0)
/// (set (fib 1) 1)
/// (rule (
///     (= f0 (fib x))
///     (= f1 (fib (+ x 1)))
/// ) (
///     (set (fib (+ x 2)) (+ f0 f1))
/// ))
/// (run 10)
///     ",
/// )?;
///
/// let big_number = 20;
///
/// // check that `(fib 20)` is not in the e-graph
/// let results = egraph.query(
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// assert!(results.is_empty());
///
/// let ruleset = "custom_ruleset";
/// add_ruleset(&mut egraph, ruleset)?;
///
/// // add the rule from `build_test_database` to the egraph
/// rust_rule(
///     &mut egraph,
///     "fib_rule",
///     ruleset,
///     vars![x: i64, f0: i64, f1: i64],
///     facts![
///         (= f0 (fib x))
///         (= f1 (fib (+ x 1)))
///     ],
///     move |mut ctx, values| {
///         let [x, f0, f1] = values else { unreachable!() };
///         let x = ctx.value_to_base::<i64>(*x);
///         let f0 = ctx.value_to_base::<i64>(*f0);
///         let f1 = ctx.value_to_base::<i64>(*f1);
///
///         let y = ctx.base_to_value::<i64>(x + 2);
///         let f2 = ctx.base_to_value::<i64>(f0 + f1);
///         ctx.set("fib", (y,), f2);
///
///         Some(())
///     },
/// )?;
///
/// // run that rule 10 times
/// for _ in 0..10 {
///     run_ruleset(&mut egraph, ruleset)?;
/// }
///
/// // check that `(fib 20)` is now in the e-graph
/// let results = egraph.query(
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// let f: Vec<i64> = results.iter().map(|m| egraph.value_to_base::<i64>(m["f"])).collect();
/// assert_eq!(f, [6765]);
///
/// # Ok::<(), egglog::Error>(())
/// ```
pub fn rust_rule(
    egraph: &mut EGraph,
    rule_name: &str,
    ruleset: &str,
    vars: &[(&str, ArcSort)],
    facts: Facts<String, String>,
    func: impl for<'a, 'db> Fn(crate::WriteState<'a, 'db>, &[Value]) -> Option<()>
    + Clone
    + Send
    + Sync
    + 'static,
) -> Result<Vec<CommandOutput>, Error> {
    if egraph.are_proofs_enabled() {
        return Err(Error::ProofsIncompatibleApi {
            api: "rust_rule",
            reason: "the rule's RHS is a Rust closure with no proof-encoding validator,\n\
                     so the proof checker cannot verify what it does.",
        });
    }
    let prim_name = egraph.parser.symbol_gen.fresh("rust_rule_prim");
    egraph.add_write_primitive(
        RustRuleRhs {
            name: prim_name.clone(),
            inputs: vars.iter().map(|(_, s)| s.clone()).collect(),
            func,
        },
        None,
    );

    let rule = Rule {
        span: span!(),
        head: GenericActions(vec![GenericAction::Expr(
            span!(),
            exprs::call(
                &prim_name,
                vars.iter().map(|(v, _)| exprs::var(v)).collect(),
            ),
        )]),
        body: facts.0,
        name: egraph.parser.symbol_gen.fresh(rule_name),
        ruleset: ruleset.into(),
        eval_mode: RuleEvalMode::Seminaive,
        no_decomp: false,
        include_subsumed: false,
    };

    egraph.run_program(vec![Command::Rule { rule }])
}

#[derive(Clone)]
struct RustRuleFullRhs<F>
where
    F: for<'a, 'db> Fn(crate::FullState<'a, 'db>, &[Value]) -> Option<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    name: String,
    inputs: Vec<ArcSort>,
    func: F,
}

impl<F> Primitive for RustRuleFullRhs<F>
where
    F: for<'a, 'db> Fn(crate::FullState<'a, 'db>, &[Value]) -> Option<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let sorts: Vec<_> = self
            .inputs
            .iter()
            .chain(once(&UnitSort.to_arcsort()))
            .cloned()
            .collect();
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
}

impl<F> crate::FullPrim for RustRuleFullRhs<F>
where
    F: for<'a, 'db> Fn(crate::FullState<'a, 'db>, &[Value]) -> Option<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    fn apply<'a, 'db>(&self, state: crate::FullState<'a, 'db>, values: &[Value]) -> Option<Value> {
        let unit = state.base_values().get(());
        (self.func)(state, values)?;
        Some(unit)
    }
}

/// Like [`rust_rule`], but the action callback receives a [`FullState`]
/// — it can read tables (`ctx.lookup`) in addition to writing them.
/// Action callbacks of `FullPrim` are only valid in `Context::Full`,
/// so this helper marks the generated rule `:naive`: the body matches
/// against the entire database every iteration instead of using the
/// seminaive delta. Use this when the action genuinely needs to look
/// up rows; prefer [`rust_rule`] when the data can be bound via the
/// matcher in the rule body.
pub fn rust_rule_full(
    egraph: &mut EGraph,
    rule_name: &str,
    ruleset: &str,
    vars: &[(&str, ArcSort)],
    facts: Facts<String, String>,
    func: impl for<'a, 'db> Fn(crate::FullState<'a, 'db>, &[Value]) -> Option<()>
    + Clone
    + Send
    + Sync
    + 'static,
) -> Result<Vec<CommandOutput>, Error> {
    if egraph.are_proofs_enabled() {
        return Err(Error::ProofsIncompatibleApi {
            api: "rust_rule_full",
            reason: "the rule's RHS is a Rust closure with no proof-encoding validator,\n\
                     so the proof checker cannot verify what it does.",
        });
    }
    let prim_name = egraph.parser.symbol_gen.fresh("rust_rule_full_prim");
    egraph.add_full_primitive(
        RustRuleFullRhs {
            name: prim_name.clone(),
            inputs: vars.iter().map(|(_, s)| s.clone()).collect(),
            func,
        },
        None,
    );

    let rule = Rule {
        span: span!(),
        head: GenericActions(vec![GenericAction::Expr(
            span!(),
            exprs::call(
                &prim_name,
                vars.iter().map(|(v, _)| exprs::var(v)).collect(),
            ),
        )]),
        body: facts.0,
        name: egraph.parser.symbol_gen.fresh(rule_name),
        ruleset: ruleset.into(),
        // FullPrim actions require `Context::Full`; use the safe whole-database
        // `:naive` path (`:unsafe-seminaive` also gets `Full` but is unsafe).
        eval_mode: RuleEvalMode::Naive,
        no_decomp: false,
        include_subsumed: false,
    };

    egraph.run_program(vec![Command::Rule { rule }])
}

/// Declare a new sort.
pub fn add_sort(egraph: &mut EGraph, name: &str) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::Sort {
        span: span!(),
        name: name.to_owned(),
        presort_and_args: None,
        uf: None,
        proof_func: None,
        container_rebuild: None,
        proof_constructors: None,
        unionable: true,
    }])
}

/// Declare a new function table.
pub fn add_function(
    egraph: &mut EGraph,
    name: &str,
    schema: Schema,
    merge: Option<GenericExpr<String, String>>,
) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::Function {
        span: span!(),
        name: name.to_owned(),
        schema,
        merge,
        hidden: false,
        let_binding: false,
        term_constructor: None,
        unextractable: false,
    }])
}

/// Declare a new constructor table.
pub fn add_constructor(
    egraph: &mut EGraph,
    name: &str,
    schema: Schema,
    cost: Option<DefaultCost>,
    unextractable: bool,
) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::Constructor {
        span: span!(),
        name: name.to_owned(),
        schema,
        cost,
        unextractable,
        hidden: false,
        let_binding: false,
        term_constructor: None,
    }])
}

/// Declare a new relation table.
pub fn add_relation(
    egraph: &mut EGraph,
    name: &str,
    inputs: Vec<String>,
) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::Relation {
        span: span!(),
        name: name.to_owned(),
        inputs,
    }])
}

/// Adds sorts and constructor tables to the database.
#[macro_export]
macro_rules! datatype {
    ($egraph:expr, (datatype $sort:ident $(($name:ident $($args:ident)* $(:cost $cost:expr)?))*)) => {
        add_sort($egraph, stringify!($sort))?;
        $(add_constructor(
            $egraph,
            stringify!($name),
            Schema {
                input: vec![$(stringify!($args).to_owned()),*],
                output: stringify!($sort).to_owned(),
            },
            [$($cost)*].first().copied(),
            false,
        )?;)*
    };
}

/// A "default" implementation of [`Sort`] for simple types
/// which just want to put some data in the e-graph. If you
/// implement this trait, do not implement `Sort` or
/// `ContainerSort. Use `add_base_sort` to register base
/// sorts with the `EGraph`. See `Sort` for documentation
/// of the methods. Do not override `to_arcsort`.
pub trait BaseSort: Any + Send + Sync + Debug {
    type Base: BaseValue;
    fn name(&self) -> &str;
    fn register_primitives(&self, _eg: &mut EGraph) {}
    fn reconstruct_termdag(&self, _: &BaseValues, _: Value, _: &mut TermDag) -> TermId;

    fn to_arcsort(self) -> ArcSort
    where
        Self: Sized,
    {
        Arc::new(BaseSortImpl(self))
    }
}

#[derive(Debug)]
struct BaseSortImpl<T: BaseSort>(T);

impl<T: BaseSort> Sort for BaseSortImpl<T> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Base(backend.base_values().get_ty::<T::Base>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.base_values_mut().register_type::<T::Base>();
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<T::Base>())
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        self.0.register_primitives(eg)
    }

    /// Reconstruct a leaf base value in a TermDag
    fn reconstruct_termdag_base(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        self.0.reconstruct_termdag(base_values, value, termdag)
    }
}

/// A "default" implementation of [`Sort`] for types which
/// just want to store a pure data structure in the e-graph.
/// If you implement this trait, do not implement `Sort` or
/// `BaseSort`. Use `add_container_sort` to register container
/// sorts with the `EGraph`. See `Sort` for documentation
/// of the methods. Do not override `to_arcsort`.
pub trait ContainerSort: Any + Send + Sync + Debug {
    type Container: ContainerValue;
    fn name(&self) -> &str;
    fn is_eq_container_sort(&self) -> bool;
    fn inner_sorts(&self) -> Vec<ArcSort>;
    fn inner_values(&self, _: &ContainerValues, _: Value) -> Vec<(ArcSort, Value)>;
    fn register_primitives(&self, _eg: &mut EGraph) {}
    fn reconstruct_termdag(
        &self,
        _: &ContainerValues,
        _: Value,
        _: &mut TermDag,
        _: Vec<TermId>,
    ) -> TermId;
    fn serialized_name(&self, container_values: &ContainerValues, value: Value) -> String;

    /// Optional: a container that supports proofs returns its canonical
    /// constructor head and the validator that canonicalizes its term form (e.g.
    /// `set-of` sorts and dedups). `None` (the default) means proofs are
    /// unsupported for this container. See [`Sort::rebuild_container_normalizer`].
    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        None
    }

    fn to_arcsort(self) -> ArcSort
    where
        Self: Sized,
    {
        Arc::new(ContainerSortImpl(self))
    }
}

#[derive(Debug)]
struct ContainerSortImpl<T: ContainerSort>(T);

impl<T: ContainerSort> Sort for ContainerSortImpl<T> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<T::Container>();
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<T::Container>())
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        self.0.inner_sorts()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        self.0.inner_values(container_values, value)
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.0.is_eq_container_sort()
    }

    fn serialized_name(&self, container_values: &ContainerValues, value: Value) -> String {
        self.0.serialized_name(container_values, value)
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        self.0.register_primitives(eg);
    }

    fn reconstruct_termdag_container(
        &self,
        container_values: &ContainerValues,
        value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        self.0
            .reconstruct_termdag(container_values, value, termdag, element_terms)
    }

    fn rebuild_container_normalizer(&self) -> Option<(String, PrimitiveValidator)> {
        self.0.rebuild_container_normalizer()
    }
}

/// Add a [`BaseSort`] to the e-graph
pub fn add_base_sort(
    egraph: &mut EGraph,
    base_sort: impl BaseSort,
    span: Span,
) -> Result<(), TypeError> {
    egraph.add_sort(BaseSortImpl(base_sort), span)
}

pub fn add_container_sort(
    egraph: &mut EGraph,
    container_sort: impl ContainerSort,
    span: Span,
) -> Result<(), TypeError> {
    egraph.add_sort(ContainerSortImpl(container_sort), span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_database() -> Result<EGraph, Error> {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(
            None,
            "
(function fib (i64) i64 :no-merge)
(set (fib 0) 0)
(set (fib 1) 1)
(rule (
    (= f0 (fib x))
    (= f1 (fib (+ x 1)))
) (
    (set (fib (+ x 2)) (+ f0 f1))
))
(run 10)
        ",
        )?;
        Ok(egraph)
    }

    #[test]
    fn rust_api_query() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        let results = egraph.query(
            vars![x: i64, y: i64],
            facts![
                (= (fib x) y)
                (= y 13)
            ],
        )?;

        assert_eq!(results.len(), 1);
        assert_eq!(egraph.value_to_base::<i64>(results[0]["x"]), 7);
        assert_eq!(egraph.value_to_base::<i64>(results[0]["y"]), 13);

        Ok(())
    }

    #[test]
    fn rust_api_rule() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        let big_number = 20;

        // check that `(fib 20)` is not in the e-graph
        let results = egraph.query(
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        assert!(results.is_empty());

        let ruleset = "custom_ruleset";
        add_ruleset(&mut egraph, ruleset)?;

        // add the rule from `build_test_database` to the egraph
        rule(
            &mut egraph,
            ruleset,
            facts![
                (= f0 (fib x))
                (= f1 (fib (+ x 1)))
            ],
            actions![
                (set (fib (+ x 2)) (+ f0 f1))
            ],
        )?;

        // run that rule 10 times
        for _ in 0..10 {
            run_ruleset(&mut egraph, ruleset)?;
        }

        // check that `(fib 20)` is now in the e-graph
        let results = egraph.query(
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        assert_eq!(results.len(), 1);
        assert_eq!(egraph.value_to_base::<i64>(results[0]["f"]), 6765);

        Ok(())
    }

    #[test]
    fn rust_api_macros() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        datatype!(&mut egraph, (datatype Expr (One) (Two Expr Expr :cost 10)));

        let ruleset = "custom_ruleset";
        add_ruleset(&mut egraph, ruleset)?;

        rule(
            &mut egraph,
            ruleset,
            facts![
                (fib 5)
                (fib x)
                (= f1 (fib (+ x 1)))
                (= 3 (unquote exprs::int(1 + 2)))
            ],
            actions![
                (let y (+ x 2))
                (set (fib (+ x 2)) (+ f1 f1))
                (delete (fib 0))
                (subsume (Two (One) (One)))
                (union (One) (Two (One) (One)))
                (panic "message")
                (+ 6 87)
            ],
        )?;

        Ok(())
    }

    #[test]
    fn rust_api_rust_rule() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        let big_number = 20;

        // check that `(fib 20)` is not in the e-graph
        let results = egraph.query(
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        assert!(results.is_empty());

        let ruleset = "custom_ruleset";
        add_ruleset(&mut egraph, ruleset)?;

        // add the rule from `build_test_database` to the egraph
        rust_rule(
            &mut egraph,
            "demo_rule",
            ruleset,
            vars![x: i64, f0: i64, f1: i64],
            facts![
                (= f0 (fib x))
                (= f1 (fib (+ x 1)))
            ],
            move |mut ctx, values| {
                let [x, f0, f1] = values else { unreachable!() };
                let x = ctx.value_to_base::<i64>(*x);
                let f0 = ctx.value_to_base::<i64>(*f0);
                let f1 = ctx.value_to_base::<i64>(*f1);

                ctx.set("fib", (x + 2,), f0 + f1).ok()?;

                Some(())
            },
        )?;

        // run that rule 10 times
        for _ in 0..10 {
            run_ruleset(&mut egraph, ruleset)?;
        }

        // check that `(fib 20)` is now in the e-graph
        let results = egraph.query(
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        assert_eq!(results.len(), 1);
        assert_eq!(egraph.value_to_base::<i64>(results[0]["f"]), 6765);

        Ok(())
    }
}
