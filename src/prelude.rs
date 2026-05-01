//! This module makes it easier to use `egglog` from Rust.
//! ```
//! use egglog::prelude::*;
//! ```
//!
//! Common entry points:
//!
//! - [`rule`] / [`rust_rule`] — add rules whose RHS is egglog code or a
//!   Rust closure (the closure receives an
//!   [`crate::WriteState`]).
//! - [`query`] — run a one-shot query and read out matches.
//! - [`BaseSort`] / [`ContainerSort`] — declare custom sort types.
//! - [`crate::PrimitiveCommon`] + one of four kind-specific traits
//!   ([`crate::PurePrim`], [`crate::WritePrim`],
//!   [`crate::ReadPrim`], [`crate::FullPrim`]) —
//!   register custom primitives. Each kind names its state wrapper
//!   directly; pick the trait matching what the body actually needs
//!   and register via [`crate::EGraph::add_pure_primitive`]
//!   etc. The Rust type checker enforces that the body only uses
//!   methods the chosen state allows.
//! - The [`add_primitive!`] / [`add_primitive_with_validator!`] /
//!   [`add_literal_prim!`] macros (re-exported via `egglog::*`) cover
//!   the "pure native function" case ergonomically — they generate
//!   `PurePrim` impls and pull base-value conversions for you.

use crate::*;
use std::any::{Any, TypeId};

// Re-exports in `prelude` for convenience.
pub use egglog::ast::{Action, Fact, Facts, GenericActions, RustSpan, Span};
pub use egglog::sort::{BigIntSort, BigRatSort, BoolSort, F64Sort, I64Sort, StringSort, UnitSort};
pub use egglog::{CommandMacro, CommandMacroRegistry};
pub use egglog::{Core, FullState, PureState, ReadState, Write, WriteState};
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
/// let results = query(
///     &mut egraph,
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// assert!(results.iter().next().is_none());
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
/// let results = query(
///     &mut egraph,
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// let y = egraph.base_to_value::<i64>(6765);
/// let results: Vec<_> = results.iter().collect();
/// assert_eq!(results, [[y]]);
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

impl<F> PrimitiveCommon for RustRuleRhs<F>
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
/// let results = query(
///     &mut egraph,
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// assert!(results.iter().next().is_none());
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
///         ctx.insert("fib", [y, f2].into_iter());
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
/// let results = query(
///     &mut egraph,
///     vars![f: i64],
///     facts![(= (fib (unquote exprs::int(big_number))) f)],
/// )?;
///
/// let y = egraph.base_to_value::<i64>(6765);
/// let results: Vec<_> = results.iter().collect();
/// assert_eq!(results, [[y]]);
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
    };

    egraph.run_program(vec![Command::Rule { rule }])
}

/// The result of a query.
pub struct QueryResult {
    rows: usize,
    cols: usize,
    data: Vec<Value>,
}

impl QueryResult {
    /// Get an iterator over the query results,
    /// where each match is a `&[Value]` in the same order
    /// as the `vars` that were passed to `query`.
    pub fn iter(&self) -> impl Iterator<Item = &[Value]> {
        assert!(self.cols > 0, "no vars; use `any_matches` instead");
        assert!(self.data.len().is_multiple_of(self.cols));
        self.data.chunks_exact(self.cols)
    }

    /// Check if any matches were returned at all.
    pub fn any_matches(&self) -> bool {
        self.rows > 0
    }
}

/// Run a query over the database.
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
/// let results = query(
///     &mut egraph,
///     vars![x: i64, y: i64],
///     facts![
///         (= (fib x) y)
///         (= y 13)
///     ],
/// )?;
///
/// let x = egraph.base_to_value::<i64>(7);
/// let y = egraph.base_to_value::<i64>(13);
/// let results: Vec<_> = results.iter().collect();
/// assert_eq!(results, [[x, y]]);
///
/// # Ok::<(), egglog::Error>(())
/// ```
pub fn query(
    egraph: &mut EGraph,
    vars: &[(&str, ArcSort)],
    facts: Facts<String, String>,
) -> Result<QueryResult, Error> {
    use std::sync::{Arc, Mutex};

    let results = Arc::new(Mutex::new(QueryResult {
        rows: 0,
        cols: vars.len(),
        data: Vec::new(),
    }));
    let results_weak = Arc::downgrade(&results);

    let ruleset = egraph.parser.symbol_gen.fresh("query_ruleset");
    add_ruleset(egraph, &ruleset)?;

    rust_rule(egraph, "query", &ruleset, vars, facts, move |_, values| {
        let arc = results_weak.upgrade().unwrap();
        let mut results = arc.lock().unwrap();
        results.rows += 1;
        results.data.extend(values);
        Some(())
    })?;

    run_ruleset(egraph, &ruleset)?;

    let ruleset = egraph.rulesets.swap_remove(&ruleset).unwrap();

    let Ruleset::Rules(rules) = ruleset else {
        unreachable!()
    };
    assert_eq!(rules.len(), 1);
    let rule = rules.into_iter().next().unwrap().1;
    egraph.backend.free_rule(rule.1);

    let Some(mutex) = Arc::into_inner(results) else {
        panic!("results_weak.upgrade() was not dropped");
    };
    Ok(mutex.into_inner().unwrap())
}

/// Declare a new sort.
pub fn add_sort(egraph: &mut EGraph, name: &str) -> Result<Vec<CommandOutput>, Error> {
    egraph.run_program(vec![Command::Sort {
        span: span!(),
        name: name.to_owned(),
        presort_and_args: None,
        uf: None,
        proof_func: None,
        unionable: true,
    }])
}

/// A builder for declaring tables (`function`s, `constructor`s, and `relation`s)
/// without manually constructing a [`Schema`].
///
/// Created via [`EGraph::declare`]. Chain `.input(sort_name)` for each input
/// column, then `.output(sort_name)` for the output column (skip for relations),
/// and finish with `.function(...)`, `.constructor(...)`, or `.relation()`.
///
/// ```
/// use egglog::prelude::*;
///
/// let mut egraph = EGraph::default();
/// egraph.declare("f").input("i64").output("i64").function(None)?;
/// egraph.declare("R").input("i64").input("i64").relation()?;
/// # Ok::<(), egglog::Error>(())
/// ```
///
/// # Missing output
/// Calling `.function(...)` or `.constructor(...)` without first calling
/// `.output(...)` panics with a clear message. Output is required to construct
/// a `Schema`, and panicking gives a tight, immediately-actionable signal to
/// the caller (no new `Error` variant is needed). `.relation()` does not
/// require an output and ignores any that was set.
pub struct DeclareTable<'a> {
    eg: &'a mut EGraph,
    name: String,
    inputs: Vec<String>,
    output: Option<String>,
}

impl<'a> DeclareTable<'a> {
    /// Append an input column with the given sort name.
    pub fn input(mut self, sort_name: &str) -> Self {
        self.inputs.push(sort_name.to_owned());
        self
    }

    /// Set the output column's sort name. Calling this multiple times keeps
    /// the most-recent value.
    pub fn output(mut self, sort_name: &str) -> Self {
        self.output = Some(sort_name.to_owned());
        self
    }

    /// Type-level shortcut for [`DeclareTable::input`]: maps a Rust primitive
    /// type to the corresponding egglog sort name (e.g. `i64` -> `"i64"`,
    /// `bool` -> `"bool"`). Only supported for built-in primitives via the
    /// [`BaseSortName`] trait. For user-defined sorts use [`DeclareTable::input`].
    pub fn input_base<T: BaseSortName>(self) -> Self {
        self.input(T::SORT_NAME)
    }

    /// Type-level shortcut for [`DeclareTable::output`]. See
    /// [`DeclareTable::input_base`].
    pub fn output_base<T: BaseSortName>(self) -> Self {
        self.output(T::SORT_NAME)
    }

    fn schema_or_panic(&self) -> Schema {
        let output = self.output.clone().unwrap_or_else(|| {
            panic!(
                "DeclareTable: no output sort set for `{}`; call .output(\"<sort>\") \
                 (or use .relation() if no output column is intended)",
                self.name
            )
        });
        Schema {
            input: self.inputs.clone(),
            output,
        }
    }

    /// Finish declaring this table as a `function` with the given (optional)
    /// merge expression. Panics if `.output(...)` was not called.
    pub fn function(
        self,
        merge: Option<GenericExpr<String, String>>,
    ) -> Result<Vec<CommandOutput>, Error> {
        let schema = self.schema_or_panic();
        self.eg.run_program(vec![Command::Function {
            span: span!(),
            name: self.name,
            schema,
            merge,
            hidden: false,
            let_binding: false,
        }])
    }

    /// Finish declaring this table as a `constructor`. Panics if
    /// `.output(...)` was not called.
    pub fn constructor(
        self,
        cost: Option<DefaultCost>,
        unextractable: bool,
    ) -> Result<Vec<CommandOutput>, Error> {
        let schema = self.schema_or_panic();
        self.eg.run_program(vec![Command::Constructor {
            span: span!(),
            name: self.name,
            schema,
            cost,
            unextractable,
            hidden: false,
            let_binding: false,
            term_constructor: None,
        }])
    }

    /// Finish declaring this table as a `relation`. Any output sort previously
    /// set via [`DeclareTable::output`] is ignored.
    pub fn relation(self) -> Result<Vec<CommandOutput>, Error> {
        self.eg.run_program(vec![Command::Relation {
            span: span!(),
            name: self.name,
            inputs: self.inputs,
        }])
    }
}

/// Maps a Rust primitive type to its egglog sort name. Used by
/// [`DeclareTable::input_base`] / [`DeclareTable::output_base`] for type-level
/// convenience when declaring tables. Implemented for the built-in primitive
/// types only; user-defined sorts should use the string-based methods.
pub trait BaseSortName {
    const SORT_NAME: &'static str;
}

impl BaseSortName for i64 {
    const SORT_NAME: &'static str = "i64";
}
impl BaseSortName for bool {
    const SORT_NAME: &'static str = "bool";
}
impl BaseSortName for () {
    const SORT_NAME: &'static str = "Unit";
}
impl BaseSortName for egglog::sort::F {
    const SORT_NAME: &'static str = "f64";
}
impl BaseSortName for egglog::sort::S {
    const SORT_NAME: &'static str = "String";
}

impl EGraph {
    /// Begin declaring a new table named `name`. See [`DeclareTable`] for
    /// usage examples.
    pub fn declare(&mut self, name: &str) -> DeclareTable<'_> {
        DeclareTable {
            eg: self,
            name: name.to_owned(),
            inputs: Vec::new(),
            output: None,
        }
    }
}

/// Declare a new function table.
#[deprecated(note = "Use `eg.declare(name).input(...).output(...).function(...)` instead")]
pub fn add_function(
    egraph: &mut EGraph,
    name: &str,
    schema: Schema,
    merge: Option<GenericExpr<String, String>>,
) -> Result<Vec<CommandOutput>, Error> {
    let mut b = egraph.declare(name);
    for inp in &schema.input {
        b = b.input(inp);
    }
    b = b.output(&schema.output);
    b.function(merge)
}

/// Declare a new constructor table.
#[deprecated(note = "Use `eg.declare(name).input(...).output(...).constructor(...)` instead")]
pub fn add_constructor(
    egraph: &mut EGraph,
    name: &str,
    schema: Schema,
    cost: Option<DefaultCost>,
    unextractable: bool,
) -> Result<Vec<CommandOutput>, Error> {
    let mut b = egraph.declare(name);
    for inp in &schema.input {
        b = b.input(inp);
    }
    b = b.output(&schema.output);
    b.constructor(cost, unextractable)
}

/// Declare a new relation table.
#[deprecated(note = "Use `eg.declare(name).input(...).input(...).relation()` instead")]
pub fn add_relation(
    egraph: &mut EGraph,
    name: &str,
    inputs: Vec<String>,
) -> Result<Vec<CommandOutput>, Error> {
    let mut b = egraph.declare(name);
    for inp in &inputs {
        b = b.input(inp);
    }
    b.relation()
}

/// Adds sorts and constructor tables to the database.
#[macro_export]
macro_rules! datatype {
    ($egraph:expr, (datatype $sort:ident $(($name:ident $($args:ident)* $(:cost $cost:expr)?))*)) => {
        add_sort($egraph, stringify!($sort))?;
        $({
            let mut __b = $egraph.declare(stringify!($name));
            $(__b = __b.input(stringify!($args));)*
            __b.output(stringify!($sort))
                .constructor([$($cost)*].first().copied(), false)?;
        })*
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

        let results = query(
            &mut egraph,
            vars![x: i64, y: i64],
            facts![
                (= (fib x) y)
                (= y 13)
            ],
        )?;

        let x = egraph.backend.base_values().get::<i64>(7);
        let y = egraph.backend.base_values().get::<i64>(13);
        assert_eq!(results.data, [x, y]);

        Ok(())
    }

    #[test]
    fn rust_api_rule() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        let big_number = 20;

        // check that `(fib 20)` is not in the e-graph
        let results = query(
            &mut egraph,
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        assert!(results.data.is_empty());

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
        let results = query(
            &mut egraph,
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        let y = egraph.backend.base_values().get::<i64>(6765);
        assert_eq!(results.data, [y]);

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
        let results = query(
            &mut egraph,
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        assert!(results.data.is_empty());

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

                let y = ctx.base_to_value::<i64>(x + 2);
                let f2 = ctx.base_to_value::<i64>(f0 + f1);
                ctx.insert("fib", [y, f2].into_iter());

                Some(())
            },
        )?;

        // run that rule 10 times
        for _ in 0..10 {
            run_ruleset(&mut egraph, ruleset)?;
        }

        // check that `(fib 20)` is now in the e-graph
        let results = query(
            &mut egraph,
            vars![f: i64],
            facts![(= (fib (unquote exprs::int(big_number))) f)],
        )?;

        let y = egraph.backend.base_values().get::<i64>(6765);
        assert_eq!(results.data, [y]);

        Ok(())
    }
}
