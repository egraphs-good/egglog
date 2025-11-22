//! This module makes it easier to use `egglog` from Rust.
//! It is intended to be imported fully.
//! ```
//! use egglog::prelude::*;
//! ```
//! See also [`rule`], [`rust_rule`], [`query`], [`BaseSort`],
//! and [`ContainerSort`].

use crate::*;
use std::any::{Any, TypeId};

// Re-exports in `prelude` for convenience.
pub use egglog::ast::{Action, Fact, Facts, GenericActions, RustSpan, Span};
pub use egglog::egraph_operations::QueryMatch;
pub use egglog::sort::{BigIntSort, BigRatSort, BoolSort, F64Sort, I64Sort, StringSort, UnitSort};
pub use egglog::{EGraph, span};
pub use egglog::{action, actions, datatype, expr, fact, facts, sort, vars};

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
    let mut rule = Rule {
        span: span!(),
        head: actions,
        body: facts.0,
        name: "".into(),
        ruleset: ruleset.into(),
    };

    rule.name = format!("{rule:?}");

    egraph.run_program(vec![Command::Rule { rule }])
}

/// A wrapper around an `ExecutionState` for rules that are written in Rust.
/// See the [`rust_rule`] documentation for an example of how to use this.
pub struct RustRuleContext<'a, 'b> {
    exec_state: &'a mut ExecutionState<'b>,
    union_action: egglog_bridge::UnionAction,
    table_actions: HashMap<String, egglog_bridge::TableAction>,
    panic_id: ExternalFunctionId,
}

impl RustRuleContext<'_, '_> {
    /// Convert from an egglog value to a Rust type.
    pub fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.exec_state.base_values().unwrap::<T>(x)
    }

    /// Convert from an egglog value to reference of Rust container type.
    ///
    /// See [`EGraph::value_to_container`].
    pub fn value_to_container<T: ContainerValue>(
        &mut self,
        x: Value,
    ) -> Option<impl Deref<Target = T>> {
        self.exec_state.container_values().get_val::<T>(x)
    }

    /// Convert from a Rust type to an egglog value.
    pub fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.exec_state.base_values().get::<T>(x)
    }

    /// Convert from a Rust container type to an egglog value.
    pub fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        self.exec_state
            .container_values()
            .register_val::<T>(x, self.exec_state)
    }

    fn get_table_action(&self, table: &str) -> egglog_bridge::TableAction {
        self.table_actions[table].clone()
    }

    /// Do a table lookup. This is potentially a mutable operation!
    /// For more information, see `egglog_bridge::TableAction::lookup`.
    pub fn lookup(&mut self, table: &str, key: &[Value]) -> Option<Value> {
        self.get_table_action(table).lookup(self.exec_state, key)
    }

    /// Union two values in the e-graph.
    /// For more information, see `egglog_bridge::UnionAction::union`.
    pub fn union(&mut self, x: Value, y: Value) {
        self.union_action.union(self.exec_state, x, y)
    }

    /// Insert a row into a table.
    /// For more information, see `egglog_bridge::TableAction::insert`.
    pub fn insert(&mut self, table: &str, row: impl Iterator<Item = Value>) {
        self.get_table_action(table).insert(self.exec_state, row)
    }

    /// Remove a row from a table.
    /// For more information, see `egglog_bridge::TableAction::remove`.
    pub fn remove(&mut self, table: &str, key: &[Value]) {
        self.get_table_action(table).remove(self.exec_state, key)
    }

    /// Subsume a row in a table.
    /// For more information, see `egglog_bridge::TableAction::subsume`.
    pub fn subsume(&mut self, table: &str, key: &[Value]) {
        self.get_table_action(table)
            .subsume(self.exec_state, key.iter().copied())
    }

    /// Panic.
    /// You should also return `None` from your callback if you call
    /// this function, which this function hopefully makes easier by
    /// always returning `None` so that you can use `?`.
    pub fn panic(&mut self) -> Option<()> {
        self.exec_state.call_external_func(self.panic_id, &[]);
        None
    }
}

#[derive(Clone)]
struct RustRuleRhs<F: Fn(&mut RustRuleContext, &[Value]) -> Option<()>> {
    name: String,
    inputs: Vec<ArcSort>,
    union_action: egglog_bridge::UnionAction,
    table_actions: HashMap<String, egglog_bridge::TableAction>,
    panic_id: ExternalFunctionId,
    func: F,
}

impl<F: Fn(&mut RustRuleContext, &[Value]) -> Option<()>> Primitive for RustRuleRhs<F> {
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

    fn apply(&self, exec_state: &mut ExecutionState, values: &[Value]) -> Option<Value> {
        let mut context = RustRuleContext {
            exec_state,
            union_action: self.union_action,
            table_actions: self.table_actions.clone(),
            panic_id: self.panic_id,
        };
        (self.func)(&mut context, values)?;
        Some(exec_state.base_values().get(()))
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
///     move |ctx, values| {
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
    func: impl Fn(&mut RustRuleContext, &[Value]) -> Option<()> + Clone + Send + Sync + 'static,
) -> Result<Vec<CommandOutput>, Error> {
    let prim_name = egraph.parser.symbol_gen.fresh("rust_rule_prim");
    let panic_id = egraph.backend.new_panic(format!("{prim_name}_panic"));
    egraph.add_primitive(RustRuleRhs {
        name: prim_name.clone(),
        inputs: vars.iter().map(|(_, s)| s.clone()).collect(),
        union_action: egglog_bridge::UnionAction::new(&egraph.backend),
        table_actions: egraph
            .functions
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    egglog_bridge::TableAction::new(&egraph.backend, v.backend_id),
                )
            })
            .collect(),
        panic_id,
        func,
    });

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
        assert!(self.data.len() % self.cols == 0);
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
    egraph.run_program(vec![Command::Sort(span!(), name.to_owned(), None)])
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
    fn reconstruct_termdag(&self, _: &BaseValues, _: Value, _: &mut TermDag) -> Term;

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
    ) -> Term {
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
        _: Vec<Term>,
    ) -> Term;
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
        element_terms: Vec<Term>,
    ) -> Term {
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
            move |ctx, values| {
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
