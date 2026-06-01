//! User-facing execution state wrappers.
//!
//! Four wrappers around `core_relations::ExecutionState` expose different
//! subsets of the database API based on the context in which a primitive runs:
//!
//! | Wrapper       | DB reads | DB writes | Used for                                 |
//! |---------------|----------|-----------|------------------------------------------|
//! | `PureState`   | no       | no        | rule LHS (seminaive on)                  |
//! | `WriteState`  | no       | yes       | rule RHS (seminaive on)                  |
//! | `ReadState`   | yes      | no        | top-level query-shaped commands          |
//! | `FullState`   | yes      | yes       | top-level action-shaped commands, `eval` |
//!
//! Methods come from sealed capability traits implemented on each
//! wrapper:
//!
//! - [`Core`] — base values, counters, container interning, conversion
//!   sugar. Implemented for all four wrappers.
//! - [`Read`] — name-indexed table lookup (`state.lookup("name", &[…])`).
//!   Implemented for [`ReadState`] and [`FullState`].
//! - [`Write`] — name-indexed writes (`set`/`add_node`/`remove`/
//!   `subsume`/`union`/`panic`). Implemented for [`WriteState`] and
//!   [`FullState`].
//!
//! Privileged seams (`call_external_func`, `table_lookup`, raw
//! `&mut ExecutionState`) used by the `FunctionContainer` higher-order
//! dispatch live on the crate-private [`Internal`] trait. User code
//! cannot reach them.
//!
//! [`PurePrim`]: crate::PurePrim
//! [`WritePrim`]: crate::WritePrim
//! [`ReadPrim`]: crate::ReadPrim
//! [`FullPrim`]: crate::FullPrim

use std::ops::Deref;

use crate::api::{ApiError, ColumnSort, IntoColumn, IntoRow};
use crate::core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, ExecutionState, ExternalFunctionId,
    Value,
};
use crate::Error;
use egglog_bridge::{ActionRegistry, TableAction, TableKind};

/// The four contexts a primitive may run in, named after the
/// capability profile they grant. Each variant maps 1:1 to one of the
/// state wrappers below: `Pure` ↔ [`PureState`], `Write` ↔ [`WriteState`],
/// `Read` ↔ [`ReadState`], `Full` ↔ [`FullState`]. The egglog
/// typechecker filters primitive definitions by whether they carry a
/// runtime id for the surrounding `Context` at each call site.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, enum_map::Enum)]
pub enum Context {
    /// No DB reads, no DB writes. The body (LHS) of a rule running
    /// under seminaive evaluation: a body read of live state means
    /// the rule won't re-fire when the read row's contents change
    /// in a later iteration; a body write makes no semantic sense.
    Pure,
    /// DB writes allowed, DB reads forbidden. The head (RHS) of a
    /// rule running under seminaive evaluation: same re-firing
    /// concern as `Pure` rules out reads, but staged writes are
    /// fine.
    Write,
    /// DB reads allowed, DB writes forbidden. Top-level query-shaped
    /// commands (`check`, condition evaluation) and the body of a
    /// `:naive` rule. Reads are safe because there is no seminaive
    /// epoch to violate.
    Read,
    /// DB reads and writes both allowed. Top-level action-shaped
    /// commands (`eval`, `let`, action-mode `run-schedule`) and the
    /// head of a `:naive` rule.
    Full,
}

impl Context {
    pub const ALL: [Context; 4] = [Context::Pure, Context::Write, Context::Read, Context::Full];
}

// =====================================================================
// Sealed traits.
//
// These traits are `pub(crate)`: external users cannot bring them
// into scope, so they cannot call the methods defined here even on
// values they have. They give the public capability traits (`Core`,
// `Write`) a way to reach the underlying `ExecutionState` from
// default methods, and carry the privileged seams used by the
// `FunctionContainer` higher-order dispatch.
// =====================================================================

/// Crate-private accessor + privileged-dispatch trait. Required
/// methods are the accessors that every wrapper supplies; default
/// methods are the privileged seams used by the `FunctionContainer`
/// higher-order dispatch.
pub(crate) trait Internal<'a, 'db: 'a>: 'a {
    fn es(&self) -> &ExecutionState<'db>;
    fn es_mut(&mut self) -> &mut ExecutionState<'db>;
    /// The call-site [`Context`] this primitive was invoked from.
    /// Stamped onto the wrapper at construction time by the
    /// `ExternalFunction` wrapper closure; read by
    /// [`Core::apply_function`] to route higher-order dispatch.
    fn ctx(&self) -> Context;

    fn call_external_func(&mut self, id: ExternalFunctionId, args: &[Value]) -> Option<Value> {
        self.es_mut().call_external_func(id, args)
    }
    fn raw_exec_state(&mut self) -> &mut ExecutionState<'db> {
        self.es_mut()
    }
}

/// Sealed accessor for the [`ActionRegistry`]. Implemented by every
/// wrapper that has a registry (`ReadState`, `WriteState`, `FullState`)
/// — the read- and write-side traits both look up `TableAction`s by
/// name through it.
pub(crate) trait RegistrySealed<'a, 'db: 'a>: Internal<'a, 'db> {
    fn registry(&self) -> &ActionRegistry;
}

// =====================================================================
// Public capability traits.
// =====================================================================

/// Core methods available on every state wrapper: base values,
/// counters, container interning, value/base/container conversion sugar.
/// Always seminaive-safe.
#[allow(private_bounds)]
pub trait Core<'a, 'db: 'a>: Internal<'a, 'db> {
    /// Base-value pool (interned primitives like `i64`, `String`, …).
    fn base_values(&self) -> &'a BaseValues {
        self.es().base_values()
    }
    /// Signal that rule execution should stop after this firing.
    fn trigger_early_stop(&self) {
        self.es().trigger_early_stop()
    }
    /// Has someone called `trigger_early_stop`?
    fn should_stop(&self) -> bool {
        self.es().should_stop()
    }
    /// Container values for this EGraph.
    fn container_values(&self) -> &'a ContainerValues {
        self.es().container_values()
    }

    /// Register a container value, returning its interned `Value`.
    fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
        // `container_values()` returns `&'a ContainerValues` — a reference
        // tied to the inner ExecutionState's lifetime, not to `&self` —
        // so it doesn't conflict with the subsequent `&mut` reborrow.
        // Avoiding the clone of `ExecutionState` here matters: this is
        // hot-path code (every container intern goes through it) and
        // the clone copies a non-trivial amount of state.
        let cv = self.container_values();
        let es = self.es_mut();
        cv.register_val(container, es)
    }

    /// Convert an egglog [`Value`] to a Rust base type. Untyped: trust
    /// that `x` belongs to sort `T`.
    fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.es().base_values().unwrap::<T>(x)
    }

    /// Convert a Rust base type to an egglog [`Value`]. Untyped.
    fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.es().base_values().get::<T>(x)
    }

    /// Look up the Rust container behind an egglog [`Value`], if any.
    fn value_to_container<T: ContainerValue>(
        &self,
        x: Value,
    ) -> Option<impl Deref<Target = T> + 'a> {
        self.es().container_values().get_val::<T>(x)
    }

    /// Intern a Rust container into the e-graph and return its
    /// [`Value`]. Sugar over `self.register_container(x)`.
    fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        self.register_container(x)
    }

    /// Dispatch a wrapped `unstable-fn` value. This is the public entry
    /// point for higher-order primitive bodies: the call-site
    /// [`Context`] is stamped onto the state by the registration
    /// wrapper (see [`EGraph::add_pure_primitive`] and the matching
    /// `add_read_primitive` / `add_write_primitive` /
    /// `add_full_primitive`), so the caller can't supply a wrong
    /// context — there is no `ctx` parameter to lie about.
    ///
    /// [`EGraph::add_pure_primitive`]: crate::EGraph::add_pure_primitive
    fn apply_function(
        &mut self,
        fc: &crate::sort::FunctionContainer,
        args: &[Value],
    ) -> Option<Value> {
        let ctx = self.ctx();
        let mut pure = PureState::wrap(self.raw_exec_state(), ctx);
        fc.apply(&mut pure, args)
    }
}

/// Read-side methods — name-indexed table lookup. Implemented for
/// [`ReadState`] and [`FullState`]; *not* for [`PureState`] or
/// [`WriteState`] (a `Write` context body must not depend on live DB
/// state). Returns `None` if the row is absent — never inserts.
///
/// Misuse (wrong table subtype, wrong arity, mismatched column sort)
/// is reported as [`crate::ApiError`] via the method's `Result`.
#[allow(private_bounds)]
pub trait Read<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Look up a function's output value at the given key. Returns
    /// `Ok(None)` if the row is absent.
    ///
    /// **Only valid for `function` tables.** Constructors error;
    /// use [`Read::eclass_of`] for those.
    fn lookup<K: IntoRow, V: BaseValue>(
        &self,
        name: &str,
        key: K,
    ) -> Result<Option<V>, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let bv = self.base_values();
        let key_values = key.into_values(bv);
        Ok(action.lookup(self.es(), &key_values).map(|v| bv.unwrap::<V>(v)))
    }

    /// Look up a constructor's eclass at the given inputs, without
    /// minting a fresh one on miss. Returns `Ok(None)` if absent.
    ///
    /// **Only valid for constructor tables.** Functions error;
    /// use [`Read::lookup`] for those.
    fn eclass_of<K: IntoRow>(
        &self,
        name: &str,
        inputs: K,
    ) -> Result<Option<Value>, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(
            name,
            &action,
            TableKind::Constructor,
            "constructor",
        )?;
        let sorts = inputs.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = inputs.into_values(self.base_values());
        Ok(action.lookup(self.es(), &key_values))
    }

    /// True iff a row with the given key exists in the table. Works
    /// for any subtype — never mints.
    fn contains<K: IntoRow>(
        &self,
        name: &str,
        key: K,
    ) -> Result<bool, Error> {
        let action = lookup_action(self.registry(), name)?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        Ok(action.lookup(self.es(), &key_values).is_some())
    }

    /// Untyped raw-`Value` lookup, escape hatch for code that already
    /// has `&[Value]` and doesn't want to round-trip through
    /// [`Read::lookup`]'s base-value conversion. Skips sort checking.
    fn lookup_raw(&self, name: &str, key: &[Value]) -> Result<Option<Value>, Error> {
        let action = lookup_action(self.registry(), name)?;
        Ok(action.lookup(self.es(), key))
    }

    /// Return the current row count for the named table, or `None` if no table
    /// with that name is registered.
    fn table_size(&self, name: &str) -> Option<usize> {
        self.registry()
            .lookup_table(name)
            .map(|action| action.row_count(self.es()))
    }

    /// Snapshot the registered table names and their current row counts.
    fn table_sizes(&self) -> Vec<(&str, usize)> {
        self.registry().table_sizes(self.es())
    }

    /// Iterate all rows of a named table, returning each one as the
    /// caller-chosen `R: FromRow`.
    ///
    /// Row shape depends on subtype:
    /// - **Function tables** (`(function f (i64) i64 :no-merge)`) —
    ///   rows are `(input..., output)`; `table_rows::<(i64, i64)>("f")`
    ///   yields `(key, value)`.
    /// - **Constructor / relation tables** — rows are
    ///   `(input..., eclass)` where `eclass` is the minted eclass
    ///   [`Value`]. Relations desugar to constructors with a synthetic
    ///   non-unionable eq-sort output column.
    ///
    /// Pass `Vec<Value>` to inspect arbitrary shapes, or use
    /// [`crate::EGraph::query`] to bind only the columns you name.
    fn table_rows<R: crate::api::FromRow>(&self, name: &str) -> Result<Vec<R>, Error> {
        let action = lookup_action(self.registry(), name)?;
        let bv = self.base_values();
        let mut out = Vec::new();
        action.for_each(self.es(), |row| {
            out.push(R::from_values(row.vals, bv));
        });
        Ok(out)
    }
}

/// Action-side write methods — name-indexed inserts/removes/subsumes
/// plus union and panic. Implemented for [`WriteState`] and
/// [`FullState`]; *not* for [`PureState`] or [`ReadState`].
///
/// Misuse (wrong table subtype, wrong arity, mismatched column or
/// output sort, cross-sort union) is reported as [`crate::ApiError`]
/// via the method's `Result`.
#[allow(private_bounds)]
pub trait Write<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Set a function table's value at the given key — mirrors the
    /// egglog `(set (f k) v)` action.
    ///
    /// **Only valid for `function` tables.** Constructors error;
    /// use [`Write::add_node`] for those.
    fn set<K: IntoRow, V: IntoColumn>(
        &mut self,
        name: &str,
        key: K,
        value: V,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        let key_sorts = key.column_sorts();
        check_input_sorts(name, &action, &key_sorts)?;
        check_output_sort(name, &action, &value.column_sort())?;
        let bv = self.base_values();
        let mut row = key.into_values(bv);
        row.push(value.into_value(bv));
        action.insert(self.es_mut(), row.into_iter());
        Ok(())
    }

    /// Mint or look up an eclass for a constructor — mirrors the
    /// egglog `(Cons k1 k2 ...)` expression form. Pass inputs only;
    /// the output eclass is minted (or returned if a row with these
    /// inputs already exists).
    ///
    /// **Only valid for constructor tables.** Functions error;
    /// use [`Write::set`] for those.
    fn add_node<R: IntoRow>(
        &mut self,
        name: &str,
        inputs: R,
    ) -> Result<Value, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(
            name,
            &action,
            TableKind::Constructor,
            "constructor",
        )?;
        let sorts = inputs.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let bv = self.base_values();
        let key = inputs.into_values(bv);
        let value = action
            .lookup_or_insert(self.es_mut(), &key)
            .expect("constructor lookup_or_insert returned None");
        Ok(value)
    }

    /// Remove a row from the named table. Works for any subtype.
    fn remove<K: IntoRow>(
        &mut self,
        name: &str,
        key: K,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        action.remove(self.es_mut(), &key_values);
        Ok(())
    }

    /// Subsume a row in the named table.
    fn subsume<K: IntoRow>(
        &mut self,
        name: &str,
        key: K,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        action.subsume(self.es_mut(), key_values.into_iter());
        Ok(())
    }

    /// Union two values in the e-graph's union-find. The caller is
    /// responsible for ensuring both values belong to the same
    /// eq-sort — there is no runtime sort check.
    fn union(&mut self, x: Value, y: Value) -> Result<(), Error> {
        let action = *self.registry().union_action();
        action.union(self.es_mut(), x, y);
        Ok(())
    }

    /// Trigger a panic from a primitive. Always returns `None` so the
    /// caller can propagate with `?`.
    fn panic(&mut self) -> Option<()> {
        let panic_id = self.registry().default_panic_id();
        self.es_mut().call_external_func(panic_id, &[]);
        None
    }
}

fn lookup_action(
    registry: &ActionRegistry,
    name: &str,
) -> Result<TableAction, Error> {
    registry
        .lookup_table(name)
        .cloned()
        .ok_or_else(|| ApiError::MissingTable { name: name.to_string() }.into())
}

fn check_subtype(
    name: &str,
    action: &TableAction,
    expected: TableKind,
    expected_label: &'static str,
) -> Result<(), Error> {
    if action.kind() == expected {
        return Ok(());
    }
    let actual_label = match action.kind() {
        TableKind::Function => "function",
        TableKind::Constructor => "constructor",
    };
    Err(ApiError::WrongSubtype {
        name: name.to_string(),
        expected: expected_label,
        actual: actual_label,
    }
    .into())
}

fn check_input_sorts(
    table: &str,
    action: &TableAction,
    provided: &[ColumnSort],
) -> Result<(), Error> {
    let expected = action.input_sort_names();
    if expected.is_empty() {
        // Table registered without sort names — skip the check
        // entirely (the typed API can't validate; raw is fine).
        return Ok(());
    }
    if provided.len() != expected.len() {
        return Err(ApiError::WrongArity {
            table: table.to_string(),
            expected: expected.len(),
            got: provided.len(),
        }
        .into());
    }
    for (i, (got, want)) in provided.iter().zip(expected.iter()).enumerate() {
        if let ColumnSort::Named(got) = got {
            if got.as_ref() != want.as_ref() {
                return Err(ApiError::WrongColumnSort {
                    table: table.to_string(),
                    column: i,
                    expected: want.to_string(),
                    actual: got.to_string(),
                }
                .into());
            }
        }
        // Unchecked columns (e.g. bare `Value`) skip.
    }
    Ok(())
}

fn check_output_sort(
    table: &str,
    action: &TableAction,
    provided: &ColumnSort,
) -> Result<(), Error> {
    let Some(expected) = action.output_sort_name() else {
        // No sort name registered — skip.
        return Ok(());
    };
    if let ColumnSort::Named(got) = provided {
        if got.as_ref() != expected.as_ref() {
            return Err(ApiError::WrongOutputSort {
                table: table.to_string(),
                expected: expected.to_string(),
                actual: got.to_string(),
            }
            .into());
        }
    }
    Ok(())
}

// =====================================================================
// The four wrapper types — plain structs, methods come from traits.
// =====================================================================

/// Wrapper for [`Context::Pure`]. Implements [`Core`] only.
///
/// ```compile_fail
/// // Pure context cannot write: `Write` is not implemented.
/// use egglog::Write;
/// fn _no_writes<'a, 'db>(state: &mut egglog::PureState<'a, 'db>) {
///     state.set("foo", (1_i64,), 2_i64);
/// }
/// ```
///
/// ```compile_fail
/// // Pure context cannot reach raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::PureState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct PureState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    /// The call-site [`Context`] the wrapping primitive was invoked
    /// from. Stamped by the wrapper closure at invocation time and
    /// read by [`PureState::apply_function`]; user code cannot
    /// observe or modify it directly.
    pub(crate) ctx: Context,
}

/// Wrapper for [`Context::Read`]. Implements [`Core`] + [`Read`].
pub struct ReadState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
    pub(crate) ctx: Context,
}

/// Wrapper for [`Context::Write`]. Implements [`Core`] + [`Write`].
///
/// ```compile_fail
/// // Write context cannot reach raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::WriteState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct WriteState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
    pub(crate) ctx: Context,
}

/// Wrapper for [`Context::Full`]. Implements [`Core`] + [`Read`] + [`Write`].
///
/// ```compile_fail
/// // Even `FullState` cannot reach the raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::FullState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct FullState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
    pub(crate) ctx: Context,
}

impl<'a, 'db: 'a> PureState<'a, 'db> {
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>, ctx: Context) -> Self {
        Self { inner: es, ctx }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
}

impl<'a, 'db: 'a> ReadState<'a, 'db> {
    pub(crate) fn wrap(
        es: &'a mut ExecutionState<'db>,
        registry: &'a ActionRegistry,
        ctx: Context,
    ) -> Self {
        Self {
            inner: es,
            registry,
            ctx,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::Read, Context::Full]
    }
}

impl<'a, 'db: 'a> WriteState<'a, 'db> {
    pub(crate) fn wrap(
        es: &'a mut ExecutionState<'db>,
        registry: &'a ActionRegistry,
        ctx: Context,
    ) -> Self {
        Self {
            inner: es,
            registry,
            ctx,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::Write, Context::Full]
    }
}

impl<'a, 'db: 'a> FullState<'a, 'db> {
    pub(crate) fn wrap(
        es: &'a mut ExecutionState<'db>,
        registry: &'a ActionRegistry,
        ctx: Context,
    ) -> Self {
        Self {
            inner: es,
            registry,
            ctx,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::Full]
    }
}

// =====================================================================
// Trait impls. The wrappers implement the sealed accessor traits;
// the public capability traits' default methods do all the rest.
// =====================================================================

impl<'a, 'db: 'a> Internal<'a, 'db> for PureState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for PureState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for ReadState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for ReadState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for ReadState<'a, 'db> {}
impl<'a, 'db: 'a> Read<'a, 'db> for ReadState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for WriteState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for WriteState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for WriteState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for WriteState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for FullState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for FullState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Read<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for FullState<'a, 'db> {}
