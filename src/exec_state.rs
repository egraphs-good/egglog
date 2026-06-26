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
//! - [`Write`] — name-indexed writes (`set`/`add`/`remove`/
//!   `subsume`/`union`/`panic`). Implemented for [`WriteState`] and
//!   [`FullState`].
//!
//! Privileged seams (`call_external_func`, raw `&mut ExecutionState`)
//! used by the `FunctionContainer` higher-order dispatch live on the
//! crate-private [`Internal`] trait. User code cannot reach them.
//!
//! [`PurePrim`]: crate::PurePrim
//! [`WritePrim`]: crate::WritePrim
//! [`ReadPrim`]: crate::ReadPrim
//! [`FullPrim`]: crate::FullPrim

use std::ops::Deref;

use crate::Error;
use crate::api::{ApiError, IntoValue, IntoValues};
use crate::core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, ExecutionState, ExternalFunctionId,
    Value,
};
use crate::{
    ast::{FunctionSubtype, Literal, ResolvedExpr},
    core::ResolvedCall,
    sort::{F, S},
    typechecking::FuncType,
};
use egglog_bridge::{ActionRegistry, TableAction, TableKind};
use smallvec::SmallVec;

/// Inline scratch for a row of column values. Matches the
/// `SmallVec<[_; 8]>` that `egglog_bridge::TableAction` uses internally.
type ValueRow = SmallVec<[Value; 8]>;

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
    fn apply_resolved_function(&mut self, _func: &FuncType, _args: &[Value]) -> Option<Value> {
        None
    }

    fn apply_table_function(
        &mut self,
        subtype: FunctionSubtype,
        action: &TableAction,
        args: &[Value],
    ) -> Option<Value> {
        match (subtype, self.ctx()) {
            (FunctionSubtype::Constructor, Context::Write | Context::Full) => {
                action.lookup_or_insert(self.es_mut(), args)
            }
            (FunctionSubtype::Custom, Context::Read | Context::Full) => {
                action.lookup(self.es(), args)
            }
            _ => None,
        }
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

    /// Convert an egglog [`Value`] to a Rust base type, assuming that the
    /// value belongs to `T`.
    fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.es().base_values().unwrap::<T>(x)
    }

    /// Convert a Rust base type to an egglog [`Value`].
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

    /// Dispatch an already type-specialized primitive in the current
    /// call-site context.
    ///
    /// This is a trusted evaluator hook, not an authorization boundary:
    /// callers must only pass primitives that were resolved for this same
    /// call-site context and whose surrounding expression has already been
    /// checked to require no more capability than this state wrapper provides.
    /// For example, a primitive body evaluator should typecheck the body under
    /// the runtime context it will register for, infer the body's required
    /// context, and register the primitive using the matching state wrapper.
    fn apply_primitive(
        &mut self,
        primitive: &crate::core::SpecializedPrimitive,
        args: &[Value],
    ) -> Option<Value> {
        let id = primitive.external_id(self.ctx());
        self.es_mut().call_external_func(id, args)
    }

    /// Evaluate an already typechecked expression in this primitive call
    /// context, using `bindings` for local variables.
    ///
    /// Primitive calls dispatch through the current call-site context.
    /// Table-backed function calls follow the same capability split as
    /// `unstable-app`: custom functions require a read-capable state,
    /// and constructors require a write-capable state that can mint on miss.
    ///
    /// This method assumes `expr` came from a trusted preparation pipeline that
    /// typechecked it for this exact runtime context and rejected expressions
    /// whose required capabilities exceed the receiver's state wrapper. It
    /// should not be used to evaluate arbitrary resolved expressions from a
    /// wider context inside a less-capable wrapper.
    fn eval_resolved_expr(
        &mut self,
        expr: &ResolvedExpr,
        bindings: &[(&str, Value)],
    ) -> Option<Value> {
        match expr {
            ResolvedExpr::Lit(_, literal) => Some(match literal {
                Literal::Int(x) => self.base_to_value(*x),
                Literal::Float(x) => self.base_to_value(F::from(*x)),
                Literal::String(x) => self.base_to_value(S::new(x.clone())),
                Literal::Bool(x) => self.base_to_value(*x),
                Literal::Unit => self.base_to_value(()),
            }),
            ResolvedExpr::Var(_, resolved_var) => {
                assert!(
                    !resolved_var.is_global_ref,
                    "global variable {:?} reached direct expression evaluation before remove_globals",
                    resolved_var.name
                );
                bindings
                    .iter()
                    .find_map(|(name, value)| (*name == resolved_var.name).then_some(*value))
            }
            ResolvedExpr::Call(_, resolved_call, children) => {
                let mut values = Vec::with_capacity(children.len());
                for child in children {
                    values.push(self.eval_resolved_expr(child, bindings)?);
                }
                match resolved_call {
                    ResolvedCall::Primitive(primitive) => self.apply_primitive(primitive, &values),
                    ResolvedCall::Func(func) => self.apply_resolved_function(func, &values),
                    ResolvedCall::Values(_) => {
                        panic!("`values` cannot be evaluated as a single-valued expression")
                    }
                }
            }
        }
    }
}

/// Read-side methods — name-indexed table lookup and iteration.
/// Implemented for [`ReadState`] and [`FullState`]; *not* for
/// [`PureState`] or [`WriteState`] (a `Write` context body must not
/// depend on live DB state).
///
/// The single-entry methods (`lookup`, `eclass_of`, `contains`)
/// return `None` if absent — never insert. The iteration /
/// introspection methods (`function_entries`, `constructor_enodes`,
/// `table_size`, `table_sizes`) walk the current contents of the
/// database.
///
/// Detectable misuse (wrong table subtype, wrong arity) is reported
/// as [`crate::ApiError`] via the method's `Result`. Per-column sort
/// matching is **not** checked at this layer.
#[allow(private_bounds)]
pub trait Read<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Look up a function's output value at the given key. Returns
    /// `Ok(None)` if the row is absent. The returned `Value` is raw —
    /// extract a Rust type via [`Core::value_to_base`] for base sorts
    /// or [`Core::value_to_container`] for containers.
    ///
    /// **Only valid for `function` tables.** Constructors error;
    /// use [`Read::eclass_of`] for those.
    fn lookup<K: IntoValues>(&self, name: &str, key: K) -> Result<Option<Value>, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        let key_values: ValueRow = key.into_values(self.base_values()).collect();
        check_arity(name, &action, key_values.len())?;
        Ok(action.lookup(self.es(), &key_values))
    }

    /// Look up a constructor's eclass at the given inputs, without
    /// minting a fresh one on miss. Returns `Ok(None)` if absent.
    ///
    /// **Only valid for constructor tables.** Functions error;
    /// use [`Read::lookup`] for those.
    fn eclass_of<K: IntoValues>(&self, name: &str, inputs: K) -> Result<Option<Value>, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Constructor, "constructor")?;
        let key_values: ValueRow = inputs.into_values(self.base_values()).collect();
        check_arity(name, &action, key_values.len())?;
        Ok(action.lookup(self.es(), &key_values))
    }

    /// True iff a row with the given key exists in the table. Works
    /// for any subtype — never mints.
    fn contains<K: IntoValues>(&self, name: &str, key: K) -> Result<bool, Error> {
        let action = lookup_action(self.registry(), name)?;
        let key_values: ValueRow = key.into_values(self.base_values()).collect();
        check_arity(name, &action, key_values.len())?;
        Ok(action.lookup(self.es(), &key_values).is_some())
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

    /// Call `f` on each [`Enode`] of a constructor / relation table.
    /// Errors with `WrongSubtype` if `name` is a function. To stop
    /// early, use [`Read::constructor_enodes_while`].
    fn constructor_enodes(&self, name: &str, mut f: impl FnMut(Enode<'_>)) -> Result<(), Error> {
        self.constructor_enodes_while(name, |enode| {
            f(enode);
            true
        })
    }

    /// Like [`Read::constructor_enodes`], but stops as soon as `f`
    /// returns `false`.
    fn constructor_enodes_while(
        &self,
        name: &str,
        mut f: impl FnMut(Enode<'_>) -> bool,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Constructor, "constructor")?;
        action.for_each_while(self.es(), |row| {
            let (eclass, children) = row
                .vals
                .split_last()
                .expect("constructor row has at least an eclass column");
            f(Enode {
                children,
                eclass: *eclass,
                subsumed: row.subsumed,
            })
        });
        Ok(())
    }

    /// Call `f` on each [`FunctionEntry`] of a function table. Errors
    /// with `WrongSubtype` if `name` is a constructor. To stop early,
    /// use [`Read::function_entries_while`].
    fn function_entries(
        &self,
        name: &str,
        mut f: impl FnMut(FunctionEntry<'_>),
    ) -> Result<(), Error> {
        self.function_entries_while(name, |entry| {
            f(entry);
            true
        })
    }

    /// Like [`Read::function_entries`], but stops as soon as `f`
    /// returns `false`.
    fn function_entries_while(
        &self,
        name: &str,
        mut f: impl FnMut(FunctionEntry<'_>) -> bool,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        action.for_each_while(self.es(), |row| {
            let (output, inputs) = row
                .vals
                .split_last()
                .expect("function row has at least an output column");
            f(FunctionEntry {
                inputs,
                output: *output,
                subsumed: row.subsumed,
            })
        });
        Ok(())
    }
}

/// One enode from [`Read::constructor_enodes`]. Columns are raw
/// [`Value`]s; convert with [`Core::value_to_base`] / [`Core::value_to_container`].
#[derive(Clone, Copy, Debug)]
pub struct Enode<'a> {
    /// The constructor's input columns.
    pub children: &'a [Value],
    /// The eclass id this enode belongs to.
    pub eclass: Value,
    /// Whether this enode has been subsumed.
    pub subsumed: bool,
}

/// One entry from [`Read::function_entries`]. Columns are raw
/// [`Value`]s; convert with [`Core::value_to_base`] / [`Core::value_to_container`].
#[derive(Clone, Copy, Debug)]
pub struct FunctionEntry<'a> {
    /// The function's key (input) columns.
    pub inputs: &'a [Value],
    /// The value the function maps the key to.
    pub output: Value,
    /// Whether this entry has been subsumed.
    pub subsumed: bool,
}

/// Action-side write methods — name-indexed inserts/removes/subsumes
/// plus union and panic. Implemented for [`WriteState`] and
/// [`FullState`]; *not* for [`PureState`] or [`ReadState`].
///
/// Detectable misuse (wrong table subtype, wrong arity) is reported
/// as [`crate::ApiError`] via the method's `Result`. Per-column sort
/// matching is **not** checked at this layer.
#[allow(private_bounds)]
pub trait Write<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Set a function table's value at the given key — mirrors the
    /// egglog `(set (f k) v)` action.
    ///
    /// **Only valid for `function` tables.** Constructors error;
    /// use [`Write::add`] for those.
    fn set<K: IntoValues, V: IntoValue>(
        &mut self,
        name: &str,
        key: K,
        value: V,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        let bv = self.base_values();
        let mut row: ValueRow = key.into_values(bv).collect();
        check_arity(name, &action, row.len())?;
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
    fn add<R: IntoValues>(&mut self, name: &str, inputs: R) -> Result<Value, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Constructor, "constructor")?;
        let key: ValueRow = inputs.into_values(self.base_values()).collect();
        check_arity(name, &action, key.len())?;
        let value = action
            .lookup_or_insert(self.es_mut(), &key)
            .expect("constructor lookup_or_insert returned None");
        Ok(value)
    }

    /// Remove a row from the named table. Works for any subtype.
    fn remove<K: IntoValues>(&mut self, name: &str, key: K) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        let key_values: ValueRow = key.into_values(self.base_values()).collect();
        check_arity(name, &action, key_values.len())?;
        action.remove(self.es_mut(), &key_values);
        Ok(())
    }

    /// Subsume a row in the named table.
    fn subsume<K: IntoValues>(&mut self, name: &str, key: K) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        let key_values: ValueRow = key.into_values(self.base_values()).collect();
        check_arity(name, &action, key_values.len())?;
        action.subsume(self.es_mut(), key_values.into_iter());
        Ok(())
    }

    /// Union two values in the e-graph's union-find. The caller is
    /// responsible for ensuring both values belong to the same
    /// eq-sort.
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

fn lookup_action(registry: &ActionRegistry, name: &str) -> Result<TableAction, Error> {
    registry.lookup_table(name).cloned().ok_or_else(|| {
        ApiError::MissingTable {
            name: name.to_string(),
        }
        .into()
    })
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

fn check_arity(table: &str, action: &TableAction, got: usize) -> Result<(), Error> {
    let expected = action.input_arity();
    if got != expected {
        return Err(ApiError::WrongArity {
            table: table.to_string(),
            expected,
            got,
        }
        .into());
    }
    Ok(())
}

fn apply_registered_function<'a, 'db: 'a>(
    state: &mut impl RegistrySealed<'a, 'db>,
    func: &FuncType,
    args: &[Value],
) -> Option<Value> {
    let action = lookup_action(state.registry(), &func.name).ok()?;
    state.apply_table_function(func.subtype, &action, args)
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
    fn apply_resolved_function(&mut self, func: &FuncType, args: &[Value]) -> Option<Value> {
        apply_registered_function(self, func, args)
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
    fn apply_resolved_function(&mut self, func: &FuncType, args: &[Value]) -> Option<Value> {
        apply_registered_function(self, func, args)
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
    fn apply_resolved_function(&mut self, func: &FuncType, args: &[Value]) -> Option<Value> {
        apply_registered_function(self, func, args)
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
