# Backend Trait Design — Phase 1

**Phase**: 1 (design + stub crate, no impls)
**Inputs**: `.tmp/backend_trait_refactor_plan.md`, `docs/backend_trait_inventory.md`
**Output crate**: `egglog-backend-trait` (top-level workspace member)
**Output doc**: this file

## Design principles

1. **Minimal IR changes.** No neutral `RuleIr`. The trait's rule-building
   surface ([`RuleBuilderOps`]) mirrors `egglog_bridge::RuleBuilder`
   one-for-one. The reference backend's impl is a thin passthrough; the
   DuckDB backend's impl accumulates calls into its existing `duck::Rule`
   data IR and submits to `compile_rule` on `build()`. Existing IRs
   (`egglog-bridge`'s callback `Query` and `egglog-bridge-duckdb`'s `Rule`)
   are untouched.
2. **No types move crates.** `FunctionId`, `RuleId`, `ColumnTy`,
   `QueryEntry`, `MergeFn`, `DefaultVal`, `FunctionConfig`, and
   `FunctionRow` remain in `egglog-bridge`. `Value`, `BaseValueId`,
   `ContainerValueId`, `ExecutionState`, `ExternalFunction`, and
   `ExternalFunctionId` remain in `egglog-core-relations`. The trait crate
   re-exports them.
3. **Dyn-compatibility.** `Backend` is `Send + Sync` and object-safe.
   Generic-over-`T: BaseValue` and generic-over-`C: ContainerValue` methods
   live on the [`BaseValuePool`] / [`ContainerPool`] sub-traits with
   `Any`-based dispatch. A small set of free helpers re-introduces the
   per-`T` sugar.
4. **DuckDB limitations are surfaced via capability flags.** Three
   booleans on `Backend` (`supports_inline_table_lookups`,
   `supports_subsumption`, `supports_complex_merge`, `supports_containers`)
   let callers gate features cleanly. Operations that the backend doesn't
   support return errors at the corresponding `RuleBuilderOps` call site
   rather than panicking later in compilation.
5. **`Value` stays `u32`.** Per the Phase 0 inventory's reverification:
   widening to `u64` is **not required** for Phase 1. The existing
   intern-table fallback in `core-relations/src/base_values/unboxed.rs:78-86`
   handles oversize `i64`. We can revisit if profiling shows the fallback
   dominates.

## Resolved open questions

| Question (from Phase 0) | Decision |
|---|---|
| `Value` widening to `u64` | **Not required.** Stay `u32`. The i64 intern-table fallback covers the only oversize case. |
| Subsumption on DuckDB | **Not supported in v1.** Trait does have a `subsume` method (bridge has it); DuckDB's impl returns an error. Capability flag: `Backend::supports_subsumption`. |
| `MergeFn::Function` / `MergeFn::Primitive` on DuckDB | **Not supported in v1.** Same pattern: trait accepts the merge fn (it's part of `FunctionConfig`); DuckDB errors at `add_table` time. Capability flag: `Backend::supports_complex_merge`. |
| `rust_rule` / `query` (user primitives that re-enter the database) | **Not supported in v1.** Capability flag: `Backend::supports_inline_table_lookups` gates them. Callers in `prelude.rs` check this flag and error at primitive registration. |
| Containers on DuckDB | **Not supported in v1.** `ContainerPool` is a stub on DuckDB (all accessors return empty / errors). The existing `program_supports_proofs` gate already excludes container files from DuckDB test combos, so the stub is never reached in practice. Capability flag: `Backend::supports_containers`. |
| `with_execution_state` | **Not on the trait.** The 4 callers in `src/` will be migrated in Phase 2 Commit 6 to dedicated methods (`add_values`/`add_term` cover sites #1 and #2; container registration is handled by the future generic `container_register_val`; the scheduler-side site #4 uses `add_values`). The reference backend's impl will call `with_execution_state` internally. |
| `TableAction` / `UnionAction` | **Stay as bridge inherent methods.** Under the minimal-change posture these are not lifted into the trait. The frontend uses them as today; DuckDB-incompatible primitives that touch them error at registration via `supports_inline_table_lookups`. |

## File map

| File | Purpose |
|---|---|
| `egglog-backend-trait/Cargo.toml` | New workspace member. Depends on `egglog-bridge` (for type re-exports), `egglog-core-relations`, `egglog-reports`, `egglog-numeric-id`, `anyhow`. |
| `egglog-backend-trait/src/lib.rs` | Trait definitions only. No impls. ~500 lines. |
| `Cargo.toml` (workspace root) | Adds the new crate to `members`; adds a `[workspace.dependencies]` entry. |
| `docs/backend_trait_design.md` | This file. |

No other files are modified in Phase 1.

## The `Backend` trait

`Backend: Send + Sync` is the central trait. Object-safe. Methods that
would otherwise be generic over a `T: BaseValue` or `C: ContainerValue`
delegate to the sub-traits returned by `base_value_pool` / `container_pool`.

### Method-by-method correspondence

Every method names the inherent method on `egglog_bridge::EGraph` it wraps.
The reference backend's `impl Backend` is a one-line passthrough per
method. DuckDB-side notes are in the `notes` column.

| Trait method | Bridge inherent method | Notes |
|---|---|---|
| `add_table(FunctionConfig) -> FunctionId` | `EGraph::add_table` | DuckDB: maps to `add_function` / `add_relation` / `add_eq_sort_constructor`; errors on `MergeFn::Function` / `MergeFn::Primitive` (gated by `supports_complex_merge`). |
| `table_size(FunctionId) -> usize` | `EGraph::table_size` | DuckDB: `db.count`. |
| `approx_table_size(FunctionId) -> usize` | `EGraph::approx_table_size` | DuckDB: same as `table_size` (estimate not needed). |
| `for_each(FunctionId, &mut dyn for<'r> FnMut(FunctionRow<'r>))` | `EGraph::for_each` | DuckDB: cursor over `SELECT * FROM <table>`. The HRTB lifetime on the closure is required because the bridge borrows each row from a transient `TaggedRowBuffer`; the per-iteration borrow is not tied to `&self`. |
| `for_each_while(FunctionId, &mut dyn for<'r> FnMut(FunctionRow<'r>) -> bool)` | `EGraph::for_each_while` | DuckDB: same cursor, break on false. Per-row buffer carries `vals: &[Value]` borrowed from a scratch slot inside the impl. |
| `lookup_id(FunctionId, &[Value]) -> Option<Value>` | `EGraph::lookup_id` | DuckDB: `SELECT cN FROM t WHERE c0=… LIMIT 1`. |
| `add_values(Box<dyn Iterator<...>>)` | `EGraph::add_values` | DuckDB: multi-row `INSERT … VALUES`. |
| `add_term(FunctionId, &[Value]) -> Value` | `EGraph::add_term` | DuckDB: `INSERT … RETURNING` against the term table. |
| `get_canon_repr(Value, ColumnTy) -> Value` | `EGraph::get_canon_repr` | DuckDB: for `ColumnTy::Id`, runs `uf_find` against the native UF; for `ColumnTy::Base(_)`, returns `val` unchanged. |
| `fresh_id() -> Value` | `EGraph::fresh_id` | DuckDB: `SELECT nextval('__egglog_eqsort_seq')`. |
| `new_rule(&str, bool) -> Box<dyn RuleBuilderOps + '_>` | `EGraph::new_rule` | Bridge: a one-line newtype wrapping `RuleBuilder<'a>`. DuckDB: a struct that accumulates calls into a `duck::Rule`. |
| `free_rule(RuleId)` | `EGraph::free_rule` | DuckDB: remove from rule registry. |
| `run_rules(&[RuleId]) -> Result<IterationReport>` | `EGraph::run_rules` | DuckDB: maps to `run_iteration_in_set` with the corresponding rule names. |
| `flush_updates() -> bool` | `EGraph::flush_updates` | DuckDB: drain staged inserts, run native-UF rebuild if dirty. |
| `register_external_func(Box<dyn ExternalFunction>) -> ExternalFunctionId` | `EGraph::register_external_func` | DuckDB: wrap as a VScalar UDF; errors if the function reflects (via `supports_inline_table_lookups`) that it needs reentrancy. |
| `free_external_func(ExternalFunctionId)` | `EGraph::free_external_func` | DuckDB: drop the VScalar. |
| `new_panic(String) -> ExternalFunctionId` | `EGraph::new_panic` | DuckDB: register a panic VScalar that sets the shared panic-message slot and aborts the current iteration. |
| `base_value_pool() -> &dyn BaseValuePool` | `EGraph::base_values` | Bridge: returns a thin shim over `&BaseValues` that implements `BaseValuePool`. DuckDB: returns its own pool. |
| `base_value_pool_mut() -> &mut dyn BaseValuePool` | `EGraph::base_values_mut` | Same. |
| `container_pool() -> &dyn ContainerPool` | `EGraph::container_values` | Bridge: shim over `&ContainerValues`. DuckDB: an empty stub. |
| `container_pool_mut() -> &mut dyn ContainerPool` | `EGraph::container_values_mut` | Same. |
| `base_value_constant_dyn(Value, BaseValueId) -> QueryEntry` | `EGraph::base_value_constant` (`<T>` form) | The trait variant takes the already-interned `Value` and `BaseValueId` to stay dyn-compatible; the generic-`T` form is provided by callers, e.g. `egraph.base_value_pool().intern_dyn(...)` + `Backend::base_value_constant_dyn(...)`. |
| `supports_inline_table_lookups() -> bool` | (no bridge analog; capability flag) | Bridge: `true`. DuckDB: `false`. |
| `supports_subsumption() -> bool` | (capability flag) | Bridge: `true`. DuckDB: `false`. |
| `supports_complex_merge() -> bool` | (capability flag) | Bridge: `true`. DuckDB: `false`. |
| `supports_containers() -> bool` | (capability flag) | Bridge: `true`. DuckDB: `false`. |
| `set_report_level(ReportLevel)` | `EGraph::set_report_level` | DuckDB: store on struct. |
| `dump_debug_info()` | `EGraph::dump_debug_info` | DuckDB: log `SELECT * FROM …` per table. |
| `clone_boxed() -> Box<dyn Backend>` | (uses `EGraph: Clone`) | Bridge: `Box::new(self.clone())`. DuckDB: bespoke (see below). |

### Cloning `Box<dyn Backend>`

The frontend's snapshot / push-pop machinery needs to clone the egraph
state. The bridge already derives `Clone` on `EGraph`, so its
`clone_boxed` is one line. DuckDB's `EGraph` holds a live `duckdb::Connection`,
which is not `Clone`. The chosen strategy for the DuckDB impl (Phase 2
Commit 8 / 9):

- Maintain a replay log of all `add_table`, `register_external_func`,
  `new_rule`, and content-mutating calls.
- On `clone_boxed`, build a fresh `duckdb::Connection`, replay the log, and
  perform a `COPY` from the source database to the clone for each table.

The replay-log approach is documented further in Phase 2 Commit 9. The
trait does not prescribe the strategy — it only requires that
`clone_boxed` produce an independent backend equivalent to the original.

`impl Clone for Box<dyn Backend>` calls `clone_boxed`, so existing code
that derives `Clone` on `Frontend::EGraph { backend: Box<dyn Backend> }`
continues to compile without manual wiring.

## The `RuleBuilderOps` trait

Mirrors `egglog_bridge::RuleBuilder` one-for-one, with two minor
differences forced by dyn-compatibility:

- `call_external_func`'s panic-message closure becomes a `String`
  argument. The bridge's `RuleBuilder::call_external_func` takes
  `impl FnOnce() -> String + 'static + Send`; this is folded to its
  forced value at the trait boundary. (The deferred-construction
  optimization is only useful when the panic message is expensive to
  build; almost all call sites already pass a `String::new()` or a
  cheap `format!`.)
- `query_prim`'s return type is `Result<()>` — the bridge's inherent method
  was always `Result<()>`, but it is documented at the trait level as a
  failure point for backends without the underlying primitive.

### Method correspondence

| Trait method | Bridge inherent | DuckDB notes |
|---|---|---|
| `new_var(ColumnTy) -> QueryEntry` | `RuleBuilder::new_var` | DuckDB: allocate `duck::Var(name)`. |
| `new_var_named(ColumnTy, &str) -> QueryEntry` | `RuleBuilder::new_var_named` | Same with explicit name. |
| `query_table(FunctionId, &[QueryEntry], Option<bool>) -> Result<()>` | `RuleBuilder::query_table` | DuckDB: append `Atom::Func { … }` to current rule. Errors if `is_subsumed = Some(true)` (subsume not supported). |
| `query_prim(ExternalFunctionId, &[QueryEntry], ColumnTy) -> Result<()>` | `RuleBuilder::query_prim` | DuckDB: append `Atom::Filter` or `Atom::Bind` depending on whether the last arg is a fresh var. |
| `call_external_func(...) -> QueryEntry` | `RuleBuilder::call_external_func` | DuckDB: append `Action::LetExpr { expr: Term::Prim(...) }`. |
| `lookup(FunctionId, &[QueryEntry], String) -> QueryEntry` | `RuleBuilder::lookup` | DuckDB: append `Action::LetCtor` or `Action::LetExpr` per `DefaultVal`. |
| `subsume(FunctionId, &[QueryEntry]) -> Result<()>` | `RuleBuilder::subsume` | DuckDB: **error** (subsumption not supported). |
| `set(FunctionId, &[QueryEntry])` | `RuleBuilder::set` | DuckDB: append `Action::Insert` with merge semantics. |
| `remove(FunctionId, &[QueryEntry])` | `RuleBuilder::remove` | DuckDB: append `Action::Delete`. |
| `union(QueryEntry, QueryEntry)` | `RuleBuilder::union` | DuckDB: append `Action::Insert` against the UF table. |
| `panic(String)` | `RuleBuilder::panic` | DuckDB: append `Action::Panic`. |
| `build(self: Box<Self>) -> Result<RuleId>` | `RuleBuilder::build` | Bridge: `Ok(self.0.build())`. DuckDB: submits accumulated `duck::Rule` to `compile_rule(...)`, registers, returns id. |

### Methods deliberately NOT on `RuleBuilderOps`

These exist on the bridge's inherent `RuleBuilder` but are internal:

- `lookup_uf` (pub(crate))
- `check_for_update` (pub(crate))
- `add_atom_with_timestamp_and_func` (pub(crate))
- `lookup_with_subsumed` (pub(crate))
- `set_with_subsume` (pub(crate))
- `rebuild_row` (pub(crate))
- `set_focus` (pub(crate))
- `set_plan_strategy` (pub(crate))

These are used only inside the bridge's `incremental_rebuild_rules` /
`nonincremental_rebuild` flow, which is internal to the bridge's
`add_table` and never visible to frontend callers. They stay as inherent
pub(crate) methods on the bridge.

`RuleBuilder::egraph(&self) -> &EGraph` is also not on the trait — the
two existing call sites (in `egglog-bridge` internals) don't need it from
the trait perspective.

## The `BaseValuePool` sub-trait

```text
pub trait BaseValuePool: Send + Sync {
    fn register_type_dyn(&mut self, type_id: TypeId) -> BaseValueId;
    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId;
    fn intern_dyn(&self, ty: BaseValueId, value: Box<dyn Any + Send + Sync>) -> Value;
    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any + Send + Sync>;
    fn has_ty(&self, type_id: TypeId) -> bool;
}
```

A set of generic-over-`T` free helpers wraps these dyn methods:

```text
pub fn pool_register_type<T: BaseValue>(pool: &mut dyn BaseValuePool) -> BaseValueId;
pub fn pool_get_ty<T: BaseValue>(pool: &dyn BaseValuePool) -> BaseValueId;
pub fn pool_get<T: BaseValue>(pool: &dyn BaseValuePool, value: T) -> Value;
pub fn pool_unwrap<T: BaseValue>(pool: &dyn BaseValuePool, val: Value) -> T;
```

`pool_get` and `pool_unwrap` honor `T::MAY_UNBOX`: when the type is
inline-encodable they short-circuit before touching the dyn pool. This
preserves the bridge's existing fast-path.

### Reference backend impl (sketch, not committed)

```text
struct BridgeBaseValuePool<'a>(&'a BaseValues);

impl BaseValuePool for BridgeBaseValuePool<'_> {
    fn register_type_dyn(&mut self, type_id: TypeId) -> BaseValueId { … }
    // ... etc
}
```

Or, simpler: implement `BaseValuePool` directly for `&BaseValues` /
`&mut BaseValues`. The bridge will pick one in Phase 2 Commit 4.

### DuckDB backend impl (sketch, not committed)

`egglog_bridge_duckdb::EGraph` holds a `BaseValues`-shaped struct (per
the Phase 0 recommendation: reuse `egglog_core_relations::BaseValues`
directly). Common `BaseValue` types (`i64`, `f64`, `bool`, `String`,
`()`) use the same inline-vs-interned encoding as the reference. The
DuckDB SQL columns store the `Value`'s `u32` representation (cast to
`BIGINT` for storage); the pool provides the mapping back to typed
values for inspection / serialization.

## The `ContainerPool` sub-trait

```text
pub trait ContainerPool: Send + Sync {
    fn has_container_type(&self, type_id: TypeId) -> bool;
    fn enabled(&self) -> bool;
    fn get_dyn(&self, ty: TypeId, val: Value) -> Option<Box<dyn Any + Send + Sync>>;
    fn register_val_dyn(&mut self, ty: TypeId, value: Box<dyn Any + Send + Sync>) -> Result<Value>;
    fn for_each_dyn(&self, ty: TypeId, f: &mut dyn FnMut(Value, &dyn Any));
    fn size(&self, ty: TypeId) -> usize;
}
```

Generic helper:

```text
pub fn container_register_val<C: ContainerValue>(
    pool: &mut dyn ContainerPool, value: C
) -> Result<Value>;
```

### Reference backend impl (sketch, not committed)

Wraps `egglog_core_relations::ContainerValues`. `enabled()` returns
`true`. All methods succeed.

### DuckDB backend impl (sketch, not committed)

An empty stub. `enabled()` returns `false`. `has_container_type` always
`false`. `get_dyn` always `None`. `register_val_dyn` always
`Err(anyhow::anyhow!("container sorts not supported on duckdb backend"))`.
`for_each_dyn` is a no-op. `size` returns `0`.

The stub is defensive: the term-encoding pass's `program_supports_proofs`
check already excludes every container-using program from DuckDB combos,
so these error paths are programmer-error guards rather than routine code
paths.

## Type re-exports / origins

As of Phase 2 Commit 3, the basic id and config types live in
`egglog-backend-trait`; `egglog-bridge` depends on the trait crate and
re-exports them. The other types live in their neutral lower-level
homes and are re-exported through the trait crate for caller convenience:

| Type | Defined in | Re-exported from `egglog-backend-trait`? |
|---|---|---|
| `FunctionId`, `RuleId`, `ColumnTy`, `QueryEntry`, `Variable`, `VariableId`, `MergeFn`, `DefaultVal`, `FunctionConfig`, `FunctionRow` | `egglog-backend-trait` (as of C3) | (origin) |
| `Value`, `BaseValueId`, `ContainerValueId`, `BaseValue`, `ContainerValue`, `ExecutionState`, `ExternalFunction`, `ExternalFunctionId` | `egglog-core-relations` | yes (`pub use`) |
| `IterationReport`, `ReportLevel` | `egglog-reports` | yes (`pub use`) |

Callers will normally
`use egglog_backend_trait::{Backend, FunctionId, Value, ...}` and never
need to import from the underlying crates directly. Existing callers
that imported from `egglog_bridge::{FunctionId, ColumnTy, …}` continue
to compile because `egglog-bridge` keeps a matching set of `pub use`
re-exports.

### Bridge-only methods on the moved types

Before C3, `MergeFn` had an inherent `impl` block in `egglog-bridge`
(`fill_deps`, `to_callback`, `resolve`) that referenced bridge-internal
state (`EGraph`, `TableId`, `core_relations::MergeFn`). Inherent impls
can only be defined in the crate that owns the type, so these methods
became private free functions in `egglog-bridge` (`merge_fn_fill_deps`,
`merge_fn_to_callback`, `merge_fn_resolve`). Likewise `VariableId::to_var`
became the private free function `id_to_var` in `egglog-bridge/src/rule.rs`.

## How `TableAction` and `UnionAction` are handled

Per the minimal-change posture, these stay as **inherent methods on the
bridge**, not on the trait. The two motivations:

1. Lifting them onto the trait would require defining new sub-traits
   `TableActionOps` / `UnionActionOps`, which would in turn need to be
   passed `&mut ExecutionState`. That leaks `ExecutionState` into the
   trait surface — exactly what we're trying to avoid.
2. The only callers of `TableAction` / `UnionAction` are
   `src/lib.rs::EGraph::input_file` (2 sites; will use `add_values` after
   Phase 2 Commit 6), `src/scheduler.rs:236` (1 site; will use
   `add_values`), and inside user-defined primitives' `apply()` body
   (`src/prelude.rs:325-360,422,544,551`). The user-primitive-side use is
   the only one that the trait surface cannot subsume by other methods.

**For user-defined primitives that need to call back into table state
during `apply()`** (the `rust_rule` / `query` constructs in
`src/prelude.rs`), we gate via the capability flag
`Backend::supports_inline_table_lookups`. The reference backend returns
`true`; the DuckDB backend returns `false`. Primitives that need this
capability check the flag at registration time and refuse to register if
the backend reports `false`. Programs that use `rust_rule` / `query` on
DuckDB get a clean error at parse / typecheck time, not a panic at
runtime.

The trait does **not** need a `with_execution_state` method. The two
remaining `with_execution_state` callers in `src/` (`lib.rs:1611` and
`lib.rs:1618`) are migrated to `add_values` / `add_term` in Phase 2
Commit 6. The third (`lib.rs:1929`) is the container-registration site,
which becomes the generic helper `container_register_val<C>(pool, c)`.
The fourth (`scheduler.rs:225`) is replaced by a multi-row `add_values`.

## Stub crate verification

Phase 1 deliverable:

- `cargo build -p egglog-backend-trait`: succeeds.
- `cargo build` (workspace): succeeds (only pre-existing dead-code
  warnings in `src/backend_duckdb.rs`).
- `cargo test --release --quiet --test files -- --test-threads=4`:
  838 / 838 pass (no regression).

No callers updated. No bridge or DuckDB code changed. The new crate is a
pure addition; if it were removed, the rest of the workspace would still
compile and test.

## What Phase 1 does NOT include

- **No impls.** The bridge does not yet `impl Backend for EGraph`. The
  DuckDB backend does not yet `impl Backend`. Those are Phase 2 Commits
  4 and 9 respectively.
- **No caller updates.** `src/lib.rs::EGraph` still holds `backend:
  egglog_bridge::EGraph` (concrete type), not `Box<dyn Backend>`. That
  is Phase 2 Commit 8.
- **No type movement.** All types live where they do today. The
  dep-direction flip (bridge depending on trait crate) is a future Phase
  2 commit.
- **No `Sort::column_ty` changes.** `Sort::column_ty(&self, &egglog_bridge::EGraph)`
  is unchanged. The trait-based signature
  `column_ty(&self, &dyn Backend)` lands in Phase 2 Commit 7.

## Design decisions made in this commit (not pre-specified by the plan)

1. **`base_value_constant_dyn` takes `Value + BaseValueId`** instead of
   exposing a generic `base_value_constant<T>` method on the trait. The
   bridge's `EGraph::base_value_constant<T>` is generic — it's not
   dyn-compatible. The dyn-friendly form pushes the
   `T -> (Value, BaseValueId)` conversion onto the caller, who already
   has access to a `BaseValuePool`. Callers in `src/lib.rs:2066,2180-2184`
   and `src/scheduler.rs:327` will be updated in Phase 2 Commit 8 to do
   this two-step (a small helper trampoline can be added if the
   boilerplate is annoying).

2. **`for_each` has an `unimplemented!()` default body, not a real
   default.** Writing a real default body requires nesting a
   `&mut dyn FnMut(FunctionRow)` inside `&mut dyn FnMut(FunctionRow) -> bool`,
   which is awkward to express through `&mut dyn` (the inner closure
   would need to capture the outer one mutably — possible, but the
   resulting code is uglier than just requiring every backend to
   implement it). Both backend impls already have a clean `for_each`
   inherent path, so requiring them to override is no extra work. The
   `unimplemented!()` default keeps the trait formally object-safe
   without offering a footgun default.

3. **`call_external_func` takes `String`, not a closure.** Bridge's
   inherent method takes `impl FnOnce() -> String + 'static + Send`.
   That can't go on a dyn trait without boxing. Callers that care about
   deferred construction can wrap an empty `String` and call
   `lazy_panic_msg` via `RuleBuilderOps` extensions if needed later.
   In practice the cost of eagerly building the panic message is
   negligible because almost all call sites already use
   `format!("...", id)` rather than something expensive.

4. **`Backend::add_term` is included.** The bridge has `add_term` as a
   public method but it's only currently called via the `add_values`
   path. It's included here because DuckDB will likely want
   `INSERT ... RETURNING` semantics for term insertion (avoiding a
   roundtrip), and exposing it on the trait makes that explicit.
   Removable from the trait surface in a future commit if it proves
   unused.

5. **Capability flags are concrete methods, not associated consts.**
   `supports_subsumption`, `supports_complex_merge`,
   `supports_inline_table_lookups`, `supports_containers` are all
   methods returning `bool`. Associated constants would be slightly
   cleaner but force impls to be `const`, which (per the existing trait
   landscape) is fine but unnecessarily restrictive. Methods are
   trivially compiled to constant returns.

6. **`Box<dyn Backend>: Clone`** is implemented via `clone_boxed`. The
   pattern is standard. The frontend `EGraph::clone` will work
   automatically once `backend: Box<dyn Backend>` lands.

7. **No `BackendBuilder` factory.** The plan suggested
   `EGraph::default()` constructs `Box::new(egglog_bridge::EGraph::default())`
   directly, and `EGraph::with_duckdb()` is parallel. No need for a
   factory trait in v1.

## Trait-signature adjustments made during Phase 2 implementation

These are changes made after Phase 1 shipped that the design above has
already been updated to reflect; they're recorded here for traceability.

1. **`for_each` / `for_each_while` use an HRTB on the row lifetime**
   (changed in Commit 4). The original Phase 1 signature was
   `fn for_each<'a>(&'a self, table, &mut dyn FnMut(FunctionRow<'a>))`,
   tying the row borrow to `&self`. In practice the bridge borrows each
   `FunctionRow` from a per-call `TaggedRowBuffer` whose lifetime is
   strictly shorter than `&self`. The correct shape is
   `&mut dyn for<'r> FnMut(FunctionRow<'r>)`. The DuckDB impl will need
   the same shape (rows come from a per-row scratch buffer).

2. **`BaseValuePool::register_type_dyn` / `intern_dyn` / `unwrap_dyn`,
   and `ContainerPool::register_val_dyn` / `get_dyn` / `for_each_dyn`,
   are `unimplemented!()` in the bridge** (Commit 4). They cannot be
   implemented through a pure `TypeId` API because the underlying
   `BaseValues::register_type<P>` / `ContainerValues::register_val<C>`
   are generic over `P: BaseValue` / `C: ContainerValue` and need the
   typed factory at compile time. For now the frontend continues to use
   the concrete `EGraph::base_values_mut().register_type::<T>()` API
   directly. A future commit (slotted between Commit 7 and Commit 8)
   will either:

   - extend `egglog-core-relations::BaseValues` / `ContainerValues` with
     a typed dyn dispatch table (e.g. an extension trait on
     `DynamicInternTable` exposing `intern_dyn` / `unwrap_dyn`), so the
     bridge wrapper can forward to it; or
   - move `BaseValuePool` / `ContainerPool` registration off the trait
     and onto a separate setup-time API.

   Tracked as an open issue for Commit 6/7 (see report).

## What gets built in Phase 2

This list is informational; not implemented in this commit.

- **Commit 4**: `impl Backend for egglog_bridge::EGraph` (trait + sub-traits
  + RuleBuilderOps). Trivial — every method is one line.
- **Commit 6**: Refactor the 4 `with_execution_state` callers in
  `src/lib.rs` and `src/scheduler.rs` to use `add_values` / `add_term` /
  `container_register_val`.
- **Commit 7**: `Sort::column_ty(&self, &dyn Backend) -> ColumnTy`.
- **Commit 8**: `EGraph::backend` field becomes `Box<dyn Backend>`.
  At this point lib.rs compiles against the trait; only the bridge is
  registered.
- **Commit 9**: Stub DuckDB `impl Backend` — compiles, doesn't run yet.
- **Commit 10**: DuckDB `impl RuleBuilderOps` — wires up the duck::Rule
  accumulator pattern.
- **Commit 11**: DuckDB `BaseValuePool` — concrete in-process pool.
- **Commit 12**: DuckDB primitives (limited; flag gates `rust_rule` /
  `query`).
- **Commit 13**: DuckDB `ContainerPool` stub.
- **Commit 14**: Flip the test harness.
- **Commits 15+**: Iterate on extract / prove-exists / serialize on
  DuckDB.

## References

- `/Users/oflatt/egglog3/.tmp/backend_trait_refactor_plan.md` — the plan.
- `/Users/oflatt/egglog3/docs/backend_trait_inventory.md` — Phase 0 inventory.
- `/Users/oflatt/egglog3/egglog-bridge/src/rule.rs` — bridge `RuleBuilder` (the
  trait mirrors this).
- `/Users/oflatt/egglog3/egglog-bridge/src/lib.rs:122-829` — bridge
  `EGraph` inherent methods (the trait wraps these).
- `/Users/oflatt/egglog3/core-relations/src/base_values/mod.rs:66-124` —
  `BaseValues` (informs `BaseValuePool`).
- `/Users/oflatt/egglog3/core-relations/src/containers/mod.rs:70-230` —
  `ContainerValues` (informs `ContainerPool`).
- `/Users/oflatt/egglog3/egglog-backend-trait/src/lib.rs` — the trait
  definitions themselves (full source).
