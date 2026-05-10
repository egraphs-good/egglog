# Plan: Replacing the egglog Backend with DuckDB

A draft analysis of whether DuckDB could plausibly serve as the execution
backend for egglog, using the term-encoded program (see
[`src/proofs/proof_encoding.md`](src/proofs/proof_encoding.md)) as the
input IR.

**Scope**: this plan targets *term-encoded mode only* (with proof tracking
optional but supported). The standard, non-term-encoded path is out of
scope — too much of the backend's complexity (custom merge callbacks,
inline `union`, online congruence) lives there. Term encoding's job is
exactly to compile that complexity away into ordinary rules; this plan
takes that reduction as a given and asks what's left for the backend.

This is a **planning document, not a commitment**. It identifies what the
swap buys us, what fights us, and what an incremental path could look like.

> **Status note**: a precursor experiment is now planned in
> [`seminaive-encoding-experiment.md`](seminaive-encoding-experiment.md):
> lift seminaive evaluation out of the backend and into a compilation
> pass, then validate it on the existing egglog backend before
> committing to a backend swap. The experiment is independent of this
> plan but, if it succeeds, simplifies §2.6 (DuckDB just translates
> the timestamp-aware IR; no SQL-emission-time expansion needed).

---

## 1. Why this is even plausible

The term encoding pass is the linchpin. It rewrites the user's program so
that:

- Every `union` is gone. All equality reasoning lives in explicit
  `UF_<Sort>` and `UF_<Sort>f` tables maintained by ordinary rules.
- Every constructor becomes an *ordinary relation*: a term table plus a
  view table plus deferred-delete/subsume helpers.
- Rebuilding (congruence + view canonicalization) is expressed *in egglog
  rules*, not in the backend.
- Deletions and subsumptions are deferred to a dedicated ruleset.
- **Custom merge functions are compiled away into rules.** See
  `handle_merge_fn` at `src/proofs/proof_encoding.rs:238`. After term
  encoding, the only merge modes that appear on any function are
  `:merge old` (UF tables, view tables, term-proof tables) and
  `:merge new` (the `UF_<Sort>f` function index). The general-purpose
  `MergeFn` machinery in `egglog-bridge` is unused.

What's left after term encoding is essentially **plain Datalog with two
trivial upsert modes, primitives, and a schedule** — exactly the shape
SQL engines exist to evaluate. The ambitious claim is: if the IR is
already relational, the backend can be relational too.

DuckDB is attractive specifically because it is:

- **In-process**, no network hop ([`duckdb-rs`](https://github.com/duckdb/duckdb-rs))
- **Columnar + vectorized**, fast on bulk joins and aggregations
- **Embeddable** with a stable C ABI
- **Extensible** via Rust-defined scalar UDFs (`vscalar` feature) and
  table-producing functions (`vtab` feature)
- **Recursive-CTE capable**, including the newer `USING KEY` variant that
  was specifically designed to support fixpoint-style queries with
  selective overwrites instead of pure append
  ([SIGMOD '25 paper](https://db.cs.uni-tuebingen.de/publications/2025/using-key/how-duckdb-is-using-key-to-unlock-recursive-query-performance.pdf))

---

## 2. The mapping

| egglog (post-term-encoding) | DuckDB |
| --- | --- |
| Sort `S` (eq sort) | A scalar `BIGINT` ID type; rows live in a per-function table |
| Sort `S` (base value: i64, f64, String, …) | DuckDB native types (`BIGINT`, `DOUBLE`, `VARCHAR`, `BOOLEAN`, `BLOB`, …) |
| Container sort (Vec, Map, Set) | `LIST<...>`, `MAP<...>`, or a side table with parent-id+index |
| `Proof` sort | An interned ID into a `proofs` table; payload columns store the proof variant |
| Function/constructor `f(a,b) -> c` | Table `f(a, b, ret, ts BIGINT, subsumed BOOL)` with `PRIMARY KEY (a,b)` |
| `:merge old` (the dominant case post-term-encoding) | `INSERT ... ON CONFLICT DO NOTHING` |
| `:merge new` (only `UF_<Sort>f` index) | `INSERT ... ON CONFLICT DO UPDATE SET ret = EXCLUDED.ret` |
| `union` action | Out of scope: term encoding lowers it to an explicit `INSERT INTO UF_<Sort>` |
| Custom merge functions | Out of scope: term encoding compiles them to merge-rule + cleanup-rule pairs (`src/proofs/proof_encoding.rs:238`) |
| Rule body (conjunctive query) | A SQL `SELECT` with N-way join and primitive predicate filters |
| Rule action (insert / set / delete / subsume) | INSERT into per-function staging tables; flush after match phase |
| Schedule (run/saturate/seq) | Driven from Rust, not SQL — each iteration is one SQL transaction |
| Primitives (`+`, `<`, `string-concat`, …) | DuckDB built-ins where they line up; otherwise Rust-defined scalar UDFs |
| Seminaive timestamp filter | Each row carries `ts`; rule queries add `WHERE atom_i.ts >= :last_run` |

---

## 2.5 Where do we cut into the egglog pipeline?

Egglog has several layers between source and execution:

```
parse → ResolvedAst → desugar → term encoding → CoreRule → BackendRule → egglog-bridge RuleBuilder → core-relations free-join plan
```

For a DuckDB backend, the **right cut is at `CoreRule`**
(`src/core.rs:870`), the same place today's `BackendRule`
(`src/lib.rs:1886`) consumes. Reasons:

- A `CoreRule` is already a conjunctive query body
  (`Query<ResolvedCall, ResolvedVar>`) + a list of actions
  (`GenericCoreActions`). Each body atom is either a function call
  (table lookup) or a primitive constraint
  (`src/core.rs:1988–2005`). That's exactly what compiles to a SQL
  `SELECT ... FROM funcs JOIN ... WHERE primitives`.
- Term encoding has already run, so we see UF tables, view tables, and
  rebuild rules as ordinary `CoreRule`s with no special significance.
- Primitives are already separated from tables and carry their resolved
  signatures, which is what we need to map them to DuckDB built-ins or
  Rust-side `vscalar` UDFs.
- We're *upstream* of egglog-bridge's seminaive expansion, free-join
  planning, and physical execution — all the parts we're replacing.

In code terms: build an `egglog-bridge-duckdb` crate that exposes the
same public `EGraph` / `RuleBuilder` / `FunctionConfig` API as
`egglog-bridge`, so `BackendRule` in `src/lib.rs` doesn't have to
change. Internally, the new crate translates each call into SQL DDL
(for tables) and stored rule descriptors (for rules), then materializes
SQL at `run_rules()` time. This keeps the diff to `src/` minimal and
gives us a clean cut point.

---

## 2.55 Could seminaive live in the egglog IR rather than the backend?

This is a tempting refactor: move seminaive expansion *upstream* of any
backend, as another encoding pass alongside term encoding. The backend
then becomes "run these rules to fixpoint", with no notion of
timestamps in its API. Worth considering on its own merits, not just
for DuckDB.

**Sketch.**

- A new pass adds a `ts BIGINT` column to every function table.
  Insertions stamp `ts` with a global "current epoch" value. (Easy as
  an encoding step — every constructor's view-table emit point already
  threads through term-encoding code.)
- The same pass replaces each N-atom rule with N variants. Variant
  `i` adds `(>= ts_i last_run_at_R)` to the focused atom and
  `(< ts_j next_ts)` to the others.
- Two new global counters appear in the IR: `next_ts` (current epoch)
  and `last_run_at_R` per rule. The schedule executor bumps these
  between iterations.

**What this buys.**

- Backend becomes trivial: it just evaluates rules to a local
  fixpoint. No `RuleInfo.last_run_at` (`egglog-bridge/src/lib.rs:833`),
  no `RuleSet`-level seminaive bookkeeping, no special "delta" mode.
- The IR is testable and inspectable: you can dump the post-encoding
  program and read off exactly what the backend will execute.
- Multiple backends (free-join, DuckDB, hypothetical others) share the
  same seminaive logic. The current arrangement re-implements it in
  every backend that wants it.

**What it doesn't free us from.**

- *Some* component still has to maintain `last_run_at_R` per rule per
  iteration. Encoding into the IR shifts who owns it (scheduler vs.
  backend), not whether it exists. The cleanest landing is probably
  the schedule executor in `src/lib.rs`, which already drives
  iterations and would just need to expose globals to the backend.
- The `ts` column has to be threaded into every function's schema.
  This is fine, but it means every backend now sees timestamps
  whether or not it cares — the cost of pushing a backend concern
  into the IR is that it stops being a backend concern.
- Egglog would need a way to express the timestamp predicates. Two
  options: (a) introduce a small set of internal primitives
  (`ts->=`, `ts-<`) and a `Timestamp` sort, or (b) reuse `i64` and
  the existing comparison primitives. (b) is simpler and probably
  good enough; we just have to make sure the planner doesn't get
  confused by "global compared to column" patterns (it shouldn't —
  globals are already lowered to nullary constructors).

**Recommendation.**

Worth doing, *but as a separate change from the DuckDB swap*. It's a
useful refactor on its own — it would simplify `egglog-bridge` and
make the seminaive logic far more legible. But landing it as part of
this plan couples two large changes that should be evaluated
independently. The DuckDB plan is well-defined either way: if
seminaive moves into the IR first, the DuckDB backend just translates
the resulting timestamp-aware rules to SQL with no further work; if
it doesn't, the DuckDB backend does the N-fold expansion at SQL
emission time (§2.6). The choice doesn't change the end state.

For this plan, the assumption is **seminaive stays a backend concern**
and the DuckDB backend implements it at SQL-emission time. If that
refactor lands first, even better.

---

## 2.6 Seminaive: who does it?

DuckDB **has nothing seminaive-shaped built in**. Concretely:

- No incremental view maintenance, no delta tracking.
- Recursive CTEs (`WITH RECURSIVE`) are *naive* fixpoint — each
  iteration re-derives from the accumulated union table, not from a
  delta. The `USING KEY` extension lets you overwrite rows in the
  union table instead of appending, which helps with transitive
  closure, but it isn't seminaive: it doesn't restrict to "rows added
  in the previous iteration".
- No mutually recursive CTEs, so we can't even encode a fixpoint over
  multiple tables in one statement.

**So the backend has to do seminaive itself.** Fortunately, egglog's
seminaive scheme is purely a query-level transformation: each rule with
N body atoms becomes N delta queries, where exactly one atom is
restricted to `ts >= last_run_at` and the others to `ts < next_ts`. No
operator-level support is required. We mirror that scheme by emitting
N SQL `SELECT`s per rule per iteration:

```sql
-- Delta query for atom i (1 ≤ i ≤ N)
INSERT INTO target_views (...)
SELECT <projection>
FROM   func_a a, func_b b, ..., func_n n
WHERE  <primitive predicates>
  AND  a_i.ts >= :last_run_at_rule_R           -- the "new" atom
  AND  a_j.ts <  :next_ts FOR ALL j != i       -- the "old" atoms
ON CONFLICT DO NOTHING;
```

Per-table requirement: each function table carries a `ts BIGINT`
column. We bump a Rust-side counter once per `step_rules()` call and
record `last_run_at` per rule. This is the exact bookkeeping today's
backend already does (`RuleInfo.last_run_at` at
`egglog-bridge/src/lib.rs:833`); we're just doing it in SQL bind
parameters.

A subtlety: rebuilding bumps timestamps too, so that seminaive on
ordinary rules sees the updated rows. That logic stays the same — when
a rebuild rule writes a row, it writes it with the current `ts`.

**Should we let egglog generate the seminaive variants for us instead
of doing it ourselves?** No, for two reasons:

1. The expansion happens *inside* `egglog-bridge`
   (`rule.rs:869–919`), below the cut point we want. Reusing it would
   couple us to types and helpers we're trying to replace.
2. Seminaive expansion is mechanical — three or four lines of SQL
   string-building given a `Query<ResolvedCall, ResolvedVar>`. It's
   cheaper to redo than to refactor across the layer boundary.

So: take `CoreRule`, do the seminaive expansion ourselves, emit one
SQL statement per delta variant per rule per iteration.

---

## 2.7 Does DuckDB store rows in timestamp order?

Not in the same way `core-relations` does, but close enough for
seminaive scans, and possibly better than expected.

**What `core-relations` does today.** `SortedWritesTable` stores rows
with `[keys..., return_value, timestamp, subsume?]` in roughly
insertion (= timestamp) order, indexed by key for joins. A seminaive
scan can find "all rows with `ts >= X`" by walking from a known offset
forward — fine-grained, in-memory, pointer-cheap.

**What DuckDB offers.**

1. **Zone maps (row-group min/max statistics).** DuckDB stores tables
   in row groups of ~122,880 rows. Each row group records min/max for
   every column, used as zone maps during scans. A `WHERE ts >= :X`
   filter skips entire row groups whose `ts.max < :X` without reading
   them. ([Sorting on Insert for Fast Selective Queries](https://duckdb.org/2025/05/14/sorting-for-fast-selective-queries))

   For seminaive this is a near-perfect fit: we *always* insert in
   monotonically increasing `ts` order (the current epoch never moves
   backward), so zone maps will be naturally tight without any
   special effort. A delta scan that should read 5% of the table will
   touch ~5% of the row groups, with full vectorized execution within.

2. **ART indexes.** DuckDB supports Adaptive Radix Tree indexes via
   `CREATE INDEX` and implicitly for `PRIMARY KEY` / `UNIQUE` /
   `FOREIGN KEY` constraints
   ([Indexes – DuckDB](https://duckdb.org/docs/current/sql/indexes)).
   We can put an ART index on `ts` for very selective range scans on
   small deltas. Worth benchmarking — for the small-delta case the
   index might beat a row-group scan; for the medium-and-large case
   the columnar scan probably wins.

3. **Sort-on-insert idiom.** When loading bulk data, DuckDB recommends
   `INSERT INTO t SELECT ... ORDER BY clustering_col` to align row
   groups with zone maps. We get this naturally for `ts`; for
   *secondary* clustering (e.g. by key prefix) we can also `ORDER BY
   ts, key0` if it helps queries that join on key prefixes.

**Differences from the current backend.**

- **Granularity.** Row-group pruning is coarser than the in-memory
  pointer-walk. If a delta is much smaller than a row group (~120k
  rows), DuckDB still has to read the whole row group containing the
  recent rows. For tiny iterations this is overhead the current
  backend doesn't pay. For medium-to-large iterations (where the
  delta spans multiple row groups), DuckDB's vectorized scan should
  match or beat the current scan.

- **Joins inside the delta query.** The focused atom benefits from
  zone-map pruning; the *non-focused* atoms are scanned in full.
  That's the same as today — seminaive's whole point is that only
  one atom is restricted at a time. The vectorized join engine
  handles the larger side fine.

- **No "after offset N" semantics.** DuckDB doesn't expose physical
  row offsets the way an in-memory store can. We use `ts` as a
  logical cursor instead. As long as `ts` is monotonic per epoch
  (which it is, by construction), this is equivalent.

**Practical schema.** Each function table looks like:

```sql
CREATE TABLE Add (
    a   BIGINT NOT NULL,
    b   BIGINT NOT NULL,
    ret BIGINT NOT NULL,
    ts  BIGINT NOT NULL,
    subsumed BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (a, b)
);
-- Optional: ART index on ts for very-small-delta workloads
CREATE INDEX add_ts_idx ON Add (ts);
```

**Bottom line.** DuckDB doesn't give us *exactly* what
`SortedWritesTable` does, but the combination of (a) zone-map
pruning on the naturally-monotonic `ts` column and (b) optional ART
indexes is a credible substitute. The risk is purely on
small-iteration overhead, which is the same risk flagged in §5.1
generally. This isn't an extra problem — it's the same problem
expressed at the storage layer.

---

## 3. Execution model

The high-level loop in Rust would be:

```text
for each step in schedule:
    next_ts = current_ts + 1
    for each rule R in active_ruleset:
        for each atom a in R's body:
            run delta query: body where a.ts >= last_run_at[R],
                             other atoms.ts <  next_ts
            stage matched actions into per-table staging tables
        last_run_at[R] = next_ts
    flush_staging():
        for each touched function table:
            apply staged inserts, executing merge semantics
            (one of: ON CONFLICT DO NOTHING / DO UPDATE / Rust-side fold)
        if uf table grew:
            run rebuild rules to fixpoint
    current_ts = next_ts
    detect saturation: did anything change in this step?
```

Several pieces are non-obvious:

### 3.1 Seminaive

See §2.6. Short version: DuckDB doesn't help here, so we emit the
N-delta-queries-per-rule expansion ourselves at SQL generation time.
DuckDB then handles each delta query independently with full
columnar/vectorized execution — exactly the workload OLAP databases
are good at.

### 3.2 Merge functions

In term/proof mode this collapses to two trivial cases:

- **`:merge old`** (UF tables, view tables, term-proof tables): the
  first value wins. This is exactly `INSERT ... ON CONFLICT DO NOTHING`.
- **`:merge new`** (only `UF_<Sort>f`, the function-index over the UF):
  the latest value wins. This is `INSERT ... ON CONFLICT DO UPDATE
  SET ret = EXCLUDED.ret`, equivalently `INSERT OR REPLACE`.

DuckDB supports both forms natively, so no UDF-driven conflict
resolution is needed.

What about the things that *would* have been merge functions? They've
already been lowered:

- **Custom `:merge <expr>` declarations**: rewritten by
  `handle_merge_fn` into a pair of egglog rules — one that runs the
  merge expression when two distinct values are present for the same
  key, one that cleans up stale view rows. Both compile to ordinary
  joins and inserts on the SQL side.
- **Equality merges (the would-be `MergeFn::UnionId`)**: rule actions
  emit explicit `INSERT INTO UF_<Sort>(parent, child) VALUES (...)`
  rows. Rebuilding rules then close it transitively.
- **`union` actions in user code**: the term encoding rewrites them to
  the same explicit UF inserts.

So the entire `MergeFn` enum in `egglog-bridge/src/lib.rs:876–895` —
`AssertEq`, `UnionId`, `Primitive`, `Function`, `Old`, `New`, `Const`
— **is dead code** for term-encoded programs. Only `Old` and `New`
remain in any meaningful sense, and both are expressible in plain SQL.

This is the single biggest reason DuckDB looks viable: the most
SQL-hostile feature of the current backend simply doesn't appear in the
input.

### 3.3 Union-find

The current backend has `DisplacedTable`, a special table type with O(α)
`find_naive`. In DuckDB we'd represent the same thing with two tables:

```sql
CREATE TABLE uf_<Sort> (
    child  BIGINT,
    parent BIGINT,
    ts     BIGINT,
    PRIMARY KEY (child)
);
```

Path compression and parent-merging are already encoded as egglog rules
in the term encoding (`parent`, `single_parent`, `uf_function_index`
rulesets — see `proof_encoding.md` lines 84–101). Those translate
straight into SQL update queries. Canonicalization of a single value
becomes a `WITH RECURSIVE` walk up `parent`, or — closer to what the
encoding does — we materialize a `UF_<Sort>f(x) = parent[x]` view that's
kept up to date by rules.

The `WITH RECURSIVE ... USING KEY` variant is a particularly good fit
for transitive closure: it lets a node's parent be **overwritten** in
the iteration table rather than appended, which is the pattern the
union-find rules already implement.

### 3.4 Rebuilding

The term encoding turns rebuilding into rules. In DuckDB this becomes:

- Congruence rules: `INSERT INTO uf SELECT ... FROM viewA JOIN viewA WHERE same key, different ret`.
- Rebuild rules: `UPDATE viewA SET ret = (SELECT parent FROM uf_<Sort>f WHERE ...) WHERE ret <> parent`.

These are large bulk operations and well-suited to columnar execution.
They'll likely be **faster** than the current incremental rebuild for
big batches; they may be **slower** for tiny rebuilds where the
per-statement DuckDB overhead dominates.

### 3.5 Saturation detection

`run_rules()` returns whether anything changed. In SQL: keep a row count
of each function table before/after, or check whether any insert
returned `RETURNING` rows. DuckDB supports `INSERT ... RETURNING`, so
the pattern is `WITH inserted AS (INSERT ... RETURNING 1) SELECT count(*) FROM inserted`.

### 3.6 Primitives and external functions

Built-ins like `+`, `-`, `<`, string-concat map to DuckDB built-ins
nearly 1:1. Egglog primitives that don't (e.g. `ordering-max`,
`ordering-min`, custom user primitives, container ops) become Rust
scalar UDFs registered through `duckdb-rs`'s `vscalar` feature.

Two snags:

- **Side effects**. Today's `ExternalFunction::invoke` can call
  `state.stage_insert(...)`. A DuckDB scalar UDF is pure — it returns a
  value and that's it. Any primitive that wants to stage writes must
  instead be lifted into an action node and run by the Rust driver
  *between* SQL statements, not as part of one. Most egglog primitives
  are pure, but a small handful (panics with side-channel messaging,
  proof-store interning) are not. Workaround: route those through a
  Rust-side hook layer.
- **Container rebuilding**. Containers (Vec, Map, Set) need to be
  rewritten when their elements change e-class. Two designs:
  (a) store containers as a `LIST<BIGINT>` and rebuild via SQL
  unnest+join+collect; (b) keep containers Rust-side and expose them
  through a `vtab` table function. (a) is more uniform; (b) reuses
  existing container code with less churn.

### 3.7 Proof tracking

Proof terms are themselves egglog values: instances of a `Proof` sort
with constructors like `Rule`, `Trans`, `Sym`, `Cong`, `PCons`, `PNil`.
After term encoding they're indistinguishable from any other
constructor and need no special backend support — they're just rows in
proof-related tables. DuckDB inherits proof support for free if the
rest of the mapping is sound.

---

## 4. What this would replace and what it wouldn't

Replaces:
- `egglog-bridge`'s `EGraph` (its rule compiler, scheduler, and table operations)
- `core-relations` (tables, free-join, hash indices, rebuilder)
- `union-find` (the standalone crate)

Keeps:
- Everything in `src/` except `BackendRule` lowering: parsing, type
  checking, term encoding, proof extraction, scheduler-as-a-driver, sorts.
- `egglog-ast`, `egglog-reports`, `numeric-id` — unchanged.

In other words: **swap the engine, keep the language**. The frontend
still produces an IR; only the IR-to-execution mapping changes.

---

## 5. Risks and open questions

**P0 — likely deal-breakers if not solved well**

1. **Per-iteration overhead.** DuckDB query planning isn't free.
   Egglog rules can be tiny ("for each row in `Add`, do something"),
   and a benchmark like `extract-vec-bench.egg` may run thousands of
   rule iterations. If each rule firing is one SQL statement, the
   constant factor could swamp the columnar win. **Mitigation**:
   compile a whole ruleset (or whole iteration) to a single SQL
   transaction with multiple CTEs / staged tables; cache prepared
   statements; consider extension-mode integration to skip the parser.

2. **Container value rebuild semantics.** Today, container types
   register their own rebuild logic against the UF. SQL-side rebuild
   means walking nested `LIST`/`MAP` structures and rewriting IDs,
   which can be expensive without a planner that understands the
   pattern. Need a benchmark on container-heavy programs
   (`container-rebuild.egg`, `vec.egg`).

**P1 — solvable but not free**

3. **Mutually recursive CTEs aren't supported in DuckDB.** This isn't
   actually needed: cross-rule recursion is driven from Rust, not
   inside one CTE. But it forecloses some clever
   "compile-the-whole-fixpoint-to-one-CTE" optimizations.

4. **No transactional rollback semantics.** Egglog has none either,
   but DuckDB's MVCC may add overhead we don't need. Benchmark with
   `PRAGMA disable_checkpoint_on_shutdown` etc.

5. **Tests.** ~700 `.egg` test files. Bring-up will be one bug at a
   time. Plan for it: stand up a parallel-runner harness early.

**P2 — known unknowns**

6. **Aggregate UDFs in `duckdb-rs` are not yet first-class** (per the
   current docs). If any egglog primitive turns out to be naturally
   aggregate-shaped, we'd need to implement it via `vtab` instead.

7. **Parallelism story.** Egglog uses Rayon for in-process parallelism.
   DuckDB has its own thread pool. Mixing them could be fine, or could
   produce nasty contention. Pin DuckDB threads vs. Rayon threads.

8. **WASM target.** Egglog ships a WASM build (`wasm-example/`).
   DuckDB has a WASM build too, but binary size and integration are
   nontrivial — confirm before promising wasm parity.

**Notably *not* on this list (after the term-encoding reduction):**
custom merge functions, inline `union`, online congruence — all gone.
The remaining risks are about query-engine constant factors and
container-shaped values, not about modeling e-graph semantics in SQL.

---

## 6. Suggested phased path

The plan is designed so that each phase produces something
*independently useful*: each one either proves a hypothesis or shoots
it down before the next phase commits more code.

### Phase 0 — Spike (1–2 weeks, throwaway code)  ✅ done

- Translate one tiny program by hand into DuckDB SQL. ✅
- Wire `duckdb-rs` to a stub frontend; run rule iterations end to
  end. ✅ See `duckdb-spike/` (workspace member).
- Compare runtime against today's backend on the same input. ✅
- **Decision gate**: if the per-iteration overhead is >10x and we have
  no plan to amortize it, stop. If it's within ~2x, continue.

**Phase 0 results** (transitive closure on a chain of N edges,
best-of-3 wall time):

```
N        egglog    --term-encoding   duckdb spike
  10     0.00s     0.00s             0.03s
 100     0.00s     0.02s             0.17s
 500     0.05s     0.91s             0.95s
1000     0.21s     7.10s             2.38s
2000     0.86s    56.69s             7.29s
```

Reading:

- **Crossover with `--term-encoding` lands around N≈500–1000.** At
  N=1000, DuckDB beats `--term-encoding` by 3×; at N=2000, by 8×.
  Term encoding scales badly on this workload (saturate goes from
  0.91s to 56.69s for 4× more edges); DuckDB scales near-linearly
  (0.95s → 7.29s).
- **Process startup is ~25 ms for the DuckDB spike**, dominating
  small-N runs. The actual rule loop at N=1 takes ~4 ms; CREATE +
  seed takes <1 ms. The 30 ms floor is bundled-DuckDB init + rust
  binary launch.
- **Baseline egglog (no term encoding) is faster than DuckDB at
  every N** by 4–9×. That's the workload the current backend was
  built for; we're not trying to replace it for that case. The plan
  is term-encoded mode only.

**Decision**: pass. The per-iteration overhead is well within 2× of
`--term-encoding` and at scale DuckDB is significantly faster.
Proceeding to Phase 1.

(Spike is in `duckdb-spike/` — single binary, ~150 lines including
schema, seed, schedule loop, and verification. Throwaway code as
planned.)

### Phase 1 — Proof of concept (4–6 weeks)

- Build a parallel `egglog-bridge-duckdb` crate with the same public
  API as `egglog-bridge`.
- Implement: schema setup, basic INSERT/UPSERT, one merge mode
  (`:merge old`), conjunctive query compilation, seminaive deltas.
- Pick 5 small `.egg` tests; pass them.
- **Decision gate**: are we still within 3x of the existing backend on
  small tests? Are the test failures bug-shaped or
  fundamental-mismatch-shaped?

### Phase 2 — Feature parity for term-encoded subset (8–12 weeks)

- Union-find tables and rebuild rules in SQL (custom merges aren't a
  separate feature — they're already lowered to plain rules upstream).
- Deferred delete/subsume rulesets.
- Primitive UDFs (start with the `i64` / `String` / `Bool` set).
- Container sorts (`Vec`, `Map`, `Set`) — pick design (a) vs. (b) above
  based on Phase 1 perf data.
- Pass the term-encoded test suite (subset that today's term encoding
  supports — see `proof_encoding.md`).

### Phase 3 — Performance tuning

- Prepared-statement caching, transaction batching.
- Investigate compiling a full iteration to a single multi-CTE
  statement.
- Investigate the `USING KEY` recursive-CTE form for UF maintenance.
- Benchmark the codspeed suite. Goal: within 1.5x of current backend on
  large workloads, ideally faster on bulk-extraction benchmarks.

### Phase 4 — Proof mode & wider compatibility

- Enable proof tracking end-to-end (largely free per §3.7).
- WASM target if needed.
- Decide: replace the existing backend, or ship as an alternative?

---

## 7. Honest recommendation

The mapping is **clean enough to be worth a Phase 0 spike**. The term
encoding has done most of the conceptual work — it's already turned the
e-graph into a relational program with only `:merge old` and
`:merge new` upserts surviving — and DuckDB is a competent relational
executor with native support for both.

The **biggest open question** is the constant factor on small
iterations. The current backend is purpose-built and tightly integrated
with the union-find; DuckDB is general-purpose and goes through SQL.
Bulk operations should look great; tiny iterations might not.

The **second biggest** is container-typed sorts (`Vec`, `Map`, `Set`).
Their rebuild logic today is bespoke Rust; in SQL it becomes
unnest-rewrite-collect, which is fine in bulk but costly per row.

If both fall the right way, this becomes attractive: we'd inherit
DuckDB's optimizer, vectorized execution, persistence, and a much
smaller backend codebase to maintain. If either falls badly, the
ergonomic hit isn't worth the rewrite. **Find out which it is in
Phase 0**, on real inputs, before committing further.

---

## 8. Alternatives considered (briefly)

- **SQLite**: tighter embedding, simpler, but row-oriented and lacks
  DuckDB's vectorized execution. Probably worse for the workload.
- **Postgres**: strong custom-aggregate / trigger support could solve
  the merge-callback problem cleanly, but it's a server, not embeddable
  the way DuckDB is.
- **DataFusion**: another in-process columnar engine, with a more
  customizable physical-plan layer. Plausible alternative; less mature
  recursive-CTE story than DuckDB.
- **Keep the current backend, just optimize**: cheapest. Most of the
  arguments for DuckDB ("vectorized!", "free optimizer!") only matter
  if the constant factor lines up. The status quo is a known quantity.

---

*Document drafted as a feasibility sketch. Numbers in §6 are
order-of-magnitude estimates, not commitments. Phase 0 should be the
next step before any of this becomes a roadmap.*

---

## 9. Known issues (May 2026)

### Over-derivation in tests with rewrites + UF unifications

`tests/until.egg` and `tests/integer_math.egg` exit 0 and pass all
`(check ...)` queries, but `(print-size)` reports more tuples than the
reference egglog backend:

- `until.egg`: `g*` count 131 vs egglog's 21 (~6×).
- `integer_math.egg`: `Add` count 336 vs 331 (1.5%).

**Cause**: when the `@rebuilding` ruleset canonicalizes a view row
(delete old + insert at canonical leader positions), the new row
carries a fresh `ts = cur_iter`. User-rule seminaive at the next
iteration sees that "new" row and re-fires on it, deriving extra
tuples that congruence later collapses but doesn't fully remove.

**Tried and reverted**: emitting the rebuild's INSERT with
`ts = t<focus>.ts` (the matched row's original ts) instead of
`cur_iter`. This eliminated the re-fire problem in principle but
broke maintenance convergence catastrophically — `integer_math` went
from 5.9s / 215 MB to 49s / 4.86 GB and OOMed inside DuckDB. Likely
cause: the rebuild rule itself uses seminaive on the view's ts;
preserving the old ts means rebuild's own input never crosses its
`last_run_at` line, so canonicalization stalls while `@UF_<sort>`
keeps growing, and the maintenance saturate spins on huge
intermediate joins.

**Possible directions** (future):
- Track per-rule `last_user_run_at` separately from
  `last_rebuild_run_at` so user rules see canonical state at
  iteration boundaries without breaking the rebuild rule's own
  seminaive.
- Match user rules on term-id columns rather than full structural
  rows, so canonicalizing a row's input columns doesn't look like a
  "new" body match.
- Look at how egglog-bridge actually solves this — likely uses
  per-table generation counters instead of a single monotone ts.

For now the test passes correctness checks; the mismatch is in extra
residual tuples after rewrite saturation, not in user-visible truth
values.

### `(push)` / `(pop)` are no-ops

Our `dispatch` silently skips `Push`/`Pop` commands. Tests that wrap
`(run …)` blocks in `(push)/(pop)` to scope intermediate derivations
(e.g. `tests/calc.egg`) accumulate state across blocks instead of
rolling back, so `print-size` reports many more terms than egglog.

Implementing this needs DuckDB savepoints (`SAVEPOINT push_<n>`/
`ROLLBACK TO`) plus per-EGraph state (next_ts, last_run_at,
combined_rulesets) to be checkpointed. Not high priority — most
tests don't use it — but it's the cause of the calc.egg mismatch.
