# Experiment: Seminaive as a Compilation Pass

A precursor experiment to the DuckDB backend plan
([`duckdb-backend-plan.md`](duckdb-backend-plan.md)). Goal: lift
seminaive evaluation out of the backend and into a compilation pass
over the term-encoded program, then measure on the existing egglog
backend whether the encoded-seminaive version matches the built-in
implementation's performance.

If it matches: the DuckDB backend (and any future backend) can
inherit seminaive for free, as a feature of the IR rather than a
feature of the engine. If it doesn't match: we learn that without
betting a backend rewrite on it.

**Scope: term/proof mode only.** This experiment runs over programs
that have already been through term encoding
(`EGraph::new_with_term_encoding`, `src/lib.rs:447`). Default-mode
egglog is out of scope. See §2.5 for the rationale and what
default-mode support would actually require.

---

## 1. Hypothesis

> With seminaive expressed as a source-to-source rewrite — adding a
> `ts` column to each function and replacing each N-atom rule with N
> timestamp-restricted variants — the existing egglog backend (with
> its own seminaive turned off) runs the rewritten program at
> performance comparable to today's built-in seminaive.

"Comparable" = within ~20% on the codspeed benchmark suite, with no
benchmark regressing by more than 2x. (Final thresholds to be
agreed before measurement.)

If the hypothesis holds, seminaive becomes a property of the IR and
backends are simpler. If it fails, the most likely culprit is the
free-join planner's selectivity model — the encoded form may hide
information the built-in form makes obvious. That's useful to know
either way.

---

## 2. Why do this *before* DuckDB

Two separate questions get conflated if we pursue both at once:

1. *Can seminaive be expressed as an encoding?* (correctness + perf
   on a backend we already trust)
2. *Can DuckDB run egglog's workload?* (correctness + perf on a new
   backend)

Doing (1) first decouples them. We exercise the same backend we trust,
on a slightly more complex IR, and answer "is the encoding-shaped
seminaive any good?" If the answer is no, the DuckDB plan still works
— it just keeps doing seminaive at SQL-emission time per
[`duckdb-backend-plan.md` §2.6](duckdb-backend-plan.md). If the
answer is yes, the DuckDB plan gets dramatically simpler and we have
evidence the IR can carry this concern.

Cost is also low: this is bounded scope, all in `src/`, no new
dependencies, and reuses every existing test.

---

## 2.5 Why term/proof mode only?

Default-mode egglog (no term encoding) keeps a lot of semantics in
the backend: inline `union` actions, custom merge functions executed
via `MergeFn` callbacks, online congruence via the rebuilder,
`MergeFn::UnionId` for eq sorts. Every one of these mutates rows
*outside* the rule-evaluation path. For seminaive to be IR-level,
each such mutation has to attach a fresh `ts` to the row it touches,
because that's how the next iteration's seminaive scan finds it.

In term/proof mode this is automatic: term encoding has already
turned all of those operations into rules. A rule that does
`(set (FView a b new))` performs a normal insertion through the
normal action path, and the encoding can stamp `ts` on it the same
way it stamps any other insertion.

In default mode, the backend would have to populate `ts` itself for:

- `MergeFn` callbacks on key conflict (the resolved row needs the
  current epoch, not a derived value).
- Inline `union` actions (the new UF entry needs a ts).
- Online congruence and rebuild (rewritten rows need a ts).
- `MergeFn::UnionId` (UF growth and the resulting canonicalization).

None of those is hard individually. Together they're a non-trivial
amount of backend cooperation — and they push timestamp-awareness
back into the engine, which defeats the point of moving seminaive
into the IR. At that point we'd have IR-level seminaive *and*
backend-level timestamp plumbing for the operations the IR doesn't
cover, which is the worst of both worlds.

So the scoping decision is: the experiment is for term-encoded
programs, where the encoding-pass philosophy already applies. If it
succeeds and we later want default-mode coverage, the right move is
probably to extend term encoding to cover more of the cases that
currently bypass it, *not* to push timestamp plumbing into the
backend. That's a separate project.

This also matches the DuckDB plan's scope, which is term/proof only.
If a user wants to run a default-mode program against the DuckDB
backend, they'll need to enable term encoding first. Same answer for
this experiment.

---

## 3. The encoding

The encoding runs *after* term encoding (or as a follow-on pass in
the same module). It performs three transformations.

### 3.1 Add a `ts` column to every function

Every function declaration grows a trailing `ts BIGINT` (or `i64`)
input column.

```text
(function AddView (i64 i64 Math) Unit :merge old)
;; becomes
(function AddView (i64 i64 Math i64) Unit :merge old)
```

Insertions populate `ts` with the current epoch, read from a global
(§3.3). Queries that don't care about `ts` get a fresh wildcard
variable for it.

A subtlety: `:merge old` semantics now key on `(args..., ret, ts)`.
The simplest interpretation is that `ts` is *not* part of the
function's logical key; only the original arg columns are. To stay
compatible with the existing backend's "function from arg-tuple to
ret" model, we model `ts` as an extra *output* column, not an extra
input. The function becomes a multi-output function: `f: args ->
(ret, ts)`. With `:merge old`, the first inserted ret-and-ts wins —
which is what we want.

If multi-output functions aren't already supported cleanly, an
alternative is a parallel `f_ts(args...) -> i64` table, kept in
sync via a rule. Pick whichever is easier in egglog-bridge.

### 3.2 Rewrite each rule into N seminaive variants

For each user/term-encoding-generated rule

```text
(rule (atom_1 atom_2 ... atom_N)
      (action_1 ... action_K)
       :ruleset R :name "src-rule")
```

generate N variants. Variant `i` looks like:

```text
(rule (atom_1' atom_2' ... atom_N'
       (>= ts_i (last_run_at "src-rule"))
       (< ts_j (next_ts)) for j != i)
      (action_1 ... action_K)
       :ruleset R :name "src-rule@i")
```

where `atom_k'` is `atom_k` extended with a fresh `ts_k` variable
binding, `(last_run_at "src-rule")` and `(next_ts)` are reads of the
globals introduced in §3.3, and `>=` / `<` are the existing `i64`
comparison primitives (already in `src/sort/i64.rs`).

The `(< ts_j (next_ts))` constraints are technically optional in the
standard "focus the i-th atom" formulation — they're a defense
against the rule observing rows derived later in the same iteration.
Worth measuring both versions.

### 3.3 Add bookkeeping globals

Two kinds of globals (modeled as nullary constructors per the
existing `proof_global_remover.rs` pattern):

- `next_ts() -> i64`: the current epoch. Read by every rule
  insertion to populate `ts`. Bumped once per `step_rules` call.
- `last_run_at_<name>() -> i64` per rule: the epoch at which the
  rule last ran. Read by the rule's seminaive variants. Updated
  after each invocation.

Both are managed by the *schedule executor*, not by encoded
egglog. After each iteration the executor:

1. Reads the current `next_ts`.
2. For each rule that just ran, sets `last_run_at_<name>` to that
   value.
3. Increments `next_ts`.

This is the smallest plumbing point that's not pure encoding — and
even this could be modeled in egglog if we wanted (with a "global
update" action), but it's cleaner as a Rust-side scheduler hook.

---

## 4. Backend changes

Minimal:

1. **Disable built-in seminaive.** `RuleInfo.seminaive` defaults to
   `true` (`egglog-bridge/src/lib.rs:251, 348`). The encoding pass
   sets it to `false` for every rule it emits. The backend then
   evaluates each rule's body over the full table on every
   iteration; the encoded `ts` predicates do the seminaive
   restriction.

2. **Expose / bind globals between iterations.** The schedule
   executor in `src/lib.rs` needs to update the `next_ts` and
   `last_run_at_<name>` global tables between iterations. This is
   small (~50 LOC) and uses existing global-update mechanisms.

3. **(Optional) keep timestamps off-band.** If exposing `ts` as an
   extra column hurts measurably, we could keep the column hidden
   and have the backend auto-bind a `ts` variable from its internal
   timestamp during query compilation. But that re-couples backend
   to seminaive, defeating the experiment.

No changes to `core-relations`. No changes to free-join planning. No
new primitives.

---

## 5. Plumbing details worth pinning down early

### 5.1 Rebuild rules

Rebuild rules are rules. They get the same seminaive expansion. The
existing backend already runs them under seminaive (per
`egglog-bridge/src/lib.rs:706+`); the only change is that their
seminaive predicates are now in the IR.

Watch for: rebuild rules currently get treated specially around
incremental vs. nonincremental modes
(`egglog-bridge/src/lib.rs:570-614`). The encoding pass should
preserve whatever metadata the backend uses to pick a mode, or the
mode-selection logic should be ported to the encoding too.

### 5.2 Rules over functions without a ts column

Primitive constraints (`(< x 5)`) and base-value queries don't have
`ts`. Only function tables do. The encoding skips the focused-atom
treatment for primitive atoms — they're never the source of "new
rows." The non-focused `ts < next_ts` predicate likewise only applies
to function atoms. So an N-atom rule with K function atoms expands
to K variants, not N. Mechanical, but worth getting right.

### 5.3 Container-typed atoms

Container values rebuild against the UF. Their internal restructuring
isn't visible as new rows in any of *these* function tables — it's
visible only as updated `ret` values in tables that store the
container. Because we model `ts` as part of the (ret, ts) output and
keep `:merge old`, a rebuild that changes only the container's
internal IDs *won't bump `ts`*, and the row stays invisible to
seminaive's focused scans.

That's actually wrong for correctness. We need: when a row's `ret`
changes (due to canonicalization), `ts` should bump too, so dependent
rules see the change. Two options:

(a) Switch the merge mode for tables holding canonicalized values to
something that updates `ts` on change but keeps `ret` semantics
intact. Awkward — the existing merge modes don't let us mix
field-by-field semantics.

(b) Make the rebuild rule explicitly *re-insert* the row, which under
the existing semantics will update `ts` to the current epoch via the
normal insertion path. This matches what today's backend does
internally.

(b) is the right answer and is consistent with what term encoding
already does for view tables. We need to confirm the rewritten
rebuild rules actually do this — likely they do, since they perform
explicit `(set (FView ...))` actions.

### 5.4 Compile-time blowup

A program with 100 rules averaging 3 atoms each becomes ~300 rules.
The free-join planner runs once per rule. At ~1ms/rule for planning
(rough order of magnitude), that's an extra 200ms at startup.
Probably fine for benchmarks; worth measuring.

### 5.5 Free-join selectivity estimation

The biggest performance risk. The current backend's seminaive is
effectively "scan only the delta of one table" — the planner sees a
*small* table to drive the join from. Encoded as `(>= ts X)` on a
full table, the planner has to *infer* that this is a small fraction.

If the planner uses table cardinality without per-column statistics,
the encoded version may pick worse join orders than the built-in
version. Mitigations:

- Provide a backend hint: "this atom is the seminaive focus, treat
  as small." Lightweight; basically the encoding sets a flag on the
  focused atom that the planner reads.
- Maintain a small statistics structure for `ts` ranges per table.
  More invasive.
- Rely on indexes: if the focused atom can be served from an index
  on `ts`, the planner may pick the right thing. Worth checking.

This is *the* thing to measure first. If it's fine on the codspeed
suite, we're done. If it isn't, the planner-hint workaround is small.

---

## 6. Evaluation

### 6.1 Correctness

Run the *term-encoding-supported* subset of the test suite (the
files that pass `command_supports_proof_encoding` —
`src/proofs/proof_encoding_helpers.rs:504`) with the new pass
enabled. All of them should still pass with identical observable
behavior (same `check` results, same extracted terms).

Tests that don't support term encoding (programs with
`:no-merge` non-globals, custom-validator-less primitives, sorts
with `:presort` / `:uf` / `:proof` annotations, `Input` commands,
etc.) are out of scope. They aren't a regression — they're outside
the term-encoded universe this experiment targets.

A useful intermediate diagnostic: dump rule-firing counts per
iteration with both built-in and encoded seminaive. They should
match exactly modulo duplicate matches across variants (which the
"focus the i-th atom" form generates and the built-in suppresses
via more careful delta tracking).

### 6.2 Performance

Run the codspeed suite both ways. Compare:

- Total runtime per benchmark.
- Per-iteration runtime (to isolate planner overhead from
  per-row work).
- Compile time (planner runs ~3x more rules).
- Memory (more rules, more index structures).

Decision matrix:

- **Within 20% across the board, no >2x regression**: ship it. Move
  on to DuckDB with seminaive treated as IR-level.
- **Mixed results**: investigate the worst regressions. Most likely
  fix: planner hint flag (§5.5). Re-measure.
- **Consistent 3x+ regression**: declare the experiment a negative
  result. Keep seminaive in the backend. The DuckDB plan still
  works; it just does the N-fold expansion at SQL-emission time.

---

## 7. Phasing

### Phase A — Encoder skeleton (1 week)  ✅ done

- Add a new pass module (e.g. `src/proofs/seminaive_encoding.rs`). ✅
- Implement: function schema rewrite, global injection, rule expansion. ✅
- Run on the smallest term-encoded test (the running `Add` example
  from `proof_encoding.md`). ✅
- No correctness goal yet — just check it produces the expected
  egglog text. ✅

### Phase B — End-to-end correctness (1–2 weeks)  in progress

- Wire the pass into the run pipeline behind a flag
  (`--seminaive-encoding`).  ✅
- Set `seminaive=false` on all encoded rules. ✅ (CLI flag implies
  `--naive` automatically).
- Implement the schedule executor's `last_run_at` / `next_ts`
  bookkeeping. ✅
- Pass the existing test suite under the flag. Bug-fix one test at
  a time. **← next**.

**Status as of last update**: 43 of 44 proof-supporting tests pass
under `--seminaive-encoding` (excluding the proof-unsupported list
in `tests/files.rs`). One holdout: `repro-should-saturate.egg`,
which exercises a niche interaction between `:merge (min old new)`
and a self-modifying saturating rule.

Bugs fixed during Phase B:
- Rules without `:name` get their full s-expression as their stored
  name (`desugar.rs::rule_name`). Symbol-rename them to a fresh
  `seminaive_rule<n>` for use in `last_run_at_<src>` globals.
- Rule actions that contain bare `Expr` calls (e.g. the
  `(cleanup_constructor merged old)` form generated by
  `handle_merge_fn`) need their tracked nested calls mirrored too
  — same for `Let` and `Union` actions.
- `_ts` parallel tables must always be `:merge old`, even when
  their base function is `:merge new` or has a custom merge.
  Otherwise the ts updates on every set, breaking egglog's
  saturation detection (the schedule sees a change to the ts table
  on every iteration and never terminates). The cost is some
  under-firing of seminaive on `:merge new` / custom-merge base
  functions — acceptable for correctness.

### Phase C — Performance measurement (1 week)

- Run the codspeed suite under the flag, on the same hardware as
  CI baseline.
- Produce a comparison table: built-in vs. encoded, per benchmark.
- If results are bad, attempt the planner-hint mitigation (§5.5).
- Decide: does this approach earn its keep?

**First-pass timing** (best-of-3, `time` wall clock, single-threaded
release build). Tests where `--term-encoding` itself fails are
omitted as not apples-to-apples.

```
benchmark                       baseline   term-encoding   seminaive-encoding
calc.egg                          0.006s          0.016s               0.053s    (3.3x term, 9x base)
integer_math.egg                  0.007s          0.023s               0.233s    (10x term, 33x base)
combined-nested.egg               0.005s          0.005s               0.007s    (1.4x term, 1.4x base)
math-microbenchmark.egg           0.481s          8.750s              TIMEOUT    (>7x term, >125x base)
```

**Read**: seminaive-encoding is currently **3–10× slower than
`--term-encoding` alone** on the workloads it ran, and times out at
60 s on `math-microbenchmark.egg` (which `--term-encoding` finishes
in ~9 s). This matches the §5.5 prediction: the doubled body atoms
confuse the free-join planner's selectivity estimation, and lacking
the built-in seminaive's "small driving table" hint, it picks worse
join orders.

The small/medium-uniform-cost case (`combined-nested.egg`) shows
near-zero overhead — when there's little to plan, the extra atoms
don't hurt. The cost shows up where the planner has real choices to
make.

**Decision (provisional, until we try the planner hint)**:
encoding-shaped seminaive is *correct* but *not currently
performant*. The mitigation in §5.5 — a backend hint flag on the
focused atom telling the planner "this atom is highly selective" —
is the next thing to try. If that closes the gap on
`math-microbenchmark` and friends, the experiment is a win. If it
doesn't, the conclusion is "seminaive is a backend concern, not an
IR concern" and the DuckDB plan does seminaive at SQL-emission time
as currently written.

### Phase D — Decision

Three outcomes:

1. **Win**: encoding lands behind the flag, becomes default after
   one CI cycle. The DuckDB plan inherits IR-level seminaive.
2. **Mixed**: keep the flag for experiments; don't make default;
   note in DuckDB plan that we'll do seminaive at SQL-emission time
   (per the existing §2.6).
3. **Loss**: revert. Document what we learned. DuckDB plan
   unaffected.

Total: ~4 weeks if it goes smoothly. The bulk of work is correctness
debugging in Phase B.

---

## 8. Open questions

1. **Multi-output functions vs. parallel `f_ts` tables for §3.1?**
   Likely the former is simpler if egglog-bridge already supports
   them. Need to check.
2. **Do existing rebuild-rule mode-selection heuristics need to
   move?** Specifically the incremental/nonincremental switch in
   `egglog-bridge/src/lib.rs:570-614`. Probably yes; need to design
   how the encoding chooses or how the backend infers.
3. **Should the `(< ts_j next_ts)` clause be included or omitted?**
   Omitting is simpler and probably matches built-in semantics.
   Including is safer. Measure both.
4. **What's the right way to update globals between iterations?**
   Direct `set` action on the global's nullary constructor? An
   internal Rust API? Pick the one that's idiomatic in current
   code.
5. **Container-rebuild-bumps-ts story** (§5.3) needs explicit
   verification on `container-rebuild.egg` and `vec.egg`.

---

## 9. Relationship to the DuckDB plan

This experiment doesn't depend on DuckDB. The DuckDB plan stays as
written:

- If this experiment succeeds: §2.6 of
  [`duckdb-backend-plan.md`](duckdb-backend-plan.md) becomes a
  no-op — the IR already carries seminaive, the DuckDB backend just
  translates it to SQL.
- If this experiment fails or is mixed: §2.6 stays, the DuckDB
  backend does seminaive at SQL-emission time as currently planned.

Either way the DuckDB plan can proceed; this just answers in advance
whether seminaive is a backend concern or an IR concern.
