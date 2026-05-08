# Souffle Backend — Long-Term Plan

This document tracks the plan for replacing egglog's core-relations backend with
Souffle (forked if necessary), preserving correctness of the proof encoding and
aiming for at-or-better performance.

The strategy is to **minimize the egglog features used in the encoding's
output** until what remains is either (a) plain Souffle, or (b) a small,
well-scoped set of Souffle extensions worth maintaining.

## Guiding principles

1. **Every change behind a flag.** The current backend stays the production
   path until a Souffle backend has reached parity. Default off; tests must
   pass with the flag both on and off, throughout.
2. **One change per phase.** Each phase isolates one risk so failures are
   diagnosable. No bundling unrelated changes.
3. **Phase-end exit criteria.** Each phase defines what "this worked" means
   quantitatively. If a phase fails, we stop and reassess.
4. **Egglog tests are the spec.** The existing test corpus + proof checker is
   the contract; we do not redefine correctness.
5. **Term identity via Skolemization with structural records.** Egglog
   encoded actions write `(let v5 (Add a b))` — minting a fresh symbol
   per first-time `(Add a b)` and reusing it via the term table. That's a
   Skolem function `Skolem_Add(a, b)` whose value is hash-consed by the
   term table. The Souffle equivalent is the structural record
   `[Add_tag, a, b]`, which Souffle's record table hash-conses by `ord()`.
   Same mechanism, different syntax. (Souffle ADTs would also work and are
   acceptable; records are the lightweight choice.)
6. **`eqrel` is out of scope.** We use plain relations + subsumption for UF
   maintenance, not Souffle's specialized equivalence-relation type.

---

## Phase 0 — Audit and investigation (2–3 weeks)

Before changing any code, lock down what we actually need from the backend
and what Souffle actually delivers.

### Deliverables

- **Feature audit:** Every egglog feature used by the encoding's *output*
  (not its input source language). Categorize each as:
  - ✅ Souffle has a direct equivalent
  - ⚠️ Souffle has a workaround (specify which)
  - ❌ Souffle is missing it (potential extension target)
- **Canary test set:** 5–10 representative tests, kept small so we can run
  them through every phase. Coverage:
  - Pure equality saturation
  - Proof tracking
  - Deletion / subsumption
  - `(run N)`-bounded programs
  - Programs that diverge under naive fixpoint (to test scheduling concerns)

### Open questions to resolve

Each question below has a concrete investigation step. These run in parallel
with the audit and gate Phase 1.

1. **Subsumption timing.** ✅ **RESOLVED: continuous.** A same-stratum
   observer rule that copies a subsumed relation sees only the
   post-subsumption state. Subsumed tuples don't leak to other rules in
   the same SCC. See `experiments/souffle-compat/q1-subsumption-timing.dl`.

2. **Souffle perf on egglog-shaped workloads.** ⚠️ **PARTIAL: blocked by
   tooling + semantic mismatch.** See `experiments/souffle-compat/Q2-FINDINGS.md`.
   - Compiled Souffle is broken on macOS Tahoe (SDK 26 path issues).
     Forced into interpreter mode, which is 5–10× slower than compiled.
   - Unbounded rule sets (e.g., distributivity) diverge in Souffle's
     fixpoint model — they need `(run N)` bounding which Souffle lacks.
     `q2-bench.dl` ran 5+ minutes at 2 GB and didn't terminate while
     egglog finished comparable work in 3 seconds.
   - Finite-term-space benchmarks reached fixpoint in <10ms in both
     engines — below the timing noise floor.
   - **Implication:** Phase 5 (extensions for bounded iteration) is more
     load-bearing than originally thought. Compiled-Souffle blocker also
     needs resolving before any clean Phase 4 number.
   - **Remaining work:** fix macOS compile (or use Linux), construct a
     finite-term-space benchmark sized for meaningful timing.

3. **Auto-indexer behavior.** ✅ **RESOLVED: covers our patterns.** Souffle
   builds prefix btree indexes for `UF(c, ?)` (function-style lookup) and
   `AddView(a, b, ?)` (co-key prefix for congruence) without hints. See
   `experiments/souffle-compat/q3-auto-indexer.dl`.

4. **Fresh ID generation under Souffle's evaluation model.** ✅ **RESOLVED:
   Skolemization via structural records.** Validated by the experiments in
   `/tmp/souffle-experiments/test{1,2,3,4}-*.dl` (May 2026).
   - The "fresh ID" minted by `(let v5 (Add a b))` in egglog is a Skolem
     term — a deterministic function of the rule's bound variables. Egglog
     hash-conses it via the term table; Souffle hash-conses it via the
     record table.
   - Test 4 (`test4-egraph.dl`) successfully built a tiny e-graph proving
     `Add(1,2) ~ Add(2,1)` using:
     - Records `[tag, a, b, n]` as term IDs.
     - Recursive record types for nested terms.
     - `ord(record)` as a replacement for `ordering-max`/`ordering-min`.
     - Subsumption rules for path compression.
     - Plain relations (no `eqrel`) for UF.
   - Custom C++ functors with counters were considered and rejected:
     interaction with semi-naive evaluation is dangerous (re-derivation in
     iteration N+1 mints a different ID).
   - Encoding-time IDs only were considered and rejected: insufficient for
     terms minted in rule actions.
   - **Implication for the translator:** The Rust→Souffle emitter rewrites
     `(let vN (Add a b))` followed by `(set (X ... vN) ...)` as inline
     record literals `[Add_tag, a, b]` in the rule head. No `let` needed
     in the Souffle output.

5. **Cross-relation deletion.** ✅ **RESOLVED: via stratified negation
   live-view, OR tombstone subsumption.** Souffle subsumption is
   intra-relation only, so direct cross-relation deletion is impossible.
   Two workarounds:
   - **Live view via negation:** `AddViewLive(a, b, l) :- AddView(a, b, l),
     !ToDelete(a, b).` — queries use the live view. Simpler.
   - **Tombstone:** inject a sentinel row from `ToDelete` into `AddView`
     itself, self-subsume the real row, filter tombstones in a live view.
     Physically deletes data; better for memory.
   See `experiments/souffle-compat/q5{a,b,c}-cross-deletion-*.dl`.

### Exit criteria

- Audit doc reviewed.
- Canary set committed.
- All five investigation questions have written answers backed by experiment.
- A go/no-go decision on whether to start Phase 1.

### Status (May 2026)

- ✅ Q1 (subsumption timing) resolved
- ⚠️ Q2 (Souffle perf comparison) — **partial.** Major findings:
  bounded iteration is essential for many real workloads (Souffle without
  it diverges where egglog terminates), and compiled Souffle on macOS
  Tahoe is currently broken.
- ✅ Q3 (auto-indexer) resolved
- ✅ Q4 (fresh ID generation) resolved
- ✅ Q5 (cross-relation deletion) resolved

Phase 1 already complete (out of order). Phase 0 closure pending a
finite-workload Q2 perf number, which requires either fixing the macOS
Souffle compile or running comparison on Linux.

---

## Phase 1 — Subsumption-only encoding ✅ DONE (May 2026)

Dropped `:merge new` and the `__UF_<Sort>f` function-index table from the
encoding's output. Behind `--souffle-compat` flag in CLI; default off.

### What landed

- **Collapsed `UF_<Sort>f` into `UF_<Sort>`.** The rebuild rule queries
  `__UF_<Sort>` directly; in proof mode the proof is the function's output
  column (no pair plumbing).
- **Dropped `__uf_function_index` ruleset and its maintenance rule** when
  flag is on.
- **Dropped `__UFPair_<Sort>` sort declaration** (proof mode only) when
  flag is on.
- **Dropped `(saturate (run __uf_function_index))` step** from the rebuild
  schedule when flag is on.
- **Kept `:merge old`** (used on UF, view, and proof tables). Maps to
  Souffle `choice-domain` later.
- **Kept all rulesets and schedules** otherwise. Scheduling is Phase 5.
- **Term tables and let-style action sequences kept as-is.** The
  Skolem-via-record rewrite is a translator concern, not an encoder one.

### Files touched

- `src/proofs/proof_encoding.rs` — flag-gated emission in `declare_sort`,
  `rebuilding_rules`, and `rebuild()` schedule.
- `src/proofs/proof_encoding_helpers.rs` — gated `term_header()`.
- `src/lib.rs` — `EGraph::with_souffle_compat()` builder method.
- `src/cli.rs` — `--souffle-compat` flag.
- `tests/files.rs` — `souffle_compat` treatment runs alongside
  `term_encoding` and `proofs`.

### Validation results

- `cargo test --release`: **843 tests pass** (up from 712); 131 new trials
  cover the supported-file corpus through the flag in both proof and
  non-proof modes.
- All 85 `.egg` files: **0 divergences** between flag-on and flag-off paths
  (47 pass-identically, 38 fail-identically — the 38 hit
  `ProofEncodingUnsupportedReason` rejections that fire before either path
  matters).
- Performance on `tests/math-microbenchmark.egg`: **flag-on is ~5% faster**
  with **~10% lower memory** (median of 3 runs). The maintenance cost of
  the function-index table outweighed its lookup-speed benefit on this
  workload.

---

## Phase 1.5 — Skolemization in the encoder (optional, 1–2 weeks)

A future pass that inlines `(let vN (Add a b))` action bindings, replacing
them with the constructor call directly: `(set (X ... (Add a b)) ...)`.
Egglog's term table hash-conses constructor calls, so this is semantically
equivalent.

### Why it might be worth doing

- The souffle_compat encoded output becomes a closer mirror of what the
  Souffle translator (Phase 3) will emit, enabling shape-level validation,
  not just behavior.
- Multi-statement actions can be fanned out to multiple rules sharing the
  same body — exactly the shape Souffle wants (one head per rule).
- One less obstacle between encoded output and pure Datalog.

### Why we deferred it

- Doesn't unblock Phase 3; the translator was always going to do the
  rewrite anyway. Skipping it doesn't remove work, it just moves work.
- Non-trivial encoder churn for no semantic benefit in the egglog runtime.
- Higher-value problems open: scheduling, cross-relation deletion at scale,
  primitive audit.

### Trigger to revisit

If Phase 3 translator work finds the let/multi-statement form costly to
emit cleanly, come back here and inline at the encoder level.

---

## Phase 2 — Primitive reduction (1–2 weeks)

Audit every primitive the encoded program uses. For each, provide a
Souffle-equivalent or note it as a gap.

- `ordering-max` / `ordering-min` — replace with `ord()` comparison on the
  structural record IDs. Validated in Phase 0 question 4 experiment.
- `!=` — Souffle has it.
- Arithmetic, string, container primitives — case-by-case.

### Exit criteria

- Every primitive in the encoded output has a documented Souffle equivalent
  or appears on the "extensions needed" list.

---

## Phase 3 — External Souffle driver prototype (3–4 weeks)

First end-to-end Souffle execution.

- Build an emitter: encoded egglog program → `.dl` file. Per Phase 0
  question 4, the emitter Skolemizes term constructors as structural
  records: `(let vN (Add a b))` followed by `(set (X ... vN) ...)` becomes
  `X(..., [Add_tag, a, b]) :- ...` with the record literal inlined into
  each Souffle rule head.
- Multi-statement egglog actions fan out to multiple Souffle rules sharing
  the same body — Souffle has one head per rule.
- Build a Rust↔Souffle FFI shim using the C++ interface
  (`SouffleProgram::run`, `Relation::insert`, `purgeInternalRelations`).
- For each `(run-schedule ...)` step, drive from outside: insert facts, call
  `run()`, read facts back, re-insert.
- This deliberately ignores rulesets — everything in one Souffle stratum,
  possibly with explicit fixpoint by re-invocation.

### What this gets us

First real signal on whether the encoded *rules* execute correctly under
Souffle's semi-naive engine, separate from the scheduling question.

### Exit criteria

- Canary test set passes through the Souffle backend.
- Any unexpected semantic divergences documented.

---

## Phase 4 — Performance characterization (2 weeks)

Profile Phase 3 against the current backend on the canary set.

### Measure

- Wall-clock time per benchmark.
- Time spent in re-serialization (Phase 3's main overhead).
- Time spent re-running fixpoint when only a few rules' data changed.
- Memory.

### Decision tree

- **Within 2× current backend:** Maybe ship as-is. Skip rulesets work.
- **2–10× current backend:** Rulesets / phased fixpoint likely worth the
  fork cost.
- **>10× current backend:** Rulesets are mandatory; revisit whether Souffle
  is the right backend at all.

### Exit criteria

A go/no-go decision on Phase 5 with concrete numbers behind it.

---

## Phase 5 — Souffle extensions (4–8 weeks, if needed)

Based on Phase 4, fork (or upstream-PR) Souffle with the minimum extensions
needed.

### Almost certainly needed

**Named rule groups + schedule expression.** Roughly:

```
.stratum rebuild_uf { ... rules ... }
.stratum rebuild_view { ... rules ... }
.schedule (saturate (seq rebuild_uf rebuild_view))
```

Souffle already has stratum-level evaluation for negation; extending to
user-named, user-scheduled strata is a moderate evaluator change.

### Likely needed

**Bounded iteration.** `(run N)` semantics for a stratum. Mechanically small;
semantically loud — breaks the "everything saturates" assumption that
optimizations like magic sets rely on.

### Possibly needed

- Per-tuple insertion timestamp (if `choice-domain` is insufficient for
  `:merge old` in some case we haven't anticipated).
- Cross-relation deletion as a first-class action (most uses can be
  expressed via subsumption + auxiliary relations, so this is a last
  resort).

### Engagement strategy

Open a Souffle issue early describing the use case; even if maintainers
don't accept the extension, it surfaces blockers we don't see. Plan for a
fork either way; budget ongoing maintenance.

### Exit criteria

Forked Souffle runs the canary set with rulesets, perf within target
multiple of current backend.

---

## Phase 6 — Production hardening (4–8 weeks)

- Full test corpus passes through Souffle backend.
- Dual-backend CI (current + Souffle) on every PR.
- Migration plan documented; users can opt in.
- Decide: deprecate core-relations or run dual-backend indefinitely?

### Exit criteria

Souffle backend reaches feature parity; deprecation timeline announced (or
explicit decision to run both).

---

## Risks and off-ramps

| Phase | Failure mode | Off-ramp |
|---|---|---|
| 0 | Fresh ID investigation shows no clean approach | Stop. Either accept structural identity (revisit ADT plan) or abandon Souffle. |
| 1 | Subsumption timing diverges from `delete` semantics | Document the divergence; revisit encoding |
| 3 | Souffle's compiled C++ slower than core-relations even on simple workloads | Stop. Souffle isn't the right backend. |
| 4 | Numbers say rulesets are mandatory but fork looks too expensive | Stop. Document findings. |
| 5 | Souffle maintainers reject extensions, fork drifts | Pin to a Souffle version; budget maintenance |

---

## Cross-cutting work

- **Snapshot test infrastructure.** Phase 1 onward produces different
  encoder output. Either separate snapshot files per flag combo or
  `cargo test --features` gating. Decide early.
- **Proof checker.** Audit before any term-identity-adjacent change to
  confirm it's flag-independent.
- **Documentation.** Each phase updates `proof_encoding.md` to reflect what
  the flag actually emits.
- **Benchmark suite.** The codspeed benchmarks become the perf yardstick.
  Add Souffle-backend variants as they come online.

---

## Total rough timeline

10–22 weeks of focused work to reach a production-grade Souffle backend,
with three real go/no-go gates (Phase 0 fresh-ID question, Phase 3
correctness, Phase 4 performance). The first 4–6 weeks (Phases 0–2) deliver
value regardless of whether Souffle pans out — they tighten the encoding's
feature surface, which is good hygiene for any backend.
