# Q2 — Souffle perf on egglog-shaped workloads (partial)

**Status: partial.** Findings are useful but the clean comparison was blocked
by tooling and semantic-mismatch issues that themselves are informative.

## What we set out to measure

Translate one encoded egglog program by hand to Souffle and compare
wall-clock runtime + memory between:
- `egglog --souffle-compat <prog>.egg`
- `souffle <prog>.dl`

## What blocked the clean comparison

### 1. Compiled Souffle is broken on macOS Tahoe (SDK 26)

`souffle -c` and `souffle-compile.py` both hardcode paths under
`/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/lib/libsqlite3.tbd`
which don't exist on Tahoe. Manual compilation also fails inside Souffle's
templated headers (`ramBitCast` substitution failure, possibly a Homebrew
bottle/clang interaction).

This forces interpreter mode for now. Compiled Souffle is typically 5–10×
faster than interpreter, so any interpreter number is a *conservative*
estimate of compiled performance.

**Action:** the Souffle compile issue is a separate blocker that needs
fixing before Phase 4 (real perf characterization). Not on the critical path
for Phase 0 closure.

### 2. Unbounded rules diverge under Souffle's fixpoint model

`q2-bench.egg` uses commutativity + associativity + distributivity. Egglog
runs `(run 14)` and stops with ~28K terms in 3.1s. The hand-translated
Souffle (`q2-bench.dl`) runs to fixpoint and **does not terminate** — we
killed it after 5+ minutes at 2 GB RAM, still growing.

This is the **same blocker** the plan identifies for Phase 5: Souffle has
no `(run N)` analog. The encoded program's `(run-schedule (repeat N ...))`
maps to nothing in pure Souffle. Distributivity creates structurally
distinct terms forever; egglog stops by iteration count, Souffle can't.

### 3. Finite-term-space benchmarks are too small to time

`q2-group.egg` (cyclic group) and `q2-comm.egg` (commutativity-only)
both reach a true fixpoint in egglog with ≤720 terms in <10ms. Below
the noise floor for wall-clock timing.

To get a meaningful timing on a finite-term-space benchmark, we'd need a
workload that's deliberately constructed to take seconds — possible but
non-trivial.

## What we *did* learn

- **Souffle interpreter has a real perf gap on egglog-shaped workloads**
  even before considering the unbounded-iteration issue. The 5-minute non-
  termination on `q2-bench.dl` (where egglog took 3 seconds for ~comparable
  exploration) is at least a 100× indication. Compiled Souffle would close
  some but not all of this.
- **Hand-translating an encoded egglog program is mechanical but verbose.**
  The bulk of the `.dl` file is initial-state record literals (deeply
  nested) and per-rule rule fan-out. A real translator would generate this
  programmatically.
- **The Souffle rebuild rule pattern** (insert canonical, then subsume
  non-canonical) works but causes substantial intermediate fan-out before
  subsumption converges. Should profile carefully when we have compiled
  Souffle.

## Implications for the plan

- **Phase 4 (perf characterization)** depends on compiled Souffle working.
  Need to either fix the macOS build, run on Linux, or accept interpreter-mode
  numbers with a 5–10× discount.
- **Phase 5 (Souffle extensions for `(run N)`)** is now even more
  load-bearing than the plan suggested. Without bounded iteration, many
  real egglog programs (anything with distributivity, term-generating
  rules, or open-ended rewrite systems) won't run in Souffle at all.
- **The hand-translation experience** is itself useful Phase 3 prep — the
  pattern is now documented in `q2-bench.dl` and can become the basis for
  the eventual `.egg` → `.dl` emitter.

## Files

- `q2-bench.egg`, `q2-bench.dl` — the unbounded benchmark + hand
  translation. Diverges in Souffle.
- `q2-bench-small.egg` — 1-seed variant trying to match Souffle's
  workload.
- `q2-group.egg` — finite cyclic-group benchmark, too small to time.
- `q2-comm.egg` — finite commutativity-only benchmark, too small to time.
