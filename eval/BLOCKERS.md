# Blockers: running the paper benchmarks across the cross product

Status as of the first `eval/bench_backends.py` runs. The harness itself works
and is correct; these are the egglog-side limitations that stop most of the
paper benchmarks from running across the full {backend} × {normal,
term-encoding, proofs} grid. Revisit later.

## Benchmark corpus reality

- **pointer-analysis/main.egg** — *unrunnable here*. It `(input …)`s cclyzer++
  CSV facts (`function.csv`, …) that are **not redistributed** in this repo
  (see `paper-benchmarks/README.md`: the 2 GB `benchmark-input/` is omitted).
  Panics on the missing file. Out of scope until the inputs are fetched.

- **Herbie dumps** (`herbie/dump-egglog/dumps.tar.zst`, 1260 files) split into
  two kinds:
  - **`taylor*`** — no `let`-globals, no `multi-extract` → **run across all 6
    conditions** today. This is the portion that exercises the full grid.
  - **`rewrite*`** — **734 `(let $N (bigrat …))` globals + `multi-extract`** →
    only `bridge-normal` works; the other 5 cells hit Blockers 1 and 2 below.

- **math-microbenchmark/math_full.egg** — runs across the grid; slow under
  term-encoding/duckdb/proofs (term encoding dominates; see separate perf
  notes — `@rebuild_rule2` / `@rebuilding` ruleset).

## Blocker 1 — term encoder emits an undeclared view for non-eq-sort globals

**Symptom:** `--term-encoding` (and therefore `--duckdb`) on any `rewrite*`
dump fails with:

```
[ERROR] In 1:126-141: (@$0View  @v840)
    Unbound function @$0View
```

**Repro:**
```
target/debug/egglog-experimental --term-encoding \
  paper-benchmarks/herbie/dump-egglog/dumps.extracted/dump-egglog/rewrite1050.egg
```

**Cause (hypothesis):** the dumps bind globals with `(let $0 (bigrat …))`.
`bigrat` is a *non-eq-sort*. In term/proof mode the encoder treats let-globals
as constructors (`src/proofs/proof_encoding.rs` ~line 1062, "in proof mode they
are constructors") and so emits a reference to the global's view `@$0View`
(via `view_name`, used in the Constructor branches of `instrument_fact_expr`
~800 and `instrument_action_expr` ~1070) — but it never *declares* that view
for a global, so the reference is unbound.

This is the exact limitation the (now-deleted) `inline-bigrat-lets.py`
worked around by inlining these globals. It is **not** actually lifted; the
dumps resurface it. Real fix lives in the term encoder's global handling:
either declare the view for non-eq-sort globals, or don't route non-eq-sort
global *uses* through a view at all.

Starting points: `proof_encoding.rs` `view_name`, `instrument_fact_expr`
(~733), `instrument_action_expr` (~1048), and the `is_global` branch (~1062);
also how globals are lowered (`remove_globals` / `proof_global_remover`).

## Blocker 2 — proofs are incompatible with `multi-extract`

**Symptom:** `--proofs` (and `--duckdb --proofs`) on any dump that ends in
`multi-extract` fails with:

```
Command is not supported by the current proof term encoding implementation.
    Offending command: (run-schedule (repeat 1 run-extract-commands))
```

**Cause:** egglog-experimental's `multi-extract` lowers to a
`run-extract-commands` ruleset; proof mode rejects it as an unsupported
command (`UnsupportedProofCommand`). Every `rewrite*` dump terminates with a
`multi-extract`, so proofs can't run them at all.

**Options when we return:** strip/ignore `multi-extract` for proof runs (we
care about saturation timing, not the extraction, under proofs); or teach the
proof encoder to tolerate the extraction ruleset; or accept that proofs don't
cover extraction-terminated programs and benchmark proofs on the rest.

## What works today

| corpus            | bridge-normal | bridge-term | bridge-proofs | duckdb(-normal/term) | duckdb-proofs |
| ----------------- | ------------- | ----------- | ------------- | -------------------- | ------------- |
| taylor* dumps     | ✓             | ✓           | ✓             | ✓                    | ✓             |
| rewrite* dumps    | ✓             | Blocker 1   | Blocker 2     | Blocker 1            | Blockers 1+2  |
| math_full.egg     | ✓             | ✓ (slow)    | ✓ (slow)      | ✓ (slow)             | ✓ (slow)      |
| pointer-analysis  | needs CSV     | needs CSV   | needs CSV     | needs CSV            | needs CSV     |

To get clean full-grid numbers now, run a **taylor-only** sample, e.g.:

```
# after extracting once, point at a taylor subset:
python3 eval/bench_backends.py \
  paper-benchmarks/herbie/dump-egglog/dumps.extracted/dump-egglog \
  --limit 25   # NOTE: sorted, so this hits rewrite* first; filter to taylor* instead
```
(A `--filter taylor` style option would help — not yet implemented.)
