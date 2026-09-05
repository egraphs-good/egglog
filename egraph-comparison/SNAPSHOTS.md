# Semantic snapshot evaluation

The comparison tool can already add useful database regression coverage. Adopt
stored semantic database snapshots incrementally, alongside the current
command-output tests. The pilot in `tests/comparison_snapshots.rs` deliberately
leaves the existing harness and `.snap` files in place.

## What the current harness checks

`tests/files.rs` runs normal, parallel, desugared, term-encoded, and proof
variants. Its shared snapshots use
`CommandOutput::snapshot_stable_under_proof_encoding` in `src/lib.rs`:

* `PrintFunction` contents and extraction variants are dropped.
* Best-term extraction is reduced to its cost.
* A final `(print-size)` adds table counts to every program.
* Snapshots are skipped for parallel runs and several known nondeterministic
  cases. Error strings and proof-support lists have separate snapshots.

Counts can stay unchanged when a function result changes, a constructor term is
replaced, or the same number of unions produces different equalities. The pilot
has three executable regressions demonstrating this: the existing snapshot
strings match, but database comparison fails and produces a valid certificate.

## Options

| Option | Coverage | Tradeoff |
| --- | --- | --- |
| Stored JSON database + semantic comparison | Changes across commits and thread counts; certificates explain failures | Larger baselines and explicit format/version maintenance |
| Compare a fresh sequential run with parallel/desugared runs | Differences between execution treatments, without stored state | Both treatments can regress identically; extra program execution |
| Snapshot a deterministic minimized serialization | Canonical bytes capture the same database semantics independently of input IDs | Extra sorting during generation; raw sidecars currently retained for failure certificates |
| Keep command-output snapshots | Errors, formatting, extraction costs, command order, intermediate outputs | Does not establish database equality |

The pilot now uses canonical JSON for accepted database contents, with raw JSON
as a diagnostic sidecar. Use differential checks to broaden threading coverage
and focused output tests for observable command behavior.
Final-state equality cannot replace intermediate-state checks, extraction-cost
checks, or error snapshots.

## Working pilot

Three checked-in canonical baselines cover constructor equalities (`eqsat-basic`),
relations (`path`), and ordinary functions (`fibonacci`). Each fresh run with
1, 4, or 32 threads is canonicalized in full-database mode and compared to the
accepted bytes. The success path reads no raw baseline and performs no joint
refinement. Both term and function changes remain observable.

Each `.canonical.json` has a raw `.json` diagnostic sidecar. On byte mismatch,
the test loads that sidecar and checks that it canonicalizes to the accepted
bytes. It then generates and verifies a disequality certificate. The actual raw
database, actual canonical bytes, and verified certificate are written under
`target/comparison-snapshots/`, with paths in the failure message. A stale or
missing sidecar produces a diagnostic error; it cannot justify a certificate.
A separate consistency test checks every accepted baseline/sidecar pair, and
regressions exercise constructor, equality, and ordinary-function failures.

Keeping sidecars lets the pilot use the existing certificate APIs, since a
minimized quotient is not necessarily valid input `Database` JSON. A future
certificate reader for [canonical graphs](CANONICAL.md) could remove these
sidecars. Canonical IDs are deterministic but a semantic change may renumber
many classes, so use the certificate to understand a large byte diff.

```sh
cargo test -p egglog --test comparison_snapshots
# Explicitly accept new databases after reviewing an intentional behavior change:
EGGLOG_UPDATE_COMPARISON_SNAPSHOTS=1 cargo test -p egglog --test comparison_snapshots golden_databases_match_sequential_and_parallel_runs
```

Updates run the sequential program once per baseline and regenerate both files.
They are never triggered by ordinary test execution. Review a proposed baseline
semantically with the comparison binary on the raw sidecars, as well as inspecting
its canonical JSON diff. The canonical format is separately versioned; a version
change requires baseline regeneration. [Canonical benchmarks](CANONICAL-PERFORMANCE.md)
measure the extra work and storage tradeoffs. Existing count, output, error, and
extraction-cost tests remain in place.

## Bounded survey

Reproduce the 12-program, five-treatment feasibility survey with:

```sh
EGGLOG_COMPARISON_SURVEY_OUTPUT=/tmp/egraph-comparison-survey.json cargo test -p egglog --test comparison_snapshots survey_snapshot_candidates -- --ignored --nocapture
```

The survey reports every unsupported case rather than silently skipping it.
On this checkout, 10 of the 12 normal exports succeeded, and all 10 compared
equal to their 32-thread runs. This is not a claim of complete coverage: one
successful export (`map`) was empty because all its values were held in globals.
The following six useful cases had nonempty exported databases:

| Program | Classes | Rows | Pretty JSON bytes | Compare with 32-thread run |
| --- | ---: | ---: | ---: | ---: |
| eqsat-basic | 11 | 11 | 2,714 | 0.38 ms |
| path | 13 | 9 | 2,585 | 0.23 ms |
| fibonacci | 11 | 9 | 2,063 | 0.22 ms |
| points-to | 20 | 13 | 4,646 | 0.44 ms |
| bignum | 3 | 1 | 615 | 0.05 ms |
| subsume | 9 | 7 | 2,024 | 0.20 ms |

These are single local debug-build measurements from before the comparator
optimizations, excluding program execution and export. They establish small-input
feasibility. The upstream [performance study](PERFORMANCE.md) measures release
builds through 16 million rows. Further optimization and measurements of ordinary
functions, certificates, and adversarial chains should precede expansion of this
pilot; the current exact refinement algorithm can still be quadratic.

## Requirements before broader adoption

1. **Container values.** The exporter rejects encountered container values.
   `set` and `unstable-fn` fail in the survey. Some other container programs export
   only their remaining scalar/constructor tables, so successful export does not
   establish container coverage. Add typed container observations: sequences for
   vectors/pairs, unordered elements for sets, multiplicities for multisets,
   key/value associations for maps, and target function identity for closures.
   Normalize unordered observations after substituting partition blocks, not by
   raw ID order. Extend certificate syntax/evaluation for those value forms.
2. **Term/proof projection.** Encoded tables are not the same schema as the user
   database; blindly dropping hidden tables is insufficient. Project constructor
   views and analysis functions back to their original signatures and canonical
   values, then compare that projection. The exporter currently rejects these
   modes. Several surveyed examples also fail earlier in encoded execution.
3. **Desugared relations.** `path` and `points-to` compare unequal after the
   harness-style resolve/print/reparse round trip. Relation desugaring marks its
   generated output sort non-unionable, but the printed sort does not preserve
   that metadata. The reparsed relation becomes a constructor to the exporter.
   Preserve this metadata in the round trip or export through an explicit schema
   projection before enabling comparisons across those treatments. Other eight
   successful normal exports compare equal to their desugared runs.
4. **Global-only values and custom sorts.** Global bindings and hidden tables
   are excluded. `map` demonstrates why this must remain explicit: an empty user
   database says nothing about values retained only by globals. Decide which
   user roots to expose as stable named observations. Custom base sorts need an
   injective, stable encoding; arbitrary `Debug` output is insufficient.
5. **Snapshot boundaries and policy.** Start at final program state, then add
   explicit checkpoints where intermediate changes matter. Keep full database
   equality for regression tests; use `terms_equal` to diagnose whether a change
   is confined to functions/activity. Costs are intentionally outside database
   equality and need their own assertions. Bisimulation ignores multiplicities
   of equivalent cycles; table-size assertions may intentionally remain stricter.

After those requirements are addressed, add semantic snapshots to selected
`Run::test_program` cases and remove their parallel snapshot skip. Expand coverage
only when their schemas and projections are supported; retain textual diagnostics
and extraction checks. This pilot provides a migration path without weakening
existing coverage or accepting unsupported exports as equal.
