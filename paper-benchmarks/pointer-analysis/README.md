# Steensgaard pointer analysis (PLDI 2023 §6.1, Fig. 8)

Unification-based points-to analysis written in egglog, compared against three
Souffle-based encodings on real-world binaries.

## Configurations (Fig. 8 bar groups)

- **egglog** — `main.egg`, the egglog version.
- **eqrel** — uses Souffle's `eqrel` keyword to maintain equivalence
  explicitly. Times out on all but one input.
- **cclyzer++** — the original `cclyzer++` Datalog encoding (choice domain +
  subsumptive rules + custom equivalence). Fast but semantically unsound on the
  `load` instruction.
- **patched** — the same as `cclyzer++` but with the bug fixed by reintroducing
  `eqrel` and adding a congruence-related rule. Sound but slower.
- **egglogNI** — egglog with semi-naïve disabled, to isolate the contribution
  of incremental evaluation.

The paper reports egglog at 4.96× over the sound `patched` baseline on average,
1.94× over `cclyzer++` (unsound), and 1.59× over `egglogNI`.

## Files

- `main.egg` — the 200-line egglog program. Reads cclyzer++-style CSV fact
  files (`function.csv`, `instruction_in_function.csv`, `assign_instruction.csv`,
  …) declared via `(input … "<csv>")`.
- `types.dl` — shared Souffle type declarations.
- `mini-cclyzerpp.dl` — the **eqrel** Souffle baseline.
- `naive-cclyzerpp.dl` — a naïve Datalog baseline (no eqrel).
- `sound-cclyzerpp.dl` — the **patched** Souffle baseline (sound version of
  cclyzer++).

(The full `cclyzer++` baseline is too large to vendor; the paper points at the
upstream [`cclyzer++` repo](https://github.com/GaloisInc/cclyzer).)

## Inputs (not vendored)

The paper's Figure 8 runs all five configurations on every program here:

**PostgreSQL 9.5.2** (30 binaries, ~2 GB of LLVM bitcode):

```
clusterdb            createdb         createlang        createuser
dict_snowball.so     dropdb           droplang          dropuser
ecpg                 initdb           libecpg_compat    libecpg
libpgtypes           libpq            libpqwalreceiver  pg_basebackup
pg_ctl               pg_dump          pg_dumpall        pg_isready
pg_receivexlog       pg_recvlogical   pg_restore        pg_rewind
pg_upgrade           pgbench          plpgsql.so        psql
reindexdb            vacuumdb
```

**coreutils 8.24** (106 binaries) — used by the `pointer-analysis-benchmark-small`
variant.

These are **LLVM bitcode (`.bc`) files** lowered from the upstream sources by
the artifact's Dockerfile (it builds PostgreSQL/coreutils with `wllvm` and
extracts `.bc`). The bitcode is then fed through `cclyzer++`'s frontend to
produce the CSV fact tables that `main.egg`'s `(input …)` directives ingest.

To fetch the prebuilt `.bc` files, pull
[`artifact.tar.gz`](https://zenodo.org/api/records/7709794/files/artifact.tar.gz/content)
from the [Zenodo artifact](https://zenodo.org/records/7709794) and look in
`artifact/pointer-analysis-benchmark/benchmark-input/`. They're omitted here
because the directory is roughly the size of the Zenodo artifact itself
(2 GB).

## Running

The paper's harness is `run.py` in the Zenodo artifact, which orchestrates
running cclyzer++ on each bitcode, dumping facts, then invoking each
configuration on those facts and timing the result. Reproducing the full
Figure 8 takes >1 hour even on x86; a small variant on coreutils runs in
minutes.
