# Backend performance eval

Benchmarks egglog across the cross product of **backend** × **mode** and
renders the results interactively with [eval-live](https://github.com/oflatt/eval-live).

|              | normal            | term-encoding              | proofs            |
| ------------ | ----------------- | -------------------------- | ----------------- |
| **bridge**   | *(no flags)*      | `--term-encoding`          | `--proofs`        |
| **duckdb**   | `--duckdb`        | `--duckdb --term-encoding` | `--duckdb --proofs` |

DuckDB is term-encoding-only, so `duckdb-normal` and `duckdb-term-encoding`
both run plain `--duckdb` (passing `--term-encoding` on top of `--duckdb`
panics the backend) and report ~the same numbers; both cells are kept so the
grid is a true cross product.

The benchmark binary is **`egglog-experimental`** (a CLI superset of plain
egglog), so the Herbie dumps' scheduler / `multi-extract` / `get-size!` forms
parse while mainline benchmarks run unchanged.

## The paper benchmark corpus

- **math-microbenchmark/math_full.egg** — runs (slow under duckdb/proofs).
- **herbie dumps** (`herbie/dump-egglog/dumps.tar.zst`, 1260 files) — run via
  egglog-experimental. Point `--path` at the tarball; it's auto-extracted to a
  sibling `.extracted/` dir (gitignored). Use `--limit N` to sample.
- **pointer-analysis/main.egg** — *excluded*: it `(input …)`s cclyzer++ CSV
  facts that aren't redistributed in this repo, so it can't run standalone.

## Run

```bash
pip install git+https://github.com/oflatt/eval-live.git   # once, for --serve

# benchmark every .egg under paper-benchmarks/ (release build), then view:
python3 eval/bench_backends.py --serve

# the Herbie dumps (auto-extracted from the tarball), sampling 20:
python3 eval/bench_backends.py paper-benchmarks/herbie/dump-egglog/dumps.tar.zst --limit 20

# point it at any file or directory:
python3 eval/bench_backends.py tests/web-demo --runs 3 --warmup 1

# re-open the viewer on existing results without re-running:
python3 eval/bench_backends.py --justserve
```

Useful flags: `--runs N`, `--warmup N`, `--timeout SECONDS`, `--debug` (use the
debug build), `--output PATH`, `--port N`.

A cell that errors or times out is recorded in the `errors` table instead of
`timings`. Results stream to `eval/results.json` after each benchmark, so you
can `--justserve` while a long run is still going.

## Files

- `bench_backends.py` — the driver (cross product, subprocess timing, JSON db, viewer).
- `graphs.py` — eval-live graphs/tables (runs in-browser via Pyodide).
- `results.json` — output (gitignored).
