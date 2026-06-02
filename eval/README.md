# Backend performance eval

Benchmarks egglog across the cross product of **backend** × **mode** and
renders the results interactively with [eval-live](https://github.com/oflatt/eval-live).

|              | normal            | term-encoding              | proofs            |
| ------------ | ----------------- | -------------------------- | ----------------- |
| **bridge**   | *(no flags)*      | `--term-encoding`          | `--proofs`        |
| **duckdb**   | `--duckdb`        | `--duckdb --term-encoding` | `--duckdb --proofs` |

DuckDB is term-encoding-only, so `duckdb-normal` and `duckdb-term-encoding`
drive the same engine and report ~the same numbers; both are kept so the grid
is a true cross product.

## Run

```bash
pip install git+https://github.com/oflatt/eval-live.git   # once, for --serve

# benchmark every .egg under paper-benchmarks/ (release build), then view:
python3 eval/bench_backends.py --serve

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
