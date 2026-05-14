# Original egglog paper benchmarks

Benchmarks from the PLDI 2023 paper [*Better Together: Unifying Datalog and
Equality Saturation*](https://doi.org/10.1145/3591239) by Zhang, Wang, Flatt,
Cao, Zucker, Rosenthal, Tatlock, and Willsey.

Pulled from the paper's [Zenodo artifact](https://zenodo.org/records/7709794)
(DOI `10.5281/zenodo.7709794`). Only the inputs that the paper actually
benchmarks are copied here — Souffle binaries, the egg / Herbie / cclyzer++
codebases, and the Docker harness are not.

## Layout

| Directory                                          | Paper section | What it measures                                                                           |
| -------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| [`math-microbenchmark/`](math-microbenchmark/)     | §5.3, Fig. 7  | Saturation throughput on the egg `math` benchmark — egglog vs egglogNI vs egg.             |
| [`pointer-analysis/`](pointer-analysis/)           | §6.1, Fig. 8  | Steensgaard points-to analysis on real binaries — egglog vs `eqrel`/`cclyzer++`/`patched`. |
| [`herbie/`](herbie/)                               | §6.2, Fig. 11/12 | Herbie's floating-point error-reduction suite with egglog as the EqSat engine.          |

Each subdirectory has its own `README.md` with the file list and what's needed
to run it.

## Comparison to in-tree tests

A few of these benchmarks already exist in `tests/` in adapted forms:

| Paper benchmark                                              | In-tree equivalent                                                 | Difference                                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `math-microbenchmark/math_full.egg`                          | [`tests/math-microbenchmark.egg`](../tests/math-microbenchmark.egg)| Constants use `i64` (`(Const 0)`) instead of `Rational` (`(Const (rational 0 1))`); identical rules otherwise. |
| `pointer-analysis/main.egg`                                  | [`tests/web-demo/points-to.egg`](../tests/web-demo/points-to.egg)  | The in-tree version is a 62-line toy demo. `main.egg` here is the 200-line paper version that consumes `cclyzer++` CSV facts. |
| `herbie/bench/`                                              | [`tests/web-demo/herbie.egg`](../tests/web-demo/herbie.egg)        | `herbie.egg` is an egglog port of Herbie's simplification *layer*. The `.fpcore` files here are Herbie's *input* programs, fed through the full Herbie pipeline. |

## What's omitted

- **Pointer-analysis `benchmark-input/`** — 2 GB of LLVM bitcode for 30 PostgreSQL
  and 106 coreutils binaries. Not redistributed here; see
  [`pointer-analysis/README.md`](pointer-analysis/README.md) for how to fetch.
- **Souffle binaries, cclyzer++, eqlog source.** All vendored in the Zenodo
  artifact's Docker image. Reproducing the paper plots needs that harness.
