# Math micro-benchmark (PLDI 2023 §5.3, Fig. 7)

Saturation throughput on the egg-derived `math` rewrite system. The paper
compares three configurations:

- **egglog** — semi-naïve evaluation enabled.
- **egglogNI** — egglog with semi-naïve disabled (the paper's claim is that
  semi-naïve gives a substantial speedup; this is the control).
- **egg** — egg's own runner over the equivalent ruleset.

The benchmark loops increase the seed term's depth and measure time/space to
reach saturation.

## Files

- `math_full.egg` — the egglog program. Datatype `Math` with `Diff`, `Integral`,
  `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Ln`, `Sqrt`, `Sin`, `Cos`, `Const`
  (`Rational`), `Var` (`String`). Rewrite rules for commutativity,
  associativity, distributivity, identity, and the standard differentiation /
  integration rules.

`tests/math-microbenchmark.egg` in this repo is essentially the same program
but with `i64` constants instead of `Rational` — see the top-level README.

## How the paper ran it

Driver code lives in `micro-benchmarks/src/main.rs` of the Zenodo artifact. It
parses `math_full.egg`, seeds it with progressively deeper test terms, and
records iteration count + wall time per configuration. The companion
`benchmarks.csv` in the artifact has the raw numbers.
