# Herbie / floating-point (PLDI 2023 §6.2, Fig. 11 & 12)

Herbie is a floating-point error-reduction tool that uses equality saturation
internally. The paper drops in egglog as Herbie's EqSat engine (replacing
egg) and compares accuracy + runtime across Herbie's standard benchmark
suite.

- **Figure 11** — error difference (egglog with sound analysis vs Herbie with
  the unsound rules). Centered near zero; 104 cases where egglog beats unsound
  Herbie, 135 where unsound Herbie beats egglog.
- **Figure 12** — runtime difference (seconds). egglog is slightly faster
  overall (73.91 min vs 81.91 min).

The paper's headline claim is *parity*: egglog can host Herbie's EqSat
without measurable accuracy regression, while staying sound by leveraging
interval analysis + a `≠`-analysis on top of egg's rules.

## Files

`bench/` contains the 289 floating-point input programs Herbie evaluates,
written in [FPCore](https://fpbench.org/). They're grouped by source:

```
bench/
├── demo.fpcore         ; standalone showcase exprs
├── tutorial.fpcore     ; from Herbie's tutorial
├── regression.fpcore   ; regression tests
├── haskell.fpcore      ; Haskell stdlib expressions
├── pbrt.fpcore         ; from the PBRT renderer
├── 2cos.fpcore         ; the 2cos motivating example
├── hamming/            ; FP gotchas from Hamming's textbook
├── libraries/          ; math.js, octave, jmatjs, ...
├── mathematics/        ; statistics, hyperbolic-functions, ...
├── numerics/           ; Kahan / Rump / Martel / libm / ...
└── physics/            ; physical-formula benchmarks
```

These are **Herbie inputs**, not egglog programs. Each `.fpcore` entry
defines a real-valued expression (e.g.
`(FPCore (x) :name "neg log" (- (log (- (/ 1 x) 1))))`). Herbie internally
lowers these into rewrite problems, builds an e-graph, runs equality
saturation, and extracts the candidate with the lowest measured FP error.

## Reproducing the paper run

The paper's setup vendors a Herbie fork (`herbie-eqlog/` in the Zenodo
artifact) where the egg backend has been swapped for egglog. The fork
inherits Herbie's `Makefile`; `make herbie-report` runs the suite end-to-end
and emits `errorhist.pdf` / `runtime.pdf`.

## Captured egglog sessions (`dump-egglog/`)

Today's upstream Herbie supports egglog directly via the
`--enable generate:egglog` flag, and `--enable dump:egglog` makes it write
every egglog session to a file. We ran Herbie 2.3 over `bench/` with both
flags and packaged the **1260 sessions from successfully-completed
benchmarks** (314 of 730 — Herbie's 60-second per-input budget timed out
the rest) into
[`dump-egglog/dumps.tar.zst`](dump-egglog/dumps.tar.zst). See
[`dump-egglog/README.md`](dump-egglog/README.md) for regeneration steps and
the compatibility caveat — these sessions use
`egglog-experimental` features (`let-scheduler`, `back-off`, `run-with`,
`get-size!`, `multi-extract`) and **won't parse** through mainline egglog
without lowering.
