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

## Reproducing

The paper's setup vendors a Herbie fork (`herbie-eqlog/` in the Zenodo
artifact) where the egg backend has been swapped for egglog. The fork
inherits Herbie's `Makefile`; `make herbie-report` runs the suite end-to-end
and emits `errorhist.pdf` / `runtime.pdf`.

Today (post-paper), upstream Herbie supports egglog directly — the user's
suggestion was to point at running Herbie against modern egglog and extract
the egglog programs it generates per benchmark. Those generated `.egg` files
aren't in the Zenodo artifact (they're transient per-input outputs), so
collecting them would mean instrumenting Herbie. Out of scope for this
folder.
