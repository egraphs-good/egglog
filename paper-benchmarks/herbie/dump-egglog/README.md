# Herbie-generated egglog programs

`dumps.tar.zst` contains **1260 egglog programs** that
[Herbie 2.3](https://github.com/herbie-fp/herbie/tree/main) emits when
processing its standard benchmark suite (`bench/`) with the egglog backend
enabled. Each `.egg` file inside the tarball is one egglog session that
Herbie spawned for a single rewriting stage of a single benchmark input.

```
$ tar -tf dumps.tar.zst | head -3
dump-egglog/
dump-egglog/rewrite0.egg
dump-egglog/rewrite1.egg
```

Compressed size: ~2.8 MB (151 MB uncompressed). The prelude — datatype `M`
with ~120 constructor declarations and the bulk of the rewrite ruleset — is
identical across files, so zstd hits ~2% ratio.

## How they were generated

```bash
# 1. Install the egglog-experimental fork (the bin Herbie spawns).
cargo install --git https://github.com/egraphs-good/egglog-experimental \
              --branch main egglog-experimental

# 2. Clone + build Herbie 2.3.
git clone --depth 1 https://github.com/herbie-fp/herbie
cd herbie
make egg-herbie     # builds the egg-herbie Rust crate + raco-installs it
make update         # raco-installs Herbie itself

# 3. Run Herbie on every .fpcore in bench/, with both flags:
#    --enable generate:egglog : actually use egglog (default is egg)
#    --enable dump:egglog     : write each egglog session to dump-egglog/<label><N>.egg
racket src/main.rkt report --enable generate:egglog \
                           --enable dump:egglog \
                           --timeout 60 \
                           bench/ /tmp/out
```

This sweep took ~58 min on an M1 laptop, ran 730 benchmark expressions, and
emitted 4047 `.egg` files. **2782** of those came from benchmarks where
Herbie hit the per-input 60-second budget (`[TIMEOUT]` / `[CRASH]` lines in
`herbie-run.log`) — egglog received those programs but the wider Herbie run
was aborted. Those dumps were dropped (see "Filtering" below) and only the
**1260** corresponding to the **314 successfully-completed benchmarks** are
in the tarball.

## Filtering

Dump filenames are sequential (`rewrite0.egg`, `rewrite1.egg`, …,
`taylor0.egg`, …) and Herbie doesn't embed the benchmark name. To map dumps
→ benchmarks we walked the run in wall-clock order: each line in
`herbie-run.log` gives a status (`OK`/`TIMEOUT`/`CRASH`) and duration for
benchmark *i*, and the dump-file mtimes line up to within ~2% of the
cumulative timing. Dumps whose mtime fell inside a successful benchmark's
window were kept; everything else (timeouts, crashes, post-last-benchmark
tail) was discarded.

The Herbie log we used (`herbie-run.log`) is checked in alongside the
tarball so this reasoning is reproducible.

## ⚠ Compatibility — these are NOT mainline-egglog programs

Each session opens with:

```scheme
(let-scheduler bo (back-off))
...
(run-schedule (saturate unsound))
(run-schedule (saturate lower))
(run-schedule
 (repeat 50
  (seq
   (run-with bo rewrite :until (<= 4000 (get-size!)))
   (run-with bo const-fold :until (<= 4000 (get-size!)))
   ...)))
(multi-extract …)
```

The forms `let-scheduler`, `back-off`, `run-with … :until`, `get-size!`, and
`multi-extract` are all
[egglog-experimental](https://github.com/egraphs-good/egglog-experimental)
extensions. Mainline egglog (this repo) won't parse them — it errors on the
first `(let-scheduler …)` line with `expected either saturate, seq, repeat,
or run`.

To run a session through egglog-experimental:

```bash
tar -xf dumps.tar.zst -C /tmp/
egglog-experimental /tmp/dump-egglog/rewrite0.egg
```

To run through this repo's mainline egglog the experimental forms have to be
lowered — e.g. replace `(run-with bo R :until …)` with
`(run-schedule (saturate R))`, drop `multi-extract` / `get-size!`. We
haven't done that here.
