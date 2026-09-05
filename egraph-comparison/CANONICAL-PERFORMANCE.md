# Canonical serialization performance

This extends the [math comparison benchmark](PERFORMANCE.md) using the same saved
run-1..12 databases and their independently renamed, row-reversed equivalents.
All comparisons use full-database mode. Both equality flags and the canonical
bytes agree for every input pair. The machine, compiler, release profile, and
single-threaded execution are the same as the earlier study.

## What is measured

- **Pairwise:** the existing public `compare(left, right)` API.
- **Canonical pair:** canonicalize both inputs, compare their bytes, and free
  the outputs. This is the cost of replacing an uncached pairwise comparison.
- **Saved baseline:** canonicalize only the fresh right input, compare it with
  previously generated left canonical bytes, and free the fresh output. Baseline
  creation is excluded, matching the pilot's ordinary success path.
- **Bytes only:** compare two distinct allocations containing the complete
  canonical output. This measures equality of already-canonical inputs.

Every API call includes validation and allocation/cleanup of its internal graph.
Input databases and original JSON buffers are retained throughout the benchmark;
parsing, file I/O, egglog execution/export, and process startup are excluded from
these API timings. A real saved-baseline test additionally avoids parsing the
old raw database, which this benchmark does not credit. Failure diagnostics and
certificates are also excluded. The workload contains constructors only; it does
not predict costs for function-heavy databases or long refinement chains.

Runs 1..11 use three samples after warmup, targeting 100 ms per sample on small
inputs. Run 12 uses one pairwise call and one canonicalization per side. Its
canonical-pair time is the sum of the baseline-generation and fresh-input passes,
including comparison and freeing both outputs; it is not a separately repeated
measurement. Bytes-only comparison still uses three samples. Do not interpret
small differences between separate benchmark sessions as code speedups.

## Results

Canonicalizing **both** inputs is 1.27–1.78x the cost of pairwise comparison
across this corpus. Checking a **saved canonical baseline** takes 0.64–0.87x the
pairwise time (13–36% faster). The extra generation cost is therefore worthwhile
on this workload when the accepted baseline is reused.

Times below are milliseconds; ratios use the pairwise time in the same row.
[Unrounded results and sample ranges](benchmarks/canonical-math-results.csv).

| Run | Pairwise (ms) | Canonical pair (ms) | Ratio | Saved baseline (ms) | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.034 | 0.061 | 1.78x | 0.030 | 0.87x |
| 2 | 0.052 | 0.085 | 1.65x | 0.042 | 0.82x |
| 3 | 0.097 | 0.156 | 1.61x | 0.076 | 0.79x |
| 4 | 0.191 | 0.298 | 1.56x | 0.149 | 0.78x |
| 5 | 0.373 | 0.573 | 1.54x | 0.277 | 0.74x |
| 6 | 0.848 | 1.270 | 1.50x | 0.605 | 0.71x |
| 7 | 1.894 | 2.868 | 1.51x | 1.440 | 0.76x |
| 8 | 5.795 | 8.788 | 1.52x | 4.394 | 0.76x |
| 9 | 21.974 | 33.614 | 1.53x | 16.764 | 0.76x |
| 10 | 147.911 | 215.606 | 1.46x | 111.142 | 0.75x |
| 11 | 1804.620 | 2436.700 | 1.35x | 1244.455 | 0.69x |
| 12 | 40494.732 | 51569.106 | 1.27x | 25893.724 | 0.64x |

At run 12 (6,865,591 classes and 15,987,528 rows per input):

- Pairwise comparison: **40.49 s**; canonical pair: **51.57 s**, **27% slower**.
- Saved-baseline check: **25.89 s**, **36% faster** than pairwise comparison.
  Creating the canonical baseline itself took **25.68 s**.
- Already-canonical byte equality: **9.43 ms** median, with 9.38–9.45 ms range.
- Canonical output: **517,844,615 bytes** (493.85 MiB), **17.4%** of the original
  2,975,308,986-byte pretty JSON input. The renamed raw file is larger still.
- The complete benchmark process took **105.05 s**, with a **25.86 GiB** peak
  observed by one-second RSS sampling. This includes the retained input buffers,
  two parsed databases, all comparison paths, and canonical output allocations;
  it is not an isolated peak for a production saved-baseline check.

The earlier study's run-12 pairwise time was 34.52 s. This session's 40.49 s
measurement is the denominator above; ratios do not mix the earlier session
with the new canonical timings. Run 12 has only one observation for each API
path, so the percentages should be treated as estimates.


## Interpretation and next steps

Canonical generation performs extra ordering work: it sorts unique block
signatures on every round and writes the minimized quotient. Pairwise comparison
can assign block numbers in discovery order and return booleans. These timings
measure the total tradeoff; they do not isolate signature sorting in a CPU
profile. Saved baselines offset that extra work by processing only one graph.

The pilot's three canonical files total 2,972 bytes versus 7,365 bytes of raw
pretty JSON. Retaining raw diagnostic sidecars makes combined storage 10,337
bytes (1.40x the original pilot). A canonical-graph certificate reader could
remove the sidecars. The canonical format uses compact integer references;
size reductions against pretty input JSON reflect that representation as well
as quotient minimization, and are not a compression benchmark.

The new API shares the existing whole-graph refinement strategy. Future work
should measure predecessor/splitter refinement and alternative ways of assigning
canonical ranks, retaining exact signature equality. Full-function inputs,
disequality diagnostics, and adversarial refinement chains still need their own
performance studies before expanding the snapshot pilot.

## Reproduction

```sh
cargo build --release -p egraph-comparison --example benchmark --bin egraph-comparison
# Reuse the corpus generated by the original benchmark; generate it if missing.
python3 scripts/bench_egraph_comparison.py measure --canonical --label canonical --samples 3 --runs 1 2 3 4 5 6 7 8 9 10 11 12
# Standalone largest case, also reporting process resources on macOS:
/usr/bin/time -l target/release/examples/benchmark target/comparison-performance/math-12.comparison.json target/comparison-performance/math-12.renamed.json --canonical --single-pass --samples 3
```

The script automatically selects a single pass for run 12 with `--canonical` and
skips the unrelated CLI comparison measurements. Raw measurements stay under
`target/comparison-performance/`. The committed CSV preserves unrounded medians,
min/max ranges, canonical sizes, and the single-pass measurement distinction.
