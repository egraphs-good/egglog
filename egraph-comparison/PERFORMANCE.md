# Math comparison performance

This experiment uses `tests/math-microbenchmark.egg`, replacing its final
`(run 11)` with `(run N)` for every N from 1 through 12. Trailing print commands
are removed. Export happens once, independently of timed comparison. Each input
is compared with an equivalent copy whose class IDs and row order are changed.
The same files are reused across implementations.

The benchmark separates the public `compare` API (including validation, graph
setup, refinement, and cleanup), validation of both inputs, parsing both JSON
buffers, and a fresh CLI process. Ordinary parsing measurements also drop the
parsed databases; the single-pass large-input measurement reports initial parse
time without that cleanup. CLI timings include process startup, file reads,
parsing, comparison, and cleanup. Certificates are not requested.

Environment: Apple M4 Max, 16 CPU cores, 128 GiB RAM, macOS 26.6.2, Rust 1.91.0,
Cargo release profile, standard system allocator. Comparisons are single-threaded.
N=1..11 use three samples after warmup, with repeated calls targeting 100 ms per
sample for small cases. N=12 uses one pass because it contains 15,987,528 rows
and its two JSON files total about 5.9 GiB. These measurements describe this
workload and machine, not a general asymptotic guarantee.

## Results

Public `compare` API times include validating both databases. Counts describe
one input; each comparison processes two. N=1..11 are medians, N=12 is a single
observation. [Machine-readable results](benchmarks/math-results.csv) retain
the timings before rounding.

| Run | Classes | Rows | Rounds | Before (ms) | After (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58 | 69 | 4 | 0.404 | 0.033 | 12.3x |
| 2 | 79 | 118 | 4 | 0.633 | 0.051 | 12.5x |
| 3 | 124 | 208 | 5 | 1.296 | 0.094 | 13.7x |
| 4 | 205 | 389 | 6 | 2.818 | 0.183 | 15.4x |
| 5 | 369 | 784 | 6 | 5.896 | 0.357 | 16.5x |
| 6 | 674 | 1,576 | 7 | 13.471 | 0.808 | 16.7x |
| 7 | 1,355 | 3,160 | 8 | 30.274 | 1.800 | 16.8x |
| 8 | 3,584 | 8,113 | 9 | 85.825 | 5.528 | 15.5x |
| 9 | 12,453 | 28,303 | 10 | 346.037 | 20.484 | 16.9x |
| 10 | 58,472 | 136,446 | 11 | 2,107.824 | 134.943 | 15.6x |
| 11 | 443,840 | 1,047,896 | 12 | 19,488.369 | 1,523.311 | 12.8x |
| 12 | 6,865,591 | 15,987,528 | 13 | 371,260.894 | 34,521.299 | 10.8x |

Both implementations report term and database equality at every size, with
identical refinement round counts. The optimized constructor-only path executes
one refinement and reports its round count for both results.

At N=12, the single-pass benchmark's total process time fell from 382.7 s to
46.3 s. One-second RSS sampling observed peaks of 61.6 GiB before and 22.9 GiB
after. These are approximate process peaks, including two parsed databases and
5.85 GiB of retained JSON buffers, not just refinement allocations. Initial
parsing took 8.74 s before and 8.46 s after; the parser in this benchmark did
not change. This largest case was measured once per implementation.

Fresh CLI process times, including file I/O, parsing, and cleanup:

| Run | Before | After |
| ---: | ---: | ---: |
| 10 | 2.274 s | 0.225 s |
| 11 | 21.140 s | 2.334 s |
| 12 | Not measured | 50.27 s |

The CSV includes CLI timings for the smaller cases as well. N=12's CLI timing
is a single wall-clock observation; other CLI entries are three-sample medians
after discarding the first run. Blank CSV cells mean a measurement was omitted.
All comparisons and CLI runs returned both equality flags as true.

## Changes guided by the measurements

The original N=10 comparison took 2.108 s; validation took 0.162 s and parsing
both buffers took 0.080 s. A ten-second `sample` profile showed allocation/free
routines and string comparisons among the largest leaf costs.

1. Reuse the constructor partition when the full database has exactly the same
   observations. This isolated change reduced N=10 to 1.036 s (2.0x). Ordinary
   functions and subsumption still trigger separate full-database refinement.
2. Intern complete operator/literal labels once across both inputs. Resolve
   serialized class IDs once; borrow their strings instead of copying them into
   internal graphs. Common node and signature arities use inline storage.
3. Sort and deduplicate compact class signatures, interning them in a hash table
   with exact key equality. Reuse scratch space for repeated signatures. Hash
   collisions cannot make unequal graphs compare equal.
4. Use borrowed hash indexes for repeated validation lookups. Keep the public
   serialization ordered and preserve its validation rules.
5. Mark output-class membership densely. At the fixed point, matching output
   blocks already implies matching rows modulo those blocks; rebuilding a large
   string-keyed row set is redundant. Release the constructor graph before
   building the full-function graph when a second pass is needed.
6. Read each CLI input into a byte buffer and parse the slice. Interleaved trials
   on the optimized comparator reduced N=10 CLI time from 291 ms to 222 ms and
   N=11 from 2.69 s to 2.25 s. This trades transient memory for parsing speed:
   one serialized file is buffered at a time and dropped before the next read.

These are improvements to setup and the existing whole-graph refinement loop.
They do not implement a splitter worklist or Hopcroft's smaller-half optimization.
Long refinement chains can still require quadratic work.

The next algorithmic improvement would be a predecessor/splitter worklist that
revisits affected classes instead of rescanning every row each round. It needs
separate correctness and performance evaluation, particularly for set-valued
class signatures and cycles. At N=11, validation is now 242 ms of the 1,523 ms
comparison, while parsing both buffers costs another 636 ms. For repeated
in-process snapshot comparisons, retaining parsed/validated input is another
option; a fresh CLI still pays the JSON and allocation costs. This workload
contains constructors only, so it does not quantify speedups on databases with
ordinary functions, subsumption, or requested disequality certificates.

## Canonical serialization

The [canonical benchmark](CANONICAL-PERFORMANCE.md) extends this corpus to measure
independent canonicalization of both inputs, checking a fresh input against saved
canonical bytes, and byte comparison alone. It reports the slowdown relative to
the pairwise API measured in the same build and session.

## Reproduction

```sh
cargo build --release -p egglog -p egraph-comparison --bins
cargo build --release -p egraph-comparison --example benchmark
python3 scripts/bench_egraph_comparison.py generate
python3 scripts/bench_egraph_comparison.py measure --label current --samples 3 --runs 1 2 3 4 5 6 7 8 9 10 11
/usr/bin/time -l target/release/examples/benchmark target/comparison-performance/math-12.comparison.json target/comparison-performance/math-12.renamed.json --single-pass
```

Save the baseline binaries before changing implementations and pass their
directory with `--bin-dir` to measure them against the same corpus. The initial
production baseline was built at commit `b3b6cad1`; its comparator and exporter
are identical to implementation commit `b3f9c619` (before the snapshot
pilot). Reordering the stack does not change the measured implementation.
Data and raw JSON results stay under
`target/comparison-performance/`; the corpus takes several GB of disk space.
For sampling, run the example with `--profile-seconds 15` and attach `sample`
(or the platform's sampling profiler) to that process.
