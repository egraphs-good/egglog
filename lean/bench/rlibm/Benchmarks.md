# bfloat16 logarithm benchmarks

The comparison uses Rutgers' original bfloat16 implementations of `log2` and `log10` at commit `8533ec0db38b2f3ab55c1b296ddaec80d578e251`. They normalize with `frexpf`, apply the Cody–Waite transformation, and evaluate a polynomial in binary64. The verified routines use LP-generated fixed-point coefficients. Thus the comparison against Rutgers measures the entire implementation choice. The Horner, Estrin and split variants use the source cubic and change its evaluation order. The `cells` schedule changes eight cubic segments (256 coefficient bytes per function) into sixteen quadratic segments (384 bytes). Its equivalence holds at the final rounding step. [Rutgers log2](https://github.com/rutgers-apl/rlibm/blob/8533ec0db38b2f3ab55c1b296ddaec80d578e251/source/bfloat16/log2.cpp) · [Rutgers log10](https://github.com/rutgers-apl/rlibm/blob/8533ec0db38b2f3ab55c1b296ddaec80d578e251/source/bfloat16/log10.cpp)

## Results

On an Apple M4 Max, MPFR 4.2.2 agreed on every returned word for both functions, the five generated schedules, their native Lean evaluations and the upstream implementations. The tables give median nanoseconds per call over 12 timed samples per cell. CPU idle samples before each rotation ranged from 79.4% to 92.8%.

### Emitted C and Rutgers

| Implementation | log2 positive | log2 all | log2 chain | log10 positive | log10 all | log10 chain |
|---|---:|---:|---:|---:|---:|---:|
| Rutgers | 16.58 | 11.16 | 20.62 | 18.47 | 11.51 | 22.99 |
| source | 3.00 | 3.46 | 12.20 | 3.07 | 3.58 | 12.13 |
| horner | 3.02 | 3.61 | 12.19 | 3.06 | 3.56 | 12.15 |
| estrin | 3.00 | 3.55 | 12.29 | 3.07 | 3.59 | 12.12 |
| split | 3.02 | 3.57 | 12.30 | 3.05 | 3.54 | 12.13 |
| cells | 2.86 | 3.47 | 11.80 | 2.93 | 3.50 | 11.90 |

The fastest emitted kernels for positive finite throughput were log2 cells at 0.17× upstream time and log10 cells at 0.16× upstream time. These ratios include normalization, polynomial evaluation and final rounding.

Apple Clang 17 emitted identical AArch64 instruction bytes and coefficient-table relocations for source, Estrin and split on log2, and for source and Estrin on log10. Small timing differences between those kernels reflect measurement variability. The quadratic uses a different table and two dependent polynomial multiply-adds in place of three. The generated object can be inspected with `otool -tv build/schedules.o` and `otool -rv build/schedules.o` from the output directory.

### Native Lean and Rutgers

| Implementation | log2 positive | log2 all | log2 chain | log10 positive | log10 all | log10 chain |
|---|---:|---:|---:|---:|---:|---:|
| Rutgers | 16.92 | 11.23 | 20.75 | 19.15 | 12.02 | 23.09 |
| source | 560.97 | 281.03 | 565.61 | 546.18 | 274.30 | 559.02 |
| horner | 688.15 | 342.57 | 687.60 | 671.13 | 335.84 | 674.49 |
| estrin | 725.56 | 362.71 | 726.92 | 687.90 | 349.82 | 708.04 |
| split | 911.30 | 456.95 | 915.11 | 891.10 | 453.75 | 899.72 |
| cells | 581.24 | 295.06 | 578.51 | 549.92 | 287.33 | 559.80 |

Across the native extracted workers, median time to run equality saturation and prepare the reusable evaluator was 49.1 µs. Preparation was outside all per-call samples. Native Lean includes the cost of its integer representation and runtime expression traversal. [Preparation and execution samples](results/samples.csv)

## Reproduction

Install the pinned Lean toolchain, Python 3, GCC or Clang with C11 and C++17 support, pkg-config, MPFR and GMP. The C kernels use count-leading-zero builtins. The CPU load sampler supports Darwin and Linux. Supply a fresh output directory. On a Mac with a conflicting SDK on the library path, use `env -u LIBRARY_PATH` before the Python command.

```sh
git clone https://github.com/rutgers-apl/rlibm.git /tmp/rlibm-upstream
git -C /tmp/rlibm-upstream checkout --detach 8533ec0db38b2f3ab55c1b296ddaec80d578e251
python3 lean/bench/rlibm/run.py /tmp/rlibm-upstream /tmp/rlibm-bench
```

Add `--verify-only` to build the same executables and skip timings. The harness runs both Lean emitters, compiles the emitted C and the two unchanged Rutgers C++ sources as separate translation units with the same optimization flags, and builds the native Lean evaluation worker. It retains the actual generated source, coefficient header, schedule manifest, oracle words and measured executable in the output directory's `build/` folder. [Harness](run.py) · [Driver and MPFR oracle](driver.cpp) · [Native Lean worker](../../RlibmBench.lean)

## Exhaustive validation

The observable is the returned 16-bit encoding under round-to-nearest, ties-to-even. For each function, MPFR evaluates the exact input dyadic directly at eight significand bits in round-to-nearest, ties-to-even mode. Apart from `log(1) = +0`, the mathematical answer for each positive finite bfloat16 input rounds to a normal, finite bfloat16 number. The oracle checks that this result converts exactly to binary32 with the low 16 bits clear, then reads its bfloat16 encoding. Both zeros map to negative infinity. Positive infinity stays positive infinity. For negative nonzero inputs and all NaN encodings, the expected word is `0xffff`, matching the upstream payload convention.

Every implementation is checked word by word for all 65,536 encodings, including 32,512 positive normals and 127 positive subnormals. The native Lean worker checks the exact same oracle words. The CSV also records a 64-bit hash of each full output vector to help compare independent reproductions. [Per-variant checks](results/validation.csv)

## Timing boundaries

- **Positive throughput** shuffles the 32,639 positive finite nonzero inputs, evaluates each complete function independently, and sums the returned encodings.
- **All-encoding throughput** shuffles all 65,536 words. Roughly half take a special-value path.
- **Dependent positive-input chain** uses the preceding output and a changing offset to select the next positive finite argument. It measures serial dependence including the next input's address calculation and load. The throughput and chain workloads therefore measure different costs.

All three distributions run for both execution paths. The C and native Lean workers read the same input sequence. C reads an array of 16-bit words, and the Lean API reads an `Array Nat` of the same words. The native Lean `source` calls the original integer kernel. The other Lean workers interpret the selected AST using the compiled Lean implementation. Each worker prepares its function once. The ledger reports preparation separately from evaluation.

All timing workers run in fresh single-threaded processes. Each performs eight untimed warm-up sweeps on its chosen input distribution. The measured section covers only function calls, input reads and output accumulation. Builds, e-graph search, oracle evaluation, input construction, validation and process startup are outside that section. Checksums must agree across schedules for each function and workload. Each distribution has a C group and a native Lean group. Each group measures Rutgers as its own reference, using the same sweep count as its candidates. In each block every candidate occupies every start position once. The second block reverses the initial ordering. Before each rotation the harness samples overall CPU idle for one second and stops below 60%. [Every sample](results/samples.csv) · [Machine and flags](results/machine.json) · [All medians](results/summary.csv)
