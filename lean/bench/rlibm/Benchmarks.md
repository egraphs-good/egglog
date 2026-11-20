# bfloat16 logarithm benchmarks

The comparison uses Rutgers' original bfloat16 implementations of `log2` and `log10` at commit `8533ec0db38b2f3ab55c1b296ddaec80d578e251`. They normalize with `frexpf`, apply the Cody–Waite transformation, and evaluate a polynomial in binary64. The verified routines use LP-generated fixed-point coefficients. Thus the comparison against Rutgers measures the entire implementation choice. The Horner, Estrin and split variants use the source cubic and change its evaluation order. The `cells` schedule changes to sixteen quadratic segments and a different coefficient table. Its equivalence holds at the final rounding step. [Rutgers log2](https://github.com/rutgers-apl/rlibm/blob/8533ec0db38b2f3ab55c1b296ddaec80d578e251/source/bfloat16/log2.cpp) · [Rutgers log10](https://github.com/rutgers-apl/rlibm/blob/8533ec0db38b2f3ab55c1b296ddaec80d578e251/source/bfloat16/log10.cpp)

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
