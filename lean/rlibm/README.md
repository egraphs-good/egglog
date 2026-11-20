# Verified bfloat16 logarithms

This example synthesizes correctly rounded `log2` and `log10` routines, then uses the Lean compiler DSL from the parent PR to change their evaluation and range reduction. The input and output are raw bfloat16 words. The contract is round-to-nearest, ties-to-even, on all 65,536 encodings of each function. The measurements compare exported C and native Lean with the original Rutgers bfloat16 library. [RLibm paper](https://www.cs.yale.edu/homes/lim-jay/publications/rlibm_POPL2021.pdf) · [Pinned upstream source](https://github.com/rutgers-apl/rlibm/tree/8533ec0db38b2f3ab55c1b296ddaec80d578e251/source/bfloat16) · [Benchmarks and reproduction](../bench/rlibm/Benchmarks.md)

## How to read the example

1. **Synthesize the baseline.** Write a positive finite input as `2^e × (1 + m/128)`, where `0 ≤ m < 128`. The baseline uses eight cubic segments. A segment covers 16 adjacent mantissas. With local coordinate `t = m % 16`, its accumulator is `e*K + c0 + t*(c1 + t*(c2 + t*c3))`. All terms share the scale 2⁴⁰. For each mantissa, the generator intersects the rounding intervals from every applicable exponent after subtracting the exponent contribution. HiGHS fits a polynomial inside those intervals. The zero-mantissa row fixes the first constant at zero. The objective maximizes the smallest distance to an endpoint over the remaining mantissas. The generator quantizes each coefficient to a signed integer. Lean verifies feasibility of these coefficients and correct rounding. The performance and optimality of the LP solver are outside the proof. [Generator](../bench/rlibm/synthesize.py) · [Coefficients](../Egglog/Examples/Rlibm/Coefficients.lean) · [Core and signed range proof](../Egglog/Examples/Rlibm/Polynomial.lean)
2. **Check the rounded result against the mathematical logarithm.** For a positive input `x=N/D`, deciding whether `log_b(x)` exceeds the dyadic rounding midpoint `p/q` reduces to comparing integers. Compare `N^q × b^max(-p,0)` with `D^q × b^max(p,0)`. The two tests enclose the exact log between the midpoints to the adjacent bfloat16 outputs. Equality belongs to the even significand. Lean's kernel checks the entire domain in pages of 128 encodings, independent of HiGHS and the decimal logarithms used for synthesis. The optional Mathlib theorem connects these comparisons to `Real.logb` and proves the result for one exactly. [Integer specification](../Egglog/Examples/Rlibm/Spec.lean) · [Core certificate](../Egglog/Examples/Rlibm/Certificate.lean) · [Mathlib theorem](mathlib/RealLog.lean)
3. **Change the arithmetic schedule.** One set of proved `egg_rule` rewrites moves the exponent compensation into the Horner chain, groups independent terms with Estrin's scheme, or splits the compensation coefficient into a power of two and a residual. Each extraction profile selects a different AST. The proofs cover every assignment to the seven operand registers. The lowering theorem relates the extracted arithmetic to unsigned 64-bit operations, and a bound justifies decoding the final accumulator as signed before rounding. [Rules, profiles and lowering](../Egglog/Examples/Rlibm/Optimizer.lean) · [Whole-routine theorems](../Egglog/Examples/Rlibm/VerifiedSchedules.lean)
4. **Change the approximation itself.** The second reduction uses 16 quadratic segments with `t = m % 8`. This doubles the segment count and removes a multiplication. On input `0x3f82` (1.015625), the cubic and quadratic log2 accumulators are 24,594,165,088 and 24,589,806,250 in Q40 units. Both round to `0x3cb7`. A rule writes the alternative into a keyed `Candidates` function with a proof that it returns the baseline’s rounded word. The `resegment2` and `resegment10` rewrites join the original call with this candidate and use its proof at the rounding boundary. The checker verifies the two actual results and the LP inequalities for each positive finite input. The cost model prefers the quadratic, and the C emitter lowers the AST that equality saturation selects. A counterexample shows the limit of this reasoning. Rounding down to the Q16 grid before the final rounding of `log2(0x3f77)` changes the result from `0xbd53` to `0xbd54`. The LP cell check rejects it. [Whole-routine rewrites](../Egglog/Examples/Rlibm/Cells.lean) · [Rounding-cell certificate](../Egglog/Examples/Rlibm/CellsCertificate.lean) · [LP and extraction theorems](../Egglog/Examples/Rlibm/Results.lean)
5. **Check the reference implementation.** A separate Lean model uses the pinned Rutgers binary64 constants, range reduction, operation order and result conversion. It checks the normalization contract and the rounded result on the full input domain. The benchmark compiles the original C++ translation units with contraction disabled and independently compares their bits with MPFR. The source transcription and C compiler are outside the Lean kernel proof. [Upstream arithmetic model](../Egglog/Examples/Rlibm/Upstream.lean) · [Upstream certificate](../Egglog/Examples/Rlibm/UpstreamCertificate.lean) · [Measurement protocol](../bench/rlibm/Benchmarks.md)

Both signed zeros return negative infinity. Positive infinity returns positive infinity. Every negative nonzero input and every NaN returns the upstream NaN word `0xffff`. The logarithm of one returns positive zero. This example covers bfloat16 result bits under nearest-even rounding. IEEE exception flags, other target formats and directed rounding modes are outside its contract.

## Reproduce the proofs and export

From the repository root, the first command checks the library, all schedules, and the upstream model with the project's pinned Lean toolchain. Its initial build runs exhaustive kernel proofs and can take several minutes. Running the emitters only needs the compiled Lean modules.

```sh
(
cd lean
lake build Egglog.Examples.Rlibm.Results Egglog.Examples.Rlibm.Emit
lake env lean --stdin <<'LEAN'
import Egglog.Examples.Rlibm.Results
#eval Rlibm.core .ten 0x4120  -- log10(10), result 0x3f80 = 1
#eval Rlibm.Cells.selected .two
#print axioms Rlibm.Cells.optimized_correct
#print axioms Rlibm.Upstream.correctly_rounded
LEAN
mkdir -p /tmp/rlibm-generated
lake env lean --run RlibmTables.lean > /tmp/rlibm-generated/rlibm_coefficients.h
lake env lean --run RlibmEmit.lean /tmp/rlibm-generated
)
```

Regenerate the candidate table with Python and HiGHS. The checked-in coefficients were produced with HiGHS 1.15.1. A different solver version may return another feasible optimum. Rebuilding the proof checks the supplied coefficients.

```sh
python3 lean/bench/rlibm/synthesize.py --output /tmp/rlibm-coefficients.lean
diff -u lean/Egglog/Examples/Rlibm/Coefficients.lean /tmp/rlibm-coefficients.lean
```

To replay the real-number theorem, provide a built Mathlib checkout or a Lake project with Mathlib in its package cache. The script recompiles the same core, coefficient table and complete certificate in that toolchain before checking the real logarithm theorem. This replay was tested with Mathlib `v4.27.0`. [Replay script](check-real.py) · [Mathlib commit](https://github.com/leanprover-community/mathlib4/tree/a3a10db0e9d66acbebf76c5e6a135066525ac900)

```sh
python3 lean/rlibm/check-real.py /path/to/built/mathlib-project
```
