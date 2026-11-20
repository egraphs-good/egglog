# ParaBit in Lean and egglog

This case study proves multi-width bitvector compiler rewrites for arbitrary integer inputs and symbolic widths. It consumes the original 165 BWLang problems from the CAV artifact. Rinaldi, Wickerson and Coward report 144 equalities found and 142 proofs verified in Isabelle. [Published paper](https://doi.org/10.1007/978-3-032-32519-8_20) · [Artifact](https://doi.org/10.5281/zenodo.19825890)

## Use the solver

From the repository root, build the Lean programs and the authors’ pinned Rust release. Set `ART` to a new directory or an existing checkout of the pinned artifact.

```sh
ART=/tmp/parabit-cav
sh lean/parabit/build-baseline.sh "$ART"
(cd lean && lake build parabit-certified parabit-relational)
CASE="$ART/benchmarks/ROVER/bwlang/add_assoc_4.bwlang"
lean/.lake/build/bin/parabit-certified "$CASE" 6
lean/.lake/build/bin/parabit-relational lean/parabit/theory.egg "$CASE" 12
```

A compiler using the library can obtain a result about the **original named BWLang AST**. `p.Holds ρ` means the assignment satisfies the file's explicit conditions and the artifact's implicit positivity constraints on width expressions. Widths and inputs range over unbounded integers. `bw w x` computes the low `w.toNat` bits of `x`. The Lean reader accepts a wider expression grammar than the artifact validator. All measured inputs come from the original validated corpus. The meanings of signed interpretation, bitwise operations, shifts, floor division and select are defined in Lean. Encoding and interning variables preserve this source denotation. [Source API](../Egglog/Examples/ParaBitInput.lean) · [ModIR semantics](../Egglog/Examples/ParaBitSemantics.lean) · [Original width assumptions](https://github.com/luigirinaldi/parabit/blob/221c2f05b2260377c3cd70b31b9ae6d0158fffab/src/utils.rs#L65-L79)

```lean
import Egglog.Examples.ParaBitInput

def checkRewrite (p : BWLang.Problem) :
    Option (PLift (∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ)) :=
  p.prove? 6
```

For word-level queries with selectors or signed interpretation, `p.proveStaged? rounds` first propagates word ranges and simplifies paired selectors and truncations. If the root classes still differ, it continues with the ordinary algebraic rules on the same graph. The round bound applies to each phase. The return type is the same source-level equivalence. Use `p.certifyRule? (staged := true)` to install a result in another compiler. The native solver and standalone theorem exporter both accept `--staged`. [Schedule](../Egglog/Examples/ParaBit.lean) · [Scaling results](../bench/Large.md)

To use a solved problem as a compiler optimization, import `Egglog.Examples.ParaBitPass`. The call `p.certifyRule?` returns an ordinary `GuardedRule` over the BWLang scalar operations. The source symbols become pattern variables and the explicit and inferred premises become equality guards. A register-IR example installs Hydra `opt_2750`, rewrites an expression inside a larger addition, and extracts a smaller AST with a proof for every register valuation. Reversing the widths disables the rewrite. [Compiler adapter](../Egglog/Examples/ParaBitPass.lean) · [Executable example](../ParaBitPassTests.lean)

Export an importable Lean theorem or a standalone egglog program from the same input. The theorem uses kernel reduction. The native solver erases semantic proofs during execution. A failing query produces no theorem.

```sh
lean/.lake/build/bin/parabit-certified --emit-lean "$CASE" 6 > /tmp/Rover.lean
(cd lean && lake env lean /tmp/Rover.lean)
lean/.lake/build/bin/parabit-relational --emit-egg lean/parabit/theory.egg "$CASE" 12 > /tmp/rover.egg
cargo run --release --bin egglog -- -j 1 /tmp/rover.egg
```

For standalone theorems, the exporter records ground instances of the proved rules, then tries deleting irrelevant instances, seed terms and premises. The kernel checks the surviving rule applications, including their premises and congruences. Boolean certificates can instead evaluate both bit circuits with the same proved transfer function as the native e-class analysis. These hints reduce the work of checking a finished proof. They are generated only when exporting a theorem. The exported ROVER associativity example can be checked directly with `(cd lean && lake env lean parabit/RoverAssociativity.lean)`. [Example theorem](RoverAssociativity.lean) · [Proof hints](../Egglog/Examples/ParaBit.lean) · [Generic ground replay](../Egglog/Certified.lean) · [Analysis replay](../Egglog/AnalysisEval.lean)

## How the analyses work

The source reader inserts both terms, the supplied hypotheses and stable identifiers for Boolean atoms. Width-order propagation, constant folding, signed ranges and unsigned precision run as table joins. For example, `fits x p` means `0 ≤ x < 2^p` at an admissible width. If both summands fit widths strictly below `w`, their sum fits `w` too. A second rule uses that fact to merge `(Bw w sum)` with `sum`. This is sufficient to remove both intermediate truncations in ROVER’s fourth associativity case. [Relational theory](theory.egg) · [Test of this derivation](../ParaBitTests.lean)

```lisp
(rule ((= e (Add a b)) (fits a p) (fits b q) (lt p w) (lt q w))
      ((fits e w)))
(rule ((= e (Bw p x)) (fits x p))
      ((union e x)))
```

Word-wise Boolean expressions use shared bit variables for operands and symbolic masks for truncations. A 64-bit truth table covers every assignment to six basis bits at one arbitrary position. Wider supports can use reduced ordered decision diagrams. Both analyses preserve arbitrary program widths. Demand-driven relations compute diagram nodes and share subproblems. Calculation of a width, range or gate value does not depend on the input's filename. [Theory and schedules](../ParaBitMain.lean) · [Negative controls](analyses-tests.egg)

In the certified analysis a width premise such as `p ≤ q` restricts the feasible truth-table rows. At every position where the low-`p` mask has a one, the low-`q` mask must also have a one. This discharges identities involving masks of different symbolic widths, while an input without that premise can lead to a different result. [Implementation](../Egglog/Examples/ParaBitDecision.lean) · [Regression tests with and without the premise](../ParaBitTests.lean)

The proof-returning solver expresses arithmetic and range transfer as proved rewrites and guarded equality rules. Its `analyze` stage accepts an e-class value transfer together with a proof that it preserves the constructor's denotation. Equal values then justify a class merge. Bit extensionality lifts equality at an arbitrary bit to integer equality. The raw relational theory and the certified implementation have independent schedules. Use `Problem.prove?` when a compiler needs evidence of equivalence. [Verified rules](../Egglog/Examples/ParaBit.lean) · [Bitwise analysis](../Egglog/Examples/ParaBitDecision.lean) · [Generic analysis interface](../Egglog/Certified.lean)

## Single-threaded comparison

The [larger benchmark suite](../bench/Large.md) scales individual symbolic-width goals to thousands of AST nodes and includes the compiler and relational workloads.

```sh
python3 lean/parabit/compare.py run "$ART" sweep.csv --seconds 2 --trials 3
python3 lean/parabit/compare.py summary sweep.csv
python3 lean/parabit/compare.py run "$ART" extended.csv --mode egg-coverage --seconds 60 --unfinished sweep.csv
python3 lean/parabit/check-kernel.py corpus "$ART" kernel.csv --rounds 6
```

The baseline links the authors' unmodified ParaBit 0.1.6 release and its pinned egg 0.10.0 dependency. Both Rust modes use `with_explanations_enabled()`. Tracking stops after finding the equivalence. Materialization also calls `make_proof()` and `explanation_string()`, which flatten, reduce, check and format the explanation. Independent Isabelle checking is outside the timed operation. [Timing driver](timing.rs) · [Upstream runner](https://github.com/luigirinaldi/parabit/blob/221c2f05b2260377c3cd70b31b9ae6d0158fffab/src/lib.rs#L189-L219)

Each sample uses fresh solver state and one worker. The shared solver timer begins after source parsing. On Lean it includes graph seeding, analysis, search and the final equality query. In Rust it includes ParaBit's rule construction, its encoding of explicit width facts and construction of the runner report. The checked Lean query encodes the artifact's implicit width-positivity premises directly. Source parsing, conversion to each solver's IR and loading the raw relational theory are outside the solver timers. The separate `runner_ms` column measures egg's iteration loop alone. The harness initializes egg’s `quanta` clock before the solver timer. Its first-use calibration reached about 200 ms on this host. Whole-process latency, including initialization, is also in the CSV. Solver timings include differences in rules and search strategy.

The harness randomizes input and engine order, samples host-wide CPU idle time before each group of 25 inputs, and stops below 60% idle. Each process has a two-second wall limit and a sampled 8 GiB resident-memory ceiling. The longer follow-up gives unresolved explanation cases 60 seconds at the same memory limit and records equivalence discovery before attempting to materialize the proof. The CAV artifact uses 60 seconds and an 8 GB virtual-memory limit on Linux for its search experiment. [Harness](compare.py) · [CAV experiment](https://github.com/luigirinaldi/CAV/blob/ea34dcb7573fbdd2049c80aa6deafe1daed2d981/scripts/run_full.sh)

## Results on the pinned corpus

On an Apple M4 Max (16 cores, 128 GB, macOS 26.6.2, Lean 4.33.1 and Rust 1.91.0), the certified solver and the relational theory each found all 163 valid equalities. The two remaining Hydra inputs have concrete counterexamples. Set `q = 2`, `r = 3`, `t = 4`, `newvar0 = 1`. Their premises hold. The left and right sides evaluate to 7 and 1 for `opt_310`, and to 15 and 1 for `opt_2703`. Lean checks both counterexamples against the source semantics. [Counterexample proofs](../ParaBitSourceTests.lean)

| Suite | Inputs | Lean certified | Lean relational | Authors’ egg found locally | Authors’ egg materialized locally |
|---|---:|---:|---:|---:|---:|
| Alive | 62 | 62 | 62 | 58 | 55 |
| Hydra | 70 | 68 | 68 | 56 | 56 |
| ROVER | 28 | 28 | 28 | 27 | 27 |
| Industry | 5 | 5 | 5 | 4 | 4 |
| Total | 165 | 163 | 163 | 145 | 142 |

The Rust columns combine all observed successes in the sweep and the 60-second follow-up. Four more explanations finished in that follow-up. Twenty processes reached the memory ceiling, three after reporting equivalence. Upstream also panicked on `Alive/AndOrXor_1388_values_0` while merging unequal constants. The paper reports 144 found and 142 externally checked by Isabelle on its hardware and limits. The last column above measures the artifact's own explanation reconstruction and checks. [Follow-up samples](results/extended.csv) · [Published evaluation](https://doi.org/10.1007/978-3-032-32519-8_20)

### Independent Lean kernel checking

Each successful native result also produced a standalone theorem about the named source AST that a fresh Lean process checked with `decide +kernel`. All 163 checked, including 111 checked from instantiated rules and 52 from Boolean analysis values. Median whole-process export took 22.6 ms, and median kernel checking took 759 ms. One Industry goal, `equation_abc`, needed the 15-second retry after exceeding the first 5-second budget. Proof export and independent checking are optional costs for a compiler using the proved pass. [Case-level proof ledger](results/kernel.csv) · [Export and check driver](check-kernel.py)

### Native solver times

The table uses the median of three runs per input, then compares only inputs both modes solved in all three trials at two seconds. Each median column summarizes the paired inputs. The ratio is the geometric mean of individual Lean/Rust time ratios, weighting each input equally. Smaller ratios favor Lean.

| Lean mode | Authors’ egg mode | Paired inputs | Lean median | Rust median | Ratio | Within 1.30× |
|---|---|---:|---:|---:|---:|---:|
| Certified | Track explanations | 143 | 0.256 ms | 1.194 ms | 0.106 | 139/143 |
| Certified | Materialize explanations | 138 | 0.274 ms | 2.709 ms | 0.048 | 138/138 |
| Relational | Track explanations | 143 | 0.629 ms | 1.194 ms | 0.322 | 134/143 |
| Relational | Materialize explanations | 138 | 0.628 ms | 2.709 ms | 0.142 | 138/138 |

The two Rust modes ran interleaved in the first three-trial sweep. Both final Lean solvers were measured in a second interleaved three-trial sweep on the same machine after optimizing relational joins and certified stop conditions. CPU idle samples ranged from 78.1% to 99.1%. The Rust iteration loop alone had a median of 0.640 ms on the 143 tracking pairs. Comparing the entire certified solver to this narrower Rust interval gives a paired ratio of 0.259. The Rust explanation timer includes dropping the consumed graph and proof string, while the tracking timer stops before dropping the graph. The difference between the two times includes those costs. [Rust trials](results/sweep.csv) · [Lean trials](results/lean.csv)

To recompute the table from the saved runs, use `python3 lean/parabit/compare.py summary lean/parabit/results/sweep.csv lean/parabit/results/lean.csv`.

### Same theory on the two egglog implementations

As a control for differences in the solving strategy, the harness emits one standalone `.egg` file per input. It feeds those exact bytes to the Lean interpreter and this checkout's plain Rust egglog CLI at `-j 1`, alternating which process runs first. Both CLIs found the same 163 equalities and failed the final equality check for the two counterexamples in all three trials. On the 163 positive inputs the geometric mean of the paired Lean/Rust wall-time ratios was 0.745. Median per-input times were 5.109 ms for Lean and 7.298 ms for Rust. Lean met 1.30× on all 163. The two expected failed checks also met the target (0.950× and 1.050×). The largest ratio was 1.273× on Alive `AndOrXor_1207` (12.686 ms against 9.965 ms). These are whole-process times including CLI startup, parsing and elaboration of the theory. [Same-theory harness](compare_raw.py) · [Paired samples](results/raw-cli.csv)

```sh
cargo build --release --locked --bin egglog
(cd lean && lake build lean-egglog parabit-relational)
python3 lean/parabit/compare_raw.py "$ART" raw-cli.csv
```
