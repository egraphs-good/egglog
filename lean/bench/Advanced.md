# Luminal and eggcc in Lean

These case studies use two distinct interfaces. The raw interpreter runs upstream egglog rules and checks their output against Rust. The proof-returning compiler uses rules proved against a Lean semantics and produces an equivalent AST. For Luminal the input to both interfaces is the same archived Llama graph. For eggcc the large relational analysis uses the PolyBench 2mm graph and the verified compiler takes normalized upstream programs with scalar expressions, conditional regions, nested post-tested loops and print effects.

## Measured results

Median single-threaded schedule times on identical egglog programs are in milliseconds. The target is Lean/Rust ≤ 1.30 for each workload.

| Raw schedule | Lean (ms) | Rust egglog (ms) | Lean/Rust |
|---|---:|---:|---:|
| Llama, complete original schedule | 62.313 | 19.418 | 3.21× |
| Llama, selected lowering and fusion rules | 5.308 | 3.909 | 1.36× |
| Llama, 225 exported projection roots | 0.085 | 0.135 | 0.63× |
| eggcc 2mm, region and context analyses | 2.766 | 3.092 | 0.89× |

The matched projection pass also has native compiler baselines. The Rust engines receive the five projection groups already isolated. Lean locates those groups in the full 2,413-binding SSA input.

| Native 225-projection pass | Optimizer (ms) | Whole process (ms) |
|---|---:|---:|
| Lean, certified projection fusion | 0.990 | 48.063 |
| Rust egglog, native projection fusion | 1.563 | 13.603 |
| Rust egg, plain | 0.365 | 210.316 |
| Rust egg, explanation tracking | 0.478 | 210.796 |
| Rust egg, all explanations materialized and checked | 21.436 | 232.040 |

Certified Lean took 0.63× as long as native Rust egglog and 2.07× as long as explanation-tracking egg. The egg run that materialized and checked all 225 explanations and their 450 rewrite steps took 21.7× as long as the certified Lean pass. On eggcc, the verified compiler took 0.073 ms for `select_simple`, 0.076 ms for `if_dead_code`, and 0.835 ms for `if_dead_code_nested`. System-wide idle readings ranged from 74.6% to 98.8% across the final trials. [Raw paired samples](results/advanced-graphs.csv) · [Native compiler samples and stage times](results/advanced-certified.csv)

## Luminal

The repository's 2,413-binding graph contains 32 Llama decoder layers. The raw interpreter runs its original declarations, all rules, and the complete schedule including cleanup. All 70 table sizes agree with the repository snapshot. A separate SSA walk identifies seven learned projections per layer and the language-model head. Post-run queries verify that every corresponding TileMatmul has its original activation and weight in the correct operand positions. Checks also cover the output dtype, RMSNorm and gated activation. The smaller lowering schedule retains the complete input and the upstream rules for dtype propagation and selected kernel fusions. It produces 65 RowRMSNorm and 32 RowSwishMul nodes. [Original workload](../../tests/luminal-llama.egg) · [Snapshot and validation harness](luminal.py) · [Luminal compiler design](https://blog.luminal.com/p/compiling-models-to-megakernels)

The verified pass implements an ordered matrix contraction. It matches a broadcast multiply followed by a sum over the contiguous inner dimension. Five e-graphs, one per distinct layout, hold the actual SSA register definitions. The rule matches sums through those aliases, and cost-based extraction chooses a direct contraction. The pass fuses **225 projections**, leaving **2,155 live bindings**. It retains any temporary with another live consumer. The theorem quantifies over the input buffers and over the meanings of surrounding pure operators. Each FP32 product rounds before the next addition and the reduction visits terms in precisely the original order. The proof needs no associative law for floating-point addition. [Rewrite and whole-graph theorems](../Egglog/Examples/Luminal.lean) · [SSA importer and emitter](../Egglog/Examples/LuminalInput.lean)

`compile_correct` equates the satisfying assignments of the original and transformed SSA graphs. `slice_correct` proves that every retained output of a dependency-closed slice agrees with the input program. The emitter checks the closure condition, reparses its output, and checks the instructions, operands, original names and output roots. Tests also mutate the stride and swap the weight and activation. The theorem starts at the typed Lean IR. The file adapter, the upstream tiled GPU kernel, and GPU code generation are outside its scope. All timings measure compilation on a CPU. [Tests](../LuminalTests.lean)

## eggcc

On the actual double-matrix-multiply Tree IR, Lean runs the upstream rules for body containment and context propagation. The harness lifts the original unconditional initialization to top-level bindings so it can query each result. Its independent DAG traversal tracks region boundaries at branches and loops. After saturation it checks **all 3,248 body-membership pairs and 2,463 context pairs**, the empty list-body relation, and the source's 10 functions, 26 conditionals, and 29 loops. Matching cardinalities rule out additional tuples. [2mm input](../../tests/eggcc-2mm.egg) · [Rule selector and independent traversal](eggcc_regions.py)

The compiler imports Tree IR from eggcc commit `8f9b0f735e26334160b2403659834a9e2c585c1d`. The expression pass converts pure conditional regions to selects. Its semantics includes wrapping 64-bit integers, signed comparisons, Booleans, tuples, branch-local arguments and a print-state token. The extracted arms are substituted into their enclosing region without capturing local arguments. The largest acyclic fixture replaces four branches with six selects and preserves three ordered prints. [Expression pass and theorem](../Egglog/Examples/EggCC.lean) · [Upstream select rule](https://github.com/egraphs-good/eggcc/blob/8f9b0f735e26334160b2403659834a9e2c585c1d/dag_in_context/src/optimizations/select.egg)

### Executing and optimizing loops

An upstream `DoWhile(initial, body)` executes its body at least once. The first field of the body's result decides whether to continue. The remaining fields become the next loop arguments or the final output. The importer binds each shared loop and print once in its own region. The Bril backend emits real backedges and preserves simultaneous updates, including swaps of loop-carried registers. A branch with a print or nested loop executes only its chosen arm. [Structured importer](../Egglog/Examples/EggCCFlowInput.lean) · [Bril emitter](../Egglog/Examples/EggCCFlowBril.lean)

The compiler runs equality saturation on the full structured program. A proved rule rewrites `loop (withFalseExit body)` to `body`, retaining everything the final iteration computes or prints. Other rules expose that false continuation through sequencing, bindings and branches. In the upstream `eliminate_loop.bril` fixture the backedge disappears and the program still prints `1`. In `select.rs` a select appears inside a loop that still executes. In `flatten_loop.bril` the output retains both nested backedges. With arguments 3 and 2 it prints 0 through 5 from the inner loop, then 3 after the outer loop. [Control rewrites and schedule](../Egglog/Examples/EggCCFlow.lean) · [Kernel tests](../EggCCFlowTests.lean)

`EggCC.Flow.optimize_correct` proves that, for every initial state, the source terminates with a given result exactly when the optimized program does. The result includes its ordered print trace. `optimize_observe` also proves equality of completed, stuck or paused observations at every finite backedge budget, including the carried state and its print history. The command-line interpreter has a configurable budget per loop invocation. The unbounded theorem and emitted Bril have no such limit. These proofs start and end at the structured Lean IR. The upstream importer and Bril emitter are checked by differential execution. [Loop semantics](../Egglog/Examples/EggCCFlowSemantics.lean) · [Optimizer theorems](../Egglog/Examples/EggCCFlow.lean) · [Backend regression tests](../EggCCFlowBrilTests.lean)

The focused suite runs **41 four-way trace comparisons**. It checks the original program in upstream eggcc, the imported and optimized Lean IRs, and the emitted Bril in upstream eggcc. The 26 loop inputs cover first-iteration exit, a 64-iteration input, loop-carried Fibonacci values, conditional prints and nested loops. The harness also checks emitted backedges and their nesting. A wider scan of the 70 top-level Rust and Bril files in upstream's `tests/passing/small` finds **28 containing DoWhile**. On their recorded arguments **23 of those 28** compile and agree across all four executions. The other five use integer division (two programs), multiple functions and calls, pointers and memory operations, or floating-point operations. [Focused traces](results/eggcc-loop-traces.csv) · [Upstream loop sweep](results/eggcc-small-loop-traces.csv) · [Differential harness](eggcc_trace.py)

Running the entire unmodified 2mm optimizer still stops at the custom merge for `LoopNumItersGuess`. The fixture also requires merges that construct new e-graph terms, sets, unextractable constructors, relation deletion and further rule forms. The 2mm timing covers the complete input and the three named analyses.

Lean reports only `propext` and `Quot.sound` as axioms of `Luminal.fuse_sound`, `Luminal.compile_correct`, `Luminal.slice_correct`, `EggCC.optimize_correct`, `EggCC.Flow.optimize_correct`, and `EggCC.Flow.optimize_observe`.

## Where the engine spent time

The initial 2mm run took a median 219.72 ms in Lean. The analysis reads a fixed constructor graph throughout saturation, so the runtime now indexes its constructor rows once and propagates newly inserted relation tuples through the joins. The optimized path still goes through the ordinary ruleset and schedule entry points. [Initial 2mm trial ledger](results/advanced-2mm-before.csv) · [Relational runtime](../Egglog/Relations.lean)

The full Llama workload also mutates constructor classes and visibility. Attaching a fresh right-hand side directly to an existing class avoids artificial class merges that had invalidated the join indices. Sampling the remaining work on the full graph put 37.5% of samples under multi-action rule heads and 34.2% under generic joins. The matcher now binds non-join columns together from one surviving row, evaluates a rule-head term in a single traversal, and reuses indices for sources that have neither gained rows nor changed representatives or visibility. A fixed-point stage can be reused only at the same database epoch, so newly added or hidden rows trigger another search. [Join implementation](../Egglog/GenericJoin.lean) · [Schedule regression](../tests/scheduled-unions.egg)

## Measurement protocol

All measurements use one worker, fresh processes, one warmup, and five measured trials per case and engine on an Apple M4 Max with 128 GiB RAM. The harness samples system-wide CPU idle time before starting a trial and stops below 60%. The workers have a 60-second wall budget and an 8 GiB resident-memory limit. Lean is 4.33.1 and the plain Rust egglog baseline is the release build of this checkout at `d89bc58adc9b9a87a8f571c5b9ad75d686d226e6`.

For raw egglog the exact same setup, schedule and validation commands go to both engines. The parent starts the clock after setup, releases the worker, and stops the clock at the end-of-schedule signal. It checks results afterward. For the exported projection workload, the schedule is one rewrite round over five layout sorts. A separate run of the full export compares all 225 extracted terms. The native certified timers include graph seeding, solving and extraction from imported IR. The Luminal timer also includes visiting the full graph and installing the selected instructions. IR parsing, dead-binding slicing and writing the output file are outside that timer. For eggcc the timer includes argument substitution, and excludes Bril emission. Whole-process times are available for every trial. [Graph harness](large_graph.py) · [Compiler harness](advanced.py)

The native Rust egglog comparator starts with the 225 original SSA product definitions and the 225 expected projection roots in typed layout groups. It seeds them in one native write transaction, runs one pre-parsed round, and uses a single cost solution and term-reconstruction cache to decode all the roots. It checks each extracted cost and the two register operands. Graph creation, schema installation and command parsing are timed separately as `setup_ns`. [Native Rust egglog comparator](../../examples/lean_comparison/luminal.rs)

The Rust `egg` 0.10.0 comparator reads the projection workload exported by the certified pass and uses identical register aliases, five layout groups, extraction costs, and 225 requested roots. All modes check the extracted activation and weight for each root. Tracking enables explanations for every union. Explanation checking additionally materializes a flattened proof for every root, checks its steps against the fusion rule and the named SSA input equations, and rejects unknown justifications. The optimizer timer includes seeding, rebuilding, one rewrite round, extraction and result checks. Its input is the projection slice rather than the full Llama SSA. The library's one-time clock calibration is outside the optimizer timer and included in whole-process wall time. [Rust comparator](projection-egg/src/main.rs) · [Pinned dependencies](projection-egg/Cargo.lock)

## Reproduction

From the repository root, use a fresh destination for each sample ledger. The upstream eggcc checkout with a release binary is required only for the interpreter cross-check.

```sh
(cd lean && lake build lean-egglog luminal-certified eggcc-certified)
cargo build --release --bin egglog --example lean_comparison
cargo build --release --locked --manifest-path lean/bench/projection-egg/Cargo.toml
sh lean/test.sh

python3 lean/bench/advanced.py graphs /tmp/compiler-graphs.csv
python3 lean/bench/advanced.py certified /tmp/compiler-passes.csv
python3 lean/bench/eggcc_trace.py /path/to/eggcc /tmp/compiler-traces.csv
python3 lean/bench/eggcc_trace.py /path/to/eggcc /tmp/small-loop-traces.csv --all-small-loops

lean/.lake/build/bin/luminal-certified --emit tests/luminal-llama.egg /tmp/lowered.egg
target/release/egglog -j 1 /tmp/lowered.egg
lean/.lake/build/bin/luminal-certified --emit-projections tests/luminal-llama.egg /tmp/projections.egg
lean/.lake/build/bin/lean-egglog --raw /tmp/projections.egg > /tmp/projections.lean.out
target/release/egglog -j 1 /tmp/projections.egg > /tmp/projections.rust.out
cmp /tmp/projections.lean.out /tmp/projections.rust.out

lean/.lake/build/bin/eggcc-certified --emit-bril lean/bench/eggcc/if_dead_code_nested.json
lean/.lake/build/bin/eggcc-certified lean/bench/eggcc/flatten_loop.json 3 2
lean/.lake/build/bin/eggcc-certified --emit-bril lean/bench/eggcc/flatten_loop.json
```

The SHA-256 hashes of the Llama and 2mm inputs in this checkout are `4373641f7823e23506a0a9cf15c27398ef356ba995304724faa36d26db9cdacf` and `ce3913074600ae3037c1af4acdc4dd9f749a2c212c9fa4d650371b2f93c50ab3`, respectively. On the measurement Mac the build commands cleared the shell’s `LIBRARY_PATH`. The Rust example build also selected the installed Xcode macOS 26.2 SDK with `SDKROOT`.
