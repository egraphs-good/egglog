# Single-threaded egglog in Lean

The Lean 4 compiler framework runs equality saturation over an existing AST and returns an optimized program with a proof that it preserves the language semantics. A separate native interpreter executes a subset of the Rust egglog language, including positive Datalog rules and partial functions. Both engines run in Lean. The Rust crate in this checkout supplies the single-threaded baseline.

The [ParaBit case study](parabit/README.md) runs the CAV multi-width bitvector corpus through a relational egglog theory and a proof-returning compiler pass. It includes a single-threaded comparison against the authors’ explanation-enabled `egg` solver.

The [Luminal and eggcc case studies](bench/Advanced.md) run the archived Llama compiler schedule and PolyBench 2mm region analyses, and provide proof-returning passes for ordered matrix projection fusion and eggcc control flow, including select conversion and post-tested loops.

## Build and run

Run these commands at the repository root with Elan and Cargo installed.

```sh
(cd lean && lake build Egglog.Examples.BoundsCheck lean-egglog)
lean/.lake/build/bin/lean-egglog --naturals tests/web-demo/naturals.egg
lean/.lake/build/bin/lean-egglog --raw tests/web-demo/path.egg
lean/.lake/build/bin/lean-egglog --raw tests/math-microbenchmark.egg
lean/test.sh
```

The build pins Lean 4.33.1 and uses its standard library. On the benchmark Mac, the shell's `LIBRARY_PATH` selected an incompatible Command Line Tools SDK. Prefixing the build commands with `env -u LIBRARY_PATH` selected the installed Xcode SDK. The default CLI profile checks the arithmetic signature `Zero One Succ Add Mul`. The `--naturals` profile checks `Z S Plus Times`. A successful `check` has no printed output. Raw mode accepts user-supplied rules and unions.

## Verified compiler passes

Import `Egglog.DSL` to define and run a compiler pass in Lean. For a new IR, `egg_language` generates the AST, interpreter, constructors and proved codec from one declaration. For an existing IR, connect its interpreter directly. Use `egg_signature` to assign constructor denotations and extraction costs, `egg_codec` to encode and decode the IR, and prove that the encoding implements the IR's semantics. Each `egg_rule` carries a theorem for every assignment of its pattern variables. A conditional rule also receives proofs of its equality premises. `egg_schedule` composes rulesets, repetition, saturation, stop conditions and stateful match-selection policies. [Compiler tutorial](Compiler.md) · [Boolean IR](Egglog/Examples/Boolean.lean) · [Guarded strength reduction](Egglog/Examples/StrengthReduction.lean)

For an input `p`, `ir.optimize phases p` returns an AST `output` and a theorem `∀ env, ir.denote output env = ir.denote p env`. The certified graph justifies each union through a proved rewrite, its premises, or congruence. Extraction follows the selected enodes and the source decoder proves that it encodes back to the extracted term. The semantic proofs erase during native compilation. A total pass keeps its input if decoding fails. `ir.optimize?` exposes that failure to the caller. `egg_prove` can also discharge equivalence goals about two programs in Lean. The persistent session API supports adding multiple roots, running a shared schedule and reusing an extraction cost solution. [Verified engine](Egglog/Certified.lean) · [Session API](Egglog/Session.lean) · [Compiler tests](CompilerTests.lean)

The bounds-check compiler defines a relation, a keyed function, a custom merge and a three-pattern rule entirely in Lean. Each stored lower bound carries a proof about its key. The consumer joins that function entry with an upper-bound relation and a bounds-check e-node. Its only input is a source AST. The same schedule produces the analysis facts, merges writes, runs copy propagation, and makes the equivalence available for extraction. [Runnable compiler](Egglog/Examples/BoundsCheck.lean) · [Function declarations and multipatterns](Compiler.md#analysis-tables-and-their-proofs)

The compiler DSL and the raw `.egg` interpreter share first-order patterns, the generic multiway join and its semi-naive search. The compiler executes over its own proof-carrying graph and tables. The CLI executes existing egglog files over a separate database and also supports commands such as deletion, subsumption and push/pop. The certified DSL currently uses one semantic carrier for function keys and values and positive rule bodies. Performance tables below identify which execution path was measured. [Compiler runtime](Egglog/Certified.lean) · [Raw database](Egglog/Relations.lean) · [Related Rocq work](https://adam.chlipala.net/papers/EgraphsPOPL27/)

## Language coverage

Raw execution handles the following parts of the Rust language.

- Equality sorts, datatype and constructor declarations, constructor costs, integer and string literals, globals, explicit unions, and ground queries.
- Mutually recursive datatype declarations, binary64 and Boolean literals, constructor deletion and subsumption, sequential rule-local bindings, constructor rewrites and birewrites, repeated variables and wildcards, named rulesets, supported comparison guards, and schedules with bounded runs, repetition, saturation and combined rulesets.
- Set-valued relations with typed columns and positive joins. Rule heads can insert facts, construct terms, set function values, and union classes. Larger rules flatten embedded constructor patterns and intersect column projections, choosing the next variable from the current candidate counts. Semi-naive variants partition matches by their first new relation, function or constructor row. Function tables support explicit merge policies including integer minimum and maximum. A changed function result triggers a rescan of that function. E-class merges rebuild keys and invalidate the saved row cutoffs.
- Extraction of a minimum additive-cost tree, push/pop scopes in the raw runner, and size queries. Extraction uses tree height and then enode ID to break equal-cost ties.

Coverage is partial. The Rust container and numeric libraries, custom primitives, arbitrary merge expressions, the Rust scheduler plugin interface, proof encoding, I/O commands, and the complete query planner remain outside this port. The main repository's nightly performance suite uses several of these features. The test driver lists original programs exercised unchanged. The harness also checks matcher equivalence round by round, including fallback joins, newly enabled matches, noncanonical handles and malformed certificates. [Tests](test.sh) · [Compiled matcher tests](CompiledTests.lean)

## Reproduce the performance comparison

The compiler comparison starts with a native AST in each language, seeds the graph, runs matching schedules, extracts and decodes the result back to a native AST. Rust inserts the input through this checkout's `Write` API. Both engines verify the full expected tree after the timer stops. This includes the cost of obtaining the certified compiler result in Lean. The raw comparison separately times an already-seeded graph running a parsed schedule.

```sh
cargo build --release --locked --example lean_comparison
(cd lean && lake build compiler-bench)
python3 lean/bench/compiler.py --size 128 --runs 7 --warmup 2 --enforce
python3 lean/bench/compiler.py --size 1024 --runs 7 --warmup 2 --enforce
```

```sh
cargo build --release --locked --example lean_comparison
(cd lean && lake build Egglog.Examples.BoundsCheck lean-egglog)
python3 lean/bench/compare.py --size 2000 --runs 7 --warmup 2 --enforce
python3 lean/bench/compare.py --math-rounds 11 --runs 7 --warmup 2 --enforce
python3 lean/bench/compare.py --paths 128 --runs 7 --warmup 2 --enforce
python3 lean/bench/compare.py --size 300 --runs 5 --warmup 1 --proofs
```

The schedule-only harness runs the same setup, bounded schedule and post-run checks on both engines. Each sample starts a fresh process. The schedule timer brackets execution of the parsed run command and excludes graph seeding and the final checks. The wall timer covers the whole process. For the generated arithmetic workloads, Lean validates the source against its proof registry during setup. Rust runs this checkout with one thread, default seminaive execution and its release allocator. The optional Rust proof run enables the crate's own proof generation. It measures recording overhead with the same queries. Kernel checking of the optional Lean snapshot and reconstruction of Rust query proofs are separate operations.

Before each workload the harness samples host CPU idle time. It interleaves engines in a deterministic shuffled order, checks the expected equalities and every constructor table size on every sample, and reports the median and interquartile range. The threshold is **Lean schedule time ≤ 1.30 × plain Rust schedule time on each workload**. The math case reuses the complete setup and schedule from `tests/math-microbenchmark.egg` and checks all 13 table sizes. Its calculus rules run under raw semantics. The path case takes the rules from the web demo and computes all 8,256 reachable pairs on a chain of 128 edges.

## Measured results

The [larger single-threaded benchmarks](bench/Large.md) cover bigger compiler ASTs, symbolic-width proofs, recursive closure, cyclic joins, calculus saturation and JSON parsing.

Measured November 18, 2026 on arm64 macOS 26.6.2 with Lean 4.33.1, rustc 1.91.0 and Rust revision `d89bc58a`. Each primary result is the median of seven interleaved, single-threaded samples after two warmups. The threshold was 1.30× plain Rust on each workload. Host CPU checks before the final primary workloads showed 93.5–98.6% idle. An earlier check at 55.6% idle skipped its run.

### Certified compiler, native input AST through native output AST

| Compiler pass | Input leaves | Rust ms | Certified Lean ms | Lean/Rust | IQR ms, Rust / Lean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boolean simplification | 128 | 0.488 | 0.438 | 0.90× | 0.028 / 0.014 |
| Guarded strength reduction | 128 | 0.329 | 0.281 | 0.85× | 0.021 / 0.008 |
| Boolean simplification | 1,024 | 2.637 | 3.318 | 1.26× | 0.019 / 0.023 |
| Guarded strength reduction | 1,024 | 1.527 | 1.913 | 1.25× | 0.044 / 0.030 |

The Boolean pass removes redundant operations and selects a branch after the two branch expressions become equivalent. Strength reduction finds the exponent through a second pattern in the multiplier's class and extracts a shift using declared operator costs. Every sample independently checked the extracted AST against the expected program. The Lean result carries the pass's semantic preservation theorem for every input valuation.

### Egglog schedule execution

| Workload | Rust ms | Lean ms | Lean/Rust | IQR ms, Rust / Lean |
| --- | ---: | ---: | ---: | ---: |
| Identities and congruence, 2,000 seeds | 2.394 | 2.536 | 1.060× | 0.077 / 0.042 |
| Commutativity, 2,000 seeds | 0.523 | 0.522 | 0.999× | 0.010 / 0.017 |
| Distribution and reassociation, 2,000 seeds | 4.946 | 3.968 | 0.802× | 0.037 / 0.040 |
| Repeated-variable matching, 2,000 seeds | 0.191 | 0.202 | 1.054× | 0.007 / 0.010 |
| Full 11-round math | 188.229 | 198.954 | 1.057× | 0.980 / 1.337 |
| 128-edge transitive closure | 1.770 | 0.749 | 0.423× | 0.283 / 0.102 |

The math run produced 1,047,896 canonical constructor rows on each engine, including 641,743 `Add` and 345,075 `Mul` rows. All 13 constructor counts and both post-run equalities agreed. Changes that mattered on this workload included packed binary rows, direct lookup of child columns, scanning only new joins across e-class merges, and compiling patterns such as `Add (Mul a b) (Mul a c)`. The path test checked the first-to-last path and all 8,256 reachable pairs.

A separate 300-seed distribution test measured the cost of recording Rust justifications. Median schedule times over five runs were 0.881 ms for plain Rust, 21.971 ms for Rust with proof generation and 0.679 ms for the Lean CLI with its rewrite registry. This comparison times saturation only. The table above for the Boolean and strength-reduction passes measures the certified source-language API end to end.
