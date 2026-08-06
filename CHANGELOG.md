# Changes

## [Unreleased] - ReleaseDate

- Fix a crash where bounded table scans with column constraints could yield stale (deleted) rows, leaking the stale value marker into primitives and panicking with an out-of-bounds intern-table read (e.g. `index out of bounds: the len is 0 but the index is 2147483647`).
- Fix a build failure when egglog is compiled without default features (as a library dependency). The `egglog-add-primitive` proc macro parses full Rust expressions and now declares `syn`'s `full` feature directly, instead of relying on another crate (`clap_derive`, via the `bin` feature) to unify it onto our `syn`. This surfaced when `clap_derive` moved to `syn` 3.x. A CI job now builds `-p egglog --no-default-features` to catch regressions.
- Convert many reachable `panic!`/`unwrap`/`expect`/`todo!` sites into recoverable errors, so malformed or edge-case programs report an error instead of aborting the process. Examples now returning `Error`/`TypeError`/`ParseError`/`ProveExistsError`: malformed sort-constructor declarations like `(sort S (Vec))` or `(sort S (UnstableFn))`; negative `extract` variant counts; duplicate rule names; `unstable-fn` referencing an unknown, non-literal, or mis-typed target; `(fail ...)` wrapping `include` or an empty expansion; subsuming a non-call rewrite; running `prove`/`prove-exists` without proofs enabled; and missing/unreadable files for `input`, `print-function`, `print-overall-statistics`, and the CLI. Several primitives became partial (returning no result instead of panicking) on out-of-range input: `vec-set`/`vec-remove` indices, `multiset-pick` on an empty multiset, count overflow in `multiset` operations, and the numeric primitives `bigint <<`/`>>`, `bigrat`, and `log2`. A few scheduler edge cases (unknown ruleset, rules with no free variables) no longer panic; variable-free rules now correctly apply their actions when scheduled. Primitive resolution now returns `TypeError::AmbiguousPrimitive`/`TypeError::UnresolvedPrimitive` instead of panicking when duplicate same-signature registrations are indistinguishable or nothing resolves; both direct calls and `unstable-fn` primitive targets report the same variants. `step_rules_with_scheduler` now restores its `rulesets`/`schedulers` on every fallible path, so an error during scheduled rule compilation no longer leaves the `EGraph` in a corrupted state.
- **Breaking:** a function's signature is stored once, shared between `TypeInfo` and `Function` instead of copied into both. `TypeInfo::get_func_type` returns `Option<Arc<FuncType>>`, `Function::schema()` becomes `Function::func_type() -> &FuncType`, and `ResolvedSchema` is removed (its `get_by_pos` moves to `FuncType`).
- **Breaking:** `EGraph::print_function` now takes its output sink as `Option<(File, PathBuf)>` plus a `Span`, so write failures return `Error::IoError` instead of panicking.
- Speed up query evaluation by building on-the-fly per-subset column indexes as sorted arrays (`SortedColumnIndex`) instead of hash maps. These indexes are typically iterated once and probed a bounded number of times over high-cardinality columns, so skipping hash-table construction is a large win (e.g. ~33% faster on the `gemma` benchmark).
- Share trie roots (and their cached sub-indexes and child nodes) across query plans within a single `run_rule_set` instead of rebuilding a fresh trie per plan. Plans that scan the same table under the same header (fast) constraints reuse one root, so on-the-fly per-subset index builds happen once rather than per plan; only roots that more than one plan uses are shared, so workloads that would not benefit keep the per-plan behavior. Large speedups on transformer workloads (e.g. ~15% faster on `whisper`, ~12% on `gemma`, ~8% on `qwen3_moe`).
- Add `make nightly` and `scripts/nightly_bench.py`, a hyperfine-based benchmark harness that measures every `tests/**/*.egg` program at 1/2/4/8 threads and (where supported) in proof-testing mode, caps each run at a 2-minute timeout, skips sub-50ms programs, and emits an HTML dashboard (one row per benchmark, one column per configuration) for nightly.cs.washington.edu. The dashboard uses [eval-live](https://github.com/oflatt/eval-live) for interactive filtering and sorting.
- Add typed `EGraph` extension state that clones with `EGraph` and is restored by `push`/`pop`.
- Fix custom scheduler queries so subsumed rows are not offered as fresh matches.
- Replace the global Rayon thread pool with an `egglog-concurrency` scoped `ThreadPool`; configure parallelism per `EGraph` via `with_num_threads` / `set_num_threads`.
- Report full source file paths in egglog span and error messages.
- Fix seminaive matching after nested containers rebuild in place by propagating dirty container ids through parent containers.
- Fix multi-column secondary index rebuilds so each value's rows come back sorted by row id, and make all rebuild paths (serial, parallel, and bulk) record a row once even when its value repeats across covered columns (#914).
- Render nullary AST calls without a trailing space, e.g. (foo) instead of (foo ).
- Escape `"` and `\` when displaying string literals so printed/serialized programs round-trip through the parser.
- Add a BigRat to-i64 primitive for integral rationals.
- Add f64 exp, log, and sqrt primitives.
- Add `RunReport::can_stop` so scheduler progress can be reported separately from database updates.
- Add `EGraph::typecheck_expr_with_bindings_and_output`, `Core::eval_resolved_expr`, and `Core::apply_primitive` for body-defined primitive support, including normal command-path global rewrites for expressions typechecked through the helper.
- Allow `unstable-fn` function containers to target primitive overloads.
- Desugar `relation`s to `constructor`s to simplify the language and implementation. Relations no longer return unit `()` values.
- Refactored API to use [`TermId`] more consistently instead of `Term` where possible, simplifying egglog code.
- **Typed primitive surface for seminaive safety (#772).** Custom primitives now pick one of `PurePrim` / `ReadPrim` / `WritePrim` / `FullPrim` based on what the body needs, and register via the matching `add_*_primitive`. Rust enforces capability bounds via the state wrapper passed to the body; the egglog typechecker enforces context bounds. See the `egglog::exec_state` module docs and the `*Prim` trait docs for the full picture. Migration: `rust_rule` callbacks now take `&mut WriteState` (replacing `RustRuleContext`); a new `rust_rule_full` gives action callbacks read access. Higher-order primitives over `unstable-fn` values dispatch via `state.apply_function(&fc, args)`.
- Expose `Read::table_size(name)` and `Read::table_sizes()` so read-capable primitives can inspect row counts without raw execution-state access, while avoiding an all-table scan when only one table is needed.
- **`:naive` and `:unsafe-seminaive` rule options** (mutually exclusive). Both compile a rule under the permissive `Read`/`Full` contexts so its RHS can read the database (read-primitives and function-table lookups). `:naive` matches the whole database every iteration; `:unsafe-seminaive` keeps seminaive (delta) matching, which is faster but **unsafe** — an RHS read observes the database mid-iteration, so results can depend on evaluation order. `:unsafe-seminaive` is rejected by the term/proof encoding.
- **Name-indexed e-graph access from primitives and `rust_rule` callbacks (#745, #751).** New `Read` / `Write` capability traits on the state wrappers let primitive bodies and rule callbacks read/write tables by name (`fs.lookup`, `fs.set`, `fs.add`, `fs.union`, `fs.function_entries`, `fs.constructor_enodes`, etc.) instead of through raw `FunctionId` + `&[Value]`; `EGraph::update(|fs| ...)` gives the same surface outside a rule, and `EGraph::function_entries` / `EGraph::constructor_enodes` expose the table scans directly at the top level. Misuse (wrong subtype, wrong arity, unknown table) surfaces as `Error::ApiError`. Also `Read::enodes_for_eclass` (a constructor's rows by output e-class, through the backend's column index rather than a scan), `Read::constructor_schema` / `Read::function_schema` / `Read::table_subtype` (a table's declared signature and subtype, which a primitive body cannot get from `TypeInfo`), and `Core::rebuild_container` (remap a container value's contents and intern the result, for container sorts whose Rust type the caller cannot name). Together these let an out-of-tree primitive walk and rebuild a sub-e-graph — see `unstable-subst` in `egglog-experimental`.
- **Container support in the term/proof encoding.** Programs using container sorts (`Vec`, `Set`, `Map`, `MultiSet`, `Pair`) now work under the term/proof encoding (previously rejected), including containers read (`vec-get`, `map-get`, …) or constructed (`vec-of`, `set-of`, …) in a rule body (`set-get` excepted: it indexes an internal runtime order that proofs cannot reproduce). A container built in the body is a *side condition* with no carryable proof: it is marked with an `Eval` proof step and re-evaluated against the typed rule when checked, so it can be read or matched in the query but not carried into an action (that is rejected). Two user-visible extraction changes: container terms extract in a deterministic, reproducible order rather than value-id order, and maps extract in a flat `(map-of k0 v0 …)` form (new `map-of` constructor) instead of nested `map-insert`s.

## [2.0.0] - 2026-02-11

Bigger changes

- Index catalog optimized for small set of indices (#719)
- Warn when globals lack the $ prefix; require globals to use the `$` prefix; missing prefixes now log a warning by default and can be upgraded to errors with `--strict-mode` or `EGraph::set_strict_mode`. (#722)
- Rename global vars in tests (#792, #800)
- Make interactive mode a delimiter (#729)
- Enable type-aware macros for fresh! sugar (#741)
- Proof preparation and term encoding (#742, #743, #765, #789)
- Export let bindings in the serialized format so they are visualized; Renames `ignore_viz` to `let_binding` (#701)
- Add snapshot tests (#778)

Bug fixes

- Fix Incorrect Unstable Function Behavior (#739)
- Run all tests in the workspace in CI (#776)

Performance improvements

- Low-level optimization for rebuilding (#754)
- Improve merge performance by being precise (#766)
- Avoid excessive cross-crate monomorphization (#773)
- Remove duplicate variables using functional dependency (#777)
- Memcpy for parallel writes and fix compilation failures (#779)

Misc. improvements

- Pin cargo codspeed version to fix CI (#734)
- Expose type constraints related APIs (#747)
- Remove lazy_static (#714)
- Simplify extract option handling (#759)
- Add longer extraction benchmark (#760)
- Specify that extractor does not support DAG costs (#763)
- Helpers for getting table sizes in primitives (#752)
- Refactor query planning (#780)
- Disable tracing tests (#787)
- Add initial early stopping support and use it for panic functions (#788)
- Update links in README for egglog resources (#798)


## [1.0.0] - 2025-10-18

This is the first release of egglog that is based on our new database-first, highly parallel backend.

**Abandoned features**

- `extract` is now a command instead of an action, which means calling `extract` within a rule is not allowed.
  Instead, the user is encouraged to use `print-function`.

Features

- Cost trait (#605)
- A new set of Rust APIs in `egglog::prelude` (#586)
- User-defined commands (#597)
- Scheduler interface for custom scheduling (#587)

Misc. Improvements

- Improves usability of `print-function` (#640)
- Desugar `rewrite`s to use `set`s when possible (#626)
- Grounded-ness check for ungrounded variables (#635)
- Don't panic when extracting nonexistent term (#629) 
- Documentation improvements (#634)
- Add parallelism flag and remove nondeterminism flag (#640, #642)
- Emit prompt and debug info when running from REPL (#672)
- Add support for the :unextractable flag for datatype variants (#712)
- Move egglog ast into its own crates (#670)

## [0.5.0] - 2025-6-9

This is the last major release before we switch to a database-first, highly parallel new backend.

Improvements

- Make `EGraph` thread-safe (#517)
- Support for egglog-python (#522)
- Throws type errors when unioning non-EqSort values (#561)
- Improvements to tests (#529)
- Improvements to error messages (#555)
- Makes union-find struct externally accessible (for container implementation) (#560)
- Disallow shadowing and interpret underscores as wildcards (#565)
- Faster `(push)` implementation

Bug fixes

- Fix value generations when `subsume`-ing a tuple in a relation (#569)
- Fixes to the new parser (#559)
- Rebuild after running commands instead of before (#573)

Benchmarks, serialization, and web demo

- Improvements to serialization (#520)
- Added eggcc benchmarks (#527)
- Fixes web demo escaping (#564, #566)
- Moves webdemo into a separate repository (#591)
- Fixes to Codspeed (#572)

## [0.4.0] - 2025-1-20

Semantic change (BREAKING)

- Split `function` into `constructor` and `functions` with merge functions. (#461)
- Remove `:default` keyword. (#461)
- Disallow lookup functions in the right hand side. (#461)
- Remove `:on_merge`, `:cost`, and `:unextractable` from functions, require `:no-merge` (#485)

Language features

- Add multi-sets (#446, #454, #471)
- Recursive datatypes with `datatype*` (#432)
- Add `BigInt` and `BigRat` and move `Rational` to `egglog-experimental` (#457, #475, #499)

Command-line interface and web demo

- Display build info when in binary mode (#427)
- Expose egglog CLI (#507, #510)
- Add a new interactive visualizer (#426)
- Disable build script for library builds (#467)

Rust interface improvements

- Make the type constraint system user-extensible (#509)
- New extensible parser (#435, #450, #484, #489, #497, #498, #506)
- Remove `Value::tag` when in release mode (#448)

Extraction

- Remove unused 'serde-1' attribute (#465)
- Extract egraph-serialize features  (#466)
- Expose extraction module publicly (#503)
- Use `set-of` instead of `set-insert` for extraction result of sets. (#514)

Bug fixes

- Fix the behavior of i64 primitives on overflow (#502)
- Fix memory blowup issue in `TermDag::to_string`
- Fix the issue that rule names are ignored (#500)

Cleanups and improvements

- Allow disabling messages for performance (#492)
- Determinize egglog (#438, #439)
- Refactor sort extraction API (#495)
- Add automated benchmarking to continuous integration (#443)
- Improvements to performance of testing (#458)
- Other small cleanups and improvements (#428, #429, #433, #434, #436, #437, #440, #442, #444, #445, #449, #453, #456, #469, #474, #477, #490, #491, #494, #501, #504, #508, #511)

## [0.3.0] - 2024-10-02

Cleanups

- Remove `declare` and `calc` keywords (#418, #419)
- Fix determinism bug from new combined ruleset code (#406)
- Fix performance bug in typechecking containers (#395)
- Minor improvements to the web demo (#413, #414, #415)
- Add power operators to i64 and f64 (#412)

Error reporting

- Report the source locations for errors (#389, #398, #405)

Serialization

- Include subsumption information in serialization (#424)
- Move splitting primitive nodes into the serialize library (#407)
- Support omitted nodes (#394)
- Support Class ID <-> Value conversion (#396)

REPL

- Evaluate multiple lines at once (#402)
- Show build information in the REPL (#427)

Higher-order functions (UNSTABLE)

- Infer types of function values based on names (#400)

Import relation from files

- Accept f64 function arguments #384

## [0.2.0] - 2024-05-24

Usability

- Improve statistics for runs (#284)
- Improve user-defined primitive support (#280, #288)
- Improve serialization (#293)
- Add more container primitives (#306)

Web demo

- Add slidemode in the web demo (#302)
- Fix box shadowing problem (#372)

Refactor

- Big refactoring to the intermediate representation (#320)
- Make global variables a syntactic sugar (#338)
- Drop experimental implementation for proofs and terms (#320, #342)

New features

- Support Subsumptions (#301)
- Add basic support for first-class, higher-order functions (UNSTABLE) (#348)
- Support combined rulesets (UNSTABLE) (#362)

Others

- Numerous bug fixes

## [0.1.0] - 2023-10-31

This is egglog's first release! Egglog is ready for use, but is still fairly experimental. Expect some significant changes in the future.

- Egglog is better than [egg](https://github.com/egraphs-good/egg) in many ways, including performance and new features.
- Egglog now includes cargo documentation for the language interface.

As of yet, the rust interface is not documented or well supported. We recommend using the language interface. Egglog also lacks proofs, a feature that egg has.


[Unreleased]: https://github.com/egraphs-good/egglog/compare/v2.0.0...HEAD
[0.1.0]: https://github.com/egraphs-good/egglog/tree/v0.1.0
[0.2.0]: https://github.com/egraphs-good/egglog/tree/v0.2.0
[0.3.0]: https://github.com/egraphs-good/egglog/tree/v0.3.0
[0.4.0]: https://github.com/egraphs-good/egglog/tree/v0.4.0
[0.5.0]: https://github.com/egraphs-good/egglog/tree/v0.5.0
[1.0.0]: https://github.com/egraphs-good/egglog/tree/v1.0.0
[2.0.0]: https://github.com/egraphs-good/egglog/tree/v2.0.0


See release-instructions.md for more information on how to do a release.
