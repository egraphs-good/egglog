# Changes

## [Unreleased] - ReleaseDate

## [3.0.0] - 2026-08-18

### Breaking changes

- **Relations are now constructors.** A `relation` is desugared to a non-unionable constructor instead of a function returning unit `()`.
- **Custom primitives use capability-specific traits (#772).** Implement `PurePrim`, `ReadPrim`, `WritePrim`, or `FullPrim` and register it with the corresponding `add_*_primitive` method. `rust_rule` callbacks now take `&mut WriteState` instead of `RustRuleContext`; use `rust_rule_full` when a callback also needs read access. Higher-order primitives dispatch through `state.apply_function`.
- **Thread configuration is now per e-graph.** The global Rayon pool is replaced by a scoped `egglog-concurrency` pool. `EGraph::set_num_threads` now takes `&mut self`; `EGraph::new`, `with_num_threads`, and `set_num_threads` configure one e-graph without affecting others.
- **The function schema API has changed.** `Function::schema()` is replaced by `Function::func_type()`, `ResolvedSchema` is replaced by `FuncType`, and `GenericFunctionDecl::resolved_schema` is removed.
- **Backend execution APIs now take an `ExternalContext`.** `Database::run_rule_set`, `with_execution_state`, `with_execution_state_tracked`, and the corresponding bridge APIs require the context; pass `None` when no external data is needed.
- **`EGraph::print_function` has a new signature.** It now takes an `Option<(File, PathBuf)>` output sink and a `Span`, allowing write failures to return `Error::IoError`.

### New features and improvements

- **Add name-indexed e-graph access for primitives and `rust_rule` callbacks (#745, #751).** The `Read` and `Write` APIs now support table lookup and mutation, table and constructor scans, schema queries, table-size queries, and indexed e-class traversal by name. `EGraph::read`, `EGraph::update`, and top-level scan methods expose the same capabilities outside callbacks. Invalid names, types, and arities return `Error::ApiError`. These APIs support out-of-tree extensions such as `unstable-subst` in `egglog-experimental`.
- **Add container support to the term/proof encoding.** Programs can read and construct `Vec`, `Set`, `Map`, `MultiSet`, and `Pair` values in rule bodies (`set-get` remains unsupported). Body-created containers are proof side conditions and cannot be carried into actions. Container extraction is now deterministic, and maps extract in flat `(map-of k0 v0 …)` form.
- **Add `:naive` and `:unsafe-seminaive` rule options.** Both permit database reads on a rule's right-hand side. `:naive` matches the full database each iteration; `:unsafe-seminaive` keeps delta matching but can be evaluation-order dependent and is rejected by the term/proof encoding. The options are mutually exclusive.
- Add typed `EGraph` extension state that clones with the e-graph and is restored by `push` and `pop`.
- Add typechecking and evaluation APIs for body-defined primitives: `EGraph::typecheck_expr_with_bindings_and_output`, `Core::eval_resolved_expr`, and `Core::apply_primitive`. `unstable-fn` containers can now target primitive overloads.
- Add `f64` `exp`, `log`, and `sqrt` primitives, plus `BigRat`-to-`i64` conversion for integral values.
- Add `RunReport::can_stop` so scheduler progress can be distinguished from database updates.
- Report full source paths in spans and error messages.

### Performance

- Speed up query evaluation with sorted-array subset indexes and by sharing trie roots across compatible query plans. Measured improvements include about 33% on `gemma` from sorted indexes and about 15% on `whisper`, 12% on `gemma`, and 8% on `qwen3_moe` from trie sharing.

### Bug fixes

- Fix a crash where constrained bounded scans could return stale, deleted rows to primitives.
- Replace many reachable panics with recoverable errors, including malformed declarations and commands, invalid primitive calls, unavailable proof operations, missing files, duplicate rule names, and unknown rulesets. Out-of-range partial primitives now return no result, variable-free scheduled rules execute correctly, and scheduler state is restored after errors.
- Fix seminaive matching after nested containers are rebuilt in place.
- Fix custom scheduler queries offering subsumed rows as fresh matches.
- Fix multi-column secondary indexes returning rows out of order or recording duplicates during rebuilds (#914).
- Fix builds of egglog as a library with default features disabled.
- Improve printed syntax: nullary calls render as `(foo)`, and string literals escape `"` and `\` so they round-trip through the parser.

## [2.0.0] - 2026-02-11

Bigger changes

- **Breaking:** extraction APIs now return `TermId` values paired with a `TermDag` instead of returning owned `Term` values. This affects `EGraph::extract_value`, `Extractor::extract_best`, `EGraph::function_to_dag`, and the extraction variants of `CommandOutput`.
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


[Unreleased]: https://github.com/egraphs-good/egglog/compare/v3.0.0...HEAD
[0.1.0]: https://github.com/egraphs-good/egglog/tree/v0.1.0
[0.2.0]: https://github.com/egraphs-good/egglog/tree/v0.2.0
[0.3.0]: https://github.com/egraphs-good/egglog/tree/v0.3.0
[0.4.0]: https://github.com/egraphs-good/egglog/tree/v0.4.0
[0.5.0]: https://github.com/egraphs-good/egglog/tree/v0.5.0
[1.0.0]: https://github.com/egraphs-good/egglog/tree/v1.0.0
[2.0.0]: https://github.com/egraphs-good/egglog/tree/v2.0.0
[3.0.0]: https://github.com/egraphs-good/egglog/tree/v3.0.0


See release-instructions.md for more information on how to do a release.
