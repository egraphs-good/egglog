# Changes

## [Unreleased] - ReleaseDate

- Desugar `relation`s to `constructor`s to simplify the language and implementation. Relations no longer return unit `()` values.
- Refactored API to use [`TermId`] more consistently instead of `Term` where possible, simplifying egglog code.
- **Typed primitive surface for seminaive safety (#772).** Custom primitives now implement `PrimitiveCommon` (name + type constraint) plus one of four kind-specific traits — `PurePrim`, `WritePrim`, `ReadPrim`, `FullPrim` — corresponding to the state wrapper their body receives (`PureState` / `WriteState` / `ReadState` / `FullState`). Methods on the wrappers come from sealed capability traits (`Core`, `Write`); the Rust type system makes only the kind's allowed operations callable, and the egglog typechecker rejects calls from contexts the kind is not valid in (e.g. a `WritePrim` used inside a query). Register with the matching `add_pure_primitive` / `add_write_primitive` / `add_read_primitive` / `add_full_primitive`. `rust_rule` callbacks now take `&mut WriteState` directly, replacing `RustRuleContext`. Higher-order primitives (`unstable-app`, `unstable-multiset-map` / `flat-map` / `filter` / `reduce`, `unstable-vec-map`) dispatch via a runtime `Context` flag on `ExecutionState` — single-bodied registration works in both queries and actions.
- **Typed `EGraph` user API (#745, #751).** First-class read/write methods on `EGraph` that mirror the egglog DSL one-to-one and don't require `parse_and_run_program`:
  - `EGraph::set(table, key, value)` — set a function table cell, mirrors `(set (f k) v)`.
  - `EGraph::add_node(table, inputs)` — mint or look up a constructor / relation eclass, mirrors `(Cons k1 k2)` and `(R k1 k2)`. Returns the eclass `Value`.
  - `EGraph::lookup::<_, V: BaseValue>(table, key)` — read a function's output value, returns `Option<V>`.
  - `EGraph::eclass_of::<_, M: EqSortMarker>(table, inputs)` — read a constructor's eclass without minting, returns `Option<EClass<M>>`.
  - `EGraph::contains` / `EGraph::remove` — work on any subtype.
  - `EGraph::query::<R: FromRow>(table)` / `EGraph::query_pattern::<R>(vars, facts)` — typed iteration / pattern matching.
  - `EGraph::intern::<T>(x)` / `EGraph::extract::<T>(v)` — base-value conversion.
  - `EClass<M>` (with the `EqSortMarker` trait) gives compile-time sort tags to eclass handles. Plugs into the row trait surface (`IntoRow`, `IntoColumn`, `FromRow`, `FromColumn` from `crate::api`).
  - Each subtype-specific method errors loudly when called on the wrong subtype (`set` on a constructor, `add_node` on a function, `lookup` on a constructor, `eclass_of` on a function).
- **Rust-rule callback ergonomics (#696).** Two changes:
  - `Core::register_container` and `container_to_value` now take `&self` (the underlying container store is interior-mutable). User closures no longer need spurious `mut ctx` for container interning.
  - New `rust_rule!` macro generates a typed bindings struct from `vars![...]`. Inside the closure, `b.x: i64` is the extracted Rust value — no manual `value_to_base::<T>(*v)` per arg, no positional `let [x, y] = values else { unreachable!() };`. The macro covers base-value vars; eclass / container vars fall back to the lower-level `add_rust_rule()` function. The function `rust_rule()` is renamed to `add_rust_rule()` so the macro can sit alongside it in the prelude.
- Hide internal IR types from the public surface (issue #751). The following items are no longer re-exported from the `egglog` crate root:
  - `Atom`, `AtomTerm`, `SpecializedPrimitive` (now `pub(crate)` re-exports; `SpecializedPrimitive` remains structurally `pub` only because it appears in `ResolvedCall::Primitive`, but its fields are private and it is no longer reachable from `egglog::*`).
  - `ResolvedExpr`, `ResolvedFact` (now `pub(crate)` type aliases; only the internal IR uses them).
  - `TypeInfo::expr_has_function_lookup` is now `pub(crate)` — it took a `ResolvedExpr` and only had internal callers.
  - `ResolvedVar` and `ResolvedCall` remain public because the public `EGraph::resolve_program` returns `Vec<ResolvedCommand>` (`= GenericCommand<ResolvedCall, ResolvedVar>`), which is consumed by external callers (e.g. `sanitize_internal_names`).
  - `egglog_bridge::FunctionId` is verified to NOT be re-exported from the `egglog` crate; the `Function` struct's `backend_id` field is private.
- Added a builder API for declaring tables: `EGraph::declare(name).input(...).output(...).function(...)` / `.constructor(...)` / `.relation()`. The free functions `add_function`, `add_constructor`, and `add_relation` in `egglog::prelude` are now `#[deprecated]` thin wrappers over the builder.
- New `egglog!` proc macro from `egglog-macros` validates an embedded egglog program at compile time and expands to `parse_and_run_program`. Typechecker errors become Rust compile errors.

### Migration notes

- If you imported `egglog::Atom`, `egglog::AtomTerm`, `egglog::SpecializedPrimitive`, `egglog::ResolvedExpr`, or `egglog::ResolvedFact`, those imports will fail. These types are internal IR; if you need them, please open an issue describing your use case. `ResolvedCall` and `ResolvedVar` (and `egglog::ast::ResolvedCommand`) remain available.
- Replace `add_function(eg, name, schema, merge)` with `eg.declare(name).input(...).output(...).function(merge)`. Same for `add_constructor` and `add_relation`. The deprecated free functions still work for one release.

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
