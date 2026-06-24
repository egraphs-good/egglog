# Changes

## [Unreleased] - ReleaseDate

- Proof/term encoding: trivial value-replacement custom-merge functions with an eq-sort OUTPUT (e.g. `(function keep (i64) T :merge old)` / `:merge new`, where `T` is a datatype) now use the functional-dependency pair-valued view like every other proof-supported merge. With this, the legacy per-rule merge-proof path (`handle_merge_fn`) is fully removed: all proof-supported custom merges are FD, so the non-FD branch is now `unreachable!` (function-reading merges and non-global `:no-merge` customs are already rejected at the file level). The now-dead `Justification::Merge` variant was removed.
- Proof/term encoding: constructor-bodied custom-merge functions with eq-sort INPUTS (e.g. rw-analysis's `(function const-prop (Loc VarT) Val :merge (merge-val old new))`) now use the functional-dependency view. A constructor-bodied `:merge` MINTS constructor enodes; in a fixpoint analysis almost all collisions are no-ops (the merged output equals the existing one), which normal egglog short-circuits via its `cur == new` check so it mints nothing. Previously the FD view's value carried an extra proof column (proof mode) or ran a minting body (term mode), so an equal-output collision would still run the body and mint a spurious `merge-val(out, out)`, diverging `(print-size)` from normal mode. This is fixed by a new backend identity-column merge short-circuit.
- egglog-bridge: `FunctionConfig` gains `identity_values: Option<usize>`. When `Some(k)`, a key collision whose leading `k` value columns are unchanged is treated as a no-op merge — the existing row is kept verbatim and the (possibly side-effecting) merge body is NOT evaluated; trailing value columns are passengers. `None` (the default for ordinary functions) preserves the classic behavior. The opt-in is scoped to proof-encoding FD views via the internal function annotation `:identity-values <n>` (stamped onto every generated FD view declaration and threaded through the function decl so it survives the desugar round-trip). Ordinary user functions carry no annotation and keep `None`, so a user `:merge` whose body changes the value on an equal-value collision (e.g. `MergeFn::Const` or `:merge (+ old 1)`) is never short-circuited.

- Proof/term encoding (Phase C): primitive-bodied custom-merge functions with eq-sort INPUTS (e.g. `(function distance (N N) i64 :merge (min old new))`) now use the functional-dependency view instead of the legacy per-rule merge proof. A rebuild canonicalizes an eq-sort input and rewrites the row's per-row proof into a non-reflexive congruence proof; the FD merge's `MergeFn` justification requires reflexive premises, so at resugaring time each such premise `p : A = B` is reflexivized to `Trans(Sym(p), p) : B = B` (landing on the canonical view row).

- Proof/term encoding (Phase B): function-bodied custom-merge functions whose `:merge` builds constructor terms (e.g. `(function f (i64) Tree :merge (C2 (C1 old new) (C2 old new)))`) now use the functional-dependency pair-valued view like constructors and primitive-bodied customs, via an internal `fd-mint` merge form that mints each nested constructor's e-class plus its view/UF/term-proof rows inside the `:merge`. This folds the per-merge congruence work into the view's own `:merge`.

- Proof/term encoding: fold the single-parent union-find invariant into the UF function index's own `:merge` instead of a separate `single_parent` ruleset. A key collision on the index (one source term with two parents) unions the two parents back into the UF table and keeps the smaller leader; path compression then removes the redundant edge. This is the same separate-rule-to-`:merge` consolidation that sped up constructor congruence.
- Multi-value function results: a function whose output sort is a `Pair` container is now stored as two value columns (its component sorts) instead of one boxed container value. Reads/lookups transparently box the columns back into a `(pair ..)` value and writes unbox into the columns; `(pair-first (f k))` / `(pair-second (f k))` fuse to a direct column read. This re-encodes proof-mode constructor views as `(children) -> (output, proof)`, so the FD `:merge` builds the congruence proof from each row's own proof — fixing proof-mode const-fold/commute correctness.

- Add typed `EGraph` extension state that clones with `EGraph` and is restored by `push`/`pop`.
- Report full source file paths in egglog span and error messages.
- Fix seminaive matching after nested containers rebuild in place by propagating dirty container ids through parent containers.
- Render nullary AST calls without a trailing space, e.g. (foo) instead of (foo ).
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
