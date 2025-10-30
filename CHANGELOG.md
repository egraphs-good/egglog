# Changes

## [Unreleased] - ReleaseDate

- Require globals to use the `$` prefix; missing prefixes now log a warning by default and can be upgraded to errors with `--strict-mode` or `EGraph::set_strict_mode`.
- Export let bindings in the serialized format so they are visualized (#701)
  - Breaking change: renames `ignore_viz` to `let_binding` in `GenericFunctionDecl`.

## [1.0.0] - 2025-8-22

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

## [0.5.0] - 2025-6-10

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

## [0.3.0] - 2024-9-12

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

## [0.2.0] - 2024-05-17

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

## [0.1.0] - 2023-10-24
This is egglog's first release! Egglog is ready for use, but is still fairly experimental. Expect some significant changes in the future.

- Egglog is better than [egg](https://github.com/egraphs-good/egg) in many ways, including performance and new features.
- Egglog now includes cargo documentation for the language interface.

As of yet, the rust interface is not documented or well supported. We reccomend using the language interface. Egglog also lacks proofs, a feature that egg has.


[Unreleased]: https://github.com/egraphs-good/egglog/compare/v1.0.0...HEAD
[0.1.0]: https://github.com/egraphs-good/egglog/tree/v0.1.0
[0.2.0]: https://github.com/egraphs-good/egglog/tree/v0.2.0
[0.3.0]: https://github.com/egraphs-good/egglog/tree/v0.3.0
[0.4.0]: https://github.com/egraphs-good/egglog/tree/v0.4.0
[0.5.0]: https://github.com/egraphs-good/egglog/tree/v0.5.0
[1.0.0]: https://github.com/egraphs-good/egglog/tree/v1.0.0


See release-instructions.md for more information on how to do a release.