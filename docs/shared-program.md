# Shared programs

`egglog::program::Program` is a versioned representation of the existing,
unresolved Egglog command tree. Text, Rust builders, and foreign-language
bindings can use the same native `Command` records. JSON contains structured
commands, expressions, facts, actions, declarations, and schedules. There is
no command encoded as a block of opaque Egglog text.

```rust
use egglog::{EGraph, program::Program};

let program = Program::parse(None, "(let $answer (+ 40 2))")?;
let json = program.to_json()?;
let imported = Program::from_json(&json)?;
EGraph::default().run_shared_program(imported)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Program::new(commands)` accepts native records, including names that cannot
be written in the text language. `Program::parse` uses the built-in parser
without constructing an e-graph. When parsing depends on registered parser
extensions, use that configured parser and pass its commands to `Program::new`.
Imported programs should remain native records: converting through a language
binding's older command wrappers may drop fields that those wrappers do not
expose.

`EGraph::run_shared_program` validates structure and enters the ordinary
`run_program` path. Macro expansion, desugaring, type inference/checking,
global removal, and existing execution still occur. This is not a resolved
IR, a direct typed linker, or a change to repeated expression evaluation.
Expressions remain trees; serialization does not introduce DAG sharing or
common-subexpression elimination.

## Format and exactness

The JSON root is `{ "format": "egglog-program-v1", "commands": [...] }`.
Data-bearing enums use adjacent `type` and `value` tags. Existing Rust tuple
variants retain positional payloads; the Rust API has not been rewritten into
synthetic `field1`/`field2` structs for serialization.

Literal kinds are explicit. `Int` contains a canonical decimal i64 string;
`Float` contains exactly sixteen lowercase hexadecimal digits representing
IEEE-754 bits. This preserves integer/float distinctions, signed zero,
infinities, and NaN payloads. `String`, `Bool`, and `Unit` retain distinct tags.
The equality behavior of `OrderedFloat` is not a claim of bitwise equality;
round-trip checks compare float bits directly.

Other numeric metadata, such as constructor costs, repeat counts, print limits,
and source offsets, remain JSON integers with their Rust type's bounds in the
schema. Consumers must use lossless integer parsing for values above 2^53;
ordinary JavaScript `JSON.parse` is not sufficient for every such program.
Counts represented by `usize` are also bounded by the destination platform.

Every command field is preserved, including proof/container metadata,
`unionable`, `term_constructor`, rule evaluation modes, and internal flags.
Source locations preserve source text and offsets or an owned Rust filename.
Shared source `Arc`s are serialized by value; a large source file referenced
by many nodes can produce substantial duplication. This format makes no
compactness claim. A source table would require a separately reviewed format.

Unknown versions, unknown fields, malformed literal strings, and invalid
UTF-8 source ranges are rejected. Unknown source spans have safe diagnostic
formatting. Checked JSON ingress/egress is limited to 16 MiB, JSON nesting to
1024 levels and 100,000 JSON objects/arrays. Checked source parsing is limited
to 256 nested lists and 100,000 lists. Native Rust construction/execution keeps
its frontend's limits and validates spans iteratively; it does not apply the
wire budgets to already-constructed commands.
Bounded stack growth protects JSON serialization and deserialization; limits
are errors, never truncation. Use `Program::from_json`, not raw serde decoding,
for the input-size/nesting checks. Runtime execution is not resource-sandboxed
by these structural limits.

## Schema and another language

`Program::schema()` uses stable Schemars against the same Rust types that
serialize programs. `schema/program-v1.schema.json` is its generated artifact.
No parallel hand-maintained Python schema defines the protocol. A consumer can
validate against this schema or use a schema-based model generator. Runtime
checks additionally enforce literal range/canonicalization, source ranges,
and aggregate resource bounds that JSON Schema alone does not express.

```text
cargo run --no-default-features --example shared_program -- schema schema/program-v1.schema.json
cargo run --no-default-features --example shared_program -- encode tests/program/portable.egg program.json
cargo run --no-default-features --example shared_program -- run-json program.json
cargo run --no-default-features --example shared_program -- run-text tests/program/portable.egg
```

The earlier [JSON Schema Support PR #736](https://github.com/egraphs-good/egglog/pull/736)
and [issue #728](https://github.com/egraphs-good/egglog/issues/728) proposed
generating foreign-language bindings from Rust schemas. PR #736 was closed
unmerged. This implementation builds on that purpose using published Schemars,
retaining the existing Rust variant shapes, validating owned spans, and using
exact tagged literals instead of ambiguous JSON numbers. It does not attempt
to serialize every `CommandOutput`, including opaque host outputs.

## Text export

`to_egglog()` is diagnostic formatting. It can display native names that the
surface parser cannot read, and the existing formatter does not emit every
internal field. It must not be used as a lossless persistence format.

`to_replayable_egglog()` renders text, parses it with the built-in parser, and
compares every structural field except source locations and singleton schedule
`Sequence` wrappers introduced by the parser. Those wrappers execute the same
single child in the same order, and core schedule normalization already removes
them. It returns an error
if this would change metadata, names, or literal bits. It never silently
renames public identities. Internal `@` names, names containing spaces, and
some internal metadata therefore require JSON. Parser-specific extensions
may also require JSON or a configured host parser.

The strict check establishes structural representability, not a hermetic
execution environment. `Include`, `Input`, and `Output` retain file paths;
their files and permissions are not bundled. `UserDefined` retains the command
name and argument trees, but its implementation is a host capability.
Custom primitives, sorts, command macros, schedulers, native cost callbacks,
and Python object registries likewise require compatible host setup. The
format does not pretend that arbitrary closures or objects are portable.

## Command recording

`EGraph::start_recording()` begins a new opt-in record. It replaces previous
history. `try_start_recording()` starts only when no record exists and returns
false without changing an existing record, allowing scoped frontend recorders
to reject nesting. `recorded_program()` returns the submitted commands so far while
recording continues. `stop_recording()` disables recording and returns a
`CommandRecord` with an outcome for each submitted command. `CommandRecord::program()`
assembles the same command sequence.
`CommandRecord::to_json` and `from_json` use the same checked, bounded JSON
codec as programs and preserve the outcomes as well as the commands.

Each entry describes an outermost `run_program` command, before command-macro
expansion, with `Pending`, `Success`, or `Failure { message }` outcome.
Included commands, nested `Fail`, and nested host calls are not duplicated.
Parser errors occur before a command exists and do not fabricate an entry.
Runtime errors are returned unchanged; they are not swallowed or reclassified.
A failed command may have committed partial effects. A host panic can leave a
`Pending` entry; recording suppression is restored before rethrowing the
original panic, so a caller that catches it can continue recording. Host
unwinding is not converted into an Egglog error.

Some hosts defer an exception in a side channel while a primitive returns a
native no-match. `run_shared_program_with_command_error` lets such a frontend
observe each outermost command once while recording and annotate a newly
reported host error. This hook changes only the recorded outcome; the native
result and whether subsequent commands execute stay unchanged. The frontend
must identify new errors without consuming/resetting its error latch. For
example, a Python exception in a rule premise can produce a recorded failure
followed by successful native commands before Python raises the deferred error.

History survives push/pop. Cloning an e-graph copies its log independently,
so later commands in two clones do not interleave in a shared log. The record
includes failed attempts and subsequent commands submitted after a failure.
Replaying the assembled `Program` normally stops at its first failing command;
it does not reproduce the state reached by a caller that caught that error
and submitted later commands. Outcomes are necessary to interpret such a
session, and no transactional or successful-state-snapshot claim is made.

This is a command record, not a complete API trace. Direct table updates,
host registration, custom extraction calls, and other operations that do not
submit commands are not replayed. Host capabilities and initial e-graph
configuration must be supplied separately.
