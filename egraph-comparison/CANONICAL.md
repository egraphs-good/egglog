# Canonical comparison graphs

```sh
cargo run -p egraph-comparison -- --canonical input.json > canonical.json
cargo run -p egraph-comparison -- --canonical --terms-only input.json > terms.json
```

The Rust API is `canonicalize(&Database, CanonicalMode) -> Result<Vec<u8>, Error>`.
For valid inputs and the same mode, byte equality is equivalent to the
corresponding `compare` result: `terms_equal` for `Terms`, `database_equal` for
`Database`. This includes ungrounded cycles, holes, and duplicate bisimilar
classes. It does not identify graphs solely by their finite term languages.
The full mode includes unused declarations and subsumption; term mode ignores
ordinary functions and subsumption. Input validation applies in both modes.

## How numbering becomes canonical

Start from row-output roots in the selected projection and retain their reachable
classes. Sort the observed sorts and complete node labels; these define the
initial block and symbol IDs. Each refinement round builds exact signatures
`(previous block, sorted unique node signatures)`. Intern signatures with full
key equality, sort the unique signatures lexicographically, and assign IDs in
that order. Each node signature is `[symbol ID, ordered child block IDs...]`.
Retaining the previous block allows only splits. Stop when no block splits.

Equivalent rooted graphs have the same reachable behaviors, the same initial
ordered partition, and inductively the same ordered signatures at every round.
At the fixed point, emit one class per block and the sorted set of root block
IDs. Conversely, identical serialized quotients supply a bisimulation between
the roots. Full-mode bisimulation also preserves the constructor projection:
matching constructor nodes identify matching constructor roots. Thus the full
encoding does not need a second copy of the term quotient.

Input IDs, row order, duplicate rows, multiplicities of equivalent classes, and
unreachable classes cannot affect bytes. Sorting unused labels or sorts would
break this property, so they are removed before numbering. Hash values are never
used as semantic identities. The algorithm is iterative and has no depth cutoff.
In addition to the pairwise algorithm's signature work, this implementation sorts
all unique block signatures on every round and serializes the quotient.

## Version 1 encoding

The output is compact UTF-8 JSON with one trailing newline, in this field order:

- `format`: `"egraph-comparison-canonical"`.
- `version`: `1`.
- `mode`: `"terms"` or `"database"`.
- `declarations`: the input declaration map, only in database mode.
- `sorts`: sorted observed sort names.
- `labels`: sorted complete labels. Literals encode as `{"Literal":"value"}`;
  calls as `{"Call":["name",function_schema,subsumed]}`. Literal labels precede
  calls; within each variant the fields follow Rust lexicographic ordering.
  Schemas order by kind (constructor before function), input sorts, output sort.
- `roots`: sorted unique class IDs.
- `classes`: records in class-ID order, each with `sort` (sort ID), then `nodes`
  (sorted unique node-signature arrays).

String and map-key ordering follows Rust `str` ordering. Schemas encode fields
in `kind`, `inputs`, `output` order. Changing canonical ordering, semantics, or
JSON encoding requires a format-version change and baseline regeneration.
Compare bytes only within the same format version and mode; cross-version
inequality is not a certificate of semantic disequality.

This is a **comparison quotient**, not version 1 input `Database` JSON. For
example, `x=f(x)` and `y=f(y)` can merge even when `g(x)` and `g(y)` have different
additional members. Their quotient contains one input tuple for `g` with two
distinct output blocks, which input validation would reject. Keep original
inputs when generating or verifying existing disequality certificates. The
format records enough structure to compare exactly, but has no certificate
reader yet. Canonical byte equality also leaves exporter coverage unchanged;
containers, custom base values, and term/proof projections still need explicit
semantics before broad snapshot migration.
