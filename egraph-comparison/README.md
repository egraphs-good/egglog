# egraph-comparison

```sh
cargo run -p egraph-comparison -- left.json right.json
cargo run -p egraph-comparison -- --terms-only left.json right.json
```

The binary reads two complete, rebuilt databases and emits JSON with
`terms_equal`, `database_equal`, and `refinement_rounds`. Exit status is 0 for
equality, 1 for disequality, and 2 for invalid input or I/O failure. `--terms-only`
selects term equality for the exit status; both results are always reported.

## Semantics

Comparison follows the partition refinement algorithm in Eli Rosenthal's
*Solving Two E-graph Puzzles With Partition Refinement*: refine the disjoint
union of the inputs and compare the sets of blocks represented on either side.
The two inputs need not already be minimized. A self-loop and two equivalent
mutually recursive classes compare equal, as do duplicated copies of a cycle.

`terms_equal` compares constructor-output classes by bisimulation. Node labels
include function name and signature; children are ordered, e-class members are
sets, and sorts and primitive literal values are exact. Literals are allowed as
constructor arguments, but primitive values occurring only in function tables
do not add constructor terms. Empty classes are observable through constructor
arguments. Bisimulation includes ungrounded cycles and is deliberately stronger
than equality of finite ground-term languages. This is not graph isomorphism.

`database_equal` additionally compares function declarations and all rows,
including constructor rows and subsumption flags, using the constructor blocks
as value identities. Non-constructor functions never become term operators.
Function-only equality-sort values without constructors are indistinguishable
when their sorts agree. All comparisons are of sets, so multiplicities of
duplicate rows or bisimilar values are ignored. Unused classes are ignored.
Costs, roots, extraction preferences, and runtime implementation details are
outside this database format.

The initial implementation recomputes exact signatures each round. It has at
most a linear number of splitting rounds and can take quadratic time (plus
ordered-map/set costs). There is no probabilistic equality or depth limit.

## Version 1 JSON

```json
{
  "version": 1,
  "classes": {"a": {"sort": "Expr"}, "n": {"sort": "i64", "literal": "1"}},
  "functions": {
    "Num": {"kind": "constructor", "inputs": ["i64"], "output": "Expr"},
    "cost": {"kind": "function", "inputs": ["Expr"], "output": "i64"}
  },
  "rows": [
    {"function": "Num", "inputs": ["n"], "output": "a"},
    {"function": "cost", "inputs": ["a"], "output": "n"}
  ]
}
```

Class IDs are arbitrary strings scoped to one file. `literal` is an opaque,
stable encoding qualified by its sort. `subsumed` defaults to false. Declarations
are explicit even for empty tables. Unknown fields, unsupported versions,
dangling IDs, wrong sorts/arity, duplicate literal identities, and conflicting
function rows are errors. Producers must canonicalize IDs and rebuild first.
This format is separate from visualization-oriented `egraph-serialize` JSON.

## Disequality certificates

Pass `--certificate` to include a `certificate` field (`null` for equal
inputs). Save that field alone to `witness.json` and verify it with:

```sh
cargo run -p egraph-comparison -- left.json right.json --verify-certificate witness.json
```

Verification emits `{"valid": true}` or `{"valid": false}` with status 0 or 1.
Malformed JSON/input still exits 2. The Rust API exposes `certificate` and
`verify` for the same operations.

* `missing_term`: a constructor term exists on only the indicated side.
* `unequal_terms`: both terms exist in both inputs, and are equal only on the
  indicated side.
* `structure`: a constructor class has a bounded bisimulation observation absent
  from all constructor classes in the other input. The class ID and refinement
  round count let the verifier replay that observation. This is needed for
  ungrounded cycles/holes, which may have no finite ground-term witness.
* `declaration` / `row`: a schema or table row differs. Row IDs are interpreted
  in the indicated input and compared modulo the joint constructor partition.

Finite terms use a topologically ordered DAG: each application references earlier
entries, and the certificate identifies its root(s). This avoids exponential
expansion and recursive evaluation. Functions never appear in these terms.
The extractor finds one ground representative per productive class, then checks
constructor rows against their output representatives in both directions.
Ground-term verification uses direct constructor lookup, independently of the
refinement algorithm. Certificates are valid witnesses, not guaranteed minimal.
Structural and row verification use exact refinement. Certificate generation
currently repeats comparison/refinement and row-witness search can be quadratic;
it is opt-in so equality checks do not pay this diagnostic cost.

## Exporting egglog databases

```sh
cargo run -p egglog -- --to-comparison-json before.egg
cargo run -p egglog -- --to-comparison-json after.egg
cargo run -p egraph-comparison -- before.comparison.json after.comparison.json --certificate
```

`EGraph::serialize_for_comparison` is available with egglog's `comparison`
feature (included by the default `bin` feature). It rebuilds before exporting
all visible user tables, empty declarations, and subsumed rows. Relations are
ordinary database tables. Global bindings and hidden helper tables are excluded.
Visualization limits, splitting, and inlining do not affect this export.

The initial exporter handles equality sorts and the built-in scalar types,
including normalized floating-point zeros/NaNs. It rejects encountered container
and custom base values, and rejects term/proof encoding. These cases require
explicit value semantics or a projection to user-visible tables; treating their
raw IDs or debug strings as values would give misleading equality results.
