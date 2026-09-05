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
