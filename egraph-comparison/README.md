# egraph-comparison

```sh
cargo run -p egraph-comparison -- left.json right.json
cargo run -p egraph-comparison -- --terms-only left.json right.json
```

The binary reads two complete, rebuilt databases and emits JSON with
`terms_equal`, `database_equal`, `refinement_rounds`, and
`database_refinement_rounds`. Exit status is 0 for
equality, 1 for disequality, and 2 for invalid input or I/O failure. `--terms-only`
selects term equality for the exit status; both results are always reported.

## Semantics

Partition refinement starts with a coarse grouping of e-classes and repeatedly
splits groups whose members have different observations. Initially, classes are
grouped by sort. A class's observation is the set of its e-node labels together
with the current groups of each node's ordered children. Refinement stops when
no group can split. Comparing two inputs refines their disjoint union and checks
that the same groups contain observable roots on both sides.

This is the idea behind [DFA minimization](https://en.wikipedia.org/wiki/DFA_minimization).
For its generalization to other transition structures, see Jacobs and Wißmann,
[Fast Coalgebraic Bisimilarity Minimization](https://arxiv.org/abs/2204.12368).
The implementation here uses whole-graph rounds; it does **not** implement
Hopcroft's smaller-half splitter worklist or the fast algorithm in that paper.

This is conservative e-graph partition refinement: cycles merge only when the
complete sets of e-nodes in their e-classes are identical modulo the refined
child groups. Merely overlapping sets do not suffice. A self-loop and two
mutually recursive classes compare equal when they have the same complete
observations, as do duplicated copies of a cycle. Adding an unmatched member to
one class can distinguish it even when both classes still contain a common term.

`terms_equal` compares constructor-output classes by bisimulation. Node labels
include function name and signature; children are ordered, e-class members are
sets, and sorts and primitive literal values are exact. Literals are allowed as
constructor arguments, but primitive values occurring only in function tables
do not add constructor terms. Empty classes are observable through constructor
arguments. Bisimulation includes ungrounded cycles and is deliberately stronger
than equality of finite ground-term languages. This is not graph isomorphism.

`database_equal` additionally compares function declarations and all rows,
including constructor rows and subsumption flags. Its full-database refinement
includes ordinary function calls in class observations. This preserves
structure carried only by function tables, even when values have no constructor
terms. Values in the rows are compared using this full-database partition.
Non-constructor functions never become term operators or enter finite term
certificates. All comparisons are of sets, so multiplicities of duplicate rows
or bisimilar values are ignored. Unused classes are ignored. This still compares
bisimulation quotients, not bijective identities or graph isomorphism: empty
classes of the same sort remain indistinguishable if no incoming rows give them
different observations.
Costs, roots, extraction preferences, and runtime implementation details are
outside this database format.

The implementation recomputes exact signatures each round. It has at most a
linear number of splitting rounds and can take quadratic time (plus signature
sorting). There is no probabilistic equality or depth limit. Performance
improvements preserve exact signature equality.

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

Class IDs are arbitrary strings scoped to one file. The numeric `version` field
is a `FormatVersion` identifying the wire format, not an e-class identifier.
Setup resolves class names to distinct typed class and partition IDs.
`literal` is an opaque, stable encoding qualified by its sort. `subsumed` defaults to false. Declarations
are explicit even for empty tables. Unknown fields, unsupported versions,
dangling IDs, wrong sorts/arity, duplicate literal identities, and conflicting
function rows are errors. Producers must canonicalize IDs and rebuild first.
This format is separate from visualization-oriented `egraph-serialize` JSON.
