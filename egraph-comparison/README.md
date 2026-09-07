# egraph-comparison

```sh
cargo run -p egraph-comparison -- left.json right.json
cargo run -p egraph-comparison -- --terms-only left.json right.json
```

The binary reads two complete, rebuilt databases and emits JSON with
`terms_equal`, `database_equal`, `refinement_steps`, and
`database_refinement_steps`. Exit status is 0 for
equality, 1 for disequality, and 2 for invalid input or I/O failure. `--terms-only`
selects term equality for the exit status; both results are always reported.
Step counts measure processed worklist blocks, not synchronous refinement depth.

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
The comparator uses the paper's smaller-half worklist idea: when a block splits,
its largest piece keeps the old ID. Only predecessors of newly numbered pieces
become dirty. Each block stores its clean members contiguously, so processing
recomputes dirty signatures plus one clean representative, without scanning all
clean members. Clean members have identical signatures and stay together.

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
includes ordinary function calls in class observations; constructor-only inputs
without subsumption can reuse the term partition. This preserves
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

Each class changes block IDs at most O(log n) times, giving O(m log n) reverse
edge visits and O(n + m log n) signature evaluations for n classes and m child
occurrences. A signature still scans, sorts, and deduplicates all of its class's
nodes, so this is not an unconditional O(m log n) runtime guarantee. The reverse
index and refinable partition use O(n + m) space, in addition to signature
scratch storage. Comparisons remain exact, with no depth limit. Internal block
IDs depend on worklist order and are not canonical serialization IDs.

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
Setup resolves class names to distinct typed class, symbol, and partition IDs.
`literal` is an opaque, stable encoding qualified by its sort. `subsumed` defaults to false. Declarations
are explicit even for empty tables. Unknown fields, unsupported versions,
dangling IDs, wrong sorts/arity, duplicate literal identities, and conflicting
function rows are errors. Producers must canonicalize IDs and rebuild first.
This format is separate from visualization-oriented `egraph-serialize` JSON.
