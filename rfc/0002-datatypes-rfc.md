# The Problem

Egglog enables users to use `function` in two ways:
1. As datatypes that can be made equal to each other using `union`. 
Under the hood these get fresh identifiers in the union-find.
Example: `(function Add (Math Math) Math)`
2. As datalog-like relations with a functional dependency between the inputs and the outputs.
These relations cannot be constructed without an output. By default, the merge function is `union`.
Users can also set a `default` output for these.
Example: `(function lower-bound (Math) Math :merge (MyMax old new))`
could be set using `(set (lower-bound (Add (Num 1) (Num 2))) (Num 3))`


However, this difference is subtle and egglog does not enforce it. This can lead to weird problems
(see issue #298).


# Proposed Solution

Make egglog `datatype` tables and `function` tables different. I'm not happy with these names please suggest better ones.

The first type of table is called a `datatype`:

```
(sort Math)
(datatype Add (Math Math) Math)
(datatype MyMax (Math Math) Math)
```

The second type of table is called a `function`:

```
(sort Math)
(function lower-bound (Math) Math :merge (MyMax old new))
```

- A `datatype` table cannot be `set`, it can only be `union`ed with other datatypes.
- A `function` table cannot be `union`ed, it can only be `set`.
- A `datatype` cannot have a `default`, nor can it have a `merge` function.
- a `function` contains datatypes- since datatypes don't have their own id, they can't be present in other tables.



# Examples

This proposal allows creating functions that have a merge function of union.

```
(function has-type (Math) Math :merge (union old new))
```

These functions behave similarly to `datatype`s, but they never have their own id- they can only be `set` to a datatype.



# Advantages

- Clarifies when to use `union` and when to use `set`.
- Clarifies which tables are constructors, and which are analysis.
- Makes it possible to perform the term encoding and proof encoding soundly and cleanly.
- Allows visualization to treat analysis functions specially, since putting them in the same
eclass as their output is confusing.



# Parsing

Parsing a `datatype` is similar to current function parsing, but without `merge`, `on_merge`, or `default`.
Parsing a `function` is similar to current function parsing, but `merge` is required.


