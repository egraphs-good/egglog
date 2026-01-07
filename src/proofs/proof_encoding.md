Rewrites an egglog program to use an encoding for equality tracking, optionally including proof tracking.

# Term Encoding
The job of the term encoding is to *remove all calls to union* in the egglog program.
The term encoding adds an explicit union-find structure per sort, and maintains it via
rules that run during scheduled maintenance.
For efficiency, every constructor becomes two tables:
  a term table that stores the actual terms, and a view table storing representative terms along with their e-class (stored as the leader term).
The term encoding enables proof tracking, done at the
  same time in this file.
The encoding keeps the operational semantics equivalent to the standard encoding (for the
subset of commands that are currently supported).

The transformation is triggered when an `EGraph` is created with
[`EGraph::new_with_term_encoding`](crate::EGraph::new_with_term_encoding) or
converted via [`EGraph::with_term_encoding_enabled`](crate::EGraph::with_term_encoding_enabled).

Consider a tiny program that defines a pure arithmetic helper and checks a fact about it:

```text
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
(run 1)
(check (= (Add 1 2) (Add 2 1)))
```

Lowering the program with the term encoding expands to a bunch of new egglog, which we'll show (most of) in pieces.

```text
(ruleset parent)
(ruleset single_parent)
(ruleset rebuilding)
(ruleset rebuilding_cleanup)
(ruleset delete_subsume_ruleset)
```

*The new rulesets* orchestrate new rules for per-sort union-find tables (`parent` and `single_parent`),
rebuild-time congruence (`rebuilding` + `rebuilding_cleanup`), and deferred deletions/subsumptions (`delete_subsume_ruleset`).

```text
(run-schedule
    (saturate
       rebuilding_cleanup ;; cleanup merged rows
       (saturate single_parent) ;; ensure each term points to single parent
       (saturate parent) ;; transitively close parent links
       rebuilding) ;; find new equalities via congruence
    delete_subsume_ruleset) ;; process deletions/subsumptions
```

*In-between* the original program's commands, the term encoding
  runs these rulesets to maintain egglog's invariants.

```text
(sort Math)
(sort uf)
(constructor UF_Math (Math Math) uf)
(rule ((UF_Math a b)
      (UF_Math b c)
      (!= b c))
     ((delete (UF_Math a b))
      (UF_Math a c))
       :ruleset parent :name "uf_update")
(rule ((UF_Math a b)
      (UF_Math a c)
      (!= b c)
      (= (ordering-max b c) b))
     ((delete (UF_Math a b))
      (UF_Math b c))
       :ruleset single_parent :name "singleparentuf_update")
```

*The union-find* tables for each sort store the equivalence 
  classes of terms of that sort.
A couple rules ensure the union-find is kept up to date as 
  equalities are added.
We use the `ordering-max` and `ordering-min` egglog primitives
  to define an arbitrary ordering on terms based on insertion order,
  so that we can deterministically choose which term becomes the parent
  in the union-find structure.


```text
(sort view)
(constructor Add (i64 i64) Math)
(constructor AddView (i64 i64 Math) view)
(constructor to_delete_Add (i64 i64) view)
(constructor to_subsume_Add (i64 i64) view)
```

Each constructor in the original program is expanded to
  a term table (`Add`), a view table (`AddView`), and helpers for deferred deletion/subsumption
  (`to_delete_Add`, `to_subsume_Add`).

```text
(rule ((AddView c0 c1 new)
       (AddView c0 c1 old)
       (!= old new)
       (= (ordering-max old new) new))
      ((UF_Math (ordering-max new old) (ordering-min new old)))
       :ruleset rebuilding :name "congruence_rule")
(rule ((AddView c0 c1 c2)
       (UF_Math c2 v)
       (!= v c2))
      ((AddView c0 c1 v)
       (delete (AddView c0 c1 c2)))
        :ruleset rebuilding :name "rebuild_rule")
```

For each constructor, we add a congruence rule and a rebuild rule.
The congruence rule adds equalities to the union-find table when two constructor applications
  have equal arguments.
The rebuild rule updates view tables so that views
  point to representative terms for child e-classes.

```
(function v2 () Math :no-merge)
(set (v2) (Add 1 2))
(AddView 1 2 (v2 ))
```

Above is the desugaring for `(Add 1 2)`.
We add to both view and term tables whenever we evaluate
  a constructor or function application.
Since global variables are not allowed after this pass,
  we use functions with no arguments to represent them.


```text
(rule ((AddView a b __v3))
      ((let __v5 (Add a b))
       (AddView a b __v5)
       (let __v6 (Add b a))
       (AddView b a __v6)
       (__UF_Math (ordering-max __v5 __v6) (ordering-min __v5 __v6)))
       :name "commutativity")
```

Here we have the instrumented commutativity rule.
The query uses the view table to find the canonical e-node.
The actions add to the term table, add to the view table,
  and add an equality to the union-find table.
We add an equality to the union-find table for the two terms, using the `ordering-max` and 
  `ordering-min` egglog primitives to correctly choose a parent.




```text
(check (AddView 1 2 __v7)
       (AddView 2 1 __v8)
       (= __v7 __v8))
```

All queries use the view tables, including check commands.
This query checks that the e-class representatives for `(Add 1 2)` and `(Add 2 1)` are equal,
  ensuring they share the same e-class.



# Proof Tracking