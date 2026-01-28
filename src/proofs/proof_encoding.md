Rewrites an egglog program to use an encoding for equality tracking, optionally including proof tracking.

# Term Encoding

The job of the term encoding is to *remove all calls to union* in the egglog program.
This makes proof production easier, since all equality reasoning is explicit and
  can be instrumented with proof tracking.
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

(delete (Add 1 2))
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
A view table stores "canonicalized" terms and their e-class representative.
A canonicalized term has representative terms for its children.
The last column of the view table is the representative term for the e-class.
The view tables are kept up to date during rebuilding.

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

```text
(function v2 () Math :no-merge)
(set (v2) (Add 1 2))
(AddView 1 2 (v2))
```

Above is the desugaring for `(Add 1 2)`.
We add to both view and term tables whenever we evaluate
  a constructor or function application.
It's straightforward except for global variables.
Since global variables are not allowed after this pass,
  we use functions with no arguments to represent them
  (see globals section below).


```text
(rule ((AddView a b v3))
      ((let v5 (Add a b))
       (AddView a b v5)
       (let v6 (Add b a))
       (AddView b a v6)
       (UF_Math (ordering-max v5 v6) (ordering-min v5 v6)))
       :name "commutativity")
```

Here we have the instrumented commutativity rule.
The query uses the view table to find the canonical e-node.
The actions add to the term table, add to the view table,
  and add an equality to the union-find table.
We add an equality to the union-find table for the two terms, using the `ordering-max` and 
  `ordering-min` egglog primitives to correctly choose a parent.




```text
(check (AddView 1 2 v7)
       (AddView 2 1 v8)
       (= v7 v8))
```

All queries use the view tables, including check commands.
This query checks that the e-class representatives for `(Add 1 2)` and `(Add 2 1)` are equal,
  ensuring they share the same e-class.

```text
(rule ((to_delete_Add c0 c1)
       (AddView c0 c1 out))
      ((delete (AddView c0 c1 out))
       (delete (to_delete_Add c0 c1)))
        :ruleset delete_subsume_ruleset :name "delete_rule")
(rule ((to_subsume_Add c0 c1)
       (AddView c0 c1 out))
      ((subsume (AddView c0 c1 out)))
        :ruleset delete_subsume_ruleset :name "delete_rule_subsume")

(to_delete_Add 1 2)
```

Finally, deletions and subsumptions are deferred via helper tables.
For every constructor, we add a `to_delete_<Constructor>` and `to_subsume_<Constructor>` table.
When a deletion or subsumption is requested, we add to these tables.
During rebuilding, we process these tables to actually delete or subsume the requested terms.
We only need to delete or subsume from the view tables,
  since the term tables are not used for queries.
This has the added benefit of allowing us to keep terms around
  for proof tracking even after they are deleted from the e-graph.


# Globals

*Before the term encoding*, egglog desugars all global
  variables to constructors with the `proof_global_remover.rs` pass.
This makes the encoding simpler and makes it so the backend
  need not worry about globals.
The above program doesn't have any global variables, so it stays the same.
A different program like this one:
```text
(sort Math)
(constructor Add (i64 i64) Math)
(let g1 (Add 1 2))
(rule ((= g1 (Add 2 3))
      ((Add 3 4))))
```

Would desugar to this before term encoding:
```text
(sort Math)
(constructor Add (i64 i64) Math)
(constructor g1 () Math)
(union (g1) (Add 1 2))
(rule ((= (g1) (Add 2 3)))
      ((Add 3 4)))
```



# Proof Tracking

During term encoding, if proof tracking is enabled,
  we also instrument the program to track proofs of equalities.
We'll continue with our example from above, showing the additions
  for proof tracking.

Original program snippet is

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


The encoding with proof tracking adds a proof header before the rest of the program.
The header defines the proof format corresponding to [`RawProof`](crate::proofs::RawProof) in Rust.
See the proof header in `proof_encoding_helpers.rs` for details.

```text
(function MathProof (Math) Proof :merge old)
```

Every sort gets a proof table storing
  a proof for that term.
The proof proves a proposition `t = t` for
  input term `t`.
We store the oldest proof currently.

```text
(function MathUFProof (Math Math) Proof :merge old)
```

Similarly, the union-find table gets a proof table storing
  proofs of equalities between terms.
If term `a` has parent `b`, it stores a 
  proof of `a = b`.
The rules that update the union-find tables
  are instrumented to produce proofs using
  symmetry (`Sym`) and transitivity (`Trans`) as needed.


```text
(function AddViewProof (i64 i64 Math) Proof :merge old)
```

View tables are the trickiest.
We store a separate proof table per view table.
Recall that view tables store a term
  along with the e-class representative.
For a term `t` with representative `r`,
  the proof proves that `r = t`.
The direction is important, making
  proof production easier later.
We store the earliest proof currently.


```text
(rule ((AddView a b v8)
       ;; proof that v8 = Add a b
       (= v9 (AddViewProof a b v8)))
      (;; proof list, one per line of the original query
       (let v10 (PCons v9 (PNil )))
       
       (let v11 (Add a b))
       ;; Proof that Add a b = Add a b
       (let v12 (Rule "commutativity" v10 (AstMath v11) (AstMath v11)))
       ;; Setting the proof for Add a b
       (set (MathProof v11) v12)

       (AddView a b v11)
       ;; Setting the proof for the view
       (set (AddViewProof a b v11) v12)

       (let v13 (Add b a))
       ;; Proof that Add b a = Add b a
       (let v14 (Rule "commutativity" v10 (AstMath v13) (AstMath v13)))
       (set (MathProof v13) v14)
       (AddView b a v13)
       (set (AddViewProof b a v13) v14)

       (UF_Math (ordering-max v11 v13) (ordering-min v11 v13))

       ;; Set the proof that (Add a b) = (Add b a)
       (set (MathUFProof (ordering-max v11 v13) (ordering-min v11 v13))
            (Rule "commutativity" v10 (AstMath (ordering-max v11 v13)) (AstMath (ordering-min v11 v13)))))
         :name "commutativity")
```

Instrumented rules with proof tracking query proof tables,
  then construct proofs for each action.
For nested terms, congruence proofs are built to ensure
  the proof terms match the original queries.

