Rewrites an egglog program to use an encoding for equality tracking, optionally including proof tracking.

# Term Encoding

The job of the term encoding is to *remove all calls to union* in the egglog program.
This makes proof production easier, since all equality reasoning is explicit and
  can be instrumented with proof tracking.
The term encoding adds an explicit union-find structure per sort, and maintains it via
  rules that run during scheduled maintenance.
To speed up rebuild queries, each sort now uses two UF tables:
  a constructor UF table (`UF_<Sort>`) that stores raw parent edges, and a function UF table
  (`UF_<Sort>f`) that stores the current parent for each term as an index.
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
(ruleset uf_function_index)
(ruleset rebuilding)
(ruleset rebuilding_cleanup)
(ruleset delete_subsume_ruleset)
```

*The new rulesets* orchestrate new rules for per-sort union-find tables (`parent` and `single_parent`),
building a fast function index over UF (`uf_function_index`),
rebuild-time congruence (`rebuilding` + `rebuilding_cleanup`), and deferred deletions/subsumptions (`delete_subsume_ruleset`).

```text
(run-schedule
    (saturate
       rebuilding_cleanup ;; cleanup merged rows
       (saturate single_parent) ;; ensure each term points to single parent
       (saturate parent) ;; transitively close parent links
       (saturate uf_function_index) ;; mirror UF constructor rows into UF function index
       rebuilding) ;; find new equalities via congruence
    delete_subsume_ruleset) ;; process deletions/subsumptions
```

*In-between* the original program's commands, the term encoding
  runs these rulesets to maintain egglog's invariants.

```text
(sort Math)
(function UF_Math (Math Math) Unit :merge old :internal-hidden)
(function UF_Mathf (Math) Math :merge new)
```

*The union-find* tables for each sort store the equivalence
  classes of terms of that sort.
`UF_<Sort>` remains the source of truth for UF maintenance updates,
  while `UF_<Sort>f` is a function-backed index used by rebuild rules.
`UF_<Sort>` is always a function whose output type is `Unit` (without proof tracking)
  or `Proof` (with proof tracking).
Using `:merge old` ensures that only the first proof/unit value is kept.
When proof tracking is enabled, proofs are stored directly in the UF table
  (e.g., `(function UF_Math (Math Math) Proof :merge old :internal-hidden)`).

```text
(rule ((UF_Math a b)
      (UF_Math b c)
      (!= b c))
     ((delete (UF_Math a b))
      (set (UF_Math a c) ()))
       :ruleset parent :name "uf_update")
(rule ((UF_Math a b)
      (UF_Math a c)
      (!= b c)
      (= (ordering-max b c) b))
     ((delete (UF_Math a b))
      (set (UF_Math b c) ()))
       :ruleset single_parent :name "singleparentuf_update")
(rule ((UF_Math a b))
      ((set (UF_Mathf a) b))
       :ruleset uf_function_index :name "uf_function_index_update")
```

*Union-find rules:*
A couple rules ensure the UF function is kept up to date as
  equalities are added, and the indexing ruleset mirrors those rows
  into the function UF.
We use the `ordering-max` and `ordering-min` egglog primitives
  to define an arbitrary ordering on terms based on insertion order,
  so that we can deterministically choose which term becomes the parent
  in the union-find structure.

**Important invariant:** every representative term must have a self-loop
  entry in the constructor union-find table (e.g., `(UF_Math v v)`).
This is because the rebuild rules query the union-find for every
  eq-sort column simultaneously, so a missing entry for any column
  prevents the rule from firing even when other columns have changed.
Self-loops are added in `add_term_and_view` whenever a constructor
  value is created.
The `uf_function_index` ruleset then copies those rows into
  `UF_<Sort>f`, so representatives also satisfy `(= (UF_<Sort>f v) v)`.
We may want to remove this invariant in the future if we move
  to a different encoding, saving some space and time.


```text
(sort view)
(constructor Add (i64 i64) Math)
(function AddView (i64 i64 Math) Unit :merge old :internal-term-constructor Add)
(constructor to_delete_Add (i64 i64) view)
(constructor to_subsume_Add (i64 i64) view)
```

Each constructor in the original program is expanded to
  a term table (`Add`), a view table (`AddView`), and helpers for deferred deletion/subsumption
  (`to_delete_Add`, `to_subsume_Add`).
The view table is always a function whose output type is `Unit` (without proof tracking)
  or `Proof` (with proof tracking), with `:merge old`.
A view table stores "canonicalized" terms and their e-class representative.
A canonicalized term has representative terms for its children.
The last column of the view table is the representative term for the e-class.
The view tables are kept up to date during rebuilding.

```text
(rule ((AddView c0 c1 new)
       (AddView c0 c1 old)
       (!= old new)
       (= (ordering-max old new) new))
      ((set (UF_Math (ordering-max new old) (ordering-min new old)) ()))
       :ruleset rebuilding :name "congruence_rule")
(rule ((= v9 (AddView c0 c1 c2))
       (= c2_leader (UF_Mathf c2))
       (guard
         (or (bool-!= c2 c2_leader))))
      ((set (AddView c0 c1 c2_leader) ())
       (delete (AddView c0 c1 c2)))
        :ruleset rebuilding :name "rebuild_rule")
```

For each constructor, we add a congruence rule and a rebuild rule.
The congruence rule adds equalities to the union-find table when two constructor applications
  have equal arguments.
The rebuild rule updates view tables so that views
  point to representative terms for child e-classes.
Rebuild rules read representatives from `UF_<Sort>f` (function lookup)
  rather than joining directly on `UF_<Sort>`,
  which avoids expensive UF joins during rebuilding.

```text
(function v2 () Math :no-merge)
(set (v2) (Add 1 2))
(set (AddView 1 2 (v2)) ())
(set (UF_Math (v2) (v2)) ())
```

Above is the desugaring for `(Add 1 2)`.
We add to both view and term tables whenever we evaluate
  a constructor or function application.
The self-loop `(UF_Math (v2) (v2))` initializes the e-class for the new term.
It's straightforward except for global variables.
Since global variables are not allowed after this pass,
  we use functions with no arguments to represent them
  (see globals section below).


```text
(rule ((= v3 (AddView a b v4)))
      ((let v5 (Add a b))
       (set (AddView a b v5) ())
       (set (UF_Math v5 v5) ())
       (let v6 (Add b a))
       (set (AddView b a v6) ())
       (set (UF_Math v6 v6) ())
       (set (UF_Math (ordering-max v5 v6) (ordering-min v5 v6)) ()))
       :name "commutativity")
```

Here we have the instrumented commutativity rule.
The query uses the view table to find the canonical e-node.
The actions add to the term table, add to the view table,
  and add an equality to the union-find table.
We add an equality to the union-find table for the two terms, using the `ordering-max` and 
  `ordering-min` egglog primitives to correctly choose a parent.




```text
(check (= v7 (AddView 1 2 v8))
       (= v9 (AddView 2 1 v10))
       (= v8 v10))
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
View functions support subsumption (via the `:internal-term-constructor` annotation).
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

When proof tracking is enabled, the union-find table's output type is `Proof` instead of `Unit`:

```text
(function UF_Math (Math Math) Proof :merge old :internal-hidden)
```

If term `a` has parent `b`, `(UF_Math a b)` returns a 
  proof of `a = b`.
The path compression and single-parent rules are instrumented to produce
  proofs using symmetry (`Sym`) and transitivity (`Trans`) as needed.


Similarly, the view table's output type is `Proof` instead of `Unit`:

```text
(function AddView (i64 i64 Math) Proof :merge old :internal-term-constructor Add)
```

Recall that view tables store a term
  along with the e-class representative.
For a term `t` with representative `r`,
  the proof (output of the view function) proves that `r = t`.
The direction is important, making
  proof production easier later.
We store the earliest proof (`:merge old`).


```text
(rule (;; query the view function directly for the proof
       (= v9 (AddView a b v8)))
      (;; proof list, one per line of the original query
       (let v10 (PCons v9 (PNil )))
       
       (let v11 (Add a b))
       ;; Proof that Add a b = Add a b
       (let v12 (Rule "commutativity" v10 (AstMath v11) (AstMath v11)))
       ;; Setting the proof for Add a b
       (set (MathProof v11) v12)

       ;; Update the view function (set instead of constructor insertion)
       (set (AddView a b v11) v12)

       (let v13 (Add b a))
       ;; Proof that Add b a = Add b a
       (let v14 (Rule "commutativity" v10 (AstMath v13) (AstMath v13)))
       (set (MathProof v13) v14)
       (set (AddView b a v13) v14)

       ;; Store a proof that (Add a b) = (Add b a).
       (set (UF_Math (ordering-max v11 v13) (ordering-min v11 v13))
            (Rule "commutativity" v10 (AstMath (ordering-max v11 v13)) (AstMath (ordering-min v11 v13)))))
         :name "commutativity")
```

Instrumented rules with proof tracking query the view function directly
  (since the proof is its output column), then construct proofs for each action.
The structure is the same as term mode — view updates use `set`, UF updates use `set` —
  but the values stored are `Proof` terms instead of `()`.
For nested terms, congruence proofs are built to ensure
  the proof terms match the original queries.

# Containers

Container sorts (`Vec`, `Set`, `Map`, `MultiSet`, `Pair`) are never unioned
directly, so they get **no** union-find tables. Instead a container is
recanonicalized structurally when its elements' e-classes change. Take:

```text
(datatype Math (Num i64))
(sort MathVec (Vec Math))
(constructor Wrap (MathVec) Math)
```

The `MathVec` argument of `Wrap` is a container column, so `Wrap`'s rebuild rule
canonicalizes it with a per-container *rebuild primitive* the encoding registers
(here `MathVec_rebuild`), alongside the usual `UF_Mathf` lookup for the
representative column:

```text
(rule ((= v (WrapView c0 c1))
       (= c0_rebuilt (MathVec_rebuild c0))
       (= c1_leader (UF_Mathf c1))
       (guard (or (bool-!= c0 c0_rebuilt) (bool-!= c1 c1_leader))))
      ((set (WrapView c0_rebuilt c1_leader) ())
       (delete (WrapView c0 c1)))
       :ruleset rebuilding :name "rebuild_rule" :naive)
```

The primitive clones the container, remaps each element to its union-find leader,
and re-interns it. Because it reads the elements' `UF_<E>f` indices rather than
joining a tracked table, the rule is marked `:naive`: an element becoming equal
to another produces no delta on the container's own view row, so the rule must
rescan the view each round. Nested containers (e.g. `(Vec (Vec Math))`) rebuild
by recursing through container-typed elements.

**Proofs.** A container's term form is the s-expr of its constructor —
`(vec-of e0 e1 …)`, `(pair a b)`, `(map-of k0 v0 …)` — so the generic `Congr`
machinery applies unchanged. Every container sort gets a reflexive `<Sort>Proof`
table (a `container = container` proof, set at creation); a `Congr` chain over
the changed elements, anchored there, proves `old = new` and folds into the
view's congruence step like an eq-sort child's UF proof.

For reordering/merging containers (`Set`, `Map`, `MultiSet`) the element-wise
`Congr` term can be out of order or hold duplicates, so a `ContainerNormalize`
step (see [`crate::proofs::proof_format`]) canonicalizes it — sort + dedup for
sets, sort for multisets, sort + last-write-wins for maps. It is emitted on every
rebuild and dropped by the proof simplifier wherever it is the identity (always
for `Vec` / `Pair`). Maps use a flat `(map-of k0 v0 …)` form so this works like
the other containers.

See [`crate::proofs::proof_container_rebuild`] for the rebuild primitives.
