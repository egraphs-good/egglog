Rewrites an egglog program to use an encoding for equality tracking, optionally including proof tracking.

# Term Encoding

The term encoding *removes all calls to union*, making equality reasoning explicit so it can be instrumented with proof tracking. It adds a per-sort union-find, maintained by rules that run during scheduled maintenance, and stores it in two tables per sort: a constructor UF table (`UF_<Sort>`) holding raw parent edges, and a function-index table (`UF_<Sort>f`) mapping each term to its parent (for fast rebuild lookups). Every constructor becomes two tables too: a term table holding the actual terms, and a view table holding canonicalized terms with their e-class leader. Proof tracking (below) is done in the same pass. The encoding keeps the operational semantics equivalent to the standard one (for the supported subset of commands).

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

Lowering it with the term encoding expands to a lot of new egglog, shown (mostly) in pieces.

```text
(ruleset parent)
(ruleset single_parent)
(ruleset uf_function_index)
(ruleset rebuilding)
(ruleset rebuilding_cleanup)
(ruleset delete_subsume_ruleset)
```

*The new rulesets* drive per-sort union-find (`parent`, `single_parent`), the fast
function index over UF (`uf_function_index`), rebuild-time congruence (`rebuilding` +
`rebuilding_cleanup`), and deferred deletions/subsumptions (`delete_subsume_ruleset`).

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

*Between* the original program's commands, the encoding runs these rulesets to
maintain egglog's invariants.

```text
(sort Math)
(function UF_Math (Math Math) Unit :merge old :internal-hidden)
(function UF_Mathf (Math) Math :merge new)
```

*The union-find tables* store each sort's equivalence classes. `UF_<Sort>` is the
source of truth for UF maintenance; `UF_<Sort>f` is the function-backed index used by
rebuild rules. `UF_<Sort>`'s output is `Unit` (without proofs) or `Proof` (with proofs);
`:merge old` keeps only the first value.

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

*Union-find rules* keep the UF up to date as equalities are added, and the indexing
ruleset mirrors its rows into the function UF. The `ordering-max`/`ordering-min`
primitives define an arbitrary (insertion-order) ordering on terms, used to pick a
parent deterministically.

**Invariant:** every representative term needs a self-loop in the constructor UF (e.g.
`(UF_Math v v)`), because the rebuild rules query the UF for every eq-sort column at
once — a missing entry blocks the rule even when other columns changed. `add_term_and_view`
adds these self-loops when a constructor value is created, and `uf_function_index` copies
them into `UF_<Sort>f`, so representatives also satisfy `(= (UF_<Sort>f v) v)`.


```text
(sort view)
(constructor Add (i64 i64) Math)
(function AddView (i64 i64 Math) Unit :merge old :internal-term-constructor Add)
(constructor to_delete_Add (i64 i64) view)
(constructor to_subsume_Add (i64 i64) view)
```

Each constructor expands to a term table (`Add`), a view table (`AddView`), and deferred
delete/subsume helpers (`to_delete_Add`, `to_subsume_Add`). The view table (`Unit`, or
`Proof` with proofs; `:merge old`) maps a canonicalized term — one whose children are
representatives — to its e-class representative (the last column), and is kept current
during rebuilding.

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

Each constructor gets a congruence rule (adds an equality when two applications have
equal arguments) and a rebuild rule (repoints views at child representatives). Rebuild
rules read representatives from `UF_<Sort>f` rather than joining on `UF_<Sort>`, avoiding
expensive UF joins.

```text
(function v2 () Math :no-merge)
(set (v2) (Add 1 2))
(set (AddView 1 2 (v2)) ())
(set (UF_Math (v2) (v2)) ())
```

The desugaring for `(Add 1 2)`: every constructor/function application is added to both
the view and term tables, and the self-loop `(UF_Math (v2) (v2))` initializes the new
term's e-class. Global variables are represented as no-arg functions (see Globals below).


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

The instrumented commutativity rule: the query uses the view table to find the canonical
e-node, and the actions add to the term and view tables and add an equality to the UF
(again using `ordering-max`/`ordering-min` to pick the parent).




```text
(check (= v7 (AddView 1 2 v8))
       (= v9 (AddView 2 1 v10))
       (= v8 v10))
```

All queries, including `check`, use the view tables. This one checks that `(Add 1 2)` and
`(Add 2 1)` share an e-class representative.

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

Deletions and subsumptions are deferred via per-constructor `to_delete_<C>` /
`to_subsume_<C>` tables, processed during rebuilding. We only delete/subsume from the
view tables (the term tables aren't queried), which keeps terms around for proof tracking
even after they leave the e-graph. Views support subsumption via `:internal-term-constructor`.


# Globals

Before term encoding, the `proof_global_remover.rs` pass desugars global variables to
constructors, so the encoding and backend need not handle globals. The example above has
none, so it is unchanged. A program with a global:
```text
(sort Math)
(constructor Add (i64 i64) Math)
(let g1 (Add 1 2))
(rule ((= g1 (Add 2 3))
      ((Add 3 4))))
```

desugars to this before term encoding:
```text
(sort Math)
(constructor Add (i64 i64) Math)
(constructor g1 () Math)
(union (g1) (Add 1 2))
(rule ((= (g1) (Add 2 3)))
      ((Add 3 4)))
```



# Proof Tracking

With proof tracking enabled, the same pass also instruments the program to track proofs
of equalities. Continuing the example above:

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


A proof header is added first, defining the proof format that corresponds to
[`RawProof`](crate::proofs::RawProof) (see `proof_encoding_helpers.rs`).

```text
(function MathProof (Math) Proof :merge old)
```

Every sort gets a proof table mapping a term `t` to a proof of `t = t` (oldest kept).

The union-find table's output becomes `Proof`:

```text
(function UF_Math (Math Math) Proof :merge old :internal-hidden)
```

If `a` has parent `b`, `(UF_Math a b)` is a proof of `a = b`. The path-compression and
single-parent rules are instrumented to compose these with `Sym`/`Trans` as needed.

The view table's output likewise becomes `Proof`:

```text
(function AddView (i64 i64 Math) Proof :merge old :internal-term-constructor Add)
```

For a term `t` with representative `r`, the view's proof proves `r = t` (this direction
eases later proof production); `:merge old` keeps the earliest.


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

Instrumented rules query the view function (whose output is the proof) and build a proof
for each action. The structure matches term mode — views and UF still use `set` — but the
stored values are `Proof` terms instead of `()`. Nested terms get congruence proofs so the
proof terms match the original queries.

# Containers

Container sorts (`Vec`, `Set`, `Map`, `MultiSet`, `Pair`) participate too. They are never
unioned directly, so they get no union-find tables; instead their elements are
canonicalized structurally during rebuilding. An eq-container column is rebuilt by a
per-container *rebuild primitive* (registered by the encoding) called inside the rebuild
rule; because that primitive reads the element sorts' `UF_<E>f` indices, the rule is
`:naive`. The primitive clones the container, remaps each eq-sort element to its
union-find leader (via `ContainerValues::rebuild_val_with`), and re-interns it.

In proof mode, a container's "term form" is the s-expr headed by its constructor —
`(vec-of e0 e1 ...)`, `(pair a b)`, etc. — built by the constructor's validator, so the
generic `Congr` machinery applies unchanged. Every container sort gets a `<Sort>Proof`
table holding a reflexive `container = container` proof (set on creation), used as the
anchor for a `Congr` chain over the changed elements proving `old = new`. That proof folds
into the view's congruence step like an eq-sort child's union-find proof. Nested containers
(e.g. `(Vec (Vec Math))`) recurse, each rebuilt container recording its own reflexive
anchor for later rebuilds.

The flat `Congr` chain replaces children in place, which for a reordering/merging container
(`Set`, `Map`, `MultiSet`) may leave them out of order or duplicated. The **container
normalization** (`ContainerNormalize` in [`crate::proofs::proof_format`]) bridges that gap:
sort + dedup for sets, sort for multisets, sort + last-write-wins for maps. Each rebuild
mints it unconditionally; the proof simplifier drops it where it is the identity (always
for order-preserving `Vec`/`Pair`, and for already-canonical sets/maps). A container's
normalization lives in its constructor validator (so a user-defined container normalizes
like the built-ins); the checker recomputes it by applying the validator for the term's
head, and `reconstruct_termdag` produces the same canonical form.

Maps use a flat `(map-of k0 v0 k1 v1 ...)` term form (like `set-of`/`vec-of`) rather than
nested `map-insert`s, so rebuild `Congr` indices are flat and key collapse works like the
other containers.

The rebuild primitives are registered programmatically with fresh names, so a re-parse of
the desugared program could not resolve them. The encoder therefore records their names on
the container's `(sort …)` as an `:internal-container-rebuild` annotation (everything else
they need is recovered from `proof_state`); when that Sort is typechecked — during encoding
*and* on re-parse — the primitives are re-registered. See
[`crate::proofs::proof_container_rebuild`].
