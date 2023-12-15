# Egglog semantics document

This (in-progress) document describes the semantics of the Egglog language.
PRs are required to keep this in sync with the implementation.

TODO:
- Add typechecking
- Running rules, schedules
- Merge functions, and on_merge
- Give semantics to rebuilding
- Add seminaive
- Handle globals
- Add something about lattices and why egglog is well-behaved when using them
- extraction meaning and guarantees
- set-option variants and their meaning

## Global State

Egglog maintains a global state $(T, D, C, E, F)$.
- $T$ is a set of terms, constructed using *constructors*. $T$ can be infinite (but is represented in a finite way in egglog's implementation).
- $D$ is a set of tuples containing terms, constructed by `set`-ing *functions*.
- $C$ is a set of equalities between terms. It is also a congruence closure, maintained by egglog's core.
- $E$ is an environment: a mapping from global variables to terms.
- $F$ is a set of functions and constructors.

## Terms

Terms are primitives or the result of a constructor on other terms.
A term has the following grammar:
```
term ::= <primitive>
        | (<constructor> <term1> ...)
```

Egglog ensures these terms are well-typed.

## Constructors

Example:
`(constructor Add (Math Math) Math)`
This declares a constructor `Add` with two arguments of sort `Math`.
After declaring a constructor, it can be used in actions or queries

## Functions

Example:
`(function LowerBound (Math) Bound :merge (Max old new))`
This declares a function `LowerBound` with one argument of sort `vMath`
and one result of sort `Bound`.

Functions contain terms built using constructors.
Egglog enforces a functional dependency between their arguments and result.
The `:merge` keyword specifies the merge function, which is used to enforce this dependency.
When the functional dependency is violated, the merge function resolves this conflict by returning a new term, created using the merge function.

## Queries

Queries have the following grammar:
```
query ::= (<constraint> ...)

fact ::= (= <qexpr> <qexpr>)
       | <qexpr>
qexpr ::= <primitive>
       | (<constructor> <qexpr> ...)
       | (<function> <qexpr> ...)
       | <variable>
```

Queries are present in `rule`, `check`, and `query-extract`.

## Rules

Rules have the following syntax:
```
(rule <name>
  <query>
  <action>)
```

## Executing Queries

When executing a query, egglog returns a set of substitutions.
Each substitution `S` is mapping from variables to terms.
Let `valid(B, S, query)` denote that substitution `S` is valid for database `B` and query `query`.
We define validity below.
Validity ensures that each constraint enforced by the query is met.
We use a helper `binds(B, S, e, t)` to denote that `e` is bound to `t` with substitution `S` in database `B`.

Example: expression `(Add a b)` with substitution `{a: 1, b: 2}` is bound to term `(Add 1 2)`.


```
;; If all constraints are valid
valid(B, S, constraint1), ..., valid(B, S, constraintN) 
------------------------------------------------------ 
;; The whole query is valid
valid(B, S, (constraint1, ..., constraintN))

;; if e1 and e2 are bound to the same term
binds(B, S, e1, t1), binds(B, S, e2, t2)
B = (T, D, C, E) ;; B is a database
t1 = t2 in C
------------------------------------------------------------------
;; the constraint (= e1 e2) is valid
valid(B, S, (= e1 e2))

binds(B, S, e, t)
----------------------------------------------
valid(B, S, e) ;; the constraint e is valid


e is a primitive
-----------------------------
binds(B, S, e, e) ;; primitives are bound to themselves

x is a variable, x in S
-----------------------------
binds(B, S, x, S[x]) ;; variables are looked up in the substitution


B = (T, D, C, E, F) ;; B is a database
c in F ;; c is a constructor with N arguments
binds(B, S, e1, t1), ..., binds(B, S, eN, tN) ;; bindings for every child
(c t1 ... tN) in T ;; the term is in the database
----------------------------------------
binds(B, S, (c e1 ... eN), (c t1 ... tN))


B = (T, D, C, E, F) ;; B is a database
f in F ;; f is a function with N arguments
binds(B, S, e1, t1), ..., binds(B, S, eN, tN) ;; bindings for every child
(f t1 ... tN o) in D ;; tuple with output o in the database
----------------------------------------
binds(B, S, (f e1 ... eN), o)
```

Let `A` be the set of every substitution `S` such that `valid((T, D, C, E, F), S, query)`.
When running `query`, egglog returns a set of substitutions $R \subseteq A$ such that:

- $\forall S \in A, \exists S' \in R$ such that:
- $\forall (v, t) \in S, \exists (v, t') \in S', (t = t') \in C$

Intuitively, egglog returns all substitutions that are valid for the query
modulo equality in the congruence closure.
Since users of egglog can't distinguish terms
that are equal in the congruence closure,
$S'$ represents $S$ soundly.

## Actions

Actions have the following grammar:
```
<action> ::= (<stmt> ...)

<stmt> ::= <aexpr>
         | (let <symbol> <aexpr>)
         | (union <aexpr> <aexpr>)
         | (set (<function> <aexpr> ...) <aexpr>)

;; importantly, aexprs do not contain
;; functions (while qexprs do)
aexpr ::= <primitive>
       | (<constructor> <aexpr> ...)
       | <variable>
```


## Executing Actions

Given a substitution `S`, an action evaluates to $(T', D')$, a set of new terms
and a set of new tuples to add to the database.

terms and tuples to the database.
Let `evaluated(B, S, A)` denote that action `A`
evaluates with substitution `S` to a new database `B`.
We also define a helper `evaluated_expr(B, S, stmt)` to denote that statement `stmt` evaluates with substitution `S` to database `B`.


```
----------------------
evaluated(B, S, ())







```


## Invariant Maintenance

After running an action, egglog needs to add new terms and tuples $(T', D')$
to the database $B$.
The resulting database must contain $T'$ and $D'$, but also maintain the
following invariants:

- $C$ is a congruence closure: it satisfies reflexivity, symmetry, transitivity, and congruence
- $T$ is closed under $C$. This means that if a term is in $T$, all variants that are equal to it are also in $T$.
$$\forall (c t_1 ... t_n) \in T, t_1 = t_1' \in C, ..., t_n = t_n' \in C \implies (c t_1' ... t_n') \in T$$

Egglog "completes" or "closes" the database in order to maintain these 
invariants.
However, it is only allowed to add terms and tuples that are directly
implied by existing ones using the axioms of the congruence closure.
TODO make this formal.

In the implementation of egglog, storing a representative $t \in T$ for
each equivalence class is sufficient to represent $T$ and
efficiently perform queries.
However, the implementation must also maintain the invariant that there is one
canonical representative per equivalence class.
We consider this an implemtation detail.
