# Egglog semantics document

This (in-progress) document describes the semantics of the Egglog language in a fairly  operational way.
PRs are required to keep this in sync with the implementation.

Things to do:
- Add typechecking
- Add something about lattices and why egglog is well-behaved in some way
- Add rebuilding
- Add seminaiive

## Global State

Egglog maintains a global state $(T, D, C, E, F)$.
- $T$ is a set of terms, constructed using *constructors*. $T$ can be inifinite (but is represented in a finite way in egglog's implementation).
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
After delaring a constructor, it can be used in actions or queries

## Functions

Example:
`(function LowerBound (Math) Bound :merge (Max old new))`
This declares a function `LowerBound` with one argument of sort `Math`
and one result of sort `Bound`.

Functions contain terms built using constructors.
They also enforce a functional dependency between their arguments and result.
The `:merge` keyword specifies the merge function, which is used to enforce this dependency.
When the functional dependency is violated, the merge function resolves this conflict by returning a new term, created using the merge function.

## Queries

Queries are have the following grammar:
```
query ::= (<constraint> ...)

fact ::= (= <expr> <expr>)
       | <expr>
expr ::= <primitive>
       | (<constructor> <expr> ...)
       | (<function> <expr> ...)
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
Each substutition `S` is mapping from variables to terms.
Let `valid(B, S, query)` denode that substitution `S` is valid for database `B` and query `query`.
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
-----------------------------------------------------------
valid(B, S, e) ;; the constraint e is valid





e is a primitive
-----------------------------
binds(B, S, e, e) ;; primitives are bound to themselves

x is a variable, x in S
-----------------------------
binds(B, S, x, S[x]) ;; variables are looked up in the substitution


binds(B, S, e1, t1), ..., binds(B, S, eN, tN) ;; bindings for every child
B = (T, D, C, E, F) ;; B is a database
c in F ;; c is a constructor with N arguments
(c t1' ... tN') in T ;; there's a term in the database that is congruent to (c t1 ... tN)
t1 = t1' in C, ..., tN = tN' in C ;; all children are equal in C
----------------------------------------
binds(B, S, (c e1 ... eN), (c t1 ... tN))


(f, e1, ..., eN) where f is a function
TODO
```

Let `A` be the set of every substitution `S` such that `valid((T, D, C, E, F), S, query)`.
When running `query`, egglog returns a set of substitutions $R \subseteq A$ such that:
- For all $S$ in $A$:
- Exists $S'$ in $R$ such that:
- For all assignments $(v, t) \in S$:
- Exists $(v, t') \in S$ such that $t = t' in C$.

Intuitively, egglog returns all substitutions that are valid for the query
modulo equality in the congruence closure.

## Running Rules

Rules are run using schedules.
A schedule runs rulesets.

When running a ruleset, egglog runs all queries from that ruleset first, before applying all actions the result.


## Action

Actions 






