# Motivation

We'd like to be able to use something similar to `unstable-fn`, but without deep support in core.
We'd also like to experiment with know what kinds of programming can be encoded using egglog.

# Constraints

- Primitives can't have type information
- Support for the REPL (not a whole-program transformation)

# Known limitations

- Primitive outputs have to be wrapped in a constructor, which can be inconvenient

# Example

This document proposes an encoding of partially applied functions in egglog.
Given an input program using the current `unstable-fn`:
```
(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math)
  (Mul Math Math))

(rewrite (Mul (Num x) (Num y)) (Num (* x y)))

(datatype MathList
  (Nil)
  (Cons Math MathList))

(sort MathFn (UnstableFn (Math) Math))


(constructor square (Math) Math)
(rewrite (square x) (Mul x x))

(let square-fn (unstable-fn "square" ))

;; test that we can call a function
(let squared-3 (unstable-app square-fn (Num 3)))
(check (= squared-3 (square (Num 3))))
```


It prodices a new program:
```
;; header- always generated
(sort Any)
(constructor PartialApp (Any Any) Any)
(ruleset partialapp)



(constructor Num (i64) Any)
(constructor Var (String) Any)
(constructor Mul (Any Any) Any)

(rewrite (Mul (Num x) (Num y)) (Num (* x y)))

(constructor Nil () Any)
(constructor Cons (Any Any) Any)

(constructor square (Math) Math)
(rewrite (square x) (Mul x x))

;; generated from unstable fn declaration
(constructor SquareOp () Any)
(rewrite (PartialApp (SquareOp) (AppCons v1 (AppNil)))
         (Square v1) :ruleset partialapp)

(let squared-3 (PartialApp SquareOp (AppCons (Num 3) (AppNil))))
;; always saturate partialapp before any database checks
(run-schedule (saturate partialapp))
(check (= squared-3 (square (Num 3))))
```

It generates new code, one command at a time. For each command, it:
- Typechecks the new command with respect to an in-progress typechecker. This can be done with the current typechecker.
- Desugars datatype to constructors
- Erases all the types, gets rid of all sort delarations
- For every action
  - Desugar unstable-app to a PartialApp call
  - Desugar unstable-fn to a new constructor and PartialApp rule
- For every schedule
  - Instrument the schedule to saturate the partialapp ruleset before every other ruleset runs


# FAQ

## Does this support multiset map?

Yes, with a new primitive implementation of multiset map.
The new implementation would simply apply `PartialApp` on each of the elements.

## How does this work with multiple partial applications?

Here's an example generated rule for a function with two arguments:
```
(let mul-fn (unstable-fn "Mul"))
```
becomes:
```
(constructor MulOp () Any)
(rewrite (PartialApp (PartialApp (MulOp) v_1) v_2) (Mul v_1 v_2)) :ruleset partialapp)
```

The partial applications are natually nested, and we can match on them to finally apply Mul.