# Generic user defined types and functions

## Abstract

This is a proposal to add support for allowing users to add generic types and functions.

## Motivation

Currently, only primitive types and functions in `egglog` can be generic, i.e. `(Vec i64)`, but user defined types cannot be.

User defined container types are helpful to implement, compared to the primitives, because they can be used in cases where the value of the expression is not known eagerly. However, implementing them currently requires re-writing their definitions and rewrite rules for every inner type.

For example, much of the Python [array API module](https://github.com/egraphs-good/egglog-python/blob/main/python/egglog/exp/array_api.py) is devoted to repeated implementations of things like `TupleInt`, `TupleNDArray`, and `OptionalInt`, `OptionalBool`, etc.

Allowing user defined generic types and functions would instead allow these to be only implemented once. Furthermore, in the Python library, these generic user defined types could be packages together into a standard module for any user to access, reducing the boilerplate and chance for mistakes.

When [trying to adapt `egglog` for use within PyTensor](https://egglog-python.readthedocs.io/en/latest/explanation/2023_11_17_pytensor.html), the lack of standard user defined container classes was one of the main points of confusion.

## Examples

Defining a generic type would require specifying the generic parameters after the name in square brackets:

```lisp
(datatype Pair [T V] 
  (pair T V))
```

When defining a generic function, the generic type can be used as a constructor with type variables:

```lisp
(function first ((Pair T V) T)
(function second ((Pair T V) V)
```

Rewrites could be defined on generic types as well:

```lisp
(rewrite (first (pair f s)) f)
(rewrite (second (pair f s)) s)
```


This is how you could make a const list:

```lisp
(datatype Tuple [T]
  (tuple-nil)
  (tuple-cons T (Tuple T)))

(function tuple-index ((Tuple T) i64) T)

(rewrite (tuple-index (tuple-cons t rest) 0) t)
(rewrite (tuple-index (tuple-cons t rest) i) (tuple-index rest (- i 1) :when (> i 0)))
```

Functions can be defined based on multiple generic types as well, like `zip`:

```lisp
(function zip ((Tuple (Pair T V)) (Pair (Tuple T) (Tuple V))))

(rewrite (zip (tuple-nil))
         (pair (tuple-nil) (tuple-nil)))
(rewrite (zip (tuple-cons (pair l r) rest))
         (pair (tuple-cons l (first (zip rest)))
               (tuple-cons r (second (zip rest)))))
```

Another common definition would be a nullable type:

```lisp
(datatype Optional [T]
   (optional-none)
   (optional-some T))
```



## Specification

The implementation of this proposal has not been investigated yet.

Generally it would require these changes

1. Change the syntax to allow type parameters in square brackets when declaring datatypes as well as optionally when declaring functions.
2. Change the syntax to allow type parameters to be s-exp’s themselves, to allow parameterized types as arguments.
3. Add generic type information to each function declaration structure.
4. Modify the type checker to use generic function information when inferring expression types. 
5. Change the rule matcher to match generic rules on any expressions which can be substituted safely in those generic rules.

## Open Questions

### Type Inference

One of the current issues with generic builtin sorts is that if the return type of a function is not clear from its arguments, i.e. creating an empty map, the type system does not use type inference to infer what type to use, instead just choosing one, see [#113](https://github.com/egraphs-good/egglog/issues/113).

If the type inference would be an issue for implementation of this proposal, it could be changed to allow more explicit type annotations. 

One option could be to allow using the type names as functions, to explicitly annotate a certain term. For example, then the first zip rewrite would look like:

```lisp
(rewrite (zip ((Tuple (Pair T V)) (tuple-nil)))
         (pair ((Tuple T) (tuple-nil)) ((Tuple V) (tuple-nil))))
```

Another option could be to allow type parameters to be passed explicitly to functions when necessary:

```lisp
(rewrite (zip (tuple-nil [(Pair T V)]))
         (pair (tuple-nil [T]) (tuple-nil [V])))
```

In this case, for functions that were defined in the `datatype` constructor, the parameters would be substituted based on that constructor list. For functions not defined in the `datatype` constructor, they could be replaced just from first instance to last, or we could allow also allowing for explicit specification of type parameters in functions, like this:

```lisp
(function tuple-nil [T] () (Tuple T))
```

Note that the Python bindings require explicit type information to be provided for all variables and so does a bottom up type inference, so has access to all the type information for any way that is needed by `egglog`.

## Syntax

This proposal currently uses the square brackets `[]` to specify the generic arguments for types and functions. This was to reduce ambiguity when parsing, compared to `()` and also corresponds closely to Python’s generic typing.

Any alternate syntax suggestions are welcome. 

## Deferred

### Higher order functions

User defined types would also benefit greatly from the ability to define higher order functions, i.e. a `tuple-map` or `tuple-reduce` function which in turn take generic functions. This proposal leaves that for future work, since there are different possibly ways of implementing higher order functions, and even without them, user defined generic would add a lot of value.
