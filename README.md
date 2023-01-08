# egg-smol

[web demo](https://www.mwillsey.com/egg-smol/)

This is the working repo for egg-smol.

## Background

See this presentation for background information:

https://youtu.be/dbgZJyw3hnk?t=3983

with this paper to go along:

https://effect.systems/doc/pldi-2022-egraphs/abstract.pdf

Also see papers about egglog.

## Prerequisites & compilation

```
apt-get install make cargo
make all
```


## Usage

```
egg-smol [-f fact-path] [-naive] <files.egg>
```

or just

```
egg-smol
```

for the REPL.

# Syntax

The syntax of the .egg files is defined in `src/ast/parse.lalrpop`.
## Commands

### `datatype` command

Declare a datatype with this syntax:

```
    ( datatype <name:Ident> <variants:(Variant)*> )
```

where variants are:

```
    ( <name:Ident> <types:(Type)*> <cost:Cost> )
```

Example:
```
(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math)
  (Mul Math Math))
```

defines a simple `Math` datatype with variants for numbers, named variables, addition and multiplication.

Datatypes are also known as algebraic data types, tagged unions and sum types.

### `function` command

```
    ( function <name:Ident> <schema:Schema> <cost:Cost>
        <merge:(:merge <Expr>)?> )
```

Defines a named function with a type schema, an optional integer cost, an optional `:merge` expression, which can refer to `old` and `new` values.

Example:
```
(function add (Math Math) Math)
```

defines a function `add` which adds two `Math` datatypes and gives a `Math` as the result.

Functions are basically lookup tables from inputs to the output. They can be considered as a kind of database table.

Explicit function values can be defined using `set`:

```
(function Fib (i64) i64)
(set (Fib 0) 0)
(set (Fib 1) 1)
```

You can extract the value of specific points of the function using `extract`:

```
(extract (Fib 1))
```

### `relation` command

The `relation` command defines a named function which returns the `Unit` type.

```
    ( relation <name:Ident> <types:List<Type>> )
```

Thus `(relation <name> <args>)` is equivalent to  `(function <name> <args> Unit)`.

Example:
```
(relation path (i64 i64))
(relation edge (i64 i64))
```

defines a `path` and an `edge` relation between two `i64`s.

```
(edge 1 2)
(edge 2 3)
```

inserts two edges into the store for the `edge` function. If your function is relation between the inputs, use `relation` and the above syntax to define the relations, since there is no syntax to define a unit value using `set`.

### `rule` command

```
    ( rule <body:List<Fact>> <head:List<Action>> )
```

defines a rule, which matches a list of facts, and runs a bunch of actions. It is useful to maintain invariants and inductive definitions.

Example:
```
(rule ((edge x y))
      ((path x y)))

(rule ((path x y) (edge y z))
      ((path x z)))
```

These rules maintains path relations for a graph: If there is an edge from `x` to `y`, there is also a path from `x` to `y`. Transitivity is handled by the second rule: If there is a path from `x` to `y` *and* there is an edge from `y` to `z`, there is also a path from `x` and `z`.

### `extract` command

```
    ( extract <variants:(:variants <UNum>)?> <e:Expr> )
```

where variants are:

```
    ( <name:Ident> <types:(Type)*> <cost:Cost> )
```

The `extract` queries the store to find the cheapest values matching the expression.

### `rewrite` and `birewrite` commands

```
    ( rewrite <lhs:Expr> <rhs:Expr>
        <conditions:(:when <List<Fact>>)?>
    )
```

defines a rule which matches the `lhs` expressions, and rewrites them to the `rhs` expression. It is possible to guard the rewrite with a condition that has to be satisfied before the rule applies.

```
    ( birewrite <lhs:Expr> <rhs:Expr>
        <conditions:(:when <List<Fact>>)?>
    )
```

does the same, but where both directions apply.

Example:

```
(rewrite (Add a b)
         (Add b a))
```

declares a rule that a `Add` variant is commutative. 

```
(birewrite (* (* a b) c) (* a (* b c)))
```

declares a rule that multiplication is associative in both directions.

### Other commands

```
    ( sort <name:Ident> ( <head:Ident> <tail:(Expr)*> ) )
    ( define <name:Ident> <expr:Expr> <cost:Cost> )
    ( run <limit:UNum>  <until:(:until <Fact>)?> )
    ( check <Fact> )
    ( clear-rules )
    ( clear )
    ( query <List<Fact>> )
    ( push <UNum?> )
    ( pop <UNum?> )
    ( print <sym:Ident> <n:UNum?> )
    ( print-size <sym:Ident> )
    ( input <name:Ident> <file:String> )
    ( fail <Command> )
    ( include <file:String> )

    ( calc ( <idents:IdentSort*> ) <exprs:Expr+> )
```

where sorts are:

```
    ( <ident:Ident> <sort:Type> )
```

### Actions

```
    ( set ( <f: Ident> <args:Expr*> ) <v:Expr> )
    ( delete ( <f: Ident> <args:Expr*> ) )
    ( union <e1:Expr> <e2:Expr> )
    ( panic <msg:String> )

    ( let <name:Ident> <expr:Expr> )
```

### Name

```
    [ <Ident> ] 
```

### Facts

```
    ( = <mut es:Expr+> <e:Expr> ) 
```

### Expressions

```
    integer
    string
    identifier
    call: ( <head:Ident> <tail:(Expr)*> )
```

## Sorts

### Sort: i64

Signed 64-bit integers supporting these primitives:

```
+ - * / %           ; arithmetic
& | ^ << >> not-i64 ; bit-wise operations
< > <= >=           ; comparisons
min max
```

### Sort: map

A map from a key type to a value type supporting these primitives:

```
empty
insert
get
not-contains
contains
set-union
set-diff
set-intersect
map-remove
```

### Sort: rational

Rational numbers (fractions) with 64-bit precision for numerator and denominator with these primitives:

```
+ - * /         ; arithmetic
min max neg abs floor ceil round
rational        ; construct from a numerator and denominator
pow log sqrt
< > <= >=       ; comparisons
```

These primitives are only defined when the result itself is a pure rational.

### Sort: string

No primitives defined.
