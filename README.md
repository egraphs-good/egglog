# egg-smol

[web demo](https://www.mwillsey.com/egg-smol/)

This is the working repo for egg-smol.

## Background

See this presentation for background information:

https://youtu.be/dbgZJyw3hnk?t=3983

with this paper to go along:

https://effect.systems/doc/pldi-2022-egraphs/abstract.pdf

Also see papers about egglog.

See also the Python binding, which provides a bit more documentation:
https://egg-smol-python.readthedocs.io/en/latest/

## Chat

There is a Zulip chat about egg-smol here:
https://egraphs.zulipchat.com/#narrow/stream/328979-Implementation/topic/Eggsmol

## Prerequisites & compilation

```
apt-get install make cargo
make all
```


## Usage

```
cargo run [-f fact-path] [-naive] <files.egg>
```

or just

```
cargo run
```

for the REPL.

## VS Code plugin

There is a VS Code extension in the vscode folder. Install using 'Install from VSIX...' in the three-dot menu of the extensions tab and pick npm install -g @vscode/vscode/eggsmol.vsix`.

### Enhancing the VS code extension

If you want to hack on the VS Code extension, install nodejs, and make your changes in the files in the `vscode/eggsmol-1.0.0` folder.

Then run

```
code vscode/eggsmol-1.0.0
```

and use F5 to run the extension in a new window. When satisfied, then install VSCE if you do not already have it:

```
npm install -g @vscode/vsce
```

Run `vsce package` in the `vscode/eggsmol-1.0.0` folder to reconstruct the .vsix file and install it manually.

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
        (:on_merge <List<Action>>)?
        (:merge <Expr>)?
        (:default <Expr>)?
```

Defines a named function with a type schema, an optional integer cost, and an optional `:on_merge` or `:merge` expression, which can refer to `old` and `new` values. You can also provide a default value using `:default`.

Example:
```
(function add (Math Math) Math)
```

defines a function `add` which adds two `Math` datatypes and gives a `Math` as the result.

Functions are basically lookup tables from input tuples to the output. They can be considered as a kind of database table.

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

If you define a `:merge` expression, you can update specific values in the function, and the function relation will be updated using the merge expression:

```
(function KeepMax (i64) i64 :merge (max new old)); when updated, keep the biggest value
(set (KeepMax 0) 0)
(set (KeepMax 1) 1)
(set (KeepMax 1) 2)   ; we redefine 1 to be 2
(set (KeepMax 1) 0)   ; this does not change since we use max
(extract (KeepMax 1)) ; this is 2  
```

### `relation` command

The `relation` command defines a named function which returns the `Unit` type.

```
    ( relation <name:Ident> <types:List<Type>> )
```

Thus `(relation <name> <args>)` is equivalent to `(function <name> <args> Unit)`.

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

### `define` command

```
    ( define <name:Ident> <expr:Expr> <cost:Cost> )
```

defines a named value. This is the same as a 0-arity function with a given, singular value.

Example:
```
(define one 1)
(define two 2)
(define three (+ one two))
(extract three); extracts 3 as a i64
```

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

### `check` and `fail` commands

```
    ( check <Fact> )
    ( fail <Command> )
```

This evaluates the fact and checks that it is true.

Example:

```
(check (= (+ 1 2) 3))
(check (<= 0 3) (>= 3 0))
(fail (check (= 1 2)))
```

prints

```
[INFO ] Checked.
[INFO ] Checked.
[ERROR] Check failed
[INFO ] Command failed as expected.
```

### Other commands

```
    ( sort <name:Ident> ( <head:Ident> <tail:(Expr)*> ) )
    ( run <limit:UNum>  <until:(:until <Fact>)?> )  ; evaluate rules N steps or until a condition is met
    ( clear-rules )                         ; clear out all rules and rewrites
    ( clear )                               ; clear data from the functions, not the function tables themselves
    ( query <List<Fact>> )
    ( push <UNum?> )                        ; saves the state of the database on the stack
    ( pop <UNum?> )                         ; restores the state of the database on the stack
    ( print <sym:Ident> <n:UNum?> )         ; print the value of an id
    ( print-size <sym:Ident> )
    ( input <name:Ident> <file:String> )
    ( output <file:String> <exprs:Expr+> )  ; Saves the expression to a file
    ( include <file:String> )
    ( add-ruleset <id:String> )             ; Saves all rules as a ruleset with a given name (EXPERIMENTAL)
    ( load-ruleset <id:String> )            ; Add the rules from a ruleset previously added (EXPERIMENTAL)
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

### Union

The underlying data structure maintained by egg-smol is an e-graph. That means that specific values can be unified to be equivalent. To extract a value, use `extract` and it will extract the cheapest option according to the costs.

```
(datatype Math (Num i64))
(union (Num 1) (Num 2)); Define that Num 1 and Num 2 are equivalent
(extract (Num 1)); Extracts Num 1
(extract (Num 2)); Extracts Num 1
```

Union only works on variants, not sorts.

### Name

```
    [ <Ident> ] 
```

### Facts

```
    ( = <mut es:Expr+> <e:Expr> )
    <Expr>
```

These are conditions used in check and other commands. There is no boolean type in egg-smol. Instead, boolean are modelled morally as `Option<Unit>`, so if something is true, it is `Some<()>`. If something is false, it does not match and is `None`.

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

### Sort: f64

64-bit floating point numbers supporting these primitives:

```
+ - * / %           ; arithmetic
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

Use double quotes to get a quote: "Foo "" Bar" is `Foo " Bar`.
No primitives defined.
