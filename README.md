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

## Syntax

The syntax of the .egg files is defined in `src/ast/parse.lalrpop`.

### Commands

```
    ( datatype <name:Ident> <variants:(Variant)*> )
    ( extract <variants:(:variants <UNum>)?> <e:Expr> )
```

where variants are:

```
    ( <name:Ident> <types:(Type)*> <cost:Cost> )
```

```
    ( sort <name:Ident> ( <head:Ident> <tail:(Expr)*> ) )
    ( function <name:Ident> <schema:Schema> <cost:Cost>
        <merge:(:merge <Expr>)?> <default:(:default <Expr>)?> )
    ( relation <name:Ident> <types:List<Type>> )
    ( rule <body:List<Fact>> <head:List<Action>> )
    ( rewrite <lhs:Expr> <rhs:Expr>
        <conditions:(:when <List<Fact>>)?>
    )
    ( birewrite <lhs:Expr> <rhs:Expr>
        <conditions:(:when <List<Fact>>)?>
    )
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
