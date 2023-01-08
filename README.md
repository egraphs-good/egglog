# egg-smol

[web demo](https://www.mwillsey.com/egg-smol/)

This is the working repo for egg-smol.

## Background

See this presentation for background information:

https://youtu.be/dbgZJyw3hnk?t=3983

with this paper to go along:

https://effect.systems/doc/pldi-2022-egraphs/abstract.pdf

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
