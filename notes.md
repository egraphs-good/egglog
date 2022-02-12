# example rewrites

egg: (+ a (+ b c)) => (+ (+ a b) c)
relational: root@(+ a bc), bc'@(+ b c), eq(bc, bc') => ...
datalog: path(x, z) :- path(x, y), path(y, z)
mixed: (div x x) = y, nz(x) => y = 1
mixed elided: (div x x), nz(x) => y      // not sure if this should be allowed
y = (sqrt x) => (nz y) = true

current design:
- all functions have an output value, no relations
- no privileged `true`
- contexts? stored on UF edges only
- contextual facts are true based on their UF edge to some `true` value
- all joins are modulo equality (and contexts), very clean

alternative:
- all facts (including term membership) have contexts
- true is implied by top context (a little cleaner for basic facts)
- how does eqrel work?

(var = pattern)*


simple math
```
(declare-fun + (Id Id) Id)
(declare-fun - (Id Id) Id)
(declare-rule add-assoc // (a b c)
  ((+ a (+ b c)))
  ((+ (+ a b) c)))
  ==>
  ab := (+ a b)

rule
  y = (sqrt x),
  ==>
  true = (nz y),
```

```
(declare-fun + (Id Id) Id)
(declare-consts a b c d)
(declare-rule add-assoc (+ x (+ y z)) => (+ (+ x y) z))
```



