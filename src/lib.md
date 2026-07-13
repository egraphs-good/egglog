# egglog

Egglog is a language for writing equality saturation applications.
It is the successor to the Rust library [egg](https://github.com/egraphs-good/egg).
Egglog is more expressive than egg and scales better to complex workloads.

# Tutorial
We have a [text tutorial](https://egraphs-good.github.io/egglog-tutorial/01-basics.html) on egglog and how to use it.
We also have a slightly outdated [video tutorial](https://www.youtube.com/watch?v=N2RDQGRBrSY).

# Language reference
The egglog language itself is documented through the [`Command`] enum
(a type alias for [`ast::GenericCommand`]): one variant per top-level
egglog command (`Datatype`, `Function`, `Rule`, `RunSchedule`, …),
each with its own doc comment describing the syntax and semantics.
Whatever you can write between parentheses at the top of an egglog
program shows up as a variant there.

# Using egglog from Rust
We encourage using the egglog language as much as possible, even from Rust.
In some cases, custom primitives or custom rules are necessary.
For this reason, we expose a Rust API in the [`prelude`] module — start
there for the full surface.

The default path: write your sorts, functions, and rules as an egglog
program and run it via [`EGraph::parse_and_run_program`]. Reach for
[`EGraph::update`] only when you really need to do reads and writes
from Rust (e.g. building rows from non-egglog data, integrating with
a Rust-side data structure). For rules whose RHS needs Rust logic,
use  [`prelude::rust_rule`] / [`prelude::rust_rule_full`]. For new
functions callable from egglog expressions, define a custom
[`Primitive`] and register it with the matching
`EGraph::add_*_primitive` — the [`add_primitive!`] macro covers
the common "pure native function" case.

To pull an extracted term back out of the e-graph, let-bind a
global name to it (`(let $root ...)`), resolve the global with
[`EGraph::eval_expr`] to get its `(sort, Value)`, then call
[`EGraph::extract_value`] (default cost model) or
[`EGraph::extract_value_with_cost_model`] / a custom
[`extract::CostModel`] when you want non-default costs. The
[`extract`] module has the full extractor API.

# Writing egglog inline from Rust
The [`expr!`], [`fact!`], [`action!`], [`command!`], [`egglog!`], and [`rule!`]
macros let you write egglog as Rust tokens and get back parsed `Command` /
`Expr` / `Fact` values; splice Rust values in with `#x`, `#..xs`, and
`:#field`, and build un-parsed fragments with [`sexp!`]. See [`egglog!`] for
the syntax (all re-exported from [`prelude`]).

```
use egglog::prelude::*;
let mut egraph = EGraph::default();
egraph.run_program(egglog!(
    (datatype Math (Num i64) (Add Math Math))
    (rule ((= e (Add (Num a) (Num b)))) ((union e (Num (+ a b)))))
    (let start (Add (Num 1) (Num 2)))
    (run 1)
    (check (= start (Num 3)))
)?)?;
let n = 2;
assert_eq!(expr!((Num #n))?.to_string(), "(Num 2)");
# Ok::<(), egglog::Error>(())
```
