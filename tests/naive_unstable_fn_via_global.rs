//! `:naive` covers the case static analysis can't: an `unstable-fn`
//! value reaches the rule **body** indirectly through a `let`-bound
//! global. Static peek-through can't tell what function the rule's
//! `unstable-app` will dispatch — we only see `$f` as a variable.
//!
//! The wrapped function here is a custom-function table lookup, which
//! `FunctionContainer::apply` returns `None` for in `Pure` (because
//! reads aren't tracked by seminaive). So:
//!
//! 1. **Without `:naive`** — the rule typechecks (the typechecker
//!    can't see that `$f` wraps a custom-function read), but at
//!    runtime the body's `unstable-app` returns `None` in the `Pure`
//!    query context, so the body match never succeeds and the rule
//!    silently doesn't fire.
//!
//! 2. **With `:naive`** — the body typechecks under `Read`,
//!    `unstable-app`'s wrapped `dbl` lookup dispatches in `Read`, the
//!    rule fires, and `out 0` ends up at 14.

use egglog::EGraph;

const PROGRAM_PREFIX: &str = "
(function dbl (i64) i64 :no-merge)
(set (dbl 7) 14)

(sort I2I (UnstableFn (i64) i64))
(let $f (unstable-fn \"dbl\"))

(function out (i64) i64 :no-merge)
";

#[test]
fn naive_rule_with_unstable_fn_via_global_works() {
    let mut egraph = EGraph::default();
    let program = format!(
        "{PROGRAM_PREFIX}
;; `unstable-app` in the body — its custom-function lookup needs
;; `Read` capability, only available with `:naive`.
(rule ((= y (unstable-app $f 7))) ((set (out 0) y)) :naive)
(run 1)
(check (= (out 0) 14))
"
    );
    egraph.parse_and_run_program(None, &program).unwrap();
}

#[test]
fn rule_without_naive_using_unstable_fn_via_global_silently_misses() {
    let mut egraph = EGraph::default();
    let program = format!(
        "{PROGRAM_PREFIX}
;; No `:naive`. The body typechecks (the typechecker can't peek
;; through the global to see the wrapped custom-function), but at
;; runtime `unstable-app` returns `None` in `Pure`, so the body
;; match never succeeds and `out 0` is never set.
(rule ((= y (unstable-app $f 7))) ((set (out 0) y)))
(run 1)
(check (= (out 0) 14))
"
    );
    let result = egraph.parse_and_run_program(None, &program);
    assert!(
        result.is_err(),
        "expected the rule without :naive to silently miss and the \
         `(check (= (out 0) 14))` to fail, but the program ran cleanly"
    );
}
