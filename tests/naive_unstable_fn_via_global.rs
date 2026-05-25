//! `:naive` covers the case static analysis can't: an `unstable-fn`
//! value reaches the rule body or action indirectly through a
//! `let`-bound global. Static peek-through can't tell what function
//! the rule's `unstable-app` will dispatch — we only see `$f` as a
//! variable.
//!
//! The wrapped function here is a custom-function table lookup,
//! which `FunctionContainer::apply` refuses in `Pure` (rule body
//! without `:naive`) and `Write` (rule action without `:naive`)
//! because seminaive can't track those reads. The mismatch triggers
//! the egglog runtime panic that was pre-registered at the
//! `unstable-fn` build site, so `run_rules` returns an `Err` rather
//! than silently dropping the match. With `:naive` the body widens
//! to `Read` / action widens to `Full` and the rule works.

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
fn rule_body_without_naive_using_unstable_fn_via_global_hard_errors() {
    let mut egraph = EGraph::default();
    let program = format!(
        "{PROGRAM_PREFIX}
;; No `:naive`. The body typechecks (the typechecker can't peek
;; through the global to see the wrapped custom-function), but at
;; runtime the capability mismatch triggers the pre-registered
;; egglog panic and `run_rules` returns `Err`.
(rule ((= y (unstable-app $f 7))) ((set (out 0) y)))
(run 1)
"
    );
    let err = egraph
        .parse_and_run_program(None, &program)
        .expect_err("expected a hard error for capability mismatch");
    let msg = format!("{err}");
    assert!(
        msg.contains("unstable-fn") && msg.contains("dbl") && msg.contains(":naive"),
        "expected error to name the wrapped fn `dbl` and mention `:naive`; got: {msg}"
    );
}

/// `unstable-app` over a custom-function lookup in a rule **action**
/// (not body) is also unsound under seminaive: the action runs at
/// `Context::Write`, where reads of live state aren't tracked, so the
/// rule won't re-fire when the read row's contents change.
/// `FunctionContainer::apply` refuses, the pre-registered panic
/// triggers, and `run_rules` returns `Err`. `:naive` widens the
/// action to `Full` and the rule works.
#[test]
fn rule_action_using_unstable_fn_custom_function_requires_naive() {
    // Without :naive — action ctx is Write, custom-function lookup
    // refuses, hard error.
    let mut egraph1 = EGraph::default();
    let bad = format!(
        "{PROGRAM_PREFIX}
(function trigger () i64 :no-merge)
(set (trigger) 1)
(rule ((= _ (trigger))) ((set (out 0) (unstable-app $f 7))))
(run 1)
"
    );
    let err = egraph1
        .parse_and_run_program(None, &bad)
        .expect_err("expected a hard error for capability mismatch");
    let msg = format!("{err}");
    assert!(
        msg.contains("unstable-fn") && msg.contains("dbl") && msg.contains(":naive"),
        "expected error to name the wrapped fn `dbl` and mention `:naive`; got: {msg}"
    );

    // With :naive — action ctx widens to Full, dispatch succeeds.
    let mut egraph2 = EGraph::default();
    let good = format!(
        "{PROGRAM_PREFIX}
(function trigger () i64 :no-merge)
(set (trigger) 1)
(rule ((= _ (trigger))) ((set (out 0) (unstable-app $f 7))) :naive)
(run 1)
(check (= (out 0) 14))
"
    );
    egraph2.parse_and_run_program(None, &good).unwrap();
}

/// Global `EGraph::seminaive = false` (the `--naive` CLI flag) must
/// widen rule contexts to `Read`/`Full` for typechecking just like
/// rule-local `:naive` does — otherwise a primitive registered as
/// `FullPrim` (e.g. `unstable-multiset-fill-index`) typechecks under
/// `:naive` but errors under `--naive` even though the backend
/// lowering already runs the rule naively.
#[test]
fn global_naive_widens_rule_context_same_as_local_naive() {
    let program = "
(datatype Math (Num i64))
(sort Maths (MultiSet Math))
(let $xs (multiset-of (Num 1) (Num 2)))

(constructor product (Maths) Math)
(let $zz (product $xs))

(function ms-count (Maths Math) i64 :merge (+ old new))
(sort MSIndexFn (UnstableFn (Maths Math) i64))

;; No rule-local `:naive`; relies on the global setting.
(rule
    ((= outer (product inner)))
    ((unstable-multiset-fill-index inner (unstable-fn \"ms-count\"))))

(run 1)
(check (= 1 (ms-count $xs (Num 1))))
";

    let mut egraph = EGraph::default();
    egraph.seminaive = false;
    egraph.parse_and_run_program(None, program).unwrap();
}
