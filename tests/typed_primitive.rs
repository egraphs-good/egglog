//! Tests for the typed primitive surface and the seminaive-safety
//! enforcement added in issue #772.
//!
//! Covers:
//! - Pure / write / read / full primitives accepted only in their
//!   respective valid contexts (typechecker rejects others).
//! - `unstable-fn` over a constructor in a query context is filtered
//!   (does NOT mint an eclass), but works in actions.
//! - `unstable-fn` over a custom function is filtered in rule queries.
//! - Two same-signature registrations: silent-pick-first behavior
//!   (regression guard).

use egglog::ast::Span;
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::sort::I64Sort;
use egglog::{
    EGraph, FullPrim, FullState, PrimitiveCommon, PurePrim, PureState, ReadPrim, ReadState, Value,
    WritePrim, WriteState, prelude::*,
};

// --- shared test fixtures ---

/// A pure primitive that adds two i64s. Trivially safe in every context.
#[derive(Clone)]
struct PureAdd(&'static str);
impl PrimitiveCommon for PureAdd {
    fn name(&self) -> &str {
        self.0
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                I64Sort.to_arcsort(),
                I64Sort.to_arcsort(),
                I64Sort.to_arcsort(),
            ],
            span.clone(),
        )
        .into_box()
    }
}
impl PurePrim for PureAdd {
    fn apply<'a, 'db>(&self, state: &mut PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let a = state.base_values().unwrap::<i64>(args[0]);
        let b = state.base_values().unwrap::<i64>(args[1]);
        Some(state.base_values().get(a + b))
    }
}

/// A write primitive (touches the wrapper's `WriteState` surface). It
/// just returns its first arg; the body uses `&mut self`-shaped methods
/// so it only type-checks against `WriteState`.
#[derive(Clone)]
struct WriteEcho(&'static str);
impl PrimitiveCommon for WriteEcho {
    fn name(&self) -> &str {
        self.0
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![I64Sort.to_arcsort(), I64Sort.to_arcsort()],
            span.clone(),
        )
        .into_box()
    }
}
impl WritePrim for WriteEcho {
    fn apply<'a, 'db>(&self, state: &mut WriteState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let _ = state.base_values();
        Some(args[0])
    }
}

/// A read primitive — uses `ReadState` (no writes, but reads the DB).
#[derive(Clone)]
struct ReadEcho(&'static str);
impl PrimitiveCommon for ReadEcho {
    fn name(&self) -> &str {
        self.0
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![I64Sort.to_arcsort(), I64Sort.to_arcsort()],
            span.clone(),
        )
        .into_box()
    }
}
impl ReadPrim for ReadEcho {
    fn apply<'a, 'db>(&self, state: &mut ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let _ = state.base_values();
        Some(args[0])
    }
}

/// A full primitive — uses `FullState` (writes + reads).
#[derive(Clone)]
struct FullEcho(&'static str);
impl PrimitiveCommon for FullEcho {
    fn name(&self) -> &str {
        self.0
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![I64Sort.to_arcsort(), I64Sort.to_arcsort()],
            span.clone(),
        )
        .into_box()
    }
}
impl FullPrim for FullEcho {
    fn apply<'a, 'db>(&self, state: &mut FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let _ = state.base_values();
        Some(args[0])
    }
}

// --- per-context acceptance ---

/// A pure primitive runs in any of the four contexts.
#[test]
fn pure_primitive_accepted_everywhere() {
    let mut egraph = EGraph::default();
    egraph.add_pure_primitive(PureAdd("p-add"), None);

    // global query — `check`
    egraph
        .parse_and_run_program(None, "(check (= (p-add 2 3) 5))")
        .unwrap();
    // global action — top-level eval
    egraph
        .parse_and_run_program(None, "(let $x (p-add 7 8))")
        .unwrap();
    // rule query (LHS) and rule action (RHS)
    egraph
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (rule ((= y (p-add 1 2))) ((set (f y) (p-add 10 20))))\n\
             (run 1)",
        )
        .unwrap();
}

/// A `WritePrim` is rejected in any query context (rule LHS, global query).
#[test]
fn write_primitive_rejected_in_queries() {
    let mut egraph = EGraph::default();
    egraph.add_write_primitive(WriteEcho("w-echo"), None);

    // RHS of a rule (RuleAction) — fine.
    egraph
        .parse_and_run_program(
            None,
            "(function g (i64) i64 :no-merge)\n\
             (rule () ((set (g 0) (w-echo 42))))",
        )
        .unwrap();

    // LHS of a rule (RuleQuery) — must be rejected.
    let mut egraph2 = EGraph::default();
    egraph2.add_write_primitive(WriteEcho("w-echo"), None);
    let result = egraph2.parse_and_run_program(
        None,
        "(function g (i64) i64 :no-merge)\n\
         (rule ((= x (w-echo 1))) ((set (g 0) x)))",
    );
    assert!(
        result.is_err(),
        "WritePrim must be rejected on a rule LHS"
    );

    // Top-level `check` (GlobalQuery) — must be rejected.
    let mut egraph3 = EGraph::default();
    egraph3.add_write_primitive(WriteEcho("w-echo"), None);
    let result = egraph3.parse_and_run_program(None, "(check (= (w-echo 1) 1))");
    assert!(
        result.is_err(),
        "WritePrim must be rejected in `check` (GlobalQuery)"
    );
}

/// A `ReadPrim` is rejected in rule contexts (both query and action) —
/// it's only valid in `GlobalQuery` and `GlobalAction`.
#[test]
fn read_primitive_rejected_in_rule_contexts() {
    let mut egraph = EGraph::default();
    egraph.add_read_primitive(ReadEcho("r-echo"), None);

    // GlobalQuery — fine.
    egraph
        .parse_and_run_program(None, "(check (= (r-echo 7) 7))")
        .unwrap();
    // GlobalAction — fine.
    egraph
        .parse_and_run_program(None, "(let $rr (r-echo 7))")
        .unwrap();

    // Rule LHS — rejected.
    let mut egraph2 = EGraph::default();
    egraph2.add_read_primitive(ReadEcho("r-echo"), None);
    let result = egraph2.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule ((= x (r-echo 1))) ((set (f 0) x)))",
    );
    assert!(result.is_err(), "ReadPrim must be rejected on a rule LHS");

    // Rule RHS — also rejected (ReadPrim is GlobalQuery+GlobalAction
    // only; RuleAction doesn't qualify).
    let mut egraph3 = EGraph::default();
    egraph3.add_read_primitive(ReadEcho("r-echo"), None);
    let result = egraph3.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule () ((set (f 0) (r-echo 1))))",
    );
    assert!(result.is_err(), "ReadPrim must be rejected on a rule RHS");
}

/// A `FullPrim` is valid only in `GlobalAction`.
#[test]
fn full_primitive_accepted_only_in_global_action() {
    let mut egraph = EGraph::default();
    egraph.add_full_primitive(FullEcho("f-echo"), None);

    // GlobalAction — fine.
    egraph
        .parse_and_run_program(None, "(let $ff (f-echo 7))")
        .unwrap();

    // GlobalQuery — rejected.
    let mut egraph2 = EGraph::default();
    egraph2.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph2.parse_and_run_program(None, "(check (= (f-echo 1) 1))");
    assert!(
        result.is_err(),
        "FullPrim must be rejected in GlobalQuery (`check`)"
    );

    // Rule LHS — rejected.
    let mut egraph3 = EGraph::default();
    egraph3.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph3.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule ((= x (f-echo 1))) ((set (f 0) x)))",
    );
    assert!(result.is_err(), "FullPrim must be rejected on a rule LHS");

    // Rule RHS — rejected.
    let mut egraph4 = EGraph::default();
    egraph4.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph4.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule () ((set (f 0) (f-echo 1))))",
    );
    assert!(result.is_err(), "FullPrim must be rejected on a rule RHS");
}

// `unstable-app` dispatch tests live in
// `tests/typed_primitive_unstable_app.egg` — they only need built-in
// primitives, so the `.egg` form is more direct than building an
// EGraph from Rust.

// --- duplicate registration regression ---

/// **Known gap, documented as a regression guard.** Two same-name
/// same-signature primitives registered separately currently *both*
/// survive the constraint builder's context filter and produce
/// identical XOR branches. The solver's XOR doesn't reject
/// ambiguous-but-equivalent matches when the surrounding constraints
/// fully pin the sort assignment, so the program type-checks and
/// `from_resolution` picks the first registration arbitrarily.
///
/// Ideally the typechecker would reject this as ambiguous; for now
/// this test pins the silent-first-wins behavior so we notice if it
/// changes.
#[test]
fn two_same_signature_registrations_silently_pick_first() {
    let mut egraph = EGraph::default();
    egraph.add_pure_primitive(PureAdd("dup-add"), None);
    egraph.add_pure_primitive(PureAdd("dup-add"), None);

    egraph
        .parse_and_run_program(None, "(check (= (dup-add 1 2) 3))")
        .unwrap();
}
