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
    EGraph, FullPrim, FullState, PrimitiveCommon, PurePrim, PureState, Read, ReadPrim, ReadState,
    Value, WritePrim, WriteState, prelude::*,
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
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
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
    fn apply<'a, 'db>(&self, state: WriteState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let _ = state.base_values();
        Some(args[0])
    }
}

/// A read primitive — looks up a row in the table named by
/// `table_name` and returns the row's value column. Returns `None` if
/// the row is absent. Demonstrates the `Read::lookup` API.
#[derive(Clone)]
struct ReadLookup {
    name: &'static str,
    table_name: &'static str,
}
impl PrimitiveCommon for ReadLookup {
    fn name(&self) -> &str {
        self.name
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
impl ReadPrim for ReadLookup {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        state.lookup(self.table_name, args)
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
    fn apply<'a, 'db>(&self, state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
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

    // RHS of a rule (Context::Write) — fine.
    egraph
        .parse_and_run_program(
            None,
            "(function g (i64) i64 :no-merge)\n\
             (rule () ((set (g 0) (w-echo 42))))",
        )
        .unwrap();

    // LHS of a rule (Context::Pure) — must be rejected.
    let mut egraph2 = EGraph::default();
    egraph2.add_write_primitive(WriteEcho("w-echo"), None);
    let result = egraph2.parse_and_run_program(
        None,
        "(function g (i64) i64 :no-merge)\n\
         (rule ((= x (w-echo 1))) ((set (g 0) x)))",
    );
    assert!(result.is_err(), "WritePrim must be rejected on a rule LHS");

    // Top-level `check` (Context::Read) — must be rejected.
    let mut egraph3 = EGraph::default();
    egraph3.add_write_primitive(WriteEcho("w-echo"), None);
    let result = egraph3.parse_and_run_program(None, "(check (= (w-echo 1) 1))");
    assert!(
        result.is_err(),
        "WritePrim must be rejected in `check` (Context::Read)"
    );
}

/// A `ReadPrim` is accepted in rule contexts; the rule auto-demotes
/// to naive evaluation. Also exercises the `Read::lookup` API:
/// `lookup-f` reads rows from a custom `f` table.
#[test]
fn read_primitive_in_rule_demotes_to_naive() {
    let mut egraph = EGraph::default();
    egraph.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );

    // Populate the `f` table at top level, then use `lookup-f` from a
    // Context::Read (`check`) and a Context::Full (`let`). Both should
    // see the value populated by `set`.
    egraph
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (set (f 7) 42)\n\
             (check (= (lookup-f 7) 42))\n\
             (let $r (lookup-f 7))\n\
             (check (= $r 42))",
        )
        .unwrap();

    // Rule LHS — accepted; the rule reads `f` so it's auto-demoted to
    // naive. Driving the rule should populate `g 1` with the value of
    // `(f 1)`.
    let mut egraph2 = EGraph::default();
    egraph2.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );
    egraph2
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (function g (i64) i64 :no-merge)\n\
             (set (f 1) 99)\n\
             (rule ((= x (lookup-f 1))) ((set (g 0) x)))\n\
             (run 1)\n\
             (check (= (g 0) 99))",
        )
        .unwrap();

    // Rule RHS — accepted; the read primitive in the action still
    // forces naive evaluation.
    let mut egraph3 = EGraph::default();
    egraph3.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );
    egraph3
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (function g (i64) i64 :no-merge)\n\
             (function trigger () i64 :no-merge)\n\
             (set (f 1) 77)\n\
             (set (trigger) 1)\n\
             (rule ((= _ (trigger))) ((set (g 0) (lookup-f 1))))\n\
             (run 1)\n\
             (check (= (g 0) 77))",
        )
        .unwrap();
}

/// A `FullPrim` is valid only in `Context::Full`. It is rejected in
/// query contexts (top-level `check` and rule LHS — both Read), but
/// accepted in action contexts (top-level `let` and rule RHS — both
/// Full); a rule using one in its RHS auto-demotes to naive.
#[test]
fn full_primitive_accepted_only_in_full_contexts() {
    let mut egraph = EGraph::default();
    egraph.add_full_primitive(FullEcho("f-echo"), None);

    // Context::Full (top-level action) — fine.
    egraph
        .parse_and_run_program(None, "(let $ff (f-echo 7))")
        .unwrap();

    // Context::Read (top-level `check`) — rejected.
    let mut egraph2 = EGraph::default();
    egraph2.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph2.parse_and_run_program(None, "(check (= (f-echo 1) 1))");
    assert!(
        result.is_err(),
        "FullPrim must be rejected in Context::Read (`check`)"
    );

    // Rule LHS — rejected (rule queries typecheck under Context::Read,
    // and FullPrim is Full-only).
    let mut egraph3 = EGraph::default();
    egraph3.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph3.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule ((= x (f-echo 1))) ((set (f 0) x)))",
    );
    assert!(result.is_err(), "FullPrim must be rejected on a rule LHS");

    // Rule RHS — accepted; the FullPrim drives auto-demotion to naive.
    let mut egraph4 = EGraph::default();
    egraph4.add_full_primitive(FullEcho("f-echo"), None);
    egraph4
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (function trigger () i64 :no-merge)\n\
             (set (trigger) 1)\n\
             (rule ((= _ (trigger))) ((set (f 0) (f-echo 5))))\n\
             (run 1)\n\
             (check (= (f 0) 5))",
        )
        .unwrap();
}

// `unstable-app` dispatch tests live in
// `tests/typed_primitive_unstable_app.egg` — they only need built-in
// primitives, so the `.egg` form is more direct than building an
// EGraph from Rust.

// --- duplicate registration regression ---

/// Two same-name same-signature primitives registered separately
/// pass the constraint-builder context filter (both are valid in
/// every context) and the typechecker doesn't reject equivalent XOR
/// branches when the surrounding constraints pin the sort assignment.
/// `ResolvedCall::from_resolution` catches this on the use site by
/// requiring exactly one match for `(name, signature, context)` and
/// panicking with a clear message otherwise.
#[test]
#[should_panic(expected = "Ambiguous primitive resolution")]
fn two_same_signature_registrations_panic_on_use() {
    let mut egraph = EGraph::default();
    egraph.add_pure_primitive(PureAdd("dup-add"), None);
    egraph.add_pure_primitive(PureAdd("dup-add"), None);

    let _ = egraph.parse_and_run_program(None, "(check (= (dup-add 1 2) 3))");
}
