//! Tests for the typed primitive surface and the seminaive-safety
//! enforcement added in issue #772.
//!
//! Covers:
//! - Pure / write / read / full primitives accepted only in their
//!   respective valid contexts (typechecker rejects others).
//! - Higher-order primitive values carry runtime ids for every context where
//!   the wrapped primitive is valid; application in other contexts hits the
//!   mismatch path.
//! - `unstable-fn` over constructors and custom functions preserves the
//!   existing function/container runtime checks.
//! - Duplicate same-signature primitive registrations are ambiguous for direct
//!   calls and higher-order primitive dispatch.

use egglog::add_primitive;
use egglog::ast::Span;
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::sort::{I64Sort, S, StringSort};
use egglog::{
    EGraph, FullPrim, FullState, Primitive, PurePrim, PureState, RawValues, Read, ReadPrim,
    ReadState, Value, WritePrim, WriteState, prelude::*,
};

// --- shared test fixtures ---

/// A pure primitive that adds two i64s. Trivially safe in every context.
#[derive(Clone)]
struct PureAdd(&'static str);
impl Primitive for PureAdd {
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
impl Primitive for WriteEcho {
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
impl Primitive for ReadLookup {
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
        state
            .lookup(self.table_name, RawValues(args.to_vec()))
            .ok()
            .flatten()
    }
}

/// A read primitive that uses the read-side table-size API.
#[derive(Clone)]
struct ReadTableSize(&'static str);
impl Primitive for ReadTableSize {
    fn name(&self) -> &str {
        self.0
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![StringSort.to_arcsort(), I64Sort.to_arcsort()],
            span.clone(),
        )
        .into_box()
    }
}
impl ReadPrim for ReadTableSize {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let table_name = state.base_values().unwrap::<S>(args[0]).0;
        let size = state.table_size(&table_name).unwrap_or(0);
        let size = i64::try_from(size).ok()?;
        Some(state.base_values().get::<i64>(size))
    }
}

/// A read primitive that uses the read-side all-table-size snapshot API.
#[derive(Clone)]
struct ReadAllTableSizes(&'static str);
impl Primitive for ReadAllTableSizes {
    fn name(&self) -> &str {
        self.0
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![I64Sort.to_arcsort()], span.clone()).into_box()
    }
}
impl ReadPrim for ReadAllTableSizes {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, _args: &[Value]) -> Option<Value> {
        let size: usize = state.table_sizes().into_iter().map(|(_, size)| size).sum();
        let size = i64::try_from(size).ok()?;
        Some(state.base_values().get::<i64>(size))
    }
}

/// A full primitive — uses `FullState` (writes + reads).
#[derive(Clone)]
struct FullEcho(&'static str);
impl Primitive for FullEcho {
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

/// A pure primitive with the (i64) -> i64 shape used by the
/// unstable-app dispatch matrix below — uniform signature lets the
/// same `unstable-fn` / `unstable-app` programs cover all four
/// registration kinds.
#[derive(Clone)]
struct PureEcho(&'static str);
impl Primitive for PureEcho {
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
impl PurePrim for PureEcho {
    fn apply<'a, 'db>(&self, _state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        Some(args[0])
    }
}

/// A read primitive that doesn't actually consult a table — its body
/// just echoes `args[0]`. The trait still wraps it as a `ReadState`
/// at dispatch time, which is what the matrix test cares about.
#[derive(Clone)]
struct ReadEcho(&'static str);
impl Primitive for ReadEcho {
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
    fn apply<'a, 'db>(&self, _state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
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

/// A `ReadPrim` is rejected in rule contexts (both query and action) —
/// it's only valid in `Context::Read` and `Context::Full`. To use one
/// inside a rule, the rule must opt out of seminaive with `:naive`.
#[test]
fn read_primitive_rejected_in_rule_contexts() {
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

    // Rule LHS without `:naive` — rejected (Context::Pure isn't in
    // `ReadPrim`'s valid contexts).
    let mut egraph2 = EGraph::default();
    egraph2.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );
    let result = egraph2.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (function g (i64) i64 :no-merge)\n\
         (rule ((= x (lookup-f 1))) ((set (g 0) x)))",
    );
    assert!(result.is_err(), "ReadPrim must be rejected on a rule LHS");

    // Rule RHS without `:naive` — also rejected (ReadPrim is Read+Full
    // only; Context::Write doesn't qualify).
    let mut egraph3 = EGraph::default();
    egraph3.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );
    let result = egraph3.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (function g (i64) i64 :no-merge)\n\
         (rule () ((set (g 0) (lookup-f 1))))",
    );
    assert!(result.is_err(), "ReadPrim must be rejected on a rule RHS");

    // With `:naive` — both LHS and RHS accepted; the rule scans the
    // whole DB each iteration, so reads are sound.
    let mut egraph4 = EGraph::default();
    egraph4.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );
    egraph4
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (function g (i64) i64 :no-merge)\n\
             (set (f 1) 99)\n\
             (rule ((= x (lookup-f 1))) ((set (g 0) x)) :naive)\n\
             (run 1)\n\
             (check (= (g 0) 99))",
        )
        .unwrap();
}

#[test]
fn read_primitive_can_observe_table_sizes() {
    let mut egraph = EGraph::default();
    egraph.add_read_primitive(ReadTableSize("table-size"), None);
    egraph.add_read_primitive(ReadAllTableSizes("all-table-sizes"), None);

    egraph
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (set (f 1) 10)\n\
             (set (f 2) 20)\n\
             (check (= (table-size \"f\") 2))\n\
             (check (= (all-table-sizes) 2))",
        )
        .unwrap();
}

/// A `FullPrim` is valid only in `Context::Full`.
#[test]
fn full_primitive_accepted_only_in_global_action() {
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

    // Rule LHS without `:naive` — rejected.
    let mut egraph3 = EGraph::default();
    egraph3.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph3.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule ((= x (f-echo 1))) ((set (f 0) x)))",
    );
    assert!(result.is_err(), "FullPrim must be rejected on a rule LHS");

    // Rule RHS without `:naive` — rejected (action ctx is Write,
    // FullPrim is Full-only).
    let mut egraph4 = EGraph::default();
    egraph4.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph4.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (rule () ((set (f 0) (f-echo 1))))",
    );
    assert!(result.is_err(), "FullPrim must be rejected on a rule RHS");

    // With `:naive`, both LHS and RHS accept FullPrim (action ctx
    // widens to Full).
    let mut egraph5 = EGraph::default();
    egraph5.add_full_primitive(FullEcho("f-echo"), None);
    egraph5
        .parse_and_run_program(
            None,
            "(function f (i64) i64 :no-merge)\n\
             (function trigger () i64 :no-merge)\n\
             (set (trigger) 1)\n\
             (rule ((= _ (trigger))) ((set (f 0) (f-echo 5))) :naive)\n\
             (run 1)\n\
             (check (= (f 0) 5))",
        )
        .unwrap();
}

/// Merge expressions are action-side writes, not top-level full actions:
/// they may use pure/write primitives, but not primitives that read live DB
/// state.
#[test]
fn merge_primitives_use_write_context() {
    let mut egraph = EGraph::default();
    egraph.add_write_primitive(WriteEcho("w-echo"), None);
    egraph
        .parse_and_run_program(None, "(function g () i64 :merge (w-echo old))")
        .unwrap();

    let mut egraph2 = EGraph::default();
    egraph2.add_read_primitive(
        ReadLookup {
            name: "lookup-f",
            table_name: "f",
        },
        None,
    );
    let result = egraph2.parse_and_run_program(
        None,
        "(function f (i64) i64 :no-merge)\n\
         (function g () i64 :merge (lookup-f old))",
    );
    assert!(result.is_err(), "ReadPrim must be rejected in :merge");

    let mut egraph3 = EGraph::default();
    egraph3.add_full_primitive(FullEcho("f-echo"), None);
    let result = egraph3.parse_and_run_program(None, "(function g () i64 :merge (f-echo old))");
    assert!(result.is_err(), "FullPrim must be rejected in :merge");
}

// `unstable-app` dispatch tests live in
// `tests/typed_primitive_unstable_app.egg` — they only need built-in
// primitives, so the `.egg` form is more direct than building an
// EGraph from Rust.

// --- duplicate registration regression ---

/// Direct primitive resolution requires exactly one matching registration for
/// `(name, signature, context)`. Two independently registered pure primitives
/// both carry valid runtime ids for every context, so a same-signature direct
/// call is ambiguous instead of silently picking one registration.
#[test]
#[should_panic(expected = "Ambiguous primitive resolution")]
fn two_same_signature_registrations_panic_on_use() {
    let mut egraph = EGraph::default();
    egraph.add_pure_primitive(PureAdd("dup-add"), None);
    egraph.add_pure_primitive(PureAdd("dup-add"), None);

    let _ = egraph.parse_and_run_program(None, "(check (= (dup-add 1 2) 3))");
}

/// Registering a primitive whose argument type has no corresponding sort must
/// fail with a message naming the missing type.
#[test]
#[should_panic(expected = "Expected exactly one sort for type `u32`")]
fn missing_sort_panics_with_type_name() {
    let mut egraph = EGraph::default();
    add_primitive!(&mut egraph, "u32-id" = |a: u32| -> i64 { a as i64 });
}

/// `unstable-fn` over a primitive must preserve the same exact-one ambiguity
/// rule as direct primitive calls. The wrapped value records valid runtime ids
/// per application context, and duplicate same-signature registrations are
/// ambiguous for every context where more than one runtime id matches.
#[test]
#[should_panic(expected = "Ambiguous primitive resolution")]
fn unstable_fn_duplicate_primitive_registration_panics_on_build() {
    let mut egraph = EGraph::default();
    egraph.add_pure_primitive(PureEcho("dup-echo"), None);
    egraph.add_pure_primitive(PureEcho("dup-echo"), None);

    let _ = egraph.parse_and_run_program(
        None,
        "(sort Fn (UnstableFn (i64) i64))\n\
         (let $f (unstable-fn \"dup-echo\"))\n\
         (check (= (unstable-app $f 7) 7))",
    );
}

// --- 4x4 unstable-app dispatch matrix ---
//
// `unstable-fn` over a primitive builds a per-context runtime id table, and
// `unstable-app` selects the id for the application context. For each
// registration kind we wrap a uniform (i64)->i64 echo primitive and apply it
// from all four application contexts. Dispatch succeeds iff the application
// context has a runtime id; otherwise the pre-registered mismatch panic
// surfaces as an error.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AppCtx {
    /// Rule LHS under default seminaive — body context is `Pure`.
    Pure,
    /// Top-level `check` — context is `Read`.
    Read,
    /// Rule RHS under default seminaive — action context is `Write`.
    Write,
    /// Top-level `let` — context is `Full`.
    Full,
}

const ALL_CTXS: [AppCtx; 4] = [AppCtx::Pure, AppCtx::Read, AppCtx::Write, AppCtx::Full];

fn matrix_program(ctx: AppCtx) -> String {
    let header = "(sort Fn (UnstableFn (i64) i64))\n\
                  (let $f (unstable-fn \"p\"))\n";
    match ctx {
        AppCtx::Pure => format!(
            "{header}(function out (i64) i64 :no-merge)\n\
             (rule ((= y (unstable-app $f 7))) ((set (out 0) y)))\n\
             (run 1)\n\
             (check (= (out 0) 7))"
        ),
        AppCtx::Read => format!("{header}(check (= (unstable-app $f 7) 7))"),
        AppCtx::Write => format!(
            "{header}(function out (i64) i64 :no-merge)\n\
             (rule () ((set (out 0) (unstable-app $f 7))))\n\
             (run 1)\n\
             (check (= (out 0) 7))"
        ),
        AppCtx::Full => format!("{header}(let $r (unstable-app $f 7))\n(check (= $r 7))"),
    }
}

fn run_matrix_cell(register: impl FnOnce(&mut EGraph), ctx: AppCtx) -> Result<(), String> {
    let mut egraph = EGraph::default();
    register(&mut egraph);
    egraph
        .parse_and_run_program(None, &matrix_program(ctx))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn unstable_app_dispatch_matrix() {
    // For each (registration kind, application ctx) cell, expected
    // success follows the trait's `valid_contexts`.
    let cells: &[(&str, fn(&mut EGraph), &[AppCtx])] = &[
        (
            "pure",
            |e: &mut EGraph| e.add_pure_primitive(PureEcho("p"), None),
            &[AppCtx::Pure, AppCtx::Read, AppCtx::Write, AppCtx::Full],
        ),
        (
            "read",
            |e: &mut EGraph| e.add_read_primitive(ReadEcho("p"), None),
            &[AppCtx::Read, AppCtx::Full],
        ),
        (
            "write",
            |e: &mut EGraph| e.add_write_primitive(WriteEcho("p"), None),
            &[AppCtx::Write, AppCtx::Full],
        ),
        (
            "full",
            |e: &mut EGraph| e.add_full_primitive(FullEcho("p"), None),
            &[AppCtx::Full],
        ),
    ];

    for (label, register, valid) in cells {
        for &ctx in &ALL_CTXS {
            let result = run_matrix_cell(*register, ctx);
            let should_succeed = valid.contains(&ctx);
            if should_succeed {
                assert!(
                    result.is_ok(),
                    "{label} prim applied via unstable-app in {ctx:?} ctx should succeed; \
                     got error: {:?}",
                    result.err()
                );
            } else {
                assert!(
                    result.is_err(),
                    "{label} prim applied via unstable-app in {ctx:?} ctx should fail \
                     (dispatch mismatch panic), but parse_and_run_program returned Ok"
                );
            }
        }
    }
}
