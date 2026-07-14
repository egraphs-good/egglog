//! Constant folding case study — uses the Rust API end-to-end
//! to build a small arithmetic e-graph, install a `rust_rule` that
//! folds `(Add (Num a) (Num b))` to `(Num (a+b))`, run to saturation,
//! and read back the canonical eclass.
//!
//! Exercises: `EGraph::update`, `Write::add` (constructor minting
//! both from outside a rule and from inside a `rust_rule` callback),
//! `Write::union` (inside the rule), `Read::eclass_of`,
//! `Read::table_sizes`, `Read::constructor_enodes`, `add_ruleset`,
//! `run_ruleset`.

use egglog::prelude::*;
use egglog::{Error, RawValues, Value};

/// Build the e-graph schema we'll work in:
///
/// ```text
/// (datatype Expr
///   (Num i64)
///   (Add Expr Expr))
/// ```
fn make_egraph() -> EGraph {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype Expr (Num i64) (Add Expr Expr))")
        .unwrap();
    eg
}

/// Register the constant-folding rule. Whenever `e = (Add n1 n2)`
/// holds and `n1 = (Num a)` and `n2 = (Num b)`, mint `(Num (a+b))`
/// and union it with `e`.
///
/// The rule body uses `value_to_base` to extract the two i64
/// payloads bound by the matcher, then `Write::add` + `Write::union`
/// on the WriteState.
fn install_const_fold_rule(eg: &mut EGraph) -> Result<(), Error> {
    let ruleset = "const_fold";
    add_ruleset(eg, ruleset)?;

    let expr_sort = eg.get_sort_by_name("Expr").unwrap().clone();
    rust_rule(
        eg,
        "fold_add",
        ruleset,
        vars![sum: (expr_sort.clone()), lhs: (expr_sort.clone()), rhs: (expr_sort), a: i64, b: i64],
        facts![
            (= sum (Add lhs rhs))
            (= lhs (Num a))
            (= rhs (Num b))
        ],
        move |mut ctx, vals| {
            let [sum, _lhs, _rhs, a, b] = vals else {
                unreachable!()
            };
            let a = ctx.value_to_base::<i64>(*a);
            let b = ctx.value_to_base::<i64>(*b);
            let folded = ctx.add("Num", a + b).ok()?;
            ctx.union(*sum, folded).ok()?;
            Some(())
        },
    )?;
    Ok(())
}

#[test]
fn const_fold_collapses_addition_chain() -> Result<(), Error> {
    // Build  (Add (Num 2) (Add (Num 3) (Num 4)))  programmatically.
    let mut eg = make_egraph();
    let root = eg.update(|mut fs| -> Result<Value, Error> {
        let n2 = fs.add("Num", 2_i64)?;
        let n3 = fs.add("Num", 3_i64)?;
        let n4 = fs.add("Num", 4_i64)?;
        let inner = fs.add("Add", RawValues(vec![n3, n4]))?;
        fs.add("Add", RawValues(vec![n2, inner]))
    })?;

    install_const_fold_rule(&mut eg)?;

    // Saturate. Two rounds: 3+4 → 7 first, then 2+7 → 9.
    for _ in 0..4 {
        run_ruleset(&mut eg, "const_fold")?;
    }

    // The root eclass should now contain `(Num 9)`. We check this
    // two ways:
    //   (a) `eclass_of("Num", 9)` exists and equals `root`;
    //   (b) enumerate the constructor rows whose output equals
    //       `root` — one of them should be a `Num` row carrying `9`.

    let nine = eg.update(|fs| fs.eclass_of("Num", 9_i64))?;
    assert_eq!(nine, Some(root), "(Num 9) should be unioned with root");

    // (b) — walk Num enodes looking for one whose eclass equals `root`
    //     and whose i64 input is `9`, stopping at the first match.
    let mut found_num_nine = false;
    let nine_value = eg.base_to_value::<i64>(9);
    eg.update(|fs| {
        fs.constructor_enodes_while("Num", |enode| {
            if enode.eclass == root && enode.children[0] == nine_value {
                found_num_nine = true;
                false
            } else {
                true
            }
        })
    })?;
    assert!(
        found_num_nine,
        "root eclass should contain a (Num 9) representative"
    );

    Ok(())
}

#[test]
fn const_fold_is_a_no_op_when_no_pair_of_nums() -> Result<(), Error> {
    // Single number; nothing to fold.
    let mut eg = make_egraph();
    let root = eg.update(|mut fs| fs.add("Num", 7_i64))?;
    install_const_fold_rule(&mut eg)?;

    let sizes_before = eg.update(|fs| -> Result<_, Error> {
        Ok(fs
            .table_sizes()
            .into_iter()
            .map(|(n, s)| (n.to_owned(), s))
            .collect::<Vec<_>>())
    })?;

    for _ in 0..3 {
        run_ruleset(&mut eg, "const_fold")?;
    }

    let sizes_after = eg.update(|fs| -> Result<_, Error> {
        Ok(fs
            .table_sizes()
            .into_iter()
            .map(|(n, s)| (n.to_owned(), s))
            .collect::<Vec<_>>())
    })?;

    assert_eq!(sizes_before, sizes_after, "no rules should have fired");

    // And `(Num 7)` is still the canonical eclass.
    let seven = eg.update(|fs| fs.eclass_of("Num", 7_i64))?;
    assert_eq!(seven, Some(root));
    Ok(())
}
