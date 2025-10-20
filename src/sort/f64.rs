use super::{
    BaseSort, BaseValues, Debug, EGraph, F, Literal, Signed, Term, TermDag, Value, add_primitive,
    i64,
};

/// 64-bit floating point numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`, `neg`, `abs`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `to-i64`, `to-string`
#[derive(Debug)]
pub struct F64Sort;

impl BaseSort for F64Sort {
    type Base = F;

    fn name(&self) -> &'static str {
        "f64"
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "+" = |a: F, b: F| -> F { a + b });
        add_primitive!(eg, "-" = |a: F, b: F| -> F { a - b });
        add_primitive!(eg, "*" = |a: F, b: F| -> F { a * b });
        add_primitive!(eg, "/" = |a: F, b: F| -?> F { (*b != 0.0).then(|| a / b) });
        add_primitive!(eg, "%" = |a: F, b: F| -?> F { (*b != 0.0).then(|| a % b) });
        add_primitive!(eg, "^" = |a: F, b: F| -> F { F::from(OrderedFloat(a.powf(**b))) });
        add_primitive!(eg, "neg" = |a: F| -> F { -a });

        add_primitive!(eg, "<" = |a: F, b: F| -?> () { (a < b).then(|| ()) });
        add_primitive!(eg, ">" = |a: F, b: F| -?> () { (a > b).then(|| ()) });
        add_primitive!(eg, "<=" = |a: F, b: F| -?> () { (a <= b).then(|| ()) });
        add_primitive!(eg, ">=" = |a: F, b: F| -?> () { (a >= b).then(|| ()) });

        add_primitive!(eg, "min" = |a: F, b: F| -> F { a.min(b) });
        add_primitive!(eg, "max" = |a: F, b: F| -> F { a.max(b) });
        add_primitive!(eg, "abs" = |a: F| -> F { F::from(a.abs()) });

        // `to-f64` should be in `i64.rs`, but `F64Sort` wouldn't exist yet
        add_primitive!(eg, "to-f64" = |a: i64| -> F { F::from(OrderedFloat(a as f64)) });
        add_primitive!(eg, "to-i64" = |a: F| -> i64 { a.0.0 as i64 });
        // Use debug instead of to_string so that decimal place is always printed
        add_primitive!(eg, "to-string" = |a: F| -> S { S::new(format!("{:?}", a.0.0)) });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let f = base_values.unwrap::<F>(value);

        termdag.lit(Literal::Float(f.0))
    }
}
