use super::*;

/// 64-bit floating point numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`, `neg`, `abs`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `exp`, `log`, `sqrt`
/// - Conversions: `to-f64`, `to-i64`, `to-string`
#[derive(Debug)]
pub struct F64Sort;

impl BaseSort for F64Sort {
    type Base = F;

    fn name(&self) -> &str {
        "f64"
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::builtin::FamilyDescription {
            name: "f64".into(), arity: 0,
        }).expect("invalid f64 family description");
        add_literal_prim!(eg, "+" [id = "egglog.core.f64.add"] = |a: F, b: F| -> F { a + b });
        add_literal_prim!(eg, "-" [id = "egglog.core.f64.sub"] = |a: F, b: F| -> F { a - b });
        add_literal_prim!(eg, "*" [id = "egglog.core.f64.mul"] = |a: F, b: F| -> F { a * b });
        add_literal_prim!(eg, "/" [id = "egglog.core.f64.div"] = |a: F, b: F| -?> F { (*b != 0.0).then(|| a / b) });
        add_literal_prim!(eg, "%" [id = "egglog.core.f64.rem"] = |a: F, b: F| -?> F { (*b != 0.0).then(|| a % b) });
        add_literal_prim!(eg, "^" [id = "egglog.core.f64.pow"] = |a: F, b: F| -> F { F::from(OrderedFloat(a.powf(**b))) });
        add_literal_prim!(eg, "neg" [id = "egglog.core.f64.neg"] = |a: F| -> F { -a });

        add_literal_prim!(eg, "<" [id = "egglog.core.f64.lt"] = |a: F, b: F| -?> () { (a < b).then(|| ()) });
        add_literal_prim!(eg, ">" [id = "egglog.core.f64.gt"] = |a: F, b: F| -?> () { (a > b).then(|| ()) });
        add_literal_prim!(eg, "<=" [id = "egglog.core.f64.le"] = |a: F, b: F| -?> () { (a <= b).then(|| ()) });
        add_literal_prim!(eg, ">=" [id = "egglog.core.f64.ge"] = |a: F, b: F| -?> () { (a >= b).then(|| ()) });

        add_literal_prim!(eg, "min" [id = "egglog.core.f64.min"] = |a: F, b: F| -> F { a.min(b) });
        add_literal_prim!(eg, "max" [id = "egglog.core.f64.max"] = |a: F, b: F| -> F { a.max(b) });
        add_literal_prim!(eg, "abs" [id = "egglog.core.f64.abs"] = |a: F| -> F { F::from(a.abs()) });
        add_literal_prim!(eg, "exp" [id = "egglog.core.f64.exp"] = |a: F| -> F { F::from(OrderedFloat(a.exp())) });
        add_literal_prim!(eg, "log" [id = "egglog.core.f64.log"] = |a: F| -?> F {
            (*a > OrderedFloat(0.0)).then(|| F::from(OrderedFloat(a.ln())))
        });
        add_literal_prim!(eg, "sqrt" [id = "egglog.core.f64.sqrt"] = |a: F| -?> F {
            (*a >= OrderedFloat(0.0)).then(|| F::from(OrderedFloat(a.sqrt())))
        });

        // `to-f64` should be in `i64.rs`, but `F64Sort` wouldn't exist yet
        add_literal_prim!(eg, "to-f64" [id = "egglog.core.f64.from_i64"] = |a: i64| -> F { F::from(OrderedFloat(a as f64)) });
        add_literal_prim!(eg, "to-i64" [id = "egglog.core.f64.to_i64"] = |a: F| -> i64 { a.0.0 as i64 });
        // Use debug instead of to_string so that decimal place is always printed
        add_literal_prim!(eg, "to-string" [id = "egglog.core.f64.to_string"] = |a: F| -> S { S::new(format!("{:?}", a.0.0)) });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let f = base_values.unwrap::<F>(value);

        termdag.lit(Literal::Float(f.0))
    }
}
