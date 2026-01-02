use super::*;

/// 64-bit floating point numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`, `neg`, `abs`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `to-i64`, `to-string`
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
        add_literal_prim!(eg, "+" = |a: F, b: F| -> F { a + b });
        add_literal_prim!(eg, "-" = |a: F, b: F| -> F { a - b });
        add_literal_prim!(eg, "*" = |a: F, b: F| -> F { a * b });
        add_literal_prim!(eg, "/" = |a: F, b: F| -?> F { (*b != 0.0).then(|| a / b) });
        add_literal_prim!(eg, "%" = |a: F, b: F| -?> F { (*b != 0.0).then(|| a % b) });
        add_literal_prim!(eg, "^" = |a: F, b: F| -> F { F::from(OrderedFloat(a.powf(**b))) });
        add_literal_prim!(eg, "neg" = |a: F| -> F { -a });

        let f64_less_than_validator = |termdag: &TermDag, args: &[TermId], _result: TermId| -> bool {
            let Term::Lit(Literal::Float(a)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::Float(b)) = termdag.get(args[1]) else { return false };
            a < b
        };
        add_primitive_with_validator!(eg, "<" = |a: F, b: F| -?> () { (a < b).then(|| ()) }, f64_less_than_validator);

        let f64_greater_than_validator = |termdag: &TermDag, args: &[TermId], _result: TermId| -> bool {
            let Term::Lit(Literal::Float(a)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::Float(b)) = termdag.get(args[1]) else { return false };
            a > b
        };
        add_primitive_with_validator!(eg, ">" = |a: F, b: F| -?> () { (a > b).then(|| ()) }, f64_greater_than_validator);

        let f64_less_equal_validator = |termdag: &TermDag, args: &[TermId], _result: TermId| -> bool {
            let Term::Lit(Literal::Float(a)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::Float(b)) = termdag.get(args[1]) else { return false };
            a <= b
        };
        add_primitive_with_validator!(eg, "<=" = |a: F, b: F| -?> () { (a <= b).then(|| ()) }, f64_less_equal_validator);

        let f64_greater_equal_validator = |termdag: &TermDag, args: &[TermId], _result: TermId| -> bool {
            let Term::Lit(Literal::Float(a)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::Float(b)) = termdag.get(args[1]) else { return false };
            a >= b
        };
        add_primitive_with_validator!(eg, ">=" = |a: F, b: F| -?> () { (a >= b).then(|| ()) }, f64_greater_equal_validator);

        add_literal_prim!(eg, "min" = |a: F, b: F| -> F { a.min(b) });
        add_literal_prim!(eg, "max" = |a: F, b: F| -> F { a.max(b) });
        add_literal_prim!(eg, "abs" = |a: F| -> F { F::from(a.abs()) });

        // `to-f64` should be in `i64.rs`, but `F64Sort` wouldn't exist yet
        let to_f64_validator = |termdag: &TermDag, args: &[TermId], result: TermId| -> bool {
            let Term::Lit(Literal::Int(a)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::Float(f)) = termdag.get(result) else { return false };
            *f == OrderedFloat(*a as f64)
        };
        add_primitive_with_validator!(eg, "to-f64" = |a: i64| -> F { F::from(OrderedFloat(a as f64)) }, to_f64_validator);

        let to_i64_validator = |termdag: &TermDag, args: &[TermId], result: TermId| -> bool {
            let Term::Lit(Literal::Float(f)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::Int(i)) = termdag.get(result) else { return false };
            *i == f.0 as i64
        };
        add_primitive_with_validator!(eg, "to-i64" = |a: F| -> i64 { a.0.0 as i64 }, to_i64_validator);

        // Use debug instead of to_string so that decimal place is always printed
        let f64_to_string_validator = |termdag: &TermDag, args: &[TermId], result: TermId| -> bool {
            let Term::Lit(Literal::Float(f)) = termdag.get(args[0]) else { return false };
            let Term::Lit(Literal::String(s)) = termdag.get(result) else { return false };
            s.as_str() == format!("{:?}", f.0)
        };
        add_primitive_with_validator!(eg, "to-string" = |a: F| -> S { S::new(format!("{:?}", a.0.0)) }, f64_to_string_validator);
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
