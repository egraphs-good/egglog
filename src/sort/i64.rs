use super::*;

/// Signed 64-bit integers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`
/// - Bitwise: `&`, `|`, `^`, `<<`, `>>`, `not-i64`
/// - Fallible comparisons: `<`, `>`, `<=`, `>=`
/// - Boolean comparisons: `bool-=`, `bool-<`, `bool->`, `bool-<=`, `bool->=`
/// - Other: `min`, `max`, `to-f64`, `to-string`, `log2`
///
/// Note: fallible comparisons are used at the top-level of a query.
/// For example, this rule will only match if `a` is less than `b`.
/// ```text
/// (rule (... (< a b)) (...))
/// ```
/// On the other hand, boolean comparisons will always match, and so
/// make sense to use inside expressions.
#[derive(Debug, Serialize, Deserialize)]
pub struct I64Sort;

impl BaseSort for I64Sort {
    type Base = i64;

    fn name(&self) -> &str {
        "i64"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "+" = |a: i64, b: i64| -?> i64 { a.checked_add(b) });
        add_primitive!(eg, "-" = |a: i64, b: i64| -?> i64 { a.checked_sub(b) });
        add_primitive!(eg, "*" = |a: i64, b: i64| -?> i64 { a.checked_mul(b) });
        add_primitive!(eg, "/" = |a: i64, b: i64| -?> i64 { a.checked_div(b) });
        add_primitive!(eg, "%" = |a: i64, b: i64| -?> i64 { a.checked_rem(b) });

        add_primitive!(eg, "&" = |a: i64, b: i64| -> i64 { a & b });
        add_primitive!(eg, "|" = |a: i64, b: i64| -> i64 { a | b });
        add_primitive!(eg, "^" = |a: i64, b: i64| -> i64 { a ^ b });
        add_primitive!(eg, "<<" = |a: i64, b: i64| -?> i64 { b.try_into().ok().and_then(|b| a.checked_shl(b)) });
        add_primitive!(eg, ">>" = |a: i64, b: i64| -?> i64 { b.try_into().ok().and_then(|b| a.checked_shr(b)) });
        add_primitive!(eg, "not-i64" = |a: i64| -> i64 { !a });

        add_primitive!(eg, "log2" = |a: i64| -> i64 { a.ilog2() as i64 });

        add_primitive!(eg, "<" = |a: i64, b: i64| -?> () { (a < b).then_some(()) });
        add_primitive!(eg, ">" = |a: i64, b: i64| -?> () { (a > b).then_some(()) });
        add_primitive!(eg, "<=" = |a: i64, b: i64| -?> () { (a <= b).then_some(()) });
        add_primitive!(eg, ">=" = |a: i64, b: i64| -?> () { (a >= b).then_some(()) });

        add_primitive!(eg, "bool-=" = |a: i64, b: i64| -> bool { a == b });
        add_primitive!(eg, "bool-<" = |a: i64, b: i64| -> bool { a < b });
        add_primitive!(eg, "bool->" = |a: i64, b: i64| -> bool { a > b });
        add_primitive!(eg, "bool-<=" = |a: i64, b: i64| -> bool { a <= b });
        add_primitive!(eg, "bool->=" = |a: i64, b: i64| -> bool { a >= b });

        add_primitive!(eg, "min" = |a: i64, b: i64| -> i64 { a.min(b) });
        add_primitive!(eg, "max" = |a: i64, b: i64| -> i64 { a.max(b) });

        add_primitive!(eg, "to-string" = |a: i64| -> S { S::new(a.to_string()) });

        // Must be in the i64 sort register function because
        // the string sort is registered before the i64 sort.
        add_primitive!(eg, "count-matches" = |a: S, b: S| -> i64 {
            a.as_str().matches(b.as_str()).count() as i64
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let i = base_values.unwrap::<i64>(value);

        termdag.lit(Literal::Int(i))
    }
}
