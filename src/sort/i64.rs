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
#[derive(Debug)]
pub struct I64Sort;

lazy_static! {
    static ref I64_SORT_NAME: Symbol = "i64".into();
}

impl Sort for I64Sort {
    fn name(&self) -> Symbol {
        *I64_SORT_NAME
    }

    fn column_ty(&self, prims: &Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<i64>())
    }

    fn register_type(&self, prims: &mut Primitives) {
        prims.register_type::<i64>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
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

        add_primitive!(eg, "log2" = |a: i64| -> i64 { (a as i64).ilog2() as i64 });

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

        add_primitive!(eg, "to-string" = |a: i64| -> Symbol { a.to_string().into() });

        // Must be in the i64 sort register function because
        // the string sort is registered before the i64 sort.
        add_primitive!(eg, "count-matches" = |a: S, b: S| -> i64 {
            a.as_str().matches(b.as_str()).count() as i64
        });
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        Some((1, termdag.lit(Literal::Int(value.bits as _))))
    }
}

impl IntoSort for i64 {
    type Sort = I64Sort;
    fn store(self, _sort: &Self::Sort) -> Value {
        Value {
            #[cfg(debug_assertions)]
            tag: I64Sort.name(),
            bits: self as u64,
        }
    }
}

impl FromSort for i64 {
    type Sort = I64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        value.bits as Self
    }
}
