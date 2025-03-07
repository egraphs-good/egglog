use super::*;
use crate::ast::Literal;
use ordered_float::OrderedFloat;

/// 64-bit floating point numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`, `neg`, `abs`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `to-i64`, `to-string`
#[derive(Debug)]
pub struct F64Sort;

lazy_static! {
    static ref F64_SORT_NAME: Symbol = "f64".into();
}

impl Sort for F64Sort {
    fn name(&self) -> Symbol {
        *F64_SORT_NAME
    }

    fn column_ty(&self, prims: &core_relations::Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<OrderedFloat<f64>>())
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        add_primitives!(eg, "+" = |a: f64, b: f64| -> f64 { a + b });
        add_primitives!(eg, "-" = |a: f64, b: f64| -> f64 { a - b });
        add_primitives!(eg, "*" = |a: f64, b: f64| -> f64 { a * b });
        add_primitives!(eg, "/" = |a: f64, b: f64| -> Option<f64> { (b != 0.0).then(|| a / b) });
        add_primitives!(eg, "%" = |a: f64, b: f64| -> Option<f64> { (b != 0.0).then(|| a % b) });
        add_primitives!(eg, "^" = |a: f64, b: f64| -> f64 { a.powf(b) });
        add_primitives!(eg, "neg" = |a: f64| -> f64 { -a });

        add_primitives!(eg, "<" = |a: f64, b: f64| -> Option<()> { (a < b).then(|| ()) });
        add_primitives!(eg, ">" = |a: f64, b: f64| -> Option<()> { (a > b).then(|| ()) });
        add_primitives!(eg, "<=" = |a: f64, b: f64| -> Option<()> { (a <= b).then(|| ()) });
        add_primitives!(eg, ">=" = |a: f64, b: f64| -> Option<()> { (a >= b).then(|| ()) });

        add_primitives!(eg, "min" = |a: f64, b: f64| -> f64 { a.min(b) });
        add_primitives!(eg, "max" = |a: f64, b: f64| -> f64 { a.max(b) });
        add_primitives!(eg, "abs" = |a: f64| -> f64 { a.abs() });

        // `to-f64` should be in `i64.rs`, but `F64Sort` wouldn't exist yet
        add_primitives!(eg, "to-f64" = |a: i64| -> f64 { a as f64 });
        add_primitives!(eg, "to-i64" = |a: f64| -> i64 { a as i64 });
        // Use debug instead of to_string so that decimal place is always printed
        add_primitives!(eg, "to-string" = |a: f64| -> Symbol { format!("{:?}", a).into() });
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        Some((
            1,
            termdag.lit(Literal::Float(OrderedFloat(f64::from_bits(value.bits)))),
        ))
    }
}

impl IntoSort for f64 {
    type Sort = F64Sort;
    fn store(self, _sort: &Self::Sort) -> Value {
        Value {
            #[cfg(debug_assertions)]
            tag: F64Sort.name(),
            bits: self.to_bits(),
        }
    }
}

impl FromSort for f64 {
    type Sort = F64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        f64::from_bits(value.bits)
    }
}
