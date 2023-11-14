use super::*;
use crate::ast::Literal;
use ordered_float::OrderedFloat;

#[derive(Debug)]
pub struct F64Sort {
    name: Symbol,
}

impl F64Sort {
    pub fn new(name: Symbol) -> Self {
        Self { name }
    }
}

impl Sort for F64Sort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "neg" = |a: f64| -> f64 { -a });

        add_primitives!(eg, "+" = |a: f64, b: f64| -> f64 { a + b });
        add_primitives!(eg, "-" = |a: f64, b: f64| -> f64 { a - b });
        add_primitives!(eg, "*" = |a: f64, b: f64| -> f64 { a * b });
        add_primitives!(eg, "/" = |a: f64, b: f64| -> Opt<f64> { (b != 0.0).then(|| a / b) });
        add_primitives!(eg, "%" = |a: f64, b: f64| -> Opt<f64> { (b != 0.0).then(|| a % b) });

        add_primitives!(eg, "<" = |a: f64, b: f64| -> Opt { (a < b).then(|| ()) });
        add_primitives!(eg, ">" = |a: f64, b: f64| -> Opt { (a > b).then(|| ()) });
        add_primitives!(eg, "<=" = |a: f64, b: f64| -> Opt { (a <= b).then(|| ()) });
        add_primitives!(eg, ">=" = |a: f64, b: f64| -> Opt { (a >= b).then(|| ()) });

        add_primitives!(eg, "min" = |a: f64, b: f64| -> f64 { a.min(b) });
        add_primitives!(eg, "max" = |a: f64, b: f64| -> f64 { a.max(b) });
        add_primitives!(eg, "abs" = |a: f64| -> f64 { a.abs() });

        add_primitives!(eg, "to-f64" = |a: i64| -> f64 { a as f64 });
        add_primitives!(eg, "to-i64" = |a: f64| -> i64 { a as i64 });
        // Use debug instead of to_string so that decimal place is always printed
        add_primitives!(eg, "to-string" = |a: f64| -> Symbol { format!("{:?}", a).into() });

    }

    fn make_expr(&self, _egraph: &EGraph, termdag: &mut TermDag, value: Value) -> CostSet {
        assert!(value.tag == self.name());
        let term = termdag.lit(Literal::F64(OrderedFloat(f64::from_bits(value.bits))));
        let costs = vec![(value, 1)].into_iter().collect();
        CostSet {
            total: 1,
            costs,
            term,
        }
    }
}

impl IntoSort for f64 {
    type Sort = F64Sort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: self.to_bits(),
        })
    }
}

impl FromSort for f64 {
    type Sort = F64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        f64::from_bits(value.bits)
    }
}
