use super::*;

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

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<F>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<F>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "+" = |a: F, b: F| -> F { a + b });
        add_primitive!(eg, "-" = |a: F, b: F| -> F { a - b });
        add_primitive!(eg, "*" = |a: F, b: F| -> F { a * b });
        add_primitive!(eg, "/" = |a: F, b: F| -?> F { (b != 0.0).then(|| a / b) });
        add_primitive!(eg, "%" = |a: F, b: F| -?> F { (b != 0.0).then(|| a % b) });
        add_primitive!(eg, "^" = |a: F, b: F| -> F { OrderedFloat(a.powf(*b)) });
        add_primitive!(eg, "neg" = |a: F| -> F { -a });

        add_primitive!(eg, "<" = |a: F, b: F| -?> () { (a < b).then(|| ()) });
        add_primitive!(eg, ">" = |a: F, b: F| -?> () { (a > b).then(|| ()) });
        add_primitive!(eg, "<=" = |a: F, b: F| -?> () { (a <= b).then(|| ()) });
        add_primitive!(eg, ">=" = |a: F, b: F| -?> () { (a >= b).then(|| ()) });

        add_primitive!(eg, "min" = |a: F, b: F| -> F { a.min(b) });
        add_primitive!(eg, "max" = |a: F, b: F| -> F { a.max(b) });
        add_primitive!(eg, "abs" = |a: F| -> F { a.abs() });

        // `to-f64` should be in `i64.rs`, but `F64Sort` wouldn't exist yet
        add_primitive!(eg, "to-f64" = |a: i64| -> F { OrderedFloat(a as f64) });
        add_primitive!(eg, "to-i64" = |a: F| -> i64 { a.0 as i64 });
        // Use debug instead of to_string so that decimal place is always printed
        add_primitive!(eg, "to-string" = |a: F| -> S { format!("{:?}", a.0).into() });
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

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<F>())
    }

    fn reconstruct_termdag_leaf(
        &self,
        primitives: &core_relations::Primitives,
        value: &core_relations::Value,
        termdag: &mut TermDag,
    ) -> Term {
        let f = primitives.unwrap_ref::<F>(*value);

        termdag.lit(Literal::Float(*f))
    }
}

impl IntoSort for F {
    type Sort = F64Sort;
    fn store(self, _sort: &Self::Sort) -> Value {
        Value {
            #[cfg(debug_assertions)]
            tag: F64Sort.name(),
            bits: self.to_bits(),
        }
    }
}

impl FromSort for F {
    type Sort = F64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        OrderedFloat(f64::from_bits(value.bits))
    }
}
