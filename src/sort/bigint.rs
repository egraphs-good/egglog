use super::*;

lazy_static! {
    static ref BIG_INT_SORT_NAME: Symbol = "BigInt".into();
    static ref INTS: Mutex<IndexSet<Z>> = Default::default();
}

#[derive(Debug)]
pub struct BigIntSort;

impl Sort for BigIntSort {
    fn name(&self) -> Symbol {
        *BIG_INT_SORT_NAME
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<Z>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<Z>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "bigint" = |a: i64| -> Z { a.into() });

        add_primitive!(eg, "+" = |a: Z, b: Z| -> Z { a + b });
        add_primitive!(eg, "-" = |a: Z, b: Z| -> Z { a - b });
        add_primitive!(eg, "*" = |a: Z, b: Z| -> Z { a * b });
        add_primitive!(eg, "/" = |a: Z, b: Z| -?> Z { (b != BigInt::ZERO).then(|| a / b) });
        add_primitive!(eg, "%" = |a: Z, b: Z| -?> Z { (b != BigInt::ZERO).then(|| a % b) });

        add_primitive!(eg, "&" = |a: Z, b: Z| -> Z { a & b });
        add_primitive!(eg, "|" = |a: Z, b: Z| -> Z { a | b });
        add_primitive!(eg, "^" = |a: Z, b: Z| -> Z { a ^ b });
        add_primitive!(eg, "<<" = |a: Z, b: i64| -> Z { a.shl(b) });
        add_primitive!(eg, ">>" = |a: Z, b: i64| -> Z { a.shr(b) });
        add_primitive!(eg, "not-Z" = |a: Z| -> Z { !a });

        add_primitive!(eg, "bits" = |a: Z| -> Z { a.bits().into() });

        add_primitive!(eg, "<" = |a: Z, b: Z| -?> () { (a < b).then_some(()) });
        add_primitive!(eg, ">" = |a: Z, b: Z| -?> () { (a > b).then_some(()) });
        add_primitive!(eg, "<=" = |a: Z, b: Z| -?> () { (a <= b).then_some(()) });
        add_primitive!(eg, ">=" = |a: Z, b: Z| -?> () { (a >= b).then_some(()) });

        add_primitive!(eg, "bool-=" = |a: Z, b: Z| -> bool { a == b });
        add_primitive!(eg, "bool-<" = |a: Z, b: Z| -> bool { a < b });
        add_primitive!(eg, "bool->" = |a: Z, b: Z| -> bool { a > b });
        add_primitive!(eg, "bool-<=" = |a: Z, b: Z| -> bool { a <= b });
        add_primitive!(eg, "bool->=" = |a: Z, b: Z| -> bool { a >= b });

        add_primitive!(eg, "min" = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitive!(eg, "max" = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitive!(eg, "to-string" = |a: Z| -> Symbol { a.to_string().into() });
        add_primitive!(eg, "from-string" = |a: Symbol| -?> Z { a.as_str().parse::<Z>().ok() });
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<Z>())
    }

    fn reconstruct_termdag_leaf(
        &self,
        primitives: &core_relations::Primitives,
        value: &core_relations::Value,
        termdag: &mut TermDag,
    ) -> Term {
        let bigint = primitives.unwrap_ref::<BigInt>(*value);

        let as_string = termdag.lit(Literal::String(bigint.to_string().into()));
        termdag.app("from_string".into(), vec![as_string])
    }
}

impl IntoSort for Z {
    type Sort = BigIntSort;
}
