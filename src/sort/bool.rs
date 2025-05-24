use super::*;

#[derive(Debug)]
pub struct BoolSort;

lazy_static! {
    static ref BOOL_SORT_NAME: Symbol = "bool".into();
}

impl Sort for BoolSort {
    fn name(&self) -> Symbol {
        *BOOL_SORT_NAME
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<bool>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<bool>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "not" = |a: bool| -> bool { !a });
        add_primitive!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        add_primitive!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        add_primitive!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_primitive!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<bool>())
    }

    fn reconstruct_termdag_leaf(
        &self,
        primitives: &Primitives,
        value: &Value,
        termdag: &mut TermDag,
    ) -> Term {
        let b = primitives.unwrap_ref::<bool>(*value);

        termdag.lit(Literal::Bool(*b))
    }
}

impl IntoSort for bool {
    type Sort = BoolSort;
}
