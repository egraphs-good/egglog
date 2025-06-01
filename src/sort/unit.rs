use super::*;

#[derive(Debug)]
pub struct UnitSort;

lazy_static! {
    static ref UNIT_SORT_NAME: Symbol = "Unit".into();
}

impl Sort for UnitSort {
    fn name(&self) -> Symbol {
        *UNIT_SORT_NAME
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<()>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<()>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<()>())
    }

    fn reconstruct_termdag_leaf(
        &self,
        _primitives: &Primitives,
        _value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        termdag.lit(Literal::Unit)
    }
}
