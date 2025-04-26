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

    fn extract_term(
        &self,
        _egraph: &EGraph,
        _value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        Some((1, termdag.lit(Literal::Unit)))
    }
    
    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<()>())
    }
}

impl IntoSort for () {
    type Sort = UnitSort;

    fn store(self, _sort: &Self::Sort) -> Value {
        Value::unit()
    }
}
