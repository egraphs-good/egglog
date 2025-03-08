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

    fn column_ty(&self, prims: &Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<()>())
    }

    fn register_type(&self, prims: &mut Primitives) {
        prims.register_type::<()>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn register_primitives(self: Arc<Self>, type_info: &mut TypeInfo) {
        add_primitive!(type_info, "!=" = |a: #, b: #| -?> () {
            (a != b).then_some(())
        })
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
}

impl IntoSort for () {
    type Sort = UnitSort;

    fn store(self, _sort: &Self::Sort) -> Value {
        Value::unit()
    }
}
