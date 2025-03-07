use super::*;
use crate::{ast::Literal, constraint::AllEqualTypeConstraint, ArcSort, PrimitiveLike};

#[derive(Debug)]
pub struct UnitSort;

lazy_static! {
    static ref UNIT_SORT_NAME: Symbol = "Unit".into();
}

impl Sort for UnitSort {
    fn name(&self) -> Symbol {
        *UNIT_SORT_NAME
    }

    fn column_ty(&self, prims: &core_relations::Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<()>())
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn register_primitives(self: Arc<Self>, type_info: &mut TypeInfo) {
        type_info.add_primitive(NotEqualPrimitive { unit: self })
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

pub struct NotEqualPrimitive {
    unit: ArcSort,
}

impl PrimitiveLike for NotEqualPrimitive {
    fn name(&self) -> Symbol {
        "!=".into()
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_exact_length(3)
            .with_output_sort(self.unit.clone())
            .into_box()
    }

    fn apply(
        &self,
        values: &[Value],
        _sorts: (&[ArcSort], &ArcSort),
        _egraph: Option<&mut EGraph>,
    ) -> Option<Value> {
        (values[0] != values[1]).then(Value::unit)
    }
}
