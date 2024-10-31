use super::*;
use crate::{ast::Literal, constraint::AllEqualTypeConstraint, ArcSort, PrimitiveLike};

#[derive(Debug)]
pub struct UnitSort;

lazy_static! {
    static ref UNIT_SORT_NAME: String = "Unit".into();
}

impl Sort for UnitSort {
    fn name(&self) -> String {
        UNIT_SORT_NAME.clone()
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn register_primitives(self: Arc<Self>, type_info: &mut TypeInfo) {
        type_info.add_primitive(NotEqualPrimitive { unit: self })
    }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        #[cfg(not(debug_assertions))]
        let _ = value;

        (1, GenericExpr::Lit(DUMMY_SPAN.clone(), Literal::Unit))
    }
}

impl IntoSort for () {
    type Sort = UnitSort;

    fn store(self, _sort: &Self::Sort) -> Option<Value> {
        Some(Value::unit())
    }
}

pub struct NotEqualPrimitive {
    unit: ArcSort,
}

impl PrimitiveLike for NotEqualPrimitive {
    fn name(&self) -> String {
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
