use super::*;
use crate::{ast::Literal, constraint::AllEqualTypeConstraint, ArcSort, PrimitiveLike};

#[derive(Debug)]
pub struct UnitSort {
    name: Symbol,
}

impl UnitSort {
    pub fn new(name: Symbol) -> Self {
        Self { name }
    }
}

impl Sort for UnitSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn register_primitives(self: Arc<Self>, type_info: &mut TypeInfo) {
        type_info.add_primitive(NotEqualPrimitive { unit: self })
    }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        assert_eq!(value.tag, self.name);
        (1, Expr::Lit((), Literal::Unit))
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
    fn name(&self) -> Symbol {
        "!=".into()
    }

    fn get_type_constraints(&self) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name())
            .with_exact_length(3)
            .with_output_sort(self.unit.clone())
            .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        (values[0] != values[1]).then(Value::unit)
    }
}
