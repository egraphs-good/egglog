use super::*;
use crate::{ast::Literal, ArcSort, PrimitiveLike};

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

    fn register_primitives(self: Arc<Self>, egraph: &mut EGraph) {
        egraph.add_primitive(NotEqualPrimitive { unit: self })
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert_eq!(value.tag, self.name);
        Expr::Lit(Literal::Unit)
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

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [a, b] if a.name() == b.name() => Some(self.unit.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        (values[0] != values[1]).then(Value::unit)
    }
}
