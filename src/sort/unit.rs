use super::*;
use crate::{
    ast::Literal, constraint::ImpossibleConstraint, typecheck::Atom, ArcSort, PrimitiveLike,
};

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
        (1, Expr::Lit(Literal::Unit))
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

    fn get_constraints(&self, arguments: &[AtomTerm]) -> Vec<Constraint<AtomTerm, ArcSort>> {
        match arguments {
            [a, b, unit] => {
                let constraints = vec![
                    Constraint::Eq(a.clone(), b.clone()),
                    Constraint::Assign(unit.clone(), self.unit.clone()),
                ];
                constraints
            }
            _ => {
                vec![Constraint::Impossible(
                    ImpossibleConstraint::ArityMismatch {
                        atom: Atom {
                            head: self.name(),
                            args: arguments.to_vec(),
                        },
                        expected: 3,
                        actual: arguments.len(),
                    },
                )]
            }
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        (values[0] != values[1]).then(Value::unit)
    }
}
