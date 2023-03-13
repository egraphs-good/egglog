use crate::ast::Literal;

use super::*;
use ordered_float::OrderedFloat;

#[derive(Debug)]
pub struct BoolSort {
    name: Symbol,
}

impl BoolSort {
    pub fn new(name: Symbol) -> Self {
        Self { name }
    }
}

impl Sort for BoolSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "assert-eq" = |a: bool, b: bool| -> bool { 
            if a == b {
                a
            } else {
                panic!("assertion failed: {:?} != {:?}", a, b);
            }
         });

        add_primitives!(eg, "abs-error" = |a: bool, b: bool| -> f64 {
            if a == b {
              0.0
            } else {
              1.0
            }
        });
        add_primitives!(eg, "rel-error" = |a: bool, b: bool| -> f64 {
            if a == b {
              0.0
            } else {
              1.0
            }
        });

        add_primitives!(eg, "furthest-from" = |a: bool| -> bool {
            !a
        });

        add_primitives!(eg, "f64-If" = |a: bool, b: f64, c: f64| -> f64 {
            if a {
              b
            } else {
              c
            }
        });

        add_primitives!(eg, "bool-And" = |a: bool, b: bool| -> bool { a && b });
        add_primitives!(eg, "bool-Or" = |a: bool, b: bool| -> bool { a || b });
        add_primitives!(eg, "bool-Not" = |a: bool| -> bool { !a });

        add_primitives!(eg, "bool-Less" = |a: f64, b: f64| -> bool { OrderedFloat(a) < OrderedFloat(b) });
        add_primitives!(eg, "bool-LessEq" = |a: f64, b: f64| -> bool { OrderedFloat(a) <= OrderedFloat(b) });
        add_primitives!(eg, "bool-Greater" = |a: f64, b: f64| -> bool { OrderedFloat(a) > OrderedFloat(b) });
        add_primitives!(eg, "bool-GreaterEq" = |a: f64, b: f64| -> bool { OrderedFloat(a) >= OrderedFloat(b) });
        add_primitives!(eg, "bool-Eq" = |a: f64, b: f64| -> bool { OrderedFloat(a) == OrderedFloat(b) });
        add_primitives!(eg, "bool-NotEq" = |a: f64, b: f64| -> bool { OrderedFloat(a) != OrderedFloat(b) });

        add_primitives!(eg, "bool-TRUE" = | | -> bool { true });
        add_primitives!(eg, "bool-FALSE" = | | -> bool { false });
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        Expr::Lit(Literal::Bool(value.bits > 0))
    }
}

impl IntoSort for bool {
    type Sort = BoolSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: self as u64,
        })
    }
}

impl FromSort for bool {
    type Sort = BoolSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        value.bits != 0
    }
}
