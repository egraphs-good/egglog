use crate::ast::Literal;

use super::*;

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
        add_primitives!(eg, "not" = |a: bool| -> bool { !a });
        add_primitives!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        add_primitives!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        add_primitives!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_primitives!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
    }

    fn make_expr(&self, _egraph: &EGraph, termdag: &mut TermDag, value: Value) -> CostSet {
        assert!(value.tag == self.name());
        let term = termdag.lit(Literal::Bool(value.bits > 0));
        let costs = vec![(value, 1)].into_iter().collect();
        CostSet {
            total: 1,
            costs,
            term,
        }
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
