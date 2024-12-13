use crate::ast::Literal;

use super::*;

#[derive(Debug)]
pub struct BoolSort;

lazy_static! {
    static ref BOOL_SORT_NAME: Symbol = "bool".into();
}

impl Sort for BoolSort {
    fn name(&self) -> Symbol {
        *BOOL_SORT_NAME
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

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        Some((1, termdag.lit(Literal::Bool(value.bits > 0))))
    }
}

impl IntoSort for bool {
    type Sort = BoolSort;
    fn store(self, _sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            #[cfg(debug_assertions)]
            tag: BoolSort.name(),
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
