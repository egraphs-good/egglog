use std::num::NonZeroU32;

use crate::ast::Literal;

use super::*;

#[derive(Debug)]
pub struct StringSort {
    name: Symbol,
}

impl StringSort {
    pub fn new(name: Symbol) -> Self {
        Self { name }
    }
}

impl Sort for StringSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> Expr {
        assert!(value.tag == self.name);
        let sym = Symbol::from(NonZeroU32::new(value.bits as _).unwrap());
        Expr::Lit(Literal::String(sym))
    }
}

// TODO could use a local symbol table

impl IntoSort for Symbol {
    type Sort = StringSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: NonZeroU32::from(self).get() as _,
        })
    }
}

impl FromSort for Symbol {
    type Sort = StringSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        NonZeroU32::new(value.bits as u32).unwrap().into()
    }
}
