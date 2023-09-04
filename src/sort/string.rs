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

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        assert!(value.tag == self.name);
        let sym = Symbol::from(NonZeroU32::new(value.bits as _).unwrap());
        (1, Expr::Lit(Literal::String(sym)))
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(Add {
            name: "+".into(),
            string: self,
        });
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

struct Add {
    name: Symbol,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Add {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if types.iter().all(|t| t.name() == self.string.name) {
            Some(self.string.clone())
        } else {
            None
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let mut res_string: String = "".to_owned();
        for value in values {
            let sym = Symbol::load(&self.string, value);
            res_string.push_str(sym.as_str());
        }
        let res_symbol: Symbol = res_string.into();
        Some(Value::from(res_symbol))
    }
}
