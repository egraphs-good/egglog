use crate::ast::Literal;

use super::*;

#[derive(Debug)]
pub struct F64Sort {
    name: Symbol,
}

impl F64Sort {
    pub fn new(name: Symbol) -> Self {
        Self { name }
    }
}

impl Sort for F64Sort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "+" = |a: F64, b: F64| -> F64 { F64::new(a.value + b.value) });
        add_primitives!(eg, "-" = |a: F64, b: F64| -> F64 { F64::new(a.value - b.value) });
        add_primitives!(eg, "*" = |a: F64, b: F64| -> F64 { F64::new(a.value * b.value) });
        add_primitives!(eg, "/" = |a: F64, b: F64| -> F64 { F64::new(a.value / b.value) });

        add_primitives!(eg, "<" = |a: F64, b: F64| -> Opt { if (a.value < b.value) {
            Some(())
        } else {
            None
        } });
        add_primitives!(eg, ">" = |a: F64, b: F64| -> Opt { 
          if (a.value > b.value) {
            Some(())
        } else {
            None
        }
         });

        add_primitives!(eg, "min" = |a: F64, b: F64| -> F64 { F64::new(a.value.min(b.value)) }); 
        add_primitives!(eg, "max" = |a: F64, b: F64| -> F64 { F64::new(a.value.max(b.value)) });

        add_primitives!(eg, "sqrt" = |a: F64| -> F64 { 
            F64::new(a.value.sqrt())
});
        add_primitives!(eg, "ln" = |a: F64| -> F64 { F64::new(a.value.ln()) });
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        Expr::Lit(Literal::Int(value.bits as _))
    }
}

impl IntoSort for F64 {
    type Sort = F64Sort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: self.value.to_bits(),
        })
    }
}

impl FromSort for F64 {
    type Sort = F64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        F64::new(f64::from_bits(value.bits))
    }
}
