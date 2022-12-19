use crate::ast::Literal;

use super::*;
use ordered_float::OrderedFloat;

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

        add_primitives!(eg, "assert-eq" = |a: F64, b: F64| -> F64 { 
            if a == b {
                a
            } else {
                panic!("assertion failed: {:?} != {:?}", a, b);
            }
         });

        add_primitives!(eg, "+" = |a: F64, b: F64| -> F64 { a + b });
        add_primitives!(eg, "-" = |a: F64, b: F64| -> F64 { a - b });
        add_primitives!(eg, "*" = |a: F64, b: F64| -> F64 { a * b });
        add_primitives!(eg, "/" = |a: F64, b: F64| -> F64 { a / b });

        add_primitives!(eg, "<" = |a: F64, b: F64| -> Opt { if a < b {
            Some(())
        } else {
            None
        } });
        add_primitives!(eg, ">" = |a: F64, b: F64| -> Opt { 
          if a > b {
            Some(())
        } else {
            None
        }
         });

        add_primitives!(eg, "min" = |a: F64, b: F64| -> F64 { a.min(b) }); 
        add_primitives!(eg, "max" = |a: F64, b: F64| -> F64 { a.max(b) });

        add_primitives!(eg, "sqrt" = |a: F64| -> F64 { 
            OrderedFloat(a.sqrt())
});
        add_primitives!(eg, "ln" = |a: F64| -> F64 { OrderedFloat(a.ln()) });
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        Expr::Lit(Literal::Float(OrderedFloat(f64::from_bits(value.bits))))
    }
}

impl IntoSort for F64 {
    type Sort = F64Sort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: self.to_bits(),
        })
    }
}

impl FromSort for F64 {
    type Sort = F64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        OrderedFloat(f64::from_bits(value.bits))
    }
}
