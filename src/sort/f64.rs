use std::f64::consts::PI;

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

        add_primitives!(eg, "dist" = |a: F64, b: F64| -> F64 {
            OrderedFloat((a - b).abs())
        });
        
        add_primitives!(eg, "f64-PI" = | | -> F64 {
            OrderedFloat(PI)
        });
        add_primitives!(eg, "f64-E" = | | -> F64 {
            OrderedFloat(std::f64::consts::E)
        });
        add_primitives!(eg, "f64-INFINITY" = | | -> F64 {
            OrderedFloat(std::f64::INFINITY)
        });

        add_primitives!(eg, "f64-Neg" = |a: F64| -> F64 {
            -a
        });

        add_primitives!(eg, "f64-Sqrt" = |a: F64| -> F64 {
            OrderedFloat(a.sqrt())
        });
        add_primitives!(eg, "f64-Cbrt" = |a: F64| -> F64 {
            OrderedFloat(a.cbrt())
        });
        add_primitives!(eg, "f64-Fabs" = |a: F64| -> F64 {
            OrderedFloat(a.abs())
        });
        add_primitives!(eg, "f64-Ceil" = |a: F64| -> F64 {
            OrderedFloat(a.ceil())
        });
        add_primitives!(eg, "f64-Floor" = |a: F64| -> F64 {
            OrderedFloat(a.floor())
        });
        add_primitives!(eg, "f64-Round" = |a: F64| -> F64 {
            OrderedFloat(a.round())
        });
        add_primitives!(eg, "f64-Log" = |a: F64| -> F64 {
            OrderedFloat(a.ln())
        });
        add_primitives!(eg, "f64-Exp" = |a: F64| -> F64 {
            OrderedFloat(a.exp())
        });
        add_primitives!(eg, "f64-Sin" = |a: F64| -> F64 {
            OrderedFloat(a.sin())
        });
        add_primitives!(eg, "f64-Cos" = |a: F64| -> F64 {
            OrderedFloat(a.cos())
        });
        add_primitives!(eg, "f64-Tan" = |a: F64| -> F64 {
            OrderedFloat(a.tan())
        });
        add_primitives!(eg, "f64-Asin" = |a: F64| -> F64 {
            OrderedFloat(a.asin())
        });
        add_primitives!(eg, "f64-Acos" = |a: F64| -> F64 {
            OrderedFloat(a.acos())
        });
        add_primitives!(eg, "f64-Atan" = |a: F64| -> F64 {
            OrderedFloat(a.atan())
        });
        add_primitives!(eg, "f64-Sinh" = |a: F64| -> F64 {
            OrderedFloat(a.sinh())
        });
        add_primitives!(eg, "f64-Cosh" = |a: F64| -> F64 {
            OrderedFloat(a.cosh())
        });
        add_primitives!(eg, "f64-Tanh" = |a: F64| -> F64 {
            OrderedFloat(a.tanh())
        });
        add_primitives!(eg, "f64-Atanh" = |a: F64| -> F64 {
            OrderedFloat(a.atanh())
        });
        add_primitives!(eg, "f64-Expm1" = |a: F64| -> F64 {
            OrderedFloat(a.exp_m1())
        });
        add_primitives!(eg, "f64-Log1p" = |a: F64| -> F64 {
            OrderedFloat(a.ln_1p())
        });

        add_primitives!(eg, "f64-Add" = |a: F64, b: F64| -> F64 { a + b });
        add_primitives!(eg, "f64-Sub" = |a: F64, b: F64| -> F64 { a - b });
        add_primitives!(eg, "f64-Mul" = |a: F64, b: F64| -> F64 { a * b });
        add_primitives!(eg, "f64-Div" = |a: F64, b: F64| -> F64 { a / b });
        add_primitives!(eg, "f64-Pow" = |a: F64, b: F64| -> F64 { OrderedFloat(a.into_inner().powf(b.into_inner())) });
        add_primitives!(eg, "f64-Atan2" = |a: F64, b: F64| -> F64 { OrderedFloat(a.into_inner().atan2(b.into_inner())) });
        add_primitives!(eg, "f64-Hypot" = |a: F64, b: F64| -> F64 { OrderedFloat(a.into_inner().hypot(b.into_inner())) });
        add_primitives!(eg, "f64-Fma" = |a: F64, b: F64, c: F64| -> F64 { OrderedFloat(a.into_inner().mul_add(b.into_inner(), c.into_inner())) });
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
