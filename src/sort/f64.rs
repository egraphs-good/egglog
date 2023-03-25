use super::*;
use crate::ast::Literal;
use ordered_float::OrderedFloat;
use std::f64::consts::PI;

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

    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        type Opt<T = ()> = Option<T>;

        add_primitives!(
            eg,
            "assert-eq" = |a: f64, b: f64| -> f64 {
                if a == b || (a.is_nan() && b.is_nan()) {
                    a
                } else {
                    panic!("assertion failed: {:?} != {:?}", a, b);
                }
            }
        );

        add_primitives!(eg, "sqrt" = |a: f64| -> f64 { a.sqrt() });
        add_primitives!(eg, "ln" = |a: f64| -> f64 { a.ln() });

        add_primitives!(
            eg,
            "rel-error" = |a: i64, b: i64| -> f64 {
                if true {
                    let a = a as f64;
                    let b = b as f64;
                    if b == 0.0 {
                        if a == 0.0 {
                            0.0
                        } else {
                            // fall back to absolute error
                            (a - b).abs()
                        }
                    } else {
                        (a - b).abs() / b.abs()
                    }
                } else {
                    panic!("TODO");
                }
            }
        );

        // calculate relative error
        add_primitives!(
            eg,
            "rel-error" = |a: f64, b: f64| -> f64 {
                if a.is_nan() && b.is_nan() {
                    0.0
                } else if a.is_nan() || b.is_nan() {
                    f64::INFINITY
                } else if b == 0.0 {
                    if a == 0.0 {
                        0.0
                    } else {
                        // fall back to absolute error
                        (a - b).abs()
                    }
                } else {
                    (a - b).abs() / b.abs()
                }
            }
        );
        add_primitives!(
            eg,
            "abs-error" = |a: f64, b: f64| -> f64 {
                if a.is_nan() || b.is_nan() {
                    f64::INFINITY
                } else {
                    (a - b).abs()
                }
            }
        );
        add_primitives!(
            eg,
            "furthest-from" = |a: f64| -> f64 {
                if a < 0.0 {
                    std::f64::INFINITY
                } else {
                    std::f64::NEG_INFINITY
                }
            }
        );

        add_primitives!(eg, "f64-PI" = || -> f64 { PI });
        add_primitives!(eg, "f64-E" = || -> f64 { std::f64::consts::E });
        add_primitives!(eg, "f64-NAN" = || -> f64 { std::f64::NAN });
        add_primitives!(eg, "f64-INFINITY" = || -> f64 { std::f64::INFINITY });

        add_primitives!(eg, "f64-Neg" = |a: f64| -> f64 { -a });

        add_primitives!(eg, "f64-Sqrt" = |a: f64| -> f64 { a.sqrt() });
        add_primitives!(eg, "f64-Cbrt" = |a: f64| -> f64 { a.cbrt() });
        add_primitives!(eg, "f64-Fabs" = |a: f64| -> f64 { a.abs() });
        add_primitives!(eg, "f64-Ceil" = |a: f64| -> f64 { a.ceil() });
        add_primitives!(eg, "f64-Floor" = |a: f64| -> f64 { a.floor() });
        add_primitives!(eg, "f64-Round" = |a: f64| -> f64 { a.round() });
        add_primitives!(eg, "f64-Fmod" = |a: f64, b: f64| -> f64 { a % b });
        add_primitives!(eg, "f64-Log" = |a: f64| -> f64 { a.ln() });
        add_primitives!(eg, "f64-Exp" = |a: f64| -> f64 { a.exp() });
        add_primitives!(eg, "f64-Sin" = |a: f64| -> f64 { a.sin() });
        add_primitives!(eg, "f64-Cos" = |a: f64| -> f64 { a.cos() });
        add_primitives!(eg, "f64-Tan" = |a: f64| -> f64 { a.tan() });
        add_primitives!(eg, "f64-Asin" = |a: f64| -> f64 { a.asin() });
        add_primitives!(eg, "f64-Acos" = |a: f64| -> f64 { a.acos() });
        add_primitives!(eg, "f64-Atan" = |a: f64| -> f64 { a.atan() });
        add_primitives!(eg, "f64-Sinh" = |a: f64| -> f64 { a.sinh() });
        add_primitives!(eg, "f64-Cosh" = |a: f64| -> f64 { a.cosh() });
        add_primitives!(eg, "f64-Tanh" = |a: f64| -> f64 { a.tanh() });
        add_primitives!(eg, "f64-Asinh" = |a: f64| -> f64 { a.asinh() });
        add_primitives!(eg, "f64-Acosh" = |a: f64| -> f64 { a.acosh() });
        add_primitives!(eg, "f64-Atanh" = |a: f64| -> f64 { a.atanh() });
        add_primitives!(eg, "f64-Expm1" = |a: f64| -> f64 { a.exp_m1() });
        add_primitives!(eg, "f64-Log1p" = |a: f64| -> f64 { a.ln_1p() });

        add_primitives!(eg, "f64-Add" = |a: f64, b: f64| -> f64 { a + b });
        add_primitives!(eg, "f64-Sub" = |a: f64, b: f64| -> f64 { a - b });
        add_primitives!(eg, "f64-Mul" = |a: f64, b: f64| -> f64 { a * b });
        add_primitives!(eg, "f64-Div" = |a: f64, b: f64| -> f64 { a / b });
        add_primitives!(eg, "f64-Pow" = |a: f64, b: f64| -> f64 { a.powf(b) });
        add_primitives!(eg, "f64-Atan2" = |a: f64, b: f64| -> f64 { a.atan2(b) });
        add_primitives!(eg, "f64-Hypot" = |a: f64, b: f64| -> f64 { a.hypot(b) });
        add_primitives!(
            eg,
            "f64-Fma" = |a: f64, b: f64, c: f64| -> f64 { a.mul_add(b, c) }
        );
        add_primitives!(
            eg,
            "f64-Copysign" = |a: f64, b: f64| -> f64 { a.copysign(b) }
        );

        add_primitives!(eg, "+" = |a: f64, b: f64| -> f64 { a + b });
        add_primitives!(eg, "-" = |a: f64, b: f64| -> f64 { a - b });
        add_primitives!(eg, "*" = |a: f64, b: f64| -> f64 { a * b });
        add_primitives!(
            eg,
            "/" = |a: f64, b: f64| -> Opt<f64> { (b != 0.0).then(|| a / b) }
        );
        add_primitives!(
            eg,
            "%" = |a: f64, b: f64| -> Opt<f64> { (b != 0.0).then(|| a % b) }
        );

        add_primitives!(eg, "<" = |a: f64, b: f64| -> Opt { (a < b).then(|| ()) });
        add_primitives!(eg, ">" = |a: f64, b: f64| -> Opt { (a > b).then(|| ()) });
        add_primitives!(eg, "<=" = |a: f64, b: f64| -> Opt { (a <= b).then(|| ()) });
        add_primitives!(eg, ">=" = |a: f64, b: f64| -> Opt { (a >= b).then(|| ()) });

        add_primitives!(eg, "min" = |a: f64, b: f64| -> f64 { a.min(b) });
        add_primitives!(eg, "max" = |a: f64, b: f64| -> f64 { a.max(b) });
    }

    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        Expr::Lit(Literal::F64(OrderedFloat(f64::from_bits(value.bits))))
    }
}

impl IntoSort for f64 {
    type Sort = F64Sort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        Some(Value {
            tag: sort.name,
            bits: self.to_bits(),
        })
    }
}

impl FromSort for f64 {
    type Sort = F64Sort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        f64::from_bits(value.bits)
    }
}
