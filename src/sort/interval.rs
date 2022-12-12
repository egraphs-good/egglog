use intervals_good::{ErrorInterval, Interval};
use rug::{float::Round, ops::*, Float, Rational};
use std::sync::Mutex;
use ordered_float::OrderedFloat;

// 53 is double precision
pub(crate) const INTERVAL_PRECISION: u32 = 200;

type R = Interval;
use crate::{ast::Literal, util::IndexSet};

use super::*;

#[derive(Debug)]
pub struct IntervalSort {
    name: Symbol,
    rats: Mutex<IndexSet<R>>,
}

impl IntervalSort {
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            rats: Default::default(),
        }
    }
}

impl Sort for IntervalSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "+" = |a: R, b: R| -> R { a.add(&b) });
        add_primitives!(eg, "-" = |a: R, b: R| -> R { a.sub(&b) });
        add_primitives!(eg, "*" = |a: R, b: R| -> R { a.mul(&b) });
        add_primitives!(eg, "/" = |a: R, b: R| -> R { a.div(&b) });

        add_primitives!(eg, "min" = |a: R, b: R| -> R { a.fmin(&b) });
        add_primitives!(eg, "max" = |a: R, b: R| -> R { a.fmax(&b) });
        add_primitives!(eg, "neg" = |a: R| -> R { a.neg() });
        add_primitives!(eg, "abs" = |a: R| -> R { a.fabs() });
        add_primitives!(eg, "floor" = |a: R| -> R { a.floor() });
        add_primitives!(eg, "ceil" = |a: R| -> R { a.ceil() });
        add_primitives!(eg, "round" = |a: R| -> R { a.round() });
        add_primitives!(eg, "interval" = |a: F64, b: F64| -> R { R::new(INTERVAL_PRECISION, a.into_inner(), b.into_inner()) });
        add_primitives!(eg, "interval" = |a: Rational, b: Rational| -> R {
            if (true) {
                let mut lo = Float::with_val(INTERVAL_PRECISION, 0.0);
                let mut hi = Float::with_val(INTERVAL_PRECISION, 0.0);
                lo.add_from_round(a, Round::Down);
                hi.add_from_round(b, Round::Up);
                Interval::make(lo, hi, ErrorInterval {
                    lo: false,
                    hi: false,
                })
        } else {
            panic!("TODO fix macro");
        }
        });

        add_primitives!(eg, "pow" = |a: R, b: R| -> R {
            a.pow(&b)
        });
        add_primitives!(eg, "ln" = |a: R| -> R {
            a.ln()
        });
        add_primitives!(eg, "sqrt" = |a: R| -> R {
            a.sqrt()
        });
        add_primitives!(eg, "cbrt" = |a: R| -> R {
            a.cbrt()
        });

        add_primitives!(eg, "intersect" = |a: R, b: R| -> Opt<R> {
            if true {
            let loF: Float = a.lo.clone().into();
            let hiF: Float = a.hi.clone().into();
            let lo = loF.max(b.lo.as_float());
            let hi = hiF.min(b.hi.as_float());
            if lo > hi {
                None
            } else {
                Some(Interval::make(lo, hi, a.err.union(&b.err)))
            }
        } else {
            None
        }});
        add_primitives!(eg, "interval-pi" = | | -> R {
            Interval::pi(INTERVAL_PRECISION)
        });
        add_primitives!(eg, "interval-e" = | | -> R {
            Interval::e(INTERVAL_PRECISION)
        });
    }
    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        let rat = R::load(self, &value);
        let left = rat.lo;
        let right = rat.hi;
        Expr::call(
            "interval",
            vec![
                Expr::Lit(Literal::Float(OrderedFloat(left.as_float().to_f64()))),
                Expr::Lit(Literal::Float(OrderedFloat(right.as_float().to_f64()))),
            ],
        )
    }
}

impl FromSort for R {
    type Sort = IntervalSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        sort.rats.lock().unwrap().get_index(i).unwrap().clone()
    }
}

impl IntoSort for R {
    type Sort = IntervalSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let (i, _) = sort.rats.lock().unwrap().insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}
