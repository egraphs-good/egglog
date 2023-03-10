use intervals_good::{ErrorInterval, Interval};
use ordered_float::OrderedFloat;
use rug::{float::Round, ops::*, Float, Rational};
use std::sync::Mutex;

// 53 is double precision
pub(crate) const INTERVAL_PRECISION: u32 = 53;

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
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "to-f64" = |a: R| -> Opt<f64> {
            if a.err.lo && a.err.hi {
                // return NaN
                Some(f64::NAN)
            } else if a.err.lo || a.err.hi {
                None
            } else {
                let loF: Float = a.lo.clone().into();
                let hiF: Float = a.hi.into();
                let top = loF.to_f64();
                let bot = hiF.to_f64();
                if top == bot {
                    Some(top)
                } else {
                    None
                }
        }});

        add_primitives!(eg, "ival-Add" = |a: R, b: R| -> R { a.add(&b) });
        add_primitives!(eg, "ival-Sub" = |a: R, b: R| -> R { a.sub(&b) });
        add_primitives!(eg, "ival-Mul" = |a: R, b: R| -> R { a.mul(&b) });
        add_primitives!(eg, "ival-Div" = |a: R, b: R| -> R { a.div(&b) });

        add_primitives!(eg, "ival-Min" = |a: R, b: R| -> R { a.fmin(&b) });
        add_primitives!(eg, "ival-Max" = |a: R, b: R| -> R { a.fmax(&b) });
        add_primitives!(eg, "ival-Neg" = |a: R| -> R { a.neg() });
        add_primitives!(eg, "ival-Abs" = |a: R| -> R { a.fabs() });
        add_primitives!(eg, "ival-Floor" = |a: R| -> R { a.floor() });
        add_primitives!(eg, "ival-Ceil" = |a: R| -> R { a.ceil() });
        add_primitives!(eg, "ival-Round" = |a: R| -> R { a.round() });
        add_primitives!(eg, "ival-Sin" = |a: R| -> R { a.sin() });
        add_primitives!(eg, "ival-Cos" = |a: R| -> R { a.cos() });
        add_primitives!(eg, "ival-Tan" = |a: R| -> R { a.tan() });
        add_primitives!(eg, "ival-Asin" = |a: R| -> R { a.asin() });
        add_primitives!(eg, "ival-Acos" = |a: R| -> R { a.acos() });
        add_primitives!(eg, "ival-Atan" = |a: R| -> R { a.atan() });
        add_primitives!(eg, "ival-Sinh" = |a: R| -> R { a.sinh() });
        add_primitives!(eg, "ival-Cosh" = |a: R| -> R { a.cosh() });
        add_primitives!(eg, "ival-Tanh" = |a: R| -> R { a.tanh() });
        add_primitives!(eg, "ival-Atanh" = |a: R| -> R { a.atanh() });
        add_primitives!(eg, "ival-Atan2" = |a: R, b: R| -> R { a.atan2(&b) });
        add_primitives!(eg, "ival-Hypot" = |a: R, b: R| -> R { a.hypot(&b) });
        add_primitives!(eg, "ival-Asinh" = |a: R| -> R { a.asinh() });
        add_primitives!(eg, "ival-Acosh" = |a: R| -> R { a.acosh() });
        add_primitives!(eg, "ival-Copysign" = |a: R, b: R| -> R { a.copysign(&b) });
        add_primitives!(eg, "ival-Fmod" = |a: R, b: R| -> R { a.fmod(&b) });
        add_primitives!(eg, "interval" = |a: f64, b: f64| -> R { R::new(INTERVAL_PRECISION, a, b) });
        add_primitives!(eg, "interval" = |a: f64, b: f64, elo: bool, ehi: bool| -> R { 
            if true {
            let lo = Float::with_val(INTERVAL_PRECISION, a);
            let hi = Float::with_val(INTERVAL_PRECISION, b);
            Interval::make(lo, hi, ErrorInterval {
                lo: elo,
                hi: ehi,
            })
            } else {
                panic!("TODO fix macro");
            }
            });
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

        add_primitives!(eg, "ival-Pow" = |a: R, b: R| -> R {
            a.pow(&b)
        });
        add_primitives!(eg, "ival-Log" = |a: R| -> R {
            a.ln()
        });
        add_primitives!(eg, "ival-Sqrt" = |a: R| -> R {
            a.sqrt()
        });
        add_primitives!(eg, "ival-Cbrt" = |a: R| -> R {
            a.cbrt()
        });

        add_primitives!(eg, "ival-disjoint" = |a: R, b: R| -> Option<()> {
            if (a.err.lo && a.err.hi && !b.err.lo && !b.err.hi) ||
                   (b.err.lo && b.err.hi && !a.err.lo && !a.err.hi) {
                    Some(())

                } else {
                    let loF: Float = a.lo.clone().into();
                    let hiF: Float = a.hi.clone().into();
                    let lo = loF.max(b.lo.as_float());
                    let hi = hiF.min(b.hi.as_float());
                    if lo > hi {
                        Some(())
                    } else {
                        None
                    }
                }  
        });

        add_primitives!(eg, "intersect" = |a: R, b: R| -> Opt<R> {
            if true {
                // they disagree on guaranteed error and no error
                if (a.err.lo && a.err.hi && !b.err.lo && !b.err.hi) ||
                   (b.err.lo && b.err.hi && !a.err.lo && !a.err.hi) {
                    panic!("Intersect failed! Intervals: {:?} and {:?}", a, b);
                } else if (a.err.lo && a.err.hi) || (b.err.lo && b.err.hi) {
                    Some(Interval::make(Float::with_val(INTERVAL_PRECISION, f64::NAN), Float::with_val(INTERVAL_PRECISION, f64::NAN), ErrorInterval {
                        lo: true,
                        hi: true
                    }))
                } else {
                    let loF: Float = a.lo.clone().into();
                    let hiF: Float = a.hi.clone().into();
                    let lo = loF.max(b.lo.as_float());
                    let hi = hiF.min(b.hi.as_float());
                    if lo > hi {
                        panic!("Intersect failed! Intervals: {:?} and {:?}", a, b);
                    } else {
                        Some(Interval::make(lo, hi, 
                            ErrorInterval {
                            lo: a.err.lo || b.err.lo, // guarantee error if either lo or hi has error
                            hi: a.err.hi && b.err.hi  // possibility of erro
                            }))
                    }
                }
        } else {
            None
        }});
        add_primitives!(eg, "interval-Pi" = | | -> R {
            Interval::pi(INTERVAL_PRECISION)
        });
        add_primitives!(eg, "interval-E" = | | -> R {
            Interval::e(INTERVAL_PRECISION)
        });
        add_primitives!(eg, "interval-Inf" = | | -> R {
            Interval::inf(INTERVAL_PRECISION)
        });

        add_primitives!(eg, "ival-Fabs" = |a: R| -> R {
            a.fabs()
        });
        add_primitives!(eg, "ival-Floor" = |a: R| -> R {
            a.floor()
        });
        add_primitives!(eg, "ival-Ceil" = |a: R| -> R {
            a.ceil()
        });
        add_primitives!(eg, "ival-Round" = |a: R| -> R {
            a.round()
        });
        add_primitives!(eg, "ival-Exp" = |a: R| -> R {
            a.exp()
        });
        add_primitives!(eg, "ival-Expm1" = |a: R| -> R {
            a.exp_m1()
        });
        add_primitives!(eg, "ival-Log1p" = |a: R| -> R {
            a.ln_1p()
        });

        add_primitives!(eg, "ival-Fma" = |a: R, b: R, c: R| -> R {
            a.fma(&b, &c)
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
                Expr::Lit(Literal::F64(OrderedFloat(
                    left.as_float().to_f64_round(Round::Down),
                ))),
                Expr::Lit(Literal::F64(OrderedFloat(
                    right.as_float().to_f64_round(Round::Up),
                ))),
                Expr::Lit(Literal::Bool(rat.err.lo)),
                Expr::Lit(Literal::Bool(rat.err.hi)),
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
