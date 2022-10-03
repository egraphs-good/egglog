use num_integer::Roots;
use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};
use std::sync::Mutex;
use intervals_good::Interval;

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

        // TODO we can't have primitives take borrows just yet, since it
        // requires returning a reference to the locked sort
        add_primitives!(eg, "+" = |a: R, b: R| -> R { a.add(&b) });
        add_primitives!(eg, "-" = |a: R, b: R| -> R { a.sub(&b) });
        add_primitives!(eg, "*" = |a: R, b: R| -> R { a.mul(&b) });
        add_primitives!(eg, "/" = |a: R, b: R| -> R { a.div(&b) });

        add_primitives!(eg, "min" = |a: R, b: R| -> R { a.fmin(b) });
        add_primitives!(eg, "max" = |a: R, b: R| -> R { a.fmax(b) });
        add_primitives!(eg, "neg" = |a: R| -> R { a.neg() });
        add_primitives!(eg, "abs" = |a: R| -> R { a.fabs() });
        add_primitives!(eg, "floor" = |a: R| -> R { a.floor() });
        add_primitives!(eg, "ceil" = |a: R| -> R { a.ceil() });
        add_primitives!(eg, "round" = |a: R| -> R { a.round() });
        add_primitives!(eg, "interval" = |a: i64, b: i64| -> R { R::new(53, a as f64, b as f64) });

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

        add_primitives!(eg, "<" = |a: R, b: R| -> Opt { (a.hi < b.lo).then(|| ()) }); 
        add_primitives!(eg, ">" = |a: R, b: R| -> Opt { (a.lo > b.hi).then(|| ()) }); 
    }
    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        let rat = R::load(self, &value);
        let left = rat.lo.floor();
        let denom = rat.hi;
        Expr::call(
            "interval",
            vec![
                Expr::Lit(Literal::Int(numer)),
                Expr::Lit(Literal::Int(denom)),
            ],
        )
    }
}

impl FromSort for R {
    type Sort = IntervalSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        *sort.rats.lock().unwrap().get_index(i).unwrap()
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
