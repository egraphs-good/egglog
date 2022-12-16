use intervals_good::{ErrorInterval, BooleanInterval, Interval};
use rug::{float::Round, ops::*, Float, Rational};
use std::sync::Mutex;
use ordered_float::OrderedFloat;

type R = BooleanInterval;
use crate::{ast::Literal, util::IndexSet};

use super::*;

#[derive(Debug)]
pub struct BoolIntervalSort {
    name: Symbol,
    rats: Mutex<IndexSet<R>>,
}

impl BoolIntervalSort {
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            rats: Default::default(),
        }
    }
}

impl Sort for BoolIntervalSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitives!(eg, "ival-And" = |a: R, b: R| -> R { a.and(&b) });
        add_primitives!(eg, "ival-Or" = |a: R, b: R| -> R { a.or(&b) });

        add_primitives!(eg, "true-interval" = | | -> R {
          R::true_interval()
        });
        add_primitives!(eg, "false-interval" = | | -> R {
          R::false_interval()
        });
        add_primitives!(eg, "unknown-interval" = | | -> R {
          R::unknown_interval()
        });

        add_primitives!(eg, "ival-Less" = |a: Interval, b: Interval| -> BooleanInterval { a.less_than(&b) });
        add_primitives!(eg, "ival-LessEq" = |a: Interval, b: Interval| -> BooleanInterval { a.less_than_or_equal(&b) });

        add_primitives!(eg, "ival-Greater" = |a: Interval, b: Interval| -> BooleanInterval { a.greater_than(&b) });
        add_primitives!(eg, "ival-GreaterEq" = |a: Interval, b: Interval| -> BooleanInterval { a.greater_than_or_equal(&b) });

        add_primitives!(eg, "ival-Eq" = |a: Interval, b: Interval| -> BooleanInterval { a.equal_to(&b) });
        add_primitives!(eg, "ival-NotEq" = |a: Interval, b: Interval| -> BooleanInterval { a.not_equal_to(&b) });

        add_primitives!(eg, "ival-Not" = |a: BooleanInterval| -> BooleanInterval { a.not() });

        add_primitives!(eg, "ival-If" = |a: BooleanInterval, b: Interval, c: Interval| -> Interval { a.if_real_result(&b, &c) });
    }
    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        let rat = R::load(self, &value);
        let left = rat.lo;
        let right = rat.hi;
        if left && right {
          Expr::call("trueinterval", vec![])
        } else if !left && !right {
          Expr::call("falseinterval", vec![])
        } else {
          Expr::call("unknowninterval", vec![])
        }
    }
}

impl FromSort for R {
    type Sort = BoolIntervalSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        sort.rats.lock().unwrap().get_index(i).unwrap().clone()
    }
}

impl IntoSort for R {
    type Sort = BoolIntervalSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let (i, _) = sort.rats.lock().unwrap().insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}
