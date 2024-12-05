use num::traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};
use num::{rational::BigRational, BigInt};
use std::sync::Mutex;

type Z = BigInt;
type Q = BigRational;
use crate::{ast::Literal, util::IndexSet};

use super::*;

lazy_static! {
    static ref BIG_RAT_SORT_NAME: Symbol = "BigRat".into();
    static ref RATS: Mutex<IndexSet<Q>> = Default::default();
}

#[derive(Debug)]
pub struct BigRatSort;

impl Sort for BigRatSort {
    fn name(&self) -> Symbol {
        *BIG_RAT_SORT_NAME
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "+" = |a: Q, b: Q| -> Opt<Q> { a.checked_add(&b) });
        add_primitives!(eg, "-" = |a: Q, b: Q| -> Opt<Q> { a.checked_sub(&b) });
        add_primitives!(eg, "*" = |a: Q, b: Q| -> Opt<Q> { a.checked_mul(&b) });
        add_primitives!(eg, "/" = |a: Q, b: Q| -> Opt<Q> { a.checked_div(&b) });

        add_primitives!(eg, "min" = |a: Q, b: Q| -> Q { a.min(b) });
        add_primitives!(eg, "max" = |a: Q, b: Q| -> Q { a.max(b) });
        add_primitives!(eg, "neg" = |a: Q| -> Q { -a });
        add_primitives!(eg, "abs" = |a: Q| -> Q { a.abs() });
        add_primitives!(eg, "floor" = |a: Q| -> Q { a.floor() });
        add_primitives!(eg, "ceil" = |a: Q| -> Q { a.ceil() });
        add_primitives!(eg, "round" = |a: Q| -> Q { a.round() });
        add_primitives!(eg, "bigrat" = |a: Z, b: Z| -> Q { Q::new(a, b) });
        add_primitives!(eg, "numer" = |a: Q| -> Z { a.numer().clone() });
        add_primitives!(eg, "denom" = |a: Q| -> Z { a.denom().clone() });

        add_primitives!(eg, "to-f64" = |a: Q| -> f64 { a.to_f64().unwrap() });

        add_primitives!(eg, "pow" = |a: Q, b: Q| -> Option<Q> {
            if a.is_zero() {
                if b.is_positive() {
                    Some(Q::zero())
                } else {
                    None
                }
            } else if b.is_zero() {
                Some(Q::one())
            } else if let Some(b) = b.to_i64() {
                if let Ok(b) = usize::try_from(b) {
                    num::traits::checked_pow(a, b)
                } else {
                    // TODO handle negative powers
                    None
                }
            } else {
                None
            }
        });
        add_primitives!(eg, "log" = |a: Q| -> Option<Q> {
            if a.is_one() {
                Some(Q::zero())
            } else {
                todo!()
            }
        });
        add_primitives!(eg, "sqrt" = |a: Q| -> Option<Q> {
            if a.numer().is_positive() && a.denom().is_positive() {
                let s1 = a.numer().sqrt();
                let s2 = a.denom().sqrt();
                let is_perfect = &(s1.clone() * s1.clone()) == a.numer() && &(s2.clone() * s2.clone()) == a.denom();
                if is_perfect {
                    Some(Q::new(s1, s2))
                } else {
                    None
                }
            } else {
                None
            }
        });
        add_primitives!(eg, "cbrt" = |a: Q| -> Option<Q> {
            if a.is_one() {
                Some(Q::one())
            } else {
                todo!()
            }
        });

        add_primitives!(eg, "<" = |a: Q, b: Q| -> Opt { if a < b {Some(())} else {None} });
        add_primitives!(eg, ">" = |a: Q, b: Q| -> Opt { if a > b {Some(())} else {None} });
        add_primitives!(eg, "<=" = |a: Q, b: Q| -> Opt { if a <= b {Some(())} else {None} });
        add_primitives!(eg, ">=" = |a: Q, b: Q| -> Opt { if a >= b {Some(())} else {None} });
   }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        let rat = Q::load(self, &value);
        let numer = rat.numer();
        let denom = rat.denom();
        (
            1,
            Expr::call_no_span(
                "bigrat",
                vec![
                    Expr::call_no_span(
                        "from-string",
                        vec![GenericExpr::Lit(
                            DUMMY_SPAN.clone(),
                            Literal::String(numer.to_string().into()),
                        )],
                    ),
                    Expr::call_no_span(
                        "from-string",
                        vec![GenericExpr::Lit(
                            DUMMY_SPAN.clone(),
                            Literal::String(denom.to_string().into()),
                        )],
                    ),
                ],
            ),
        )
    }
}

impl FromSort for Q {
    type Sort = BigRatSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        RATS.lock().unwrap().get_index(i).unwrap().clone()
    }
}

impl IntoSort for Q {
    type Sort = BigRatSort;
    fn store(self, _sort: &Self::Sort) -> Option<Value> {
        let (i, _) = RATS.lock().unwrap().insert_full(self);
        Some(Value {
            #[cfg(debug_assertions)]
            tag: BigRatSort.name(),
            bits: i as u64,
        })
    }
}
