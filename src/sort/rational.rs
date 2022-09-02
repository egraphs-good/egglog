use num_integer::Roots;
use num_traits::{CheckedDiv, One, Signed, ToPrimitive, Zero, CheckedAdd, CheckedSub, CheckedMul};
use std::sync::Mutex;

type R = num_rational::Rational64;
use crate::{ast::Literal, util::IndexSet};

use super::*;

#[derive(Debug)]
pub struct RationalSort {
    name: Symbol,
    rats: Mutex<IndexSet<R>>,
}

impl RationalSort {
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            rats: Default::default(),
        }
    }
}

impl Sort for RationalSort {
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
        add_primitives!(eg, "+" = |a: R, b: R| -> Opt<R> { a.checked_add(&b) });
        add_primitives!(eg, "-" = |a: R, b: R| -> Opt<R> { a.checked_sub(&b) });
        add_primitives!(eg, "*" = |a: R, b: R| -> Opt<R> { a.checked_mul(&b) });
        add_primitives!(eg, "/" = |a: R, b: R| -> Opt<R> { a.checked_div(&b) });

        add_primitives!(eg, "min" = |a: R, b: R| -> R { a.min(b) });
        add_primitives!(eg, "max" = |a: R, b: R| -> R { a.max(b) });
        add_primitives!(eg, "neg" = |a: R| -> R { -a });
        add_primitives!(eg, "abs" = |a: R| -> R { a.abs() });
        add_primitives!(eg, "floor" = |a: R| -> R { a.floor() });
        add_primitives!(eg, "ceil" = |a: R| -> R { a.ceil() });
        add_primitives!(eg, "round" = |a: R| -> R { a.round() });
        add_primitives!(eg, "rational" = |a: i64, b: i64| -> R { R::new(a, b) });

        add_primitives!(eg, "pow" = |a: R, b: R| -> Option<R> {
            if a.is_zero() {
                if b.is_positive() {
                    Some(R::zero())
                } else {
                    None
                }
            } else if b.is_zero() {
                Some(R::one())
            } else if let Some(b) = b.to_i64() { 
                if let Ok(b) = usize::try_from(b) {
                    num_traits::checked_pow(a, b)
                } else {
                    // TODO handle negative powers
                    None
                }
            } else {
                None
            }
        });
        add_primitives!(eg, "log" = |a: R| -> Option<R> {
            if a.is_one() {
                Some(R::zero())
            } else {
                todo!()
            }
        });
        add_primitives!(eg, "sqrt" = |a: R| -> Option<R> {
            if a.numer().is_positive() && a.denom().is_positive() {
                let s1 = a.numer().sqrt();
                let s2 = a.denom().sqrt();
                let is_perfect = &(s1 * s1) == a.numer() && &(s2 * s2) == a.denom();
                if is_perfect {
                    Some(R::new(s1, s2))
                } else {
                    None
                }
            } else {
                None
            }
        });
        add_primitives!(eg, "cbrt" = |a: R| -> Option<R> {
            if a.is_one() {
                Some(R::one())
            } else {
                todo!()
            }
        });

        add_primitives!(eg, "<" = |a: R, b: R| -> Opt { (a < b).then(|| ()) }); 
        add_primitives!(eg, ">" = |a: R, b: R| -> Opt { (a > b).then(|| ()) }); 
    }
    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        let rat = R::load(self, &value);
        let numer = *rat.numer();
        let denom = *rat.denom();
        Expr::call(
            "rational",
            vec![
                Expr::Lit(Literal::Int(numer)),
                Expr::Lit(Literal::Int(denom)),
            ],
        )
    }
}

impl FromSort for R {
    type Sort = RationalSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        *sort.rats.lock().unwrap().get_index(i).unwrap()
    }
}

impl IntoSort for R {
    type Sort = RationalSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let (i, _) = sort.rats.lock().unwrap().insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}
