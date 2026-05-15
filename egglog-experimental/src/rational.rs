use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed, F, OrderedFloat};
use num::integer::Roots;
use num::rational::Rational64;
use num::traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};

pub type R = Boxed<Rational64>;
use crate::ast::Literal;

use super::*;

#[derive(Debug)]
pub struct RationalSort;

impl BaseSort for RationalSort {
    type Base = R;

    fn name(&self) -> &str {
        "Rational"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "+" = |a: R, b: R| -?> R { a.0.checked_add(&b.0).map(R::new) });
        add_primitive!(eg, "-" = |a: R, b: R| -?> R { a.0.checked_sub(&b.0).map(R::new) });
        add_primitive!(eg, "*" = |a: R, b: R| -?> R { a.0.checked_mul(&b.0).map(R::new) });
        add_primitive!(eg, "/" = |a: R, b: R| -?> R { a.0.checked_div(&b.0).map(R::new) });

        add_primitive!(eg, "min" = |a: R, b: R| -> R { R::new(a.0.min(b.0)) });
        add_primitive!(eg, "max" = |a: R, b: R| -> R { R::new(a.0.max(b.0)) });
        add_primitive!(eg, "neg" = |a: R| -> R { R::new(-a.0) });
        add_primitive!(eg, "abs" = |a: R| -> R { R::new(a.0.abs()) });
        add_primitive!(eg, "floor" = |a: R| -> R { R::new(a.0.floor()) });
        add_primitive!(eg, "ceil" = |a: R| -> R { R::new(a.0.ceil()) });
        add_primitive!(eg, "round" = |a: R| -> R { R::new(a.0.round()) });
        add_primitive!(eg, "rational" = |a: i64, b: i64| -> R { R::new(Rational64::new(a, b)) });
        add_primitive!(eg, "numer" = |a: R| -> i64 { *a.0.numer() });
        add_primitive!(eg, "denom" = |a: R| -> i64 { *a.0.denom() });

        add_primitive!(eg, "to-f64" = |a: R| -> F { F::new(OrderedFloat(a.0.to_f64().unwrap())) });

        add_primitive!(eg, "pow" = |a: R, b: R| -?> R {
            if a.0.is_zero() {
                if b.0.is_positive() {
                    Some(R::new(Rational64::zero()))
                } else {
                    None
                }
            } else if b.0.is_zero() {
                Some(R::new(Rational64::one()))
            } else if let Some(b) = b.0.to_i64() {
                if let Ok(b) = usize::try_from(b) {
                    num::traits::checked_pow(a.0, b).map(R::new)
                } else {
                    // TODO handle negative powers
                    None
                }
            } else {
                None
            }
        });
        add_primitive!(eg, "log" = |a: R| -?> R {
            if a.0.is_one() {
                Some(R::new(Rational64::zero()))
            } else {
                todo!()
            }
        });
        add_primitive!(eg, "sqrt" = |a: R| -?> R {
            if a.0.numer().is_positive() && a.0.denom().is_positive() {
                let s1 = a.0.numer().sqrt();
                let s2 = a.0.denom().sqrt();
                let is_perfect = &(s1 * s1) == a.0.numer() && &(s2 * s2) == a.0.denom();
                if is_perfect {
                    Some(R::new(Rational64::new(s1, s2)))
                } else {
                    None
                }
            } else {
                None
            }
        });
        add_primitive!(eg, "cbrt" = |a: R| -?> R {
            if a.0.is_one() {
                Some(R::new(Rational64::one()))
            } else {
                todo!()
            }
        });

        add_primitive!(eg, "<" = |a: R, b: R| -?> () { if a.0 < b.0 {Some(())} else {None} });
        add_primitive!(eg, ">" = |a: R, b: R| -?> () { if a.0 > b.0 {Some(())} else {None} });
        add_primitive!(eg, "<=" = |a: R, b: R| -?> () { if a.0 <= b.0 {Some(())} else {None} });
        add_primitive!(eg, ">=" = |a: R, b: R| -?> () { if a.0 >= b.0 {Some(())} else {None} });
   }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let rat = base_values.unwrap::<R>(value);

        let numer = rat.0.numer();
        let denom = rat.0.denom();

        let numer = termdag.lit(Literal::Int(*numer));
        let denom = termdag.lit(Literal::Int(*denom));

        termdag.app("rational".into(), vec![numer, denom])
    }
}
