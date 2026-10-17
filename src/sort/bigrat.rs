use super::*;

/// Rational numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `neg`, `abs`
/// - Exponential: `pow`, `log`, `sqrt`, `cbrt`
/// - Rounding: `floor`, `ceil`, `round`
/// - Con/Destruction: `bigrat`, `numer`, `denom`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `to-f64`, `to-i64`
#[derive(Debug)]
pub struct BigRatSort;

impl BaseSort for BigRatSort {
    type Base = Q;

    fn name(&self) -> &str {
        "BigRat"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::builtin::FamilyDescription {
            name: "BigRat".into(), arity: 0,
        }).expect("invalid BigRat family description");
        add_primitive!(eg, "+" [id = "egglog.core.bigrat.add"] = |a: Q, b: Q| -?> Q { a.checked_add(&b).map(Q::new) });
        add_primitive!(eg, "-" [id = "egglog.core.bigrat.sub"] = |a: Q, b: Q| -?> Q { a.checked_sub(&b).map(Q::new) });
        add_primitive!(eg, "*" [id = "egglog.core.bigrat.mul"] = |a: Q, b: Q| -?> Q { a.checked_mul(&b).map(Q::new) });
        add_primitive!(eg, "/" [id = "egglog.core.bigrat.div"] = |a: Q, b: Q| -?> Q { a.checked_div(&b).map(Q::new) });

        add_primitive!(eg, "min" [id = "egglog.core.bigrat.min"] = |a: Q, b: Q| -> Q { a.min(b) });
        add_primitive!(eg, "max" [id = "egglog.core.bigrat.max"] = |a: Q, b: Q| -> Q { a.max(b) });
        add_primitive!(eg, "neg" [id = "egglog.core.bigrat.neg"] = |a: Q| -> Q { Q::new(-a.0) });
        add_primitive!(eg, "abs" [id = "egglog.core.bigrat.abs"] = |a: Q| -> Q { Q::new(a.0.abs()) });
        add_primitive!(eg, "floor" [id = "egglog.core.bigrat.floor"] = |a: Q| -> Q { Q::new(a.0.floor()) });
        add_primitive!(eg, "ceil" [id = "egglog.core.bigrat.ceil"] = |a: Q| -> Q { Q::new(a.0.ceil()) });
        add_primitive!(eg, "round" [id = "egglog.core.bigrat.round"] = |a: Q| -> Q { Q::new(a.round()) });

        add_primitive!(eg, "bigrat" [id = "egglog.core.bigrat.make"] = |a: Z, b: Z| -?> Q { (!b.0.is_zero()).then(|| Q::new(BigRational::new(a.0, b.0))) });
        add_primitive!(eg, "numer" [id = "egglog.core.bigrat.numer"] = |a: Q| -> Z { Z::new(a.numer().clone()) });
        add_primitive!(eg, "denom" [id = "egglog.core.bigrat.denom"] = |a: Q| -> Z { Z::new(a.denom().clone()) });
        add_primitive!(eg, "to-f64" [id = "egglog.core.bigrat.to_f64"] = |a: Q| -> F { F::new(OrderedFloat(a.to_f64().unwrap())) });
        add_primitive!(eg, "to-i64" [id = "egglog.core.bigrat.to_i64"] = |a: Q| -?> i64 { a.is_integer().then(|| a.to_integer()).and_then(|a| a.to_i64()) });

        add_primitive!(eg, "pow" [id = "egglog.core.bigrat.pow"] = |a: Q, b: Q| -?> Q {
            if !b.is_integer() {
                // fractional powers are forbidden.
                // reject this even for the zero case
                None
            } else if a.is_zero() {
                // remove zero from the field of rationals
                // so that multiplicative inverse is always safe
                if b.is_zero() {
                    // 0^0 = 1 by common convention
                    Some(BigRational::one().into())
                } else if b.is_positive() {
                    // 0^n = 0 where (n > 0)
                    Some(BigRational::zero().into())
                } else {
                    // 0^n => (1/0)^(abs n) where (n < 0)
                    None
                }
            } else {
                let is_neg_pow = b.is_negative();
                let (adj_base, adj_exp) = if is_neg_pow {
                    (Q::new(a.recip()), Q::new(b.abs()))
                } else {
                    (a, b)
                };
                // series of type-conversions
                // to match the `checked_pow` signature
                let adj_exp_int = adj_exp.to_i64()?;
                let adj_exp_usize = usize::try_from(adj_exp_int).ok()?;

                num::traits::checked_pow(adj_base.into_inner(), adj_exp_usize).map(Q::new)
            }
        });
        add_primitive!(eg, "log" [id = "egglog.core.bigrat.log"] = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::new(BigRational::zero()))
            } else {
                todo!("log of bigrat")
            }
        });
        add_primitive!(eg, "sqrt" [id = "egglog.core.bigrat.sqrt"] = |a: Q| -?> Q {
            if a.numer().is_positive() && a.denom().is_positive() {
                let s1 = a.numer().sqrt();
                let s2 = a.denom().sqrt();
                let is_perfect = &(s1.clone() * s1.clone()) == a.numer() && &(s2.clone() * s2.clone()) == a.denom();
                if is_perfect {
                    Some(Q::new(BigRational::new(s1, s2)))
                } else {
                    None
                }
            } else {
                None
            }
        });
        add_primitive!(eg, "cbrt" [id = "egglog.core.bigrat.cbrt"] = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::new(BigRational::one()))
            } else {
                todo!("cbrt of bigrat")
            }
        });

        add_primitive!(eg, "<" [id = "egglog.core.bigrat.lt"] = |a: Q, b: Q| -?> () { if a < b {Some(())} else {None} });
        add_primitive!(eg, ">" [id = "egglog.core.bigrat.gt"] = |a: Q, b: Q| -?> () { if a > b {Some(())} else {None} });
        add_primitive!(eg, "<=" [id = "egglog.core.bigrat.le"] = |a: Q, b: Q| -?> () { if a <= b {Some(())} else {None} });
        add_primitive!(eg, ">=" [id = "egglog.core.bigrat.ge"] = |a: Q, b: Q| -?> () { if a >= b {Some(())} else {None} });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let rat = base_values.unwrap::<Q>(value);
        let numer = rat.numer();
        let denom = rat.denom();

        let numer_as_string = termdag.lit(Literal::String(numer.to_string()));
        let denom_as_string = termdag.lit(Literal::String(denom.to_string()));

        let numer_term = termdag.app("from-string".to_owned(), vec![numer_as_string]);
        let denom_term = termdag.app("from-string".to_owned(), vec![denom_as_string]);

        termdag.app("bigrat".to_owned(), vec![numer_term, denom_term])
    }
}
