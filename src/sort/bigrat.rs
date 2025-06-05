use super::*;

lazy_static! {
    static ref BIG_RAT_SORT_NAME: Symbol = "BigRat".into();
    static ref RATS: Mutex<IndexSet<Q>> = Default::default();
}

/// Rational numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `neg`, `abs`
/// - Exponential: `pow`, `log`, `sqrt`, `cbrt`
/// - Rounding: `floor`, `ceil`, `round`
/// - Con/Destruction: `bigrat`, `numer`, `denom`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `to-f64`
#[derive(Debug)]
pub struct BigRatSort;

impl Sort for BigRatSort {
    fn name(&self) -> Symbol {
        *BIG_RAT_SORT_NAME
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<Q>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<Q>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "+" = |a: Q, b: Q| -?> Q { (&*a).checked_add(&b).map(Q::new) });
        add_primitive!(eg, "-" = |a: Q, b: Q| -?> Q { (&*a).checked_sub(&b).map(Q::new) });
        add_primitive!(eg, "*" = |a: Q, b: Q| -?> Q { (&*a).checked_mul(&b).map(Q::new) });
        add_primitive!(eg, "/" = |a: Q, b: Q| -?> Q { (&*a).checked_div(&b).map(Q::new) });

        add_primitive!(eg, "min" = |a: Q, b: Q| -> Q { a.min(b) });
        add_primitive!(eg, "max" = |a: Q, b: Q| -> Q { a.max(b) });
        add_primitive!(eg, "neg" = |a: Q| -> Q { Q::new(-a.0) });
        add_primitive!(eg, "abs" = |a: Q| -> Q { Q::new(a.0.abs()) });
        add_primitive!(eg, "floor" = |a: Q| -> Q { Q::new(a.0.floor()) });
        add_primitive!(eg, "ceil" = |a: Q| -> Q { Q::new(a.0.ceil()) });
        add_primitive!(eg, "round" = |a: Q| -> Q { Q::new((&*a).round()) });

        add_primitive!(eg, "bigrat" = |a: Z, b: Z| -> Q { Q::new(BigRational::new(a.0, b.0)) });
        add_primitive!(eg, "numer" = |a: Q| -> Z { Z::new((&*a).numer().clone()) });
        add_primitive!(eg, "denom" = |a: Q| -> Z { Z::new((&*a).denom().clone()) });
        add_primitive!(eg, "to-f64" = |a: Q| -> F { F::new(OrderedFloat(a.to_f64().unwrap())) });

        add_primitive!(eg, "pow" = |a: Q, b: Q| -?> Q {
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
        add_primitive!(eg, "log" = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::new(BigRational::zero()))
            } else {
                todo!("log of bigrat")
            }
        });
        add_primitive!(eg, "sqrt" = |a: Q| -?> Q {
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
        add_primitive!(eg, "cbrt" = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::new(BigRational::one()))
            } else {
                todo!("cbrt of bigrat")
            }
        });

        add_primitive!(eg, "<" = |a: Q, b: Q| -?> () { if a < b {Some(())} else {None} });
        add_primitive!(eg, ">" = |a: Q, b: Q| -?> () { if a > b {Some(())} else {None} });
        add_primitive!(eg, "<=" = |a: Q, b: Q| -?> () { if a <= b {Some(())} else {None} });
        add_primitive!(eg, ">=" = |a: Q, b: Q| -?> () { if a >= b {Some(())} else {None} });
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<Q>())
    }

    fn reconstruct_termdag_leaf(
        &self,
        primitives: &Primitives,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let rat = primitives.unwrap::<Q>(value);
        let numer = rat.numer();
        let denom = rat.denom();

        let numer_as_string = termdag.lit(Literal::String(numer.to_string().into()));
        let denom_as_string = termdag.lit(Literal::String(denom.to_string().into()));

        let numer_term = termdag.app("from_string".into(), vec![numer_as_string]);
        let denom_term = termdag.app("from_string".into(), vec![denom_as_string]);

        termdag.app("bigrat".into(), vec![numer_term, denom_term])
    }
}

impl IntoSort for Q {
    type Sort = BigRatSort;
}
