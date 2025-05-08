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
        add_primitive!(eg, "+" = |a: Q, b: Q| -?> Q { a.checked_add(&b) });
        add_primitive!(eg, "-" = |a: Q, b: Q| -?> Q { a.checked_sub(&b) });
        add_primitive!(eg, "*" = |a: Q, b: Q| -?> Q { a.checked_mul(&b) });
        add_primitive!(eg, "/" = |a: Q, b: Q| -?> Q { a.checked_div(&b) });

        add_primitive!(eg, "min" = |a: Q, b: Q| -> Q { a.min(b) });
        add_primitive!(eg, "max" = |a: Q, b: Q| -> Q { a.max(b) });
        add_primitive!(eg, "neg" = |a: Q| -> Q { -a });
        add_primitive!(eg, "abs" = |a: Q| -> Q { a.abs() });
        add_primitive!(eg, "floor" = |a: Q| -> Q { a.floor() });
        add_primitive!(eg, "ceil" = |a: Q| -> Q { a.ceil() });
        add_primitive!(eg, "round" = |a: Q| -> Q { a.round() });

        add_primitive!(eg, "bigrat" = |a: Z, b: Z| -> Q { Q::new(a, b) });
        add_primitive!(eg, "numer" = |a: Q| -> Z { a.numer().clone() });
        add_primitive!(eg, "denom" = |a: Q| -> Z { a.denom().clone() });
        add_primitive!(eg, "to-f64" = |a: Q| -> F { OrderedFloat(a.to_f64().unwrap()) });

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
                    Some(Q::one())
                } else if b.is_positive() {
                    // 0^n = 0 where (n > 0)
                    Some(Q::zero())
                } else {
                    // 0^n => (1/0)^(abs n) where (n < 0)
                    None
                }
            } else {
                let is_neg_pow = b.is_negative();
                let (adj_base, adj_exp) = if is_neg_pow {
                    (a.recip(), b.abs())
                } else {
                    (a, b)
                };
                // series of type-conversions
                // to match the `checked_pow` signature
                let adj_exp_int = adj_exp.to_i64()?;
                let adj_exp_usize = usize::try_from(adj_exp_int).ok()?;

                num::traits::checked_pow(adj_base, adj_exp_usize)
            }
        });
        add_primitive!(eg, "log" = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::zero())
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
                    Some(Q::new(s1, s2))
                } else {
                    None
                }
            } else {
                None
            }
        });
        add_primitive!(eg, "cbrt" = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::one())
            } else {
                todo!("cbrt of bigrat")
            }
        });

        add_primitive!(eg, "<" = |a: Q, b: Q| -?> () { if a < b {Some(())} else {None} });
        add_primitive!(eg, ">" = |a: Q, b: Q| -?> () { if a > b {Some(())} else {None} });
        add_primitive!(eg, "<=" = |a: Q, b: Q| -?> () { if a <= b {Some(())} else {None} });
        add_primitive!(eg, ">=" = |a: Q, b: Q| -?> () { if a >= b {Some(())} else {None} });
   }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        let rat = Q::load(self, &value);
        let numer = rat.numer();
        let denom = rat.denom();

        let numer_as_string = termdag.lit(Literal::String(numer.to_string().into()));
        let denom_as_string = termdag.lit(Literal::String(denom.to_string().into()));

        let numer_term = termdag.app("from-string".into(), vec![numer_as_string]);
        let denom_term = termdag.app("from-string".into(), vec![denom_as_string]);

        Some((
            1,
            termdag.app("bigrat".into(), vec![numer_term, denom_term]),
        ))
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<Q>())
    }
    
    fn reconstruct_termdag_leaf(
        &self,
        exec_state: &core_relations::ExecutionState,
        value: &core_relations::Value,
        termdag: &mut TermDag,
    ) -> Term {
        let rat = exec_state.prims().unwrap_ref::<BigRational>(*value);
        let numer = rat.numer();
        let denom = rat.denom();

        let numer_as_string = termdag.lit(Literal::String(numer.to_string().into()));
        let denom_as_string = termdag.lit(Literal::String(denom.to_string().into()));

        let numer_term = termdag.app("from_string".into(), vec![numer_as_string]);
        let denom_term = termdag.app("from_string".into(), vec![denom_as_string]);

        termdag.app("bigrat".into(), vec![numer_term, denom_term])
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
    fn store(self, _sort: &Self::Sort) -> Value {
        let (i, _) = RATS.lock().unwrap().insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: BigRatSort.name(),
            bits: i as u64,
        }
    }
}
