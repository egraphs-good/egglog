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
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "BigRat".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "BigRat".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "BigRat".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid BigRat family presentation");
        add_primitive!(eg, "+" [id = "egglog.core.bigrat.add", bindings = |s| crate::builtin::scalar_member_bindings(s, "__add__", "add", &["other"], Some("Add"), None)] = |a: Q, b: Q| -?> Q { a.checked_add(&b).map(Q::new) });
        add_primitive!(eg, "-" [id = "egglog.core.bigrat.sub", bindings = |s| crate::builtin::scalar_member_bindings(s, "__sub__", "sub", &["other"], Some("Sub"), None)] = |a: Q, b: Q| -?> Q { a.checked_sub(&b).map(Q::new) });
        add_primitive!(eg, "*" [id = "egglog.core.bigrat.mul", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mul__", "mul", &["other"], Some("Mul"), None)] = |a: Q, b: Q| -?> Q { a.checked_mul(&b).map(Q::new) });
        add_primitive!(eg, "/" [id = "egglog.core.bigrat.div", bindings = |s| crate::builtin::scalar_member_bindings(s, "__truediv__", "div", &["other"], Some("Div"), None)] = |a: Q, b: Q| -?> Q { a.checked_div(&b).map(Q::new) });

        add_primitive!(eg, "min" [id = "egglog.core.bigrat.min", bindings = |s| crate::builtin::scalar_member_bindings(s, "min", "min", &["other"], None, None)] = |a: Q, b: Q| -> Q { a.min(b) });
        add_primitive!(eg, "max" [id = "egglog.core.bigrat.max", bindings = |s| crate::builtin::scalar_member_bindings(s, "max", "max", &["other"], None, None)] = |a: Q, b: Q| -> Q { a.max(b) });
        add_primitive!(eg, "neg" [id = "egglog.core.bigrat.neg", bindings = |s| crate::builtin::scalar_member_bindings(s, "__neg__", "neg", &[], Some("Neg"), None)] = |a: Q| -> Q { Q::new(-a.0) });
        add_primitive!(eg, "abs" [id = "egglog.core.bigrat.abs", bindings = |s| crate::builtin::scalar_member_bindings(s, "__abs__", "abs", &[], None, None)] = |a: Q| -> Q { Q::new(a.0.abs()) });
        add_primitive!(eg, "floor" [id = "egglog.core.bigrat.floor", bindings = |s| crate::builtin::scalar_member_bindings(s, "floor", "floor", &[], None, None)] = |a: Q| -> Q { Q::new(a.0.floor()) });
        add_primitive!(eg, "ceil" [id = "egglog.core.bigrat.ceil", bindings = |s| crate::builtin::scalar_member_bindings(s, "ceil", "ceil", &[], None, None)] = |a: Q| -> Q { Q::new(a.0.ceil()) });
        add_primitive!(eg, "round" [id = "egglog.core.bigrat.round", bindings = |s| crate::builtin::scalar_member_bindings(s, "round", "round", &[], None, None)] = |a: Q| -> Q { Q::new(a.round()) });

        add_primitive!(eg, "bigrat" [id = "egglog.core.bigrat.make", bindings = |s: &crate::proto::GenericSignature| {
            use crate::proto as pb;
            let owner = Some(pb::BindingOwner { kind: Some(pb::binding_owner::Kind::Sort(s.output.unwrap())) });
            pb::CallableBindings {
                python: Some(pb::PythonBindings { views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Initializer.into(), owner,
                    params: ["num", "den"].iter().enumerate().map(|(i, name)| pb::PythonParameter { core_input: Some(i as u32), name: (*name).into(), default_expr: None }).collect(),
                    ..Default::default()
                }] }),
                rust: Some(pb::RustBindings { views: vec![pb::RustCallable {
                    path: vec!["new".into()], owner,
                    params: ["num", "den"].iter().enumerate().map(|(i, name)| pb::RustParameter { core_input: Some(i as u32), name: (*name).into(), borrowed: false }).collect(),
                    ..Default::default()
                }] }),
                ..Default::default()
            }
        }] = |a: Z, b: Z| -?> Q { (!b.0.is_zero()).then(|| Q::new(BigRational::new(a.0, b.0))) });
        add_primitive!(eg, "numer" [id = "egglog.core.bigrat.numer", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "numer", "numer", &[], None, None);
            bindings.python.as_mut().unwrap().views[0].kind = crate::proto::PythonCallKind::Property.into();
            bindings
        }] = |a: Q| -> Z { Z::new(a.numer().clone()) });
        add_primitive!(eg, "denom" [id = "egglog.core.bigrat.denom", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "denom", "denom", &[], None, None);
            bindings.python.as_mut().unwrap().views[0].kind = crate::proto::PythonCallKind::Property.into();
            bindings
        }] = |a: Q| -> Z { Z::new(a.denom().clone()) });
        add_primitive!(eg, "to-f64" [id = "egglog.core.bigrat.to_f64", bindings = |s| crate::builtin::scalar_member_bindings(s, "to_f64", "to_f64", &[], None, None)] = |a: Q| -> F { F::new(OrderedFloat(a.to_f64().unwrap())) });
        add_primitive!(eg, "to-i64" [id = "egglog.core.bigrat.to_i64", bindings = |s| crate::builtin::scalar_member_bindings(s, "to_i64", "to_i64", &[], None, None)] = |a: Q| -?> i64 { a.is_integer().then(|| a.to_integer()).and_then(|a| a.to_i64()) });

        add_primitive!(eg, "pow" [id = "egglog.core.bigrat.pow", bindings = |s| crate::builtin::scalar_member_bindings(s, "__pow__", "pow", &["other"], None, None)] = |a: Q, b: Q| -?> Q {
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
        add_primitive!(eg, "log" [id = "egglog.core.bigrat.log", bindings = |s| crate::builtin::scalar_member_bindings(s, "log", "log", &[], None, None)] = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::new(BigRational::zero()))
            } else {
                todo!("log of bigrat")
            }
        });
        add_primitive!(eg, "sqrt" [id = "egglog.core.bigrat.sqrt", bindings = |s| crate::builtin::scalar_member_bindings(s, "sqrt", "sqrt", &[], None, None)] = |a: Q| -?> Q {
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
        add_primitive!(eg, "cbrt" [id = "egglog.core.bigrat.cbrt", bindings = |s| crate::builtin::scalar_member_bindings(s, "cbrt", "cbrt", &[], None, None)] = |a: Q| -?> Q {
            if a.is_one() {
                Some(Q::new(BigRational::one()))
            } else {
                todo!("cbrt of bigrat")
            }
        });

        add_primitive!(eg, "<" [id = "egglog.core.bigrat.lt", bindings = |s| crate::builtin::scalar_member_bindings(s, "__lt__", "lt", &["other"], None, None)] = |a: Q, b: Q| -?> () { if a < b {Some(())} else {None} });
        add_primitive!(eg, ">" [id = "egglog.core.bigrat.gt", bindings = |s| crate::builtin::scalar_member_bindings(s, "__gt__", "gt", &["other"], None, None)] = |a: Q, b: Q| -?> () { if a > b {Some(())} else {None} });
        add_primitive!(eg, "<=" [id = "egglog.core.bigrat.le", bindings = |s| crate::builtin::scalar_member_bindings(s, "__le__", "le", &["other"], None, None)] = |a: Q, b: Q| -?> () { if a <= b {Some(())} else {None} });
        add_primitive!(eg, ">=" [id = "egglog.core.bigrat.ge", bindings = |s| crate::builtin::scalar_member_bindings(s, "__ge__", "ge", &["other"], None, None)] = |a: Q, b: Q| -?> () { if a >= b {Some(())} else {None} });
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
