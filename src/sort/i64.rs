use super::*;

/// Signed 64-bit integers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`
/// - Bitwise: `&`, `|`, `^`, `<<`, `>>`, `not-i64`
/// - Fallible comparisons: `<`, `>`, `<=`, `>=`
/// - Boolean comparisons: `bool-=`, `bool-<`, `bool->`, `bool-<=`, `bool->=`
/// - Other: `min`, `max`, `to-f64`, `to-string`, `log2`
///
/// Note: fallible comparisons are used at the top-level of a query.
/// For example, this rule will only match if `a` is less than `b`.
/// ```text
/// (rule (... (< a b)) (...))
/// ```
/// On the other hand, boolean comparisons will always match, and so
/// make sense to use inside expressions.
#[derive(Debug)]
pub struct I64Sort;

impl BaseSort for I64Sort {
    type Base = i64;

    fn name(&self) -> &str {
        "i64"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "i64".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "i64".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "I64".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid i64 family presentation");
        add_literal_prim!(eg, "+" [id = "egglog.core.i64.add", bindings = |s| crate::builtin::scalar_member_bindings(s, "__add__", "add", &["other"], Some("Add"), None)] = |a: i64, b: i64| -?> i64 { a.checked_add(b) });
        add_literal_prim!(eg, "-" [id = "egglog.core.i64.sub", bindings = |s| crate::builtin::scalar_member_bindings(s, "__sub__", "sub", &["other"], Some("Sub"), None)] = |a: i64, b: i64| -?> i64 { a.checked_sub(b) });
        add_literal_prim!(eg, "*" [id = "egglog.core.i64.mul", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mul__", "mul", &["other"], Some("Mul"), None)] = |a: i64, b: i64| -?> i64 { a.checked_mul(b) });
        add_literal_prim!(eg, "/" [id = "egglog.core.i64.div", bindings = |s| crate::builtin::scalar_member_bindings(s, "__truediv__", "div", &["other"], Some("Div"), None)] = |a: i64, b: i64| -?> i64 { a.checked_div(b) });
        add_literal_prim!(eg, "%" [id = "egglog.core.i64.rem", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mod__", "rem", &["other"], Some("Rem"), None)] = |a: i64, b: i64| -?> i64 { a.checked_rem(b) });

        add_literal_prim!(eg, "&" [id = "egglog.core.i64.bitand", bindings = |s| crate::builtin::scalar_member_bindings(s, "__and__", "bitand", &["other"], Some("BitAnd"), None)] = |a: i64, b: i64| -> i64 { a & b });
        add_literal_prim!(eg, "|" = |a: i64, b: i64| -> i64 { a | b });
        add_literal_prim!(eg, "^" = |a: i64, b: i64| -> i64 { a ^ b });
        add_literal_prim!(eg, "<<" = |a: i64, b: i64| -?> i64 { b.try_into().ok().and_then(|b| a.checked_shl(b)) });
        add_literal_prim!(eg, ">>" = |a: i64, b: i64| -?> i64 { b.try_into().ok().and_then(|b| a.checked_shr(b)) });
        add_literal_prim!(eg, "not-i64" = |a: i64| -> i64 { !a });

        add_literal_prim!(eg, "log2" = |a: i64| -?> i64 { a.checked_ilog2().map(|x| x as i64) });

        add_literal_prim!(eg, "abs" = |a: i64| -?> i64 { a.checked_abs() });

        add_literal_prim!(eg, "<" [id = "egglog.core.i64.lt", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__lt__", "lt", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -?> () { (a < b).then_some(()) });
        add_literal_prim!(eg, ">" = |a: i64, b: i64| -?> () { (a > b).then_some(()) });
        add_literal_prim!(eg, "<=" = |a: i64, b: i64| -?> () { (a <= b).then_some(()) });
        add_literal_prim!(eg, ">=" [id = "egglog.core.i64.ge", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__ge__", "ge", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -?> () { (a >= b).then_some(()) });

        add_literal_prim!(eg, "bool-=" = |a: i64, b: i64| -> bool { a == b });
        add_literal_prim!(eg, "bool-<" = |a: i64, b: i64| -> bool { a < b });
        add_literal_prim!(eg, "bool->" = |a: i64, b: i64| -> bool { a > b });
        add_literal_prim!(eg, "bool-<=" = |a: i64, b: i64| -> bool { a <= b });
        add_literal_prim!(eg, "bool->=" = |a: i64, b: i64| -> bool { a >= b });

        add_literal_prim!(eg, "min" [id = "egglog.core.i64.min", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "min", "min", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -> i64 { a.min(b) });
        add_literal_prim!(eg, "max" [id = "egglog.core.i64.max", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "max", "max", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -> i64 { a.max(b) });

        add_literal_prim!(eg, "to-string" [id = "egglog.core.i64.to_string", bindings = |s| crate::builtin::scalar_member_bindings(s, "to_string", "to_string", &[], None, None)] = |a: i64| -> S { S::new(a.to_string()) });

        // Must be in the i64 sort register function because
        // the string sort is registered before the i64 sort.
        add_literal_prim!(eg, "count-matches" = |a: S, b: S| -> i64 {
            a.as_str().matches(b.as_str()).count() as i64
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let i = base_values.unwrap::<i64>(value);

        termdag.lit(Literal::Int(i))
    }
}
