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
        add_literal_prim!(eg, "|" [id = "egglog.core.i64.bitor", bindings = |s| crate::builtin::scalar_member_bindings(s, "__or__", "bitor", &["other"], Some("BitOr"), None)] = |a: i64, b: i64| -> i64 { a | b });
        add_literal_prim!(eg, "^" [id = "egglog.core.i64.bitxor", bindings = |s| crate::builtin::scalar_member_bindings(s, "__xor__", "bitxor", &["other"], Some("BitXor"), None)] = |a: i64, b: i64| -> i64 { a ^ b });
        add_literal_prim!(eg, "<<" [id = "egglog.core.i64.shl", bindings = |s| crate::builtin::scalar_member_bindings(s, "__lshift__", "shl", &["other"], Some("Shl"), None)] = |a: i64, b: i64| -?> i64 { b.try_into().ok().and_then(|b| a.checked_shl(b)) });
        add_literal_prim!(eg, ">>" [id = "egglog.core.i64.shr", bindings = |s| crate::builtin::scalar_member_bindings(s, "__rshift__", "shr", &["other"], Some("Shr"), None)] = |a: i64, b: i64| -?> i64 { b.try_into().ok().and_then(|b| a.checked_shr(b)) });
        add_literal_prim!(eg, "not-i64" [id = "egglog.core.i64.not", bindings = |s| crate::builtin::scalar_member_bindings(s, "__invert__", "not", &[], Some("Not"), None)] = |a: i64| -> i64 { !a });

        add_literal_prim!(eg, "log2" [id = "egglog.core.i64.log2", bindings = |s| crate::builtin::scalar_member_bindings(s, "log2", "log2", &[], None, None)] = |a: i64| -?> i64 { a.checked_ilog2().map(|x| x as i64) });

        add_literal_prim!(eg, "abs" [id = "egglog.core.i64.abs", bindings = |s| crate::builtin::scalar_member_bindings(s, "__abs__", "abs", &[], None, None)] = |a: i64| -?> i64 { a.checked_abs() });

        add_literal_prim!(eg, "<" [id = "egglog.core.i64.lt", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__lt__", "lt", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -?> () { (a < b).then_some(()) });
        add_literal_prim!(eg, ">" [id = "egglog.core.i64.gt", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__gt__", "gt", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -?> () { (a > b).then_some(()) });
        add_literal_prim!(eg, "<=" [id = "egglog.core.i64.le", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__le__", "le", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -?> () { (a <= b).then_some(()) });
        add_literal_prim!(eg, ">=" [id = "egglog.core.i64.ge", bindings = |s| {
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__ge__", "ge", &["other"], None, None);
            bindings.rust.as_mut().unwrap().views[0].receiver.as_mut().unwrap().borrowed = true;
            bindings
        }] = |a: i64, b: i64| -?> () { (a >= b).then_some(()) });

        add_literal_prim!(eg, "bool-=" [id = "egglog.core.i64.bool_eq"] = |a: i64, b: i64| -> bool { a == b });
        add_literal_prim!(eg, "bool-<" [id = "egglog.core.i64.bool_lt", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_lt", "bool_lt", &["other"], None, None)] = |a: i64, b: i64| -> bool { a < b });
        add_literal_prim!(eg, "bool->" [id = "egglog.core.i64.bool_gt", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_gt", "bool_gt", &["other"], None, None)] = |a: i64, b: i64| -> bool { a > b });
        add_literal_prim!(eg, "bool-<=" [id = "egglog.core.i64.bool_le", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_le", "bool_le", &["other"], None, None)] = |a: i64, b: i64| -> bool { a <= b });
        add_literal_prim!(eg, "bool->=" [id = "egglog.core.i64.bool_ge", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_ge", "bool_ge", &["other"], None, None)] = |a: i64, b: i64| -> bool { a >= b });

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
        add_literal_prim!(eg, "count-matches" [id = "egglog.core.string.count_matches", bindings = |_: &crate::proto::GenericSignature| {
            use crate::proto as pb;
            pb::CallableBindings {
                python: Some(pb::PythonBindings { views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Function.into(),
                    path: vec!["egglog".into(), "builtins".into(), "count_matches".into()],
                    params: ["s", "pattern"].iter().enumerate().map(|(i, name)| pb::PythonParameter { core_input: Some(i as u32), name: (*name).into(), default_expr: None }).collect(),
                    ..Default::default()
                }] }),
                rust: Some(pb::RustBindings { views: vec![pb::RustCallable {
                    path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "count_matches".into()],
                    params: ["s", "pattern"].iter().enumerate().map(|(i, name)| pb::RustParameter { core_input: Some(i as u32), name: (*name).into(), borrowed: false }).collect(),
                    ..Default::default()
                }] }),
                ..Default::default()
            }
        }] = |a: S, b: S| -> i64 {
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
