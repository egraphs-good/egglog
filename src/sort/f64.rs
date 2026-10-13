use super::*;

/// 64-bit floating point numbers supporting these primitives:
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`, `neg`, `abs`
/// - Comparisons: `<`, `>`, `<=`, `>=`
/// - Other: `min`, `max`, `exp`, `log`, `sqrt`
/// - Conversions: `to-f64`, `to-i64`, `to-string`
#[derive(Debug)]
pub struct F64Sort;

impl BaseSort for F64Sort {
    type Base = F;

    fn name(&self) -> &str {
        "f64"
    }

    #[rustfmt::skip]
    // We need the closure for division and mod operations, as they can panic.
    // cf https://github.com/rust-lang/rust-clippy/issues/9422
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "f64".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "f64".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "F64".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid f64 family presentation");
        add_literal_prim!(eg, "+" [id = "egglog.core.f64.add", bindings = |s| crate::builtin::scalar_member_bindings(s, "__add__", "add", &["other"], Some("Add"), None)] = |a: F, b: F| -> F { a + b });
        add_literal_prim!(eg, "-" [id = "egglog.core.f64.sub", bindings = |s| crate::builtin::scalar_member_bindings(s, "__sub__", "sub", &["other"], Some("Sub"), None)] = |a: F, b: F| -> F { a - b });
        add_literal_prim!(eg, "*" [id = "egglog.core.f64.mul", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mul__", "mul", &["other"], Some("Mul"), None)] = |a: F, b: F| -> F { a * b });
        add_literal_prim!(eg, "/" [id = "egglog.core.f64.div", bindings = |s| crate::builtin::scalar_member_bindings(s, "__truediv__", "div", &["other"], Some("Div"), None)] = |a: F, b: F| -?> F { (*b != 0.0).then(|| a / b) });
        add_literal_prim!(eg, "%" [id = "egglog.core.f64.rem", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mod__", "rem", &["other"], Some("Rem"), None)] = |a: F, b: F| -?> F { (*b != 0.0).then(|| a % b) });
        add_literal_prim!(eg, "^" [id = "egglog.core.f64.pow", bindings = |s| crate::builtin::scalar_member_bindings(s, "__pow__", "pow", &["other"], None, None)] = |a: F, b: F| -> F { F::from(OrderedFloat(a.powf(**b))) });
        add_literal_prim!(eg, "neg" [id = "egglog.core.f64.neg", bindings = |s| crate::builtin::scalar_member_bindings(s, "__neg__", "neg", &[], Some("Neg"), None)] = |a: F| -> F { -a });

        add_literal_prim!(eg, "<" [id = "egglog.core.f64.lt", bindings = |s| crate::builtin::scalar_member_bindings(s, "__lt__", "lt", &["other"], None, None)] = |a: F, b: F| -?> () { (a < b).then(|| ()) });
        add_literal_prim!(eg, ">" [id = "egglog.core.f64.gt", bindings = |s| crate::builtin::scalar_member_bindings(s, "__gt__", "gt", &["other"], None, None)] = |a: F, b: F| -?> () { (a > b).then(|| ()) });
        add_literal_prim!(eg, "<=" [id = "egglog.core.f64.le", bindings = |s| crate::builtin::scalar_member_bindings(s, "__le__", "le", &["other"], None, None)] = |a: F, b: F| -?> () { (a <= b).then(|| ()) });
        add_literal_prim!(eg, ">=" [id = "egglog.core.f64.ge", bindings = |s| crate::builtin::scalar_member_bindings(s, "__ge__", "ge", &["other"], None, None)] = |a: F, b: F| -?> () { (a >= b).then(|| ()) });

        add_literal_prim!(eg, "min" [id = "egglog.core.f64.min", bindings = |s| crate::builtin::scalar_member_bindings(s, "min", "min", &["other"], None, None)] = |a: F, b: F| -> F { a.min(b) });
        add_literal_prim!(eg, "max" [id = "egglog.core.f64.max", bindings = |s| crate::builtin::scalar_member_bindings(s, "max", "max", &["other"], None, None)] = |a: F, b: F| -> F { a.max(b) });
        add_literal_prim!(eg, "abs" [id = "egglog.core.f64.abs", bindings = |s| crate::builtin::scalar_member_bindings(s, "__abs__", "abs", &[], None, None)] = |a: F| -> F { F::from(a.abs()) });
        add_literal_prim!(eg, "exp" [id = "egglog.core.f64.exp", bindings = |s| crate::builtin::scalar_member_bindings(s, "exp", "exp", &[], None, None)] = |a: F| -> F { F::from(OrderedFloat(a.exp())) });
        add_literal_prim!(eg, "log" [id = "egglog.core.f64.log", bindings = |s| crate::builtin::scalar_member_bindings(s, "log", "log", &[], None, None)] = |a: F| -?> F {
            (*a > OrderedFloat(0.0)).then(|| F::from(OrderedFloat(a.ln())))
        });
        add_literal_prim!(eg, "sqrt" [id = "egglog.core.f64.sqrt", bindings = |s| crate::builtin::scalar_member_bindings(s, "sqrt", "sqrt", &[], None, None)] = |a: F| -?> F {
            (*a >= OrderedFloat(0.0)).then(|| F::from(OrderedFloat(a.sqrt())))
        });

        // `to-f64` should be in `i64.rs`, but `F64Sort` wouldn't exist yet
        add_literal_prim!(eg, "to-f64" [id = "egglog.core.f64.from_i64", bindings = |s: &crate::proto::GenericSignature| {
            use crate::proto as pb;
            let owner = Some(pb::BindingOwner { kind: Some(pb::binding_owner::Kind::Sort(s.output.unwrap())) });
            pb::CallableBindings {
                python: Some(pb::PythonBindings { views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::ClassMethod.into(), path: vec!["from_i64".into()], owner,
                    params: vec![pb::PythonParameter { core_input: Some(0), name: "i".into(), default_expr: None }],
                    ..Default::default()
                }] }),
                rust: Some(pb::RustBindings { views: vec![pb::RustCallable {
                    path: vec!["from_i64".into()], owner,
                    params: vec![pb::RustParameter { core_input: Some(0), name: "i".into(), borrowed: false }],
                    ..Default::default()
                }] }),
                ..Default::default()
            }
        }] = |a: i64| -> F { F::from(OrderedFloat(a as f64)) });
        add_literal_prim!(eg, "to-i64" [id = "egglog.core.f64.to_i64", bindings = |s| crate::builtin::scalar_member_bindings(s, "to_i64", "to_i64", &[], None, None)] = |a: F| -> i64 { a.0.0 as i64 });
        // Use debug instead of to_string so that decimal place is always printed
        add_literal_prim!(eg, "to-string" [id = "egglog.core.f64.to_string", bindings = |s| crate::builtin::scalar_member_bindings(s, "to_string", "to_string", &[], None, None)] = |a: F| -> S { S::new(format!("{:?}", a.0.0)) });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let f = base_values.unwrap::<F>(value);

        termdag.lit(Literal::Float(f.0))
    }
}
