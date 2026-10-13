use super::*;

#[derive(Debug)]
pub struct BigIntSort;

impl BaseSort for BigIntSort {
    type Base = Z;

    fn name(&self) -> &str {
        "BigInt"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "BigInt".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "BigInt".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "BigInt".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid BigInt family presentation");
        add_primitive!(eg, "bigint" [id = "egglog.core.bigint.from_i64", bindings = |s: &crate::proto::GenericSignature| {
            use crate::proto as pb;
            let owner = Some(pb::BindingOwner { kind: Some(pb::binding_owner::Kind::Sort(s.output.unwrap())) });
            pb::CallableBindings {
                python: Some(pb::PythonBindings { views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Initializer.into(), owner,
                    params: vec![pb::PythonParameter { core_input: Some(0), name: "value".into(), default_expr: None }],
                    ..Default::default()
                }] }),
                rust: Some(pb::RustBindings { views: vec![pb::RustCallable {
                    path: vec!["new".into()], owner,
                    params: vec![pb::RustParameter { core_input: Some(0), name: "value".into(), borrowed: false }],
                    ..Default::default()
                }] }),
                ..Default::default()
            }
        }] = |a: i64| -> Z { Z::new(a.into()) });

        add_primitive!(eg, "+" [id = "egglog.core.bigint.add", bindings = |s| crate::builtin::scalar_member_bindings(s, "__add__", "add", &["other"], Some("Add"), None)] = |a: Z, b: Z| -> Z { a + b });
        add_primitive!(eg, "-" [id = "egglog.core.bigint.sub", bindings = |s| crate::builtin::scalar_member_bindings(s, "__sub__", "sub", &["other"], Some("Sub"), None)] = |a: Z, b: Z| -> Z { a - b });
        add_primitive!(eg, "*" [id = "egglog.core.bigint.mul", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mul__", "mul", &["other"], Some("Mul"), None)] = |a: Z, b: Z| -> Z { a * b });
        add_primitive!(eg, "/" [id = "egglog.core.bigint.div", bindings = |s| crate::builtin::scalar_member_bindings(s, "__truediv__", "div", &["other"], Some("Div"), None)] = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a / b) });
        add_primitive!(eg, "%" [id = "egglog.core.bigint.rem", bindings = |s| crate::builtin::scalar_member_bindings(s, "__mod__", "rem", &["other"], Some("Rem"), None)] = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a % b) });

        add_primitive!(eg, "&" [id = "egglog.core.bigint.bitand", bindings = |s| crate::builtin::scalar_member_bindings(s, "__and__", "bitand", &["other"], Some("BitAnd"), None)] = |a: Z, b: Z| -> Z { a & b });
        add_primitive!(eg, "|" [id = "egglog.core.bigint.bitor", bindings = |s| crate::builtin::scalar_member_bindings(s, "__or__", "bitor", &["other"], Some("BitOr"), None)] = |a: Z, b: Z| -> Z { a | b });
        add_primitive!(eg, "^" [id = "egglog.core.bigint.bitxor", bindings = |s| crate::builtin::scalar_member_bindings(s, "__xor__", "bitxor", &["other"], Some("BitXor"), None)] = |a: Z, b: Z| -> Z { a ^ b });
        add_primitive!(eg, "<<" [id = "egglog.core.bigint.shl", bindings = |s| crate::builtin::scalar_member_bindings(s, "__lshift__", "shl", &["other"], Some("Shl"), None)] = |a: Z, b: i64| -?> Z { u32::try_from(b).ok().map(|s| (&*a).shl(s as usize).into()) });
        add_primitive!(eg, ">>" [id = "egglog.core.bigint.shr", bindings = |s| crate::builtin::scalar_member_bindings(s, "__rshift__", "shr", &["other"], Some("Shr"), None)] = |a: Z, b: i64| -?> Z { u32::try_from(b).ok().map(|s| (&*a).shr(s as usize).into()) });
        add_primitive!(eg, "not-Z" [id = "egglog.core.bigint.not", bindings = |s| crate::builtin::scalar_member_bindings(s, "__invert__", "not", &[], Some("Not"), None)] = |a: Z| -> Z { Z::new(!&*a) });

        add_primitive!(eg, "bits" [id = "egglog.core.bigint.bits", bindings = |s| crate::builtin::scalar_member_bindings(s, "bits", "bits", &[], None, None)] = |a: Z| -> Z { Z::new(a.bits().into()) });

        add_primitive!(eg, "<" [id = "egglog.core.bigint.lt", bindings = |s| crate::builtin::scalar_member_bindings(s, "__lt__", "lt", &["other"], None, None)] = |a: Z, b: Z| -?> () { (a < b).then_some(()) });
        add_primitive!(eg, ">" [id = "egglog.core.bigint.gt", bindings = |s| crate::builtin::scalar_member_bindings(s, "__gt__", "gt", &["other"], None, None)] = |a: Z, b: Z| -?> () { (a > b).then_some(()) });
        add_primitive!(eg, "<=" [id = "egglog.core.bigint.le", bindings = |s| crate::builtin::scalar_member_bindings(s, "__le__", "le", &["other"], None, None)] = |a: Z, b: Z| -?> () { (a <= b).then_some(()) });
        add_primitive!(eg, ">=" [id = "egglog.core.bigint.ge", bindings = |s| crate::builtin::scalar_member_bindings(s, "__ge__", "ge", &["other"], None, None)] = |a: Z, b: Z| -?> () { (a >= b).then_some(()) });

        add_primitive!(eg, "bool-=" [id = "egglog.core.bigint.bool_eq", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_eq", "bool_eq", &["other"], None, None)] = |a: Z, b: Z| -> bool { a == b });
        add_primitive!(eg, "bool-<" [id = "egglog.core.bigint.bool_lt", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_lt", "bool_lt", &["other"], None, None)] = |a: Z, b: Z| -> bool { a < b });
        add_primitive!(eg, "bool->" [id = "egglog.core.bigint.bool_gt", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_gt", "bool_gt", &["other"], None, None)] = |a: Z, b: Z| -> bool { a > b });
        add_primitive!(eg, "bool-<=" [id = "egglog.core.bigint.bool_le", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_le", "bool_le", &["other"], None, None)] = |a: Z, b: Z| -> bool { a <= b });
        add_primitive!(eg, "bool->=" [id = "egglog.core.bigint.bool_ge", bindings = |s| crate::builtin::scalar_member_bindings(s, "bool_ge", "bool_ge", &["other"], None, None)] = |a: Z, b: Z| -> bool { a >= b });

        add_primitive!(eg, "min" [id = "egglog.core.bigint.min", bindings = |s| crate::builtin::scalar_member_bindings(s, "min", "min", &["other"], None, None)] = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitive!(eg, "max" [id = "egglog.core.bigint.max", bindings = |s| crate::builtin::scalar_member_bindings(s, "max", "max", &["other"], None, None)] = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitive!(eg, "to-string" [id = "egglog.core.bigint.to_string", bindings = |s| crate::builtin::scalar_member_bindings(s, "to_string", "to_string", &[], None, None)] = |a: Z| -> S { S::new(a.to_string()) });
        add_primitive!(eg, "from-string" [id = "egglog.core.bigint.from_string", bindings = |s: &crate::proto::GenericSignature| {
            use crate::proto as pb;
            let owner = Some(pb::BindingOwner { kind: Some(pb::binding_owner::Kind::Sort(s.output.unwrap())) });
            pb::CallableBindings {
                python: Some(pb::PythonBindings { views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::ClassMethod.into(), path: vec!["from_string".into()], owner,
                    params: vec![pb::PythonParameter { core_input: Some(0), name: "s".into(), default_expr: None }],
                    ..Default::default()
                }] }),
                rust: Some(pb::RustBindings { views: vec![pb::RustCallable {
                    path: vec!["from_string".into()], owner,
                    params: vec![pb::RustParameter { core_input: Some(0), name: "s".into(), borrowed: false }],
                    ..Default::default()
                }] }),
                ..Default::default()
            }
        }] = |a: S| -?> Z {
            a.as_str().parse::<BigInt>().ok().map(Z::new)
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let bigint = base_values.unwrap::<Z>(value);

        let as_string = termdag.lit(Literal::String(bigint.0.to_string()));
        termdag.app("from-string".to_owned(), vec![as_string])
    }
}
