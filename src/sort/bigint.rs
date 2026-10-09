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

        add_primitive!(eg, "+" = |a: Z, b: Z| -> Z { a + b });
        add_primitive!(eg, "-" = |a: Z, b: Z| -> Z { a - b });
        add_primitive!(eg, "*" = |a: Z, b: Z| -> Z { a * b });
        add_primitive!(eg, "/" = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a / b) });
        add_primitive!(eg, "%" = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a % b) });

        add_primitive!(eg, "&" = |a: Z, b: Z| -> Z { a & b });
        add_primitive!(eg, "|" = |a: Z, b: Z| -> Z { a | b });
        add_primitive!(eg, "^" = |a: Z, b: Z| -> Z { a ^ b });
        add_primitive!(eg, "<<" = |a: Z, b: i64| -?> Z { u32::try_from(b).ok().map(|s| (&*a).shl(s as usize).into()) });
        add_primitive!(eg, ">>" = |a: Z, b: i64| -?> Z { u32::try_from(b).ok().map(|s| (&*a).shr(s as usize).into()) });
        add_primitive!(eg, "not-Z" = |a: Z| -> Z { Z::new(!&*a) });

        add_primitive!(eg, "bits" = |a: Z| -> Z { Z::new(a.bits().into()) });

        add_primitive!(eg, "<" = |a: Z, b: Z| -?> () { (a < b).then_some(()) });
        add_primitive!(eg, ">" = |a: Z, b: Z| -?> () { (a > b).then_some(()) });
        add_primitive!(eg, "<=" = |a: Z, b: Z| -?> () { (a <= b).then_some(()) });
        add_primitive!(eg, ">=" = |a: Z, b: Z| -?> () { (a >= b).then_some(()) });

        add_primitive!(eg, "bool-=" = |a: Z, b: Z| -> bool { a == b });
        add_primitive!(eg, "bool-<" = |a: Z, b: Z| -> bool { a < b });
        add_primitive!(eg, "bool->" = |a: Z, b: Z| -> bool { a > b });
        add_primitive!(eg, "bool-<=" = |a: Z, b: Z| -> bool { a <= b });
        add_primitive!(eg, "bool->=" = |a: Z, b: Z| -> bool { a >= b });

        add_primitive!(eg, "min" = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitive!(eg, "max" = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitive!(eg, "to-string" = |a: Z| -> S { S::new(a.to_string()) });
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
