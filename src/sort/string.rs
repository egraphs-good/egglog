use super::*;

#[derive(Debug)]
pub struct StringSort;

impl BaseSort for StringSort {
    type Base = S;

    fn name(&self) -> &str {
        "String"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "String".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "String".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "String".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid String family presentation");
        let string_concat_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let mut concatenated = String::new();
            for &arg in args {
                let Term::Lit(Literal::String(s)) = termdag.get(arg) else {
                    return None;
                };
                concatenated.push_str(s.as_str());
            }
            let result_lit = Literal::String(concatenated);
            let result_term = termdag.lit(result_lit);
            Some(result_term)
        };
        add_primitive_with_validator!(eg, "+" [id = "egglog.core.string.concat", bindings = |s: &crate::proto::GenericSignature| {
            use crate::proto as pb;
            let mut bindings = crate::builtin::scalar_member_bindings(s, "__add__", "add", &["other"], Some("Add"), Some(2));
            bindings.python.as_mut().unwrap().views.insert(0, pb::PythonCallable {
                kind: pb::PythonCallKind::Function.into(),
                path: vec!["egglog".into(), "builtins".into(), "join".into()],
                params: vec![pb::PythonParameter { core_input: Some(0), name: "strings".into(), default_expr: None }],
                ..Default::default()
            });
            bindings.rust.as_mut().unwrap().views.insert(0, pb::RustCallable {
                path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "join".into()],
                params: vec![pb::RustParameter { core_input: Some(0), name: "strings".into(), borrowed: false }],
                ..Default::default()
            });
            bindings
        }] = [xs: S] -> S {{
            let mut y = String::new();
            xs.for_each(|x| y.push_str(x.as_str()));
            y.into()
        }}, string_concat_validator);

        add_literal_prim!(eg, "replace" [id = "egglog.core.string.replace", bindings = |s| crate::builtin::scalar_member_bindings(s, "replace", "replace", &["old", "new"], None, None)] = |a: S, b: S, c: S| -> S {
            a.as_str().replace(b.as_str(), c.as_str()).into()
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let s = base_values.unwrap::<S>(value);

        termdag.lit(Literal::String(s.0))
    }
}
