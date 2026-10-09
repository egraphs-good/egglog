use super::*;

#[derive(Debug)]
pub struct BoolSort;

impl BaseSort for BoolSort {
    type Base = bool;

    fn name(&self) -> &str {
        "bool"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "bool".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "Bool".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "Bool".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid bool family presentation");
        add_literal_prim!(eg, "not" [id = "egglog.core.bool.not", bindings = |s| crate::builtin::scalar_member_bindings(s, "__invert__", "not", &[], Some("Not"), None)] = |a: bool| -> bool { !a });
        add_literal_prim!(eg, "and" [id = "egglog.core.bool.and", bindings = |s| crate::builtin::scalar_member_bindings(s, "__and__", "bitand", &["other"], Some("BitAnd"), None)] = |a: bool, b: bool| -> bool { a && b });
        let or_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let mut result = false;
            for &arg in args {
                let Term::Lit(Literal::Bool(b)) = termdag.get(arg) else {
                    return None;
                };
                result = result || *b;
            }
            Some(termdag.lit(Literal::Bool(result)))
        };
        add_primitive_with_validator!(eg, "or" [id = "egglog.core.bool.or", bindings = |s| crate::builtin::scalar_member_bindings(s, "__or__", "bitor", &["other"], Some("BitOr"), Some(2))] = [xs: bool] -> bool {
            xs.fold(false, |acc, x| acc || x)
        }, or_validator);
        add_literal_prim!(eg, "xor" [id = "egglog.core.bool.xor", bindings = |s| crate::builtin::scalar_member_bindings(s, "__xor__", "bitxor", &["other"], Some("BitXor"), None)] = |a: bool, b: bool| -> bool { a ^ b });
        add_literal_prim!(eg, "=>" [id = "egglog.core.bool.implies", bindings = |s| crate::builtin::scalar_member_bindings(s, "implies", "implies", &["other"], None, None)] = |a: bool, b: bool| -> bool { !a || b });
        // A guard primitive that fails the query if the boolean is false
        add_literal_prim!(eg, "guard" = |a: bool| -?> () { a.then_some(()) });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let b = base_values.unwrap::<bool>(value);

        termdag.lit(Literal::Bool(b))
    }
}
