use super::*;

#[derive(Debug)]
pub struct UnitSort;

impl BaseSort for UnitSort {
    type Base = ();

    fn name(&self) -> &str {
        "Unit"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::proto::HostSortFamily {
            name: "Unit".into(), arity: 0,
            bindings: Some(crate::proto::SortBindings {
                python: Some(crate::proto::TypeBinding { path: vec!["egglog".into(), "builtins".into(), "Unit".into()], type_params: vec![] }),
                rust: Some(crate::proto::TypeBinding { path: vec!["egglog_experimental".into(), "typed".into(), "builtins".into(), "Unit".into()], type_params: vec![] }),
                ..Default::default()
            }),
        }).expect("invalid Unit family presentation");
    }

    fn reconstruct_termdag(
        &self,
        _base_values: &BaseValues,
        _value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        termdag.lit(Literal::Unit)
    }
}
