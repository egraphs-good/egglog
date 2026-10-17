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
        eg.type_info().register_builtin_family(crate::builtin::FamilyDescription {
            name: "Unit".into(), arity: 0,
        }).expect("invalid Unit family description");
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
