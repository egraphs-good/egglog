use super::*;

#[derive(Debug)]
pub struct UnitSort;

impl BaseSort for UnitSort {
    type Base = ();

    fn name(&self) -> &str {
        "Unit"
    }

    fn reconstruct_termdag(
        &self,
        _base_values: &BaseValues,
        _value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        termdag.lit(Literal::Unit)
    }
}
