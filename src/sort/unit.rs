use super::{BaseSort, BaseValues, Debug, Literal, Term, TermDag, Value};

#[derive(Debug)]
pub struct UnitSort;

impl BaseSort for UnitSort {
    type Base = ();

    fn name(&self) -> &'static str {
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
