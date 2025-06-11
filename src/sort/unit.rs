use super::*;

#[derive(Debug)]
pub struct UnitSort;

impl LeafSort for UnitSort {
    type Leaf = ();

    fn name(&self) -> &str {
        "Unit"
    }

    fn reconstruct_termdag(
        &self,
        _primitives: &Primitives,
        _value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        termdag.lit(Literal::Unit)
    }
}
