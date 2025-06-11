use super::*;

#[derive(Debug)]
pub struct BoolSort;

impl LeafSort for BoolSort {
    type Leaf = bool;

    fn name(&self) -> &str {
        "bool"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "not" = |a: bool| -> bool { !a });
        add_primitive!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        add_primitive!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        add_primitive!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_primitive!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
    }

    fn reconstruct_termdag(
        &self,
        primitives: &Primitives,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let b = primitives.unwrap::<bool>(value);

        termdag.lit(Literal::Bool(b))
    }
}
