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
        add_primitive!(eg, "not" = |a: bool| -> bool { !a });
        add_primitive!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        add_primitive!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        add_primitive!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_primitive!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let b = base_values.unwrap::<bool>(value);

        termdag.lit(Literal::Bool(b))
    }
}
