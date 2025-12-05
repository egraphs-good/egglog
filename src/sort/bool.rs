use super::*;
use egglog_add_primitive::add_literal_prim;

#[derive(Debug)]
pub struct BoolSort;

impl BaseSort for BoolSort {
    type Base = bool;

    fn name(&self) -> &str {
        "bool"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_literal_prim!(eg, "not" = |a: bool| -> bool { !a });
        add_literal_prim!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        add_literal_prim!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        add_literal_prim!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_literal_prim!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        use egglog_ast::generic_ast::Literal;

        let b = base_values.unwrap::<bool>(value);
        termdag.lit(Literal::Bool(b))
    }
}
