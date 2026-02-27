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
        add_literal_prim!(eg, "not" = |a: bool| -> bool { !a });
        add_literal_prim!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
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
        add_primitive_with_validator!(eg, "or" = [xs: bool] -> bool {
            xs.fold(false, |acc, x| acc || x)
        }, or_validator);
        add_literal_prim!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_literal_prim!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
        // A filter primitive that fails the query if the boolean is false
        add_literal_prim!(eg, "filter" = |a: bool| -?> () { a.then_some(()) });
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
