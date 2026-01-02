use super::*;

#[derive(Debug)]
pub struct StringSort;

impl BaseSort for StringSort {
    type Base = S;

    fn name(&self) -> &str {
        "String"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        let string_concat_validator = |termdag: &TermDag, args: &[TermId], result: TermId| -> bool {
            let Term::Lit(Literal::String(result_str)) = termdag.get(result) else { return false };
            let mut concatenated = String::new();
            for &arg in args {
                let Term::Lit(Literal::String(s)) = termdag.get(arg) else { return false };
                concatenated.push_str(s.as_str());
            }
            result_str.as_str() == concatenated
        };
        add_primitive_with_validator!(eg, "+" = [xs: S] -> S {{
            let mut y = String::new();
            xs.for_each(|x| y.push_str(x.as_str()));
            y.into()
        }}, string_concat_validator);

        add_literal_prim!(eg, "replace" = |a: S, b: S, c: S| -> S {
            a.as_str().replace(b.as_str(), c.as_str()).into()
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let s = base_values.unwrap::<S>(value);

        termdag.lit(Literal::String(s.0))
    }
}
