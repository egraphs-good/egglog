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
        let string_concat_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let mut concatenated = String::new();
            for &arg in args {
                let Term::Lit(Literal::String(s)) = termdag.get(arg) else {
                    return None;
                };
                concatenated.push_str(s.as_str());
            }
            let result_lit = Literal::String(concatenated);
            let result_term = termdag.lit(result_lit);
            Some(result_term)
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
    ) -> TermId {
        let s = base_values.unwrap::<S>(value);

        termdag.lit(Literal::String(s.0))
    }
}
