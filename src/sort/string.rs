use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct StringSort;

impl BaseSort for StringSort {
    type Base = S;

    fn name(&self) -> &str {
        "String"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "+" = [xs: S] -> S {{
            let mut y = String::new();
            xs.for_each(|x| y.push_str(x.as_str()));
            y.into()
        }});
        add_primitive!(eg, "replace" = |a: S, b: S, c: S| -> S {
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
