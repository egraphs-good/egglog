use super::*;

#[derive(Debug)]
pub struct StringSort;

impl LeafSort for StringSort {
    type Leaf = S;

    fn name(&self) -> &str {
        "String"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "+" = [xs: S] -> S {{
            let mut y = String::new();
            xs.for_each(|x| y.push_str(x.as_str()));
            y.into()
        }});
        add_primitive!(
            eg,
            "replace" =
                |a: S, b: S, c: S| -> S { a.as_str().replace(b.as_str(), c.as_str()).into() }
        );
    }

    fn reconstruct_termdag(
        &self,
        primitives: &Primitives,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let s = primitives.unwrap_ref::<S>(value);

        termdag.lit(Literal::String(*s))
    }
}
