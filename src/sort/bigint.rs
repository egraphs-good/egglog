use super::*;

#[derive(Debug, Serialize)]
pub struct BigIntSort;

impl BaseSort for BigIntSort {
    type Base = Z;

    fn name(&self) -> &str {
        "BigInt"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "bigint" = |a: i64| -> Z { Z::new(a.into()) });

        add_primitive!(eg, "+" = |a: Z, b: Z| -> Z { a + b });
        add_primitive!(eg, "-" = |a: Z, b: Z| -> Z { a - b });
        add_primitive!(eg, "*" = |a: Z, b: Z| -> Z { a * b });
        add_primitive!(eg, "/" = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a / b) });
        add_primitive!(eg, "%" = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a % b) });

        add_primitive!(eg, "&" = |a: Z, b: Z| -> Z { a & b });
        add_primitive!(eg, "|" = |a: Z, b: Z| -> Z { a | b });
        add_primitive!(eg, "^" = |a: Z, b: Z| -> Z { a ^ b });
        add_primitive!(eg, "<<" = |a: Z, b: i64| -> Z { (&*a).shl(b).into() });
        add_primitive!(eg, ">>" = |a: Z, b: i64| -> Z { (&*a).shr(b).into() });
        add_primitive!(eg, "not-Z" = |a: Z| -> Z { Z::new(!&*a) });

        add_primitive!(eg, "bits" = |a: Z| -> Z { Z::new(a.bits().into()) });

        add_primitive!(eg, "<" = |a: Z, b: Z| -?> () { (a < b).then_some(()) });
        add_primitive!(eg, ">" = |a: Z, b: Z| -?> () { (a > b).then_some(()) });
        add_primitive!(eg, "<=" = |a: Z, b: Z| -?> () { (a <= b).then_some(()) });
        add_primitive!(eg, ">=" = |a: Z, b: Z| -?> () { (a >= b).then_some(()) });

        add_primitive!(eg, "bool-=" = |a: Z, b: Z| -> bool { a == b });
        add_primitive!(eg, "bool-<" = |a: Z, b: Z| -> bool { a < b });
        add_primitive!(eg, "bool->" = |a: Z, b: Z| -> bool { a > b });
        add_primitive!(eg, "bool-<=" = |a: Z, b: Z| -> bool { a <= b });
        add_primitive!(eg, "bool->=" = |a: Z, b: Z| -> bool { a >= b });

        add_primitive!(eg, "min" = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitive!(eg, "max" = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitive!(eg, "to-string" = |a: Z| -> S { S::new(a.to_string()) });
        add_primitive!(eg, "from-string" = |a: S| -?> Z {
            a.as_str().parse::<BigInt>().ok().map(Z::new)
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let bigint = base_values.unwrap::<Z>(value);

        let as_string = termdag.lit(Literal::String(bigint.0.to_string()));
        termdag.app("from-string".to_owned(), vec![as_string])
    }
}
