use super::*;

#[derive(Debug)]
pub struct BigIntSort;

impl BaseSort for BigIntSort {
    type Base = Z;

    fn name(&self) -> &str {
        "BigInt"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info().register_builtin_family(crate::builtin::FamilyDescription {
            name: "BigInt".into(), arity: 0,
        }).expect("invalid BigInt family description");
        add_primitive!(eg, "bigint" [id = "egglog.core.bigint.from_i64"] = |a: i64| -> Z { Z::new(a.into()) });

        add_primitive!(eg, "+" [id = "egglog.core.bigint.add"] = |a: Z, b: Z| -> Z { a + b });
        add_primitive!(eg, "-" [id = "egglog.core.bigint.sub"] = |a: Z, b: Z| -> Z { a - b });
        add_primitive!(eg, "*" [id = "egglog.core.bigint.mul"] = |a: Z, b: Z| -> Z { a * b });
        add_primitive!(eg, "/" [id = "egglog.core.bigint.div"] = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a / b) });
        add_primitive!(eg, "%" [id = "egglog.core.bigint.rem"] = |a: Z, b: Z| -?> Z { (*b != BigInt::ZERO).then(|| a % b) });

        add_primitive!(eg, "&" [id = "egglog.core.bigint.bitand"] = |a: Z, b: Z| -> Z { a & b });
        add_primitive!(eg, "|" [id = "egglog.core.bigint.bitor"] = |a: Z, b: Z| -> Z { a | b });
        add_primitive!(eg, "^" [id = "egglog.core.bigint.bitxor"] = |a: Z, b: Z| -> Z { a ^ b });
        add_primitive!(eg, "<<" [id = "egglog.core.bigint.shl"] = |a: Z, b: i64| -?> Z { u32::try_from(b).ok().map(|s| (&*a).shl(s as usize).into()) });
        add_primitive!(eg, ">>" [id = "egglog.core.bigint.shr"] = |a: Z, b: i64| -?> Z { u32::try_from(b).ok().map(|s| (&*a).shr(s as usize).into()) });
        add_primitive!(eg, "not-Z" [id = "egglog.core.bigint.not"] = |a: Z| -> Z { Z::new(!&*a) });

        add_primitive!(eg, "bits" [id = "egglog.core.bigint.bits"] = |a: Z| -> Z { Z::new(a.bits().into()) });

        add_primitive!(eg, "<" [id = "egglog.core.bigint.lt"] = |a: Z, b: Z| -?> () { (a < b).then_some(()) });
        add_primitive!(eg, ">" [id = "egglog.core.bigint.gt"] = |a: Z, b: Z| -?> () { (a > b).then_some(()) });
        add_primitive!(eg, "<=" [id = "egglog.core.bigint.le"] = |a: Z, b: Z| -?> () { (a <= b).then_some(()) });
        add_primitive!(eg, ">=" [id = "egglog.core.bigint.ge"] = |a: Z, b: Z| -?> () { (a >= b).then_some(()) });

        add_primitive!(eg, "bool-=" [id = "egglog.core.bigint.bool_eq"] = |a: Z, b: Z| -> bool { a == b });
        add_primitive!(eg, "bool-<" [id = "egglog.core.bigint.bool_lt"] = |a: Z, b: Z| -> bool { a < b });
        add_primitive!(eg, "bool->" [id = "egglog.core.bigint.bool_gt"] = |a: Z, b: Z| -> bool { a > b });
        add_primitive!(eg, "bool-<=" [id = "egglog.core.bigint.bool_le"] = |a: Z, b: Z| -> bool { a <= b });
        add_primitive!(eg, "bool->=" [id = "egglog.core.bigint.bool_ge"] = |a: Z, b: Z| -> bool { a >= b });

        add_primitive!(eg, "min" [id = "egglog.core.bigint.min"] = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitive!(eg, "max" [id = "egglog.core.bigint.max"] = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitive!(eg, "to-string" [id = "egglog.core.bigint.to_string"] = |a: Z| -> S { S::new(a.to_string()) });
        add_primitive!(eg, "from-string" [id = "egglog.core.bigint.from_string"] = |a: S| -?> Z {
            a.as_str().parse::<BigInt>().ok().map(Z::new)
        });
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let bigint = base_values.unwrap::<Z>(value);

        let as_string = termdag.lit(Literal::String(bigint.0.to_string()));
        termdag.app("from-string".to_owned(), vec![as_string])
    }
}
