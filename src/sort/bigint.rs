use super::*;

lazy_static! {
    static ref BIG_INT_SORT_NAME: Symbol = "BigInt".into();
    static ref INTS: Mutex<IndexSet<Z>> = Default::default();
}

#[derive(Debug)]
pub struct BigIntSort;

impl Sort for BigIntSort {
    fn name(&self) -> Symbol {
        *BIG_INT_SORT_NAME
    }

    fn column_ty(&self, prims: &Primitives) -> ColumnTy {
        ColumnTy::Primitive(prims.get_ty::<Z>())
    }

    fn register_type(&self, prims: &mut Primitives) {
        prims.register_type::<Z>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        add_primitives!(eg, "bigint" = |a: i64| -> Z { a.into() });

        add_primitives!(eg, "+" = |a: Z, b: Z| -> Z { a + b });
        add_primitives!(eg, "-" = |a: Z, b: Z| -> Z { a - b });
        add_primitives!(eg, "*" = |a: Z, b: Z| -> Z { a * b });
        add_primitives!(eg, "/" = |a: Z, b: Z| -?> Z { (b != BigInt::ZERO).then(|| a / b) });
        add_primitives!(eg, "%" = |a: Z, b: Z| -?> Z { (b != BigInt::ZERO).then(|| a % b) });

        add_primitives!(eg, "&" = |a: Z, b: Z| -> Z { a & b });
        add_primitives!(eg, "|" = |a: Z, b: Z| -> Z { a | b });
        add_primitives!(eg, "^" = |a: Z, b: Z| -> Z { a ^ b });
        add_primitives!(eg, "<<" = |a: Z, b: i64| -> Z { a.shl(b) });
        add_primitives!(eg, ">>" = |a: Z, b: i64| -> Z { a.shr(b) });
        add_primitives!(eg, "not-Z" = |a: Z| -> Z { !a });

        add_primitives!(eg, "bits" = |a: Z| -> Z { a.bits().into() });

        add_primitives!(eg, "<" = |a: Z, b: Z| -?> () { (a < b).then_some(()) });
        add_primitives!(eg, ">" = |a: Z, b: Z| -?> () { (a > b).then_some(()) });
        add_primitives!(eg, "<=" = |a: Z, b: Z| -?> () { (a <= b).then_some(()) });
        add_primitives!(eg, ">=" = |a: Z, b: Z| -?> () { (a >= b).then_some(()) });

        add_primitives!(eg, "bool-=" = |a: Z, b: Z| -> bool { a == b });
        add_primitives!(eg, "bool-<" = |a: Z, b: Z| -> bool { a < b });
        add_primitives!(eg, "bool->" = |a: Z, b: Z| -> bool { a > b });
        add_primitives!(eg, "bool-<=" = |a: Z, b: Z| -> bool { a <= b });
        add_primitives!(eg, "bool->=" = |a: Z, b: Z| -> bool { a >= b });

        add_primitives!(eg, "min" = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitives!(eg, "max" = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitives!(eg, "to-string" = |a: Z| -> Symbol { a.to_string().into() });
        add_primitives!(eg, "from-string" = |a: Symbol| -?> Z { a.as_str().parse::<Z>().ok() });
   }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        let bigint = Z::load(self, &value);

        let as_string = termdag.lit(Literal::String(bigint.to_string().into()));
        Some((1, termdag.app("from-string".into(), vec![as_string])))
    }
}

impl FromSort for Z {
    type Sort = BigIntSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        INTS.lock().unwrap().get_index(i).unwrap().clone()
    }
}

impl IntoSort for Z {
    type Sort = BigIntSort;
    fn store(self, _sort: &Self::Sort) -> Value {
        let (i, _) = INTS.lock().unwrap().insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: BigIntSort.name(),
            bits: i as u64,
        }
    }
}
