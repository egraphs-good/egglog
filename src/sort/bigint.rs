use num::BigInt;
use std::sync::Mutex;

type Z = BigInt;
use crate::{ast::Literal, util::IndexSet};

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

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut TypeInfo) {
        type Opt<T=()> = Option<T>;

        add_primitives!(eg, "+" = |a: Z, b: Z| -> Z { a + b });
        add_primitives!(eg, "-" = |a: Z, b: Z| -> Z { a - b });
        add_primitives!(eg, "*" = |a: Z, b: Z| -> Z { a * b });
        add_primitives!(eg, "/" = |a: Z, b: Z| -> Opt<Z> { (b != BigInt::ZERO).then(|| a / b) });
        add_primitives!(eg, "%" = |a: Z, b: Z| -> Opt<Z> { (b != BigInt::ZERO).then(|| a % b) });

        add_primitives!(eg, "&" = |a: Z, b: Z| -> Z { a & b });
        add_primitives!(eg, "|" = |a: Z, b: Z| -> Z { a | b });
        add_primitives!(eg, "^" = |a: Z, b: Z| -> Z { a ^ b });
        // TODO: figure out type error
        // add_primitives!(eg, "<<" = |a: Z, b: i64| -> Z { a.shl(b) });
        // add_primitives!(eg, ">>" = |a: Z, b: i64| -> Z { a.shr(b) });
        add_primitives!(eg, "not-Z" = |a: Z| -> Z { !a });

        // TODO: use `BigInt::bits`?
        // add_primitives!(eg, "log2" = |a: Z| -> Z { (a as Z).ilog2() as Z });

        add_primitives!(eg, "<" = |a: Z, b: Z| -> Opt { (a < b).then(|| ()) });
        add_primitives!(eg, ">" = |a: Z, b: Z| -> Opt { (a > b).then(|| ()) });
        add_primitives!(eg, "<=" = |a: Z, b: Z| -> Opt { (a <= b).then(|| ()) });
        add_primitives!(eg, ">=" = |a: Z, b: Z| -> Opt { (a >= b).then(|| ()) });

        add_primitives!(eg, "bool-=" = |a: Z, b: Z| -> bool { a == b });
        add_primitives!(eg, "bool-<" = |a: Z, b: Z| -> bool { a < b });
        add_primitives!(eg, "bool->" = |a: Z, b: Z| -> bool { a > b });
        add_primitives!(eg, "bool-<=" = |a: Z, b: Z| -> bool { a <= b });
        add_primitives!(eg, "bool->=" = |a: Z, b: Z| -> bool { a >= b });

        add_primitives!(eg, "min" = |a: Z, b: Z| -> Z { a.min(b) });
        add_primitives!(eg, "max" = |a: Z, b: Z| -> Z { a.max(b) });

        add_primitives!(eg, "to-string" = |a: Z| -> Symbol { a.to_string().into() });
        add_primitives!(eg, "from-string" = |a: Symbol| -> Opt<Z> { a.as_str().parse::<Z>().ok() })
   }

    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (Cost, Expr) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        let bigint = Z::load(self, &value);
        (
            1,
            Expr::call_no_span(
                "from-string",
                vec![GenericExpr::Lit(
                    DUMMY_SPAN.clone(),
                    Literal::String(bigint.to_string().into()),
                )],
            ),
        )
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
    fn store(self, _sort: &Self::Sort) -> Option<Value> {
        let (i, _) = INTS.lock().unwrap().insert_full(self);
        Some(Value {
            #[cfg(debug_assertions)]
            tag: BigIntSort.name(),
            bits: i as u64,
        })
    }
}
