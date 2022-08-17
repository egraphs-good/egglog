use num_rational::BigRational;
use std::sync::Mutex;

use crate::util::IndexSet;

use super::*;

#[derive(Debug)]
pub struct RationalSort {
    name: Symbol,
    rats: Mutex<IndexSet<BigRational>>,
}

impl RationalSort {
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            rats: Default::default(),
        }
    }
}

impl Sort for RationalSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        type R = BigRational;
        // TODO we can't have primitives take borrows just yet, since it
        // requires returning a reference to the locked sort
        add_primitives!(eg, "+" = |a: R, b: R| -> R { a + b });
        add_primitives!(eg, "-" = |a: R, b: R| -> R { a - b });
        add_primitives!(eg, "*" = |a: R, b: R| -> R { a * b });
        add_primitives!(eg, "/" = |a: R, b: R| -> R { a / b }); // TODO option
        add_primitives!(eg, "min" = |a: R, b: R| -> R { a.min(b) });
        add_primitives!(eg, "max" = |a: R, b: R| -> R { a.max(b) });
        add_primitives!(eg, "rational" = |a: i64, b: i64| -> R { R::new(a.into(), b.into()) });
    }
    fn make_expr(&self, value: Value) -> Expr {
        assert!(value.tag == self.name());
        // Expr::Lit(Literal::Int(value.bits as _))
        todo!()
    }
}

// impl<'a> TypedSort<'a, BigRational> for RationalSort {
//     fn load(&'a self, value: &Value) -> BigRational {
//         let i = value.bits as usize;
//         self.rats.lock().unwrap().get_index(i).unwrap().clone()
//     }
//     fn store(&'a self, t: BigRational) -> Value {
//         let i = self.rats.lock().unwrap().insert(t);
//         Value {
//             tag: self.name,
//             bits: i as u64,
//         }
//     }
// }

// impl TypedSort<'_> for BigRational {
//     type Load = BigRational;
//     type Store = BigRational;

//     fn load(&self) -> BigRational {
//         self.clone()
//     }
//     fn store(&self) -> BigRational {
//         self.clone()
//     }
// }

impl FromSort for BigRational {
    type Sort = RationalSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let i = value.bits as usize;
        sort.rats.lock().unwrap().get_index(i).unwrap().clone()
    }
}

impl IntoSort for BigRational {
    type Sort = RationalSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let (i, _) = sort.rats.lock().unwrap().insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

// impl<'a> TypedSort<'a, &'a BigRational> for RationalSort {
//     fn load(&'a self, value: &Value) -> &'a BigRational {
//         let i = value.bits as usize;
//         self.rats.lock().unwrap().get_index(i).unwrap()
//     }

//     fn store(&self, t: &BigRational) -> Value {
//         let i = self.rats.lock().unwrap().insert(t.clone());
//         Value {
//             tag: self.name,
//             bits: i as u64,
//         }
//     }
// }
