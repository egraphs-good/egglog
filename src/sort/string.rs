use numeric_id::NumericId;
use std::num::NonZeroU32;

use super::*;

#[derive(Debug)]
pub struct StringSort;

lazy_static! {
    static ref STRING_SORT_NAME: Symbol = "String".into();
}

impl Sort for StringSort {
    fn name(&self) -> Symbol {
        *STRING_SORT_NAME
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<S>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<S>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "+" = [xs: S] -> S {{
            let mut y = String::new();
            xs.for_each(|x| y.push_str(x.as_str()));
            let x: Symbol = y.into();
            SymbolWrapper(x)
        }});
        add_primitive!(
            eg,
            "replace" = |a: S, b: S, c: S| -> S {
                SymbolWrapper(a.as_str().replace(b.as_str(), c.as_str()).into())
            }
        );
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<S>())
    }
    fn reconstruct_termdag_leaf(
        &self,
        primitives: &Primitives,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let s = primitives.unwrap::<S>(value);

        termdag.lit(Literal::String(s.0))
    }
}

impl IntoSort for S {
    type Sort = StringSort;
}

/// A newtype wrapper for [`Symbol`] to allow for a custom implementation of the
/// [`core_relations::Primitive`] trait.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolWrapper(pub Symbol);

impl SymbolWrapper {
    pub fn new(symbol: Symbol) -> Self {
        SymbolWrapper(symbol)
    }
}

impl core_relations::Primitive for SymbolWrapper {
    const MAY_UNBOX: bool = true;
    fn try_box(&self) -> Option<core_relations::Value> {
        let x: NonZeroU32 = self.0.into();
        Some(core_relations::Value::new_const(x.get()))
    }
    fn try_unbox(val: core_relations::Value) -> Option<Self> {
        Some(SymbolWrapper(NonZeroU32::new(val.rep()).unwrap().into()))
    }
}

impl std::ops::Deref for SymbolWrapper {
    type Target = Symbol;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
