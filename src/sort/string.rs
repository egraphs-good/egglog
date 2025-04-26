use super::*;
use std::num::NonZeroU32;

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

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(value.tag, self.name());

        let sym = Symbol::from(NonZeroU32::new(value.bits as _).unwrap());
        Some((1, termdag.lit(Literal::String(sym))))
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
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
    
    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<S>())
    }
}

// TODO could use a local symbol table

impl IntoSort for Symbol {
    type Sort = StringSort;
    fn store(self, _sort: &Self::Sort) -> Value {
        Value {
            #[cfg(debug_assertions)]
            tag: StringSort.name(),
            bits: NonZeroU32::from(self).get() as _,
        }
    }
}

impl FromSort for Symbol {
    type Sort = StringSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        NonZeroU32::new(value.bits as u32).unwrap().into()
    }
}
