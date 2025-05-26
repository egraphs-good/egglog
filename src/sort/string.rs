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
    fn reconstruct_termdag_leaf(
        &self,
        primitives: &Primitives,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let s = primitives.unwrap_ref::<S>(value);

        termdag.lit(Literal::String(*s))
    }
}

impl IntoSort for Symbol {
    type Sort = StringSort;
}
