use super::*;

#[derive(Debug)]
pub struct BoolSort;

lazy_static! {
    static ref BOOL_SORT_NAME: Symbol = "bool".into();
}

impl Sort for BoolSort {
    fn name(&self) -> Symbol {
        *BOOL_SORT_NAME
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<bool>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<bool>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "not" = |a: bool| -> bool { !a });
        add_primitive!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        add_primitive!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        add_primitive!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        add_primitive!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
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

        Some((1, termdag.lit(Literal::Bool(value.bits > 0))))
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<bool>())
    }
    
    fn reconstruct_termdag_leaf(
        &self,
        exec_state: &core_relations::ExecutionState,
        value: &core_relations::Value,
        termdag: &mut TermDag,
    ) -> Term {
        let b = exec_state.prims().unwrap_ref::<bool>(*value);

        termdag.lit(Literal::Bool(*b))
    }
    
}

impl IntoSort for bool {
    type Sort = BoolSort;
    fn store(self, _sort: &Self::Sort) -> Value {
        Value {
            #[cfg(debug_assertions)]
            tag: BoolSort.name(),
            bits: self as u64,
        }
    }
}

impl FromSort for bool {
    type Sort = BoolSort;
    fn load(_sort: &Self::Sort, value: &Value) -> Self {
        value.bits != 0
    }
}
