use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetContainer {
    do_rebuild: bool,
    pub data: BTreeSet<core_relations::Value>,
}

impl Container for SetContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            let mut xs: Vec<_> = self.data.iter().copied().collect();
            let changed = rebuilder.rebuild_slice(&mut xs);
            self.data = xs.into_iter().collect();
            changed
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = core_relations::Value> + '_ {
        self.data.iter().copied()
    }
}

#[derive(Debug)]
pub struct SetSort {
    name: Symbol,
    element: ArcSort,
}

impl SetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for SetSort {
    fn presort_name() -> Symbol {
        "Set".into()
    }

    fn reserved_primitives() -> Vec<Symbol> {
        vec![
            "set-of".into(),
            "set-empty".into(),
            "set-insert".into(),
            "set-not-contains".into(),
            "set-contains".into(),
            "set-remove".into(),
            "set-union".into(),
            "set-diff".into(),
            "set-intersect".into(),
            "set-get".into(),
            "set-length".into(),
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(span, e)] = args {
            let e = typeinfo
                .get_sort_by_name(e)
                .ok_or(TypeError::UndefinedSort(*e, span.clone()))?;

            if e.is_eq_container_sort() {
                return Err(TypeError::DisallowedSort(
                    name,
                    "Sets nested with other EqSort containers are not allowed".into(),
                    span.clone(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                element: e.clone(),
            }))
        } else {
            panic!()
        }
    }
}

impl Sort for SetSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<SetContainer>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.element.clone()]
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort()
    }

    fn inner_values(
        &self,
        containers: &core_relations::Containers,
        value: &core_relations::Value,
    ) -> Vec<(ArcSort, core_relations::Value)> {
        let val = containers.get_val::<SetContainer>(*value).unwrap().clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "set-empty" = |                      | -> @SetContainer (self.clone()) { SetContainer { do_rebuild: self.__y.is_eq_container_sort(), data: BTreeSet::new() } });
        add_primitive!(eg, "set-of"    = [xs: # (self.element())] -> @SetContainer (self.clone()) { SetContainer { do_rebuild: self.__y.is_eq_container_sort(), data: xs.collect()    } });

        add_primitive!(eg, "set-get" = |xs: @SetContainer (self.clone()), i: i64| -?> # (self.element()) { xs.data.iter().nth(i as usize).copied() });
        add_primitive!(eg, "set-insert" = |mut xs: @SetContainer (self.clone()), x: # (self.element())| -> @SetContainer (self.clone()) {{ xs.data.insert( x); xs }});
        add_primitive!(eg, "set-remove" = |mut xs: @SetContainer (self.clone()), x: # (self.element())| -> @SetContainer (self.clone()) {{ xs.data.remove(&x); xs }});

        add_primitive!(eg, "set-length"       = |xs: @SetContainer (self.clone())| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "set-contains"     = |xs: @SetContainer (self.clone()), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "set-not-contains" = |xs: @SetContainer (self.clone()), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });

        add_primitive!(eg, "set-union"      = |mut xs: @SetContainer (self.clone()), ys: @SetContainer (self.clone())| -> @SetContainer (self.clone()) {{ xs.data.extend(ys.data);                  xs }});
        add_primitive!(eg, "set-diff"       = |mut xs: @SetContainer (self.clone()), ys: @SetContainer (self.clone())| -> @SetContainer (self.clone()) {{ xs.data.retain(|k| !ys.data.contains(k)); xs }});
        add_primitive!(eg, "set-intersect"  = |mut xs: @SetContainer (self.clone()), ys: @SetContainer (self.clone())| -> @SetContainer (self.clone()) {{ xs.data.retain(|k|  ys.data.contains(k)); xs }});
    }

    fn reconstruct_termdag_container(
        &self,
        _containers: &core_relations::Containers,
        _value: &core_relations::Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        termdag.app("set-of".into(), element_terms)
    }

    fn serialized_name(&self, _value: &core_relations::Value) -> Symbol {
        "set-of".into()
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<SetContainer>())
    }
}

impl IntoSort for SetContainer {
    type Sort = SetSort;
}
