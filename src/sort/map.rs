use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapContainer {
    do_rebuild_keys: bool,
    do_rebuild_vals: bool,
    pub data: BTreeMap<Value, Value>,
}

impl Container for MapContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        let mut changed = false;
        if self.do_rebuild_keys {
            self.data = self
                .data
                .iter()
                .map(|(old, v)| {
                    let new = rebuilder.rebuild_val(*old);
                    changed |= *old != new;
                    (new, *v)
                })
                .collect();
        }
        if self.do_rebuild_vals {
            for old in self.data.values_mut() {
                let new = rebuilder.rebuild_val(*old);
                changed |= *old != new;
                *old = new;
            }
        }
        changed
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().flat_map(|(k, v)| [k, v]).copied()
    }
}

/// A map from a key type to a value type supporting these primitives:
/// - `map-empty`
/// - `map-insert`
/// - `map-get`
/// - `map-contains`
/// - `map-not-contains`
/// - `map-remove`
/// - `map-length`
#[derive(Debug)]
pub struct MapSort {
    name: Symbol,
    key: ArcSort,
    value: ArcSort,
}

impl MapSort {
    pub fn key(&self) -> ArcSort {
        self.key.clone()
    }

    pub fn value(&self) -> ArcSort {
        self.value.clone()
    }
}

impl Presort for MapSort {
    fn presort_name() -> Symbol {
        "Map".into()
    }

    fn reserved_primitives() -> Vec<Symbol> {
        vec![
            "map-empty".into(),
            "map-insert".into(),
            "map-get".into(),
            "map-not-contains".into(),
            "map-contains".into(),
            "map-remove".into(),
            "map-length".into(),
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(k_span, k), Expr::Var(v_span, v)] = args {
            let k = typeinfo
                .get_sort_by_name(k)
                .ok_or(TypeError::UndefinedSort(*k, k_span.clone()))?;
            let v = typeinfo
                .get_sort_by_name(v)
                .ok_or(TypeError::UndefinedSort(*v, v_span.clone()))?;

            // TODO: specialize the error message
            if k.is_eq_container_sort() {
                return Err(TypeError::DisallowedSort(
                    name,
                    "Maps nested with other EqSort containers are not allowed".into(),
                    k_span.clone(),
                ));
            }

            if v.is_container_sort() {
                return Err(TypeError::DisallowedSort(
                    name,
                    "Maps nested with other containers are not allowed".into(),
                    v_span.clone(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                key: k.clone(),
                value: v.clone(),
            }))
        } else {
            panic!()
        }
    }
}

impl Sort for MapSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.register_container_ty::<MapContainer>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.key.clone(), self.value.clone()]
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.key.is_eq_sort() || self.value.is_eq_sort()
    }

    fn inner_values(&self, containers: &Containers, value: Value) -> Vec<(ArcSort, Value)> {
        let val = containers.get_val::<MapContainer>(value).unwrap().clone();
        val.data
            .iter()
            .flat_map(|(k, v)| [(self.key.clone(), *k), (self.value.clone(), *v)])
            .collect()
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "map-empty" = || -> @MapContainer (self.clone()) { MapContainer {
            do_rebuild_keys: self.__y.inner_sorts()[0].is_eq_sort(),
            do_rebuild_vals: self.__y.inner_sorts()[1].is_eq_sort(),
            data: BTreeMap::new()
        } });

        add_primitive!(eg, "map-get"    = |    xs: @MapContainer (self.clone()), x: # (self.key())                     | -?> # (self.value()) { xs.data.get(&x).copied() });
        add_primitive!(eg, "map-insert" = |mut xs: @MapContainer (self.clone()), x: # (self.key()), y: # (self.value())| -> @MapContainer (self.clone()) {{ xs.data.insert(x, y); xs }});
        add_primitive!(eg, "map-remove" = |mut xs: @MapContainer (self.clone()), x: # (self.key())                     | -> @MapContainer (self.clone()) {{ xs.data.remove(&x);   xs }});

        add_primitive!(eg, "map-length"       = |xs: @MapContainer (self.clone())| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "map-contains"     = |xs: @MapContainer (self.clone()), x: # (self.key())| -?> () { ( xs.data.contains_key(&x)).then_some(()) });
        add_primitive!(eg, "map-not-contains" = |xs: @MapContainer (self.clone()), x: # (self.key())| -?> () { (!xs.data.contains_key(&x)).then_some(()) });
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<MapContainer>())
    }

    fn reconstruct_termdag_container(
        &self,
        _containers: &Containers,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        let mut term = termdag.app("map-empty".into(), vec![]);

        for x in element_terms.chunks(2) {
            term = termdag.app("map-insert".into(), vec![term, x[0].clone(), x[1].clone()])
        }

        term
    }
}
