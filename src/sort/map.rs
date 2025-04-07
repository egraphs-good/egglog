use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapContainer<V> {
    do_rebuild_keys: bool,
    do_rebuild_vals: bool,
    pub data: BTreeMap<V, V>,
}

impl Container for MapContainer<core_relations::Value> {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild_keys || self.do_rebuild_vals {
            let mut changed = false;
            let (mut keys, mut vals): (Vec<_>, Vec<_>) = self.data.iter().unzip();
            if self.do_rebuild_keys {
                changed |= rebuilder.rebuild_slice(&mut keys);
            }
            if self.do_rebuild_vals {
                changed |= rebuilder.rebuild_slice(&mut vals);
            }
            self.data = keys.into_iter().zip(vals).collect();
            changed
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = core_relations::Value> + '_ {
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
    maps: Mutex<IndexSet<MapContainer<Value>>>,
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
                .get_sort(k)
                .ok_or(TypeError::UndefinedSort(*k, k_span.clone()))?;
            let v = typeinfo
                .get_sort(v)
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
                    "Maps nested with other EqSort containers are not allowed".into(),
                    v_span.clone(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                key: k.clone(),
                value: v.clone(),
                maps: Default::default(),
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
        backend.register_container_ty::<MapContainer<core_relations::Value>>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.key.is_eq_sort() || self.value.is_eq_sort()
    }

    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        let maps = self.maps.lock().unwrap();
        let map = maps.get_index(value.bits as usize).unwrap();
        let mut result = Vec::new();
        for (k, v) in map.data.iter() {
            result.push((self.key.clone(), *k));
            result.push((self.value.clone(), *v));
        }
        result
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let maps = self.maps.lock().unwrap();
        let map = maps.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_map = MapContainer {
            do_rebuild_keys: map.do_rebuild_keys,
            do_rebuild_vals: map.do_rebuild_vals,
            data: map
                .data
                .iter()
                .map(|(k, v)| {
                    let (mut k, mut v) = (*k, *v);
                    changed |= self.key.canonicalize(&mut k, unionfind);
                    changed |= self.value.canonicalize(&mut v, unionfind);
                    (k, v)
                })
                .collect(),
        };
        drop(maps);
        *value = new_map.store(self);
        changed
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        add_primitive!(eg, "map-empty" = || -> @MapContainer<Value> (self.clone()) { MapContainer { do_rebuild_keys: self.__y.key.is_eq_sort(), do_rebuild_vals: self.__y.value.is_eq_sort(), data: BTreeMap::new() } });

        add_primitive!(eg, "map-get"    = |    xs: @MapContainer<Value> (self.clone()), x: # (self.key())                     | -?> # (self.value()) { xs.data.get(&x).copied() });
        add_primitive!(eg, "map-insert" = |mut xs: @MapContainer<Value> (self.clone()), x: # (self.key()), y: # (self.value())| -> @MapContainer<Value> (self.clone()) {{ xs.data.insert(x, y); xs }});
        add_primitive!(eg, "map-remove" = |mut xs: @MapContainer<Value> (self.clone()), x: # (self.key())                     | -> @MapContainer<Value> (self.clone()) {{ xs.data.remove(&x);   xs }});

        add_primitive!(eg, "map-length"       = |xs: @MapContainer<Value> (self.clone())| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "map-contains"     = |xs: @MapContainer<Value> (self.clone()), x: # (self.key())| -?> () { ( xs.data.contains_key(&x)).then_some(()) });
        add_primitive!(eg, "map-not-contains" = |xs: @MapContainer<Value> (self.clone()), x: # (self.key())| -?> () { (!xs.data.contains_key(&x)).then_some(()) });
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        let map = MapContainer::load(self, &value);
        let mut term = termdag.app("map-empty".into(), vec![]);
        let mut cost = 0usize;
        for (k, v) in map.data.iter().rev() {
            let k = extractor.find_best(*k, termdag, &self.key)?;
            let v = extractor.find_best(*v, termdag, &self.value)?;
            cost = cost.saturating_add(k.0).saturating_add(v.0);
            term = termdag.app("map-insert".into(), vec![term, k.1, v.1]);
        }
        Some((cost, term))
    }
}

impl IntoSort for MapContainer<Value> {
    type Sort = MapSort;
    fn store(self, sort: &Self::Sort) -> Value {
        let mut maps = sort.maps.lock().unwrap();
        let (i, _) = maps.insert_full(self);
        Value {
            #[cfg(debug_assertions)]
            tag: sort.name,
            bits: i as u64,
        }
    }
}

impl FromSort for MapContainer<Value> {
    type Sort = MapSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let maps = sort.maps.lock().unwrap();
        maps.get_index(value.bits as usize).unwrap().clone()
    }
}
