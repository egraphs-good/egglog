use std::collections::BTreeMap;
use std::sync::Mutex;

use super::*;

type ValueMap = BTreeMap<Value, Value>;

#[derive(Debug)]
pub struct MapSort {
    name: Symbol,
    key: ArcSort,
    value: ArcSort,
    maps: Mutex<IndexSet<ValueMap>>,
}

impl MapSort {
    fn kv_names(&self) -> (Symbol, Symbol) {
        (self.key.name(), self.value.name())
    }

    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(k), Expr::Var(v)] = args {
            let k = typeinfo.sorts.get(k).ok_or(TypeError::UndefinedSort(*k))?;
            let v = typeinfo.sorts.get(v).ok_or(TypeError::UndefinedSort(*v))?;

            if k.is_eq_container_sort() || v.is_container_sort() {
                return Err(TypeError::UndefinedSort(
                    "Maps nested with other EqSort containers are not allowed".into(),
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

impl MapSort {
    pub fn presort_names() -> Vec<Symbol> {
        vec![
            "rebuild".into(),
            "map-empty".into(),
            "map-insert".into(),
            "map-get".into(),
            "map-not-contains".into(),
            "map-contains".into(),
            "map-remove".into(),
        ]
    }
}

impl Sort for MapSort {
    fn name(&self) -> Symbol {
        self.name
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

    fn inner_values(&self, value: &Value) -> Vec<(&ArcSort, Value)> {
        let maps = self.maps.lock().unwrap();
        let map = maps.get_index(value.bits as usize).unwrap();
        let mut result = Vec::new();
        for (k, v) in map.iter() {
            result.push((&self.key, *k));
            result.push((&self.value, *v));
        }
        result
    }

    fn canonicalize(&self, _value: &mut Value, _unionfind: &UnionFind) -> bool {
        false
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(MapRebuild {
            name: "rebuild".into(),
            map: self.clone(),
        });
        typeinfo.add_primitive(Ctor {
            name: "map-empty".into(),
            map: self.clone(),
        });
        typeinfo.add_primitive(Insert {
            name: "map-insert".into(),
            map: self.clone(),
        });
        typeinfo.add_primitive(Get {
            name: "map-get".into(),
            map: self.clone(),
        });
        typeinfo.add_primitive(NotContains {
            name: "map-not-contains".into(),
            map: self.clone(),
            unit: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Contains {
            name: "map-contains".into(),
            map: self.clone(),
            unit: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Remove {
            name: "map-remove".into(),
            map: self,
        });
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> (Cost, Expr) {
        let mut termdag = TermDag::default();
        let extractor = Extractor::new(egraph, &mut termdag);
        self.extract_expr(egraph, value, &extractor, &mut termdag)
            .expect("Extraction should be successful since extractor has been fully initialized")
    }

    fn extract_expr(
        &self,
        _egraph: &EGraph,
        value: Value,
        extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Expr)> {
        let map = ValueMap::load(self, &value);
        let mut expr = Expr::call("map-empty", []);
        let mut cost = 0usize;
        for (k, v) in map.iter().rev() {
            let k = extractor.find_best(*k, termdag, &self.key)?;
            let v = extractor.find_best(*v, termdag, &self.value)?;
            cost = cost.saturating_add(k.0).saturating_add(v.0);
            expr = Expr::call(
                "map-insert",
                [expr, termdag.term_to_expr(&k.1), termdag.term_to_expr(&v.1)],
            )
        }
        Some((cost, expr))
    }
}

impl IntoSort for ValueMap {
    type Sort = MapSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut maps = sort.maps.lock().unwrap();
        let (i, _) = maps.insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueMap {
    type Sort = MapSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let maps = sort.maps.lock().unwrap();
        maps.get_index(value.bits as usize).unwrap().clone()
    }
}

struct MapRebuild {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for MapRebuild {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map] if map.name() == self.map.name => Some(self.map.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], egraph: &EGraph) -> Option<Value> {
        let maps = self.map.maps.lock().unwrap();
        let map = maps.get_index(values[0].bits as usize).unwrap();
        let mut changed = false;
        let new_map: ValueMap = map
            .iter()
            .map(|(k, v)| {
                let (k, v) = (*k, *v);
                let updated_k = egraph.find(k);
                let updated_v = egraph.find(v);
                changed |= updated_k != k || updated_v != v;
                (updated_k, updated_v)
            })
            .collect();

        drop(maps);

        let res = new_map.store(&self.map).unwrap();
        Some(res)
    }
}

struct Ctor {
    name: Symbol,
    map: Arc<MapSort>,
}

pub(crate) struct TermOrderingMin {}

impl PrimitiveLike for TermOrderingMin {
    fn name(&self) -> Symbol {
        "ordering-min".into()
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [a, b] if a.name() == b.name() => Some(a.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        assert_eq!(values.len(), 2);
        if values[0] < values[1] {
            Some(values[0])
        } else {
            Some(values[1])
        }
    }
}

pub(crate) struct TermOrderingMax {}

impl PrimitiveLike for TermOrderingMax {
    fn name(&self) -> Symbol {
        "ordering-max".into()
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [a, b] if a.name() == b.name() => Some(a.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        assert_eq!(values.len(), 2);
        if values[0] > values[1] {
            Some(values[0])
        } else {
            Some(values[1])
        }
    }
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [] => Some(self.map.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        assert!(values.is_empty());
        ValueMap::default().store(&self.map)
    }
}

struct Insert {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for Insert {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map, key, value]
                if (map.name(), (key.name(), value.name()))
                    == (self.map.name, self.map.kv_names()) =>
            {
                Some(self.map.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let mut map = ValueMap::load(&self.map, &values[0]);
        map.insert(values[1], values[2]);
        map.store(&self.map)
    }
}

struct Get {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for Get {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map, key] if (map.name(), key.name()) == (self.map.name, self.map.key.name()) => {
                Some(self.map.value.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let map = ValueMap::load(&self.map, &values[0]);
        map.get(&values[1]).copied()
    }
}

struct NotContains {
    name: Symbol,
    map: Arc<MapSort>,
    unit: Arc<UnitSort>,
}

impl PrimitiveLike for NotContains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map, key] if (map.name(), key.name()) == (self.map.name, self.map.key.name()) => {
                Some(self.unit.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let map = ValueMap::load(&self.map, &values[0]);
        if map.contains_key(&values[1]) {
            None
        } else {
            Some(Value::unit())
        }
    }
}

struct Contains {
    name: Symbol,
    map: Arc<MapSort>,
    unit: Arc<UnitSort>,
}

impl PrimitiveLike for Contains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map, key] if (map.name(), key.name()) == (self.map.name, self.map.key.name()) => {
                Some(self.unit.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let map = ValueMap::load(&self.map, &values[0]);
        if map.contains_key(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}

struct Remove {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for Remove {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map, key] if (map.name(), key.name()) == (self.map.name, self.map.key.name()) => {
                Some(self.map.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let mut map = ValueMap::load(&self.map, &values[0]);
        map.remove(&values[1]);
        map.store(&self.map)
    }
}
