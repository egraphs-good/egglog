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

    fn foreach_tracked_values<'a>(&'a self, value: &'a Value, mut f: Box<dyn FnMut(Value) + 'a>) {
        // TODO: Potential duplication of code
        let maps = self.maps.lock().unwrap();
        let map = maps.get_index(value.bits as usize).unwrap();

        if self.key.is_eq_sort() {
            for key in map.keys() {
                f(*key)
            }
        }

        if self.value.is_eq_sort() {
            for value in map.values() {
                f(*value)
            }
        }
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let maps = self.maps.lock().unwrap();
        let map = maps.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_map: ValueMap = map
            .iter()
            .map(|(k, v)| {
                let (mut k, mut v) = (*k, *v);
                changed |= self.key.canonicalize(&mut k, unionfind);
                changed |= self.value.canonicalize(&mut v, unionfind);
                (k, v)
            })
            .collect();
        drop(maps);
        *value = new_map.store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
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

    fn make_expr(&self, value: Value) -> Expr {
        let map = ValueMap::load(self, &value);
        let mut expr = Expr::call("map-empty", []);
        for (k, v) in map.iter().rev() {
            let k = self.key.make_expr(*k);
            let v = self.value.make_expr(*v);
            expr = Expr::call("map-insert", [expr, k, v])
        }
        expr
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

struct Ctor {
    name: Symbol,
    map: Arc<MapSort>,
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

    fn apply(&self, values: &[Value]) -> Option<Value> {
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

    fn apply(&self, values: &[Value]) -> Option<Value> {
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

    fn apply(&self, values: &[Value]) -> Option<Value> {
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

    fn apply(&self, values: &[Value]) -> Option<Value> {
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

    fn apply(&self, values: &[Value]) -> Option<Value> {
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

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut map = ValueMap::load(&self.map, &values[0]);
        map.remove(&values[1]);
        map.store(&self.map)
    }
}
