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

    pub fn make_sort(egraph: &mut EGraph, name: Symbol, args: &[Expr]) -> Result<ArcSort, Error> {
        if let [Expr::Var(k), Expr::Var(v)] = args {
            let k = egraph.sorts.get(k).ok_or(TypeError::UndefinedSort(*k))?;
            let v = egraph.sorts.get(v).ok_or(TypeError::UndefinedSort(*v))?;
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

    fn register_primitives(self: Arc<Self>, egraph: &mut EGraph) {
        egraph.add_primitive(Ctor {
            name: "empty".into(),
            map: self.clone(),
        });
        egraph.add_primitive(Insert {
            name: "insert".into(),
            map: self.clone(),
        });
        egraph.add_primitive(Get {
            name: "get".into(),
            map: self.clone(),
        });
        egraph.add_primitive(NotContains {
            name: "not-contains".into(),
            map: self.clone(),
            unit: egraph.get_sort(),
        });
        egraph.add_primitive(Contains {
            name: "contains".into(),
            map: self.clone(),
            unit: egraph.get_sort(),
        });
        egraph.add_primitive(Union {
            name: "set-union".into(),
            map: self.clone(),
        });
        egraph.add_primitive(Diff {
            name: "set-diff".into(),
            map: self.clone(),
        });
        egraph.add_primitive(Intersect {
            name: "set-intersect".into(),
            map: self.clone(),
        });
        egraph.add_primitive(Remove {
            name: "map-remove".into(),
            map: self,
        });
    }

    fn make_expr(&self, value: Value) -> Expr {
        let map = ValueMap::load(self, &value);
        let mut expr = Expr::call("empty", []);
        for (k, v) in map.iter().rev() {
            let k = self.key.make_expr(*k);
            let v = self.value.make_expr(*v);
            expr = Expr::call("insert", [expr, k, v])
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
    fn arity(&self) -> usize {
        0
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

    fn arity(&self) -> usize {
        3
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

    fn arity(&self) -> usize {
        2
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

    fn arity(&self) -> usize {
        2
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

    fn arity(&self) -> usize {
        2
    }
}

struct Union {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for Union {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map1, map2] if map1.name() == self.map.name && map2.name() == self.map.name() => {
                Some(self.map.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut map1 = ValueMap::load(&self.map, &values[0]);
        let map2 = ValueMap::load(&self.map, &values[1]);
        map1.extend(map2.iter());
        // map.insert(values[1], values[2]);
        map1.store(&self.map)
    }

    fn arity(&self) -> usize {
        2
    }
}

struct Intersect {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for Intersect {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map1, map2] if map1.name() == self.map.name && map2.name() == self.map.name() => {
                Some(self.map.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut map1 = ValueMap::load(&self.map, &values[0]);
        let map2 = ValueMap::load(&self.map, &values[1]);
        map1.retain(|k, _| map2.contains_key(k));
        // map.insert(values[1], values[2]);
        map1.store(&self.map)
    }

    fn arity(&self) -> usize {
        2
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

    fn arity(&self) -> usize {
        2
    }
}

struct Diff {
    name: Symbol,
    map: Arc<MapSort>,
}

impl PrimitiveLike for Diff {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [map1, map2] if map1.name() == self.map.name && map2.name() == self.map.name() => {
                Some(self.map.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut map1 = ValueMap::load(&self.map, &values[0]);
        let map2 = ValueMap::load(&self.map, &values[1]);
        map1.retain(|k, _| !map2.contains_key(k));
        map1.store(&self.map)
    }

    fn arity(&self) -> usize {
        2
    }
}
