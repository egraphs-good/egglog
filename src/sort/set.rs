use std::collections::BTreeSet;
use std::sync::Mutex;

use super::*;

type ValueSet = BTreeSet<Value>;

#[derive(Debug)]
pub struct SetSort {
    name: Symbol,
    element: ArcSort,
    sets: Mutex<IndexSet<ValueSet>>,
}

impl SetSort {
    pub fn element_name(&self) -> Symbol {
        self.element.name()
    }

    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(e)] = args {
            let e = typeinfo.sorts.get(e).ok_or(TypeError::UndefinedSort(*e))?;

            if e.is_eq_container_sort() {
                return Err(TypeError::UndefinedSort(
                    "Sets nested with other EqSort containers are not allowed".into(),
                ));
            }

            Ok(Arc::new(Self {
                name,
                element: e.clone(),
                sets: Default::default(),
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

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort()
    }

    fn foreach_tracked_values<'a>(&'a self, value: &'a Value, mut f: Box<dyn FnMut(Value) + 'a>) {
        // TODO: Potential duplication of code
        let sets = self.sets.lock().unwrap();
        let set = sets.get_index(value.bits as usize).unwrap();

        if self.element.is_eq_sort() {
            for e in set.iter() {
                f(*e)
            }
        }
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let sets = self.sets.lock().unwrap();
        let set = sets.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_set: ValueSet = set
            .iter()
            .map(|e| {
                let mut e = *e;
                changed |= self.element.canonicalize(&mut e, unionfind);
                e
            })
            .collect();
        drop(sets);
        *value = new_set.store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(SetOf {
            name: "set-of".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(Ctor {
            name: "set-empty".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(Insert {
            name: "set-insert".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(NotContains {
            name: "set-not-contains".into(),
            set: self.clone(),
            unit: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Contains {
            name: "set-contains".into(),
            set: self.clone(),
            unit: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Remove {
            name: "set-remove".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(Union {
            name: "set-union".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(Diff {
            name: "set-diff".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(Intersect {
            name: "set-intersect".into(),
            set: self,
        });
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> Expr {
        let set = ValueSet::load(self, &value);
        let mut expr = Expr::call("set-empty", []);
        for e in set.iter().rev() {
            let e = egraph.extract(*e, Some(&self.element)).1;
            expr = Expr::call("set-insert", [expr, e])
        }
        expr
    }
}

impl IntoSort for ValueSet {
    type Sort = SetSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut sets = sort.sets.lock().unwrap();
        let (i, _) = sets.insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueSet {
    type Sort = SetSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let sets = sort.sets.lock().unwrap();
        sets.get_index(value.bits as usize).unwrap().clone()
    }
}

struct SetOf {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for SetOf {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if types.iter().all(|t| t.name() == self.set.element_name()) {
            Some(self.set.clone())
        } else {
            None
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let set = ValueSet::from_iter(values.iter().copied());
        set.store(&self.set)
    }
}

struct Ctor {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [] => Some(self.set.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        assert!(values.is_empty());
        ValueSet::default().store(&self.set)
    }
}

struct Insert {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for Insert {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set, key] if (set.name(), key.name()) == (self.set.name, self.set.element_name()) => {
                Some(self.set.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut set = ValueSet::load(&self.set, &values[0]);
        set.insert(values[1]);
        set.store(&self.set)
    }
}

struct NotContains {
    name: Symbol,
    set: Arc<SetSort>,
    unit: Arc<UnitSort>,
}

impl PrimitiveLike for NotContains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set, element]
                if (set.name(), element.name()) == (self.set.name, self.set.element_name()) =>
            {
                Some(self.unit.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let set = ValueSet::load(&self.set, &values[0]);
        if set.contains(&values[1]) {
            None
        } else {
            Some(Value::unit())
        }
    }
}

struct Contains {
    name: Symbol,
    set: Arc<SetSort>,
    unit: Arc<UnitSort>,
}

impl PrimitiveLike for Contains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set, key] if (set.name(), key.name()) == (self.set.name, self.set.element_name()) => {
                Some(self.unit.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let set = ValueSet::load(&self.set, &values[0]);
        if set.contains(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}

struct Union {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for Union {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set1, set2] if set1.name() == self.set.name() && set2.name() == self.set.name() => {
                Some(self.set.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut set1 = ValueSet::load(&self.set, &values[0]);
        let set2 = ValueSet::load(&self.set, &values[1]);
        set1.extend(set2.iter());
        set1.store(&self.set)
    }
}

struct Intersect {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for Intersect {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set1, set2] if set1.name() == self.set.name && set2.name() == self.set.name() => {
                Some(self.set.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut set1 = ValueSet::load(&self.set, &values[0]);
        let set2 = ValueSet::load(&self.set, &values[1]);
        set1.retain(|k| set2.contains(k));
        // set.insert(values[1], values[2]);
        set1.store(&self.set)
    }
}

struct Remove {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for Remove {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set, key] if (set.name(), key.name()) == (self.set.name, self.set.element_name()) => {
                Some(self.set.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut set = ValueSet::load(&self.set, &values[0]);
        set.remove(&values[1]);
        set.store(&self.set)
    }
}

struct Diff {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for Diff {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [set1, set2] if set1.name() == self.set.name && set2.name() == self.set.name() => {
                Some(self.set.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut set1 = ValueSet::load(&self.set, &values[0]);
        let set2 = ValueSet::load(&self.set, &values[1]);
        set1.retain(|k| !set2.contains(k));
        set1.store(&self.set)
    }
}
