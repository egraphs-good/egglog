use std::sync::Mutex;

use super::*;

type ValueVec = Vec<Value>;

#[derive(Debug)]
pub struct VecSort {
    name: Symbol,
    element: ArcSort,
    vecs: Mutex<IndexSet<ValueVec>>,
}

impl VecSort {
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
                vecs: Default::default(),
            }))
        } else {
            panic!()
        }
    }
}

impl Sort for VecSort {
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
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();

        if self.element.is_eq_sort() {
            for e in vec.iter() {
                f(*e)
            }
        }
    }

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        let mut changed = false;
        let new_set: ValueVec = vec
            .iter()
            .map(|e| {
                let mut e = *e;
                changed |= self.element.canonicalize(&mut e, unionfind);
                e
            })
            .collect();
        drop(vecs);
        *value = new_set.store(self).unwrap();
        changed
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(VecOf {
            name: "vec-of".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Append {
            name: "vec-append".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Ctor {
            name: "vec-empty".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Push {
            name: "vec-push".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(Pop {
            name: "vec-pop".into(),
            vec: self.clone(),
        });
        typeinfo.add_primitive(NotContains {
            name: "vec-not-contains".into(),
            vec: self.clone(),
            unit: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Contains {
            name: "vec-contains".into(),
            vec: self,
            unit: typeinfo.get_sort(),
        });
    }

    fn make_expr(&self, value: Value) -> Expr {
        let vec = ValueVec::load(self, &value);
        let mut expr = Expr::call("vec-empty", []);
        for e in vec.iter().rev() {
            let e = self.element.make_expr(*e);
            expr = Expr::call("vec-insert", [expr, e])
        }
        expr
    }
}

impl IntoSort for ValueVec {
    type Sort = VecSort;
    fn store(self, sort: &Self::Sort) -> Option<Value> {
        let mut vecs = sort.vecs.lock().unwrap();
        let (i, _) = vecs.insert_full(self);
        Some(Value {
            tag: sort.name,
            bits: i as u64,
        })
    }
}

impl FromSort for ValueVec {
    type Sort = VecSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let vecs = sort.vecs.lock().unwrap();
        vecs.get_index(value.bits as usize).unwrap().clone()
    }
}

struct VecOf {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for VecOf {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if types.iter().all(|t| t.name() == self.vec.element_name()) {
            Some(self.vec.clone())
        } else {
            None
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let vec = ValueVec::from_iter(values.iter().copied());
        vec.store(&self.vec)
    }
}

struct Append {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Append {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if types.iter().all(|t| t.name() == self.vec.name()) {
            Some(self.vec.clone())
        } else {
            None
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let vec = ValueVec::from_iter(values.iter().flat_map(|v| ValueVec::load(&self.vec, v)));
        vec.store(&self.vec)
    }
}

struct Ctor {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [] => Some(self.vec.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        assert!(values.is_empty());
        ValueVec::default().store(&self.vec)
    }
}

struct Push {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Push {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [vec, key] if (vec.name(), key.name()) == (self.vec.name, self.vec.element_name()) => {
                Some(self.vec.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut vec = ValueVec::load(&self.vec, &values[0]);
        vec.push(values[1]);
        vec.store(&self.vec)
    }
}

struct Pop {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for Pop {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [vec] if vec.name() == self.vec.name => Some(self.vec.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let mut vec = ValueVec::load(&self.vec, &values[0]);
        vec.pop();
        vec.store(&self.vec)
    }
}

struct NotContains {
    name: Symbol,
    vec: Arc<VecSort>,
    unit: Arc<UnitSort>,
}

impl PrimitiveLike for NotContains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [vec, element]
                if (vec.name(), element.name()) == (self.vec.name, self.vec.element_name()) =>
            {
                Some(self.unit.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        if vec.contains(&values[1]) {
            None
        } else {
            Some(Value::unit())
        }
    }
}

struct Contains {
    name: Symbol,
    vec: Arc<VecSort>,
    unit: Arc<UnitSort>,
}

impl PrimitiveLike for Contains {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [vec, key] if (vec.name(), key.name()) == (self.vec.name, self.vec.element_name()) => {
                Some(self.unit.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        if vec.contains(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}
