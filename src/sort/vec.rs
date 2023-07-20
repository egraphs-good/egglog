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

    pub fn presort_names() -> Vec<Symbol> {
        vec![
            "rebuild".into(),
            "vec-of".into(),
            "vec-append".into(),
            "vec-empty".into(),
            "vec-push".into(),
            "vec-pop".into(),
            "vec-not-contains".into(),
            "vec-contains".into(),
            "vec-length".into(),
            "vec-get".into(),
        ]
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
            panic!("Vec sort must have sort as argument. Got {:?}", args)
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

    fn inner_values(&self, value: &Value) -> Vec<(&ArcSort, Value)> {
        // TODO: Potential duplication of code
        let vecs = self.vecs.lock().unwrap();
        let vec = vecs.get_index(value.bits as usize).unwrap();
        let mut result: Vec<(&Arc<dyn Sort>, Value)> = Vec::new();
        for e in vec.iter() {
            result.push((&self.element, *e));
        }
        result
    }

    fn canonicalize(&self, _value: &mut Value, _unionfind: &UnionFind) -> bool {
        false
    }

    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(VecRebuild {
            name: "rebuild".into(),
            vec: self.clone(),
        });
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
            vec: self.clone(),
            unit: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Length {
            name: "vec-length".into(),
            vec: self.clone(),
            i64: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Get {
            name: "vec-get".into(),
            vec: self,
            i64: typeinfo.get_sort(),
        })
    }

    fn make_expr(&self, egraph: &EGraph, value: Value) -> Expr {
        let vec = ValueVec::load(self, &value);
        let mut expr = Expr::call("vec-empty", []);
        let mut termdag = TermDag::default();
        for e in vec.iter().rev() {
            let e = egraph.extract(*e, &mut termdag, &self.element).1;
            expr = Expr::call("vec-push", [expr, termdag.term_to_expr(&e)])
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

struct VecRebuild {
    name: Symbol,
    vec: Arc<VecSort>,
}

impl PrimitiveLike for VecRebuild {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        if let [vec] = types {
            if vec.name() == self.vec.name() {
                return Some(self.vec.clone());
            }
        }
        None
    }

    fn apply(&self, values: &[Value], egraph: &EGraph) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);

        let mut changed = false;
        let new_set: ValueVec = vec
            .iter()
            .map(|e| {
                let updated = egraph.find(*e);
                changed |= updated != *e;
                updated
            })
            .collect();
        Some(new_set.store(&self.vec).unwrap())
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
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

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        if vec.contains(&values[1]) {
            Some(Value::unit())
        } else {
            None
        }
    }
}

struct Length {
    name: Symbol,
    vec: Arc<VecSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for Length {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [vec] if vec.name() == self.vec.name => Some(self.i64.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        Some(Value::from(vec.len() as i64))
    }
}

struct Get {
    name: Symbol,
    vec: Arc<VecSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for Get {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [vec, index] if (vec.name(), index.name()) == (self.vec.name, "i64".into()) => {
                Some(self.vec.element.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let vec = ValueVec::load(&self.vec, &values[0]);
        let index = i64::load(&self.i64, &values[1]);
        vec.get(index as usize).copied()
    }
}
