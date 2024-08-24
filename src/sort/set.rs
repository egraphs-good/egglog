use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::constraint::{AllEqualTypeConstraint, SimpleTypeConstraint};

use super::*;

type ValueSet = BTreeSet<Value>;

#[derive(Debug)]
pub struct SetSort {
    name: Symbol,
    element: ArcSort,
    sets: Mutex<IndexSet<ValueSet>>,
}

impl SetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }

    pub fn element_name(&self) -> Symbol {
        self.element.name()
    }

    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(span, e)] = args {
            let e = typeinfo
                .sorts
                .get(e)
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
                sets: Default::default(),
            }))
        } else {
            panic!()
        }
    }
}

impl SetSort {
    pub fn presort_names() -> Vec<Symbol> {
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

    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        // TODO: Potential duplication of code
        let sets = self.sets.lock().unwrap();
        let set = sets.get_index(value.bits as usize).unwrap();
        let mut result = Vec::new();
        for e in set.iter() {
            result.push((self.element.clone(), *e));
        }
        result
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
        typeinfo.add_primitive(SetRebuild {
            name: "rebuild".into(),
            set: self.clone(),
        });
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
            unit: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Contains {
            name: "set-contains".into(),
            set: self.clone(),
            unit: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Remove {
            name: "set-remove".into(),
            set: self.clone(),
        });
        typeinfo.add_primitive(Get {
            name: "set-get".into(),
            set: self.clone(),
            i64: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Length {
            name: "set-length".into(),
            set: self.clone(),
            i64: typeinfo.get_sort_nofail(),
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
        let set = ValueSet::load(self, &value);
        let mut expr = Expr::call_no_span("set-empty", []);
        let mut cost = 0usize;
        for e in set.iter().rev() {
            let e = extractor.find_best(*e, termdag, &self.element)?;
            cost = cost.saturating_add(e.0);
            expr = Expr::call_no_span("set-insert", [expr, termdag.term_to_expr(&e.1)])
        }
        Some((cost, expr))
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_all_arguments_sort(self.set.element())
            .with_output_sort(self.set.clone())
            .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let set = ValueSet::from_iter(values.iter().copied());
        Some(set.store(&self.set).unwrap())
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.set.clone()], span.clone()).into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        assert!(values.is_empty());
        ValueSet::default().store(&self.set)
    }
}

struct SetRebuild {
    name: Symbol,
    set: Arc<SetSort>,
}

impl PrimitiveLike for SetRebuild {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let egraph = egraph.unwrap();
        let set = ValueSet::load(&self.set, &values[0]);
        let new_set: ValueSet = set.iter().map(|e| egraph.find(*e)).collect();
        // drop set to make sure we lose lock
        drop(set);
        new_set.store(&self.set)
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.element(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.element(), self.unit.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.element(), self.unit.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.clone(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.clone(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let mut set1 = ValueSet::load(&self.set, &values[0]);
        let set2 = ValueSet::load(&self.set, &values[1]);
        set1.retain(|k| set2.contains(k));
        // set.insert(values[1], values[2]);
        set1.store(&self.set)
    }
}

struct Length {
    name: Symbol,
    set: Arc<SetSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for Length {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.i64.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let set = ValueSet::load(&self.set, &values[0]);
        Some(Value::from(set.len() as i64))
    }
}
struct Get {
    name: Symbol,
    set: Arc<SetSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for Get {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.i64.clone(), self.set.element()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let set = ValueSet::load(&self.set, &values[0]);
        let index = i64::load(&self.i64, &values[1]);
        set.iter().nth(index as usize).copied()
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.element(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.set.clone(), self.set.clone(), self.set.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let mut set1 = ValueSet::load(&self.set, &values[0]);
        let set2 = ValueSet::load(&self.set, &values[1]);
        set1.retain(|k| !set2.contains(k));
        set1.store(&self.set)
    }
}
