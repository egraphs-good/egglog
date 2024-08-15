use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::constraint::{AllEqualTypeConstraint, SimpleTypeConstraint};

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
    fn key(&self) -> ArcSort {
        self.key.clone()
    }

    fn value(&self) -> ArcSort {
        self.value.clone()
    }

    pub fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(k_span, k), Expr::Var(v_span, v)] = args {
            let k = typeinfo
                .sorts
                .get(k)
                .ok_or(TypeError::UndefinedSort(*k, k_span.clone()))?;
            let v = typeinfo
                .sorts
                .get(v)
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
            "map-length".into(),
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

    fn inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        let maps = self.maps.lock().unwrap();
        let map = maps.get_index(value.bits as usize).unwrap();
        let mut result = Vec::new();
        for (k, v) in map.iter() {
            result.push((self.key.clone(), *k));
            result.push((self.value.clone(), *v));
        }
        result
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
            unit: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Contains {
            name: "map-contains".into(),
            map: self.clone(),
            unit: typeinfo.get_sort_nofail(),
        });
        typeinfo.add_primitive(Remove {
            name: "map-remove".into(),
            map: self.clone(),
        });
        typeinfo.add_primitive(Length {
            name: "map-length".into(),
            i64: typeinfo.get_sort_nofail(),
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
        let mut expr = Expr::call_no_span("map-empty", []);
        let mut cost = 0usize;
        for (k, v) in map.iter().rev() {
            let k = extractor.find_best(*k, termdag, &self.key)?;
            let v = extractor.find_best(*v, termdag, &self.value)?;
            cost = cost.saturating_add(k.0).saturating_add(v.0);
            expr = Expr::call_no_span(
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.map.clone(), self.map.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], egraph: Option<&mut EGraph>) -> Option<Value> {
        let egraph = egraph.unwrap();
        let maps = self.map.maps.lock().unwrap();
        let map = maps.get_index(values[0].bits as usize).unwrap();
        let new_map: ValueMap = map
            .iter()
            .map(|(k, v)| (egraph.find(*k), egraph.find(*v)))
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

// TODO: move term ordering min/max to its own mod
pub(crate) struct TermOrderingMin {}

impl PrimitiveLike for TermOrderingMin {
    fn name(&self) -> Symbol {
        "ordering-min".into()
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_exact_length(3)
            .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_exact_length(3)
            .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.map.clone()], span.clone()).into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                self.map.clone(),
                self.map.key(),
                self.map.value(),
                self.map.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.map.clone(), self.map.key(), self.map.value()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.map.clone(), self.map.key(), self.unit.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.map.clone(), self.map.key(), self.unit.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
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

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.map.clone(), self.map.key(), self.map.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let mut map = ValueMap::load(&self.map, &values[0]);
        map.remove(&values[1]);
        map.store(&self.map)
    }
}

struct Length {
    name: Symbol,
    map: Arc<MapSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for Length {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.map.clone(), self.i64.clone()],
            span.clone(),
        )
        .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: Option<&mut EGraph>) -> Option<Value> {
        let map = ValueMap::load(&self.map, &values[0]);
        Some(Value::from(map.len() as i64))
    }
}
