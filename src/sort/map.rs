use crate::constraint::{AllEqualTypeConstraint, NoTypeConstraint};

use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapContainer {
    do_rebuild_keys: bool,
    do_rebuild_vals: bool,
    pub data: BTreeMap<Value, Value>,
}

impl ContainerValue for MapContainer {
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
#[derive(Clone, Debug)]
pub struct MapSort {
    name: String,
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
    fn presort_name() -> &'static str {
        "Map"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "map-empty",
            "map-insert",
            "map-get",
            "map-not-contains",
            "map-contains",
            "map-remove",
            "map-length",
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(k_span, k), Expr::Var(v_span, v)] = args {
            let k = typeinfo
                .get_sort_by_name(k)
                .ok_or(TypeError::UndefinedSort(k.clone(), k_span.clone()))?;
            let v = typeinfo
                .get_sort_by_name(v)
                .ok_or(TypeError::UndefinedSort(v.clone(), v_span.clone()))?;

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

            let out = Self {
                name,
                key: k.clone(),
                value: v.clone(),
            };
            Ok(out.to_arcsort())
        } else {
            panic!()
        }
    }
}

impl ContainerSort for MapSort {
    type Container = MapContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.key.clone(), self.value.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.key.is_eq_sort() || self.value.is_eq_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<MapContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .flat_map(|(k, v)| [(self.key.clone(), *k), (self.value.clone(), *v)])
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        // Takes two name maps that map child keys to input names and canonicalizes them, producing a "shape"
        add_primitive!(eg, "map-empty" = {self.clone(): MapSort} || -> @MapContainer (arc) { MapContainer {
            do_rebuild_keys: self.ctx.key.is_eq_sort(),
            do_rebuild_vals: self.ctx.value.is_eq_sort(),
            data: BTreeMap::new()
        } });

        add_primitive!(eg, "map-get"    = |    xs: @MapContainer (arc), x: # (self.key())                     | -?> # (self.value()) { xs.data.get(&x).copied() });
        add_primitive!(eg, "map-insert" = |mut xs: @MapContainer (arc), x: # (self.key()), y: # (self.value())| -> @MapContainer (arc) {{ xs.data.insert(x, y); xs }});
        add_primitive!(eg, "map-remove" = |mut xs: @MapContainer (arc), x: # (self.key())                     | -> @MapContainer (arc) {{ xs.data.remove(&x);   xs }});

        add_primitive!(eg, "map-length"       = |xs: @MapContainer (arc)| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "map-contains"     = |xs: @MapContainer (arc), x: # (self.key())| -?> () { ( xs.data.contains_key(&x)).then_some(()) });
        add_primitive!(eg, "map-not-contains" = |xs: @MapContainer (arc), x: # (self.key())| -?> () { (!xs.data.contains_key(&x)).then_some(()) });

        add_primitive!(eg, "map-inverse" = |xs: @MapContainer (arc)| -> @MapContainer (arc) {{
            let mut new_map = BTreeMap::new();
            for (k, v) in xs.data.iter() {
                new_map.insert(*v, *k);
            }
            MapContainer {
                do_rebuild_keys: xs.do_rebuild_vals,
                do_rebuild_vals: xs.do_rebuild_keys,
                data: new_map
            }
        }});

        // add shape primitive
        eg.add_primitive(Shape {});
        eg.add_primitive(ComposeInvert {});
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
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

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        self.name().to_owned()
    }
}

#[derive(Clone, Debug)]
struct Shape {}

impl Primitive for Shape {
    fn name(&self) -> &str {
        "shape"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn crate::constraint::TypeConstraint> {
        // todo no type contraints
        Box::new(NoTypeConstraint::new())
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let mut original_pairs = vec![];
        let mut maps = vec![];
        for arg in args {
            let m = exec_state
                .container_values()
                .get_val::<MapContainer>(*arg)?;
            original_pairs.push(m.clone().data);
            let kv_pairs: Vec<(Value, Value)> = m.clone().data.into_iter().collect();

            let mut kv_pairs: Vec<(i64, Value)> = kv_pairs
                .into_iter()
                .map(|(k, v)| (exec_state.base_values().unwrap::<i64>(k), v))
                .collect();

            kv_pairs.sort();

            maps.push(kv_pairs);
        }

        let res: Vec<Value> = maps.into_iter().flatten().map(|(k, v)| v).collect();

        // maps from shape slots to original slots
        let mut m: BTreeMap<Value, Value> = BTreeMap::new();
        for r in res {
            if !m.contains_key(&r) {
                m.insert(r, exec_state.base_values().get::<i64>(m.len() as i64));
            }
        }

        // we want mappings from arguements to shape slots
        let shape_parts: Vec<BTreeMap<Value, Value>> =
            original_pairs.iter().map(|x| compose(&m, x)).collect();

        let mut res = shape_parts;
        res.push(m);

        // turn each btree into a value
        let mut res_values = vec![];
        for m in res {
            let map_value = exec_state.container_values().register_val(
                MapContainer {
                    do_rebuild_keys: false,
                    do_rebuild_vals: false,
                    data: m,
                },
                exec_state,
            );
            res_values.push(map_value);
        }

        // run res into a vector value
        let vec_value = exec_state.container_values().register_val(
            VecContainer {
                do_rebuild: false,
                data: res_values,
            },
            exec_state,
        );

        Some(vec_value)
    }
}

#[derive(Clone, Debug)]
struct ComposeInvert {}

// This computes (compose (invert m1) m2) for two arguments m1 and m2.
impl Primitive for ComposeInvert {
    fn name(&self) -> &str {
        "compose-invert"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn crate::constraint::TypeConstraint> {
        // must be vecs of integer sort
        Box::new(AllEqualTypeConstraint::new("compose-invert", span.clone()))
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let m1 = exec_state
            .container_values()
            .get_val::<MapContainer>(args[0])?
            .clone();
        let m2 = exec_state
            .container_values()
            .get_val::<MapContainer>(args[1])?
            .clone();
        let res = compose(&invert(&m1.data), &m2.data);
        eprintln!(
            "compose-invert: m1 = {:?}, m2 = {:?}, res = {:?}",
            m1.data, m2.data, res
        );
        let map_value = exec_state.container_values().register_val(
            MapContainer {
                do_rebuild_keys: false,
                do_rebuild_vals: false,
                data: res,
            },
            exec_state,
        );

        Some(map_value)
    }
}

// Given a mapping m1 and a mapping m2, gives a mapping which is like m1(m2(x)).
// In other words maps through m2 and then m1. For example, if m1 maps a to b and m2 maps c to a, then the output maps c to b.
fn compose(m1: &BTreeMap<Value, Value>, m2: &BTreeMap<Value, Value>) -> BTreeMap<Value, Value> {
    let mut res = BTreeMap::new();
    for (k, v) in m2.iter() {
        if let Some(v2) = m1.get(v) {
            res.insert(*k, *v2);
        }
    }
    res
}

fn invert(m1: &BTreeMap<Value, Value>) -> BTreeMap<Value, Value> {
    let mut res = BTreeMap::new();
    for (k, v) in m1.iter() {
        res.insert(*v, *k);
    }
    res
}
