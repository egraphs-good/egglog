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
