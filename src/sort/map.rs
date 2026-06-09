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
        self.key.is_eq_sort()
            || self.value.is_eq_sort()
            || self.key.is_eq_container_sort()
            || self.value.is_eq_container_sort()
    }

    fn proof_normalizes(&self) -> bool {
        true
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

        // Proof term form of a map: nested `(map-insert (map-empty) k v ...)`,
        // matching `reconstruct_termdag`. (Key collapse / last-write-wins for
        // proof checking is refined in the Map proof stage.)
        let map_empty_validator = |termdag: &mut TermDag, _args: &[TermId]| -> Option<TermId> {
            Some(termdag.app("map-empty".into(), vec![]))
        };
        let map_insert_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            if args.len() != 3 {
                return None;
            }
            let raw = termdag.app("map-insert".into(), args.to_vec());
            Some(termdag.normalize_container_term(raw))
        };

        add_primitive_with_validator!(eg, "map-empty" = {self.clone(): MapSort} || -> @MapContainer (arc) { MapContainer {
            do_rebuild_keys: self.ctx.key.is_eq_sort() || self.ctx.key.is_eq_container_sort(),
            do_rebuild_vals: self.ctx.value.is_eq_sort() || self.ctx.value.is_eq_container_sort(),
            data: BTreeMap::new()
        } }, map_empty_validator);

        add_primitive!(eg, "map-get"    = |    xs: @MapContainer (arc), x: # (self.key())                     | -?> # (self.value()) { xs.data.get(&x).copied() });
        add_primitive_with_validator!(eg, "map-insert" = |mut xs: @MapContainer (arc), x: # (self.key()), y: # (self.value())| -> @MapContainer (arc) {{ xs.data.insert(x, y); xs }}, map_insert_validator);
        add_primitive!(eg, "map-remove" = |mut xs: @MapContainer (arc), x: # (self.key())                     | -> @MapContainer (arc) {{ xs.data.remove(&x);   xs }});

        add_primitive!(eg, "map-length"       = |xs: @MapContainer (arc)| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "map-contains"     = |xs: @MapContainer (arc), x: # (self.key())| -?> () { ( xs.data.contains_key(&x)).then_some(()) });
        add_primitive!(eg, "map-not-contains" = |xs: @MapContainer (arc), x: # (self.key())| -?> () { (!xs.data.contains_key(&x)).then_some(()) });
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        // element_terms is [k0, v0, k1, v1, ...] (unique keys). Build the nested
        // insert chain in deterministic AST key order so proof checking can
        // reproduce it from terms alone.
        let mut pairs: Vec<(TermId, TermId)> =
            element_terms.chunks(2).map(|c| (c[0], c[1])).collect();
        pairs.sort_by(|(ka, _), (kb, _)| termdag.ast_cmp(*ka, *kb));
        let mut term = termdag.app("map-empty".into(), vec![]);
        for (k, v) in pairs {
            term = termdag.app("map-insert".into(), vec![term, k, v]);
        }
        term
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        self.name().to_owned()
    }
}
