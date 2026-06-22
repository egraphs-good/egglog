use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetContainer {
    pub do_rebuild: bool,
    pub data: BTreeSet<Value>,
}

impl ContainerValue for SetContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        if self.do_rebuild {
            let mut xs: Vec<_> = self.data.iter().copied().collect();
            let changed = rebuilder.rebuild_slice(&mut xs);
            self.data = xs.into_iter().collect();
            changed
        } else {
            false
        }
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.data.iter().copied()
    }
}

#[derive(Clone, Debug)]
pub struct SetSort {
    name: String,
    element: ArcSort,
}

impl SetSort {
    pub fn element(&self) -> ArcSort {
        self.element.clone()
    }
}

impl Presort for SetSort {
    fn presort_name() -> &'static str {
        "Set"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec![
            "set-of",
            "set-empty",
            "set-insert",
            "set-not-contains",
            "set-contains",
            "set-remove",
            "set-union",
            "set-diff",
            "set-intersect",
            "set-get",
            "set-length",
        ]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError> {
        if let [Expr::Var(span, e)] = args {
            let e = typeinfo
                .get_sort_by_name(e)
                .ok_or(TypeError::UndefinedSort(e.clone(), span.clone()))?;

            let out = Self {
                name,
                element: e.clone(),
            };
            Ok(out.to_arcsort())
        } else {
            panic!()
        }
    }
}

impl ContainerSort for SetSort {
    type Container = SetContainer;

    fn name(&self) -> &str {
        &self.name
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        vec![self.element.clone()]
    }

    fn is_eq_container_sort(&self) -> bool {
        self.element.is_eq_sort() || self.element.is_eq_container_sort()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<SetContainer>(value)
            .unwrap()
            .clone();
        val.data
            .iter()
            .map(|e| (self.element.clone(), *e))
            .collect()
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        let arc = self.clone().to_arcsort();

        // Proof term form of a set: `(set-of e0 e1 ...)`, matching
        // `reconstruct_termdag`. (Element dedup/ordering for proof checking of
        // collapsing sets is refined in the Set proof stage.)
        let set_of_validator = |termdag: &mut TermDag, args: &[TermId]| -> Option<TermId> {
            let raw = termdag.app("set-of".into(), args.to_vec());
            Some(termdag.normalize_container_term(raw))
        };
        let set_empty_validator = |termdag: &mut TermDag, _args: &[TermId]| -> Option<TermId> {
            Some(termdag.app("set-of".into(), vec![]))
        };

        add_primitive_with_validator!(eg, "set-empty" = {self.clone(): SetSort} |                      | -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: BTreeSet::new()
        } }, set_empty_validator);
        add_primitive_with_validator!(eg, "set-of"    = {self.clone(): SetSort} [xs: # (self.element())] -> @SetContainer (arc) { SetContainer {
            do_rebuild: self.ctx.is_eq_container_sort(),
            data: xs.collect()
        } }, set_of_validator);

        add_primitive!(eg, "set-get" = |xs: @SetContainer (arc), i: i64| -?> # (self.element()) { xs.data.iter().nth(i as usize).copied() });
        add_primitive!(eg, "set-insert" = |mut xs: @SetContainer (arc), x: # (self.element())| -> @SetContainer (arc) {{ xs.data.insert( x); xs }});
        add_primitive!(eg, "set-remove" = |mut xs: @SetContainer (arc), x: # (self.element())| -> @SetContainer (arc) {{ xs.data.remove(&x); xs }});

        add_primitive!(eg, "set-length"       = |xs: @SetContainer (arc)| -> i64 { xs.data.len() as i64 });
        add_primitive!(eg, "set-contains"     = |xs: @SetContainer (arc), x: # (self.element())| -?> () { ( xs.data.contains(&x)).then_some(()) });
        add_primitive!(eg, "set-not-contains" = |xs: @SetContainer (arc), x: # (self.element())| -?> () { (!xs.data.contains(&x)).then_some(()) });

        add_primitive!(eg, "set-union"      = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{ xs.data.extend(ys.data);                  xs }});
        add_primitive!(eg, "set-diff"       = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{ xs.data.retain(|k| !ys.data.contains(k)); xs }});
        add_primitive!(eg, "set-intersect"  = |mut xs: @SetContainer (arc), ys: @SetContainer (arc)| -> @SetContainer (arc) {{ xs.data.retain(|k|  ys.data.contains(k)); xs }});
    }

    fn reconstruct_termdag(
        &self,
        _container_values: &ContainerValues,
        _value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<TermId>,
    ) -> TermId {
        // Canonical form (sorted by deterministic AST order, deduped) via the
        // shared `normalize_container_term`, so proof checking can reproduce it
        // from terms alone.
        let raw = termdag.app("set-of".into(), element_terms);
        termdag.normalize_container_term(raw)
    }

    fn serialized_name(&self, _container_values: &ContainerValues, _: Value) -> String {
        "set-of".to_owned()
    }
}
