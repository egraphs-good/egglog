use egglog_bridge::EGraph;
use egglog_core_relations::{ExternalFunctionId, Rebuilder, Value, make_external_func};
use smallvec::SmallVec;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct FreeVarSet {
    values: SmallVec<[Value; 4]>,
}

impl FreeVarSet {
    pub fn empty() -> Self {
        Self {
            values: SmallVec::new(),
        }
    }

    pub fn singleton(value: Value) -> Self {
        let mut values = SmallVec::new();
        values.push(value);
        Self { values }
    }

    pub fn remove(&mut self, value: Value) -> bool {
        match self.values.binary_search(&value) {
            Ok(idx) => {
                self.values.remove(idx);
                true
            }
            Err(_) => false,
        }
    }

    pub fn union_with(&mut self, other: &FreeVarSet) -> bool {
        if other.values.is_empty() {
            return false;
        }
        if self.values.is_empty() {
            self.values = other.values.clone();
            return true;
        }

        let mut merged =
            SmallVec::<[Value; 4]>::with_capacity(self.values.len() + other.values.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.values.len() && j < other.values.len() {
            let left = self.values[i];
            let right = other.values[j];
            match left.cmp(&right) {
                std::cmp::Ordering::Less => {
                    merged.push(left);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(right);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    merged.push(left);
                    i += 1;
                    j += 1;
                }
            }
        }
        if i < self.values.len() {
            merged.extend_from_slice(&self.values[i..]);
        }
        if j < other.values.len() {
            merged.extend_from_slice(&other.values[j..]);
        }

        if merged == self.values {
            false
        } else {
            self.values = merged;
            true
        }
    }

    fn normalize(&mut self) {
        self.values.sort_unstable();
        self.values.dedup();
    }
}

pub struct FreeVarSetExternalFns {
    pub empty: ExternalFunctionId,
    pub singleton: ExternalFunctionId,
    pub remove: ExternalFunctionId,
    pub union: ExternalFunctionId,
}

pub fn register_free_var_set_functions(egraph: &mut EGraph) -> FreeVarSetExternalFns {
    egraph.register_container_ty::<FreeVarSet>();

    let empty = egraph.register_external_func(Box::new(make_external_func(|state, vals| {
        if !vals.is_empty() {
            panic!("[free-var-set-empty] expected 0 values, got {vals:?}");
        }
        Some(
            state
                .container_values()
                .register_val(FreeVarSet::empty(), state),
        )
    })));

    let singleton = egraph.register_external_func(Box::new(make_external_func(|state, vals| {
        let [value] = vals else {
            panic!("[free-var-set-singleton] expected 1 value, got {vals:?}");
        };
        let set = FreeVarSet::singleton(*value);
        Some(state.container_values().register_val(set, state))
    })));

    let remove = egraph.register_external_func(Box::new(make_external_func(|state, vals| {
        let [set_id, value] = vals else {
            panic!("[free-var-set-remove] expected 2 values, got {vals:?}");
        };
        let mut set: FreeVarSet = state
            .container_values()
            .get_val::<FreeVarSet>(*set_id)?
            .clone();
        set.remove(*value);
        Some(state.container_values().register_val(set, state))
    })));

    let union = egraph.register_external_func(Box::new(make_external_func(|state, vals| {
        let [left_id, right_id] = vals else {
            panic!("[free-var-set-union] expected 2 values, got {vals:?}");
        };
        let mut left: FreeVarSet = state
            .container_values()
            .get_val::<FreeVarSet>(*left_id)
            .expect("l")
            .clone();
        let right: FreeVarSet = state
            .container_values()
            .get_val::<FreeVarSet>(*right_id)
            .expect("r")
            .clone();
        left.union_with(&right);
        Some(state.container_values().register_val(left, state))
    })));

    FreeVarSetExternalFns {
        empty,
        singleton,
        remove,
        union,
    }
}

impl Default for FreeVarSet {
    fn default() -> Self {
        Self::empty()
    }
}

impl egglog_core_relations::ContainerValue for FreeVarSet {
    fn rebuild_contents(&mut self, rebuilder: &dyn Rebuilder) -> bool {
        let res = rebuilder.rebuild_slice(&mut self.values);
        self.normalize();
        res
    }

    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.values.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{FreeVarSet, register_free_var_set_functions};
    use egglog_bridge::EGraph;
    use egglog_core_relations::{ColumnId, ContainerValue, DisplacedTable, Table, Value};
    use egglog_numeric_id::NumericId;
    use smallvec::smallvec;

    fn collect_values(set: &FreeVarSet) -> Vec<Value> {
        set.iter().collect()
    }

    #[test]
    fn empty_is_empty() {
        let set = FreeVarSet::empty();
        assert!(collect_values(&set).is_empty());
    }

    #[test]
    fn singleton_contains_value() {
        let value = Value::new(3);
        let set = FreeVarSet::singleton(value);
        assert_eq!(collect_values(&set), vec![value]);
    }

    #[test]
    fn normalize_orders_and_dedups() {
        let mut set = FreeVarSet {
            values: smallvec![Value::new(3), Value::new(1), Value::new(2), Value::new(2)],
        };
        set.normalize();
        assert_eq!(
            collect_values(&set),
            vec![Value::new(1), Value::new(2), Value::new(3)]
        );
    }

    #[test]
    fn remove_eliminates_value() {
        let mut set = FreeVarSet {
            values: smallvec![Value::new(1), Value::new(2)],
        };
        assert!(set.remove(Value::new(1)));
        assert!(!set.remove(Value::new(1)));
        assert_eq!(collect_values(&set), vec![Value::new(2)]);
    }

    #[test]
    fn union_merges_sets() {
        let mut left = FreeVarSet {
            values: smallvec![Value::new(2), Value::new(4)],
        };
        let right = FreeVarSet {
            values: smallvec![Value::new(1), Value::new(4)],
        };
        assert!(left.union_with(&right));
        assert_eq!(
            collect_values(&left),
            vec![Value::new(1), Value::new(2), Value::new(4)]
        );
    }

    #[test]
    fn rebuild_sorts_and_dedups() {
        // Bit of a hack: we just need a placeholder execution state to propagate updates to the
        // DisplacedTable
        let egraph = EGraph::default();
        let mut set = FreeVarSet {
            values: smallvec![Value::new(1), Value::new(3), Value::new(4)],
        };
        let mut table = DisplacedTable::default();
        table
            .new_buffer()
            .stage_insert(&[Value::new(3), Value::new(0), Value::new(0)]);
        egraph.with_execution_state(|es| table.merge(es));
        let rebuilder = table
            .rebuilder(&[ColumnId::new(0)])
            .expect("rebuilder should be available");
        let changed = set.rebuild_contents(rebuilder.as_ref());
        assert!(changed);
        assert_eq!(
            collect_values(&set),
            vec![Value::new(0), Value::new(1), Value::new(4)]
        );
    }

    #[test]
    fn external_empty_set() {
        let mut egraph = EGraph::default();
        let funcs = register_free_var_set_functions(&mut egraph);
        let set_id = egraph
            .with_execution_state(|state| state.call_external_func(funcs.empty, &[]))
            .expect("expected free-var-set-empty result");
        let set = egraph
            .container_values()
            .get_val::<FreeVarSet>(set_id)
            .expect("expected container value for empty set");
        assert!(collect_values(&set).is_empty());
    }

    #[test]
    fn external_singleton_set() {
        let mut egraph = EGraph::default();
        let funcs = register_free_var_set_functions(&mut egraph);
        let value = Value::new(7);
        let set_id = egraph
            .with_execution_state(|state| state.call_external_func(funcs.singleton, &[value]))
            .expect("expected free-var-set-singleton result");
        let set = egraph
            .container_values()
            .get_val::<FreeVarSet>(set_id)
            .expect("expected container value for singleton set");
        assert_eq!(collect_values(&set), vec![value]);
    }

    #[test]
    fn external_remove_set() {
        let mut egraph = EGraph::default();
        let funcs = register_free_var_set_functions(&mut egraph);
        let value = Value::new(9);
        let set_id = egraph
            .with_execution_state(|state| state.call_external_func(funcs.singleton, &[value]))
            .expect("expected free-var-set-singleton result");
        let updated_id = egraph
            .with_execution_state(|state| state.call_external_func(funcs.remove, &[set_id, value]))
            .expect("expected free-var-set-remove result");
        let set = egraph
            .container_values()
            .get_val::<FreeVarSet>(updated_id)
            .expect("expected container value for removed set");
        assert!(collect_values(&set).is_empty());
    }

    #[test]
    fn external_union_set() {
        let mut egraph = EGraph::default();
        let funcs = register_free_var_set_functions(&mut egraph);
        let set_left = egraph
            .with_execution_state(|state| {
                state.call_external_func(funcs.singleton, &[Value::new(1)])
            })
            .expect("expected free-var-set-singleton result");
        let set_right = egraph
            .with_execution_state(|state| {
                state.call_external_func(funcs.singleton, &[Value::new(2)])
            })
            .expect("expected free-var-set-singleton result");
        let union_id = egraph
            .with_execution_state(|state| {
                state.call_external_func(funcs.union, &[set_left, set_right])
            })
            .expect("expected free-var-set-union result");
        let set = egraph
            .container_values()
            .get_val::<FreeVarSet>(union_id)
            .expect("expected container value for union set");
        assert_eq!(collect_values(&set), vec![Value::new(1), Value::new(2)]);
    }
}
