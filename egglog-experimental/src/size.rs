use std::{collections::HashSet, convert::TryFrom};

use egglog::{
    ExecutionState, Primitive, Value,
    constraint::{AllEqualTypeConstraint, TypeConstraint},
    prelude::BaseSort,
    prelude::{I64Sort, Span, StringSort},
    sort::S,
    util::INTERNAL_SYMBOL_PREFIX,
};

#[derive(Clone)]
pub struct GetSizePrimitive;

impl Primitive for GetSizePrimitive {
    fn name(&self) -> &str {
        "get-size!"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_output_sort(I64Sort.to_arcsort())
            .with_all_arguments_sort(StringSort.to_arcsort())
            .into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let filters: Option<HashSet<String>> = if args.is_empty() {
            None
        } else {
            Some(
                args.iter()
                    .map(|value| exec_state.base_values().unwrap::<S>(*value).0)
                    .collect::<HashSet<_>>(),
            )
        };

        let total_size: usize = exec_state
            .table_ids()
            .filter_map(|table_id| {
                let name = exec_state.table_name(table_id)?;
                if name.starts_with(INTERNAL_SYMBOL_PREFIX) {
                    return None;
                }
                if let Some(filter) = &filters
                    && !filter.contains(name)
                {
                    return None;
                }
                Some(exec_state.get_table(table_id).len())
            })
            .sum();
        let total_size = i64::try_from(total_size).ok()?;
        Some(exec_state.base_values().get::<i64>(total_size))
    }
}
