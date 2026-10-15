//! A query-source API experiment, not an integration with the query planner.
//!
//! Run with `cargo run -p egglog-core-relations --example forward_query_source`.
//! Stored and computed relations can share forward lookup without sharing the
//! storage contract: a primitive has no enumerable `all()` subset, stable row
//! IDs, or persistent indexes. Implementing `Table` with dummy versions of those
//! operations would violate assumptions in query preparation and execution.

use egglog_core_relations::{
    ColumnId, Database, ExecutionState, ExternalFunctionId, SortedWritesTable, Value, WrappedTable,
    make_external_func,
};
use egglog_numeric_id::NumericId;

#[derive(Debug, PartialEq, Eq)]
enum AccessMode {
    Enumerable,
    /// All input columns must be bound before evaluating this relation.
    BoundInputs {
        n_inputs: usize,
    },
}

/// Both variants satisfy keys/inputs -> values/output. That functional
/// dependency expresses uniqueness, not permission to enumerate or invert a
/// function. A planner must additionally respect `access_mode()`.
enum QuerySource<'a> {
    Stored(&'a WrappedTable),
    /// The caller must register a query-safe callback. `ExternalFunction` itself
    /// permits writes; the frontend's Pure/Read contexts supply that restriction.
    Computed {
        function: ExternalFunctionId,
        n_inputs: usize,
    },
}

#[derive(Debug, PartialEq, Eq)]
struct ArityError {
    expected: usize,
    actual: usize,
}

impl QuerySource<'_> {
    fn access_mode(&self) -> AccessMode {
        match self {
            Self::Stored(_) => AccessMode::Enumerable,
            Self::Computed { n_inputs, .. } => AccessMode::BoundInputs {
                n_inputs: *n_inputs,
            },
        }
    }

    fn n_inputs(&self) -> usize {
        match self {
            Self::Stored(table) => table.spec().n_keys,
            Self::Computed { n_inputs, .. } => *n_inputs,
        }
    }

    /// Return the complete row, inputs followed by outputs, without a RowId.
    /// `None` means a missing stored row or a failed partial primitive; both
    /// eliminate the current candidate binding in a future lookup join stage.
    fn lookup(
        &self,
        state: &mut ExecutionState<'_>,
        inputs: &[Value],
    ) -> Result<Option<Vec<Value>>, ArityError> {
        if inputs.len() != self.n_inputs() {
            return Err(ArityError {
                expected: self.n_inputs(),
                actual: inputs.len(),
            });
        }
        Ok(match self {
            Self::Stored(table) => table.get_row(inputs).map(|row| row.vals.to_vec()),
            Self::Computed { function, .. } => {
                state.call_external_func(*function, inputs).map(|output| {
                    let mut row = Vec::with_capacity(inputs.len() + 1);
                    row.extend_from_slice(inputs);
                    row.push(output);
                    row
                })
            }
        })
    }
}

#[allow(clippy::disallowed_macros)]
fn main() {
    let mut db = Database::new();
    let table = db.add_table(
        SortedWritesTable::new(
            1,
            2,
            ColumnId::new(0),
            vec![],
            Box::new(|_, old, new, _| {
                assert_eq!(old, new);
                false
            }),
        ),
        [],
        [],
    );
    {
        let mut rows = db.new_buffer(table);
        for input in 1..=3 {
            rows.stage_insert(&[Value::from_usize(input), Value::from_usize(input - 1)]);
        }
    }
    db.merge_all();

    let predecessor = db.add_external_function(Box::new(make_external_func(|_, args| {
        args[0].index().checked_sub(1).map(Value::from_usize)
    })));
    let constant = db.add_external_function(Box::new(make_external_func(|_, args| {
        assert!(args.is_empty());
        Some(Value::from_usize(42))
    })));

    db.with_execution_state(None, |state| {
        let stored = QuerySource::Stored(db.get_table(table));
        let computed = QuerySource::Computed {
            function: predecessor,
            n_inputs: 1,
        };
        assert_eq!(stored.access_mode(), AccessMode::Enumerable);
        assert_eq!(
            computed.access_mode(),
            AccessMode::BoundInputs { n_inputs: 1 }
        );

        for input in 1..=3 {
            let key = [Value::from_usize(input)];
            let expected = Some(vec![key[0], Value::from_usize(input - 1)]);
            assert_eq!(stored.lookup(state, &key), Ok(expected.clone()));
            assert_eq!(computed.lookup(state, &key), Ok(expected));
        }

        // A missing stored row and an undefined primitive both prune.
        for source in [&stored, &computed] {
            assert_eq!(source.lookup(state, &[Value::new(0)]), Ok(None));
            assert_eq!(
                source.lookup(state, &[]),
                Err(ArityError {
                    expected: 1,
                    actual: 0,
                })
            );
            assert_eq!(
                source.lookup(state, &[Value::new(1), Value::new(2)]),
                Err(ArityError {
                    expected: 1,
                    actual: 2,
                })
            );
        }

        // The computed relation does not depend on pre-materialized rows.
        assert_eq!(stored.lookup(state, &[Value::new(4)]), Ok(None));
        assert_eq!(
            computed.lookup(state, &[Value::new(4)]),
            Ok(Some(vec![Value::new(4), Value::new(3)]))
        );

        // An empty prerequisite set is immediately ready. FD closure alone
        // cannot establish this: FunDeps currently discards empty antecedents.
        let nullary = QuerySource::Computed {
            function: constant,
            n_inputs: 0,
        };
        assert_eq!(
            nullary.access_mode(),
            AccessMode::BoundInputs { n_inputs: 0 }
        );
        assert_eq!(nullary.lookup(state, &[]), Ok(Some(vec![Value::new(42)])));
    });

    println!("Forward lookup assertions passed; no planner behavior was changed.");
}
