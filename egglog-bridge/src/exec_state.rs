//! [`ActionRegistry`]: the live mapping from table name to
//! [`TableAction`] (plus the shared [`UnionAction`] and default-panic
//! external-function id) that backs the name-indexed action methods on
//! the egglog crate's typed state wrappers.
//!
//! The state wrappers themselves (`PureState`/`ReadState`/`WriteState`/
//! `FullState`) live in the `egglog` crate so their privileged seams
//! (raw `ExecutionState` access, external-function dispatch, raw table
//! reads) can be `pub(crate)`-sealed from external users. This file
//! holds only the registry, which the bridge mutates as new tables are
//! added.

use std::collections::HashMap;

use crate::core_relations::ExternalFunctionId;

use crate::{TableAction, UnionAction};

/// A live registry of action handles for use by typed primitives.
/// Owned by the bridge `EGraph` and shared with action-side state
/// wrappers (in the `egglog` crate) at invoke time.
#[derive(Clone)]
pub struct ActionRegistry {
    table_actions: HashMap<String, TableAction>,
    union_action: UnionAction,
    default_panic_id: ExternalFunctionId,
}

impl ActionRegistry {
    pub(crate) fn new(union_action: UnionAction, default_panic_id: ExternalFunctionId) -> Self {
        Self {
            table_actions: HashMap::new(),
            union_action,
            default_panic_id,
        }
    }

    pub(crate) fn register_table(&mut self, name: String, action: TableAction) {
        self.table_actions.insert(name, action);
    }

    /// Look up the [`TableAction`] for a table by name, or `None` if
    /// no table with that name has been registered.
    pub fn lookup_table(&self, name: &str) -> Option<&TableAction> {
        self.table_actions.get(name)
    }

    /// The shared [`UnionAction`] for this EGraph's union-find.
    pub fn union_action(&self) -> &UnionAction {
        &self.union_action
    }

    /// The default panic external function id, used by the egglog
    /// crate's `ActionView::panic`.
    pub fn default_panic_id(&self) -> ExternalFunctionId {
        self.default_panic_id
    }
}
