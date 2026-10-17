use crate::{FormatVersion, HashMap, RowId};
use egglog_numeric_id::NumericId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A complete rebuilt database without visualization limits. Class names are
/// arbitrary input-local strings; they are resolved to typed dense IDs at setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Database {
    /// Wire-format version; unrelated to database contents or e-class identity.
    pub version: FormatVersion,
    pub classes: BTreeMap<String, Class>,
    pub functions: BTreeMap<String, Function>,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Class {
    pub sort: String,
    /// A stable, exact encoding of a primitive value, qualified by `sort`.
    /// Equality-sort classes have no literal, including empty classes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub literal: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub kind: FunctionKind,
    pub inputs: Vec<String>,
    pub output: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionKind {
    Constructor,
    Function,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub function: String,
    pub inputs: Vec<String>,
    pub output: String,
    /// Subsumed constructors still denote terms; activity affects database equality.
    #[serde(default)]
    pub subsumed: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported database version {version:?}")]
    UnsupportedVersion { version: FormatVersion },
    #[error("duplicate literal classes {first} and {second}")]
    DuplicateLiteral { first: String, second: String },
    #[error("row {row:?}: undeclared function {function}")]
    UndeclaredFunction { row: RowId, function: String },
    #[error("row {row:?}: wrong arity for {function}: expected {expected}, got {actual}")]
    Arity {
        row: RowId,
        function: String,
        expected: usize,
        actual: usize,
    },
    #[error("row {row:?}: unknown class {class}")]
    UnknownClass { row: RowId, class: String },
    #[error("row {row:?}: class {class} has sort {actual}, expected {expected}")]
    SortMismatch {
        row: RowId,
        class: String,
        expected: String,
        actual: String,
    },
    #[error("row {row:?}: constructor output {class} cannot be a literal")]
    ConstructorLiteral { row: RowId, class: String },
    #[error(
        "row {row:?}: conflicting rows for the same inputs to {function}; rebuild before exporting"
    )]
    ConflictingRows { row: RowId, function: String },
    #[error("disequality without a witness (internal error)")]
    MissingWitness,
    #[error("cannot serialize canonical graph: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl Default for Database {
    fn default() -> Self {
        Self {
            version: FormatVersion::V1,
            classes: BTreeMap::new(),
            functions: BTreeMap::new(),
            rows: Vec::new(),
        }
    }
}

impl Database {
    /// Reject malformed or non-canonical input rather than silently repairing it.
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != FormatVersion::V1 {
            return Err(Error::UnsupportedVersion {
                version: self.version,
            });
        }
        let mut literals = HashMap::default();
        for (id, class) in &self.classes {
            if let Some(value) = &class.literal
                && let Some(previous) = literals.insert((&class.sort, value), id)
            {
                return Err(Error::DuplicateLiteral {
                    first: previous.clone(),
                    second: id.clone(),
                });
            }
        }
        // Borrow the serialized names; repeated row lookups should not traverse
        // a string-keyed tree. The public, ordered serialization stays unchanged.
        let classes: HashMap<_, _> = self
            .classes
            .iter()
            .map(|(id, c)| (id.as_str(), c))
            .collect();
        let mut calls = HashMap::default();
        calls.reserve(self.rows.len());
        for (index, row) in self.rows.iter().enumerate() {
            let function =
                self.functions
                    .get(&row.function)
                    .ok_or_else(|| Error::UndeclaredFunction {
                        row: RowId::from_usize(index),
                        function: row.function.clone(),
                    })?;
            if row.inputs.len() != function.inputs.len() {
                return Err(Error::Arity {
                    row: RowId::from_usize(index),
                    function: row.function.clone(),
                    expected: function.inputs.len(),
                    actual: row.inputs.len(),
                });
            }
            for (id, sort) in row
                .inputs
                .iter()
                .zip(&function.inputs)
                .chain(std::iter::once((&row.output, &function.output)))
            {
                let class = classes
                    .get(id.as_str())
                    .ok_or_else(|| Error::UnknownClass {
                        row: RowId::from_usize(index),
                        class: id.clone(),
                    })?;
                if &class.sort != sort {
                    return Err(Error::SortMismatch {
                        row: RowId::from_usize(index),
                        class: id.clone(),
                        expected: sort.clone(),
                        actual: class.sort.clone(),
                    });
                }
            }
            if function.kind == FunctionKind::Constructor
                && classes[row.output.as_str()].literal.is_some()
            {
                return Err(Error::ConstructorLiteral {
                    row: RowId::from_usize(index),
                    class: row.output.clone(),
                });
            }
            if let Some(previous) =
                calls.insert((&row.function, &row.inputs), (&row.output, row.subsumed))
                && previous != (&row.output, row.subsumed)
            {
                return Err(Error::ConflictingRows {
                    row: RowId::from_usize(index),
                    function: row.function.clone(),
                });
            }
        }
        Ok(())
    }
}
