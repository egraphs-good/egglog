use crate::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Version 1 is a complete, canonicalized database, without visualization limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Database {
    pub version: u32,
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
#[error("{0}")]
pub struct Error(pub String);

impl Default for Database {
    fn default() -> Self {
        Self {
            version: 1,
            classes: BTreeMap::new(),
            functions: BTreeMap::new(),
            rows: Vec::new(),
        }
    }
}

impl Database {
    /// Reject malformed or non-canonical input rather than silently repairing it.
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != 1 {
            return Err(Error(format!(
                "unsupported database version {}",
                self.version
            )));
        }
        let mut literals = HashMap::default();
        for (id, class) in &self.classes {
            if let Some(value) = &class.literal
                && let Some(previous) = literals.insert((&class.sort, value), id)
            {
                return Err(Error(format!(
                    "duplicate literal classes {previous} and {id}"
                )));
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
            let error = |message: String| Error(format!("row {index}: {message}"));
            let function = self
                .functions
                .get(&row.function)
                .ok_or_else(|| error(format!("undeclared function {}", row.function)))?;
            if row.inputs.len() != function.inputs.len() {
                return Err(error(format!("wrong arity for {}", row.function)));
            }
            for (id, sort) in row
                .inputs
                .iter()
                .zip(&function.inputs)
                .chain(std::iter::once((&row.output, &function.output)))
            {
                let class = classes
                    .get(id.as_str())
                    .ok_or_else(|| error(format!("unknown class {id}")))?;
                if &class.sort != sort {
                    return Err(error(format!(
                        "class {id} has sort {}, expected {sort}",
                        class.sort
                    )));
                }
            }
            if function.kind == FunctionKind::Constructor
                && classes[row.output.as_str()].literal.is_some()
            {
                return Err(error("constructor output cannot be a literal".into()));
            }
            if let Some(previous) =
                calls.insert((&row.function, &row.inputs), (&row.output, row.subsumed))
                && previous != (&row.output, row.subsumed)
            {
                return Err(error(
                    "conflicting rows for the same function inputs; rebuild before exporting"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}
