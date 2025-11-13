use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::hash::Hash;

use ordered_float::OrderedFloat;
use schemars::JsonSchema;

use crate::span::Span;

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Deserialize, Serialize, JsonSchema,
)]
#[serde(untagged)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Bool(bool),
    Unit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(_unstable_ref_variants)]
pub enum GenericExpr<Head, Leaf> {
    Var {
        span: Span,
        name: Leaf,
    },
    Call {
        field1: Span,
        field2: Head,
        field3: Vec<GenericExpr<Head, Leaf>>,
    },
    Lit {
        field1: Span,
        field2: Literal,
    },
}

/// Facts are the left-hand side of a [`Command::Rule`].
/// They represent a part of a database query.
/// Facts can be expressions or equality constraints between expressions.
///
/// Note that primitives such as  `!=` are partial.
/// When two things are equal, it returns nothing and the query does not match.
/// For example, the following egglog code runs:
/// ```text
/// (fail (check (!= 1 1)))
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(_unstable_ref_variants)]
pub enum GenericFact<Head, Leaf> {
    Eq {
        field1: Span,
        field2: GenericExpr<Head, Leaf>,
        field3: GenericExpr<Head, Leaf>,
    },
    Fact {
        field1: GenericExpr<Head, Leaf>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
pub struct GenericActions<Head: Clone + Display, Leaf: Clone + PartialEq + Eq + Display + Hash>(
    pub Vec<GenericAction<Head, Leaf>>,
);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(_unstable_ref_variants)]
pub enum GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// Bind a variable to a particular datatype or primitive.
    /// At the top level (in a [`Command::Action`]), this defines a global variable.
    /// In a [`Command::Rule`], this defines a local variable in the actions.
    Let {
        field1: Span,
        field2: Leaf,
        field3: GenericExpr<Head, Leaf>,
    },
    /// `set` a function to a particular result.
    /// `set` should not be used on datatypes-
    /// instead, use `union`.
    Set {
        field1: Span,
        field2: Head,
        field3: Vec<GenericExpr<Head, Leaf>>,
        field4: GenericExpr<Head, Leaf>,
    },
    /// Delete or subsume (mark as hidden from future rewrites and unextractable) an entry from a function.
    Change {
        field1: Span,
        field2: Change,
        field3: Head,
        field4: Vec<GenericExpr<Head, Leaf>>,
    },
    /// `union` two datatypes, making them equal
    /// in the implicit, global equality relation
    /// of egglog.
    /// All rules match modulo this equality relation.
    ///
    /// Example:
    /// ```text
    /// (datatype Math (Num i64))
    /// (union (Num 1) (Num 2)); Define that Num 1 and Num 2 are equivalent
    /// (extract (Num 1)); Extracts Num 1
    /// (extract (Num 2)); Extracts Num 1
    /// ```
    Union {
        field1: Span,
        field2: GenericExpr<Head, Leaf>,
        field3: GenericExpr<Head, Leaf>,
    },
    Panic {
        field1: Span,
        field2: String,
    },
    Expr {
        field1: Span,
        field2: GenericExpr<Head, Leaf>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
pub struct GenericRule<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub span: Span,
    pub head: GenericActions<Head, Leaf>,
    pub body: Vec<GenericFact<Head, Leaf>>,
    /// A globally unique name for this rule in the EGraph.
    pub name: String,
    /// The ruleset this rule belongs to. Defaults to `""`.
    pub ruleset: String,
}

/// Change a function entry.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
#[schemars(untagged)]
pub enum Change {
    /// `delete` this entry from a function.
    /// Be wary! Only delete entries that are guaranteed to be not useful.
    Delete,
    /// `subsume` this entry so that it cannot be queried or extracted, but still can be checked.
    /// Note that this is currently forbidden for functions with custom merges.
    Subsume,
}
