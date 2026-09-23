use std::fmt::Display;
use std::hash::Hash;

use ordered_float::OrderedFloat;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::span::Span;

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Serialize, Deserialize, JsonSchema,
)]
#[serde(tag = "type", content = "value", deny_unknown_fields)]
pub enum Literal {
    Int(
        #[serde(with = "integer_literal")]
        #[schemars(with = "String", regex(pattern = "^-?(0|[1-9][0-9]*)$"))]
        i64,
    ),
    Float(
        #[serde(with = "float_literal")]
        #[schemars(with = "String", regex(pattern = "^[0-9a-f]{16}$"))]
        OrderedFloat<f64>,
    ),
    String(String),
    Bool(bool),
    Unit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "value", deny_unknown_fields)]
pub enum GenericExpr<Head, Leaf> {
    Var(Span, Leaf),
    Call(Span, Head, Vec<GenericExpr<Head, Leaf>>),
    Lit(Span, Literal),
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "value", deny_unknown_fields)]
pub enum GenericFact<Head, Leaf> {
    Eq(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    Fact(GenericExpr<Head, Leaf>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct GenericActions<Head: Clone + Display, Leaf: Clone + PartialEq + Eq + Display + Hash>(
    pub Vec<GenericAction<Head, Leaf>>,
);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "value", deny_unknown_fields)]
pub enum GenericAction<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    /// Bind a variable to a particular datatype or primitive.
    /// At the top level (in a [`Command::Action`]), this defines a global variable.
    /// In a [`Command::Rule`], this defines a local variable in the actions.
    Let(Span, Leaf, GenericExpr<Head, Leaf>),
    /// `set` a function to a particular result.
    /// `set` should not be used on datatypes-
    /// instead, use `union`.
    Set(
        Span,
        Head,
        Vec<GenericExpr<Head, Leaf>>,
        GenericExpr<Head, Leaf>,
    ),
    /// Delete or subsume (mark as hidden from future rewrites and unextractable) an entry from a function.
    Change(Span, Change, Head, Vec<GenericExpr<Head, Leaf>>),
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
    Union(Span, GenericExpr<Head, Leaf>, GenericExpr<Head, Leaf>),
    Panic(Span, String),
    Expr(Span, GenericExpr<Head, Leaf>),
}

/// How a rule is evaluated. The three modes are mutually exclusive, so they
/// share one field on [`GenericRule`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum RuleEvalMode {
    /// Default: seminaive (delta) evaluation with restrictive `Pure`/`Write`
    /// primitive contexts (no database reads in the RHS).
    #[default]
    Seminaive,
    /// `:naive`: match the whole database every iteration, with permissive
    /// `Read`/`Full` contexts so the RHS may read the database.
    Naive,
    /// `:unsafe-seminaive`: like `:naive`'s `Read`/`Full` contexts (the RHS may
    /// read the database) but keeps delta evaluation. **Unsafe**: an RHS read
    /// observes the database mid-iteration and isn't re-evaluated if it changes.
    UnsafeSeminaive,
}

impl RuleEvalMode {
    /// `:naive` — disables seminaive evaluation (unlike `:unsafe-seminaive`).
    pub fn is_naive(self) -> bool {
        matches!(self, RuleEvalMode::Naive)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    /// How this rule is evaluated; set by `:naive` / `:unsafe-seminaive`.
    pub eval_mode: RuleEvalMode,
    /// If `true`, this rule skips tree-decomposition during query
    /// planning and evaluate rules as a single-bag (without decomposing
    /// it into smaller queries). Set via the `:no-decomp` rule option.
    pub no_decomp: bool,
    /// If `true`, table atoms in this rule match subsumed rows as well as
    /// live rows. This is intended for internal maintenance rules, not
    /// ordinary user rewrites.
    pub include_subsumed: bool,
}

/// Change a function entry.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum Change {
    /// `delete` this entry from a function.
    /// Be wary! Only delete entries that are guaranteed to be not useful.
    Delete,
    /// `subsume` this entry so that it cannot be queried or extracted, but still can be checked.
    /// Note that this is currently forbidden for functions with custom merges.
    Subsume,
}

// JSON numbers cannot represent every i64 in all consumers. The tagged string
// representation also keeps integer literals distinct from floating literals.
mod integer_literal {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(value: &i64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        let text = String::deserialize(deserializer)?;
        let value: i64 = text.parse().map_err(D::Error::custom)?;
        if value.to_string() != text {
            return Err(D::Error::custom("expected a canonical decimal i64 string"));
        }
        Ok(value)
    }
}

// Serializing the bits preserves signed zero, infinities, and NaN payloads.
mod float_literal {
    use ordered_float::OrderedFloat;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(
        value: &OrderedFloat<f64>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{:016x}", value.0.to_bits()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<OrderedFloat<f64>, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.len() != 16
            || !text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(D::Error::custom(
                "expected 16 lowercase hexadecimal f64 bits",
            ));
        }
        let bits = u64::from_str_radix(&text, 16).map_err(D::Error::custom)?;
        Ok(OrderedFloat(f64::from_bits(bits)))
    }
}
