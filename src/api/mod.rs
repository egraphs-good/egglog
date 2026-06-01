//! Row encoding for the [`crate::EGraph`] surface API.
//!
//! [`IntoRow`] / [`IntoColumn`] convert Rust values into egglog row data on
//! the way *in* (insert, lookup keys, query patterns).  [`FromRow`] /
//! [`FromColumn`] convert row data back into Rust values on the way *out*
//! (lookup return values, query iteration).
//!
//! Every input column carries a runtime [`ColumnSort`] tag. The trait
//! methods that consume rows ([`crate::Read::lookup`],
//! [`crate::Write::set`], etc.) validate each column's tag against the
//! table's declared schema and return [`crate::ApiError::WrongColumnSort`]
//! on mismatch. Base values (`i64`, `String`, `f64`, …) tag with the
//! corresponding egglog sort name automatically; eclass ids flow through
//! as bare [`Value`]s, which carry no sort tag — callers track their
//! eclass-vs-base provenance themselves.

use std::sync::Arc;

use crate::core_relations::{BaseValues, Value};
use crate::sort;
use thiserror::Error;

// ---------------------------------------------------------------------
// ApiError — runtime check failures from the `Read` / `Write` trait
// methods and from [`crate::EGraph::with_full_state`] callers.
// ---------------------------------------------------------------------

/// Runtime errors from the Rust API surface.
///
/// These signal a misuse the API can detect at runtime — wrong table
/// subtype, wrong arity, mismatched column sorts, etc. They are *not*
/// egglog typecheck errors and *not* backend / e-graph failures; for
/// those, see [`enum@crate::Error`].
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("no table named `{name}` is registered")]
    MissingTable { name: String },

    #[error(
        "table `{name}` is a {actual}; this method is only valid for {expected} tables"
    )]
    WrongSubtype {
        name: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("table `{table}`: expected {expected} input columns, got {got}")]
    WrongArity {
        table: String,
        expected: usize,
        got: usize,
    },

    #[error(
        "table `{table}`, column {column}: expected sort `{expected}`, got value of sort `{actual}`"
    )]
    WrongColumnSort {
        table: String,
        column: usize,
        expected: String,
        actual: String,
    },

    #[error(
        "table `{table}` output: expected sort `{expected}`, got value of sort `{actual}`"
    )]
    WrongOutputSort {
        table: String,
        expected: String,
        actual: String,
    },
}

mod sealed {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------
// Input side: IntoRow + IntoColumn
// ---------------------------------------------------------------------

/// Runtime sort tag for a column on the input side. Used by the
/// `Read` / `Write` trait methods to validate inputs against a
/// table's declared schema.
#[derive(Clone, Debug)]
pub enum ColumnSort {
    /// The column has a known sort name. Compared exactly against the
    /// table's expected sort for that column position.
    Named(Arc<str>),
    /// No sort information attached. Skip the runtime check — the
    /// caller is responsible for getting the column right. Used by
    /// the bare [`Value`] [`IntoColumn`] impl.
    Unchecked,
}

/// Convert a Rust value into a row of egglog [`Value`]s, with a sort
/// tag per column.
///
/// Implemented for:
/// - A bare [`IntoColumn`] value (e.g. `1_i64` or a [`Value`]) —
///   produces a single-column row.
/// - Tuples up to arity 8 of [`IntoColumn`] values.
/// - [`RawValues`] as an escape hatch for already-converted multi-column
///   rows (sort checks are skipped for every column).
pub trait IntoRow {
    fn into_values(self, bv: &BaseValues) -> Vec<Value>;
    /// Sort tags for each column, in order. Length must equal the
    /// `into_values` result length.
    fn column_sorts(&self) -> Vec<ColumnSort>;
}

/// A single column of an egglog row, on the input side.
///
/// This is a sealed trait — additional impls live in the egglog crate.
pub trait IntoColumn: sealed::Sealed {
    fn into_value(self, bv: &BaseValues) -> Value;
    /// Runtime sort tag for this column. Consulted by the `Read` /
    /// `Write` methods to validate against the table's declared schema.
    fn column_sort(&self) -> ColumnSort;
}

// ---------------------------------------------------------------------
// Output side: FromRow + FromColumn
// ---------------------------------------------------------------------

/// Convert a row of egglog [`Value`]s back into a Rust value.
///
/// Implemented for:
/// - `()` — discards the row, useful for "did this match" queries.
/// - A bare [`FromColumn`] type — extracts a single column.
/// - Tuples up to arity 8 of [`FromColumn`] types.
/// - `Vec<Value>` as an escape hatch for rows with non-base columns.
pub trait FromRow: Sized {
    fn from_values(values: &[Value], bv: &BaseValues) -> Self;
}

/// A single column of an egglog row, on the output side.
///
/// This is a sealed trait — additional impls live in the egglog crate.
pub trait FromColumn: sealed::Sealed {
    fn from_value(value: Value, bv: &BaseValues) -> Self;
}

// ---------------------------------------------------------------------
// Escape hatch: RawValues
// ---------------------------------------------------------------------

/// Escape hatch wrapper — pass already-converted [`Value`] columns when
/// the caller wants to bypass per-column sort checking entirely.
#[derive(Clone, Debug)]
pub struct RawValues(pub Vec<Value>);

impl IntoRow for RawValues {
    fn into_values(self, _bv: &BaseValues) -> Vec<Value> {
        self.0
    }
    fn column_sorts(&self) -> Vec<ColumnSort> {
        vec![ColumnSort::Unchecked; self.0.len()]
    }
}

impl FromRow for Vec<Value> {
    fn from_values(values: &[Value], _bv: &BaseValues) -> Self {
        values.to_vec()
    }
}

impl FromRow for () {
    fn from_values(_values: &[Value], _bv: &BaseValues) -> Self {}
}

// ---------------------------------------------------------------------
// Base column impls — symmetric for IntoColumn / FromColumn
// ---------------------------------------------------------------------

macro_rules! impl_column_for_base {
    ( $( ($ty:ty, $sort_name:literal) ),+ $(,)? ) => {
        $(
            impl sealed::Sealed for $ty {}
            impl IntoColumn for $ty {
                fn into_value(self, bv: &BaseValues) -> Value {
                    bv.get::<$ty>(self)
                }
                fn column_sort(&self) -> ColumnSort {
                    ColumnSort::Named(Arc::from($sort_name))
                }
            }
            impl FromColumn for $ty {
                fn from_value(value: Value, bv: &BaseValues) -> Self {
                    bv.unwrap::<$ty>(value)
                }
            }
        )+
    };
}

impl_column_for_base!(
    (i64, "i64"),
    (bool, "bool"),
    ((), "Unit"),
    (sort::F, "f64"),
    (sort::S, "String"),
    (sort::Z, "BigInt"),
    (sort::Q, "BigRat"),
);

// `String` is sugar — egglog's String sort uses `sort::S` (`Boxed<String>`).
impl sealed::Sealed for String {}
impl IntoColumn for String {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(Arc::from("String"))
    }
}
impl FromColumn for String {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::S>(value).0
    }
}

// `&str` is one-directional input sugar.
impl sealed::Sealed for &str {}
impl IntoColumn for &str {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.to_string().into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(Arc::from("String"))
    }
}

// `f64` is sugar for `sort::F`.
impl sealed::Sealed for f64 {}
impl IntoColumn for f64 {
    fn into_value(self, bv: &BaseValues) -> Value {
        use ordered_float::OrderedFloat;
        bv.get::<sort::F>(OrderedFloat(self).into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(Arc::from("f64"))
    }
}
impl FromColumn for f64 {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::F>(value).0.0
    }
}

// Bare `Value` is the unchecked escape hatch.
impl sealed::Sealed for Value {}
impl IntoColumn for Value {
    fn into_value(self, _bv: &BaseValues) -> Value {
        self
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Unchecked
    }
}
impl FromColumn for Value {
    fn from_value(value: Value, _bv: &BaseValues) -> Self {
        value
    }
}

// ---------------------------------------------------------------------
// Single-column blanket impls
// ---------------------------------------------------------------------

impl<A: IntoColumn> IntoRow for A {
    fn into_values(self, bv: &BaseValues) -> Vec<Value> {
        vec![self.into_value(bv)]
    }
    fn column_sorts(&self) -> Vec<ColumnSort> {
        vec![self.column_sort()]
    }
}

// Note: no blanket `impl<A: FromColumn> FromRow for A` — would conflict
// with the `Vec<Value>` impl. Single-column outputs use the (A,) tuple form.

// ---------------------------------------------------------------------
// Tuple impls — symmetric for IntoRow / FromRow
// ---------------------------------------------------------------------

macro_rules! impl_row_for_tuple {
    ( $( ($($name:ident),+) ),+ $(,)? ) => {
        $(
            #[allow(non_snake_case)]
            impl<$($name: IntoColumn),+> IntoRow for ($($name,)+) {
                fn into_values(self, bv: &BaseValues) -> Vec<Value> {
                    let ($($name,)+) = self;
                    vec![ $( $name.into_value(bv) ),+ ]
                }
                fn column_sorts(&self) -> Vec<ColumnSort> {
                    let ($($name,)+) = self;
                    vec![ $( $name.column_sort() ),+ ]
                }
            }

            #[allow(non_snake_case)]
            impl<$($name: FromColumn),+> FromRow for ($($name,)+) {
                fn from_values(values: &[Value], bv: &BaseValues) -> Self {
                    let arity = [$(stringify!($name)),+].len();
                    assert!(
                        values.len() == arity,
                        "FromRow: expected {} values for tuple of arity {}, got {} (use Vec<Value> or () to discard extras)",
                        arity, arity, values.len(),
                    );
                    let mut iter = values.iter().copied();
                    ( $( $name::from_value(iter.next().unwrap(), bv), )+ )
                }
            }
        )+
    };
}

impl_row_for_tuple! {
    (T1),
    (T1, T2),
    (T1, T2, T3),
    (T1, T2, T3, T4),
    (T1, T2, T3, T4, T5),
    (T1, T2, T3, T4, T5, T6),
    (T1, T2, T3, T4, T5, T6, T7),
    (T1, T2, T3, T4, T5, T6, T7, T8),
}
