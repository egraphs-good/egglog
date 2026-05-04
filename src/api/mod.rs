//! Typed row encoding for the [`crate::EGraph`] surface API.
//!
//! [`IntoRow`] / [`IntoColumn`] convert Rust values into egglog row data on
//! the way *in* (insert, lookup keys, query patterns).  [`FromRow`] /
//! [`FromColumn`] convert row data back into Rust values on the way *out*
//! (lookup return values, query iteration).
//!
//! Both surfaces share the same column impls — the standard base types
//! (`i64`, `bool`, `()`, `String`, `&str`, `f64`, `sort::F` / `S` / `Z` /
//! `Q`) plus [`Value`] itself as an "already-converted" pass-through.
//!
//! For rows whose columns are not base values (eclasses, container values),
//! use [`RawValues`] as an escape hatch.

use crate::core_relations::{BaseValues, Value};
use crate::sort;

mod eclass;
pub use eclass::{EClass, EqSortMarker};

mod sealed {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------
// Input side: IntoRow + IntoColumn
// ---------------------------------------------------------------------

/// Convert a Rust value into a row of egglog [`Value`]s.
///
/// Implemented for:
/// - A bare [`IntoColumn`] value (e.g. `1_i64` or a [`Value`] you already
///   computed) — produces a single-column row.
/// - Tuples up to arity 8 of [`IntoColumn`] values.
/// - [`RawValues`] as an escape hatch for already-converted multi-column
///   rows.
pub trait IntoRow {
    fn into_values(self, bv: &BaseValues) -> Vec<Value>;
}

/// A single column of an egglog row, on the input side.
///
/// This is a sealed trait — additional impls live in the egglog crate.
pub trait IntoColumn: sealed::Sealed {
    fn into_value(self, bv: &BaseValues) -> Value;
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

/// Escape hatch wrapper — pass already-converted [`Value`] columns when the
/// row contains non-base sorts that can't go through [`IntoColumn`] /
/// [`FromColumn`].
#[derive(Clone, Debug)]
pub struct RawValues(pub Vec<Value>);

impl IntoRow for RawValues {
    fn into_values(self, _bv: &BaseValues) -> Vec<Value> {
        self.0
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
    ( $( $ty:ty ),+ $(,)? ) => {
        $(
            impl sealed::Sealed for $ty {}
            impl IntoColumn for $ty {
                fn into_value(self, bv: &BaseValues) -> Value {
                    bv.get::<$ty>(self)
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

impl_column_for_base!(i64, bool, (), sort::F, sort::S, sort::Z, sort::Q);

// `String` is sugar — egglog's String sort uses `sort::S` (`Boxed<String>`).
impl sealed::Sealed for String {}
impl IntoColumn for String {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.into())
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
}

// `f64` is sugar for `sort::F`.
impl sealed::Sealed for f64 {}
impl IntoColumn for f64 {
    fn into_value(self, bv: &BaseValues) -> Value {
        use ordered_float::OrderedFloat;
        bv.get::<sort::F>(OrderedFloat(self).into())
    }
}
impl FromColumn for f64 {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::F>(value).0.0
    }
}

// `Value` passes through.
impl sealed::Sealed for Value {}
impl IntoColumn for Value {
    fn into_value(self, _bv: &BaseValues) -> Value {
        self
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
            }

            #[allow(non_snake_case)]
            impl<$($name: FromColumn),+> FromRow for ($($name,)+) {
                fn from_values(values: &[Value], bv: &BaseValues) -> Self {
                    let mut iter = values.iter().copied();
                    ( $( $name::from_value(iter.next().expect("FromRow: too few values"), bv), )+ )
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
