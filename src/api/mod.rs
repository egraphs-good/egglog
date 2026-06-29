//! Row encoding for the [`crate::EGraph`] surface API.
//!
//! Four traits map Rust values to and from egglog row data:
//!
//! - [`IntoValue`] / [`FromValue`] — one Rust value ↔ one egglog
//!   [`Value`]. These are **open** traits: users implementing custom
//!   sorts (or just custom Rust wrapper types) can implement them to
//!   make their types flow through the row-encoding APIs.
//! - [`IntoValues`] / [`FromValues`] — a whole row of Rust values
//!   ↔ a `&[Value]`. These are **sealed**: row shape is fixed to
//!   tuples up to arity 8, plus the [`RawValues`] / `Vec<Value>` /
//!   `()` escape hatches.
//!
//! The API is **type-unsafe at the column level**: every column flows
//! as a bare [`Value`]. The Rust trait machinery here only enforces
//! arity (you can't pass a 3-tuple to a 2-column table) and subtype
//! (you can't `set` on a constructor or `add` on a function). Per-
//! column sort matching (e.g., "this column wants an `i64`, you
//! passed a `String`") is the caller's responsibility — see the
//! parked dynamic-typing work for what real safety would cost.

use crate::core_relations::{BaseValues, Value};
use crate::sort;
use thiserror::Error;

// ---------------------------------------------------------------------
// ApiError — runtime check failures from the `Read` / `Write` trait
// methods and from [`crate::EGraph::update`] callers.
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

    #[error("table `{name}` is a {actual}; this method is only valid for {expected} tables")]
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
}

mod sealed {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------
// Per-column conversion: IntoValue + FromValue (open)
// ---------------------------------------------------------------------

/// Convert a Rust value into an egglog [`Value`]. Open trait: impl
/// for your own types if you want them to flow through `set` / `add`
/// / `lookup` keys as column inputs.
pub trait IntoValue {
    fn into_value(self, bv: &BaseValues) -> Value;
}

/// Convert an egglog [`Value`] back into a Rust value. Open trait:
/// impl for your own types if you want them to flow out of
/// `function_entries` / `query` rows.
pub trait FromValue: Sized {
    fn from_value(value: Value, bv: &BaseValues) -> Self;
}

// ---------------------------------------------------------------------
// Row conversion: IntoValues + FromValues (sealed)
// ---------------------------------------------------------------------

/// Convert a Rust value into a row of egglog [`Value`]s.
///
/// Sealed; impls cover bare [`IntoValue`] (single-column row),
/// tuples up to arity 8 of [`IntoValue`] values, and [`RawValues`]
/// for variadic / pre-converted rows.
pub trait IntoValues: sealed::Sealed {
    fn into_values(self, bv: &BaseValues) -> impl Iterator<Item = Value>;
}

/// Convert a row of egglog [`Value`]s back into a Rust value.
///
/// Sealed; impls cover `()` (discard), `Vec<Value>` (raw escape
/// hatch), and tuples up to arity 8 of [`FromValue`] types.
pub trait FromValues: Sized + sealed::Sealed {
    fn from_values(values: &[Value], bv: &BaseValues) -> Self;
}

// ---------------------------------------------------------------------
// Built-in column impls — symmetric for IntoValue / FromValue
// ---------------------------------------------------------------------

macro_rules! impl_value_for_base {
    ( $( $ty:ty ),+ $(,)? ) => {
        $(
            impl IntoValue for $ty {
                fn into_value(self, bv: &BaseValues) -> Value {
                    bv.get::<$ty>(self)
                }
            }
            impl FromValue for $ty {
                fn from_value(value: Value, bv: &BaseValues) -> Self {
                    bv.unwrap::<$ty>(value)
                }
            }
        )+
    };
}

impl_value_for_base!(i64, bool, (), sort::F, sort::S, sort::Z, sort::Q);

// `String` is sugar — egglog's String sort uses `sort::S` (`Boxed<String>`).
impl IntoValue for String {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.into())
    }
}
impl FromValue for String {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::S>(value).0
    }
}

// `&str` is one-directional input sugar.
impl IntoValue for &str {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.to_string().into())
    }
}

// `f64` is sugar for `sort::F`.
impl IntoValue for f64 {
    fn into_value(self, bv: &BaseValues) -> Value {
        use ordered_float::OrderedFloat;
        bv.get::<sort::F>(OrderedFloat(self).into())
    }
}
impl FromValue for f64 {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::F>(value).0.0
    }
}

// Bare `Value` passes through unchanged.
impl IntoValue for Value {
    fn into_value(self, _bv: &BaseValues) -> Value {
        self
    }
}
impl FromValue for Value {
    fn from_value(value: Value, _bv: &BaseValues) -> Self {
        value
    }
}

// ---------------------------------------------------------------------
// Variadic / pre-converted rows: RawValues
// ---------------------------------------------------------------------

/// Wrap a `Vec<Value>` as an [`IntoValues`]. Useful for zero-arg
/// constructors (`RawValues(vec![])`) and for rows whose column
/// count isn't known until runtime.
#[derive(Clone, Debug)]
pub struct RawValues(pub Vec<Value>);

impl sealed::Sealed for RawValues {}
impl IntoValues for RawValues {
    fn into_values(self, _bv: &BaseValues) -> impl Iterator<Item = Value> {
        self.0.into_iter()
    }
}

impl sealed::Sealed for Vec<Value> {}
impl FromValues for Vec<Value> {
    fn from_values(values: &[Value], _bv: &BaseValues) -> Self {
        values.to_vec()
    }
}

// `()` as a FromValues discards whatever it was handed (useful for
// "did this match" queries). `()` as an IntoValues is a single Unit
// column, handled by the blanket `impl<A: IntoValue> IntoValues for A`
// below. For a zero-arg row, use `RawValues(vec![])`.
impl FromValues for () {
    fn from_values(_values: &[Value], _bv: &BaseValues) -> Self {}
}

// ---------------------------------------------------------------------
// Single-column blanket impl: any T: IntoValue is a 1-column IntoValues.
//
// Note: this does NOT cover FromValues — that would conflict with the
// `Vec<Value>` impl above. Single-column outputs use the (A,) tuple form.
// ---------------------------------------------------------------------

impl<A: IntoValue> sealed::Sealed for A {}
impl<A: IntoValue> IntoValues for A {
    fn into_values(self, bv: &BaseValues) -> impl Iterator<Item = Value> {
        std::iter::once(self.into_value(bv))
    }
}

// ---------------------------------------------------------------------
// Tuple impls — IntoValues / FromValues for tuples of arity 1..=8
// ---------------------------------------------------------------------

macro_rules! impl_values_for_tuple {
    ( $( ($($name:ident),+) ),+ $(,)? ) => {
        $(
            #[allow(non_snake_case)]
            impl<$($name),+> sealed::Sealed for ($($name,)+) {}

            #[allow(non_snake_case)]
            impl<$($name: IntoValue),+> IntoValues for ($($name,)+) {
                fn into_values(self, bv: &BaseValues) -> impl Iterator<Item = Value> {
                    let ($($name,)+) = self;
                    [ $( $name.into_value(bv) ),+ ].into_iter()
                }
            }

            #[allow(non_snake_case)]
            impl<$($name: FromValue),+> FromValues for ($($name,)+) {
                fn from_values(values: &[Value], bv: &BaseValues) -> Self {
                    let mut iter = values.iter().copied();
                    let tuple = ( $( $name::from_value(iter.next().expect("FromValues: row shorter than tuple arity"), bv), )+ );
                    assert!(
                        iter.next().is_none(),
                        "FromValues: row longer than tuple arity",
                    );
                    tuple
                }
            }
        )+
    };
}

impl_values_for_tuple! {
    (T1),
    (T1, T2),
    (T1, T2, T3),
    (T1, T2, T3, T4),
    (T1, T2, T3, T4, T5),
    (T1, T2, T3, T4, T5, T6),
    (T1, T2, T3, T4, T5, T6, T7),
    (T1, T2, T3, T4, T5, T6, T7, T8),
}
