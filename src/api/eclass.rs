//! Typed eclass handles.
//!
//! Today, e-class IDs of user datatypes flow through the API as opaque
//! [`Value`]s — there's no compile-time check that an e-class for sort
//! `Math` isn't accidentally passed where a `List` was expected.
//!
//! [`EClass<M>`] is a phantom-typed wrapper around [`Value`], tagged at
//! the type level by an [`EqSortMarker`] — a zero-sized marker the user
//! defines once per sort. Wrapping a value claims the value belongs to
//! sort `M::NAME`. [`EGraph::typed_eclass`] performs the check.
//!
//! ```
//! use egglog::api::{EClass, EqSortMarker};
//! use egglog::prelude::*;
//!
//! struct Math;
//! impl EqSortMarker for Math {
//!     const NAME: &'static str = "Math";
//! }
//!
//! let mut eg = EGraph::default();
//! eg.parse_and_run_program(None, "(datatype Math (Num i64))").unwrap();
//! eg.parse_and_run_program(None, "(let $n (Num 7))").unwrap();
//!
//! let raw = eg.lookup_function("Num", &[eg.intern::<i64>(7)]).unwrap();
//! let typed: EClass<Math> = eg.typed_eclass::<Math>(raw).unwrap();
//! assert_eq!(typed.value(), raw);
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::core_relations::{BaseValues, Value};

use super::{FromColumn, IntoColumn, sealed};

/// Compile-time marker for a user-defined egglog sort.
///
/// Implement on a zero-sized type to get typed eclass handles. The
/// associated `NAME` constant is the egglog sort name.
///
/// ```
/// use egglog::api::EqSortMarker;
///
/// struct Math;
/// impl EqSortMarker for Math {
///     const NAME: &'static str = "Math";
/// }
/// ```
pub trait EqSortMarker: 'static {
    const NAME: &'static str;
}

/// A typed e-class id. Wraps a [`Value`] tagged at the type level
/// with the [`EqSortMarker`] for the sort it belongs to.
///
/// `EClass<M>` is `Copy`, `Clone`, `Debug`, `Eq`, `Hash` for any `M`.
/// Two `EClass<M>` values compare equal iff their underlying [`Value`]s
/// are equal — i.e. the e-graph hasn't unioned them and gone through
/// rebuild yet. After a rebuild, equal e-classes may still be stored as
/// distinct `Value`s; users wanting canonical comparison should call
/// `EGraph::canonicalize` (todo) or `union-find::find` themselves.
pub struct EClass<M: EqSortMarker> {
    value: Value,
    _marker: PhantomData<fn() -> M>,
}

impl<M: EqSortMarker> EClass<M> {
    /// Construct from a raw [`Value`]. The caller asserts the value is
    /// actually an e-class id of sort `M::NAME`. Prefer
    /// [`crate::EGraph::typed_eclass`] which checks at runtime.
    pub fn from_value_unchecked(value: Value) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }

    /// The underlying raw [`Value`].
    pub fn value(self) -> Value {
        self.value
    }

    /// The sort name this handle is tagged with (compile-time constant).
    pub fn sort_name() -> &'static str {
        M::NAME
    }
}

impl<M: EqSortMarker> Clone for EClass<M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: EqSortMarker> Copy for EClass<M> {}

impl<M: EqSortMarker> PartialEq for EClass<M> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<M: EqSortMarker> Eq for EClass<M> {}

impl<M: EqSortMarker> Hash for EClass<M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<M: EqSortMarker> fmt::Debug for EClass<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EClass<{}>({:?})", M::NAME, self.value)
    }
}

// ---------------------------------------------------------------------
// IntoColumn / FromColumn — typed eclasses pass through as the raw Value
// ---------------------------------------------------------------------

impl<M: EqSortMarker> sealed::Sealed for EClass<M> {}

impl<M: EqSortMarker> IntoColumn for EClass<M> {
    fn into_value(self, _bv: &BaseValues) -> Value {
        self.value
    }
}

impl<M: EqSortMarker> FromColumn for EClass<M> {
    fn from_value(value: Value, _bv: &BaseValues) -> Self {
        // The caller (a typed query/lookup at this column position) is
        // asserting the column is sort `M::NAME`. There's no runtime
        // sort check at this point — the EGraph::query / lookup methods
        // could add one in the future if it becomes a footgun.
        EClass::from_value_unchecked(value)
    }
}
