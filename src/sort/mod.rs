use num::traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};
use num::{BigInt, BigRational};
pub use ordered_float::OrderedFloat;
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::{Shl, Shr};
use std::{any::Any, sync::Arc};

use crate::core_relations;
pub use core_relations::{
    BaseValues, Boxed, ContainerValue, ContainerValues, ExecutionState, Rebuilder,
};
pub use egglog_bridge::ColumnTy;

use crate::*;

pub type Z = core_relations::Boxed<BigInt>;
pub type Q = core_relations::Boxed<BigRational>;
pub type F = core_relations::Boxed<OrderedFloat<f64>>;
pub type S = core_relations::Boxed<String>;

mod bigint;
pub use bigint::*;
mod bigrat;
pub use bigrat::*;
mod bool;
pub use self::bool::*;
mod string;
pub use string::*;
mod unit;
pub use unit::*;
mod i64;
pub use self::i64::*;
mod f64;
pub use self::f64::*;
mod map;
pub use map::*;
mod set;
pub use set::*;
mod vec;
pub use vec::*;
mod r#fn;
pub use r#fn::*;
mod multiset;
pub use multiset::*;

/// A sort (type) in the e-graph that defines values in egglog. Sorts are user-extensible (e.g., [`prelude::BaseSort`]).
#[typetag::serialize]
pub trait Sort: Any + Send + Sync + Debug {
    /// Returns the name of this sort.
    fn name(&self) -> &str;

    /// Returns the backend-specific column type. See [`ColumnTy`].
    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy;

    /// return the inner sorts if a container sort
    /// remember that containers can contain containers
    /// and this only unfolds one level
    fn inner_sorts(&self) -> Vec<ArcSort> {
        if self.is_container_sort() {
            todo!("inner_sorts: {}", self.name());
        } else {
            panic!("inner_sort called on non-container sort: {}", self.name());
        }
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph);

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static>;

    fn is_eq_sort(&self) -> bool {
        false
    }

    // return true if it is a container sort.
    fn is_container_sort(&self) -> bool {
        false
    }

    // return true if it is a container sort that contains ids.
    // only eq_sort and eq_container_sort need to be canonicalized.
    fn is_eq_container_sort(&self) -> bool {
        false
    }

    /// Return the serialized name of the sort
    ///
    /// Only used for container sorts, which cannot be serialized with make_expr so need an explicit name
    fn serialized_name(&self, _container_values: &ContainerValues, _value: Value) -> String {
        self.name().to_owned()
    }

    /// Return the inner values and sorts.
    /// Only container sort need to implement this method,
    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        debug_assert!(!self.is_container_sort());
        let _ = value;
        let _ = container_values;
        vec![]
    }

    /// Return the type id of values that this sort represents.
    ///
    /// Every non-EqSort sort should return Some(TypeId).
    fn value_type(&self) -> Option<TypeId>;

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        let _ = eg;
    }

    /// Reconstruct a container value in a TermDag
    fn reconstruct_termdag_container(
        &self,
        container_values: &ContainerValues,
        value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        let _container_values = container_values;
        let _value = value;
        let _termdag = termdag;
        let _element_terms = element_terms;
        todo!("reconstruct_termdag_container: {}", self.name());
    }

    /// Reconstruct a leaf primitive value in a TermDag
    fn reconstruct_termdag_base(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let _base_values = base_values;
        let _value = value;
        let _termdag = termdag;
        todo!("reconstruct_termdag_leaf: {}", self.name());
    }
}

// Note: this trait is currently intended to be implemented on the
// same struct as `Sort`. If in the future we have dynamic presorts
// (for example, we want to add partial application) we should revisit
// this and make the methods take a `self` parameter.
pub trait Presort {
    fn presort_name() -> &'static str;
    fn reserved_primitives() -> Vec<&'static str>;
    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError>;
}

pub type MkSort = fn(&mut TypeInfo, String, &[Expr]) -> Result<ArcSort, TypeError>;

#[derive(Debug, Serialize)]
pub struct EqSort {
    pub name: String,
}

#[typetag::serialize]
impl Sort for EqSort {
    fn name(&self) -> &str {
        &self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, _backend: &mut egglog_bridge::EGraph) {}

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn is_eq_sort(&self) -> bool {
        true
    }

    fn value_type(&self) -> Option<TypeId> {
        None
    }
}

pub fn literal_sort(lit: &Literal) -> ArcSort {
    match lit {
        Literal::Int(_) => I64Sort.to_arcsort(),
        Literal::Float(_) => F64Sort.to_arcsort(),
        Literal::String(_) => StringSort.to_arcsort(),
        Literal::Bool(_) => BoolSort.to_arcsort(),
        Literal::Unit => UnitSort.to_arcsort(),
    }
}
