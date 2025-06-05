use lazy_static::lazy_static;
use num::traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};
use num::{rational::BigRational, BigInt};
use ordered_float::OrderedFloat;
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::{Shl, Shr};
use std::sync::Mutex;
use std::{any::Any, sync::Arc};

pub use core_relations::{Container, Containers, ExecutionState, Primitives, Rebuilder};
pub use egglog_bridge::ColumnTy;

use crate::ast::Literal;
use crate::extract::Cost;
use crate::util::IndexSet;
use crate::*;

/// A newtype wrapper used to implement the [`core_relations::Primitive`] trait on types not
/// defined in this crate.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseTypeWrapper<T>(pub T);

impl<T> BaseTypeWrapper<T> {
    pub fn new(value: T) -> Self {
        BaseTypeWrapper(value)
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::ops::Deref for BaseTypeWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for BaseTypeWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<T> for BaseTypeWrapper<T> {
    fn from(value: T) -> Self {
        BaseTypeWrapper(value)
    }
}

impl<T: Copy> From<&T> for BaseTypeWrapper<T> {
    fn from(value: &T) -> Self {
        BaseTypeWrapper(*value)
    }
}

impl<T: std::ops::Add<Output = T>> std::ops::Add for BaseTypeWrapper<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 + other.0)
    }
}

impl<T: std::ops::Sub<Output = T>> std::ops::Sub for BaseTypeWrapper<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 - other.0)
    }
}

impl<T: std::ops::Mul<Output = T>> std::ops::Mul for BaseTypeWrapper<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 * other.0)
    }
}

impl<T: std::ops::Div<Output = T>> std::ops::Div for BaseTypeWrapper<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 / other.0)
    }
}

impl<T: std::ops::Rem<Output = T>> std::ops::Rem for BaseTypeWrapper<T> {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 % other.0)
    }
}

impl<T: std::ops::Neg<Output = T>> std::ops::Neg for BaseTypeWrapper<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        BaseTypeWrapper(-self.0)
    }
}

impl<T: std::ops::BitAnd<Output = T>> std::ops::BitAnd for BaseTypeWrapper<T> {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 & other.0)
    }
}

impl<T: std::ops::BitOr<Output = T>> std::ops::BitOr for BaseTypeWrapper<T> {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 | other.0)
    }
}

impl<T: std::ops::BitXor<Output = T>> std::ops::BitXor for BaseTypeWrapper<T> {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        BaseTypeWrapper(self.0 ^ other.0)
    }
}

impl core_relations::Primitive for BaseTypeWrapper<BigInt> {}
impl core_relations::Primitive for BaseTypeWrapper<BigRational> {}
impl core_relations::Primitive for BaseTypeWrapper<OrderedFloat<f64>> {}

pub type Z = BaseTypeWrapper<BigInt>;
pub type Q = BaseTypeWrapper<BigRational>;
pub type F = BaseTypeWrapper<OrderedFloat<f64>>;
pub type S = BaseTypeWrapper<Symbol>;

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

pub trait Sort: Any + Send + Sync + Debug {
    fn name(&self) -> Symbol;

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
    fn serialized_name(&self, _value: Value) -> Symbol {
        self.name()
    }

    /// Return the inner values and sorts.
    /// Only container sort need to implement this method,
    fn inner_values(&self, containers: &Containers, value: Value) -> Vec<(ArcSort, Value)> {
        debug_assert!(!self.is_container_sort());
        let _ = value;
        let _ = containers;
        vec![]
    }

    /// Return the type id of values that this sort represents.
    ///
    /// Every non-EqSort sort should return Some(TypeId).
    fn value_type(&self) -> Option<TypeId>;

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        let _ = eg;
    }

    /// Default cost for containers when the cost model does not specify the cost
    fn default_container_cost(
        &self,
        _containers: &Containers,
        _value: Value,
        element_costs: &[Cost],
    ) -> Cost {
        element_costs.iter().fold(0, |s, c| s.saturating_add(*c))
    }

    /// Default cost for leaf primitives when the cost model does not specify the cost
    fn default_leaf_cost(&self, _primitives: &Primitives, _value: Value) -> Cost {
        1
    }

    /// Reconstruct a container value in a TermDag
    fn reconstruct_termdag_container(
        &self,
        containers: &Containers,
        value: Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        let _containers = containers;
        let _value = value;
        let _termdag = termdag;
        let _element_terms = element_terms;
        todo!("reconstruct_termdag_container: {}", self.name());
    }

    /// Reconstruct a leaf primitive value in a TermDag
    fn reconstruct_termdag_leaf(
        &self,
        primitives: &Primitives,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let _primitives = primitives;
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
    fn presort_name() -> Symbol;
    fn reserved_primitives() -> Vec<Symbol>;
    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: Symbol,
        args: &[Expr],
    ) -> Result<ArcSort, TypeError>;
}

#[derive(Debug)]
pub struct EqSort {
    pub name: Symbol,
}

impl Sort for EqSort {
    fn name(&self) -> Symbol {
        self.name
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

/// This trait is used by the `add_primitive` macro to infer the sort type
/// that corresponds to a given type.
pub trait IntoSort: Sized {
    type Sort: Sort;
}

pub type PreSort =
    fn(typeinfo: &mut TypeInfo, name: Symbol, params: &[Expr]) -> Result<ArcSort, TypeError>;

pub fn literal_sort(lit: &Literal) -> ArcSort {
    match lit {
        Literal::Int(_) => Arc::new(I64Sort) as ArcSort,
        Literal::Float(_) => Arc::new(F64Sort) as ArcSort,
        Literal::String(_) => Arc::new(StringSort) as ArcSort,
        Literal::Bool(_) => Arc::new(BoolSort) as ArcSort,
        Literal::Unit => Arc::new(UnitSort) as ArcSort,
    }
}
