use num::traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};
use num::{BigInt, BigRational};
use ordered_float::OrderedFloat;
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::{Shl, Shr};
use std::{any::Any, sync::Arc};

pub use core_relations::{Container, Containers, ExecutionState, Primitives, Rebuilder};
pub use egglog_bridge::ColumnTy;

use crate::ast::Literal;
use crate::extract::Cost;
use crate::*;

pub type Z = core_relations::Boxed<BigInt>;
pub type Q = core_relations::Boxed<BigRational>;
pub type F = core_relations::Boxed<OrderedFloat<f64>>;
pub type S = SymbolWrapper;

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
        egraph: &mut EGraph,
        name: Symbol,
        args: &[Expr],
        span: Span,
    ) -> Result<(), TypeError>;
}

pub type MkSort = fn(&mut EGraph, Symbol, &[Expr], Span) -> Result<(), TypeError>;

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

pub fn literal_sort(lit: &Literal) -> ArcSort {
    match lit {
        Literal::Int(_) => I64Sort.to_arcsort(),
        Literal::Float(_) => F64Sort.to_arcsort(),
        Literal::String(_) => StringSort.to_arcsort(),
        Literal::Bool(_) => BoolSort.to_arcsort(),
        Literal::Unit => UnitSort.to_arcsort(),
    }
}
