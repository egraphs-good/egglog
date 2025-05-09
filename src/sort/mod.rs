use lazy_static::lazy_static;
use num::traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};
use num::{rational::BigRational, BigInt};
use ordered_float::OrderedFloat;
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::{Shl, Shr};
use std::sync::Mutex;
use std::{any::Any, sync::Arc};

pub use core_relations::{Container, ExecutionState, ExternalFunction, Rebuilder};
pub use egglog_bridge::ColumnTy;

use crate::ast::Literal;
use crate::extract::{Cost, Extractor};
use crate::util::IndexSet;
use crate::*;

pub type Z = BigInt;
pub type Q = BigRational;
pub type F = OrderedFloat<f64>;
pub type S = Symbol;

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

    // return the inner sorts if a container sort
    // remember that containers can contain containers
    // and this only unfold one level
    fn inner_sorts(&self) -> Vec<&Arc<dyn Sort>> {
        vec![]
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

    // Only eq_container_sort need to implement this method,
    // which returns a list of ids to be tracked.
    fn foreach_tracked_values<'a>(
        &'a self,
        value: &'a Value,
        mut f: Box<dyn FnMut(ArcSort, Value) + 'a>,
    ) {
        for (sort, value) in self.old_inner_values(value) {
            if sort.is_eq_sort() {
                f(sort, value)
            }
        }
    }

    // Sort-wise canonicalization. Return true if value is modified.
    // Only EqSort or containers of EqSort should override.
    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.name(), value.tag);

        #[cfg(not(debug_assertions))]
        let _ = value;
        let _ = unionfind;
        false
    }

    /// Return the serialized name of the sort
    ///
    /// Only used for container sorts, which cannot be serialized with make_expr so need an explicit name
    fn serialized_name(&self, _value: &core_relations::Value) -> Symbol {
        self.name()
    }

    /// Return the inner values and sorts.
    /// Only container sort need to implement this method,
    fn inner_values(
        &self,
        containers: &core_relations::Containers,
        value: &core_relations::Value,
    ) -> Vec<(ArcSort, core_relations::Value)> {
        debug_assert!(!self.is_container_sort());
        let _ = value;
        let _ = containers;
        vec![]
    }

    /// Return the type id of values that this sort represents.
    ///
    /// Every non-EqSort sort should return Some(TypeId).
    fn value_type(&self) -> Option<TypeId>;

    /// Only eq_container_sort need to implement this method,
    fn old_inner_values(&self, value: &Value) -> Vec<(ArcSort, Value)> {
        let _ = value;
        vec![]
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        let _ = eg;
    }

    /// Extracting a term (with smallest cost) out of a primitive value
    fn extract_term(
        &self,
        egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        _termdag: &mut TermDag,
    ) -> Option<(Cost, Term)>;

    /// Default cost for containers when the cost model does not specify the cost
    fn default_container_cost(
        &self,
        _exec_state: &core_relations::ExecutionState,
        _value: &core_relations::Value,
        element_costs: &Vec<Cost>,
    ) -> Cost {
        element_costs.iter().fold(0, |s, c| { s.saturating_add(*c) })
    }

    /// Default cost for leaf primitives when the cost model does not specify the cost
    fn default_leaf_cost(
        &self,
        _exec_state: &core_relations::ExecutionState,
        _value: &core_relations::Value,
    ) -> Cost {
        1
    }
    
    /// Reconstruct a container value in a TermDag
    fn reconstruct_termdag_container(
        &self,
        exec_state: &core_relations::ExecutionState,
        value: &core_relations::Value,
        termdag: &mut TermDag,
        element_terms: Vec<Term>,
    ) -> Term {
        let _exec_state = exec_state;
        let _value = value;
        let _termdag = termdag;
        let _element_terms = element_terms;
        todo!("reconstruct_termdag_container : {}", self.name());
    }

    /// Reconstruct a leaf primitive value in a TermDag
    fn reconstruct_termdag_leaf(
        &self,
        exec_state: &core_relations::ExecutionState,
        value: &core_relations::Value,
        termdag: &mut TermDag,
    ) -> Term {
        let _exec_state = exec_state;
        let _value = value;
        let _termdag = termdag;
        todo!("reconstruct_termdag_leaf : {}", self.name());
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

    fn canonicalize(&self, value: &mut Value, unionfind: &UnionFind) -> bool {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.name(), value.tag);

        let bits = unionfind.find(value.bits);
        if bits != value.bits {
            value.bits = bits;
            true
        } else {
            false
        }
    }

    fn extract_term(
        &self,
        _egraph: &EGraph,
        _value: Value,
        _extractor: &Extractor,
        _termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        unimplemented!("No extract_term for EqSort {}", self.name)
    }

    fn value_type(&self) -> Option<TypeId> {
        None
    }
}

pub trait FromSort: Sized {
    type Sort: Sort;
    fn load(sort: &Self::Sort, value: &Value) -> Self;
}

pub trait IntoSort: Sized {
    type Sort: Sort;
    fn store(self, sort: &Self::Sort) -> Value;
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
