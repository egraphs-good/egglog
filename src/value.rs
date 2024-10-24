use ordered_float::OrderedFloat;
use std::num::NonZeroU32;

use lazy_static::lazy_static;

use crate::ast::Symbol;

#[cfg(debug_assertions)]
use crate::{BoolSort, F64Sort, I64Sort, Sort, StringSort};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
// FIXME this shouldn't be pub
pub struct Value {
    // since egglog is type-safe, we don't need to store the tag
    // however, it is useful in debugging, so we keep it in debug builds
    #[cfg(debug_assertions)]
    pub tag: Symbol,
    pub bits: u64,
}

lazy_static! {
    static ref BOGUS: Symbol = "__bogus__".into();
    static ref UNIT: Symbol = "Unit".into();
}

impl Value {
    pub fn unit() -> Self {
        Value {
            #[cfg(debug_assertions)]
            tag: *UNIT,
            bits: 0,
        }
    }

    pub fn fake() -> Self {
        Value {
            #[cfg(debug_assertions)]
            tag: *BOGUS,
            bits: 1234567890,
        }
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Self {
            #[cfg(debug_assertions)]
            tag: I64Sort.name(),
            bits: i as u64,
        }
    }
}

impl From<OrderedFloat<f64>> for Value {
    fn from(f: OrderedFloat<f64>) -> Self {
        Self {
            #[cfg(debug_assertions)]
            tag: F64Sort.name(),
            bits: f.into_inner().to_bits(),
        }
    }
}

impl From<Symbol> for Value {
    fn from(s: Symbol) -> Self {
        Self {
            #[cfg(debug_assertions)]
            tag: StringSort.name(),
            bits: NonZeroU32::from(s).get().into(),
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self {
            #[cfg(debug_assertions)]
            tag: BoolSort.name(),
            bits: b as u64,
        }
    }
}
