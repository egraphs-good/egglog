use ordered_float::OrderedFloat;

use lazy_static::lazy_static;

#[cfg(debug_assertions)]
use crate::{BoolSort, F64Sort, I64Sort, Sort};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
// FIXME this shouldn't be pub
pub struct Value {
    // since egglog is type-safe, we don't need to store the tag
    // however, it is useful in debugging, so we keep it in debug builds
    #[cfg(debug_assertions)]
    pub tag: String,
    pub bits: u64,
}

lazy_static! {
    static ref BOGUS: String = "__bogus__".into();
    static ref UNIT: String = "Unit".into();
}

impl Value {
    pub fn unit() -> Self {
        Value {
            #[cfg(debug_assertions)]
            tag: UNIT.clone(),
            bits: 0,
        }
    }

    pub fn fake() -> Self {
        Value {
            #[cfg(debug_assertions)]
            tag: BOGUS.clone(),
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

// impl From<String> for Value {
//     fn from(s: String) -> Self {
//         Self {
//             #[cfg(debug_assertions)]
//             tag: StringSort.name(),
//             bits: NonZeroU32::from(s).get().into(),
//         }
//     }
// }

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self {
            #[cfg(debug_assertions)]
            tag: BoolSort.name(),
            bits: b as u64,
        }
    }
}
