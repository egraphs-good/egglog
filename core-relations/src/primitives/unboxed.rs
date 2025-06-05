//! Special handling for small integer types to avoid lookups in an [`InternTable`].
//!
//!
//! [`InternTable`]: crate::common::InternTable

use numeric_id::NumericId;

use crate::Value;

use super::Primitive;

macro_rules! impl_small_primitive {
    ($ty:ty) => {
        impl Primitive for $ty {
            const MAY_UNBOX: bool = true;
            fn try_unbox(val: Value) -> Option<Self> {
                Some(val.rep() as $ty)
            }
            fn try_box(&self) -> Option<Value> {
                Some(Value::new(*self as u32))
            }
        }
    };
    ($ty:ty, $($rest:ty),+) => {
        impl_small_primitive!($ty);
        impl_small_primitive!($($rest),+);
    };
}

impl_small_primitive!(u8, u16, u32, i8, i16, i32);

impl Primitive for bool {
    const MAY_UNBOX: bool = true;
    fn try_unbox(val: Value) -> Option<Self> {
        Some(val.rep() != 0)
    }
    fn try_box(&self) -> Option<Value> {
        Some(Value::new(if *self { 1 } else { 0 }))
    }
}

impl Primitive for () {
    const MAY_UNBOX: bool = true;
    fn try_unbox(_val: Value) -> Option<Self> {
        Some(())
    }
    fn try_box(&self) -> Option<Value> {
        Some(Value::new(0))
    }
}

const VAL_BITS: u32 = std::mem::size_of::<Value>() as u32 * 8;
const VAL_MASK: u32 = 1 << (VAL_BITS - 1);

macro_rules! impl_medium_primitive {
    ($ty:ty) => {
        impl Primitive for $ty {
            const MAY_UNBOX: bool = true;
            fn try_box(&self) -> Option<Value> {
                if *self & (VAL_MASK-1) as $ty == *self {
                    // If the top bit is clear, we can box it directly.
                    Some(Value::new(*self as u32))
                } else {
                    // If the top bit is set, we need to intern it.
                    None
                }
            }
            fn try_unbox(val: Value) -> Option<Self> {
                let top_bit_clear = val.rep() & VAL_MASK == 0;
                if top_bit_clear {
                    Some(val.rep() as $ty)
                } else {
                    // If the top bit is set, look this up in an intern table.
                    None
                }
            }
        }
    };

    ($ty:ty, $($rest:ty),+) => {
        impl_medium_primitive!($ty);
        impl_medium_primitive!($($rest),+);
    };
}

impl_medium_primitive!(u64, i64, usize, isize);
