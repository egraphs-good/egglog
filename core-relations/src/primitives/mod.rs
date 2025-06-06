//! Mechanisms for declaring primitive types and functions on them.

use std::{
    any::{Any, TypeId},
    fmt::{self, Debug},
    hash::Hash,
};

use numeric_id::{define_id, DenseIdMap, NumericId};

use crate::common::{HashMap, InternTable, Value};

#[cfg(test)]
mod tests;
mod unboxed;

define_id!(pub PrimitiveId, u32, "an identifier for primitive types");

/// A simple primitive type that can be interned in a database.
///
/// Most callers can simply implement this trait on their desired type, with no overrides needed.
/// For types that are particularly small, users can override the `try_box` and `try_unbox`
/// methods and set `MAY_UNBOX` to `true` to allow the primitive to be stored directly in a
/// `Value`.
///
/// Regardless, all primitive types should be registered in a [`Primitives`] instance using the
/// [`Primitives::register_type`] method before they can be used in the database.
pub trait Primitive: Clone + Hash + Eq + Any + Debug + Send + Sync {
    const MAY_UNBOX: bool = false;
    fn intern(&self, table: &InternTable<Self, Value>) -> Value {
        table.intern(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn try_box(&self) -> Option<Value> {
        None
    }
    fn try_unbox(_val: Value) -> Option<Self> {
        None
    }
}

impl Primitive for String {}
impl Primitive for &'static str {}
impl Primitive for num::Rational64 {}

/// A wrapper used to print a primitive value.
///
/// The given primitive must be registered with the `Primitives` instance,
/// otherwise attempting to call the [`Debug`] implementation will panic.
pub struct PrimitivePrinter<'a> {
    pub prim: &'a Primitives,
    pub ty: PrimitiveId,
    pub val: Value,
}

impl Debug for PrimitivePrinter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.prim.tables[self.ty].print_value(self.val, f)
    }
}

/// A registry for primitive values and functions on them.
#[derive(Clone, Default)]
pub struct Primitives {
    type_ids: HashMap<TypeId, PrimitiveId>,
    tables: DenseIdMap<PrimitiveId, Box<dyn DynamicInternTable>>,
}

impl Primitives {
    /// Register the given type `P` as a primitive type in this registry.
    pub fn register_type<P: Primitive>(&mut self) -> PrimitiveId {
        let type_id = TypeId::of::<P>();
        let next_primitive_id = PrimitiveId::from_usize(self.type_ids.len());
        let id = *self.type_ids.entry(type_id).or_insert(next_primitive_id);
        self.tables
            .get_or_insert(id, || Box::<PrimitiveInternTable<P>>::default());
        id
    }

    /// Get the [`PrimitiveId`] for the given primitive type `P`.
    pub fn get_ty<P: Primitive>(&self) -> PrimitiveId {
        self.type_ids[&TypeId::of::<P>()]
    }

    /// Get the [`PrimitiveId`] for the given primitive type id.
    pub fn get_ty_by_id(&self, id: TypeId) -> PrimitiveId {
        self.type_ids[&id]
    }

    /// Get a [`Value`] representing the given primitive `p`.
    pub fn get<P: Primitive>(&self, p: P) -> Value {
        if P::MAY_UNBOX {
            if let Some(v) = p.try_box() {
                return v;
            }
        }
        let id = self.get_ty::<P>();
        let table = self.tables[id]
            .as_any()
            .downcast_ref::<PrimitiveInternTable<P>>()
            .unwrap();
        table.intern(p)
    }

    pub fn unwrap<P: Primitive>(&self, v: Value) -> P {
        if P::MAY_UNBOX {
            if let Some(p) = P::try_unbox(v) {
                return p;
            }
        }
        let id = self.get_ty::<P>();
        let table = self
            .tables
            .get(id)
            .expect("types must be registered before unwrapping")
            .as_any()
            .downcast_ref::<PrimitiveInternTable<P>>()
            .unwrap();
        table.get(v)
    }
}

trait DynamicInternTable: Any + dyn_clone::DynClone + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn print_value(&self, val: Value, f: &mut fmt::Formatter) -> fmt::Result;
}

// Implements `Clone` for `Box<dyn DynamicInternTable>`.
dyn_clone::clone_trait_object!(DynamicInternTable);

#[derive(Clone)]
struct PrimitiveInternTable<P> {
    table: InternTable<P, Value>,
}

impl<P> Default for PrimitiveInternTable<P> {
    fn default() -> Self {
        Self {
            table: InternTable::default(),
        }
    }
}

impl<P: Primitive> DynamicInternTable for PrimitiveInternTable<P> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn print_value(&self, val: Value, f: &mut fmt::Formatter) -> fmt::Result {
        let p = self.get(val);
        write!(f, "{p:?}")
    }
}

const VAL_OFFSET: u32 = 1 << (std::mem::size_of::<Value>() as u32 * 8 - 1);

impl<P: Primitive> PrimitiveInternTable<P> {
    pub fn intern(&self, p: P) -> Value {
        if P::MAY_UNBOX {
            p.try_box().unwrap_or_else(|| {
                // If the primitive type is too large to fit in a Value, we intern it and return
                // the corresponding Value with its top bit set. We use add to ensure we overflow
                // if the number of interned values is too large.
                Value::new(
                    self.table
                        .intern(&p)
                        .rep()
                        .checked_add(VAL_OFFSET)
                        .expect("interned value overflowed"),
                )
            })
        } else {
            self.table.intern(&p)
        }
    }

    pub fn get(&self, v: Value) -> P {
        if P::MAY_UNBOX {
            P::try_unbox(v)
                .unwrap_or_else(|| self.table.get_cloned(Value::new(v.rep() - VAL_OFFSET)))
        } else {
            self.table.get_cloned(v)
        }
    }
}

/// A newtype wrapper used to implement the [`Primitive`] trait on types not
/// defined in this crate.
///
/// This type is just a helper: users can also implement the [`Primitive`] trait directly on their
/// types if the type is defined in the crate in which the implementation is defined, or if they
/// need custom logic for boxing or unboxing the type.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Boxed<T>(pub T);

impl<T> Boxed<T> {
    pub fn new(value: T) -> Self {
        Boxed(value)
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Hash + Eq + Debug + Clone + Send + Sync + 'static> Primitive for Boxed<T> {}

impl<T> std::ops::Deref for Boxed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Boxed<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<T> for Boxed<T> {
    fn from(value: T) -> Self {
        Boxed(value)
    }
}

impl<T: Copy> From<&T> for Boxed<T> {
    fn from(value: &T) -> Self {
        Boxed(*value)
    }
}

impl<T: std::ops::Add<Output = T>> std::ops::Add for Boxed<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Boxed(self.0 + other.0)
    }
}

impl<T: std::ops::Sub<Output = T>> std::ops::Sub for Boxed<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Boxed(self.0 - other.0)
    }
}

impl<T: std::ops::Mul<Output = T>> std::ops::Mul for Boxed<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Boxed(self.0 * other.0)
    }
}

impl<T: std::ops::Div<Output = T>> std::ops::Div for Boxed<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Boxed(self.0 / other.0)
    }
}

impl<T: std::ops::Rem<Output = T>> std::ops::Rem for Boxed<T> {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        Boxed(self.0 % other.0)
    }
}

impl<T: std::ops::Neg<Output = T>> std::ops::Neg for Boxed<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Boxed(-self.0)
    }
}

impl<T: std::ops::BitAnd<Output = T>> std::ops::BitAnd for Boxed<T> {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Boxed(self.0 & other.0)
    }
}

impl<T: std::ops::BitOr<Output = T>> std::ops::BitOr for Boxed<T> {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Boxed(self.0 | other.0)
    }
}

impl<T: std::ops::BitXor<Output = T>> std::ops::BitXor for Boxed<T> {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Boxed(self.0 ^ other.0)
    }
}
