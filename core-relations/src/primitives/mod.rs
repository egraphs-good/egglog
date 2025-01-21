//! Mechanisms for declaring primitive types and operations on them.

use std::{
    any::{Any, TypeId},
    fmt::{self, Debug},
    hash::Hash,
    ops::Deref,
};

use numeric_id::{define_id, DenseIdMap};

use crate::{
    action::{
        mask::{Mask, MaskIter, ValueSource},
        Bindings,
    },
    common::{InternTable, Value},
    pool::Pool,
    QueryEntry, Variable,
};

#[cfg(test)]
mod tests;

define_id!(pub PrimitiveId, u32, "an identifier for primitive types");
define_id!(pub PrimitiveFunctionId, u32, "an identifier for primitive operations");

pub trait Primitive: Clone + Hash + Eq + Any + Debug + Send + Sync {
    fn intern(&self, table: &InternTable<Self, Value>) -> Value {
        table.intern(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T: Clone + Hash + Eq + Any + Debug + Send + Sync> Primitive for T {}

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

/// A registry for primitive values and operations on them.
#[derive(Default)]
pub struct Primitives {
    type_ids: InternTable<TypeId, PrimitiveId>,
    tables: DenseIdMap<PrimitiveId, Box<dyn DynamicInternTable>>,
    operations: DenseIdMap<PrimitiveFunctionId, DynamicPrimitveOperation>,
}

impl Primitives {
    /// Register the given type `P` as a primitive type in this registry.
    pub fn register_type<P: Primitive>(&mut self) -> PrimitiveId {
        let id = self.get_ty::<P>();
        self.tables
            .get_or_insert(id, || Box::<InternTable<P, Value>>::default());
        id
    }

    /// Get the [`PrimitiveId`] for the given primitive type `P`.
    pub fn get_ty<P: Primitive>(&self) -> PrimitiveId {
        self.type_ids.intern(&TypeId::of::<P>())
    }

    /// Get a [`Value`] representing the given primitive `p`.
    pub fn get<P: Primitive>(&self, p: P) -> Value {
        let id = self.get_ty::<P>();
        let table = self.tables[id]
            .as_any()
            .downcast_ref::<InternTable<P, Value>>()
            .unwrap();
        p.intern(table)
    }

    /// Get a reference to the primitive value represented by the given [`Value`].
    pub fn unwrap_ref<P: Primitive>(&self, v: Value) -> impl Deref<Target = P> + '_ {
        let id = self.get_ty::<P>();
        let table = self
            .tables
            .get(id)
            .expect("types must be registered before unwrapping")
            .as_any()
            .downcast_ref::<InternTable<P, Value>>()
            .unwrap();
        table.get(v)
    }
    pub fn unwrap<P: Primitive>(&self, v: Value) -> P {
        self.unwrap_ref::<P>(v).clone()
    }

    pub fn register_op(&mut self, op: impl PrimitiveOperation + 'static) -> PrimitiveFunctionId {
        op.register_types(self);
        self.operations.push(DynamicPrimitveOperation::new(op))
    }

    pub fn get_schema(&self, id: PrimitiveFunctionId) -> PrimitiveFunctionSignature {
        self.operations[id].op.signature()
    }

    /// Apply the given operation to the supplied arguments.
    ///
    /// This operation is not particularly efficient, but it is useful when
    /// writing tests or external proof checkers.
    pub fn apply_op(&mut self, id: PrimitiveFunctionId, args: &[Value]) -> Option<Value> {
        let dyn_op = self.operations.unwrap_val(id);
        let res = dyn_op.op.apply(self, args);
        self.operations.insert(id, dyn_op);
        res
    }
    pub(crate) fn apply_vectorized(
        &self,
        id: PrimitiveFunctionId,
        pool: Pool<Vec<Value>>,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        out_var: Variable,
    ) {
        let dyn_op = &self.operations[id];
        dyn_op
            .op
            .apply_vectorized(self, pool, mask, bindings, args, out_var);
    }
}

struct DynamicPrimitveOperation {
    op: Box<dyn PrimitiveOperationExt>,
}

impl DynamicPrimitveOperation {
    fn new(op: impl PrimitiveOperation + 'static) -> Self {
        Self { op: Box::new(op) }
    }
}

trait DynamicInternTable: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn print_value(&self, val: Value, f: &mut fmt::Formatter) -> fmt::Result;
}

impl<P: Primitive> DynamicInternTable for InternTable<P, Value> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn print_value(&self, val: Value, f: &mut fmt::Formatter) -> fmt::Result {
        let p = self.get(val);
        write!(f, "{:?}", &*p)
    }
}

/// The type signature for a primitive function.
pub struct PrimitiveFunctionSignature<'a> {
    pub args: &'a [PrimitiveId],
    pub ret: PrimitiveId,
}

/// A primitive operation on a set of primitive types.
///
/// Most of the time you can get away with using the `lift_operation` macro,
/// which implements this under the hood.
pub trait PrimitiveOperation: Send + Sync {
    fn signature(&self) -> PrimitiveFunctionSignature;
    fn register_types(&self, prims: &mut Primitives);
    fn apply(&self, prims: &Primitives, args: &[Value]) -> Option<Value>;
}

pub(crate) trait PrimitiveOperationExt: PrimitiveOperation {
    fn apply_vectorized(
        &self,
        prims: &Primitives,
        pool: Pool<Vec<Value>>,
        mask: &mut Mask,
        bindings: &mut Bindings,
        args: &[QueryEntry],
        out_var: Variable,
    ) {
        let mut out = pool.get();
        mask.iter_dynamic(
            pool,
            args.iter().map(|v| match v {
                QueryEntry::Var(v) => ValueSource::Slice(&bindings[*v]),
                QueryEntry::Const(c) => ValueSource::Const(*c),
            }),
        )
        .fill_vec(&mut out, Value::stale, |_, args| self.apply(prims, &args));
        bindings.insert(out_var, out);
    }
}

impl<T: PrimitiveOperation> PrimitiveOperationExt for T {}

#[macro_export]
macro_rules! lift_operation_impl {
    ([$arity:expr, $table:expr] fn $name:ident ( $($id:ident : $ty:ty : $n:tt),* ) -> $ret:ty { $body:expr }) => {
         {
            use $crate::{Primitives, PrimitiveOperation, PrimitiveId, PrimitiveFunctionSignature};
            use $crate::Value;
            fn $name(prims: &mut Primitives) -> $crate::PrimitiveFunctionId {
                struct Impl<F> {
                    arg_prims: Vec<PrimitiveId>,
                    ret: PrimitiveId,
                    f: F,
                }

                impl<F: FnMut($($ty),*) -> $ret> Impl<F> {
                    fn new(f: F, prims: &mut Primitives) -> Self {
                        Self {
                            arg_prims: vec![$(prims.get_ty::<$ty>()),*],
                            ret: prims.get_ty::<$ret>(),
                            f,
                        }
                    }
                }

                impl<F: Fn($($ty),*) -> $ret + Send + Sync> PrimitiveOperation for Impl<F> {
                    fn signature(&self) -> PrimitiveFunctionSignature {
                        PrimitiveFunctionSignature {
                            args: &self.arg_prims,
                            ret: self.ret,
                        }
                    }

                    fn apply(&self, prims: &Primitives, args: &[Value]) -> Option<Value> {
                        assert_eq!(args.len(), $arity, "wrong number of arguments to {}", stringify!($name));
                        let ret = (self.f)($(prims.unwrap::<$ty>(args[$n]).clone()),*);
                        Some(prims.get(ret))
                    }

                    fn register_types(&self, prims: &mut Primitives) {
                        $( prims.register_type::<$ty>();)*
                        prims.register_type::<$ret>();
                    }
                }

                fn __impl($($id: $ty),*) -> $ret {
                    $body
                }

                let op = Impl::new(__impl, prims);
                prims.register_op(op)
            }
            $name($table)
        }
    };
}

#[macro_export]
macro_rules! lift_operation_count {
    ([$next:expr, $table:expr] [$($x:ident : $ty:ty : $n: expr),*] fn $name:ident() -> $ret:ty { $body:expr }) => {
        $crate::lift_operation_impl!(
            [$next, $table] fn $name($($x : $ty : $n),*) -> $ret {
                $body
            }
        )
    };
    ([$next:expr, $table:expr] [$($p:ident : $ty:ty : $n:expr),*] fn $name:ident($id:ident : $hd:ty $(,$rest:ident : $tl:ty)*) -> $ret:ty { $body:expr }) => {
        $crate::lift_operation_count!(
            [($next + 1), $table]
            [$($p : $ty : $n,)* $id : $hd : $next]
            fn $name($($rest:$tl),*) -> $ret {
                $body
            }
        )
    };
}

/// Lifts a function into a primitive operation, adding it to the supplied table
/// of primitives.
#[macro_export]
macro_rules! lift_operation {
    ([$table:expr] fn $name:ident ( $($id:ident : $ty:ty),* ) -> $ret:ty { $($body:tt)* } ) => {
        $crate::lift_operation_count!(
            [0, $table]
            []
            fn $name($($id : $ty),*) -> $ret {{
                $($body)*
            }}
        )
    };
}
