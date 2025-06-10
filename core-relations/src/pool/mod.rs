//! Utilities for pooling object allocations.

use std::{
    cell::{Cell, RefCell},
    fmt,
    hash::{Hash, Hasher},
    mem::{self, ManuallyDrop},
    ops::{Deref, DerefMut},
    ptr,
    rc::Rc,
};

use fixedbitset::FixedBitSet;
use hashbrown::HashTable;
use numeric_id::{DenseIdMap, IdVec};

use crate::{
    action::{Instr, PredictedVals},
    common::{HashMap, HashSet, IndexMap, IndexSet, ShardId, Value},
    free_join::execute::FrameUpdate,
    hash_index::{BufferedSubset, ColumnIndex, TableEntry},
    offsets::SortedOffsetVector,
    table::TableEntry as SwTableEntry,
    table_spec::Constraint,
    ColumnId, RowId,
};

#[cfg(test)]
mod tests;

/// A trait for types whose allocations can be reused.
pub trait Clear: Default {
    /// Clear the object.
    ///
    /// The end result must be equivalent to `Self::default()`.
    fn clear(&mut self);
    /// Indicate whether or not this object should be reused.
    fn reuse(&self) -> bool {
        true
    }
    /// A rough approximation for the in-memory overhead of this object.
    fn bytes(&self) -> usize;
}

impl<T> Clear for Vec<T> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 256
    }
    fn bytes(&self) -> usize {
        self.capacity() * mem::size_of::<T>()
    }
}

impl<T: Clear> Clear for Rc<T> {
    fn clear(&mut self) {
        Rc::get_mut(self).unwrap().clear()
    }
    fn reuse(&self) -> bool {
        Rc::strong_count(self) == 1 && Rc::weak_count(self) == 0
    }
    fn bytes(&self) -> usize {
        mem::size_of::<T>()
    }
}

impl<T: Clear> Clone for Pooled<Rc<T>>
where
    Rc<T>: InPoolSet<PoolSet>,
{
    fn clone(&self) -> Self {
        Pooled {
            data: self.data.clone(),
        }
    }
}

impl<T> Clear for HashSet<T> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.capacity() * mem::size_of::<T>()
    }
}

impl<T> Clear for HashTable<T> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.capacity() * mem::size_of::<T>()
    }
}

impl<K, V> Clear for HashMap<K, V> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.capacity() * mem::size_of::<(K, V)>()
    }
}

impl<K, V> Clear for IndexMap<K, V> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.capacity() * (mem::size_of::<u64>() + mem::size_of::<(K, V)>())
    }
}

impl<T> Clear for IndexSet<T> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.capacity() * (mem::size_of::<u64>() + mem::size_of::<T>())
    }
}

impl Clear for FixedBitSet {
    fn clear(&mut self) {
        self.clone_from(&Default::default());
    }
    fn reuse(&self) -> bool {
        !self.is_empty()
    }
    fn bytes(&self) -> usize {
        self.len() / 8
    }
}

impl<K, V> Clear for IdVec<K, V> {
    fn clear(&mut self) {
        self.clear()
    }
    fn reuse(&self) -> bool {
        self.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.capacity() * mem::size_of::<V>()
    }
}

struct PoolState<T> {
    data: Vec<T>,
    bytes: usize,
    limit: usize,
}

impl<T: Clear> PoolState<T> {
    fn new(limit: usize) -> Self {
        PoolState {
            data: Vec::new(),
            bytes: 0,
            limit,
        }
    }

    fn push(&mut self, mut item: T) {
        if !item.reuse() {
            return;
        }
        if self.bytes + item.bytes() > self.limit {
            return;
        }
        item.clear();
        self.bytes += item.bytes();
        self.data.push(item);
    }

    fn pop(&mut self) -> T {
        if let Some(got) = self.data.pop() {
            self.bytes -= got.bytes();
            got
        } else {
            Default::default()
        }
    }

    fn clear_and_shrink(&mut self) {
        self.data.clear();
        self.bytes = 0;
        self.data.shrink_to_fit();
    }
}

/// A shared pool of objects.
pub struct Pool<T> {
    data: Rc<RefCell<PoolState<T>>>,
}

impl<T> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Pool {
            data: self.data.clone(),
        }
    }
}

impl<T: Clear> Default for Pool<T> {
    fn default() -> Self {
        Pool {
            data: Rc::new(RefCell::new(PoolState::new(usize::MAX))),
        }
    }
}

impl<T: Clear + InPoolSet<PoolSet>> Pool<T> {
    pub(crate) fn new(limit: usize) -> Pool<T> {
        Pool {
            data: Rc::new(RefCell::new(PoolState::new(limit))),
        }
    }
    /// Get an empty value of type `T`, potentially reused from the pool.
    pub(crate) fn get(&self) -> Pooled<T> {
        let empty = self.data.borrow_mut().pop();

        Pooled {
            data: ManuallyDrop::new(empty),
        }
    }

    /// Clear the contents of the pool and release any memory associated with it.
    pub(crate) fn clear(&self) {
        let mut data_mut = self.data.borrow_mut();
        data_mut.clear_and_shrink();
    }
}

/// An owned value of type `T` that can be returned to a memory pool when it is
/// no longer used.
pub struct Pooled<T: Clear + InPoolSet<PoolSet>> {
    data: ManuallyDrop<T>,
}

impl<T: Clear + InPoolSet<PoolSet>> Default for Pooled<T> {
    fn default() -> Self {
        with_pool_set(|ps| ps.get::<T>())
    }
}

impl<T: Clear + fmt::Debug + InPoolSet<PoolSet>> fmt::Debug for Pooled<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &T = &self.data;
        data.fmt(f)
    }
}
impl<T: Clear + PartialEq + InPoolSet<PoolSet>> PartialEq for Pooled<T> {
    fn eq(&self, other: &Self) -> bool {
        // This form rid of a spuriou clippy warning about unconditional recursion.
        <T as PartialEq>::eq(&self.data, &other.data)
    }
}

impl<T: Clear + InPoolSet<PoolSet> + Eq> Eq for Pooled<T> {}

impl<T: Clear + Hash + InPoolSet<PoolSet>> Hash for Pooled<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<T: Clear + InPoolSet<PoolSet> + 'static> Pooled<T> {
    /// Clear the contents the wrapped object. If the object cannot be reused,
    /// attempt to fetch another value from the pool.
    ///
    /// This method can be used in concert with `relinquish` to provide a
    /// `clear` operation that hands data back to the pool, and then grabs it
    /// back again if it needs to be reused.
    ///
    /// This pattern is likely only suitable for "temporary" buffers.
    pub(crate) fn refresh(this: &mut Pooled<T>) {
        this.data.clear();
        if this.data.reuse() {
            return;
        }
        let pool = with_pool_set(|ps| ps.get_pool::<T>());
        let mut other = pool.data.borrow_mut().pop();
        if !other.reuse() {
            return;
        }
        let slot: &mut T = &mut this.data;
        mem::swap(slot, &mut other);
    }

    pub(crate) fn into_inner(this: Pooled<T>) -> T {
        // SAFETY: ownership of `this.data` is transferred to the caller. We
        // will not drop `this` or use it again.
        let inner = unsafe { ptr::read(&this.data) };
        mem::forget(this);
        ManuallyDrop::into_inner(inner)
    }

    pub(crate) fn new(data: T) -> Pooled<T> {
        Pooled {
            data: ManuallyDrop::new(data),
        }
    }
}

impl<T: Clear + Clone + InPoolSet<PoolSet>> Pooled<T> {
    pub(crate) fn cloned(this: &Pooled<T>) -> Pooled<T> {
        let mut res = with_pool_set(|ps| ps.get::<T>());
        res.clone_from(this);
        res
    }
}

impl<T: Clear + InPoolSet<PoolSet>> Drop for Pooled<T> {
    fn drop(&mut self) {
        let reuse = self.data.reuse();
        if !reuse {
            // SAFETY: we own `self.data` and being in the drop method means no
            // one else will access it.
            unsafe { ManuallyDrop::drop(&mut self.data) };
            return;
        }
        self.data.clear();
        let t: &T = &self.data;
        // SAFETY: ownership of `self.data` is transferred to the pool
        with_pool_set(|ps| {
            T::with_pool(ps, |pool| {
                pool.data.borrow_mut().push(unsafe { ptr::read(t) })
            })
        });
    }
}

impl<T: Clear + InPoolSet<PoolSet>> Deref for Pooled<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T: Clear + InPoolSet<PoolSet>> DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

/// Helper trait for allowing the trait resolution system to infer the correct
/// pool type during allocation.
pub trait InPoolSet<PoolSet>
where
    Self: Sized + Clear,
{
    fn with_pool<R>(pool_set: &PoolSet, f: impl FnOnce(&Pool<Self>) -> R) -> R;
}

macro_rules! pool_set {
    ($vis:vis $name:ident { $($ident:ident : $ty:ty [ $bytes:expr ],)* }) => {
        $vis struct $name {
            $(
                $ident: Pool<$ty>,
            )*
        }

        impl Default for $name {
            fn default() -> Self {
                $name {
                $(
                    $ident: Pool::new($bytes),
                )*
                }
            }
        }

        impl $name {
            $vis fn get_pool<T: InPoolSet<Self>>(&self) -> Pool<T> {
                T::with_pool(self, Pool::clone)
            }

            $vis fn get<T: InPoolSet<Self> + Default>(&self) -> Pooled<T> {
                self.get_pool().get()
            }
            $vis fn clear(&self) {
                $( self.$ident.clear(); )*
            }
        }

        $(
            impl InPoolSet<$name> for $ty {
                fn with_pool<R>(pool_set: &$name, f: impl FnOnce(&Pool<Self>) -> R) -> R {
                    f(&pool_set.$ident)
                }
            }
        )*
    }
}

// The main thread-local memory pool used for reusing allocations. The syntax is:
//
// <name> : <type> [ <bytes> ],
//
// Where `name` is not used for anything, `type` feeds into the `InPoolSet` machinery and allows
// anything of that type to be allocated using `with_pool_set`, and `bytes` is a per-type limit on
// the total bytes that can be buffered in a single (per-thread) memory pool.

pool_set! {
    pub PoolSet {
        vec_vals: Vec<Value> [ 1 << 25 ],
        vec_cell_vals: Vec<Cell<Value>> [ 1 << 25 ],
        // TODO: work on scaffolding/DI/etc. so that we can share allocations
        // between vec_vals and shared_vals.
        rows: Vec<RowId> [ 1 << 25 ],
        offset_vec: SortedOffsetVector [ 1 << 20 ],
        column_index: IndexMap<Value, BufferedSubset> [ 1 << 20 ],
        constraints: Vec<Constraint> [ 1 << 20 ],
        bitsets: FixedBitSet [ 1 << 20 ],
        instrs: Vec<Instr> [ 1 << 20 ],
        frame_updates: FrameUpdate [ 1 << 25 ],
        frame_update_vecs: Vec<Pooled<FrameUpdate>> [ 1 << 20 ],
        tuple_indexes: HashTable<TableEntry<BufferedSubset>> [ 1 << 20 ],
        staged_outputs: HashTable<SwTableEntry> [ 1 << 25 ],
        predicted_vals: PredictedVals [ 1 << 20 ],
        shard_hist: DenseIdMap<ShardId, usize> [ 1 << 20 ],
        instr_indexes: Vec<u32> [ 1 << 20 ],
        cached_subsets: IdVec<ColumnId, std::sync::OnceLock<std::sync::Arc<ColumnIndex>>> [ 4 << 20 ],
    }
}

/// Run `f` on the thread-local [`PoolSet`].
pub(crate) fn with_pool_set<R>(f: impl FnOnce(&PoolSet) -> R) -> R {
    POOL_SET.with(|pool_set| f(pool_set))
}

thread_local! {
    /// A thread-local pool set. All pooled allocations land back in the local thread.
    ///
    /// We don't drop this PoolSet because it does not contain any resources
    /// that need to be released, other than memory (which will be reclaimed
    /// when the process exits, right after drop runs).
    ///
    /// For large egraphs, this be a big runtime win. The main egglog binary
    /// avoids dropping the egraph for the same reason.
    static POOL_SET: ManuallyDrop<PoolSet> = Default::default();
}
