//! Execution masks used for vectorized evaluation of actions.

use std::ops::Range;

use fixedbitset::FixedBitSet;
use smallvec::SmallVec;

use crate::{
    pool::{InPoolSet, Pool, Pooled},
    PoolSet,
};

/// A subset of offsets that are still active.
#[derive(Debug)]
pub(crate) struct Mask {
    data: Pooled<FixedBitSet>,
}

impl Clone for Mask {
    fn clone(&self) -> Self {
        Mask {
            data: Pooled::cloned(&self.data),
        }
    }
}

// NB: this is currently a very basic implementation of execution masks, and for
// highly "sparse" masks with only a few bits set, it's not very efficient. (For
// "dense" masks, it's probably fine, but there's still plenty to do there too.)
//
// We'll want to get end to end tests passing first, but there is probably a
// good amount of low-hanging fruit here for sparse programs, if we have a use
// for them. (e.g. even using a Subset rather than a bitset would be better
// here; with an API for getting the next index, slightly harder but sitll
// doable would be using a TaggedRowBuffer to store bindings, so fill_vec can
// only fill in the needed offsets. Scarrier one would be to use uninitialized
// memory / set_len. We're caching these vectors so it'd probably be fast, but
// then it's not clear how to give them a save API; the mask would have to "own"
// the bindings.).

impl Mask {
    pub(super) fn new(range: Range<usize>, ps: &PoolSet) -> Mask {
        let mut data = ps.get::<FixedBitSet>();
        data.grow(range.end);
        data.set_range(range, true);
        Mask { data }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.data.is_clear()
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(super) fn symmetric_difference(&mut self, other: &Mask) {
        debug_assert!(self.data.is_subset(&other.data));
        self.data.symmetric_difference_with(&other.data);
    }

    pub(super) fn union(&mut self, other: &Mask) {
        self.data.union_with(&other.data);
    }

    /// Iterate over the offsets in the slice that correspond to set offsets in
    /// the `Mask`.
    pub(crate) fn iter<'slice, T>(
        &'slice mut self,
        slice: &'slice [T],
    ) -> MaskIterBase<'slice, 'slice, T> {
        MaskIterBase {
            counter: 0,
            slice,
            mask: &mut self.data,
        }
    }

    pub(crate) fn iter_dynamic<'a, T>(
        &'a mut self,
        pool: Pool<Vec<T>>,
        sources: impl Iterator<Item = ValueSource<'a, T>>,
    ) -> MaskIterDynamicSource<'a, 'a, T> {
        MaskIterDynamicSource {
            counter: 0,
            data: SmallVec::from_iter(sources),
            pool,
            mask: &mut self.data,
        }
    }

    /// Set all entries in the mask to false.
    pub(super) fn clear(&mut self) {
        self.data.clear();
    }
}

pub(crate) enum IterResult<T> {
    Item(T),
    Skip,
    Done,
}

pub(crate) trait MaskIter {
    type Item;
    // Internal operations; callers should not use these directly.

    fn inc_counter(&mut self) -> usize;
    fn get_at(&mut self, idx: usize) -> IterResult<Self::Item>;
    fn remove(&mut self, idx: usize);

    /// Iterate over the contents of the iterator: if the function `f` returns
    /// false, the corresponding item is removed from the mask.
    fn retain(mut self, mut f: impl FnMut(Self::Item) -> bool)
    where
        Self: Sized,
    {
        loop {
            let cur = self.inc_counter();
            let next = match self.get_at(cur) {
                IterResult::Item(item) => item,
                IterResult::Skip => continue,
                IterResult::Done => break,
            };
            if !f(next) {
                self.remove(cur);
            }
        }
    }

    /// A variant of `retain` that supports writing to an output vector.
    ///
    /// If the function `f` returns `None`, the corresponding item is removed.
    /// All "removed" items, (including ones corresponding to entries in the
    /// mask that are removed in the current call) have `default()` added at the
    /// given offset to `out`.
    ///
    /// N.B. `f` also gets the current offset. This is because an action
    /// instruction requires it.
    fn fill_vec<Out>(
        mut self,
        out: &mut Vec<Out>,
        mut default: impl FnMut() -> Out,
        mut f: impl FnMut(usize, Self::Item) -> Option<Out>,
    ) where
        Self: Sized,
    {
        loop {
            let cur = self.inc_counter();
            let next = match self.get_at(cur) {
                IterResult::Item(item) => item,
                IterResult::Skip => {
                    out.push(default());
                    continue;
                }
                IterResult::Done => break,
            };
            match f(cur, next) {
                Some(next) => out.push(next),
                None => {
                    out.push(default());
                    self.remove(cur);
                }
            }
        }
    }
    fn assign_vec<Out>(mut self, out: &mut [Out], mut f: impl FnMut(usize, Self::Item) -> Out)
    where
        Self: Sized,
    {
        loop {
            let cur = self.inc_counter();
            let next = match self.get_at(cur) {
                IterResult::Item(item) => item,
                IterResult::Skip => {
                    continue;
                }
                IterResult::Done => break,
            };
            out[cur] = f(cur, next);
        }
    }

    fn assign_vec_and_retain<Out>(
        mut self,
        out: &mut [Out],
        mut f: impl FnMut(usize, Self::Item) -> Option<Out>,
    ) where
        Self: Sized,
    {
        loop {
            let cur = self.inc_counter();
            let next = match self.get_at(cur) {
                IterResult::Item(item) => item,
                IterResult::Skip => {
                    continue;
                }
                IterResult::Done => break,
            };
            match f(cur, next) {
                Some(next) => out[cur] = next,
                None => {
                    self.remove(cur);
                }
            }
        }
    }

    /// Iterate over the contents of the iterator.
    fn for_each(self, mut f: impl for<'a> FnMut(Self::Item))
    where
        Self: Sized,
    {
        self.retain(|item| {
            f(item);
            true
        })
    }

    fn zip<T>(self, slice: &[T]) -> ZipIter<Self, T>
    where
        Self: Sized,
    {
        ZipIter { base: self, slice }
    }
}

pub(crate) struct MaskIterDynamicSource<'slice, 'mask, T> {
    counter: usize,
    data: SmallVec<[ValueSource<'slice, T>; 4]>,
    pool: Pool<Vec<T>>,
    mask: &'mask mut FixedBitSet,
}

// NB: We could get this to work by passing references as well. This way is just
// a bit easier when `T = Value`
impl<T: Clone> MaskIter for MaskIterDynamicSource<'_, '_, T>
where
    Vec<T>: InPoolSet<PoolSet>,
{
    type Item = Pooled<Vec<T>>;
    fn inc_counter(&mut self) -> usize {
        let res = self.counter;
        self.counter += 1;
        res
    }
    fn get_at(&mut self, idx: usize) -> IterResult<Self::Item> {
        if self.mask.contains(idx) {
            let mut result = self.pool.get();
            result.reserve(self.data.len());
            result.extend(self.data.iter().map(|x| match x {
                ValueSource::Const(x) => (*x).clone(),
                ValueSource::Slice(x) => x[idx].clone(),
            }));
            IterResult::Item(result)
        } else if idx < self.mask.len() {
            IterResult::Skip
        } else {
            IterResult::Done
        }
    }
    fn remove(&mut self, idx: usize) {
        self.mask.set(idx, false);
    }
}

pub(crate) struct MaskIterBase<'slice, 'mask, T> {
    counter: usize,
    slice: &'slice [T],
    mask: &'mask mut FixedBitSet,
}

impl<'slice, T> MaskIter for MaskIterBase<'slice, '_, T> {
    type Item = &'slice T;
    fn inc_counter(&mut self) -> usize {
        let res = self.counter;
        self.counter += 1;
        res
    }
    fn get_at(&mut self, idx: usize) -> IterResult<&'slice T> {
        if self.mask.contains(idx) {
            IterResult::Item(&self.slice[idx])
        } else if idx < self.slice.len() {
            IterResult::Skip
        } else {
            IterResult::Done
        }
    }
    fn remove(&mut self, idx: usize) {
        self.mask.set(idx, false);
    }
}
pub(crate) struct ZipIter<'slice, Base, T> {
    base: Base,
    slice: &'slice [T],
}

impl<'slice, Base: MaskIter, T> MaskIter for ZipIter<'slice, Base, T> {
    type Item = (Base::Item, &'slice T);
    fn inc_counter(&mut self) -> usize {
        self.base.inc_counter()
    }
    fn get_at(&mut self, idx: usize) -> IterResult<Self::Item> {
        match self.base.get_at(idx) {
            IterResult::Item(base) => IterResult::Item((base, &self.slice[idx])),
            IterResult::Skip => IterResult::Skip,
            IterResult::Done => IterResult::Done,
        }
    }

    fn remove(&mut self, idx: usize) {
        self.base.remove(idx);
    }
}

pub(crate) enum ValueSource<'a, T> {
    Const(T),
    Slice(&'a [T]),
}
