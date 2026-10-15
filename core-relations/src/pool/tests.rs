use std::{cell::RefCell, mem, rc::Rc};

use super::{Clear, InPoolSet, Pool, PoolSet, PoolState, Pooled, ReusableBitSet, with_pool_set};

#[test]
fn bitset_pool_retains_allocation_and_restores_logical_length() {
    let pool = with_pool_set(|ps| ps.get_pool::<ReusableBitSet>());
    pool.clear();
    let mut source = Pooled::new(ReusableBitSet::default());
    source.grow(17);
    source.set_range(3..9, true);

    let mut spare = pool.get();
    spare.grow(4097);
    spare.set(4096, true);
    let pointer = spare.as_slice().as_ptr();
    let bytes = spare.bytes();
    assert_eq!(bytes, 1024);
    drop(spare);
    assert_eq!(pool.data.borrow().data.len(), 1);
    assert_eq!(pool.data.borrow().bytes, bytes);

    let cloned = Pooled::cloned(&source);
    assert_eq!(cloned.len(), 17);
    assert_eq!(
        cloned.ones().collect::<Vec<_>>(),
        (3..9).collect::<Vec<_>>()
    );
    assert_eq!(cloned.as_slice().as_ptr(), pointer);
    assert_eq!(cloned.bytes(), bytes);
    assert_eq!(pool.data.borrow().bytes, 0);
    drop(cloned);
    assert_eq!(pool.data.borrow().bytes, bytes);

    let mut reused = pool.get();
    assert_eq!(reused.len(), 0);
    assert!(reused.is_clear());
    reused.grow(5);
    assert_eq!(reused.len(), 5);
    assert!(reused.is_clear());
    assert_eq!(reused.as_slice().as_ptr(), pointer);
    reused.grow(4097);
    assert!(
        reused.is_clear(),
        "bits from the previous owner survived reuse"
    );
}

#[test]
fn bitset_pool_budget_counts_retained_capacity_after_shrinking() {
    let mut bitset = ReusableBitSet::default();
    bitset.grow(257);
    let bytes = bitset.bytes();
    assert_eq!(bytes, 64);
    bitset.clone_from(&ReusableBitSet::default());
    assert_eq!(bitset.len(), 0);
    assert_eq!(bitset.bytes(), bytes);

    let mut too_small = PoolState::new(bytes - 1);
    too_small.push(bitset.clone());
    // A fresh clone of an empty value owns no allocation.
    assert!(too_small.data.is_empty());
    too_small.push(bitset);
    assert!(too_small.data.is_empty());
    assert_eq!(too_small.bytes, 0);

    let mut bounded = PoolState::new(bytes);
    for _ in 0..2 {
        let mut bitset = ReusableBitSet::default();
        bitset.grow(257);
        bounded.push(bitset);
    }
    assert_eq!(bounded.data.len(), 1);
    assert_eq!(bounded.bytes, bytes);
    let recycled = bounded.pop();
    assert_eq!(recycled.len(), 0);
    assert_eq!(recycled.bytes(), bytes);
    assert_eq!(bounded.bytes, 0);
}

#[test]
fn bitset_growth_and_set_operations_preserve_logical_contents() {
    let mut left = ReusableBitSet::default();
    left.grow(5);
    left.set(2, true);
    let mut right = ReusableBitSet::default();
    right.grow(513);
    right.set(512, true);
    left.union_with(&right);
    assert_eq!(left.len(), 513);
    assert_eq!(left.ones().collect::<Vec<_>>(), vec![2, 512]);
    assert_eq!(left.bytes(), 128);
    left.symmetric_difference_with(&right);
    assert_eq!(left.ones().collect::<Vec<_>>(), vec![2]);
    left.clear();
    assert_eq!(left.len(), 513);
    assert!(left.is_clear());
    Clear::clear(&mut left);
    assert_eq!(left.len(), 0);
    assert_eq!(left.bytes(), 128);
}

#[test]
fn pooled_does_not_drop() {
    let start = DROP_COUNT.with(|k| k.get());
    {
        // NB: the pools in these tests are really just guiding the types at
        // this point. The objects themselves now end up in the thread-local
        // pools.
        let pool = Pool::<Dropper>::default();
        let mut d1 = pool.get();
        let d2 = pool.get();
        assert!(d1.cleared);
        assert!(d2.cleared);
        d1.set_reuse();
        mem::drop(d1);
        let d3 = pool.get();
        assert!(d3.cleared);
    }
    assert_eq!(DROP_COUNT.with(|k| { k.get() }), start + 3);
}

#[test]
fn refcount() {
    let start = DROP_COUNT.with(|k| k.get());
    let pool = Pool::<Rc<Dropper>>::default();
    let mut d1 = pool.get();
    Rc::get_mut(&mut d1).unwrap().set_reuse();
    let d2 = d1.clone();
    mem::drop(d1);
    assert_eq!(DROP_COUNT.with(|k| { k.get() }), start);
    mem::drop(d2);
    // Reset the pool; dropping its current contents.
    DROP_RC_POOL.with(|pool| mem::take(&mut *pool.borrow_mut()));
    assert_eq!(DROP_COUNT.with(|k| { k.get() }), start + 1);
}

impl InPoolSet<PoolSet> for Dropper {
    fn with_pool<R>(_: &PoolSet, f: impl FnOnce(&Pool<Self>) -> R) -> R {
        DROP_POOL.with(|pool| f(&pool.borrow()))
    }
}

impl InPoolSet<PoolSet> for Rc<Dropper> {
    fn with_pool<R>(_: &PoolSet, f: impl FnOnce(&Pool<Self>) -> R) -> R {
        DROP_RC_POOL.with(|pool| f(&pool.borrow()))
    }
}

// Hacks around the fact that you cannot really have "constructor arguments" for
// a pool.
thread_local! {
    static DROP_COUNT: DropCount = DropCount(Rc::new(RefCell::new(0)));
    static DROP_POOL: RefCell<Pool<Dropper>> = Default::default();
    static DROP_RC_POOL: RefCell<Pool<Rc<Dropper>>> = Default::default();
}

struct Dropper {
    cleared: bool,
    reuse: bool,
}

impl Dropper {
    fn set_reuse(&mut self) {
        self.reuse = true;
    }
}

impl Default for Dropper {
    fn default() -> Self {
        Dropper {
            reuse: false,
            cleared: true,
        }
    }
}

impl Clear for Dropper {
    fn clear(&mut self) {
        self.cleared = true;
        self.reuse = false
    }
    fn reuse(&self) -> bool {
        self.reuse
    }
    fn bytes(&self) -> usize {
        0
    }
}

impl Drop for Dropper {
    fn drop(&mut self) {
        DROP_COUNT.with(|c| c.inc());
    }
}

struct DropCount(Rc<RefCell<usize>>);

impl DropCount {
    fn inc(&self) {
        *self.0.borrow_mut() += 1;
    }
    fn get(&self) -> usize {
        *self.0.borrow()
    }
}
