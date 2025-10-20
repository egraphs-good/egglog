use std::{cell::RefCell, mem, rc::Rc};

use super::{Clear, InPoolSet, Pool, PoolSet};

#[test]
fn pooled_does_not_drop() {
    let start = DROP_COUNT.with(DropCount::get);
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
    let start = DROP_COUNT.with(DropCount::get);
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
    static DROP_POOL: RefCell<Pool<Dropper>> = RefCell::default();
    static DROP_RC_POOL: RefCell<Pool<Rc<Dropper>>> = RefCell::default();
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
        self.reuse = false;
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
        DROP_COUNT.with(DropCount::inc);
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
