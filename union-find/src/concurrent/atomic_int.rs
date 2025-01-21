//! Basic trait for encapsulating atomic integer operations.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// A simple trait used to abstract over atomic integer types.
///
/// As a bonus, this allows us to abstract away the memory orderings of choice,
/// making the actual algorithms a bit easier to read.
pub trait AtomicInt: Default + Send + Sync + 'static {
    type Underlying: Copy + Eq + Ord;
    fn from_usize(value: usize) -> Self;
    fn as_usize(value: Self::Underlying) -> usize;
    fn load(&self) -> Self::Underlying;
    fn store(&self, value: Self::Underlying);
    fn cas(
        &self,
        current: Self::Underlying,
        new: Self::Underlying,
    ) -> Result<Self::Underlying, Self::Underlying>;
}

impl AtomicInt for AtomicU32 {
    type Underlying = u32;
    fn from_usize(value: usize) -> Self {
        AtomicU32::new(u32::try_from(value).expect("usize doesn't fit in u32"))
    }
    fn as_usize(value: u32) -> usize {
        value as usize
    }
    fn load(&self) -> u32 {
        self.load(LOAD_ORDERING)
    }
    fn store(&self, value: u32) {
        self.store(value, STORE_ORDERING);
    }
    fn cas(&self, current: u32, new: u32) -> Result<u32, u32> {
        self.compare_exchange(current, new, CAS_SUCCESS_ORDERING, CAS_FAILURE_ORDERING)
    }
}

impl AtomicInt for AtomicU64 {
    type Underlying = u64;
    fn from_usize(value: usize) -> Self {
        AtomicU64::new(value as u64)
    }
    fn as_usize(value: u64) -> usize {
        value as usize
    }
    fn load(&self) -> u64 {
        self.load(LOAD_ORDERING)
    }
    fn store(&self, value: u64) {
        self.store(value, STORE_ORDERING);
    }
    fn cas(&self, current: u64, new: u64) -> Result<u64, u64> {
        self.compare_exchange(current, new, CAS_SUCCESS_ORDERING, CAS_FAILURE_ORDERING)
    }
}
impl AtomicInt for AtomicUsize {
    type Underlying = usize;

    fn from_usize(value: usize) -> Self {
        AtomicUsize::new(value)
    }
    fn as_usize(value: usize) -> usize {
        value
    }

    fn load(&self) -> usize {
        self.load(LOAD_ORDERING)
    }
    fn store(&self, value: usize) {
        self.store(value, STORE_ORDERING);
    }
    fn cas(&self, current: usize, new: usize) -> Result<usize, usize> {
        self.compare_exchange(current, new, CAS_SUCCESS_ORDERING, CAS_FAILURE_ORDERING)
    }
}

impl<T: AtomicInt> AtomicInt for Padded<T> {
    type Underlying = T::Underlying;

    fn from_usize(value: usize) -> Self {
        Padded(T::from_usize(value))
    }
    fn as_usize(value: T::Underlying) -> usize {
        T::as_usize(value)
    }

    fn load(&self) -> T::Underlying {
        self.0.load()
    }
    fn store(&self, value: T::Underlying) {
        self.0.store(value)
    }
    fn cas(
        &self,
        current: T::Underlying,
        new: T::Underlying,
    ) -> Result<T::Underlying, T::Underlying> {
        self.0.cas(current, new)
    }
}

/// Pad a given atomic integer type out to a full cache line, avoiding false
/// sharing.
///
/// We aren't currently using this because it doesn't appear to help.
#[repr(align(64))]
#[derive(Default)]
pub(crate) struct Padded<T>(T);

/*

We could do the following, but we don't have good reason to believe it is
correct, or preserves the linearizability guarantees of the baseline
data-structure.

Still, code like this is helpful in microbenchmarks to measure the overhead of
memory barriers for the code. We've observed 5-10% overhead on an M1 Ultra CPU
with high concurrency.

const STORE_ORDERING: Ordering = Ordering::Relaxed;
const LOAD_ORDERING: Ordering = Ordering::Relaxed;
const CAS_SUCCESS_ORDERING: Ordering = Ordering::Relaxed;
const CAS_FAILURE_ORDERING: Ordering = Ordering::Relaxed;

*/

const STORE_ORDERING: Ordering = Ordering::Release;
const LOAD_ORDERING: Ordering = Ordering::Acquire;
const CAS_SUCCESS_ORDERING: Ordering = Ordering::AcqRel;
const CAS_FAILURE_ORDERING: Ordering = Ordering::Acquire;
