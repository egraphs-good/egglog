//! Scoped thread-safe arena allocation.
//!
//! This module provides [`SharedArena`], a bump-allocation arena that can be
//! shared by reference across scoped worker threads. Each worker creates a
//! non-`Send`, non-`Sync` [`Handle`] and performs unsynchronized allocations
//! into that handle's private local arena. Allocated references remain valid
//! for the lifetime of the borrowed [`SharedArena`].
//!
//! Values allocated with [`Handle::alloc`] are dropped with the arena. Since
//! the arena itself is `Send`, those values must be `Send`: their destructors
//! may run on a different thread from the one that allocated them.
//!
//! [`Handle::alloc_layout`] is the narrow escape hatch for packed allocations.
//! It returns an uninitialized, non-`Send` [`RawAllocation`]. The caller writes
//! a header and any trailing data, then explicitly publishes a shared reference
//! to the header. Raw allocations reclaim only their storage; they never run
//! destructors.

use std::{
    alloc::Layout, cell::RefCell, marker::PhantomData, mem, pin::Pin, ptr::NonNull, rc::Rc,
    sync::Mutex,
};

use bumpalo::Bump;

/// A thread-safe scoped arena for bump-allocating shared immutable references.
///
/// `SharedArena` is designed for scoped parallelism: share `&SharedArena` with
/// worker tasks, create one [`Handle`] inside each task, and allocate through
/// that handle. Handles are deliberately not `Send` or `Sync`, so allocation
/// from a single local arena stays thread-local.
///
/// Values returned by [`Handle::alloc`] remain valid until the arena is dropped.
/// Their destructors run last-in, first-out within one handle; order across
/// handles is unspecified.
///
/// # Examples
///
/// Sharing an arena by reference across a Rayon scope:
///
/// ```
/// use egglog_concurrency::SharedArena;
///
/// let arena = SharedArena::new();
/// let mut values = Vec::new();
///
/// rayon::scope(|scope| {
///     let (left, right) = (&arena, &arena);
///     scope.spawn(move |_| {
///         let handle = left.new_handle();
///         assert_eq!(*handle.alloc(1), 1);
///     });
///     scope.spawn(move |_| {
///         let handle = right.new_handle();
///         assert_eq!(*handle.alloc(2), 2);
///     });
/// });
///
/// let handle = arena.new_handle();
/// values.push(handle.alloc(3));
/// assert_eq!(*values[0], 3);
/// ```
///
/// Allocated references may outlive the handle that created them:
///
/// ```
/// use egglog_concurrency::SharedArena;
///
/// let arena = SharedArena::new();
/// let value;
/// {
///     let handle = arena.new_handle();
///     value = handle.alloc(String::from("scoped"));
/// }
///
/// assert_eq!(value.as_str(), "scoped");
/// ```
pub struct SharedArena {
    locals: Mutex<Vec<Pin<Box<LocalArena>>>>,
}

impl Default for SharedArena {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedArena {
    /// Create an empty arena.
    ///
    /// # Examples
    ///
    /// ```
    /// use egglog_concurrency::SharedArena;
    ///
    /// let arena = SharedArena::new();
    /// let handle = arena.new_handle();
    /// assert_eq!(*handle.alloc(10), 10);
    /// ```
    pub fn new() -> Self {
        Self {
            locals: Mutex::new(Vec::new()),
        }
    }

    /// Create a thread-local allocation handle for this arena.
    ///
    /// Handles are cheap to allocate and are intended to be created inside the
    /// scoped thread that will use them. A handle cannot be sent to another
    /// thread, but references allocated by the handle can be shared when their
    /// payload type is `Sync`.
    ///
    /// # Examples
    ///
    /// ```
    /// use egglog_concurrency::SharedArena;
    ///
    /// let arena = SharedArena::new();
    /// std::thread::scope(|scope| {
    ///     let arena = &arena;
    ///     scope.spawn(move || {
    ///         let handle = arena.new_handle();
    ///         assert_eq!(*handle.alloc(99), 99);
    ///     });
    /// });
    /// ```
    ///
    /// Handles intentionally do not implement `Send`:
    ///
    /// ```compile_fail
    /// use egglog_concurrency::SharedArena;
    ///
    /// fn assert_send<T: Send>(_: T) {}
    ///
    /// let arena = SharedArena::new();
    /// let handle = arena.new_handle();
    /// assert_send(handle);
    /// ```
    pub fn new_handle(&self) -> Handle<'_> {
        let mut locals = self.locals.lock().unwrap();
        locals.push(Box::pin(LocalArena::new()));
        let local = NonNull::from(locals.last().unwrap().as_ref().get_ref());
        Handle {
            local,
            _arena: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

/// A thread-local handle for allocating values in a [`SharedArena`].
///
/// `Handle` values borrow their parent arena and cannot be sent or shared
/// across threads. Create a separate handle in each scoped worker that needs to
/// allocate.
pub struct Handle<'arena> {
    local: NonNull<LocalArena>,
    _arena: PhantomData<&'arena SharedArena>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'arena> Handle<'arena> {
    /// Allocate `value` in the parent [`SharedArena`].
    ///
    /// The returned reference remains valid until the arena is dropped. `T`
    /// must be `Send` because the arena may be moved to and dropped on a
    /// different thread from the allocating handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use egglog_concurrency::SharedArena;
    ///
    /// let arena = SharedArena::new();
    /// let handle = arena.new_handle();
    /// let value = handle.alloc(vec![1, 2, 3]);
    ///
    /// assert_eq!(value.as_slice(), &[1, 2, 3]);
    /// ```
    pub fn alloc<T>(&self, value: T) -> &'arena T
    where
        T: Send + 'arena,
    {
        // SAFETY: `self.local` points to a boxed `LocalArena` stored in the
        // parent `SharedArena`. Boxes are never removed from that vector before
        // the `SharedArena` is dropped, and this handle's lifetime prevents the
        // arena from being dropped while the handle is alive.
        let local = unsafe { self.local.as_ref() };
        let value = local.alloc(value);
        // SAFETY: the pinned LocalArena remains owned by the parent arena for
        // `'arena`, and its bump allocation is never moved or mutably exposed.
        unsafe { value.as_ref() }
    }

    /// Allocate uninitialized storage with exactly `layout` in the parent
    /// [`SharedArena`].
    ///
    /// The returned [`RawAllocation`] is an initialization token. It exposes a
    /// raw pointer for constructing a packed allocation, but no reference to
    /// the storage exists until
    /// [`RawAllocation::assume_init_no_drop`] publishes a header at the start
    /// of the allocation. Dropping the token simply abandons the storage until
    /// the arena itself is reclaimed.
    ///
    /// Raw allocations never run destructors. Use [`Handle::alloc`] instead
    /// when values need cleanup.
    ///
    /// The token intentionally does not implement `Send` or `Sync`:
    ///
    /// ```compile_fail
    /// use std::alloc::Layout;
    /// use egglog_concurrency::SharedArena;
    ///
    /// let arena = SharedArena::new();
    /// let handle = arena.new_handle();
    /// let raw = handle.alloc_layout(Layout::new::<usize>());
    /// std::thread::scope(|scope| {
    ///     scope.spawn(move || drop(raw));
    /// });
    /// ```
    pub fn alloc_layout(&self, layout: Layout) -> RawAllocation<'arena> {
        // SAFETY: `self.local` points to a boxed `LocalArena` stored in the
        // parent `SharedArena`. Boxes are never removed from that vector before
        // the `SharedArena` is dropped, and the raw allocation's lifetime keeps
        // that arena borrowed until the initialization token is consumed or
        // dropped.
        let local = unsafe { self.local.as_ref() };
        RawAllocation {
            ptr: local.alloc_layout(layout),
            layout,
            _arena: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

/// Uninitialized storage owned by a [`SharedArena`].
///
/// A raw allocation is an opaque, single-use publication token returned by
/// [`Handle::alloc_layout`]. It is deliberately not `Send` or `Sync`: initialize
/// it on the same thread that owns the allocating [`Handle`]. Its storage stays
/// allocated until the parent arena is dropped, even if the token is abandoned
/// because initialization panics.
///
/// No destructor is registered for any part of this allocation; dropping the
/// arena reclaims only its storage.
pub struct RawAllocation<'arena> {
    ptr: NonNull<u8>,
    layout: Layout,
    _arena: PhantomData<&'arena SharedArena>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'arena> RawAllocation<'arena> {
    /// Return the allocation's exact requested layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Return a pointer to the start of the uninitialized allocation.
    ///
    /// Dereferencing this pointer is unsafe. Although this method requires a
    /// mutable borrow of the initialization token, raw pointers copied from it
    /// are not tracked by Rust's borrow checker.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Publish an initialized `T` header without registering its destructor.
    ///
    /// The allocation may be larger than `T`; this is the intended way to
    /// publish a sized header followed by packed trailing arrays. This method
    /// checks, before treating the storage as `T`, that the original layout is
    /// large enough and sufficiently aligned for `T`. A failed check panics
    /// without publishing a reference.
    ///
    /// `T` may require drop, but no destructor will run. Use [`Handle::alloc`]
    /// when values need cleanup.
    ///
    /// # Safety
    ///
    /// If the checked layout requirements hold, the caller must ensure that:
    ///
    /// - a valid `T` has been completely initialized at [`Self::as_mut_ptr`];
    /// - every byte that `T`'s safe interface can read, including trailing
    ///   storage reached through offsets or pointers in `T`, is initialized;
    /// - the `Send` and `Sync` behavior of `T` accurately accounts for any
    ///   trailing values its interface can access; and
    /// - no pointer retained from [`Self::as_mut_ptr`] is used to mutate the
    ///   allocation after publication, except through valid interior
    ///   mutability.
    ///
    /// Raw trailing storage is not described by Rust's type system, so these
    /// invariants cannot be checked by this API.
    pub unsafe fn assume_init_no_drop<T>(self) -> &'arena T
    where
        T: Send + 'arena,
    {
        assert!(
            self.layout.size() >= mem::size_of::<T>(),
            "raw arena allocation of {} bytes is too small for a {}-byte header",
            self.layout.size(),
            mem::size_of::<T>()
        );
        assert!(
            self.layout.align() >= mem::align_of::<T>(),
            "raw arena allocation alignment {} is insufficient for header alignment {}",
            self.layout.align(),
            mem::align_of::<T>()
        );
        // SAFETY: the caller initialized a valid `T`, the checked allocation
        // can hold it, and the method's remaining invariants prohibit later
        // mutation. The allocation remains valid for `'arena`.
        unsafe { self.ptr.cast::<T>().as_ref() }
    }
}

struct LocalArena {
    bump: Bump,
    drops: RefCell<Vec<DropEntry>>,
}

// SAFETY: `LocalArena` is stored behind a `SharedArena` mutex and only exposed
// to a single non-`Send`, non-`Sync` `Handle` for allocation. It may be dropped
// on another thread when its parent `SharedArena` is moved, but `Handle::alloc`
// requires allocated values to be `Send`, so running destructors on that thread
// is sound.
unsafe impl Send for LocalArena {}

impl LocalArena {
    fn new() -> Self {
        Self {
            bump: Bump::new(),
            drops: RefCell::new(Vec::new()),
        }
    }

    fn alloc<T>(&self, value: T) -> NonNull<T>
    where
        T: Send,
    {
        let value = self.bump.alloc(value);
        let ptr = NonNull::from(value);

        if mem::needs_drop::<T>() {
            self.drops.borrow_mut().push(DropEntry {
                ptr: ptr.cast(),
                drop: drop_value::<T>,
            });
        }

        ptr
    }

    fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        self.bump.alloc_layout(layout)
    }
}

impl Drop for LocalArena {
    fn drop(&mut self) {
        for entry in self.drops.get_mut().iter().rev() {
            // SAFETY: Every drop entry is registered immediately after a
            // successful allocation of a value of the corresponding type. This
            // is the only place where registered destructors are run.
            unsafe {
                (entry.drop)(entry.ptr.as_ptr());
            }
        }
    }
}

struct DropEntry {
    ptr: NonNull<()>,
    drop: unsafe fn(*mut ()),
}

unsafe fn drop_value<T>(ptr: *mut ()) {
    // SAFETY: `ptr` was recorded from a live `T` allocation in `LocalArena::alloc`.
    unsafe {
        ptr.cast::<T>().drop_in_place();
    }
}
