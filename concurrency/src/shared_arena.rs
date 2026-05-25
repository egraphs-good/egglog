//! Scoped thread-safe arena allocation.
//!
//! This module provides [`SharedArena`], a bump-allocation arena that can be
//! shared by reference across scoped worker threads. Each worker creates a
//! non-`Send`, non-`Sync` [`Handle`] and performs unsynchronized allocations
//! into that handle's private local arena. Allocated values are exposed through
//! [`SharedRef`], an immutable reference-like handle whose lifetime is tied to
//! the borrowed [`SharedArena`].
//!
//! The arena owns all allocated values and runs destructors when the
//! [`SharedArena`] is dropped. Since the arena itself is `Send`, values must be
//! `Send` when they are allocated: the arena may be dropped on a different
//! thread from the one that performed the allocation.

use std::{
    cell::RefCell, marker::PhantomData, mem, ops::Deref, pin::Pin, ptr::NonNull, rc::Rc,
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
/// Values allocated in the arena remain valid until the arena is dropped. For
/// types with destructors, drops run when the arena is dropped. Drop order is
/// last-in, first-out within a single handle's local arena; order across
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
    /// thread, but [`SharedRef`] values allocated by the handle can be shared
    /// when their payload type is `Sync`.
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
    /// The returned [`SharedRef`] remains valid until the arena is dropped.
    /// `T` must be `Send` because the arena may be moved to and dropped on a
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
    pub fn alloc<T>(&self, value: T) -> SharedRef<'arena, T>
    where
        T: Send + 'arena,
    {
        // SAFETY: `self.local` points to a boxed `LocalArena` stored in the
        // parent `SharedArena`. Boxes are never removed from that vector before
        // the `SharedArena` is dropped, and this handle's lifetime prevents the
        // arena from being dropped while the handle is alive.
        let local = unsafe { self.local.as_ref() };
        SharedRef {
            ptr: local.alloc(value),
            _lifetime: PhantomData,
        }
    }
}

/// An immutable reference to a value allocated in a [`SharedArena`].
///
/// `SharedRef` is copyable and dereferences to `T`. It implements `Send` and
/// `Sync` when `T: Sync`, matching the sharing behavior of `&T`.
///
/// # Examples
///
/// ```
/// use egglog_concurrency::{SharedArena, SharedRef};
///
/// let arena = SharedArena::new();
/// let handle = arena.new_handle();
/// let value: SharedRef<'_, usize> = handle.alloc(5);
///
/// assert_eq!(*value, 5);
/// assert_eq!(*value, *value.clone());
/// ```
pub struct SharedRef<'arena, T> {
    ptr: NonNull<T>,
    _lifetime: PhantomData<&'arena T>,
}

impl<T> Copy for SharedRef<'_, T> {}

impl<T> Clone for SharedRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Deref for SharedRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `ptr` is created from a valid bump allocation and the
        // `SharedRef` lifetime is tied to the parent arena. The API never
        // exposes mutable references to allocated values after allocation.
        unsafe { self.ptr.as_ref() }
    }
}

unsafe impl<T: Sync> Send for SharedRef<'_, T> {}
unsafe impl<T: Sync> Sync for SharedRef<'_, T> {}

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
