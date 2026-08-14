use super::*;

#[test]
fn arena_handle_is_created_only_on_first_packed_allocation() {
    let arena = SharedArena::new();
    let handle = LazyArenaHandle::new(&arena);
    assert!(handle.handle.get().is_none());
    handle.get();
    assert!(handle.handle.get().is_some());
}
