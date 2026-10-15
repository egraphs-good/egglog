use crate::{DenseIdMap, IdVec, NumericId, define_id};

define_id!(pub(crate) Id, u32, "a unique id");

#[test]
#[should_panic]
fn id_out_of_bounds() {
    Id::from_usize(u32::MAX as usize + 5);
}

fn to_vec(map: &DenseIdMap<Id, &'static str>) -> Vec<(Id, &'static str)> {
    map.iter().map(|(x, y)| (x, *y)).collect::<Vec<_>>()
}

#[test]
fn push() {
    let mut map = DenseIdMap::<Id, &'static str>::new();
    let first = map.next_id();
    assert_eq!(first, map.push("zero"));
    let second = map.next_id();
    assert_eq!(second, map.push("one"));
    assert_eq!(to_vec(&map), vec![(first, "zero"), (second, "one")]);
}

#[test]
fn basic_id_map() {
    let mut map = DenseIdMap::<Id, &'static str>::new();
    map.insert(Id::from_usize(0), "zero");
    map.insert(Id::from_usize(4), "four");

    assert_eq!(
        to_vec(&map),
        vec![(Id::from_usize(0), "zero"), (Id::from_usize(4), "four")]
    );

    assert_eq!(map.unwrap_val(Id::from_usize(0)), "zero");

    assert_eq!(to_vec(&map), vec![(Id::from_usize(4), "four")]);

    map.insert(Id::from_usize(2), "two");
    assert_eq!(
        to_vec(&map),
        vec![(Id::from_usize(2), "two"), (Id::from_usize(4), "four")]
    );
}

#[test]
fn get_or_insert() {
    let mut map = DenseIdMap::<Id, &'static str>::new();
    let id = Id::from_usize(3);
    assert_eq!(map.get_or_default(id), &"");
    assert_eq!(
        map.get_or_insert(id, || panic!("this should not be called")),
        &""
    );

    map.get_or_insert(id, || "three");
    assert_eq!(to_vec(&map), vec![(id, "")]);
}

#[test]
fn basic_id_vec() {
    let mut v = IdVec::<Id, usize>::default();
    let id0 = v.push(0);
    let id1 = v.push(1);
    assert_eq!(0, v[id0]);
    assert_eq!(1, v[id1]);
    let mut i = 2;
    v.resize_with(100, || {
        let res = i;
        i += 1;
        res
    });
}

#[test]
fn dense_id_map_clone_from_reuses_capacity_and_preserves_holes() {
    let mut source = DenseIdMap::<Id, String>::default();
    source.insert(Id::new(2), "two".into());
    source.insert(Id::new(7), "seven".into());
    let mut destination = DenseIdMap::<Id, String>::with_capacity(64);
    destination.insert(Id::new(20), "old".into());
    let pointer = destination.data.as_ptr();
    let capacity = destination.capacity();
    destination.clear();
    destination.clone_from(&source);
    assert_eq!(destination, source);
    assert_eq!(destination.data.as_ptr(), pointer);
    assert_eq!(destination.capacity(), capacity);

    destination.clone_from(&DenseIdMap::default());
    assert_eq!(destination.n_ids(), 0);
    assert_eq!(destination.data.as_ptr(), pointer);
    assert_eq!(destination.capacity(), capacity);
}

#[test]
fn id_vec_clone_from_reuses_capacity() {
    let mut source = IdVec::<Id, String>::default();
    source.push("first".into());
    source.push("second".into());
    let mut destination = IdVec::<Id, String>::default();
    destination.resize_with(64, || "old".into());
    let pointer = destination.data.as_ptr();
    let capacity = destination.capacity();
    destination.clear();
    destination.clone_from(&source);
    assert_eq!(destination.data, source.data);
    assert_eq!(destination.data.as_ptr(), pointer);
    assert_eq!(destination.capacity(), capacity);

    destination.clone_from(&IdVec::default());
    assert_eq!(destination.len(), 0);
    assert_eq!(destination.data.as_ptr(), pointer);
    assert_eq!(destination.capacity(), capacity);
}
