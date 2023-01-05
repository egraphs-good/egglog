use super::*;
use binary_search::binary_search_table_by_key;

fn make_value(bits: u32) -> Value {
    Value {
        tag: "testing".into(),
        bits: bits as u64,
    }
}

fn insert_to_map(table: &mut Table, i: u32, ts: u32) {
    let v = make_value(i);
    table.insert(&[v], v, ts);
}

#[test]
fn binary_search() {
    let mut map = Table::default();
    assert_eq!(binary_search_table_by_key(&map, 0), None);
    insert_to_map(&mut map, 1, 1);
    assert_eq!(binary_search_table_by_key(&map, 0), Some(0));
    map.clear();
    for i in 0..128 {
        // have a run of 4 24s and then skip to 26
        let v = if i == 50 || i == 51 { 24 } else { i / 2 };
        insert_to_map(&mut map, i, v);
    }

    assert_eq!(binary_search_table_by_key(&map, 3), Some(6));
    assert_eq!(binary_search_table_by_key(&map, 0), Some(0));
    assert_eq!(binary_search_table_by_key(&map, 63), Some(126));
    assert_eq!(binary_search_table_by_key(&map, 200), None);
    assert_eq!(binary_search_table_by_key(&map, 24), Some(48));
    assert_eq!(binary_search_table_by_key(&map, 25), Some(52));
}
