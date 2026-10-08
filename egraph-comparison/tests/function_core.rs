use egraph_comparison::{Database, compare};
#[test]
fn function_only_structure_is_observable_at_the_core_layer() {
    let mut left: Database = serde_json::from_value(serde_json::json!({
        "version":1,"classes":{"x":{"sort":"E"},"y":{"sort":"E"}},
        "functions":{"f":{"kind":"function","inputs":["E"],"output":"E"},"g":{"kind":"function","inputs":["E"],"output":"E"}},
        "rows":[{"function":"f","inputs":["x"],"output":"y"},{"function":"g","inputs":["x"],"output":"x"}]
    })).unwrap();
    let right = left.clone();
    left.rows[1].inputs[0] = "y".into();
    left.rows[1].output = "y".into();
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
}
