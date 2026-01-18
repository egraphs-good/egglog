use crate::core_relations::{ExternalFunction, Value, make_external_func};
use crate::numeric_id::NumericId;

/// Compute a CRC32 hash over the Value representations in `values`.
pub fn crc32_hash(values: &[Value]) -> Value {
    let mut hasher = crc32fast::Hasher::new();
    for value in values {
        hasher.update(&value.rep().to_le_bytes());
    }
    Value::new(hasher.finalize())
}

/// Create an ExternalFunction wrapper for `crc32_hash`.
pub fn crc32_external_func() -> Box<dyn ExternalFunction + 'static> {
    Box::new(make_external_func(|_, values| Some(crc32_hash(values))))
}

#[cfg(test)]
mod tests {
    use super::{crc32_external_func, crc32_hash};
    use crate::core_relations::{Database, Value};

    #[test]
    fn crc32_hash_is_deterministic() {
        let values = [Value::new_const(1), Value::new_const(2), Value::new_const(3)];
        let first = crc32_hash(&values);
        let second = crc32_hash(&values);
        assert_eq!(first, second);
    }

    #[test]
    fn crc32_hash_changes_with_input() {
        let values_a = [Value::new_const(1), Value::new_const(2), Value::new_const(3)];
        let values_b = [Value::new_const(1), Value::new_const(2), Value::new_const(4)];
        assert_ne!(crc32_hash(&values_a), crc32_hash(&values_b));
    }

    #[test]
    fn crc32_external_matches_direct_hash() {
        let values = [Value::new_const(10), Value::new_const(20), Value::new_const(30)];
        let expected = crc32_hash(&values);
        let func = crc32_external_func();
        let db = Database::new();
        db.with_execution_state(|state| {
            let actual = func.invoke(state, &values).expect("crc32 should return a value");
            assert_eq!(actual, expected);
        });
    }
}
