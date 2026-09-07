use egraph_comparison::{Database, Error, FormatVersion, RowId};

#[test]
fn validation_errors_expose_context_without_parsing_messages() {
    let mut db: Database = serde_json::from_value(serde_json::json!({
        "version":1, "classes":{"x":{"sort":"E"}},
        "functions":{"f":{"kind":"constructor","inputs":["E"],"output":"E"}},
        "rows":[{"function":"f","inputs":[],"output":"x"}]
    }))
    .unwrap();
    assert!(
        matches!(db.validate(), Err(Error::Arity { row, function, expected: 1, actual: 0 })
        if row == RowId::new_const(0) && function == "f")
    );
    db.rows[0].inputs.push("missing".into());
    assert!(
        matches!(db.validate(), Err(Error::UnknownClass { row, class })
        if row == RowId::new_const(0) && class == "missing")
    );
    db.version = FormatVersion::new_const(99);
    assert!(
        matches!(db.validate(), Err(Error::UnsupportedVersion { version })
        if version == FormatVersion::new_const(99))
    );
}

#[test]
fn typed_format_version_preserves_scalar_wire_encoding() {
    let db = Database::default();
    assert_eq!(serde_json::to_value(&db).unwrap()["version"], 1);
    let parsed: Database = serde_json::from_value(serde_json::to_value(&db).unwrap()).unwrap();
    assert_eq!(parsed.version, FormatVersion::V1);
}
