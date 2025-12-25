use egglog::EGraph;

#[test]
fn test_desugar_includes() {
    let mut egraph = EGraph::default();

    // Create a temporary test file to include
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(format!(
        "egglog_test_include_{}.egg",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&file_path, "(datatype Math (Test))\n(let x (Test))\n").unwrap();
    let include_path = file_path.to_string_lossy().replace('\\', "/");

    // Test desugar with include
    let input = format!(
        r#"
        (datatype Main (Test2))
        (include "{}")
        (let y (Test))
    "#,
        include_path
    );

    let desugared = egraph
        .desugar_program(None, &input)
        .unwrap()
        .iter()
        .map(|cmd| format!("{}", cmd))
        .collect::<Vec<_>>();

    let snapshot = desugared.join("\n");
    insta::assert_snapshot!("desugar_includes", snapshot);

    // Clean up
    std::fs::remove_file(&file_path).ok();
}
