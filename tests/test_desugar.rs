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
        .map(|cmd| format!("{:?}", cmd))
        .collect::<Vec<_>>();

    // Should contain datatype, include, and let commands
    assert!(
        desugared.iter().any(|cmd| cmd.contains("datatype Main")),
        "desugared output missing datatype definition: {:?}",
        desugared
    );
    assert!(
        desugared
            .iter()
            .any(|cmd| cmd.contains(&format!("include \"{}\"", include_path))),
        "desugared output missing include for {}: {:?}",
        include_path,
        desugared
    );
    assert!(
        desugared.iter().any(|cmd| cmd.contains("let y")),
        "desugared output missing let binding: {:?}",
        desugared
    );
    assert!(
        desugared.iter().all(|cmd| !cmd.contains("datatype Math")),
        "desugared output should not include code from include file (datatype Math): {:?}",
        desugared
    );
    assert!(
        desugared.iter().all(|cmd| !cmd.contains("let x")),
        "desugared output should not include code from include file (let x): {:?}",
        desugared
    );

    // Clean up
    std::fs::remove_file(&file_path).ok();
}

#[test]
fn test_desugar_without_includes() {
    let mut egraph = EGraph::default();

    let input = r#"
        (datatype Test)
        (let x (Test))
        (union x (Test))
    "#;

    let desugared = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|cmd| format!("{:?}", cmd))
        .collect::<Vec<_>>();

    // Should have 3 commands
    assert_eq!(desugared.len(), 3);
    assert!(desugared[0].contains("datatype Test"));
    assert!(desugared[1].contains("let x"));
    assert!(desugared[2].contains("union"));
}
