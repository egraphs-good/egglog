use egglog::EGraph;

#[test]
fn test_desugar_includes() {
    let mut egraph = EGraph::default();

    // Create a test file to include
    std::fs::write("/tmp/test_include.egg", "(datatype Test)\n(let x (Test))\n").unwrap();

    // Test desugar with include
    let input = r#"
        (datatype Main)
        (include "/tmp/test_include.egg")
        (let y (Main))
    "#;

    let desugared = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|cmd| format!("{:?}", cmd))
        .collect::<Vec<_>>();

    // Should have datatype, include (not expanded), and let commands
    assert_eq!(desugared.len(), 3);
    assert!(desugared[0].contains("datatype Main"));
    assert!(desugared[1].contains("include \"/tmp/test_include.egg\""));
    assert!(desugared[2].contains("let y"));

    // Clean up
    std::fs::remove_file("/tmp/test_include.egg").ok();
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
