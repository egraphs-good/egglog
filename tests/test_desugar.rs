use egglog::EGraph;

#[test]
fn rule_names_are_stable() {
    let rule_names = EGraph::default()
        .resolve_program(
            None,
            r#"
                (datatype Math (Var String) (Wrap Math))
                (ruleset generated-names)
                (rule ((= x (Var "a"))) ((union x (Var "b")))
                    :ruleset generated-names :unsafe-seminaive :no-decomp
                    :internal-include-subsumed)
                (rewrite (Wrap (Var "c")) (Var "c") :subsume
                    :when ((= (Var "c") (Var "c"))) :ruleset generated-names)
                (birewrite (Wrap (Var "d")) (Var "d")
                    :when ((= (Var "d") (Var "d"))) :ruleset generated-names)
                (rule ((= x (Var "e"))) ((union x (Var "f"))) :name "named-rule")
                (rewrite (Wrap (Var "g")) (Var "g") :name "named-rewrite")
                (birewrite (Wrap (Var "h")) (Var "h") :name "named-birewrite")
            "#,
        )
        .unwrap()
        .into_iter()
        .filter_map(|command| match command {
            egglog::ast::GenericCommand::Rule { rule } => Some(rule.name),
            _ => None,
        })
        .collect::<Vec<_>>();

    let expected = [
        "(rule ((= x (Var 'a')))\n      ((union x (Var 'b')))\n        :ruleset generated-names  :unsafe-seminaive :no-decomp :internal-include-subsumed)",
        "(rewrite (Wrap (Var 'c')) (Var 'c') :subsume :when ((= (Var 'c') (Var 'c'))) :ruleset generated-names)",
        "(birewrite (Wrap (Var 'd')) (Var 'd') :when ((= (Var 'd') (Var 'd'))) :ruleset generated-names)=>",
        "(birewrite (Wrap (Var 'd')) (Var 'd') :when ((= (Var 'd') (Var 'd'))) :ruleset generated-names)<=",
        "named-rule",
        "named-rewrite",
        "named-birewrite=>",
        "named-birewrite<=",
    ];
    assert_eq!(rule_names, expected);
}

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
        (include "{include_path}")
        (let y (Test))
    "#
    );

    let desugared = egraph
        .resolve_program(None, &input)
        .unwrap()
        .iter()
        .map(|cmd| format!("{cmd}"))
        .collect::<Vec<_>>();

    let snapshot = desugared.join("\n");
    insta::assert_snapshot!("desugar_includes", snapshot);

    // Clean up
    std::fs::remove_file(&file_path).ok();
}
