use egglog::{EGraph, ast::GenericCommand};

#[test]
fn rule_names_are_stable() {
    let mut egraph = EGraph::default();
    let commands = egraph
        .resolve_program(
            None,
            r#"
                (sort Math)
                (constructor Var (String) Math)
                (constructor Wrap (Math) Math)
                (rule ((= x (Var "a"))) ((union x (Var "b"))))
                (function table (Math) Math :no-merge)
                (rewrite (Wrap (Var "c")) (Var "c"))
                (relation edge (Math Math))
                (birewrite (Wrap (Var "d")) (Var "d"))
                (rule ((= x (Var "e"))) ((union x (Var "f"))) :name "named-rule")
                (rewrite (Wrap (Var "g")) (Var "g") :name "named-rewrite")
                (birewrite (Wrap (Var "h")) (Var "h") :name "named-birewrite")
            "#,
        )
        .unwrap();

    let rule_names = commands
        .into_iter()
        .filter_map(|command| match command {
            GenericCommand::Rule { rule } => Some(rule.name),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        rule_names,
        [
            "(rule ((= x (Var 'a')))\n      ((union x (Var 'b')))\n         )",
            "(rewrite (Wrap (Var 'c')) (Var 'c'))",
            "(birewrite (Wrap (Var 'd')) (Var 'd'))=>",
            "(birewrite (Wrap (Var 'd')) (Var 'd'))<=",
            "named-rule",
            "named-rewrite",
            "named-birewrite=>",
            "named-birewrite<=",
        ]
    );
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
