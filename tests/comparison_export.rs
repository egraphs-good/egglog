#![cfg(feature = "comparison")]
use egglog::EGraph;
use egraph_comparison::{Certificate, Database, FunctionKind, certificate, compare, verify};

fn export(program: &str) -> Database {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, program).unwrap();
    graph.serialize_for_comparison().unwrap()
}

#[test]
fn exports_constructor_function_relation_and_empty_declarations() {
    let db = export(
        r#"
        (datatype E (A) (B) (F E))
        (function cost (E) i64 :merge (max old new))
        (relation seen (E))
        (A)
        (set (cost (A)) 7)
        (seen (A))
        (let $global (A))
    "#,
    );
    assert_eq!(db.functions["A"].kind, FunctionKind::Constructor);
    assert_eq!(db.functions["cost"].kind, FunctionKind::Function);
    assert_eq!(db.functions["seen"].kind, FunctionKind::Function);
    assert!(db.functions.contains_key("B"));
    assert!(!db.functions.contains_key("$global"));
    assert_eq!(db.functions.len(), 5);
    assert!(db.rows.iter().any(|row| row.function == "cost"));
    assert!(db.rows.iter().any(|row| row.function == "seen"));
    let decoded = serde_json::from_str(&serde_json::to_string(&db).unwrap()).unwrap();
    assert!(compare(&db, &decoded).unwrap().database_equal);
}

#[test]
fn compares_reordered_insertions_and_union_representatives() {
    let header = "(datatype E (A) (B) (F E))";
    let left = export(&format!("{header} (F (A)) (B) (union (A) (B))"));
    let right = export(&format!("{header} (B) (F (B)) (A) (union (B) (A))"));
    assert!(compare(&left, &right).unwrap().database_equal);
}

#[test]
fn compares_real_cyclic_egraphs() {
    let header = "(datatype E (Placeholder) (X E))";
    let cycle = "(let $c (X (Placeholder))) (union (Placeholder) $c) (delete (Placeholder))";
    let left = export(&format!("{header} {cycle}"));
    let second = cycle.replace("$c", "$d");
    let right = export(&format!("{header} {cycle} {second}"));
    assert!(compare(&left, &right).unwrap().database_equal);
}

#[test]
fn term_witnesses_and_function_only_differences() {
    let header = "(datatype E (A) (B)) (function cost (E) i64 :merge (max old new))";
    let left = export(&format!("{header} (set (cost (A)) 1)"));
    let right = export(&format!("{header} (set (cost (A)) 2)"));
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
    let witness = certificate(&left, &right).unwrap().unwrap();
    assert!(matches!(witness, Certificate::Row { .. }));
    assert!(verify(&witness, &left, &right).unwrap());
    let right = export(&format!("{header} (set (cost (A)) 1) (B)"));
    assert!(matches!(
        certificate(&left, &right).unwrap().unwrap(),
        Certificate::MissingTerm { .. }
    ));
}

#[test]
fn subsumed_rows_remain_in_the_export() {
    let left = export("(datatype E (A) (B)) (A) (B)");
    let right = export("(datatype E (A) (B)) (A) (B) (subsume (A))");
    assert!(right.rows.iter().any(|r| r.function == "A" && r.subsumed));
    let result = compare(&left, &right).unwrap();
    assert!(result.terms_equal);
    assert!(!result.database_equal);
}

#[test]
fn scalar_values_are_exact_and_zeros_are_normalized() {
    let header = "(datatype E (Float f64) (Str String) (Int i64))";
    let left = export(&format!(
        r#"{header} (Float -0.0) (Str "a\n\"b") (Int 9223372036854775807)"#
    ));
    let right = export(&format!(
        r#"{header} (Int 9223372036854775807) (Str "a\n\"b") (Float 0.0)"#
    ));
    assert!(compare(&left, &right).unwrap().database_equal);
    let right = export(&format!(
        r#"{header} (Int 9223372036854775806) (Str "a\n\"b") (Float 0.0)"#
    ));
    assert!(!compare(&left, &right).unwrap().terms_equal);
}

#[test]
fn unsupported_exports_fail_instead_of_truncating() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            "(sort V (Vec i64)) (datatype E (C V)) (C (vec-of 1 2))",
        )
        .unwrap();
    assert!(
        graph
            .serialize_for_comparison()
            .unwrap_err()
            .to_string()
            .contains("does not support sort V")
    );
    let mut graph = EGraph::new_with_term_encoding();
    graph
        .parse_and_run_program(None, "(datatype E (A)) (A)")
        .unwrap();
    assert!(
        graph
            .serialize_for_comparison()
            .unwrap_err()
            .to_string()
            .contains("term/proof encoding")
    );
}

#[cfg(feature = "bin")]
#[test]
fn cli_exports_complete_database_even_with_visualization_limits() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("egglog-comparison-export-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.egg");
    fs::write(&input, "(datatype E (A) (B)) (A) (B)").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_egglog"))
        .args([
            "--to-comparison-json",
            "--max-functions",
            "0",
            "--max-calls-per-function",
            "0",
        ])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let db: Database =
        serde_json::from_slice(&fs::read(input.with_extension("comparison.json")).unwrap())
            .unwrap();
    assert_eq!(db.rows.len(), 2);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn relation_names_are_independent_of_desugaring_order() {
    let left = export("(relation a-b (i64)) (relation ab (i64)) (a-b 1) (ab 2)");
    let right = export("(relation ab (i64)) (relation a-b (i64)) (ab 2) (a-b 1)");
    assert!(compare(&left, &right).unwrap().database_equal);
    let missing = export("(relation a-b (i64)) (relation ab (i64)) (a-b 1)");
    assert!(compare(&left, &missing).unwrap().terms_equal);
    assert!(!compare(&left, &missing).unwrap().database_equal);
}

#[test]
fn user_nonunionable_constructors_still_build_terms() {
    let mut graph = EGraph::default();
    let mut program = graph
        .parse_program(None, "(sort E) (constructor A () E) (A)")
        .unwrap();
    let egglog::ast::Command::Sort { unionable, .. } = &mut program[0] else {
        unreachable!()
    };
    *unionable = false;
    graph.run_program(program).unwrap();
    let db = graph.serialize_for_comparison().unwrap();
    assert_eq!(db.functions["A"].kind, FunctionKind::Constructor);
}
