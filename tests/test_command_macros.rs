use egglog::ast::*;
use egglog::util::FreshGen;
use egglog::*;
use egglog_ast::span::{RustSpan, Span};
use std::sync::Arc;

// Macro that prefixes rule names with a specific prefix
struct PrefixRuleMacro {
    prefix: String,
}

impl CommandMacro for PrefixRuleMacro {
    fn transform(
        &self,
        command: Command,
        symbol_gen: &mut util::SymbolGen,
        _type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        match command {
            Command::Rule { mut rule } => {
                rule.name = symbol_gen.fresh(&format!("{}_{}", self.prefix, rule.name));
                Ok(vec![Command::Rule { rule }])
            }
            cmd => Ok(vec![cmd]),
        }
    }
}

// Macro that duplicates every rule
struct DuplicateRuleMacro;

impl CommandMacro for DuplicateRuleMacro {
    fn transform(
        &self,
        command: Command,
        symbol_gen: &mut util::SymbolGen,
        _type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        match command {
            Command::Rule { rule } => {
                let mut rule1 = rule.clone();
                let mut rule2 = rule;
                rule1.name = symbol_gen.fresh(&format!("dup1_{}", rule1.name));
                rule2.name = symbol_gen.fresh(&format!("dup2_{}", rule2.name));
                Ok(vec![
                    Command::Rule { rule: rule1 },
                    Command::Rule { rule: rule2 },
                ])
            }
            cmd => Ok(vec![cmd]),
        }
    }
}

// Macro that adds a comment after each rule showing its name
struct CommentAfterRuleMacro;

impl CommandMacro for CommentAfterRuleMacro {
    fn transform(
        &self,
        command: Command,
        _symbol_gen: &mut util::SymbolGen,
        _type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        match command {
            Command::Rule { rule } => {
                let rule_name = rule.name.clone();
                Ok(vec![
                    Command::Rule { rule },
                    // Comments don't exist in the AST but we can use PrintSize with None
                    // as a marker that the macro ran
                    Command::PrintSize(span!(), Some(format!("rule_{}", rule_name))),
                ])
            }
            cmd => Ok(vec![cmd]),
        }
    }
}

#[test]
fn test_single_macro_with_desugar_program() {
    let mut egraph = EGraph::default();
    egraph
        .command_macros_mut()
        .register(Arc::new(PrefixRuleMacro {
            prefix: "test".to_string(),
        }));

    let input = r#"
        (datatype Math (Num i64))
        (rule ((Num x)) ((Num (+ x 1))))
        (let a (Num 1))
    "#;

    let result = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    // The output should have the rule name prefixed
    let output = result.join("\n");

    // Check that the datatype was desugared into a sort and constructor
    assert!(output.contains("(sort Math)"), "Expected sort declaration");
    assert!(
        output.contains("(function Num (i64) Math :cost 1)"),
        "Expected Num constructor"
    );

    // Check that the rule name was prefixed - it should have test_ prefix
    // The original rule name is generated as $rule_<n>, so it becomes test_$rule_<n>
    assert!(
        output.contains("test_$rule_"),
        "Expected rule name to be prefixed with test_: {}",
        output
    );

    // Check that let is desugared correctly
    assert!(output.contains("(let a (Num 1))"), "Expected let statement");
}

#[test]
fn test_multiple_macros_compose_with_desugar_program() {
    let mut egraph = EGraph::default();

    // First macro prefixes with "first_", second prefixes with "second_"
    egraph
        .command_macros_mut()
        .register(Arc::new(PrefixRuleMacro {
            prefix: "first".to_string(),
        }));
    egraph
        .command_macros_mut()
        .register(Arc::new(PrefixRuleMacro {
            prefix: "second".to_string(),
        }));

    let input = r#"(rule ((Num x)) ((Num (+ x 1))))"#;

    let result = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let output = result.join("\n");

    // Both prefixes should be applied in order: second_first_$rule_<n>
    assert!(
        output.contains("second_first_$rule_"),
        "Expected rule name to have both prefixes in order: {}",
        output
    );
}

#[test]
fn test_duplicate_macro_creates_two_rules() {
    let mut egraph = EGraph::default();
    egraph
        .command_macros_mut()
        .register(Arc::new(DuplicateRuleMacro));

    let input = r#"
        (datatype Math (Num i64))
        (rule ((Num x)) ((Num (+ x 1))))
    "#;

    let result = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let output = result.join("\n");

    // Should have two rules with dup1_ and dup2_ prefixes
    assert!(
        output.contains("dup1_$rule_"),
        "Expected first duplicated rule: {}",
        output
    );
    assert!(
        output.contains("dup2_$rule_"),
        "Expected second duplicated rule: {}",
        output
    );

    // Count the number of rule declarations - should be 2
    let rule_count = output.matches("(rule ").count();
    assert_eq!(
        rule_count, 2,
        "Expected exactly 2 rules, got {}",
        rule_count
    );
}

#[test]
fn test_macro_adds_commands_after_rules() {
    let mut egraph = EGraph::default();
    egraph
        .command_macros_mut()
        .register(Arc::new(CommentAfterRuleMacro));

    let input = r#"
        (rule ((Num x)) ((Num (+ x 1))))
        (let a (Num 1))
    "#;

    let result = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let output = result.join("\n");

    // Should have a print-size command after the rule with the rule's name
    assert!(
        output.contains("(print-size rule_$rule_"),
        "Expected print-size marker after rule: {}",
        output
    );

    // The let should still be there, unaffected
    assert!(
        output.contains("(let a (Num 1))"),
        "Expected let statement: {}",
        output
    );
}

#[test]
fn test_complex_macro_composition() {
    let mut egraph = EGraph::default();

    // Apply duplicate first, then prefix both duplicated rules
    egraph
        .command_macros_mut()
        .register(Arc::new(DuplicateRuleMacro));
    egraph
        .command_macros_mut()
        .register(Arc::new(PrefixRuleMacro {
            prefix: "prefixed".to_string(),
        }));

    let input = r#"(rule ((Num x)) ((Num (+ x 1))))"#;

    let result = egraph
        .desugar_program(None, input)
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let output = result.join("\n");

    // DuplicateRuleMacro creates two rules with dup1_ and dup2_ prefixes
    // Then PrefixRuleMacro adds "prefixed_" to both
    // So we should have: prefixed_dup1_$rule_<n> and prefixed_dup2_$rule_<n>
    assert!(
        output.contains("prefixed_dup1_$rule_"),
        "Expected first rule with both transformations: {}",
        output
    );
    assert!(
        output.contains("prefixed_dup2_$rule_"),
        "Expected second rule with both transformations: {}",
        output
    );

    // Should have exactly 2 rules
    let rule_count = output.matches("(rule ").count();
    assert_eq!(rule_count, 2, "Expected exactly 2 rules after duplication");
}

#[test]
fn test_macros_work_with_actual_program_execution() {
    // Test that macros not only transform the syntax but the program still runs
    let mut egraph = EGraph::default();
    egraph
        .command_macros_mut()
        .register(Arc::new(DuplicateRuleMacro));

    let result = egraph.parse_and_run_program(
        None,
        r#"
        (datatype Math (Num i64))
        (rule ((Num x)) ((Num (+ x 1))))
        (let a (Num 1))
        (run 1)
        (check (= a (Num 2)))
        "#,
    );

    // The program should run successfully with duplicated rules
    assert!(
        result.is_ok(),
        "Program with duplicated rules should run: {:?}",
        result
    );
}

struct TypeInfoReader;

impl CommandMacro for TypeInfoReader {
    fn transform(
        &self,
        command: Command,
        symbol_gen: &mut util::SymbolGen,
        type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        // if this is a rule command, typecheck the query
        match command {
            Command::Rule { rule } => {
                type_info.typecheck_facts(symbol_gen, &rule.body)?;
                Ok(vec![Command::Rule { rule }])
            }
            cmd => Ok(vec![cmd]),
        }
    }
}

#[test]
fn test_macro_accesses_type_info() {
    let mut egraph = EGraph::default();
    egraph
        .command_macros_mut()
        .register(Arc::new(TypeInfoReader));
    let result = egraph.parse_and_run_program(
        None,
        r#"
        (datatype Math (Num i64))
        (rule ((Num x)) ((Num (+ x 1))))
        (let a (Num 1))
        (constructor Math () B)
        (union a (B))
        (check (= (B) (Num 1)))
        "#,
    );
    // The program should run successfully with the macro accessing type info
    assert!(
        result.is_ok(),
        "Program with type info reading macro should run: {:?}",
        result
    );
}
