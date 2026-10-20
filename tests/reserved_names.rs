use egglog::ast::{Command, Parser, Span};
use egglog::util::{FreshGen, SymbolGen};
use egglog::{CommandMacro, CommandMacroRegistry, EGraph, Error, TypeInfo};
use std::sync::Arc;

const SOURCE: &str = r#"
    (datatype Row (Pair i64 i64) (Unique i64))
    (relation seen (i64))
    (Pair 1 2)
    (check (Pair _ @v) (= @v 2))
    (rule ((Unique @Unique)) ((seen @Unique) (subsume (Unique @Unique))))
    (Unique 3)
    (run 1)
    (check (seen 3))
    (let @x@1 (Unique 11))
    (let @x___1 (Unique 22))
    (check (= @x@1 (Unique 11)) (= @x___1 (Unique 22)))
"#;

#[test]
fn source_names_do_not_capture_generated_variables() {
    for mut egraph in [
        EGraph::default(),
        EGraph::new_with_term_encoding(),
        EGraph::new_with_proofs(),
    ] {
        // `_` needs a fresh `v`; subsume typechecking needs a fresh `Unique`.
        egraph.parse_and_run_program(None, SOURCE).unwrap();
    }
}

#[test]
fn independently_parsed_ast_reserves_names_in_the_receiving_egraph() {
    let program = Parser::default()
        .get_program_from_string(None, SOURCE)
        .unwrap();
    for mut egraph in [
        EGraph::default(),
        EGraph::new_with_term_encoding(),
        EGraph::new_with_proofs(),
    ] {
        egraph.run_program(program.clone()).unwrap();
    }
}

#[test]
fn user_proof_sort_is_reserved_before_internal_proof_names() {
    EGraph::new_with_proofs()
        .with_proof_testing()
        .parse_and_run_program(
            None,
            "(datatype @Proof (Value i64)) (Value 1) (check (Value 1))",
        )
        .unwrap();
}

#[test]
fn printed_term_and_proof_programs_run_through_the_default_parser() {
    for mut egraph in [EGraph::new_with_term_encoding(), EGraph::new_with_proofs()] {
        let printed = egraph
            .resolve_program(None, SOURCE)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(printed.contains("@x@1"));
        assert!(printed.contains("@x___1"));
        let parsed = Parser::default()
            .get_program_from_string(None, &printed)
            .unwrap();
        let reprinted = parsed
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        // Parsing adds schedule sequences; identifier spellings must stay exact.
        let names = |text: &str| {
            text.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '"'))
                .filter(|word| word.contains('@'))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&reprinted), names(&printed));
        EGraph::default().run_program(parsed).unwrap();
    }
}

#[test]
fn include_names_are_reserved_before_later_parent_wildcards() {
    for mut egraph in [
        EGraph::default(),
        EGraph::new_with_term_encoding(),
        EGraph::new_with_proofs(),
    ] {
        egraph
            .parse_and_run_program(
                None,
                r#"(include "tests/header/reserved-names.egg") (Included 2) (check (= _ (Included 2)))"#,
            )
            .unwrap();
    }
}

struct AppendRuleset {
    fresh: bool,
}

impl CommandMacro for AppendRuleset {
    fn transform(
        &self,
        command: Command,
        symbol_gen: &mut SymbolGen,
        _type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        let name = if self.fresh {
            symbol_gen.fresh("x")
        } else {
            "@x".into()
        };
        Ok(vec![command, Command::AddRuleset(Span::Panic, name)])
    }
}

#[test]
fn later_command_macros_avoid_literal_names_from_earlier_macros() {
    let mut registry = CommandMacroRegistry::new();
    registry.register(Arc::new(AppendRuleset { fresh: false }));
    registry.register(Arc::new(AppendRuleset { fresh: true }));
    let mut egraph = EGraph::default();
    let commands = registry
        .apply(
            Command::AddRuleset(Span::Panic, "seed".into()),
            &mut SymbolGen::new("@".into()),
            egraph.type_info(),
        )
        .unwrap();
    let names = commands
        .iter()
        .map(|command| match command {
            Command::AddRuleset(_, name) => name.as_str(),
            _ => panic!("expected a ruleset"),
        })
        .collect::<Vec<_>>();
    assert_eq!(names, ["seed", "@x@1", "@x", "@x@2"]);
    egraph.run_program(commands).unwrap();
}

#[test]
fn reservation_does_not_treat_string_data_as_names() {
    let source = format!("(let $text \"@v@{}\") (check (= _ 1))", usize::MAX);
    EGraph::default()
        .parse_and_run_program(None, &source)
        .unwrap();
}

#[test]
fn reservation_defers_include_errors_until_execution() {
    let path = std::env::temp_dir().join(format!("egglog-reservation-{}.egg", std::process::id()));
    std::fs::write(&path, "(").unwrap();
    let mut egraph = EGraph::default();
    let result = egraph.parse_and_run_program(
        None,
        &format!("(let $before 1) (include \"{}\")", path.display()),
    );
    std::fs::remove_file(path).unwrap();
    assert!(result.is_err());
    let output = egraph
        .parse_and_run_program(None, "(extract $before)")
        .unwrap();
    assert_eq!(output[0].to_string(), "1\n");
}
