#[cfg(test)]
mod tests {
    use crate::CommandOutput;
    use crate::ast::{
        FunctionSubtype, GenericAction, GenericExpr, ResolvedCommand, RuleEvalMode,
        sanitize_internal_names,
    };
    use crate::core::ResolvedCall;

    fn term_encode(source: &str) -> Vec<ResolvedCommand> {
        let mut egraph = crate::EGraph::new_with_term_encoding();
        egraph.resolve_program(None, source).unwrap()
    }

    /// The proof encoder reads body variables' `term_proof`s from the RHS via
    /// `:unsafe-seminaive` lookups. Assert this produces the same database as
    /// the safe baseline (the same rules annotated `:naive`), for a hardcoded
    /// handful of files (running it across all tests would be too slow).
    #[test]
    fn unsafe_seminaive_matches_naive() {
        let files = [
            "tests/calc.egg",
            "tests/integer_math.egg",
            "tests/fibonacci-demand.egg",
            "tests/until.egg",
        ];

        for file in files {
            let source = std::fs::read_to_string(file)
                .unwrap_or_else(|e| panic!("couldn't read {file}: {e}"));

            // Guard against a vacuous comparison: the two encodings must differ.
            let encode = |naive: bool| -> String {
                let mut egraph = crate::EGraph::new_with_proofs();
                egraph.proof_state.force_proof_naive = naive;
                egraph
                    .resolve_program(Some(file.to_string()), &source)
                    .unwrap_or_else(|e| panic!("{file} resolve (naive={naive}) failed: {e}"))
                    .iter()
                    .map(|cmd| cmd.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            assert!(
                encode(false).contains(":unsafe-seminaive") && encode(false) != encode(true),
                "expected {file} to exercise the `:unsafe-seminaive` encoding path"
            );

            // `print-size` summarizes the whole database (per-function row
            // counts, sorted) deterministically.
            let program = format!("{source}\n(print-size)");

            let run = |naive: bool| -> Vec<CommandOutput> {
                let mut egraph = crate::EGraph::new_with_proofs();
                egraph.proof_state.force_proof_naive = naive;
                egraph
                    .parse_and_run_program(Some(file.to_string()), &program)
                    .unwrap_or_else(|e| panic!("{file} (naive={naive}) failed: {e}"))
            };

            let unsafe_seminaive = CommandOutput::snapshot_stable_under_proof_encoding(&run(false));
            let naive = CommandOutput::snapshot_stable_under_proof_encoding(&run(true));

            assert_eq!(
                unsafe_seminaive, naive,
                ":unsafe-seminaive and :naive proof encodings disagree for {file}"
            );
        }
    }

    /// A user rule marked `:naive` must stay `:naive` through proof encoding;
    /// dropping it would silently switch the rule to seminaive evaluation.
    #[test]
    fn proof_encoding_preserves_naive() {
        // The second case binds an eq-sort body var, whose `term_proof` RHS
        // read would otherwise force `:unsafe-seminaive`. Both must stay naive.
        let cases = [
            r#"(relation r (i64))
               (relation s (i64))
               (rule ((r x)) ((s x)) :naive :name "keep")"#,
            r#"(sort Math)
               (constructor Num (i64) Math)
               (constructor Neg (Math) Math)
               (relation seen (Math))
               (rule ((Neg m)) ((seen m)) :naive :name "keep")"#,
        ];
        for source in cases {
            let mut egraph = crate::EGraph::new_with_proofs();
            let resolved = egraph.resolve_program(None, source).unwrap();
            let rule = resolved
                .iter()
                .find_map(|c| match c {
                    ResolvedCommand::Rule { rule } if rule.name == "keep" => Some(rule),
                    _ => None,
                })
                .expect("instrumented rule not found");
            assert_eq!(
                rule.eval_mode,
                RuleEvalMode::Naive,
                "proof encoding did not preserve :naive for:\n{source}"
            );
        }
    }

    #[test]
    fn generated_proof_rhs_lookups_are_allowed() {
        let source = r#"
            (sort Math)
            (constructor Neg (Math) Math)
            (relation seen (Math))
            (rule ((= m (Neg n))) ((seen m)) :name "uses-proof-read")
        "#;

        let mut egraph = crate::EGraph::new_with_proofs();
        let resolved = egraph.resolve_program(None, source).unwrap();
        let rule = resolved
            .iter()
            .find_map(|command| match command {
                ResolvedCommand::Rule { rule } if rule.name == "uses-proof-read" => Some(rule),
                _ => None,
            })
            .expect("proof-instrumented rule not found");

        assert_eq!(rule.eval_mode, RuleEvalMode::UnsafeSeminaive);

        let mut found_internal_hidden_lookup = false;
        for action in rule.head.0.iter() {
            let GenericAction::Let(_, _, rhs) = action else {
                continue;
            };
            let GenericExpr::Call(_, ResolvedCall::Func(func_type), _) = rhs else {
                continue;
            };
            if func_type.subtype == FunctionSubtype::Custom && func_type.internal_hidden {
                found_internal_hidden_lookup = true;
                assert!(
                    egraph.type_info.expr_has_function_lookup(rhs).is_none(),
                    "internal-hidden proof lookup was treated as a user RHS function lookup"
                );
            }
        }

        assert!(
            found_internal_hidden_lookup,
            "expected proof instrumentation to emit an internal-hidden RHS lookup"
        );
    }

    #[test]
    fn user_rhs_function_lookup_still_rejected() {
        let source = r#"
            (function input (i64) i64 :merge old)
            (function output (i64) i64 :merge old)
            (relation trigger ())
            (rule ((trigger)) ((set (output 0) (input 0))))
        "#;

        let mut egraph = crate::EGraph::default();
        let err = egraph
            .resolve_program(None, source)
            .expect_err("user RHS function lookup should be rejected")
            .to_string();
        assert!(
            err.contains("Value lookup of non-constructor function function in rule is disallowed"),
            "unexpected error for user RHS function lookup: {err}"
        );
    }

    #[test]
    fn doc_example_add_function2() {
        let commands = term_encode(
            r#"
            (function add (i64 i64) i64 :merge old)
            (check (= (add 0 0) 0))
            "#,
        );

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        insta::assert_snapshot!("doc_example_add_function2", snapshot);
    }

    #[test]
    fn doc_example_add_function1() {
        let commands = term_encode(
            r#"
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
(check (= (Add 1 2) (Add 2 1)))
            "#,
        );

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        insta::assert_snapshot!("doc_example_add_function1", snapshot);
    }
}
