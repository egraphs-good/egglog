#[cfg(test)]
mod tests {
    use crate::CommandOutput;
    use crate::ast::{ResolvedCommand, RuleEvalMode, sanitize_internal_names};

    fn term_encode(source: &str) -> Vec<ResolvedCommand> {
        let mut egraph = crate::EGraph::new_with_term_encoding();
        egraph.resolve_program(None, source).unwrap()
    }

    fn proof_encode(source: &str) -> String {
        let mut egraph = crate::EGraph::new_with_proofs();
        let commands = egraph.resolve_program(None, source).unwrap();
        sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A trivial eq-sort-output custom merge (`:merge old`) is value-replacement
    /// and routes onto the FD pair-valued view (no legacy `handle_merge_fn`
    /// merge rule). The FD view declaration carries the `:identity-values`
    /// stamp; its `:merge` builds a `(pair ...)` value. This is what makes the
    /// `unreachable!` at the `handle_merge_or_congruence` else branch safe.
    #[test]
    fn trivial_eq_sort_output_merge_is_fd() {
        let encoding = proof_encode(
            r#"
            (datatype T (A) (B))
            (function keep (i64) T :merge old)
            (set (keep 0) (A))
            (set (keep 0) (B))
            (check (= (keep 0) (A)))
            "#,
        );
        // The FD view stamps `:identity-values 1` on the generated view function.
        assert!(
            encoding.contains(":identity-values"),
            "trivial eq-sort merge should produce an FD view (got: {encoding})"
        );
        // FD merges produce a `(pair ...)` value rather than the legacy
        // output-in-key `:merge old` view shape with a separate merge rule.
        assert!(
            encoding.contains("(pair"),
            "FD view merge should build a pair value (got: {encoding})"
        );
    }

    /// A `:merge` that READS a non-constructor function (e.g. `:merge (foo)`) is a
    /// live-DB read — merges are pure writes — so it is proof-UNSUPPORTED and must be
    /// rejected cleanly, never reaching the `handle_merge_or_congruence` `unreachable!`.
    /// (Before the gate covered merge bodies, this panicked at the unreachable.)
    #[test]
    fn function_reading_merge_is_proof_unsupported() {
        let source = r#"
            (function foo () i64 :merge (min old new))
            (function bar () i64 :merge (foo))
        "#;
        // File-level classification excludes it.
        let mut e = crate::EGraph::default();
        let resolved = e.resolve_program(None, source).unwrap();
        assert!(
            !crate::program_supports_proofs(&resolved, &e.type_info),
            "function-reading merge must be proof-unsupported"
        );
        // The real `--proofs` path returns a graceful error, NOT a panic.
        let mut pe = crate::EGraph::new_with_proofs();
        let err = pe.parse_and_run_program(None, source);
        assert!(
            matches!(err, Err(crate::Error::UnsupportedProofCommand { .. })),
            "expected UnsupportedProofCommand, got {err:?}"
        );
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
