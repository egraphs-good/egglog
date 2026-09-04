#[cfg(test)]
mod tests {
    use crate::ast::{ResolvedCommand, RuleEvalMode, sanitize_internal_names};
    use crate::{
        CommandOutput, EGraph, Error, ProofEncodingUnsupportedReason, TermDag, TermId,
        add_primitive_with_validator,
    };

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
    fn proof_mode_allows_eq_sort_primitive_results_in_facts() {
        let mut egraph = EGraph::default();
        let validator =
            |_: &mut TermDag, args: &[TermId]| -> Option<TermId> { args.first().copied() };
        add_primitive_with_validator!(
            &mut egraph,
            "proof-id" = |x: #| -> # { x },
            validator
        );
        let mut egraph = egraph.with_proofs_enabled();

        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype Math
                  (Done)
                  (Num i64))
                (relation Seed (Math))

                (Seed (Num 1))

                (rule ((Seed y)
                       (= x (proof-id y)))
                      ((Done))
                      :name "use-proof-id")

                (run 1)
                (prove (Done))
                "#,
            )
            .unwrap();
    }

    #[test]
    fn proof_support_rejects_naive_eq_sort_primitive_results_in_facts() {
        let mut egraph = EGraph::default();
        let validator =
            |_: &mut TermDag, args: &[TermId]| -> Option<TermId> { args.first().copied() };
        add_primitive_with_validator!(
            &mut egraph,
            "proof-id" = |x: #| -> # { x },
            validator
        );
        let mut egraph = egraph.with_proofs_enabled();

        let err = egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype Math
                  (Done)
                  (Num i64))
                (relation Seed (Math))

                (rule ((Seed y)
                       (= x (proof-id y)))
                      ((Done))
                      :naive
                      :name "naive-use-proof-id")
                "#,
            )
            .unwrap_err();

        assert!(
            matches!(
                err,
                Error::UnsupportedProofCommand {
                    reason: ProofEncodingUnsupportedReason::NaiveEqSortPrimitiveFact,
                    ..
                }
            ),
            "expected NaiveEqSortPrimitiveFact, got {err:?}"
        );
    }

    #[test]
    fn proof_mode_allows_eq_container_primitive_results_in_facts() {
        // A real (presort-declared) eq-container sort, so the term/proof
        // encoding builds its rebuild primitive. A custom identity primitive
        // returns an existing eq-container value, exercising the
        // eq-container-primitive-result-in-a-fact path under proofs.
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype E (Mk))
                (sort EqContainer (Vec E))
                "#,
            )
            .unwrap();

        let eq_container_sort = egraph
            .type_info
            .get_sort_by_name("EqContainer")
            .expect("EqContainer sort")
            .clone();
        let validator =
            |_: &mut TermDag, args: &[TermId]| -> Option<TermId> { args.first().copied() };
        add_primitive_with_validator!(
            &mut egraph,
            "proof-container-id" = |x: # (eq_container_sort)| -> # (eq_container_sort) { x },
            validator
        );
        let validator =
            |_: &mut TermDag, args: &[TermId]| -> Option<TermId> { args.first().copied() };
        let original_typechecking = egraph.proof_state.original_typechecking.as_mut().unwrap();
        add_primitive_with_validator!(
            &mut **original_typechecking,
            "proof-container-id" = |x: # (eq_container_sort)| -> # (eq_container_sort) { x },
            validator
        );

        egraph
            .parse_and_run_program(
                None,
                r#"
                (relation SeedContainer (EqContainer))
                (relation Done ())

                (SeedContainer (vec-of (Mk)))

                (rule ((SeedContainer ys)
                       (= xs (proof-container-id ys)))
                      ((Done))
                      :name "use-proof-container-id")

                (run 1)
                (prove (Done))
                "#,
            )
            .unwrap();
    }

    // A container constructed in the query body and not used in an action: the
    // binding fact's proof is the container's reflexive `Eval`, which the rule
    // check re-derives with the typed primitive.
    #[test]
    fn proof_mode_query_constructed_container_not_used_in_action() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype E (Mk))
                (sort EqContainer (Vec E))
                (relation SeedElem (E))
                (relation Done ())

                (SeedElem (Mk))

                (rule ((SeedElem e)
                       (= xs (vec-of e)))
                      ((Done))
                      :name "new-container-in-body")

                (run 1)
                (prove (Done))
                "#,
            )
            .unwrap();
    }

    #[test]
    fn no_merge_conflict_matches_normal_mode() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (function f (i64) i64 :no-merge)
                    (set (f 0) 1)
                    "#,
                )
                .unwrap_or_else(|error| panic!("{mode} setup failed: {error}"));

            let error = egraph
                .parse_and_run_program(None, "(set (f 0) 2)")
                .expect_err("a conflicting no-merge set must fail");
            assert_eq!(
                error.to_string(),
                "Panic: Illegal merge attempted for function f",
                "{mode} mode exposed a different conflict surface"
            );
        }
    }

    #[test]
    fn no_merge_repeated_value_succeeds_in_all_modes() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (function f (i64) i64 :no-merge)
                    (set (f 0) 1)
                    (set (f 0) 1)
                    (check (= (f 0) 1))
                    "#,
                )
                .unwrap_or_else(|error| panic!("{mode} mode rejected an idempotent set: {error}"));
        }
    }

    /// `:no-merge` compares canonical values, not the constructor spelling
    /// used to reach them. Once two outputs are unioned, writing either member
    /// of that e-class is an idempotent update rather than a conflict.
    #[test]
    fn no_merge_equal_eclass_outputs_succeed_in_all_modes() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (datatype E (A) (B))
                    (function f (i64) E :no-merge)
                    (set (f 0) (A))
                    (union (A) (B))
                    (set (f 0) (B))
                    (check (= (f 0) (A)))
                    "#,
                )
                .unwrap_or_else(|error| {
                    panic!("{mode} mode rejected equal e-class outputs: {error}")
                });
        }
    }

    /// Equality containers are rebuilt from their elements' canonical
    /// representatives. Two structurally equal rebuilt outputs must therefore
    /// satisfy the same no-merge slot just as two unioned scalar outputs do.
    #[test]
    fn no_merge_equal_container_outputs_succeed_in_all_modes() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (datatype E (A) (B))
                    (sort Es (Vec E))
                    (function f (i64) Es :no-merge)
                    (set (f 0) (vec-of (A)))
                    (union (A) (B))
                    (set (f 0) (vec-of (B)))
                    (check (= (f 0) (vec-of (A))))
                    "#,
                )
                .unwrap_or_else(|error| {
                    panic!("{mode} mode rejected equal container outputs: {error}")
                });
            if mode == "proof" {
                egraph
                    .parse_and_run_program(None, "(prove (= (f 0) (vec-of (B))))")
                    .expect("canonical container output proof must pass the checker");
            }
        }
    }

    /// Container canonicalization in an authority guard is a database read.
    /// Rule instrumentation must select a safe evaluation mode and still emit
    /// a checker-valid proof for the original, uninstrumented premise.
    #[test]
    fn no_merge_container_authority_is_safe_in_rule_queries() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype E (A) (B))
                (sort Es (Vec E))
                (function f (i64) Es :no-merge)
                (relation Observed ())
                (set (f 0) (vec-of (A)))
                (union (A) (B))
                (rule ((= (f 0) value))
                      ((Observed))
                      :name "observe-no-merge-container")
                (run 1)
                (prove (Observed))
                "#,
            )
            .expect("container authority queries in rules must remain proof-checkable");
    }

    #[test]
    fn no_merge_distinct_eclass_outputs_still_conflict() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype E (A) (B))
                (function f (i64) E :no-merge)
                (set (f 0) (A))
                "#,
            )
            .unwrap();
        let error = egraph
            .parse_and_run_program(None, "(set (f 0) (B))")
            .expect_err("distinct e-classes must conflict");
        assert_eq!(
            error.to_string(),
            "Panic: Illegal merge attempted for function f"
        );
    }

    #[test]
    fn no_merge_zero_arity_function_preserves_first_value() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (function answer () i64 :no-merge)
                (set (answer) 42)
                (set (answer) 42)
                (prove (= (answer) 42))
                "#,
            )
            .expect("zero-arity idempotent updates must succeed");
        let error = egraph
            .parse_and_run_program(None, "(set (answer) 7)")
            .expect_err("zero-arity conflict must fail");
        assert_eq!(
            error.to_string(),
            "Panic: Illegal merge attempted for function answer"
        );
    }

    /// Merge expressions run through action instrumentation. A custom-function
    /// lookup there is independently unsupported and must be rejected by the
    /// capability gate, even when the looked-up function itself is no-merge.
    #[test]
    fn proof_support_rejects_function_lookup_in_merge_expression() {
        let mut egraph = EGraph::new_with_proofs();
        let error = egraph
            .parse_and_run_program(
                None,
                r#"
                (function source () i64 :no-merge)
                (function sink () i64 :merge (source))
                "#,
            )
            .expect_err("merge-expression function lookup must fail at the support gate");
        assert!(
            matches!(
                error,
                Error::UnsupportedProofCommand {
                    reason: ProofEncodingUnsupportedReason::FunctionLookupInAction,
                    ..
                }
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Function keys over equality sorts are canonicalized as the e-graph is
    /// rebuilt. An idempotent write through the new representative must find
    /// the original key instead of creating an independent authority row.
    #[test]
    fn no_merge_keys_follow_eclass_rebuild_without_false_conflict() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (datatype E (A) (B))
                    (function f (E) i64 :no-merge)
                    (set (f (A)) 1)
                    (union (A) (B))
                    (set (f (B)) 1)
                    (check (= (f (A)) 1))
                    "#,
                )
                .unwrap_or_else(|error| panic!("{mode} mode lost a rebuilt no-merge key: {error}"));
        }
    }

    #[test]
    fn no_merge_conflict_does_not_leave_a_provable_row() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (function f (i64) i64 :no-merge)
                (set (f 0) 1)
                "#,
            )
            .unwrap();
        egraph
            .parse_and_run_program(None, "(set (f 0) 2)")
            .expect_err("conflicting set must fail");

        let ghost = egraph
            .parse_and_run_program(None, "(check (= (f 0) 2))")
            .expect_err("the rejected row must not be queryable");
        assert!(
            matches!(ghost, Error::CheckError(..)),
            "unexpected error: {ghost}"
        );

        let outputs = egraph
            .parse_and_run_program(None, "(prove (= (f 0) 1))")
            .expect("the accepted row must remain provable after the conflict");
        assert!(
            outputs
                .iter()
                .any(|output| matches!(output, CommandOutput::ProveExists { .. })),
            "prove command emitted no checked proof"
        );
    }

    /// An expected conflict is still a complete command boundary after term
    /// instrumentation expands one surface `set` into several internal writes
    /// plus rebuilding. The rejected candidate must not escape that boundary.
    #[test]
    fn no_merge_expected_conflict_preserves_the_accepted_row() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(
                    None,
                    r#"
                    (function f (i64) i64 :no-merge)
                    (set (f 0) 1)
                    (fail (set (f 0) 2))
                    (check (= (f 0) 1))
                    "#,
                )
                .unwrap_or_else(|error| {
                    panic!("{mode} mode did not contain the expected conflict: {error}")
                });

            let sizes = egraph
                .parse_and_run_program(None, "(print-size f)")
                .unwrap_or_else(|error| panic!("{mode} mode could not size f: {error}"));
            assert!(
                sizes
                    .iter()
                    .any(|output| matches!(output, CommandOutput::PrintFunctionSize(1))),
                "{mode} mode counted a rejected no-merge row: {sizes:?}"
            );

            if mode == "proof" {
                egraph
                    .parse_and_run_program(None, "(prove (= (f 0) 1))")
                    .expect("the accepted row proof must remain checker-valid");
            }
        }
    }

    /// Rebuild-deferred no-merge handling must not move an immediate failure
    /// out of its `fail` boundary. This protects the generic command contract
    /// while the no-merge path receives its additional completion rebuild.
    #[test]
    fn expected_panic_remains_wrapped_in_all_modes() {
        for (mode, mut egraph) in [
            ("normal", EGraph::default()),
            ("term", EGraph::new_with_term_encoding()),
            ("proof", EGraph::new_with_proofs()),
        ] {
            egraph
                .parse_and_run_program(None, r#"(fail (panic "expected"))"#)
                .unwrap_or_else(|error| panic!("{mode} mode leaked the expected panic: {error}"));
        }
    }

    /// Rule actions execute in one backend batch. A conflict must not make the
    /// rejected row visible even if later actions in that batch have already run.
    #[test]
    fn no_merge_rule_conflict_does_not_leave_a_provable_row() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (function f (i64) i64 :no-merge)
                (relation Trigger (i64))
                (set (f 0) 1)
                (Trigger 2)
                (rule ((Trigger value))
                      ((set (f 0) value))
                      :name "write-conflicting-value")
                "#,
            )
            .unwrap();
        let error = egraph
            .parse_and_run_program(None, "(run 1)")
            .expect_err("rule-produced no-merge conflict must fail");
        assert_eq!(
            error.to_string(),
            "Panic: Illegal merge attempted for function f"
        );

        let ghost = egraph
            .parse_and_run_program(None, "(check (= (f 0) 2))")
            .expect_err("the rejected rule row must not be queryable");
        assert!(
            matches!(ghost, Error::CheckError(..)),
            "unexpected error: {ghost}"
        );
        egraph
            .parse_and_run_program(None, "(prove (= (f 0) 1))")
            .expect("checker must still accept the pre-conflict row proof");
    }

    #[test]
    fn no_merge_rule_repeating_the_same_value_succeeds() {
        let mut egraph = EGraph::new_with_proofs();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (function f (i64) i64 :no-merge)
                (relation Trigger (i64))
                (set (f 0) 1)
                (Trigger 1)
                (rule ((Trigger value))
                      ((set (f 0) value))
                      :name "repeat-same-value")
                (run 1)
                (prove (= (f 0) 1))
                "#,
            )
            .expect("same-value rule update must remain idempotent");
    }

    // A container constructed in the query is a side condition with no carryable
    // proof (just an `Eval` marker), so it can't be used in an action. Proof mode
    // rejects such a rule rather than producing an unsound proof.
    #[test]
    fn proof_support_rejects_query_constructed_container_used_in_action() {
        let mut egraph = EGraph::new_with_proofs();
        let err = egraph
            .parse_and_run_program(
                None,
                r#"
                (datatype E (Mk))
                (sort EqContainer (Vec E))
                (relation SeedElem (E))
                (relation Out (EqContainer))

                (rule ((SeedElem e)
                       (= xs (vec-of e)))
                      ((Out xs))
                      :name "new-container-in-action")
                "#,
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::UnsupportedProofCommand {
                    reason: ProofEncodingUnsupportedReason::ContainerCreatedInQueryUsedInAction,
                    ..
                }
            ),
            "expected ContainerCreatedInQueryUsedInAction, got {err:?}"
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
