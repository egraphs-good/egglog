#[cfg(test)]
mod tests {
    use crate::ast::{ResolvedCommand, sanitize_internal_names};

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
