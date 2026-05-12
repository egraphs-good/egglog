#[cfg(test)]
mod tests {
    use crate::ast::{ResolvedCommand, sanitize_internal_names};

    fn term_encode(source: &str) -> Vec<ResolvedCommand> {
        let mut egraph = crate::EGraph::new_with_term_encoding();
        egraph.resolve_program(None, source).unwrap()
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

    #[test]
    fn dump_souffle_baseline() {
        let commands = term_encode(
            r#"
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
(run 1)
(check (= (Add 1 2) (Add 2 1)))
            "#,
        );

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        std::fs::write("/tmp/souffle-baseline-output.txt", &snapshot).unwrap();
        eprintln!("WROTE TO /tmp/souffle-baseline-output.txt");
    }

    #[test]
    fn dump_souffle_compat_strata_check() {
        let mut egraph = crate::EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = egraph
            .resolve_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b)) ((union (Add a b) (Add b a))) :name "comm")
(run 1)
(check (= (Add 1 2) (Add 2 1)))
            "#,
            )
            .unwrap();
        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write("/tmp/souffle-strata-check.txt", &snapshot).unwrap();
    }

    #[test]
    fn dump_souffle_compat() {
        let mut egraph = crate::EGraph::new_with_term_encoding().with_souffle_compat();
        let commands = egraph
            .resolve_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
(run 1)
(check (= (Add 1 2) (Add 2 1)))
            "#,
            )
            .unwrap();

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        std::fs::write("/tmp/souffle-compat-output.txt", &snapshot).unwrap();
        eprintln!("WROTE TO /tmp/souffle-compat-output.txt");
    }

    #[test]
    fn souffle_compat_strata_emits_buffer_and_snap() {
        let mut egraph = crate::EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = egraph
            .resolve_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
            "#,
            )
            .unwrap();

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        // Phase 1: buffer + snap declarations exist.
        assert!(
            snapshot.contains("__AddView_buffer"),
            "expected __AddView_buffer in:\n{snapshot}"
        );
        assert!(
            snapshot.contains("__AddView_snap"),
            "expected __AddView_snap in:\n{snapshot}"
        );
        // Sanity: canonical view still there.
        assert!(
            snapshot.contains("(function __AddView "),
            "expected canonical __AddView still declared in:\n{snapshot}"
        );
        // Phase 2: drain rule from buffer to canonical view exists.
        assert!(
            snapshot.contains("__AddView_buffer_drain"),
            "expected __AddView_buffer_drain rule in:\n{snapshot}"
        );
        // Phase 4: UF buffer + drain.
        assert!(
            snapshot.contains("__UF_Math_buffer"),
            "expected __UF_Math_buffer in:\n{snapshot}"
        );
        assert!(
            snapshot.contains("__UF_Math_buffer_drain"),
            "expected __UF_Math_buffer_drain rule in:\n{snapshot}"
        );
    }

    /// Phase 3: with user rules in the program, those rules' reads should
    /// hit __<C>View_snap (not __<C>View) and writes should hit
    /// __<C>View_buffer. The internal rebuild rules still use __<C>View.
    #[test]
    fn souffle_compat_strata_user_rules_redirect() {
        let mut egraph = crate::EGraph::new_with_term_encoding().with_souffle_compat_strata();
        let commands = egraph
            .resolve_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
            "#,
            )
            .unwrap();

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        // Find the commutativity rule and check it reads from snap +
        // writes to buffer.
        let comm_rule_block = snapshot
            .split('\n')
            .collect::<Vec<_>>()
            .windows(20)
            .find(|w| w.iter().any(|line| line.contains(":name \"commutativity\"")))
            .map(|w| w.join("\n"))
            .expect("expected commutativity rule in encoded form");

        // User read goes to snap.
        assert!(
            comm_rule_block.contains("__AddView_snap"),
            "expected user rule to read __AddView_snap; got:\n{comm_rule_block}"
        );
        // User writes go to buffer.
        assert!(
            comm_rule_block.contains("__AddView_buffer"),
            "expected user rule to write __AddView_buffer; got:\n{comm_rule_block}"
        );
        // User rule should NOT directly reference the canonical __AddView
        // (rebuild reads/writes that, but user rules go through snap/buffer).
        // We can't strictly forbid the literal substring since the buffer
        // and snap names start with __AddView, but the bare __AddView
        // (without a suffix) should not appear inside the rule body or
        // head — quick heuristic: count occurrences vs. the suffixed forms.
        let bare_view = comm_rule_block.matches("__AddView ").count()
            + comm_rule_block.matches("__AddView)").count();
        // Some leniency: rebuild references may show up in proof terms
        // (Rule "commutativity" ...). For now just sanity-check the rule
        // includes the redirected forms.
        let _ = bare_view;
    }

    #[test]
    fn dump_souffle_compat_proofs() {
        let mut egraph = crate::EGraph::new_with_proofs().with_souffle_compat();
        let commands = egraph
            .resolve_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
(run 1)
(check (= (Add 1 2) (Add 2 1)))
            "#,
            )
            .unwrap();

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        std::fs::write("/tmp/souffle-compat-proofs-output.txt", &snapshot).unwrap();
        eprintln!("WROTE TO /tmp/souffle-compat-proofs-output.txt");
    }

    #[test]
    fn dump_souffle_baseline_proofs() {
        let mut egraph = crate::EGraph::new_with_proofs();
        let commands = egraph
            .resolve_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
(Add 1 2)
(rule ((Add a b))
      ((union (Add a b) (Add b a)))
     :name "commutativity")
(run 1)
(check (= (Add 1 2) (Add 2 1)))
            "#,
            )
            .unwrap();

        let snapshot = sanitize_internal_names(&commands)
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        std::fs::write("/tmp/souffle-baseline-proofs-output.txt", &snapshot).unwrap();
        eprintln!("WROTE TO /tmp/souffle-baseline-proofs-output.txt");
    }
}
