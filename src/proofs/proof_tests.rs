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

    /// A user-defined command must pass through term encoding (proofs off)
    /// instead of panicking. The command here just runs a sub-program that
    /// inserts a row, so we can observe that it actually executed.
    #[test]
    fn user_defined_command_passes_through_term_encoding() {
        use crate::ast::{Expr, GenericCommand};
        use crate::{CommandOutput, EGraph, Error, UserDefinedCommand};
        use std::sync::Arc;

        #[derive(Debug)]
        struct InsertOne;

        impl UserDefinedCommand for InsertOne {
            fn update(
                &self,
                egraph: &mut EGraph,
                _args: &[Expr],
            ) -> Result<Vec<CommandOutput>, Error> {
                egraph.parse_and_run_program(None, "(Add 1 2)")?;
                Ok(vec![])
            }
        }

        let mut egraph = EGraph::new_with_term_encoding();
        egraph
            .add_command("insert-one".to_string(), Arc::new(InsertOne))
            .unwrap();

        // This program contains a user-defined command. Before the fix this
        // panicked in term-encoding instrumentation.
        egraph
            .parse_and_run_program(
                None,
                r#"
(sort Math)
(constructor Add (i64 i64) Math)
(insert-one)
(check (Add 1 2))
            "#,
            )
            .unwrap();

        // Sanity check that resolve (the instrumentation path) also doesn't
        // panic and forwards the user-defined command.
        let resolved = egraph.resolve_program(None, "(insert-one)").unwrap();
        assert!(
            resolved
                .iter()
                .any(|c| matches!(c, GenericCommand::UserDefined(..))),
            "user-defined command should be forwarded by term encoding, got: {resolved:?}"
        );
    }
}
