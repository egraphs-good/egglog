#[cfg(test)]
mod tests {
    use crate::ast::{Command, sanitize_internal_names};
    use crate::proof_encoding::ProofInstrumentor;

    use egglog::ast::Parser;
    use egglog::ast::desugar::desugar_command;
    use egglog::ast::proof_global_remover;

    fn term_encode(source: &str) -> Vec<Command> {
        let mut egraph = crate::EGraph::new_with_term_encoding();
        let mut parser = Parser::default();
        let program = parser
            .get_program_from_string(None, source)
            .expect("failed to parse program");
        let mut ncommands = Vec::new();
        for command in program {
            let desugared =
                desugar_command(command, &mut parser).expect("failed to desugar command");
            ncommands.extend(desugared);
        }

        let mut resolved = egraph
            .typecheck_program(&ncommands)
            .expect("failed to typecheck program");
        resolved = proof_global_remover::remove_globals(resolved, &mut parser.symbol_gen);
        ProofInstrumentor::add_term_encoding(&mut egraph, resolved)
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
            (union (Add 1 2) (Add 2 1))
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
