use egglog::*;
use egglog::prelude::*;
use egglog::sort::I64Sort;
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_ast::generic_ast::Literal;

#[test]
fn test_add_primitive_validator() {
    // Test that we can add a validator to a primitive with the macro
    let mut egraph = EGraph::default();
    
    // Create a validator
    let validator = |termdag: &TermDag, result_term: TermId| -> Option<Literal> {
        // For testing, just return the result as-is if it's a literal
        match termdag.get(result_term) {
            Term::Lit(lit) => Some(lit.clone()),
            _ => None,
        }
    };
    
    // Use the macro to add a primitive with a validator
    add_primitive_with_validator!(
        &mut egraph,
        "test-add" = |a: i64, b: i64| -> i64 { a + b },
        validator
    );
    
    // Verify the primitive works
    egraph.parse_and_run_program(
        None,
        "(check (= (test-add 2 3) 5))"
    ).unwrap();
}

#[test]
fn test_add_primitive_with_validator_method() {
    // Test that we can add a validator to a primitive using the direct method
    let mut egraph = EGraph::default();
    
    // Create a simple primitive that adds two numbers
    #[derive(Clone)]
    struct TestAdd;
    impl Primitive for TestAdd {
        fn name(&self) -> &str {
            "test-add"
        }
        fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
            SimpleTypeConstraint::new(
                self.name(),
                vec![I64Sort.to_arcsort(), I64Sort.to_arcsort(), I64Sort.to_arcsort()],
                span.clone(),
            )
            .into_box()
        }
        fn apply(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
            let a = exec_state.base_values().unwrap::<i64>(args[0]);
            let b = exec_state.base_values().unwrap::<i64>(args[1]);
            Some(exec_state.base_values().get(a + b))
        }
    }
    
    // Create a validator
    let validator = std::sync::Arc::new(|termdag: &TermDag, result_term: TermId| -> Option<Literal> {
        // For testing, just return the result as-is if it's a literal
        match termdag.get(result_term) {
            Term::Lit(lit) => Some(lit.clone()),
            _ => None,
        }
    });
    
    // Add the primitive with validator using the direct method
    egraph.add_primitive_with_validator(TestAdd, Some(validator));
    
    // Verify the primitive works
    egraph.parse_and_run_program(
        None,
        "(check (= (test-add 2 3) 5))"
    ).unwrap();
}
