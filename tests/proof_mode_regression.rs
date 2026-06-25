use egglog::ast::Command;
use egglog::util::SymbolGen;
use egglog::*;
use std::sync::{Arc, Mutex};

struct RecordFunctionInputArity {
    name: String,
    seen: Arc<Mutex<Vec<usize>>>,
}

impl CommandMacro for RecordFunctionInputArity {
    fn transform(
        &self,
        command: Command,
        _symbol_gen: &mut SymbolGen,
        type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        if let Some(func) = type_info.get_func_type(&self.name) {
            self.seen.lock().unwrap().push(func.input.len());
        }
        Ok(vec![command])
    }
}

#[test]
fn proof_mode_command_macros_see_original_function_arities() {
    let seen = Arc::new(Mutex::new(vec![]));
    let mut egraph = EGraph::new_with_proofs();
    egraph
        .command_macros_mut()
        .register(Arc::new(RecordFunctionInputArity {
            name: "score".to_string(),
            seen: seen.clone(),
        }));

    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype Math (Num i64))
            (function score (Math) i64 :merge old)
            (let x (Num 1))
            "#,
        )
        .unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![1]);
}
