use dyn_clone::DynClone;
use std::mem::take;

use crate::ast::ResolvedVar;
use crate::core::ResolvedCall;
use crate::prelude::BaseSort;
use crate::sort::UnitSort;
use crate::{ast::GenericNCommand, Error};
use crate::{ExtractReport, Function, RunReport, Term, TermDag};

/**
 *
 */
pub trait OutputHandler: DynClone + Send + Sync {
    /**
     * Called on each command after it has been resolved.
     *
     * Returns a boolean indicating whether the command should be run.
     */
    fn handle_resolved_command(
        &mut self,
        _command: &GenericNCommand<ResolvedCall, ResolvedVar>,
    ) -> Result<bool, Error> {
        Ok(true)
    }

    fn handle_command_result(&mut self, _result: Result<(), Error>) -> Result<(), Error> {
        Ok(())
    }

    fn handle_print_function_size(&mut self, _name: &String, _size: usize) -> Result<(), Error> {
        Ok(())
    }

    fn handle_print_all_functions_size(
        &mut self,
        _names_and_sizes: Vec<(&String, usize)>,
    ) -> Result<(), Error> {
        Ok(())
    }

    /**
     * Returns the output of the command.
     */
    fn return_output(&mut self) -> Result<Vec<String>, Error> {
        Ok(vec![])
    }

    fn remove_globals(&self) -> bool {
        true
    }

    fn set_interactive_mode(&mut self, _interactive: bool) {}
    fn handle_extract_best(
        &mut self,
        _termdag: TermDag,
        _cost: u64,
        _term: Term,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn handle_extract_variants(
        &mut self,
        _termdag: TermDag,
        _terms: Vec<Term>,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn handle_overall_statistics(&mut self, _run_report: &RunReport) -> Result<(), Error> {
        Ok(())
    }
    fn handle_print_function(
        &mut self,
        _function: &Function,
        _termdag: TermDag,
        _terms_and_outputs: Vec<(Term, Term)>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn handle_run_schedule(&mut self, _report: RunReport) -> Result<(), Error> {
        Ok(())
    }
}

dyn_clone::clone_trait_object!(OutputHandler);

#[derive(Default, Clone)]
pub struct NoOpOutputHandler;

impl OutputHandler for NoOpOutputHandler {}

pub trait NormalOutputHandler: OutputHandler {
    /**
     * Handles one line of output.
     */
    fn handle_output_string(&mut self, output: String) -> Result<(), Error>;

    fn get_interactive_mode(&self) -> bool;

    fn handle_command_result(&mut self, result: Result<(), Error>) -> Result<(), Error> {
        if self.get_interactive_mode() {
            self.handle_output_string(match result {
                Ok(()) => "(done)".into(),
                Err(_) => "(error)".into(),
            })?;
        }
        Ok(())
    }

    fn handle_print_function_size(&mut self, _name: &String, size: usize) -> Result<(), Error> {
        self.handle_output_string(size.to_string())
    }
    fn handle_print_all_functions_size(
        &mut self,
        names_and_sizes: Vec<(&String, usize)>,
    ) -> Result<(), Error> {
        self.handle_output_string(
            names_and_sizes
                .into_iter()
                .map(|(name, len)| format!("{}: {}", name, len))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    fn handle_extract_best(
        &mut self,
        termdag: TermDag,
        _cost: u64,
        term: Term,
    ) -> Result<(), Error> {
        let extracted = termdag.to_string(&term);
        self.handle_output_string(extracted)
    }
    fn handle_extract_variants(&mut self, termdag: TermDag, terms: Vec<Term>) -> Result<(), Error> {
        self.handle_output_string("(".to_string())?;
        for expr in &terms {
            let str = termdag.to_string(expr);
            self.handle_output_string(format!("   {str}"))?;
        }
        self.handle_output_string(")".to_string())
    }

    fn handle_overall_statistics(&mut self, run_report: &RunReport) -> Result<(), Error> {
        self.handle_output_string(format!("Overall statistics:\n{}", run_report))
    }
    fn handle_print_function(
        &mut self,
        function: &Function,
        termdag: TermDag,
        terms_and_outputs: Vec<(Term, Term)>,
    ) -> Result<(), Error> {
        self.handle_output_string(print_function_string(function, termdag, terms_and_outputs))
    }
}

pub(crate) fn print_function_string(
    function: &Function,
    termdag: TermDag,
    terms_and_outputs: Vec<(Term, Term)>,
) -> String {
    let out_is_unit = function.schema.output.name() == UnitSort.name();
    let mut buf = "(\n".to_string();
    let s = &mut buf;
    for (term, output) in terms_and_outputs {
        let tuple_str = format!(
            "   {}{}",
            termdag.to_string(&term),
            if !out_is_unit {
                format!(" -> {}", termdag.to_string(&output))
            } else {
                "".into()
            },
        );
        log::info!("{}", tuple_str);
        s.push_str(&tuple_str);
    }
    s.push_str("\n)\n");
    buf
}

#[derive(Default, Clone)]
pub struct PrintlnOutputHandler {
    interactive: bool,
}
impl OutputHandler for PrintlnOutputHandler {
    fn set_interactive_mode(&mut self, interactive: bool) {
        self.interactive = interactive;
    }
}
impl NormalOutputHandler for PrintlnOutputHandler {
    fn handle_output_string(&mut self, output: String) -> Result<(), Error> {
        println!("{output}");
        Ok(())
    }

    fn get_interactive_mode(&self) -> bool {
        self.interactive
    }
}

#[derive(Default, Clone)]
pub struct CollectedOutputHandler {
    interactive: bool,
    outputs: Vec<String>,
}

impl OutputHandler for CollectedOutputHandler {
    fn return_output(&mut self) -> Result<Vec<String>, Error> {
        Ok(take(&mut self.outputs))
    }
    fn set_interactive_mode(&mut self, interactive: bool) {
        self.interactive = interactive;
    }
}

impl NormalOutputHandler for CollectedOutputHandler {
    fn handle_output_string(&mut self, output: String) -> Result<(), Error> {
        self.outputs.push(output);
        Ok(())
    }
    fn get_interactive_mode(&self) -> bool {
        self.interactive
    }
}

#[derive(Default, Clone)]
pub struct DesugarOutputHandler {
    commands: Vec<String>,
}

impl OutputHandler for DesugarOutputHandler {
    fn handle_resolved_command(
        &mut self,
        command: &GenericNCommand<ResolvedCall, ResolvedVar>,
    ) -> Result<bool, Error> {
        self.commands.push(command.to_command().to_string());
        // In show_egglog mode, we still need to run scope-related commands (Push/Pop) to make
        // the program well-scoped.
        match command {
            GenericNCommand::Push(..) | GenericNCommand::Pop(..) => Ok(true),
            _ => Ok(false),
        }
    }
    fn return_output(&mut self) -> Result<Vec<String>, Error> {
        Ok(take(&mut self.commands))
    }

    fn remove_globals(&self) -> bool {
        false
    }
}
pub struct TestingOutputHandler {
    pub last_extract_report: Option<ExtractReport>,
}

impl OutputHandler for TestingOutputHandler {
    fn handle_extract_best(
        &mut self,
        termdag: TermDag,
        _cost: u64,
        term: Term,
    ) -> Result<(), Error> {
    }
    fn handle_extract_variants(
        &mut self,
        _termdag: TermDag,
        _terms: Vec<Term>,
    ) -> Result<(), Error> {
        Ok(())
    }
}
