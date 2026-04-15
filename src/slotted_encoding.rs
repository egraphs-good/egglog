use crate::{
    EGraph, ResolvedFact, ast::{Command, GenericNCommand, ResolvedNCommand}
};

/// Thin wrapper around an [`EGraph`] for the slotted encoding
pub(crate) struct SlottedInstrumentor<'a> {
    pub(crate) egraph: &'a mut EGraph,
}

impl<'a> SlottedInstrumentor<'a> {
    pub(crate) fn add_slotted_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_slotted_encoding_helper(program)
    }

    fn add_slotted_encoding_helper(&mut self, program: Vec<ResolvedNCommand>) -> Vec<Command> {
        let mut res = vec![];

        for r in program {
            res.push(self.add_slotted_encoding_one(r));
        }

        res
    }

    // Adds an intermediate Rename around every sub-expression
    fn slottify_query(&mut self, generic_fact: &ResolvedFact) {
      match generic_fact
    }

    fn add_slotted_encoding_one(&mut self, command: ResolvedNCommand) -> Command {
        match command {
            // TODO just doing "user rules" we don't have a header
            GenericNCommand::NormRule { rule } if rule.name.contains("user") => {
              todo!()
            }
            _ => command.to_command().make_unresolved(),
        }
    }
}
