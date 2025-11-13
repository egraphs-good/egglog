use crate::*;

#[derive(Clone, Debug, Default)]
pub(crate) struct Names(HashMap<String, Span>);

impl Names {
    fn check(&mut self, name: String, new: Span) -> Result<(), Error> {
        if let Some(old) = self.0.get(&name) {
            Err(Error::Shadowing(name, old.clone(), new))
        } else {
            self.0.insert(name, new);
            Ok(())
        }
    }

    /// WARNING: this function does not handle `push` and `pop`.
    /// Because `Names` is contained on the `EGraph`, this will
    /// work correctly when executed from `process_command`, but
    /// a unit test that called this function multiple times without
    /// changing the `EGraph` will be wrong.
    pub(crate) fn check_shadowing(&mut self, command: &ResolvedNCommand) -> Result<(), Error> {
        match command {
            ResolvedNCommand::Sort(span, name, _args) => self.check(name.clone(), span.clone()),
            ResolvedNCommand::Function(decl) => self.check(decl.name.clone(), decl.span.clone()),
            ResolvedNCommand::AddRuleset(span, name) => self.check(name.clone(), span.clone()),
            ResolvedNCommand::UnstableCombinedRuleset(span, name, _args) => {
                self.check(name.clone(), span.clone())
            }
            ResolvedNCommand::NormRule { rule, .. } => {
                let mut inner = self.clone();
                inner.check_shadowing_query(&rule.body)?;
                for action in rule.head.iter() {
                    inner.check_shadowing_action(action)?;
                }
                Ok(())
            }
            ResolvedNCommand::CoreAction(action) => self.check_shadowing_action(action),
            ResolvedNCommand::Check(_span, query) => {
                let mut inner = self.clone();
                inner.check_shadowing_query(query)
            }
            ResolvedNCommand::Fail(_span, command) => {
                let mut inner = self.clone();
                inner.check_shadowing(command)
            }
            ResolvedNCommand::Extract(..) => Ok(()),
            ResolvedNCommand::RunSchedule(..) => Ok(()),
            ResolvedNCommand::PrintOverallStatistics(..) => Ok(()),
            ResolvedNCommand::PrintFunction(..) => Ok(()),
            ResolvedNCommand::PrintSize(..) => Ok(()),
            ResolvedNCommand::Input { .. } => Ok(()),
            ResolvedNCommand::Output { .. } => Ok(()),
            ResolvedNCommand::Push(..) => Ok(()),
            ResolvedNCommand::Pop(..) => Ok(()),
            ResolvedNCommand::UserDefined(..) => Ok(()),
        }
    }

    fn check_shadowing_query(&mut self, query: &[ResolvedFact]) -> Result<(), Error> {
        // we want to allow names in queries to shadow each other, so we first collect
        // all of the variable names, and then we check each of those names once
        fn get_expr_names(expr: &ResolvedExpr, inner: &mut Names) {
            match expr {
                ResolvedExpr::Lit { .. } => {}
                ResolvedExpr::Var { span, name } => {
                    if !inner.0.contains_key(&name.name) {
                        inner.0.insert(name.name.clone(), span.clone());
                    }
                }
                ResolvedExpr::Call {
                    field1: _span,
                    field2: _func,
                    field3: args,
                } => args.iter().for_each(|e| get_expr_names(e, inner)),
            };
        }

        let mut inner = Names::default();

        for fact in query {
            match fact {
                ResolvedFact::Eq {
                    field1: _span,
                    field2: e1,
                    field3: e2,
                } => {
                    get_expr_names(e1, &mut inner);
                    get_expr_names(e2, &mut inner);
                }
                ResolvedFact::Fact { field1: e } => get_expr_names(e, &mut inner),
            }
        }

        for (name, span) in inner.0 {
            self.check(name, span.clone())?;
        }

        Ok(())
    }

    fn check_shadowing_action(&mut self, action: &ResolvedAction) -> Result<(), Error> {
        if let ResolvedAction::Let {
            field1: span,
            field2: name,
            field3: _args,
        } = action
        {
            self.check(name.name.clone(), span.clone())
        } else {
            Ok(())
        }
    }
}
