use crate::*;

#[derive(Clone, Debug, Default)]
pub(crate) struct Names(HashMap<Symbol, Span>);

impl Names {
    pub(crate) fn check_shadowing(&mut self, program: &[ResolvedNCommand]) -> Result<(), Error> {
        for command in program {
            self.check_shadowing_command(command)?;
        }
        Ok(())
    }

    fn check(&mut self, name: Symbol, new: Span) -> Result<(), Error> {
        if let Some(old) = self.0.get(&name) {
            Err(Error::Shadowing(name, old.clone(), new))
        } else {
            self.0.insert(name, new);
            Ok(())
        }
    }

    fn check_shadowing_command(&mut self, command: &ResolvedNCommand) -> Result<(), Error> {
        match command {
            ResolvedNCommand::Sort(span, name, _args) => self.check(*name, span.clone()),
            ResolvedNCommand::Function(decl) => self.check(decl.name, decl.span.clone()),
            ResolvedNCommand::AddRuleset(span, name) => self.check(*name, span.clone()),
            ResolvedNCommand::UnstableCombinedRuleset(span, name, _args) => {
                self.check(*name, span.clone())
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
                inner.check_shadowing_command(command)
            }
            ResolvedNCommand::SetOption { .. } => Ok(()),
            ResolvedNCommand::RunSchedule(..) => Ok(()),
            ResolvedNCommand::PrintOverallStatistics => Ok(()),
            ResolvedNCommand::PrintTable(..) => Ok(()),
            ResolvedNCommand::PrintSize(..) => Ok(()),
            ResolvedNCommand::Input { .. } => Ok(()),
            ResolvedNCommand::Output { .. } => Ok(()),
            ResolvedNCommand::Push(..) => Ok(()),
            ResolvedNCommand::Pop(..) => Ok(()),
        }
    }

    fn check_shadowing_query(&mut self, query: &[ResolvedFact]) -> Result<(), Error> {
        // we want to allow names in queries to shadow each other, so we first collect
        // all of the variable names, and then we check each of those names once
        fn get_expr_names(expr: &ResolvedExpr, inner: &mut Names) {
            match expr {
                ResolvedExpr::Lit(..) => {}
                ResolvedExpr::Var(span, name) => {
                    if !inner.0.contains_key(&name.name) {
                        inner.0.insert(name.name, span.clone());
                    }
                }
                ResolvedExpr::Call(_span, _func, args) => {
                    args.iter().for_each(|e| get_expr_names(e, inner))
                }
            };
        }

        let mut inner = Names::default();

        for fact in query {
            match fact {
                ResolvedFact::Eq(_span, e1, e2) => {
                    get_expr_names(e1, &mut inner);
                    get_expr_names(e2, &mut inner);
                }
                ResolvedFact::Fact(e) => get_expr_names(e, &mut inner),
            }
        }

        for (name, span) in inner.0 {
            self.check(name, span.clone())?;
        }

        Ok(())
    }

    fn check_shadowing_action(&mut self, action: &ResolvedAction) -> Result<(), Error> {
        if let ResolvedAction::Let(span, name, _args) = action {
            self.check(name.name, span.clone())
        } else {
            Ok(())
        }
    }
}
