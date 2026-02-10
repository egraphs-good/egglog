use crate::{util::HashMap, *};

#[derive(Clone, Debug, Default)]
pub(crate) struct Names {
    seen: HashMap<String, Span>,
    global_aliases: HashMap<String, (String, Span)>,
}

impl Names {
    fn check(&mut self, name: String, new: Span) -> Result<(), Error> {
        if let Some(old) = self.seen.get(&name) {
            Err(Error::Shadowing(name, old.clone(), new))
        } else {
            self.seen.insert(name, new);
            Ok(())
        }
    }

    fn track_global_alias(&mut self, name: &str, span: &Span) {
        if let Some(stripped) = name.strip_prefix(GLOBAL_NAME_PREFIX) {
            self.global_aliases
                .insert(stripped.to_owned(), (name.to_owned(), span.clone()));
        }
    }

    fn check_pattern_name(&mut self, name: &str, span: &Span) -> Result<(), Error> {
        let canonical = name
            .strip_prefix(GLOBAL_NAME_PREFIX)
            .unwrap_or(name)
            .to_owned();
        if let Some((global_name, global_span)) = self.global_aliases.get(&canonical) {
            return Err(Error::Shadowing(
                format!(
                    "pattern variable `{}` conflicts with global `{}`",
                    name, global_name
                ),
                global_span.clone(),
                span.clone(),
            ));
        }
        self.check(name.to_owned(), span.clone())
    }

    /// WARNING: this function does not handle `push` and `pop`.
    /// Because `Names` is contained on the `EGraph`, this will
    /// work correctly when executed from `process_command`, but
    /// a unit test that called this function multiple times without
    /// changing the `EGraph` will be wrong.
    pub(crate) fn check_shadowing(&mut self, command: &ResolvedNCommand) -> Result<(), Error> {
        match command {
            ResolvedNCommand::Sort { span, name, .. } => self.check(name.clone(), span.clone()),
            ResolvedNCommand::Function(decl) => {
                self.check(decl.name.clone(), decl.span.clone())?;
                if decl.let_binding {
                    self.track_global_alias(&decl.name, &decl.span);
                }
                Ok(())
            }
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
            ResolvedNCommand::ProveExists(..) => Ok(()),
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
        fn collect_expr_names(expr: &ResolvedExpr, out: &mut HashMap<String, Span>) {
            match expr {
                ResolvedExpr::Lit(..) => {}
                ResolvedExpr::Var(span, name) => {
                    out.entry(name.name.clone()).or_insert_with(|| span.clone());
                }
                ResolvedExpr::Call(_span, _func, args) => {
                    args.iter().for_each(|e| collect_expr_names(e, out));
                }
            }
        }

        let mut collected = HashMap::default();

        for fact in query {
            match fact {
                ResolvedFact::Eq(_span, e1, e2) => {
                    collect_expr_names(e1, &mut collected);
                    collect_expr_names(e2, &mut collected);
                }
                ResolvedFact::Fact(e) => collect_expr_names(e, &mut collected),
            }
        }

        for (name, span) in collected {
            self.check_pattern_name(&name, &span)?;
        }

        Ok(())
    }

    fn check_shadowing_action(&mut self, action: &ResolvedAction) -> Result<(), Error> {
        if let ResolvedAction::Let(span, name, _args) = action {
            self.check_pattern_name(&name.name, span)
        } else {
            Ok(())
        }
    }
}
