use crate::*;

pub(crate) struct TermState<'a> {
    pub(crate) current_ctx: CommandId,
    pub(crate) egraph: &'a mut EGraph,
}

impl<'a> TermState<'a> {
    pub(crate) fn add_term_encoding(
        egraph: &'a mut EGraph,
        program: Vec<NormCommand>,
    ) -> Vec<Command> {
        Self {
            // default id is overwritten when we call add_term_encoding
            current_ctx: CommandId::default(),
            egraph,
        }
        .add_term_encoding_helper(program)
    }

    fn desugar(&self) -> &Desugar {
        &self.egraph.desugar
    }

    fn type_info(&self) -> &TypeInfo {
        &self.egraph.desugar.type_info
    }

    pub(crate) fn parent_name(&self, sort: Symbol) -> Symbol {
        self.desugar().parent_name(sort)
    }

    pub(crate) fn init(&self, type_name: Symbol, expr: &str) -> String {
        let pname = self.parent_name(type_name);
        format!("(set ({pname} {expr}) {expr})",)
    }

    pub(crate) fn union(&self, type_name: Symbol, lhs: &str, rhs: &str) -> String {
        let pname = self.parent_name(type_name);
        format!(
            "(set ({pname}
                   (ordering-max {lhs} {rhs}))
                  (ordering-min {lhs} {rhs}))",
        )
    }

    fn make_parent_table(&self, name: Symbol) -> Vec<Command> {
        let pname = self.parent_name(name);
        let union_old_new = self.union(name, "old", "new");
        self.parse_program(&format!(
            "(function {pname} ({name}) {name} 
                        :on_merge ({union_old_new})
                        :merge (ordering-min old new)
                        )
            (rule ((= ({pname} a) b)
                   (= ({pname} b) c))
                  ((set ({pname} a) c))
                   :ruleset {})",
            self.parent_ruleset_name()
        ))
        .unwrap()
    }

    fn view_name(&self, name: Symbol) -> Symbol {
        Symbol::from(format!("{}View{}", name, self.desugar().number_underscores))
    }

    fn make_term_view(&self, fdecl: &NormFunctionDecl) -> Vec<Command> {
        let types = self.type_info().lookup_user_func(fdecl.name).unwrap();
        if !types.is_datatype {
            vec![]
        } else {
            let view_name = self.view_name(fdecl.name);
            let sort = types.output.name();
            let child_sorts = ListDisplay(types.input.iter().map(|s| s.name()), " ");
            let union_old_new = self.union(sort, "old", "new");
            self.desugar()
                .parser
                .parse(&format!(
                    "(function {view_name} ({child_sorts}) {sort} 
                    :on_merge ({union_old_new})
                    :merge (ordering-min old new))"
                ))
                .unwrap()
        }
    }

    fn make_canonicalize_func(&mut self, fdecl: &NormFunctionDecl) -> Vec<Command> {
        let types = self.type_info().lookup_user_func(fdecl.name).unwrap();

        let view_name = if types.is_datatype {
            self.view_name(fdecl.name)
        } else {
            fdecl.name
        };
        let child = |i| format!("c{i}_");
        let child_parent = |i| {
            #[allow(clippy::iter_nth)]
            let child_t: ArcSort = types.input.iter().nth(i).unwrap().clone();
            self.wrap_parent_or_rebuild(child(i), child_t)
                .unwrap_or_else(|| child(i))
        };
        let children = format!(
            "{}",
            ListDisplay(
                (0..fdecl.schema.input.len()).map(child).collect::<Vec<_>>(),
                " "
            )
        );
        let children_updated = format!(
            "{}",
            ListDisplay(
                (0..fdecl.schema.input.len())
                    .map(child_parent)
                    .collect::<Vec<_>>(),
                " "
            )
        );
        let lhs_updated = self
            .wrap_parent_or_rebuild("lhs".to_string(), types.output.clone())
            .unwrap_or_else(|| "lhs".to_string());
        let rule = format!(
            "(rule ((= lhs ({view_name} {children}))
                    {children_updated})
                   ((set ({view_name} {children_updated}) {lhs_updated}))
                    :ruleset {})",
            self.rebuilding_ruleset_name()
        );
        self.desugar().parser.parse(&rule).unwrap()
    }

    /// Instrument fact replaces terms with looking up
    /// canonical versions in the view.
    /// It also needs to look up references to globals.
    fn instrument_fact(&mut self, fact: &NormFact) -> Fact {
        match fact {
            NormFact::Assign(lhs, NormExpr::Call(head, body)) => {
                let func_type = self
                    .type_info()
                    .lookup_expr(self.current_ctx, &NormExpr::Call(*head, body.clone()))
                    .unwrap();
                if func_type.is_datatype {
                    let view_name = self.view_name(*head);
                    NormFact::Assign(*lhs, NormExpr::Call(view_name, body.clone()))
                } else {
                    fact.clone()
                }
            }
            _ => fact.clone(),
        }
        .map_use(&mut |var| {
            let vtype = self.type_info().lookup(self.current_ctx, var).unwrap();
            if self.type_info().is_global(var) && vtype.is_eq_sort() {
                let parent = self.parent_name(vtype.name());
                Expr::Call(parent, vec![Expr::Var(var)])
            } else {
                Expr::Var(var)
            }
        })
    }

    fn wrap_parent_or_rebuild(&mut self, var: String, sort: ArcSort) -> Option<String> {
        if sort.is_container_sort() {
            Some(format!("(rebuild {})", var))
        } else {
            self.wrap_parent(var, sort)
        }
    }

    fn wrap_parent(&mut self, var: String, sort: ArcSort) -> Option<String> {
        // TODO make all containers also eq sort
        if sort.is_eq_sort() {
            let parent = self.parent_name(sort.name());
            Some(format!("({parent} {var})"))
        } else {
            None
        }
    }

    fn instrument_facts(&mut self, facts: &[NormFact]) -> Vec<Fact> {
        facts.iter().map(|f| self.instrument_fact(f)).collect()
    }

    pub(crate) fn parse_actions(&self, actions: Vec<String>) -> Vec<Action> {
        actions
            .into_iter()
            .map(|s| self.desugar().action_parser.parse(&s).unwrap())
            .collect()
    }

    fn instrument_action(&mut self, action: &NormAction) -> Vec<Action> {
        match action {
            NormAction::Delete(_) => {
                // delete from view instead of from terms
                vec![action.to_action()]
            }
            NormAction::Let(lhs, NormExpr::Call(op, body)) => {
                let func_type = self
                    .type_info()
                    .lookup_expr(self.current_ctx, &NormExpr::Call(*op, body.clone()))
                    .unwrap();
                let lhs_type = func_type.output;
                let mut res = vec![action.to_action()];

                // add the new term to the union-find
                if let Some(lhs_wrapped) = self.wrap_parent(lhs.to_string(), lhs_type.clone()) {
                    res.extend(self.parse_actions(vec![format!("(set {lhs_wrapped} {lhs})",)]))
                }

                // add the new term to the view
                if func_type.is_datatype {
                    let view_name = self.view_name(*op);
                    res.extend(self.parse_actions(vec![format!(
                        "(set ({view_name} {}) {lhs})",
                        ListDisplay(body, " ")
                    )]));
                }

                res
            }
            // Set doesn't touch terms, only tables
            NormAction::Set(..) => {
                vec![action.to_action()]
            }
            NormAction::Union(lhs, rhs) => {
                let lhs_type = self.type_info().lookup(self.current_ctx, *lhs).unwrap();
                let rhs_type = self.type_info().lookup(self.current_ctx, *rhs).unwrap();
                assert_eq!(lhs_type.name(), rhs_type.name());
                assert!(lhs_type.is_eq_sort());

                self.parse_actions(vec![self.union(
                    lhs_type.name(),
                    &lhs.to_string(),
                    &rhs.to_string(),
                )])
            }
            _ => vec![action.to_action()],
        }
    }

    fn instrument_actions(&mut self, actions: &[NormAction]) -> Vec<Action> {
        actions
            .iter()
            .flat_map(|a| self.instrument_action(a))
            .collect()
    }

    fn instrument_rule(&mut self, ruleset: Symbol, name: Symbol, rule: &NormRule) -> Vec<Command> {
        let rule = Rule {
            head: self.instrument_actions(&rule.head),
            body: self.instrument_facts(&rule.body),
        };
        vec![Command::Rule {
            ruleset,
            name,
            rule,
        }]
    }

    fn rebuild(&self) -> Schedule {
        Schedule::Saturate(Box::new(Schedule::Sequence(vec![
            Schedule::Saturate(Box::new(Schedule::Run(RunConfig {
                ruleset: self.parent_ruleset_name(),
                until: None,
            }))),
            Schedule::Saturate(Box::new(Schedule::Run(RunConfig {
                ruleset: self.rebuilding_ruleset_name(),
                until: None,
            }))),
        ])))
    }

    fn instrument_schedule(&mut self, schedule: &NormSchedule) -> Schedule {
        schedule.map_run_commands(&mut |run_config| {
            Schedule::Sequence(vec![
                self.rebuild(),
                Schedule::Run(RunConfig {
                    ruleset: run_config.ruleset,
                    until: run_config
                        .until
                        .as_ref()
                        .map(|facts| self.instrument_facts(facts)),
                }),
            ])
        })
    }

    // TODO we need to also instrument merge actions and merge because they can add new terms that need representatives
    // the egraph is the initial egraph with only default sorts
    pub(crate) fn add_term_encoding_helper(&mut self, program: Vec<NormCommand>) -> Vec<Command> {
        let mut res = vec![];

        if !self.egraph.term_header_added {
            res.extend(self.term_header());
            self.egraph.term_header_added = true;
        }
        //eprintln!("program: {}", ListDisplay(&program, "\n"));

        for command in program {
            self.current_ctx = command.metadata.id;

            // run rebuilding before most commands
            if let NCommand::Function(..) | NCommand::NormRule { .. } | NCommand::Sort(..) =
                &command.command
            {
            } else {
                res.push(Command::RunSchedule(self.rebuild()));
            }

            match &command.command {
                NCommand::Sort(name, _presort_and_args) => {
                    res.push(command.to_command());
                    res.extend(self.make_parent_table(*name));
                }
                NCommand::Function(fdecl) => {
                    res.push(command.to_command());
                    res.extend(self.make_term_view(fdecl));
                    res.extend(self.make_canonicalize_func(fdecl));
                }
                NCommand::NormRule {
                    ruleset,
                    name,
                    rule,
                } => {
                    res.extend(self.instrument_rule(*ruleset, *name, rule));
                }
                NCommand::NormAction(action) => {
                    res.extend(
                        self.instrument_action(action)
                            .into_iter()
                            .map(Command::Action),
                    );
                }
                NCommand::Check(facts) => {
                    res.push(Command::Check(self.instrument_facts(facts)));
                }
                NCommand::RunSchedule(schedule) => {
                    res.push(Command::RunSchedule(self.instrument_schedule(schedule)));
                }
                NCommand::Fail(cmd) => {
                    let mut with_term_encoding = self.add_term_encoding_helper(vec![NormCommand {
                        command: *cmd.clone(),
                        metadata: command.metadata.clone(),
                    }]);
                    let last = with_term_encoding.pop().unwrap();
                    res.extend(with_term_encoding);
                    res.push(Command::Fail(Box::new(last)));
                }
                NCommand::SetOption { .. }
                | NCommand::Pop(..)
                | NCommand::Push(..)
                | NCommand::AddRuleset(..)
                | NCommand::PrintSize(..)
                | NCommand::PrintTable(..)
                | NCommand::Output { .. }
                | NCommand::Input { .. }
                | NCommand::CheckProof => {
                    res.push(command.to_command());
                }
            }
        }

        res
    }

    fn parent_ruleset_name(&self) -> Symbol {
        Symbol::from(format!(
            "parent{}",
            "_".repeat(self.desugar().number_underscores)
        ))
    }

    fn rebuilding_ruleset_name(&self) -> Symbol {
        Symbol::from(format!(
            "rebuilding{}",
            "_".repeat(self.desugar().number_underscores)
        ))
    }

    pub(crate) fn term_header(&self) -> Vec<Command> {
        let str = format!(
            "(ruleset {})
                         (ruleset {})",
            self.parent_ruleset_name(),
            self.rebuilding_ruleset_name()
        );
        self.parse_program(&str).unwrap()
    }

    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        self.desugar().parse_program(input)
    }
}
