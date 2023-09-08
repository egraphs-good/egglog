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
        Symbol::from(format!(
            "{}View{}",
            name,
            "_".repeat(self.desugar().number_underscores)
        ))
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
                .parse_program(&format!(
                    "(function {view_name} ({child_sorts}) {sort} 
                    :on_merge ({union_old_new})
                    :merge (ordering-min old new))"
                ))
                .unwrap()
        }
    }

    fn canonicalize_rules(&mut self, fdecl: &NormFunctionDecl) -> Vec<Command> {
        let types = self.type_info().lookup_user_func(fdecl.name).unwrap();
        let has_eq_type = types.output.is_eq_sort() || types.input.iter().any(|s| s.is_eq_sort());

        if !has_eq_type {
            return vec![];
        }

        let view_name = if types.is_datatype {
            self.view_name(fdecl.name)
        } else {
            fdecl.name
        };
        let child = |i| format!("c{i}_");
        let child_parent = |myself: &Self, i| {
            #[allow(clippy::iter_nth)]
            let child_t: ArcSort = types.input.iter().nth(i).unwrap().clone();
            myself
                .wrap_parent_or_rebuild(child(i), child_t)
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
                    .map(|v| child_parent(self, v))
                    .collect::<Vec<_>>(),
                " "
            )
        );
        let lhs_updated = self
            .wrap_parent_or_rebuild("lhs".to_string(), types.output.clone())
            .unwrap_or_else(|| "lhs".to_string());
        let fresh = self.fresh_var();

        let mut res = vec![];
        // This rule updates each row of the table,
        // doing canonicalization
        // (and through the view merge function, rebuilding)
        let rule = format!(
            "(rule ((= lhs ({view_name} {children}))
                           {children_updated}
                           (= {fresh} {lhs_updated}))
                       (
                        (replace ({view_name} {children})
                                 ({view_name} {children_updated})
                                 {lhs_updated})
                       )
                        :ruleset {})",
            self.rebuilding_ruleset_name()
        );
        res.extend(self.desugar().parse_program(&rule).unwrap());

        res
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
            if !self.type_info().is_global(var) {
                Expr::Var(var)
            } else if let Some(wrapped) = self.wrap_parent_or_rebuild(var.to_string(), vtype) {
                self.desugar().expr_parser.parse(&wrapped).unwrap()
            } else {
                Expr::Var(var)
            }
        })
    }

    fn wrap_parent_or_rebuild(&self, var: String, sort: ArcSort) -> Option<String> {
        if sort.is_container_sort() {
            Some(format!("(rebuild {})", var))
        } else {
            self.wrap_parent(var, sort)
        }
    }

    fn wrap_parent(&self, var: String, sort: ArcSort) -> Option<String> {
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

    fn fresh_var(&mut self) -> Symbol {
        self.egraph.desugar.get_fresh()
    }

    fn look_up_globals(&mut self, action: &NormAction, res: &mut Vec<Action>) -> NormAction {
        action.map_def_use(&mut |var, is_def| {
            let vtype = self.type_info().lookup(self.current_ctx, var).unwrap();

            if is_def || !self.type_info().is_global(var) {
                var
            } else if let Some(wrapped) = self.wrap_parent_or_rebuild(var.to_string(), vtype) {
                let fresh_var = self.fresh_var();
                res.extend(
                    self.desugar()
                        .action_parser
                        .parse(&format!("(let {fresh_var} {wrapped})")),
                );
                fresh_var
            } else {
                var
            }
        })
    }

    // Actions need to be instrumented to add to the view
    // as well as to the terms tables.
    // In addition, we need to look up canonical versions
    // of globals.
    fn instrument_action(&mut self, action: &NormAction) -> Vec<Action> {
        let mut res = vec![];

        // compute the func type before we mess with
        // the action
        let func_type = match action {
            NormAction::Delete(expr) => Some(
                self.type_info()
                    .lookup_expr(self.current_ctx, expr)
                    .unwrap(),
            ),
            NormAction::Let(_lhs, expr) => Some(
                self.type_info()
                    .lookup_expr(self.current_ctx, expr)
                    .unwrap(),
            ),
            _ => None,
        };
        let union_type = if let NormAction::Union(lhs, _rhs) = action {
            Some(self.type_info().lookup(self.current_ctx, *lhs).unwrap())
        } else {
            None
        };

        let globals_replaced = self.look_up_globals(action, &mut res);

        res.extend(match &globals_replaced {
            NormAction::Delete(NormExpr::Call(op, body)) => {
                // delete from view instead of from terms
                // TODO does this do the right thing? What if it misses because of canonicalization?
                let func_type = func_type.unwrap();
                if func_type.is_datatype {
                    let view_name = self.view_name(*op);
                    vec![Action::Delete(
                        view_name,
                        body.iter().cloned().map(Expr::Var).collect(),
                    )]
                } else {
                    vec![globals_replaced.to_action()]
                }
            }
            NormAction::Let(lhs, NormExpr::Call(op, body)) => {
                let func_type = func_type.unwrap();
                let lhs_type = func_type.output;
                let mut res = vec![globals_replaced.to_action()];

                // add the new term to the union-find
                if let Some(lhs_wrapped) = self.wrap_parent(lhs.to_string(), lhs_type.clone()) {
                    res.extend(self.parse_actions(vec![format!("(set {lhs_wrapped} {lhs})",)]))
                }

                let lhs_wrapped = self
                    .wrap_parent_or_rebuild(lhs.to_string(), lhs_type.clone())
                    .unwrap_or(lhs.to_string());

                let body_wrapped = body
                    .iter()
                    .zip(func_type.input.iter())
                    .map(|(v, vtype)| {
                        self.wrap_parent_or_rebuild(v.to_string(), vtype.clone())
                            .unwrap_or(v.to_string())
                    })
                    .collect::<Vec<_>>();

                // add the new term to the view
                if func_type.is_datatype {
                    let view_name = self.view_name(*op);
                    res.extend(self.parse_actions(vec![format!(
                        "(set ({view_name} {}) {lhs_wrapped})",
                        ListDisplay(body_wrapped, " ")
                    )]));
                }

                res
            }

            NormAction::Union(lhs, rhs) => {
                let lhs_type = union_type.unwrap();

                self.parse_actions(vec![self.union(
                    lhs_type.name(),
                    &lhs.to_string(),
                    &rhs.to_string(),
                )])
            }
            // Set doesn't touch terms, only tables
            _ => vec![globals_replaced.to_action()],
        });
        res
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
        Schedule::Sequence(vec![
            Schedule::Run(RunConfig {
                ruleset: self.parent_ruleset_name(),
                until: None,
            })
            .saturate(),
            Schedule::Run(RunConfig {
                ruleset: self.rebuilding_ruleset_name(),
                until: None,
            }),
        ])
        .saturate()
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
                    res.extend(self.canonicalize_rules(fdecl));
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
                | NCommand::CheckProof
                | NCommand::PrintOverallStatistics => {
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
