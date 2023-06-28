use crate::*;

impl ProofState {
    pub fn parent_name(&self, sort: Symbol) -> Symbol {
        Symbol::from(format!(
            "{}_Parent{}",
            sort,
            "_".repeat(self.desugar.number_underscores)
        ))
    }

    pub(crate) fn union(&self, type_name: Symbol, lhs: &str, rhs: &str) -> String {
        let pname = self.parent_name(type_name);
        format!(
            "(set ({pname} ({pname} {lhs})) ({pname} {rhs}))
             (set ({pname} ({pname} {rhs})) ({pname} {lhs}))",
        )
    }

    fn make_rebuilding(&self, name: Symbol) -> Vec<Command> {
        let pname = self.parent_name(name);
        vec![
            format!(
                "(function {pname} ({name}) {name} :unextractable :merge (ordering-less old new))"
            ),
            format!(
                "(rule ((= ({pname} a) b)
                        (= ({pname} b) c))
                       ((set ({pname} a) c))
                            :ruleset {})",
                self.parent_ruleset_name()
            ),
        ]
        .into_iter()
        .flat_map(|s| self.desugar.parser.parse(&s).unwrap())
        .collect()
    }

    fn make_canonicalize_func(&mut self, fdecl: &FunctionDecl) -> Vec<Command> {
        let types = self.type_info.func_types.get(&fdecl.name).unwrap().clone();
        if !types.output.is_eq_sort() {
            return vec![];
        }
        let op = fdecl.name;
        let pname = self.parent_name(fdecl.schema.output);
        let child = |i| format!("c{i}_");
        let child_parent = |i| {
            #[allow(clippy::iter_nth)]
            let child_t: ArcSort = types.input.iter().nth(i).unwrap().clone();
            self.wrap_parent(child(i), child_t)
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
        vec![format!(
            "(rule ((= lhs ({op} {children}))
                    {children_updated})
                   ((let rhs ({op} {children_updated}))
                    (set ({pname} rhs) rhs)
                    {})
                    :ruleset {})",
            self.union(fdecl.schema.output, "lhs", "rhs"),
            self.rebuilding_ruleset_name()
        )]
        .into_iter()
        .flat_map(|s| self.desugar.parser.parse(&s).unwrap())
        .collect()
    }

    fn instrument_fact(&mut self, fact: &NormFact) -> Vec<Fact> {
        match fact {
            NormFact::ConstrainEq(lhs, rhs) => {
                let lhs_t = self.type_info.lookup(self.current_ctx, *lhs).unwrap();
                let rhs_t = self.type_info.lookup(self.current_ctx, *rhs).unwrap();
                assert!(lhs_t.name() == rhs_t.name());

                if let (Some(lhs_wrapped), Some(rhs_wrapped)) = (
                    self.wrap_parent(lhs.to_string(), lhs_t),
                    self.wrap_parent(rhs.to_string(), rhs_t),
                ) {
                    vec![
                        format!("(= {} {})", lhs_wrapped, rhs_wrapped),
                        format!("(= {} {})", rhs_wrapped, lhs_wrapped),
                    ]
                    .into_iter()
                    .map(|s| self.desugar.fact_parser.parse(&s).unwrap())
                    .collect::<Vec<Fact>>()
                } else {
                    vec![fact.to_fact()]
                }
            }
            NormFact::Compute(lhs, expr) => {
                let NormExpr::Call(head, children) = expr;
                let children_parents = children
                    .iter()
                    .map(|child| {
                        self.wrap_parent(
                            child.to_string(),
                            self.type_info.lookup(self.current_ctx, *child).unwrap(),
                        )
                        .unwrap_or_else(|| child.to_string())
                    })
                    .collect::<Vec<String>>();

                self.parse_facts(vec![format!(
                    "(= {lhs} ({head} {}))",
                    ListDisplay(children_parents, " ")
                )])
            }
            NormFact::Assign(_lhs, NormExpr::Call(_head, body)) => {
                vec![fact.to_fact()]
                    .into_iter()
                    .chain({
                        let facts = body
                            .iter()
                            .filter_map(|child| {
                                let child_t =
                                    self.type_info.lookup(self.current_ctx, *child).unwrap();
                                self.wrap_parent(child.to_string(), child_t)
                                    .map(|child_wrapped| format!("(= {} {})", child_wrapped, child))
                            })
                            .collect();
                        // optimization: only match on
                        // canonical enodes
                        self.parse_facts(facts)
                    })
                    .collect()
            }
            NormFact::AssignLit(..) => {
                vec![fact.to_fact()]
            }
        }
    }

    fn wrap_parent(&mut self, var: String, sort: ArcSort) -> Option<String> {
        if sort.is_eq_sort() {
            let parent = self.parent_name(sort.name());
            Some(format!("({parent} {var})"))
        } else {
            None
        }
    }

    fn instrument_facts(&mut self, facts: &[NormFact]) -> Vec<Fact> {
        facts.iter().flat_map(|f| self.instrument_fact(f)).collect()
    }

    fn parse_facts(&self, facts: Vec<String>) -> Vec<Fact> {
        facts
            .into_iter()
            .map(|s| self.desugar.fact_parser.parse(&s).unwrap())
            .collect()
    }

    pub fn parse_actions(&self, actions: Vec<String>) -> Vec<Action> {
        actions
            .into_iter()
            .map(|s| self.desugar.action_parser.parse(&s).unwrap())
            .collect()
    }

    fn instrument_action(&mut self, action: &NormAction) -> Vec<Action> {
        match action {
            NormAction::Delete(_) => {
                // TODO what to do about delete?
                vec![action.to_action()]
            }
            NormAction::Let(lhs, _expr) => {
                let lhs_type = self.type_info.lookup(self.current_ctx, *lhs).unwrap();
                if let Some(lhs_wrapped) = self.wrap_parent(lhs.to_string(), lhs_type) {
                    vec![action.to_action()]
                        .into_iter()
                        .chain(self.parse_actions(vec![format!("(set {lhs_wrapped} {lhs})",)]))
                        .collect()
                } else {
                    vec![action.to_action()]
                }
            }
            NormAction::LetLit(..)
            | NormAction::LetIteration(..)
            | NormAction::LetVar(..)
            | NormAction::Panic(..) => vec![action.to_action()],

            // handled by merge action
            NormAction::Set(_expr, _rhs) => vec![action.to_action()],
            NormAction::Union(lhs, rhs) => {
                let lhs_type = self.type_info.lookup(self.current_ctx, *lhs).unwrap();
                let rhs_type = self.type_info.lookup(self.current_ctx, *rhs).unwrap();
                assert_eq!(lhs_type.name(), rhs_type.name());
                assert!(lhs_type.is_eq_sort());

                self.parse_actions(
                    self.union(lhs_type.name(), &lhs.to_string(), &rhs.to_string())
                        .split('\n')
                        .map(|s| s.to_string())
                        .collect(),
                )
            }
        }
    }

    fn instrument_actions(&mut self, actions: &[NormAction]) -> Vec<Action> {
        actions
            .iter()
            .flat_map(|a| self.instrument_action(a))
            .collect()
    }

    fn instrument_rule(&mut self, ruleset: Symbol, name: Symbol, rule: &NormRule) -> Vec<Command> {
        vec![Command::Rule {
            ruleset,
            name,
            rule: Rule {
                head: self.instrument_actions(&rule.head),
                body: self.instrument_facts(&rule.body),
            },
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

    fn instrument_fdecl(&mut self, fdecl: &FunctionDecl) -> FunctionDecl {
        let mut res = fdecl.clone();
        let types = self.type_info.func_types.get(&fdecl.name).unwrap().clone();

        if res.merge.is_none() {
            res.merge = Some(Expr::Var("new".into()));
        }

        if types.output.is_eq_sort() && !types.has_merge {
            res.merge_action.extend(
                self.parse_actions(
                    self.union(types.output.name(), "old", "new")
                        .split('\n')
                        .map(|s| s.to_string())
                        .collect(),
                ),
            );

            res
        } else {
            res
        }
    }

    // TODO we need to also instrument merge actions and merge because they can add new terms that need representatives
    // the egraph is the initial egraph with only default sorts
    pub(crate) fn add_term_encoding(&mut self, program: Vec<NormCommand>) -> Vec<Command> {
        let mut res = vec![];

        if !self.term_header_added {
            res.extend(self.term_header());
            self.term_header_added = true;
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
                NCommand::Push(_num) => {
                    res.push(command.to_command());
                }
                NCommand::Sort(name, _presort_and_args) => {
                    res.push(command.to_command());
                    res.extend(self.make_rebuilding(*name));
                }
                NCommand::Function(fdecl) => {
                    res.push(Command::Function(self.instrument_fdecl(fdecl)));
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
                NCommand::Extract { variants, var } => {
                    //res.extend(self.desugar_extract(variants, expr));
                    res.push(command.to_command());
                }
                _ => {
                    res.push(command.to_command());
                }
            }
        }

        res
    }

    fn desugar_extract(&mut self, variants: usize, expr: Expr) -> Vec<Command> {
        // TODO handle variants
        let mut res = vec![Command::Push(1)];
        let lhs = self.get_fresh();
        res.push(Command::Action(Action::Let(lhs, expr)));

        /*for (sort, _) in self.type_info.sorts {
          res.push(format!("(function {} ({sort}) {sort}"
        }*/

        res.push(Command::Pop(1));
        res
    }

    fn best_of_name(&self, sort: Symbol) -> String {
        format!(
            "{}_BestOf{}",
            sort,
            "_".repeat(self.desugar.number_underscores)
        )
    }

    fn get_fresh(&mut self) -> Symbol {
        self.desugar.get_fresh()
    }

    fn parent_ruleset_name(&self) -> Symbol {
        Symbol::from(format!(
            "parent{}",
            "_".repeat(self.desugar.number_underscores)
        ))
    }

    fn rebuilding_ruleset_name(&self) -> Symbol {
        Symbol::from(format!(
            "rebuilding{}",
            "_".repeat(self.desugar.number_underscores)
        ))
    }

    pub(crate) fn term_header(&self) -> Vec<Command> {
        let str = format!(
            "(ruleset {})
                         (ruleset {})",
            self.parent_ruleset_name(),
            self.rebuilding_ruleset_name()
        );
        self.parse_program(&str).unwrap().into_iter().collect()
    }

    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        self.desugar.parse_program(input)
    }
}
