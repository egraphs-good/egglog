use crate::*;

#[derive(Default, Clone)]
pub(crate) struct ProofConstants {
    pub uf_parent: HashMap<String, String>,
    pub to_union: HashMap<String, String>,
    pub term_header_added: bool,
    pub term_mode_enabled: bool,
}

pub(crate) struct TermState<'a> {
    egraph: &'a mut EGraph,
}

impl<'a> TermState<'a> {
    /// Make a term state and use it to instrument the code.
    pub(crate) fn add_term_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_term_encoding_helper(program)
    }

    pub(crate) fn parent_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_constants.uf_parent.get(sort) {
            name.clone()
        } else {
            self.egraph.proof_constants.uf_parent.insert(
                sort.to_string(),
                String::from(format!("ufparent_{}", sort,)),
            );
            self.egraph.proof_constants.uf_parent[sort].clone()
        }
    }

    pub(crate) fn to_union_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_constants.to_union.get(sort) {
            name.clone()
        } else {
            self.egraph
                .proof_constants
                .to_union
                .insert(sort.to_string(), String::from(format!("ufunion_{}", sort)));
            self.egraph.proof_constants.to_union[sort].clone()
        }
    }

    fn to_union_ruleset_name(&self) -> String {
        String::from(format!("to_union",))
    }

    /// Mark two things for unioning.
    pub(crate) fn union(&mut self, type_name: &str, lhs: &str, rhs: &str) -> String {
        let to_union_name = self.to_union_name(type_name);
        format!("({} {} {})", to_union_name, lhs, rhs)
    }

    /// The parent table is the database representation of a union-find datastructure.
    /// When one term has two parents, those parents are unioned in the merge action.
    /// Also, we have a rule that maintains the invariant that each term points to its
    /// canonical representative.
    fn make_parent_table(&mut self, name: &str) -> Vec<Command> {
        let pname = self.parent_name(name);
        let to_union_name = self.to_union_name(name);
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("uf");
        self.parse_program(&format!(
            "(sort {fresh_sort})
             (constructor {pname} ({name} {name}) {fresh_sort})
             (rule (({pname} a b)
                    ({pname} b c)
                    (!= b c)
                    (!= a c))
                  ((delete ({pname} a b))
                   ({pname} a c))
                   :ruleset {})
                   
                   
             (rule (({to_union_name} a b)
                    ({pname} a pa)
                    ({pname} b pb)
                    (!= pa pb))
                   (({pname} (ordering-min pa pb) (ordering-max pa pb)))
                   :ruleset {})",
            self.parent_direct_ruleset_name(),
            self.to_union_ruleset_name(),
        ))
        .unwrap()
    }

    fn view_name(&self, name: &str) -> String {
        String::from(format!("{}View", name,))
    }

    /// Each function/constructor gets a term table and a view table.
    /// The term table stores underlying representative terms.
    /// The view table stores child terms and their eclass.
    /// The view table is mutated using delete, but we never delete from term tables.
    /// We re-use the original name of the function for the term table.
    ///
    /// TODO need a rule to handle merge functions
    fn term_and_view(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let schema = &fdecl.schema;
        let mut types = schema.input.clone();
        if fdecl.subtype == FunctionSubtype::Custom {
            types.push(schema.output.clone());
        }

        let view_name = self.view_name(&fdecl.name);
        let child_sorts = ListDisplay(types, " ");
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("view");
        // the term table has child_sorts as inputs
        // the view table has child_sorts + the leader term for the eclass
        self.parse_program(&format!(
            "
            (constructor {} ({child_sorts}) {})

            (sort {fresh_sort})
            (constructor {view_name} ({child_sorts} {}) {fresh_sort})",
            fdecl.name,
            schema.output,
            if fdecl.subtype == FunctionSubtype::Constructor {
                fdecl.schema.output.clone()
            } else {
                "".to_string()
            }
        ))
        .unwrap()
    }

    fn rebuilding_rules(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let mut type_strs = fdecl.schema.input.clone();
        if fdecl.subtype == FunctionSubtype::Custom {
            type_strs.push(fdecl.schema.output.clone());
        }
        let types = fdecl.resolved_schema.term_types();
        let types2 = types.clone();

        let view_name = self.view_name(&fdecl.name);
        let child = |i| format!("c{i}_");
        let child_parent = move |myself: &mut Self, i| {
            // TODO weird type error when I don't use iter here
            let child_t: ArcSort = types2.iter().nth(i).unwrap().clone();
            myself
                .get_canonical_expr_of(child(i), child_t)
                .unwrap_or_else(|| child(i))
        };
        let children = format!(
            "{}",
            ListDisplay((0..types.len()).map(child).collect::<Vec<_>>(), " ")
        );
        let children_updated = format!(
            "{}",
            ListDisplay(
                (0..types.len())
                    .map(|v| child_parent(self, v))
                    .collect::<Vec<_>>(),
                " "
            )
        );

        let mut res = vec![];
        // Make a rule that updates the view
        let rule = format!(
            "(rule ((= lhs ({view_name} {children})))
                     (
                      ({view_name} {children_updated})
                     )
                      :ruleset {})",
            self.rebuilding_ruleset_name()
        );

        // And a rule that cleans up the view
        let rule2 = format!(
            "(rule ((!= ({view_name} {children})
                        ({view_name} {children_updated})))
                     (
                      (delete ({view_name} {children}))
                     )
                      :ruleset {})",
            self.rebuilding_ruleset_name()
        );

        res
    }

    /// Instrument fact replaces terms with looking up
    /// canonical versions in the view.
    /// It also needs to look up references to globals.
    fn instrument_fact(&mut self, fact: &ResolvedFact, res: &mut Vec<String>) {
        match fact {
            GenericFact::Eq(_span, generic_expr, generic_expr1) => {
                let v1 = self.instrument_fact_expr(generic_expr, res);
                let v2 = self.instrument_fact_expr(generic_expr1, res);
                res.push(format!("(= {} {})", v1, v2));
            }
            GenericFact::Fact(generic_expr) => {
                let v = self.instrument_fact_expr(generic_expr, res);
            }
        }
    }

    fn instrument_fact_expr(&mut self, expr: &ResolvedExpr, res: &mut Vec<String>) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{}", lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_fact_expr(arg, res))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        let fv = self.fresh_var();
                        res.push(format!(
                            "({} {} {fv})",
                            self.view_name(&func_type.name),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(let {} ({} {}))",
                            fv,
                            specialized_primitive.name(),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                }
            }
        }
    }

    fn get_canonical_expr_of(&mut self, var: String, sort: ArcSort) -> Option<String> {
        if sort.is_eq_container_sort() {
            // todo containers need a custom rebuild
            Some(format!("(rebuild {})", var))
        } else {
            self.wrap_parent(var, sort)
        }
    }

    fn wrap_parent(&mut self, var: String, sort: ArcSort) -> Option<String> {
        if sort.is_eq_sort() {
            let parent = self.parent_name(&sort.name());
            Some(format!("({parent} {var})"))
        } else {
            None
        }
    }

    fn instrument_facts(&mut self, facts: &[ResolvedFact]) -> Vec<String> {
        let mut res = vec![];
        for fact in facts {
            self.instrument_fact(fact, &mut res);
        }
        res
    }

    fn fresh_var(&mut self) -> String {
        self.egraph.parser.symbol_gen.fresh("v")
    }

    // Actions need to be instrumented to add to the view
    // as well as to the terms tables.
    fn instrument_action(&mut self, action: &ResolvedAction) -> Vec<String> {
        let mut res = vec![];

        match action {
            ResolvedAction::Let(_span, v, generic_expr) => {
                let v2 = self.instrument_action_expr(generic_expr, &mut res);
                res.push(format!("(let {} {})", v.name, v2));
            }
            ResolvedAction::Set(_span, h, generic_exprs, generic_expr) => {
                let mut exprs = vec![];
                for e in generic_exprs
                    .into_iter()
                    .chain(std::iter::once(generic_expr))
                {
                    exprs.push(self.instrument_action_expr(e, &mut res));
                }
                res.push(format!("({} {})", h.name(), ListDisplay(exprs, " ")));
            }
            ResolvedAction::Change(_span, change, h, generic_exprs) => {
                todo!()
            }
            ResolvedAction::Union(_span, generic_expr, generic_expr1) => {
                let v1 = self.instrument_action_expr(generic_expr, &mut res);
                let v2 = self.instrument_action_expr(generic_expr1, &mut res);
                let ot = generic_expr.output_type();
                let type_name = ot.name();
                let unioned = self.union(type_name, &v1, &v2);
                res.push(unioned);
            }
            ResolvedAction::Panic(_span, msg) => {
                res.push(format!("(panic {})", msg));
            }
            ResolvedAction::Expr(_span, generic_expr) => {
                let v = self.instrument_action_expr(generic_expr, &mut res);
            }
        }

        res
    }

    // Add to view and term tables, returning a variable for the created term.
    fn instrument_action_expr(&mut self, expr: &ResolvedExpr, res: &mut Vec<String>) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{}", lit),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_action_expr(arg, res))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        if func_type.subtype == FunctionSubtype::Custom {
                            panic!(
                                "Found a function lookup in actions, should have been prevented by typechecking"
                            );
                        }

                        let fv = self.fresh_var();
                        // add to term table
                        res.push(format!(
                            "(let fv ({} {}))",
                            func_type.name,
                            ListDisplay(args.clone(), " ")
                        ));
                        // add to view table
                        res.push(format!(
                            "({} {} {fv})",
                            self.view_name(&func_type.name),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(let {} ({} {}))",
                            fv,
                            specialized_primitive.name(),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                }
            }
        }
    }

    fn instrument_actions(&mut self, actions: &[ResolvedAction]) -> Vec<String> {
        let mut res = vec![];
        for action in actions {
            let mut instrumented = self.instrument_action(action);
            res.append(&mut instrumented);
        }
        res
    }

    fn instrument_rule(&mut self, rule: &ResolvedRule) -> Vec<Command> {
        let facts = self.instrument_facts(&rule.body);
        let actions = self.instrument_actions(&rule.head.0);
        self.parse_program(&format!(
            "(rule ({} )
                   ({} )
                    :ruleset {})",
            ListDisplay(facts, " "),
            ListDisplay(actions, " "),
            rule.ruleset,
        ))
        .unwrap()
    }

    /// TODO experiment with schedule- unclear what is fastest.
    /// Any schedules should be sound.
    fn rebuild(&self) -> Schedule {
        Schedule::Saturate(
            span!(),
            Box::new(Schedule::Sequence(
                span!(),
                vec![
                    Schedule::Saturate(
                        span!(),
                        Box::new(Schedule::Run(
                            span!(),
                            RunConfig {
                                ruleset: self.parent_direct_ruleset_name(),
                                until: None,
                            },
                        )),
                    ),
                    Schedule::Run(
                        span!(),
                        RunConfig {
                            ruleset: self.to_union_ruleset_name(),
                            until: None,
                        },
                    ),
                    Schedule::Run(
                        span!(),
                        RunConfig {
                            ruleset: self.rebuilding_ruleset_name(),
                            until: None,
                        },
                    ),
                ],
            )),
        )
    }

    // TODO schedules contain queries we need to instrument
    fn instrument_schedule(&mut self, schedule: &ResolvedSchedule) -> Schedule {
        match schedule {
            ResolvedSchedule::Run(span, config) => {
                let new_run = match config.until {
                    Some(ref facts) => {
                        let instrumented = self.instrument_facts(facts);
                        let instrumented_facts = self.parse_facts(&instrumented);
                        Schedule::Run(
                            span.clone(),
                            RunConfig {
                                ruleset: config.ruleset.clone(),
                                until: Some(instrumented_facts),
                            },
                        )
                    }
                    None => Schedule::Run(
                        span.clone(),
                        RunConfig {
                            ruleset: config.ruleset.clone(),
                            until: None,
                        },
                    ),
                };
                Schedule::Sequence(span.clone(), vec![new_run, self.rebuild()])
            }
            ResolvedSchedule::Sequence(span, schedules) => Schedule::Sequence(
                span.clone(),
                schedules
                    .iter()
                    .map(|s| self.instrument_schedule(s))
                    .collect(),
            ),
            ResolvedSchedule::Saturate(span, schedule) => {
                Schedule::Saturate(span.clone(), Box::new(self.instrument_schedule(schedule)))
            }
            GenericSchedule::Repeat(span, n, schedule) => Schedule::Repeat(
                span.clone(),
                *n,
                Box::new(self.instrument_schedule(schedule)),
            ),
        }
    }

    pub(crate) fn add_term_encoding_helper(
        &mut self,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        let mut res = vec![];

        if !self.egraph.proof_constants.term_header_added {
            res.extend(self.term_header());
            self.egraph.proof_constants.term_header_added = true;
        }

        for command in program {
            match command {
                ResolvedNCommand::Sort(_, name, _presort_and_args) => {
                    res.push(command.to_command().make_unresolved());
                    res.extend(self.make_parent_table(&name));
                }
                ResolvedNCommand::Function(fdecl) => {
                    res.extend(self.term_and_view(&fdecl));
                    res.extend(self.rebuilding_rules(&fdecl));
                }
                ResolvedNCommand::NormRule { rule } => {
                    res.extend(self.instrument_rule(&rule));
                }
                ResolvedNCommand::CoreAction(action) => {
                    res.extend(
                        self.parse_program(&self.instrument_action(&action).join("\n"))
                            .unwrap(),
                    );
                }
                ResolvedNCommand::Check(span, facts) => {
                    res.push(Command::Check(
                        span,
                        self.parse_facts(&self.instrument_facts(&facts)),
                    ));
                }
                ResolvedNCommand::RunSchedule(schedule) => {
                    res.push(Command::RunSchedule(self.instrument_schedule(&schedule)));
                }
                ResolvedNCommand::Fail(span, cmd) => {
                    let mut with_term_encoding = self.add_term_encoding_helper(vec![*cmd]);
                    let last = with_term_encoding.pop().unwrap();
                    res.extend(with_term_encoding);
                    res.push(Command::Fail(span, Box::new(last)));
                }
                ResolvedNCommand::Pop(..)
                | ResolvedNCommand::Push(..)
                | ResolvedNCommand::AddRuleset(..)
                | ResolvedNCommand::PrintSize(..)
                | ResolvedNCommand::Output { .. }
                | ResolvedNCommand::Input { .. }
                | ResolvedNCommand::UnstableCombinedRuleset(..)
                | ResolvedNCommand::PrintOverallStatistics(..)
                | ResolvedNCommand::PrintFunction(..) => {
                    res.push(command.to_command().make_unresolved());
                }
                ResolvedNCommand::Extract(..) => {
                    panic!("Extract commands unsupported in term encoding");
                }
                ResolvedNCommand::UserDefined(..) => {
                    panic!("User defined commands unsupported in term encoding");
                }
            }

            // run rebuilding after every command except a few
            if let ResolvedNCommand::Function(..)
            | ResolvedNCommand::NormRule { .. }
            | ResolvedNCommand::Sort(..) = &command
            {
            } else {
                res.push(Command::RunSchedule(self.rebuild()));
            }
        }

        res
    }

    fn parent_direct_ruleset_name(&self) -> String {
        String::from(format!("parent",))
    }

    fn rebuilding_ruleset_name(&self) -> String {
        String::from(format!("rebuilding",))
    }

    pub(crate) fn term_header(&mut self) -> Vec<Command> {
        let str = format!(
            "(ruleset {})
             (ruleset {})
             (ruleset {})",
            self.parent_direct_ruleset_name(),
            self.to_union_ruleset_name(),
            self.rebuilding_ruleset_name()
        );
        self.parse_program(&str).unwrap()
    }

    fn parse_program(&mut self, input: &str) -> Result<Vec<Command>, ParseError> {
        self.egraph.parser.get_program_from_string(None, input)
    }

    fn parse_facts(&mut self, input: &Vec<String>) -> Vec<Fact> {
        input
            .into_iter()
            .map(|f| self.egraph.parser.get_fact_from_string(None, f).unwrap())
            .collect()
    }
}
