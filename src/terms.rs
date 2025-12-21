use crate::ast::GenericCommand;
use crate::*;
use std::path::{Path, PathBuf};

#[derive(Default, Clone)]
pub(crate) struct ProofConstants {
    pub uf_parent: HashMap<String, String>,
    pub term_header_added: bool,
    // TODO this is very ugly- we should separate out a typechecking struct
    // since we didn't need an entire e-graph
    // When Some term encoding is enabled.
    pub original_typechecking: Option<Box<EGraph>>,
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

    pub(crate) fn uf_name(&mut self, sort: &str) -> String {
        if let Some(name) = self.egraph.proof_state.uf_parent.get(sort) {
            name.clone()
        } else {
            self.egraph
                .proof_state
                .uf_parent
                .insert(sort.to_string(), format!("ufparent_{}", sort,));
            self.egraph.proof_state.uf_parent[sort].clone()
        }
    }

    fn single_parent_ruleset_name(&self) -> String {
        String::from(format!("single_parent",))
    }

    /// Mark two things for unioning.
    pub(crate) fn union(&mut self, type_name: &str, lhs: &str, rhs: &str) -> String {
        let uf_name = self.uf_name(type_name);
        format!("({uf_name} (ordering-max {lhs} {rhs}) (ordering-min {lhs} {rhs}))",)
    }

    /// The parent table is the database representation of a union-find datastructure.
    /// When one term has two parents, those parents are unioned in the merge action.
    /// Also, we have a rule that maintains the invariant that each term points to its
    /// canonical representative.
    fn make_uf_table(&mut self, name: &str) -> Vec<Command> {
        let pname = self.uf_name(name);
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("uf");
        let fresh_name = self.egraph.parser.symbol_gen.fresh("uf_update");
        self.parse_program(&format!(
            "(sort {fresh_sort})
             (constructor {pname} ({name} {name}) {fresh_sort})
             (rule (({pname} a b)
                    ({pname} b c)
                    (!= b c))
                  ((delete ({pname} a b))
                   ({pname} a c))
                   :ruleset {}
                   :name \"{fresh_name}\")
             (rule (({pname} a b)
                    ({pname} a c)
                    (!= b c))
                  ((delete ({pname} a (ordering-max b c)))
                   ({pname} (ordering-max b c) (ordering-min b c)))
                   :ruleset {}
                   :name \"singleparent{fresh_name}\")
                   ",
            self.parent_direct_ruleset_name(),
            self.single_parent_ruleset_name(),
        ))
        .unwrap()
    }

    // Each function/constructor gets a view table, the canonicalized e-nodes to accelerate e-matching.
    fn view_name(&self, name: &str) -> String {
        String::from(format!("{}View", name,))
    }

    // Generate a rule that runs the merge function for custom functions.
    fn handle_merge_fn(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>()
            .join(" ");
        let rebuilding_ruleset = self.rebuilding_ruleset_name();
        let view_name = self.view_name(&fdecl.name);
        if fdecl.subtype == FunctionSubtype::Custom {
            let name = &fdecl.name;

            let merge_fn = &fdecl
                .merge
                .as_ref()
                .unwrap_or_else(|| panic!("Proofs don't support :no-merge"));

            let rebuilding_cleanup_ruleset = self.rebuilding_cleanup_ruleset_name();
            let fresh_name = self.egraph.parser.symbol_gen.fresh("merge_rule");
            let cleanup_name = self.egraph.parser.symbol_gen.fresh("merge_cleanup");
            let cleanup_name2 = self.egraph.parser.symbol_gen.fresh("merge_cleanup2");
            let res_fresh = self.egraph.parser.symbol_gen.fresh("r");

            // TODO these cleanup rules are expensive. Could speed up with actual functional dependencies.
            // The first runs the merge function adding a new row.
            // The second deletes rows with old values for the old variable, while the third deletes rows with new values for the new variable.
            format!(
                "(rule (({view_name} {child_names} old)
                        ({view_name} {child_names} new)
                        (!= old new)
                        (= (ordering-max old new) new))
                       (({view_name} {child_names} {merge_fn})
                        ({name} {child_names} {merge_fn})
                       )
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")
                 (rule ((= {res_fresh} {merge_fn})
                        ({view_name} {child_names} {res_fresh})
                        ({view_name} {child_names} new)
                        ({view_name} {child_names} old)
                        (!= {res_fresh} old))
                       ((delete ({view_name} {child_names} old)))
                        :ruleset {rebuilding_cleanup_ruleset}
                        :name \"{cleanup_name}\")
                 (rule ((= {res_fresh} {merge_fn})
                        ({view_name} {child_names} {res_fresh})
                        ({view_name} {child_names} new)
                        ({view_name} {child_names} old)
                        (!= {res_fresh} new))
                       ((delete ({view_name} {child_names} new)))
                        :ruleset {rebuilding_cleanup_ruleset}
                        :name \"{cleanup_name2}\")
                ",
            )
        } else {
            // Congruence rule
            let fresh_name = self.egraph.parser.symbol_gen.fresh("congruence_rule");
            let uf_name = self.uf_name(&fdecl.schema.output);
            format!(
                "(rule (({view_name} {child_names} old)
                        ({view_name} {child_names} new)
                        (!= old new)
                        (= (ordering-max old new) new))
                       (({uf_name} new old))
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")"
            )
        }
    }

    /// Each function/constructor gets a term table and a view table.
    /// The term table stores underlying representative terms.
    /// The view table stores child terms and their eclass.
    /// The view table is mutated using delete, but we never delete from term tables.
    /// We re-use the original name of the function for the term table.
    fn term_and_view(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let schema = &fdecl.schema;
        let mut types = schema.input.clone();
        if fdecl.subtype == FunctionSubtype::Custom {
            types.push(schema.output.clone());
        }

        let view_name = self.view_name(&fdecl.name);
        let child_sorts = ListDisplay(types, " ");
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("view");
        let merge_rule = self.handle_merge_fn(fdecl);
        // the term table has child_sorts as inputs
        // the view table has child_sorts + the leader term for the eclass
        self.parse_program(&format!(
            "
            (sort {fresh_sort})
            (constructor {} ({child_sorts}) {})
            (constructor {view_name} ({child_sorts} {}) {fresh_sort})
            {}",
            fdecl.name,
            if fdecl.subtype == FunctionSubtype::Constructor {
                schema.output.clone()
            } else {
                fresh_sort.clone()
            },
            if fdecl.subtype == FunctionSubtype::Constructor {
                fdecl.schema.output.clone()
            } else {
                "".to_string()
            },
            merge_rule
        ))
        .unwrap()
    }

    fn rebuilding_rules(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let mut type_strs = fdecl.schema.input.clone();
        if fdecl.subtype == FunctionSubtype::Custom {
            type_strs.push(fdecl.schema.output.clone());
        }
        let types = fdecl.resolved_schema.view_types();

        let types2 = types.clone();

        let view_name = self.view_name(&fdecl.name);
        let name = &fdecl.name;
        let child = |i| format!("c{i}_");
        let children = format!(
            "{}",
            ListDisplay((0..types.len()).map(child).collect::<Vec<_>>(), " ")
        );
        let mut children_updated_query = vec![];
        let mut children_updated = vec![];
        for i in 0..types.len() {
            if let Some((query, var)) = self.get_canonical_expr_of(child(i), types2[i].clone()) {
                children_updated_query.push(query);
                children_updated.push(var);
            } else {
                children_updated.push(child(i));
            }
        }

        let children_updated_query = format!(
            "{}",
            ListDisplay(children_updated_query.into_iter().collect::<Vec<_>>(), " ")
        );
        let children_updated = format!(
            "{}",
            ListDisplay(children_updated.into_iter().collect::<Vec<_>>(), " ")
        );

        let mut res = vec![];
        let fresh_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");

        // Make a rule that updates the view
        let rule = format!(
            "(rule ((= lhs ({view_name} {children}))
                        {children_updated_query})
                     (
                      ({view_name} {children_updated})
                     )
                      :ruleset {} :name \"{fresh_name}\")",
            self.rebuilding_ruleset_name(),
        );
        let cleanup_name = self.egraph.parser.symbol_gen.fresh("cleanup_rule");

        // And a rule that cleans up the view
        let rule2 = format!(
            "(rule (
                        {children_updated_query}
                        (!= ({view_name} {children})
                            ({view_name} {children_updated})))
                     (
                      (delete ({view_name} {children}))
                     )
                      :ruleset {} :name \"{cleanup_name}\")",
            self.rebuilding_ruleset_name()
        );

        // don't extend res if none of them needed updating
        if !children_updated_query.is_empty() {
            res.extend(self.parse_program(&rule).unwrap());
            res.extend(self.parse_program(&rule2).unwrap());
        }

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
                let _ = self.instrument_fact_expr(generic_expr, res);
            }
        }
    }

    /// Instruments a fact expression to use the view tables.
    /// Returns a variable representing the expression.
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
                        if specialized_primitive.output().is_eq_sort() {
                            panic!(
                                "Term encoding does not support eq-sort primitive expressions in facts"
                            );
                        }
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(= {fv} ({} {}))",
                            specialized_primitive.name(),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                }
            }
        }
    }

    // Returns a query for the canonical expression of the given var of the given sort.
    fn get_canonical_expr_of(&mut self, var: String, sort: ArcSort) -> Option<(String, String)> {
        if sort.is_eq_container_sort() {
            let fresh = self.fresh_var();
            // todo containers need a custom rebuild
            Some((format!("(let {fresh} (rebuild {var}))"), fresh))
        } else {
            self.wrap_parent(var, sort)
        }
    }

    fn wrap_parent(&mut self, var: String, sort: ArcSort) -> Option<(String, String)> {
        if sort.is_eq_sort() {
            let fresh = self.fresh_var();
            let parent = self.uf_name(&sort.name());
            Some((format!("({parent} {var} {fresh})"), fresh))
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

                // todo let else panic
                let ResolvedCall::Func(func_type) = h else {
                    panic!(
                        "Set action on non-function, should have been prevented by typechecking"
                    );
                };

                // add to view
                res.push(format!(
                    "({} {})",
                    self.view_name(&func_type.name),
                    ListDisplay(exprs.clone(), " ")
                ));
                // add to term table
                res.push(format!(
                    "({} {})",
                    func_type.name,
                    ListDisplay(exprs.clone(), " ")
                ));
            }
            ResolvedAction::Change(_span, change, h, generic_exprs) => {
                if let ResolvedCall::Func(func_type) = h {
                    if func_type.subtype == FunctionSubtype::Custom {
                        panic!("proofs don't support deleting function rows");
                    } else {
                        let symbol = match change {
                            Change::Delete => "delete",
                            Change::Subsume => "subsume",
                        };
                        let children = generic_exprs
                            .iter()
                            .map(|e| self.instrument_action_expr(e, &mut res))
                            .collect::<Vec<_>>();

                        res.push(format!(
                            "({symbol} ({} {}))",
                            self.view_name(&func_type.name),
                            ListDisplay(children, " ")
                        ));
                    }
                } else {
                    panic!(
                        "Delete action on non-function, should have been prevented by typechecking"
                    );
                }
            }
            ResolvedAction::Union(_span, generic_expr, generic_expr1) => {
                let v1 = self.instrument_action_expr(generic_expr, &mut res);
                let v2 = self.instrument_action_expr(generic_expr1, &mut res);
                let ot = generic_expr.output_type();
                let type_name = ot.name();
                let unioned = self.union(type_name, &v1, &v2);
                res.push(unioned);
            }
            ResolvedAction::Panic(..) => {
                res.push(format!("{}", action));
            }
            ResolvedAction::Expr(_span, generic_expr) => {
                self.instrument_action_expr(generic_expr, &mut res);
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
                            "(let {fv} ({} {}))",
                            func_type.name,
                            ListDisplay(args.clone(), " ")
                        ));
                        // add to view table
                        res.push(format!(
                            "({} {} {fv})",
                            self.view_name(&func_type.name),
                            ListDisplay(args.clone(), " ")
                        ));
                        // add to uf table to initialize eclass for eq sorts
                        if func_type.output.is_eq_sort() {
                            let uf_name = self.uf_name(func_type.output.name());
                            res.push(format!("({} {fv} {})", uf_name, fv));
                        }

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
            res.extend(self.instrument_action(action));
        }
        res
    }

    fn instrument_rule(&mut self, rule: &ResolvedRule) -> Vec<Command> {
        let facts = self.instrument_facts(&rule.body);
        let actions = self.instrument_actions(&rule.head.0);
        let instrumented = format!(
            "(rule ({} )
                   ({} )
                    {})",
            ListDisplay(facts, " "),
            ListDisplay(actions, " "),
            if rule.ruleset == "" {
                "".to_string()
            } else {
                format!(":ruleset {}", rule.ruleset)
            }
        );
        self.parse_program(&instrumented).unwrap()
    }

    /// TODO experiment with schedule- unclear what is fastest.
    /// Any schedules should be sound.
    fn rebuild(&mut self) -> Schedule {
        let parent_direct_ruleset = self.parent_direct_ruleset_name();
        let single_parent = self.single_parent_ruleset_name();
        let rebuilding_cleanup_ruleset = self.rebuilding_cleanup_ruleset_name();
        let rebuilding_ruleset = self.rebuilding_ruleset_name();
        self.parse_schedule(format!(
            "(saturate
                  {rebuilding_cleanup_ruleset}
                  (saturate {single_parent})
                  (saturate {parent_direct_ruleset})
                  {rebuilding_ruleset})
                   "
        ))
        .unwrap()
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

    fn term_encode_command(&mut self, command: &ResolvedNCommand, res: &mut Vec<Command>) {
        match &command {
            ResolvedNCommand::Sort(_span, name, _presort_and_args) => {
                res.push(command.to_command().make_unresolved());
                res.extend(self.make_uf_table(name));
            }
            ResolvedNCommand::Function(fdecl) => {
                res.extend(self.term_and_view(fdecl));
                res.extend(self.rebuilding_rules(fdecl));
            }
            ResolvedNCommand::NormRule { rule } => {
                res.extend(self.instrument_rule(rule));
            }
            ResolvedNCommand::CoreAction(action) => {
                let instrumented = self.instrument_action(action).join("\n");
                res.extend(self.parse_program(&instrumented).unwrap());
            }
            ResolvedNCommand::Check(span, facts) => {
                let instrumented = self.instrument_facts(facts);
                res.push(Command::Check(
                    span.clone(),
                    self.parse_facts(&instrumented),
                ));
            }
            ResolvedNCommand::RunSchedule(schedule) => {
                res.push(Command::RunSchedule(self.instrument_schedule(schedule)));
            }
            ResolvedNCommand::Fail(span, cmd) => {
                self.term_encode_command(&cmd, res);
                let last = res.pop().unwrap();
                res.push(Command::Fail(span.clone(), Box::new(last)));
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
                // TODO we just omit extract for now, support in future
            }
            ResolvedNCommand::UserDefined(..) => {
                panic!("User defined commands unsupported in term encoding");
            }
        }
    }

    pub(crate) fn add_term_encoding_helper(
        &mut self,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        let mut res = vec![];

        if !self.egraph.proof_state.term_header_added {
            res.extend(self.term_header());
            self.egraph.proof_state.term_header_added = true;
        }

        for command in program {
            self.term_encode_command(&command, &mut res);

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
        "parent".to_string()
    }

    fn rebuilding_ruleset_name(&self) -> String {
        "rebuilding".to_string()
    }

    fn rebuilding_cleanup_ruleset_name(&self) -> String {
        "rebuilding_cleanup".to_string()
    }

    pub(crate) fn term_header(&mut self) -> Vec<Command> {
        let str = format!(
            "(ruleset {})
             (ruleset {})
             (ruleset {})
             (ruleset {})",
            self.parent_direct_ruleset_name(),
            self.single_parent_ruleset_name(),
            self.rebuilding_ruleset_name(),
            self.rebuilding_cleanup_ruleset_name(),
        );
        self.parse_program(&str).unwrap()
    }

    fn parse_program(&mut self, input: &str) -> Result<Vec<Command>, ParseError> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_program_from_string(None, input);
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res
    }

    fn parse_schedule(&mut self, input: String) -> Result<Schedule, ParseError> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = self.egraph.parser.get_schedule_from_string(None, &input);
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res
    }

    fn parse_facts(&mut self, input: &Vec<String>) -> Vec<Fact> {
        self.egraph.parser.ensure_no_reserved_symbols = false;
        let res = input
            .into_iter()
            .map(|f| self.egraph.parser.get_fact_from_string(None, f).unwrap())
            .collect();
        self.egraph.parser.ensure_no_reserved_symbols = true;
        res
    }
}

pub fn term_encoding_supported(path: &Path) -> bool {
    let mut visited = HashSet::default();
    term_encoding_supported_impl(path, &mut visited)
}

fn term_encoding_supported_impl(path: &Path, visited: &mut HashSet<PathBuf>) -> bool {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(_) => return false,
    };

    let canonical = match std::fs::canonicalize(path) {
        Ok(canonical) => canonical,
        Err(_) => return false,
    };

    if !visited.insert(canonical.clone()) {
        return true;
    }

    let mut egraph = EGraph::default();
    let filename = canonical.to_string_lossy().into_owned();
    let desugared = match egraph.desugar_program(Some(filename), &contents) {
        Ok(commands) => commands,
        Err(_) => return false,
    };

    commands_support_proof_encoding(&desugared)
}

pub fn commands_support_proof_encoding(commands: &Vec<ResolvedCommand>) -> bool {
    for command in commands {
        if !command_supports_proof_encoding(command) {
            return false;
        }
    }
    true
}

pub fn command_supports_proof_encoding(command: &ResolvedCommand) -> bool {
    match command {
        GenericCommand::Sort(_, _, Some(_))
        | GenericCommand::UserDefined(..)
        | GenericCommand::Relation { .. }
        | GenericCommand::Input { .. } => false,
        ResolvedCommand::Action(ResolvedAction::Let(_, _, expr)) => expr.output_type().is_eq_sort(),
        // no-merge isn't supported right now
        ResolvedCommand::Function { merge: None, .. } => false,
        // delete or subsume on custom functions isn't supported
        _ => {
            let mut res = true;
            command.clone().visit_actions(&mut |action| {
                if let ResolvedAction::Change(_, _change, ResolvedCall::Func(func_type), _) =
                    &action
                {
                    if func_type.subtype == FunctionSubtype::Custom {
                        res = false;
                    }
                }
                action
            });
            res
        }
    }
}
