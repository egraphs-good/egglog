use super::{Rewrite, Rule};
use crate::*;

fn desugar_datatype(name: Symbol, variants: Vec<Variant>) -> Vec<NCommand> {
    vec![NCommand::Sort(name, None)]
        .into_iter()
        .chain(variants.into_iter().map(|variant| {
            NCommand::Function(FunctionDecl {
                name: variant.name,
                schema: Schema {
                    input: variant.types,
                    output: name,
                },
                merge: None,
                merge_action: Actions::default(),
                default: None,
                cost: variant.cost,
                unextractable: false,
            })
        }))
        .collect()
}

fn desugar_rewrite(ruleset: Symbol, name: Symbol, rewrite: &Rewrite) -> Vec<NCommand> {
    let var = Symbol::from("rewrite_var__");
    // make two rules- one to insert the rhs, and one to union
    // this way, the union rule can only be fired once,
    // which helps proofs not add too much info
    vec![NCommand::NormRule {
        ruleset,
        name,
        rule: Rule {
            body: [Fact::Eq(vec![Expr::Var((), var), rewrite.lhs.clone()])]
                .into_iter()
                .chain(rewrite.conditions.clone())
                .collect(),
            head: Actions::singleton(Action::Union((), Expr::Var((), var), rewrite.rhs.clone())),
        },
    }]
}

fn desugar_birewrite(ruleset: Symbol, name: Symbol, rewrite: &Rewrite) -> Vec<NCommand> {
    let rw2 = Rewrite {
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(ruleset, format!("{}=>", name).into(), rewrite)
        .into_iter()
        .chain(desugar_rewrite(ruleset, format!("{}<=", name).into(), &rw2))
        .collect()
}

fn add_semi_naive_rule(desugar: &mut Desugar, rule: Rule) -> Option<Rule> {
    let mut new_rule = rule;
    // Whenever an Let(_, expr@Call(...)) or Set(_, expr@Call(...)) is present in action,
    // an additional seminaive rule should be created.
    // Moreover, for each such expr, expr and all variable definitions that it relies on should be moved to trigger.
    let mut new_head_atoms = vec![];
    let mut add_new_rule = false;

    let mut var_set = HashSet::default();
    for head_slice in new_rule.head.0.iter_mut().rev() {
        match head_slice {
            Action::Set(_ann, _, _, expr) => {
                var_set.extend(expr.vars());
                if let Expr::Call((), _, _) = expr {
                    add_new_rule = true;

                    let fresh_symbol = desugar.get_fresh();
                    let fresh_var = Expr::Var((), fresh_symbol);
                    let expr = std::mem::replace(expr, fresh_var.clone());
                    new_head_atoms.push(Fact::Eq(vec![fresh_var, expr]));
                };
            }
            Action::Let(_ann, symbol, expr) if var_set.contains(symbol) => {
                var_set.extend(expr.vars());
                if let Expr::Call((), _, _) = expr {
                    add_new_rule = true;

                    let var = Expr::Var((), *symbol);
                    new_head_atoms.push(Fact::Eq(vec![var, expr.clone()]));
                }
            }
            _ => (),
        }
    }

    if add_new_rule {
        new_rule.body.extend(new_head_atoms.into_iter().rev());
        // remove all let action
        new_rule.head.0.retain_mut(
            |action| !matches!(action, Action::Let(_ann, var, _) if var_set.contains(var)),
        );
        log::debug!("Added a semi-naive desugared rule:\n{}", new_rule);
        Some(new_rule)
    } else {
        None
    }
}

/// The Desugar struct stores all the state needed
/// during desugaring a program.
/// While desugaring doesn't need type information, it
/// needs to know what global variables exist.
/// It also needs to know what functions are primitives
/// (it uses the [`TypeInfo`] for that.
/// After desugaring, typechecking happens and the
/// type_info field is used for that.
pub struct Desugar {
    next_fresh: usize,
    // Store the parser because it takes some time
    // on startup for some reason
    parser: ast::parse::ProgramParser,
    // yz (dec 5): Comment out since they are only used in terms.rs which are deleted
    // pub(crate) expr_parser: ast::parse::ExprParser,
    // pub(crate) action_parser: ast::parse::ActionParser,
    // TODO fix getting fresh names using modules
    pub(crate) number_underscores: usize,
    pub(crate) type_info: TypeInfo,
}

impl Default for Desugar {
    fn default() -> Self {
        let type_info = TypeInfo::default();
        Self {
            next_fresh: Default::default(),
            // these come from lalrpop and don't have default impls
            parser: ast::parse::ProgramParser::new(),
            // expr_parser: ast::parse::ExprParser::new(),
            // action_parser: ast::parse::ActionParser::new(),
            number_underscores: 3,
            type_info,
        }
    }
}

fn desugar_simplify(desugar: &mut Desugar, expr: &Expr, schedule: &Schedule) -> Vec<NCommand> {
    let mut res = vec![NCommand::Push(1)];
    let lhs = desugar.get_fresh();
    res.push(NCommand::NormAction(Action::Let((), lhs, expr.clone())));
    res.push(NCommand::RunSchedule(schedule.clone()));
    res.extend(
        desugar_command(
            Command::QueryExtract {
                variants: 0,
                expr: Expr::Var((), lhs),
            },
            desugar,
            false,
            false,
        )
        .unwrap(),
    );

    res.push(NCommand::Pop(1));
    res
}

pub(crate) fn desugar_calc(
    desugar: &mut Desugar,
    idents: Vec<IdentSort>,
    exprs: Vec<Expr>,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let mut res = vec![];

    // first, push all the idents
    for IdentSort { ident, sort } in idents {
        res.push(Command::Declare { name: ident, sort });
    }

    // now, for every pair of exprs we need to prove them equal
    for expr1and2 in exprs.windows(2) {
        let expr1 = &expr1and2[0];
        let expr2 = &expr1and2[1];
        res.push(Command::Push(1));

        // add the two exprs only when they are calls (consts and vars don't need to be populated).
        if let Expr::Call(..) = expr1 {
            res.push(Command::Action(Action::Expr((), expr1.clone())));
        }
        if let Expr::Call(..) = expr2 {
            res.push(Command::Action(Action::Expr((), expr2.clone())));
        }

        res.push(Command::RunSchedule(Schedule::Saturate(Box::new(
            Schedule::Run(RunConfig {
                ruleset: "".into(),
                until: Some(vec![Fact::Eq(vec![expr1.clone(), expr2.clone()])]),
            }),
        ))));

        res.push(Command::Check(vec![Fact::Eq(vec![
            expr1.clone(),
            expr2.clone(),
        ])]));

        res.push(Command::Pop(1));
    }

    desugar_commands(res, desugar, false, seminaive_transform)
}

pub(crate) fn rewrite_name(rewrite: &Rewrite) -> String {
    rewrite.to_string().replace('\"', "'")
}

/// Desugars a single command into the normalized form.
/// Gets rid of a bunch of syntactic sugar, but also
/// makes rules into a SSA-like format (see [`NormFact`]).
pub(crate) fn desugar_command(
    command: Command,
    desugar: &mut Desugar,
    get_all_proofs: bool,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let res = match command {
        Command::SetOption { name, value } => {
            vec![NCommand::SetOption { name, value }]
        }
        Command::Function(fdecl) => desugar.desugar_function(&fdecl),
        Command::Relation {
            constructor,
            inputs,
        } => desugar.desugar_function(&FunctionDecl::relation(constructor, inputs)),
        Command::Declare { name, sort } => desugar.declare(name, sort),
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Rewrite(ruleset, rewrite) => {
            desugar_rewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite)
        }
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite)
        }
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            return desugar_commands(
                desugar.parse_program(&s)?,
                desugar,
                get_all_proofs,
                seminaive_transform,
            );
        }
        Command::Rule {
            ruleset,
            mut name,
            rule,
        } => {
            if name == "".into() {
                name = rule.to_string().replace('\"', "'").into();
            }

            let mut result = vec![NCommand::NormRule {
                ruleset,
                name,
                rule: rule.clone(),
            }];

            if seminaive_transform {
                if let Some(new_rule) = add_semi_naive_rule(desugar, rule) {
                    result.push(NCommand::NormRule {
                        ruleset,
                        name,
                        rule: new_rule,
                    });
                }
            }

            result
        }
        Command::Sort(sort, option) => vec![NCommand::Sort(sort, option)],
        // TODO ignoring cost for now
        Command::AddRuleset(name) => vec![NCommand::AddRuleset(name)],
        Command::Action(action) => vec![NCommand::NormAction(action)],
        Command::Simplify { expr, schedule } => desugar_simplify(desugar, &expr, &schedule),
        Command::Calc(idents, exprs) => desugar_calc(desugar, idents, exprs, seminaive_transform)?,
        Command::RunSchedule(sched) => {
            vec![NCommand::RunSchedule(sched.clone())]
        }
        Command::PrintOverallStatistics => {
            vec![NCommand::PrintOverallStatistics]
        }
        Command::QueryExtract { variants, expr } => {
            let fresh = desugar.get_fresh();
            let fresh_ruleset = desugar.get_fresh();
            let desugaring = if let Expr::Var((), v) = expr {
                format!("(extract {v} {variants})")
            } else {
                format!(
                    "(check {expr})
                    (ruleset {fresh_ruleset})
                    (rule ((= {fresh} {expr}))
                          ((extract {fresh} {variants}))
                          :ruleset {fresh_ruleset})
                    (run {fresh_ruleset} 1)"
                )
            };

            desugar.desugar_program(
                desugar.parse_program(&desugaring).unwrap(),
                get_all_proofs,
                seminaive_transform,
            )?
        }
        Command::Check(facts) => {
            let res = vec![NCommand::Check(facts)];

            if get_all_proofs {
                // TODO check proofs
            }

            res
        }
        Command::CheckProof => vec![NCommand::CheckProof],
        Command::PrintFunction(symbol, size) => {
            vec![NCommand::PrintTable(symbol, size)]
        }
        Command::PrintSize(symbol) => vec![NCommand::PrintSize(symbol)],
        Command::Output { file, exprs } => vec![NCommand::Output { file, exprs }],
        Command::Push(num) => {
            vec![NCommand::Push(num)]
        }
        Command::Pop(num) => {
            vec![NCommand::Pop(num)]
        }
        Command::Fail(cmd) => {
            let mut desugared = desugar_command(*cmd, desugar, false, seminaive_transform)?;

            let last = desugared.pop().unwrap();
            desugared.push(NCommand::Fail(Box::new(last)));
            return Ok(desugared);
        }
        Command::Input { name, file } => {
            vec![NCommand::Input { name, file }]
        }
    };

    Ok(res)
}

pub(crate) fn desugar_commands(
    program: Vec<Command>,
    desugar: &mut Desugar,
    get_all_proofs: bool,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let mut res = vec![];
    for command in program {
        let desugared = desugar_command(command, desugar, get_all_proofs, seminaive_transform)?;
        res.extend(desugared);
    }
    Ok(res)
}

impl Clone for Desugar {
    fn clone(&self) -> Self {
        Self {
            next_fresh: self.next_fresh,
            parser: ast::parse::ProgramParser::new(),
            // expr_parser: ast::parse::ExprParser::new(),
            // action_parser: ast::parse::ActionParser::new(),
            number_underscores: self.number_underscores,
            type_info: self.type_info.clone(),
        }
    }
}

impl Desugar {
    pub fn merge_ruleset_name(&self) -> Symbol {
        Symbol::from(format!(
            "merge_ruleset{}",
            "_".repeat(self.number_underscores)
        ))
    }

    pub fn get_fresh(&mut self) -> Symbol {
        self.next_fresh += 1;
        format!(
            "v{}{}",
            self.next_fresh - 1,
            "_".repeat(self.number_underscores)
        )
        .into()
    }

    pub(crate) fn desugar_program(
        &mut self,
        program: Vec<Command>,
        get_all_proofs: bool,
        seminaive_transform: bool,
    ) -> Result<Vec<NCommand>, Error> {
        let res = desugar_commands(program, self, get_all_proofs, seminaive_transform)?;
        Ok(res)
    }

    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        Ok(self
            .parser
            .parse(input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?)
    }

    pub fn declare(&mut self, name: Symbol, sort: Symbol) -> Vec<NCommand> {
        let fresh = self.get_fresh();
        vec![
            NCommand::Function(FunctionDecl {
                name: fresh,
                schema: Schema {
                    input: vec![],
                    output: sort,
                },
                default: None,
                merge: None,
                merge_action: Actions::default(),
                cost: None,
                unextractable: false,
            }),
            NCommand::NormAction(Action::Let((), name, Expr::Call((), fresh, vec![]))),
        ]
    }

    pub fn desugar_function(&mut self, fdecl: &FunctionDecl) -> Vec<NCommand> {
        vec![NCommand::Function(FunctionDecl {
            name: fdecl.name,
            schema: fdecl.schema.clone(),
            default: fdecl.default.clone(),
            merge: fdecl.merge.clone(),
            merge_action: fdecl.merge_action.clone(),
            cost: fdecl.cost,
            unextractable: fdecl.unextractable,
        })]
    }

    /// Get the name of the parent table for a sort
    /// for the term encoding (not related to desugaring)
    pub(crate) fn parent_name(&self, sort: Symbol) -> Symbol {
        Symbol::from(format!(
            "{}_Parent{}",
            sort,
            "_".repeat(self.number_underscores)
        ))
    }
}
