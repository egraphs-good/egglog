use super::{Rewrite, Rule};
use crate::*;

pub struct Desugar {
    pub(crate) fresh_gen: SymbolGen,
    // Store the parser because it takes some time
    // on startup for some reason
    parser: ast::parse::ProgramParser,
}

impl Default for Desugar {
    fn default() -> Self {
        Self {
            // the default reserved string in egglog is "_"
            fresh_gen: SymbolGen::new("_".repeat(2)),
            parser: ast::parse::ProgramParser::new(),
        }
    }
}

fn desugar_datatype(span: Span, name: Symbol, variants: Vec<Variant>) -> Vec<NCommand> {
    vec![NCommand::Sort(span.clone(), name, None)]
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
                ignore_viz: false,
                span: variant.span,
            })
        }))
        .collect()
}

fn desugar_rewrite(
    ruleset: Symbol,
    name: Symbol,
    rewrite: &Rewrite,
    subsume: bool,
) -> Vec<NCommand> {
    let span = rewrite.span.clone();
    let var = Symbol::from("rewrite_var__");
    let mut head = Actions::singleton(Action::Union(
        span.clone(),
        Expr::Var(span.clone(), var),
        rewrite.rhs.clone(),
    ));
    if subsume {
        match &rewrite.lhs {
            Expr::Call(_, f, args) => {
                head.0.push(Action::Change(
                    span.clone(),
                    Change::Subsume,
                    *f,
                    args.to_vec(),
                ));
            }
            _ => {
                panic!("Subsumed rewrite must have a function call on the lhs");
            }
        }
    }
    // make two rules- one to insert the rhs, and one to union
    // this way, the union rule can only be fired once,
    // which helps proofs not add too much info
    vec![NCommand::NormRule {
        ruleset,
        name,
        rule: Rule {
            span: span.clone(),
            body: [Fact::Eq(
                span.clone(),
                vec![Expr::Var(span, var), rewrite.lhs.clone()],
            )]
            .into_iter()
            .chain(rewrite.conditions.clone())
            .collect(),
            head,
        },
    }]
}

fn desugar_birewrite(ruleset: Symbol, name: Symbol, rewrite: &Rewrite) -> Vec<NCommand> {
    let span = rewrite.span.clone();
    let rw2 = Rewrite {
        span,
        lhs: rewrite.rhs.clone(),
        rhs: rewrite.lhs.clone(),
        conditions: rewrite.conditions.clone(),
    };
    desugar_rewrite(ruleset, format!("{}=>", name).into(), rewrite, false)
        .into_iter()
        .chain(desugar_rewrite(
            ruleset,
            format!("{}<=", name).into(),
            &rw2,
            false,
        ))
        .collect()
}

// TODO(yz): we can delete this code once we enforce that all rule bodies cannot read the database (except EqSort).
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
            Action::Set(span, _, _, expr) => {
                var_set.extend(expr.vars());
                if let Expr::Call(..) = expr {
                    add_new_rule = true;

                    let fresh_symbol = desugar.get_fresh();
                    let fresh_var = Expr::Var(span.clone(), fresh_symbol);
                    let expr = std::mem::replace(expr, fresh_var.clone());
                    new_head_atoms.push(Fact::Eq(span.clone(), vec![fresh_var, expr]));
                };
            }
            Action::Let(span, symbol, expr) if var_set.contains(symbol) => {
                var_set.extend(expr.vars());
                if let Expr::Call(..) = expr {
                    add_new_rule = true;

                    let var = Expr::Var(span.clone(), *symbol);
                    new_head_atoms.push(Fact::Eq(span.clone(), vec![var, expr.clone()]));
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

fn desugar_simplify(
    desugar: &mut Desugar,
    expr: &Expr,
    schedule: &Schedule,
    span: Span,
) -> Vec<NCommand> {
    let mut res = vec![NCommand::Push(1)];
    let lhs = desugar.get_fresh();
    res.push(NCommand::CoreAction(Action::Let(
        span.clone(),
        lhs,
        expr.clone(),
    )));
    res.push(NCommand::RunSchedule(schedule.clone()));
    res.extend(
        desugar_command(
            Command::QueryExtract {
                span: span.clone(),
                variants: 0,
                expr: Expr::Var(span.clone(), lhs),
            },
            desugar,
            false,
            false,
        )
        .unwrap(),
    );

    res.push(NCommand::Pop(span, 1));
    res
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
        Command::Function(fdecl) => vec![NCommand::Function(fdecl)],
        Command::Relation {
            span,
            constructor,
            inputs,
        } => vec![NCommand::Function(FunctionDecl::relation(
            span,
            constructor,
            inputs,
        ))],
        Command::Datatype {
            span,
            name,
            variants,
        } => desugar_datatype(span, name, variants),
        Command::Rewrite(ruleset, rewrite, subsume) => {
            desugar_rewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite, subsume)
        }
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite)
        }
        Command::Include(span, file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("{} Failed to read file {file}", span.get_quote()));
            return desugar_commands(
                desugar.parse_program(Some(file), &s)?,
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
        Command::Sort(span, sort, option) => vec![NCommand::Sort(span, sort, option)],
        Command::AddRuleset(name) => vec![NCommand::AddRuleset(name)],
        Command::UnstableCombinedRuleset(name, subrulesets) => {
            vec![NCommand::UnstableCombinedRuleset(name, subrulesets)]
        }
        Command::Action(action) => vec![NCommand::CoreAction(action)],
        Command::Simplify {
            span,
            expr,
            schedule,
        } => desugar_simplify(desugar, &expr, &schedule, span),
        Command::RunSchedule(sched) => {
            vec![NCommand::RunSchedule(sched.clone())]
        }
        Command::PrintOverallStatistics => {
            vec![NCommand::PrintOverallStatistics]
        }
        Command::QueryExtract {
            span,
            variants,
            expr,
        } => {
            let variants = Expr::Lit(span.clone(), Literal::Int(variants.try_into().unwrap()));
            if let Expr::Var(..) = expr {
                // (extract {v} {variants})
                vec![NCommand::CoreAction(Action::Extract(
                    span.clone(),
                    expr,
                    variants,
                ))]
            } else {
                // (check {expr})
                // (ruleset {fresh_ruleset})
                // (rule ((= {fresh} {expr}))
                //       ((extract {fresh} {variants}))
                //       :ruleset {fresh_ruleset})
                // (run {fresh_ruleset} 1)
                let fresh = desugar.get_fresh();
                let fresh_ruleset = desugar.get_fresh();
                let fresh_rulename = desugar.get_fresh();
                let rule = Rule {
                    span: span.clone(),
                    body: vec![Fact::Eq(
                        span.clone(),
                        vec![Expr::Var(span.clone(), fresh), expr.clone()],
                    )],
                    head: Actions::singleton(Action::Extract(
                        span.clone(),
                        Expr::Var(span.clone(), fresh),
                        variants,
                    )),
                };
                vec![
                    NCommand::Check(span.clone(), vec![Fact::Fact(expr.clone())]),
                    NCommand::AddRuleset(fresh_ruleset),
                    NCommand::NormRule {
                        name: fresh_rulename,
                        ruleset: fresh_ruleset,
                        rule,
                    },
                    NCommand::RunSchedule(Schedule::Run(
                        span.clone(),
                        RunConfig {
                            ruleset: fresh_ruleset,
                            until: None,
                        },
                    )),
                ]
            }
        }
        Command::Check(span, facts) => {
            let res = vec![NCommand::Check(span, facts)];

            if get_all_proofs {
                // TODO check proofs
            }

            res
        }
        Command::CheckProof => vec![NCommand::CheckProof],
        Command::PrintFunction(span, symbol, size) => {
            vec![NCommand::PrintTable(span, symbol, size)]
        }
        Command::PrintSize(span, symbol) => vec![NCommand::PrintSize(span, symbol)],
        Command::Output { span, file, exprs } => vec![NCommand::Output { span, file, exprs }],
        Command::Push(num) => {
            vec![NCommand::Push(num)]
        }
        Command::Pop(span, num) => {
            vec![NCommand::Pop(span, num)]
        }
        Command::Fail(span, cmd) => {
            let mut desugared = desugar_command(*cmd, desugar, false, seminaive_transform)?;

            let last = desugared.pop().unwrap();
            desugared.push(NCommand::Fail(span, Box::new(last)));
            return Ok(desugared);
        }
        Command::Input { span, name, file } => {
            vec![NCommand::Input { span, name, file }]
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
            fresh_gen: self.fresh_gen.clone(),
            parser: ast::parse::ProgramParser::new(),
        }
    }
}

impl Desugar {
    pub fn get_fresh(&mut self) -> Symbol {
        self.fresh_gen.fresh(&"v".into())
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

    pub fn parse_program(
        &self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<Command>, Error> {
        let filename = filename.unwrap_or_else(|| DEFAULT_FILENAME.to_string());
        let srcfile = Arc::new(SrcFile {
            name: filename,
            contents: Some(input.to_string()),
        });
        Ok(self
            .parser
            .parse(&srcfile, input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?)
    }

    pub fn parent_name(&mut self, eqsort_name: Symbol) -> Symbol {
        self.fresh_gen
            .generate_special(&format!("{}Parent", eqsort_name).into())
    }

    pub fn lookup_parent_name(&self, eqsort_name: Symbol) -> Option<Symbol> {
        self.fresh_gen
            .lookup_special(&format!("{}Parent", eqsort_name).into())
    }
}
