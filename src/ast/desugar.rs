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
                ignore_viz: false,
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
    let span = rewrite.ann;
    let var = Symbol::from("rewrite_var__");
    let mut head = Actions::singleton(Action::Union(
        span,
        Expr::Var(span, var),
        rewrite.rhs.clone(),
    ));
    if subsume {
        match &rewrite.lhs {
            Expr::Call(_, f, args) => {
                head.0
                    .push(Action::Change(span, Change::Subsume, *f, args.to_vec()));
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
            ann: span,
            body: [Fact::Eq(span, vec![Expr::Var(span, var), rewrite.lhs.clone()])]
                .into_iter()
                .chain(rewrite.conditions.clone())
                .collect(),
            head,
        },
    }]
}

fn desugar_birewrite(ruleset: Symbol, name: Symbol, rewrite: &Rewrite) -> Vec<NCommand> {
    let span = rewrite.ann;
    let rw2 = Rewrite {
        ann: span,
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
            Action::Set(ann, _, _, expr) => {
                var_set.extend(expr.vars());
                if let Expr::Call(..) = expr {
                    add_new_rule = true;

                    let fresh_symbol = desugar.get_fresh();
                    let fresh_var = Expr::Var(*ann, fresh_symbol);
                    let expr = std::mem::replace(expr, fresh_var.clone());
                    new_head_atoms.push(Fact::Eq(*ann, vec![fresh_var, expr]));
                };
            }
            Action::Let(ann, symbol, expr) if var_set.contains(symbol) => {
                var_set.extend(expr.vars());
                if let Expr::Call(..) = expr {
                    add_new_rule = true;

                    let var = Expr::Var(*ann, *symbol);
                    new_head_atoms.push(Fact::Eq(*ann, vec![var, expr.clone()]));
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

fn desugar_simplify(desugar: &mut Desugar, expr: &Expr, schedule: &Schedule) -> Vec<NCommand> {
    let ann = expr.ann();
    let mut res = vec![NCommand::Push(1)];
    let lhs = desugar.get_fresh();
    res.push(NCommand::CoreAction(Action::Let(ann, lhs, expr.clone())));
    res.push(NCommand::RunSchedule(schedule.clone()));
    res.extend(
        desugar_command(
            Command::QueryExtract {
                variants: 0,
                expr: Expr::Var(ann, lhs),
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
    span: Span,
    idents: Vec<IdentSort>,
    exprs: Vec<Expr>,
    seminaive_transform: bool,
) -> Result<Vec<NCommand>, Error> {
    let mut res = vec![];

    // first, push all the idents
    for IdentSort { ident, sort } in idents {
        res.push(Command::Declare {
            ann: span,
            name: ident,
            sort,
        });
    }

    // now, for every pair of exprs we need to prove them equal
    for expr1and2 in exprs.windows(2) {
        let expr1 = &expr1and2[0];
        let expr2 = &expr1and2[1];
        res.push(Command::Push(1));

        // add the two exprs only when they are calls (consts and vars don't need to be populated).
        if let Expr::Call(..) = expr1 {
            res.push(Command::Action(Action::Expr(expr1.ann(), expr1.clone())));
        }
        if let Expr::Call(..) = expr2 {
            res.push(Command::Action(Action::Expr(expr2.ann(), expr2.clone())));
        }

        res.push(Command::RunSchedule(Schedule::Saturate(
            span,
            Box::new(Schedule::Run(
                span,
                RunConfig {
                    ruleset: "".into(),
                    until: Some(vec![Fact::Eq(span, vec![expr1.clone(), expr2.clone()])]),
                },
            )),
        )));

        res.push(Command::Check(
            span,
            vec![Fact::Eq(span, vec![expr1.clone(), expr2.clone()])],
        ));

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
        Command::Declare { ann, name, sort } => desugar.declare(ann, name, sort),
        Command::Datatype { name, variants } => desugar_datatype(name, variants),
        Command::Rewrite(ruleset, rewrite, subsume) => {
            desugar_rewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite, subsume)
        }
        Command::BiRewrite(ruleset, rewrite) => {
            desugar_birewrite(ruleset, rewrite_name(&rewrite).into(), &rewrite)
        }
        Command::Include(file) => {
            let s = std::fs::read_to_string(&file)
                .unwrap_or_else(|_| panic!("Failed to read file {file}"));
            return desugar_commands(
                desugar.parse_program(Some(file.into()), &s)?,
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
        Command::AddRuleset(name) => vec![NCommand::AddRuleset(name)],
        Command::UnstableCombinedRuleset(name, subrulesets) => {
            vec![NCommand::UnstableCombinedRuleset(name, subrulesets)]
        }
        Command::Action(action) => vec![NCommand::CoreAction(action)],
        Command::Simplify { expr, schedule } => desugar_simplify(desugar, &expr, &schedule),
        Command::Calc(span, idents, exprs) => {
            desugar_calc(desugar, span, idents, exprs, seminaive_transform)?
        }
        Command::RunSchedule(sched) => {
            vec![NCommand::RunSchedule(sched.clone())]
        }
        Command::PrintOverallStatistics => {
            vec![NCommand::PrintOverallStatistics]
        }
        Command::QueryExtract { variants, expr } => {
            let fresh = desugar.get_fresh();
            let fresh_ruleset = desugar.get_fresh();
            let desugaring = if let Expr::Var(_, v) = expr {
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
                desugar.parse_program(todo!("our filename should be richer to better support locating desugared rule. Alternatively, we can just use the same source location"), &desugaring).unwrap(),
                get_all_proofs,
                seminaive_transform,
            )?
        }
        Command::Check(span, facts) => {
            let res = vec![NCommand::Check(span, facts)];

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
        filename: Option<Symbol>,
        input: &str,
    ) -> Result<Vec<Command>, Error> {
        let filename = filename.unwrap_or_else(|| Symbol::from(DEFAULT_FILENAME));
        Ok(self
            .parser
            .parse(filename, input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?)
    }

    // TODO declare by creating a new global function. See issue #334
    pub fn declare(&mut self, span: Span, name: Symbol, sort: Symbol) -> Vec<NCommand> {
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
                ignore_viz: false,
            }),
            NCommand::CoreAction(Action::Let(span, name, Expr::Call(span, fresh, vec![]))),
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
            ignore_viz: fdecl.ignore_viz,
        })]
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
