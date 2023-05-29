use crate::*;

use crate::ast::desugar::Desugar;
use crate::typechecking::FuncType;

use symbolic_expressions::Sexp;

pub const RULE_PROOF_KEYWORD: &str = "rule-proof";

#[derive(Default, Clone)]
pub(crate) struct ProofState {
    pub(crate) current_ctx: CommandId,
    pub(crate) desugar: Desugar,
    pub(crate) type_info: TypeInfo,
}

impl ProofState {
    pub fn parse_program(&self, input: &str) -> Result<Vec<Command>, Error> {
        self.desugar.parse_program(input)
    }

    fn parent_name(&self, sort: Symbol) -> Symbol {
        Symbol::from(format!("{}_Parent__", sort.to_string()))
    }

    fn make_rebuilding(&self, name: Symbol) -> Vec<Command> {
        let pname = self.parent_name(name);
        vec![
            format!("(function {pname} ({name}) {name} :merge (ordering-less old new))"),
            format!(
                "(rule (({pname} a b)
                        ({pname} b c))
                       (({pname} a c))
                            :ruleset parent__)"
            ),
        ]
        .into_iter()
        .flat_map(|s| self.desugar.parser.parse(&s).unwrap())
        .collect()
    }

    fn make_rebuilding_func(&self, fdecl: &FunctionDecl) -> Vec<Command> {
        let op = fdecl.name;
        let parent = self.parent_name(fdecl.schema.output);
        let child = |i| Symbol::from(format!("c{i}_"));
        let updated = |i| Symbol::from(format!("u{i}_"));
        let children = format!(
            "{}",
            ListDisplay(
                (0..fdecl.schema.input.len()).map(child).collect::<Vec<_>>(),
                " "
            )
        );
        let updateds = format!(
            "{}",
            ListDisplay(
                (0..fdecl.schema.input.len())
                    .map(updated)
                    .collect::<Vec<_>>(),
                " "
            )
        );
        let child_updates_to = format!(
            "{}",
            ListDisplay(
                (0..fdecl.schema.input.len())
                    .map(|i| format!("({parent} {} {})", (child)(i), (updated)(i)))
                    .collect::<Vec<_>>(),
                " "
            )
        );
        vec![format!(
            "(rule ((= e ({op} {children}))
                    ({parent} e ep)
                    {child_updates_to})
                   (({parent} ({op} {updateds}) ep))
                    :ruleset rebuilding__)",
        )]
        .into_iter()
        .flat_map(|s| self.desugar.parser.parse(&s).unwrap())
        .collect()
    }

    fn var_to_parent(&self, var: Symbol) -> Symbol {
        Symbol::from(format!("{}_parent__", var.to_string()))
    }

    fn instrument_fact(&mut self, fact: &NormFact) -> Vec<Fact> {
        match fact {
            NormFact::Assign(lhs, expr) => {
                let NormExpr::Call(head, body) = expr;
                let typeinfo = self
                    .type_info
                    .typecheck_expr(self.current_ctx, expr, true)
                    .unwrap();

                vec![fact.to_fact()]
                    .into_iter()
                    .chain(
                        vec![format!(
                            "({} {lhs} {})",
                            self.parent_name(typeinfo.output.name()),
                            self.var_to_parent(*lhs)
                        )]
                        .into_iter()
                        .chain(
                            body.iter()
                                .zip(typeinfo.input.iter())
                                .map(|(arg, argtype)| {
                                    format!(
                                        "({} {arg} {})",
                                        self.parent_name(argtype.name()),
                                        self.var_to_parent(*arg)
                                    )
                                }),
                        )
                        .map(|s| self.desugar.fact_parser.parse(&s).unwrap()),
                    )
                    .collect::<Vec<Fact>>()
            }
            NormFact::AssignLit(lhs, lit) => vec![fact.to_fact()],
            NormFact::ConstrainEq(lhs, rhs) => vec![format!(
                "(= {} {})",
                self.var_to_parent(*lhs),
                self.var_to_parent(*rhs),
            )]
            .into_iter()
            .map(|s| self.desugar.fact_parser.parse(&s).unwrap())
            .collect::<Vec<Fact>>(),
        }
    }

    fn instrument_facts(&mut self, facts: &Vec<NormFact>) -> Vec<Fact> {
        facts.iter().flat_map(|f| self.instrument_fact(f)).collect()
    }

    /*fn instrument_action(&self, action: &NormAction) -> Action {
        match action {
            Action::Union(lhs, rhs) => Action::Set(
        }
    }

    fn instrument_actions(&self, actions: &Vec<NormAction>) -> Vec<Action> {
        actions.iter().flat_map(|a| self.instrument_action(a)).collect()
    }*/

    fn instrument_rule(&mut self, ruleset: Symbol, name: Symbol, rule: &NormRule) -> Vec<Command> {
        vec![Command::Rule {
            ruleset,
            name,
            rule: Rule {
                head: rule.head.iter().map(|head| head.to_action()).collect(),
                body: self.instrument_facts(&rule.body),
            },
        }]
    }

    // TODO we need to also instrument merge actions and merge because they can add new terms that need representatives
    // the egraph is the initial egraph with only default sorts
    pub(crate) fn add_proofs(&mut self, program: Vec<NormCommand>) -> Vec<Command> {
        let mut res = vec![];

        for command in program {
            self.current_ctx = command.metadata.id;

            match &command.command {
                NCommand::Push(_num) => {
                    res.push(command.to_command());
                }
                NCommand::Sort(name, presort_and_args) => {
                    res.push(command.to_command());
                    res.extend(self.make_rebuilding(*name));
                }
                NCommand::Function(fdecl) => {
                    res.push(command.to_command());
                    res.extend(self.make_rebuilding_func(fdecl));
                }
                NCommand::NormRule {
                    ruleset,
                    name,
                    rule,
                } => {
                    res.extend(self.instrument_rule(*ruleset, *name, rule));
                }
                _ => {
                    res.push(command.to_command());
                }
            }
        }

        res
    }

    pub(crate) fn get_fresh(&mut self) -> Symbol {
        self.desugar.get_fresh()
    }

    pub(crate) fn make_term_prf(
        &mut self,
        trm_prf: Symbol,
        rep_trm: Symbol,
        rep_prf: String,
    ) -> Vec<Action> {
        vec![
            format!("(let {trm_prf} (MakeTrmPrf__ {rep_trm} {rep_prf}))"),
            format!("(set (TrmOf__ {trm_prf}) {rep_trm})"),
            format!("(set (PrfOf__ {trm_prf}) {rep_prf})"),
        ]
        .into_iter()
        .map(|s| self.desugar.action_parser.parse(&s).unwrap())
        .collect()
    }

    pub(crate) fn proof_header(&self) -> Vec<Command> {
        let str = include_str!("proofheader.egg");
        self.parse_program(str).unwrap().into_iter().collect()
    }

    pub(crate) fn literal_name(&self, lit: &Literal) -> Symbol {
        self.type_info.infer_literal(lit).name()
    }
}
