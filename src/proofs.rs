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
                "(rule ((= ({pname} a) b)
                        (= ({pname} b) c))
                       ((set ({pname} a) c))
                            :ruleset parent__)"
            ),
        ]
        .into_iter()
        .flat_map(|s| self.desugar.parser.parse(&s).unwrap())
        .collect()
    }

    fn make_rebuilding_func(&self, fdecl: &FunctionDecl) -> Vec<Command> {
        let op = fdecl.name;
        let pname = self.parent_name(fdecl.schema.output);
        let child = |i| format!("c{i}_");
        let child_parent =
            |i| format!("({} {})", self.parent_name(fdecl.schema.input[i]), child(i));
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
            "(rule ((= e ({op} {children})))
                   ((let lhs ({op} {children_updated}))
                    (let rhs ({pname} e))
                    (set ({pname} lhs) rhs)
                    (set ({pname} rhs) lhs))
                    :ruleset rebuilding__)",
        )]
        .into_iter()
        .flat_map(|s| self.desugar.parser.parse(&s).unwrap())
        .collect()
    }

    fn var_to_parent(&self, var: Symbol) -> Symbol {
        Symbol::from(format!("{}_parent__", var))
    }

    fn instrument_fact(&mut self, fact: &NormFact) -> Vec<Fact> {
        match fact {
            NormFact::ConstrainEq(lhs, rhs) => {
                let lhs_t = self.type_info.lookup(self.current_ctx, *lhs).unwrap();
                let rhs_t = self.type_info.lookup(self.current_ctx, *rhs).unwrap();
                assert!(lhs_t.name() == rhs_t.name());
                let parent = self.parent_name(lhs_t.name());

                vec![format!("(= ({parent} {lhs}) ({parent} {rhs}))")]
                    .into_iter()
                    .map(|s| self.desugar.fact_parser.parse(&s).unwrap())
                    .collect::<Vec<Fact>>()
            }
            _ => vec![fact.to_fact()],
        }
    }

    fn instrument_facts(&mut self, facts: &Vec<NormFact>) -> Vec<Fact> {
        facts.iter().flat_map(|f| self.instrument_fact(f)).collect()
    }

    fn parse_actions(&self, actions: Vec<String>) -> Vec<Action> {
        actions
            .into_iter()
            .map(|s| self.desugar.action_parser.parse(&s).unwrap())
            .collect()
    }

    fn instrument_action(&mut self, action: &NormAction) -> Vec<Action> {
        [
            vec![action.to_action()],
            match action {
                NormAction::Delete(_) => {
                    // TODO what to do about delete?
                    vec![]
                }
                NormAction::Let(lhs, _expr) => {
                    let lhs_type = self.type_info.lookup(self.current_ctx, *lhs).unwrap();
                    let pname = self.parent_name(lhs_type.name());
                    self.parse_actions(vec![format!("(set ({pname} {lhs}) {lhs})")])
                }
                NormAction::LetLit(..) => vec![],
                NormAction::LetIteration(..) => vec![],
                NormAction::LetVar(..) => vec![],
                NormAction::Panic(..) => vec![],
                NormAction::Set(expr, rhs) => {
                    let type_info = self
                        .type_info
                        .typecheck_expr(self.current_ctx, expr, true)
                        .unwrap();
                    if !type_info.has_merge && type_info.output.is_eq_sort() {
                        let pname = self.parent_name(type_info.output.name());
                        self.parse_actions(vec![
                            format!("(set ({pname} {expr}) ({pname} {rhs}))"),
                            format!("(set ({pname} {rhs}) ({pname} {expr}))"),
                        ])
                    } else {
                        vec![]
                    }
                }
                NormAction::Union(lhs, rhs) => {
                    let lhs_type = self.type_info.lookup(self.current_ctx, *lhs).unwrap();
                    let rhs_type = self.type_info.lookup(self.current_ctx, *rhs).unwrap();
                    assert_eq!(lhs_type.name(), rhs_type.name());
                    let pname = self.parent_name(lhs_type.name());
                    self.parse_actions(vec![
                        format!("(set ({pname} {lhs}) ({pname} {rhs}))"),
                        format!("(set ({pname} {rhs}) ({pname} {lhs}))"),
                    ])
                }
            },
        ]
        .concat()
    }

    fn instrument_actions(&mut self, actions: &Vec<NormAction>) -> Vec<Action> {
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
                NCommand::Sort(name, _presort_and_args) => {
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
                NCommand::NormAction(action) => {
                    res.extend(
                        self.instrument_action(action)
                            .into_iter()
                            .map(Command::Action),
                    );
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

    pub(crate) fn proof_header(&self) -> Vec<Command> {
        let str = include_str!("termheader.egg");
        self.parse_program(str).unwrap().into_iter().collect()
    }

    pub(crate) fn literal_name(&self, lit: &Literal) -> Symbol {
        self.type_info.infer_literal(lit).name()
    }
}
