use egglog::ast::*;
use egglog_ast::span::Span;

pub struct WithRuleset;

impl Macro<Vec<Command>> for WithRuleset {
    fn name(&self) -> &str {
        "with-ruleset"
    }

    fn parse(
        &self,
        args: &[Sexp],
        span: Span,
        parser: &mut Parser,
    ) -> Result<Vec<Command>, ParseError> {
        match args {
            [Sexp::Atom(ruleset, _), rest @ ..] => {
                let parsed_cmds = rest
                    .iter()
                    .map(|c| parser.parse_command(c))
                    .collect::<Result<Vec<_>, _>>()?;
                parsed_cmds
                    .into_iter()
                    .flatten()
                    .map(|cmd| {
                        Ok(match cmd {
                            Command::Rule { rule } => {
                                if !rule.ruleset.is_empty() {
                                    return Err(ParseError(
                                        rule.span,
                                        "expected rules in `with-ruleset` to have empty ruleset"
                                            .to_string(),
                                    ));
                                }

                                Command::Rule {
                                    rule: GenericRule {
                                        ruleset: ruleset.clone(),
                                        ..rule
                                    },
                                }
                            }
                            Command::Rewrite(rule_ruleset, rewrite, subsume) => {
                                if !rule_ruleset.is_empty() {
                                    return Err(ParseError(
                                        rewrite.span,
                                        "expected rules in `with-ruleset` to have empty ruleset"
                                            .to_string(),
                                    ));
                                }

                                Command::Rewrite(ruleset.clone(), rewrite, subsume)
                            }
                            Command::BiRewrite(rule_ruleset, rewrite) => {
                                if !rule_ruleset.is_empty() {
                                    return Err(ParseError(
                                        rewrite.span,
                                        "expected rules in `with-ruleset` to have empty ruleset"
                                            .to_string(),
                                    ));
                                }

                                Command::BiRewrite(ruleset.clone(), rewrite)
                            }
                            _ => {
                                // Ideally the span should be the current command's span (i.e. cmd.span()),
                                // but currently not all of our commands have a span field.
                                return Err(ParseError(
                                    span.clone(),
                                    "expected rule or rewrite".to_string(),
                                ));
                            }
                        })
                    })
                    .collect()
            }
            _ => Err(ParseError(
                span,
                "expected (with-ruleset <ruleset> <command>*)".to_string(),
            )),
        }
    }
}
