use egglog::{ast::*, util::FreshGen};
use egglog_ast::span::Span;

pub struct For;

impl Macro<Vec<Command>> for For {
    fn name(&self) -> &str {
        "for"
    }

    fn parse(
        &self,
        args: &[Sexp],
        span: Span,
        parser: &mut Parser,
    ) -> Result<Vec<Command>, ParseError> {
        if args.len() != 2 {
            return Err(ParseError(
                span,
                "expected (for <query> <action>)".to_string(),
            ));
        }

        let ruleset = parser.symbol_gen.fresh("for_ruleset");
        let rulename = parser.symbol_gen.fresh("for_rule");
        let query = args[0]
            .expect_list("query")?
            .iter()
            .map(|s| parser.parse_fact(s))
            .collect::<Result<Vec<_>, _>>()?;
        let action = args[1]
            .expect_list("action")?
            .iter()
            .map(|s| parser.parse_action(s))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let rule = Rule {
            span: span.clone(),
            head: Actions::new(action),
            body: query,
            name: rulename,
            ruleset: ruleset.clone(),
        };

        Ok(vec![
            Command::AddRuleset(span.clone(), ruleset.clone()),
            Command::Rule { rule },
            Command::RunSchedule(Schedule::Run(
                span.clone(),
                RunConfig {
                    ruleset,
                    until: None,
                },
            )),
        ])
    }
}
