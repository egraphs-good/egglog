use crate::{
    Action, Actions, GenericAction, GenericNCommand, NCommand, ResolvedAction, ResolvedActions,
    ResolvedNCommand, ResolvedRule,
};

use super::Rule;

impl ResolvedExpr {
    pub fn to_unresolved(self) -> Expr {}
}

impl ResolvedAction {
    pub fn to_unresolved(self) -> Action {
        match self {
            GenericAction::Let(ann, var, expr) => {
                Action::Let(ann, var.to_unresolved(), expr.to_unresolved())
            }
            GenericAction::Set(ann, var, sort, expr) => {
                Action::Set(ann, var.to_unresolved(), sort, expr.to_unresolved())
            }
            GenericAction::Change(ann, change, sort, expr) => {
                Action::Change(ann, change, sort, expr.to_unresolved())
            }
            GenericAction::Union(ann, lhs, rhs) => {
                Action::Union(ann, lhs.to_unresolved(), rhs.to_unresolved())
            }
            GenericAction::Extract(ann, expr, expr2) => {
                Action::Extract(ann, expr.to_unresolved(), expr2.to_unresolved())
            }
            GenericAction::Panic(ann, msg) => Action::Panic(ann, msg),
            GenericAction::Expr(ann, expr) => Action::Expr(ann, expr.to_unresolved()),
        }
    }
}

impl ResolvedActions {
    pub fn to_unresolved(self) -> Actions {
        Actions(self.0.into_iter().map(|x| x.to_unresolved()).collect())
    }
}

impl ResolvedRule {
    pub fn to_unresolved(self) -> Rule {
        Rule {
            head: self.head.to_unresolved(),
            body: self.body.into_iter().map(|x| x.to_unresolved()).collect(),
        }
    }
}

impl ResolvedNCommand {
    pub fn to_unresolved(self) -> NCommand {
        match self {
            GenericNCommand::SetOption { name, value } => NCommand::SetOption { name, value },
            GenericNCommand::Sort(name, params) => NCommand::Sort(name, params),
            GenericNCommand::Function(func) => NCommand::Function(func.to_unresolved()),
            GenericNCommand::AddRuleset(name) => NCommand::AddRuleset(name),
            GenericNCommand::NormRule {
                name,
                ruleset,
                rule,
            } => NCommand::NormRule {
                name,
                ruleset,
                rule: rule.to_unresolved(),
            },
            GenericNCommand::RunSchedule(schedule) => {
                NCommand::RunSchedule(schedule.to_unresolved())
            }
            GenericNCommand::PrintOverallStatistics => NCommand::PrintOverallStatistics,
            GenericNCommand::CoreAction(action) => NCommand::CoreAction(action.to_unresolved()),
            GenericNCommand::CheckProof => NCommand::CheckProof,
            GenericNCommand::PrintTable(name, n) => NCommand::PrintTable(name, n),
            GenericNCommand::PrintSize(name) => NCommand::PrintSize(name),
            GenericNCommand::Output { file, exprs } => NCommand::Output { file, exprs },
            GenericNCommand::Push(n) => NCommand::Push(n),
            GenericNCommand::Pop(n) => NCommand::Pop(n),
            GenericNCommand::Fail(cmd) => NCommand::Fail(Box::new((*cmd).to_unresolved())),
            GenericNCommand::Input { name, file } => NCommand::Input { name, file },
        }
    }
}
