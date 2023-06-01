use crate::*;

use crate::ast::desugar::Desugar;

pub const RULE_PROOF_KEYWORD: &str = "rule-proof";

#[derive(Default, Clone)]
pub(crate) struct ProofState {
    pub(crate) current_ctx: CommandId,
    pub(crate) desugar: Desugar,
    pub(crate) type_info: TypeInfo,
    pub(crate) term_header_added: bool,
}
