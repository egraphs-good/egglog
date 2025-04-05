use std::rc::Rc;

use symbol_table::Symbol;

use crate::{HashMap, Term, TermDag};

type ProofId = u32;

struct Proof {
    store: Vec<ProofTerm>,
    hashcons: HashMap<ProofTerm, ProofId>,
    termdag: TermDag,
}

type Substitution = HashMap<Symbol, Term>;

enum Proposition {
    TOk(Term),
    TEq(Term, Term),
}

/// Projects the appropriate expression of an action
enum ActionProof {
    APExprOK,
    APExprEq,
    APLetOK,
    APLetAct(Rc<ActionProof>),
    APUnionOk1,
    APUnionOk2,
    APUnion,
    APSeq1(Rc<ActionProof>),
    APSeq2(Rc<ActionProof>),
}

// todo how to ignore this warning?
#[warn(clippy::enum_variant_names)]
enum ProofTerm {
    /// proves a Proposition based on a rule application
    /// the subsitution gives the mapping from variables to terms
    /// the body_pfs gives proofs for each of the conditions in the query of the rule
    /// the act_pf gives a location in the action of the proposition
    PRule {
        rule_name: Symbol,
        subst: Substitution,
        body_pfs: Vec<ProofId>,
        act_pf: ActionProof,
        result: Proposition,
    },
    /// A term is equal to itself- proves the proposition t = t
    PRefl {
        t_ok_pf: ProofId,
        t: Term,
    },
    /// The symmetric equality of eq_pf
    PSym {
        eq_pf: ProofId,
    },
    PTrans {
        pfxy: ProofId,
        pfyz: ProofId,
    },
    /// get a proof for the child of a term given a proof of a term
    PProj {
        pf_f_args_ok: ProofId,
        arg_idx: u32,
    },
    /// Proves f(x1, y1, ...) = f(x2, y2, ...) where f is fun_sym
    /// A proof via congruence- one proof for each child of the term
    /// pf_f_args_ok is a proof that the term with the lhs children is valid
    ///
    PCong {
        pf_args_eq: Vec<ProofId>,
        pf_f_args_ok: ProofId,
        fun_sym: Symbol,
    },
}
