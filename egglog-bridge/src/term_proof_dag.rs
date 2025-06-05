use anyhow::Context;
use core_relations::{PrimitiveId, PrimitivePrinter, Value};
use thiserror::Error;

use std::{
    collections::BTreeMap,
    fmt,
    io::{self, Write},
    rc::Rc,
    sync::Arc,
};

use hashbrown::{hash_map::Entry, HashMap};

use crate::{
    rule::Variable,
    syntax::{Binding, Entry as SyntaxEntry, Statement, TermFragment},
    ColumnTy, EGraph, FunctionId, Result, RuleId,
};

#[derive(Debug)]
pub enum TermValue<Prf> {
    Prim(PrimitiveConstant),
    SubTerm(Rc<Prf>),
}

impl<Prf> TermValue<Prf> {
    fn get_subterm(&self) -> Option<&Prf> {
        match self {
            TermValue::Prim(_) => None,
            TermValue::SubTerm(subterm) => Some(subterm),
        }
    }
}

impl<Prf> Clone for TermValue<Prf> {
    fn clone(&self) -> Self {
        match self {
            TermValue::Prim(p) => TermValue::Prim(p.clone()),
            TermValue::SubTerm(p) => TermValue::SubTerm(p.clone()),
        }
    }
}

#[derive(Debug)]
pub struct EqProof {
    pub lhs: Rc<TermProof>,
    pub rhs: Rc<TermProof>,
    pub reason: EqReason,
}

#[derive(Debug)]
pub enum EqReason {
    /// The (trivial) proof that a row equals itself.
    Id(Rc<TermProof>),
    /// An explanation of the existence of a row in the union-find.
    Base(Rc<TermProof>),
    /// A proof that `x = y` by way of `y = x`.
    Backwards(Rc<EqProof>),
    /// A proof that `x = z` by way of `x = y`, `y = z` (for any number of
    /// intermediate `y`s).
    Trans(Vec<Rc<EqProof>>),
}

pub enum TermProof {
    /// A proof that a term `r'` exists because:
    /// * Another term `r` exists, and
    /// * Each argument in `r` is equal to `r'`.
    Cong {
        func_id: FunctionId,
        func: Arc<str>,
        old_term: Rc<TermProof>,
        pairwise_eq: Vec<TermValue<EqProof>>,
    },
    /// The base case of a proof. Terms that were added as base values to the
    /// database.
    Fiat {
        desc: Arc<str>,
        func: Arc<str>,
        func_id: FunctionId,
        row: Vec<TermValue<TermProof>>,
    },

    /// A proof of the existence of a term by applying a rule to the databse.
    FromRule {
        rule_id: RuleId,
        lhs_atoms: usize,
        rhs_atoms: usize,
        rule_desc: Arc<str>,
        atom_desc: Rc<str>,
        func: Arc<str>,
        // NB: "none" means that this is a non-function, like "union".
        func_id: RuleTarget,
        row: Vec<TermValue<TermProof>>,
        premises: Vec<Rc<TermProof>>,
        premises_eq: Vec<Rc<EqProof>>,
    },
}

impl fmt::Debug for TermProof {
    /// We have a custom impl of fmt::Debug for TermProof as the default
    /// implementaiton can easily cause exponential blowup by printing out the
    /// tree for a DAG proof with lots of redundancy.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermProof::Cong { func, .. } => write!(f, "(cong {func} ...)"),
            TermProof::Fiat { desc, func, .. } => write!(f, "(fiat {desc} ({func} ... ))"),
            TermProof::FromRule {
                func,
                rule_desc,
                rule_id,
                ..
            } => {
                write!(f, "({rule_id:?} / {rule_desc} => ({func} ...))")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuleTarget {
    Union,
    Row(FunctionId, Rc<TermFragment<Variable>>),
}

impl RuleTarget {
    fn func_id(&self) -> FunctionId {
        let RuleTarget::Row(id, _) = self else {
            panic!("attempting to get func_id from non-row target")
        };
        *id
    }
}

impl TermProof {
    pub fn dump_explanation(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut printer = Printer::default();
        printer.print_term(self, "", writer)
    }
}

#[derive(Default)]
struct Printer {
    ids: HashMap<usize, usize>,
}

impl Printer {
    /// Get the id associated with the given pointer, creating a new one if one
    /// does not exist.
    ///
    /// The second value in the tuple is `true` if a new id was created.
    fn get_id<T>(&mut self, node: &T) -> (usize, bool) {
        let len = self.ids.len();
        match self.ids.entry(node as *const T as *const () as usize) {
            Entry::Occupied(o) => (*o.get(), false),
            Entry::Vacant(v) => (*v.insert(len), true),
        }
    }

    fn get_term_id(&mut self, term: &TermProof, writer: &mut impl Write) -> io::Result<String> {
        let (id, is_new) = self.get_id(term);
        if is_new {
            self.print_term(term, &format!("let t{id} = "), writer)?;
        }
        Ok(format!("t{id}"))
    }

    fn get_eq_id(&mut self, eq: &EqProof, writer: &mut impl Write) -> io::Result<String> {
        let (id, is_new) = self.get_id(eq);
        if is_new {
            self.print_eq(eq, &format!("let e{id} = "), writer)?;
        }
        Ok(format!("e{id}"))
    }

    fn print_term(
        &mut self,
        term: &TermProof,
        prefix: &str,
        writer: &mut impl Write,
    ) -> io::Result<()> {
        match term {
            TermProof::Cong {
                func,
                old_term,
                pairwise_eq,
                ..
            } => {
                let old_term = self.get_term_id(old_term.as_ref(), writer)?;
                let eq_subproofs = DisplayList(
                    try_collect(pairwise_eq.iter().map(|t| match t {
                        TermValue::Prim(s) => Ok(format!("{s}")),
                        TermValue::SubTerm(subterm) => self.get_eq_id(subterm.as_ref(), writer),
                    }))?,
                    " ",
                );
                writeln!(
                    writer,
                    "{prefix}Cong {{ {old_term} => [{func} {eq_subproofs}] }}"
                )?;
            }
            TermProof::Fiat {
                desc, func, row, ..
            } => {
                let term = DisplayList(
                    try_collect(row.iter().map(|t| match t {
                        TermValue::Prim(s) => Ok(format!("{s}")),
                        TermValue::SubTerm(subterm) => self.get_term_id(subterm.as_ref(), writer),
                    }))?,
                    " ",
                );
                writeln!(writer, "{prefix}Fiat {{ {desc}, ({func} {term}) }}")?;
            }
            TermProof::FromRule {
                rule_desc,
                atom_desc,
                func,
                row,
                premises,
                premises_eq,
                ..
            } => {
                let premises = DisplayList(
                    try_collect(
                        premises
                            .iter()
                            .map(|p| self.get_term_id(p.as_ref(), writer)),
                    )?,
                    " ",
                );
                let premises_eq = DisplayList(
                    try_collect(
                        premises_eq
                            .iter()
                            .map(|p| self.get_eq_id(p.as_ref(), writer)),
                    )?,
                    " ",
                );
                let row = DisplayList(
                    try_collect(row.iter().map(|t| match t {
                        TermValue::Prim(s) => Ok(format!("{s}")),
                        TermValue::SubTerm(subterm) => self.get_term_id(subterm.as_ref(), writer),
                    }))?,
                    " ",
                );
                writeln!(
                    writer,
                    "{prefix}FromRule {{\n\trule: {rule_desc}\n\tatom: {atom_desc}\n\t({func} {row})\n\tpremises: {premises}\n\tpremises_eq: {premises_eq}\n}}"
                )?;
            }
        }
        Ok(())
    }

    fn print_eq(
        &mut self,
        EqProof { lhs, rhs, reason }: &EqProof,
        prefix: &str,
        writer: &mut impl Write,
    ) -> io::Result<()> {
        match reason {
            EqReason::Id(row) => {
                let id = self.get_term_id(row.as_ref(), writer)?;
                writeln!(writer, "{prefix}Id {{ {id} }}")?;
            }
            EqReason::Base(b) => {
                let lhs = self.get_term_id(lhs, writer)?;
                let rhs = self.get_term_id(rhs, writer)?;
                let id = self.get_term_id(b.as_ref(), writer)?;
                writeln!(writer, "{prefix}Union-from-rule {{ {id} }} ({lhs} = {rhs})")?;
            }
            EqReason::Backwards(b) => {
                let lhs = self.get_term_id(lhs, writer)?;
                let rhs = self.get_term_id(rhs, writer)?;
                let id = self.get_eq_id(b.as_ref(), writer)?;
                writeln!(writer, "{prefix}Backwards {{ {id} }} ({lhs} = {rhs})")?;
            }
            EqReason::Trans(eqs) => {
                let lhs = self.get_term_id(lhs, writer)?;
                let rhs = self.get_term_id(rhs, writer)?;
                let eqs = DisplayList(
                    try_collect(eqs.iter().map(|e| self.get_eq_id(e.as_ref(), writer)))?,
                    " -> ",
                );
                writeln!(writer, "{prefix}Transitivity {{ {eqs} }} ({lhs} = {rhs})")?;
            }
        }
        Ok(())
    }
}

fn try_collect<T, E, I>(iter: I) -> std::result::Result<Vec<T>, E>
where
    I: Iterator<Item = std::result::Result<T, E>>,
{
    iter.collect()
}

/// A basic helper for display-formatting lists of items.
pub(crate) struct DisplayList<T>(pub Vec<T>, pub &'static str);

impl<T: std::fmt::Display> std::fmt::Display for DisplayList<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.0.iter();
        if let Some(first) = iter.next() {
            write!(f, "{first}")?;
            for item in iter {
                write!(f, "{}{item}", self.1)?;
            }
        }
        Ok(())
    }
}

struct ByPtr<T>(Rc<T>);

impl std::cmp::PartialEq for ByPtr<TermProof> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl std::cmp::Eq for ByPtr<TermProof> {}

impl std::hash::Hash for ByPtr<TermProof> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state)
    }
}

#[derive(Default)]
pub(crate) struct TermEnv {
    terms: HashMap<ByPtr<TermProof>, Rc<Term>>,
    check_cache: HashMap<usize, CacheState>,
}

enum CacheState {
    Marked,
    Checked,
}

impl TermEnv {
    /// Get the underlying term for a given proof.
    pub fn get_term(&mut self, term: Rc<TermProof>) -> Rc<Term> {
        let by_ptr = ByPtr(term);
        if let Some(term) = self.terms.get(&by_ptr) {
            return term.clone();
        }
        let term = match by_ptr.0.as_ref() {
            TermProof::Cong {
                func,
                pairwise_eq,
                func_id,
                ..
            } => {
                let new_subterms = Vec::from_iter(pairwise_eq.iter().map(|t| match t {
                    TermValue::Prim(p) => Rc::new(Term::Prim(p.clone())),
                    TermValue::SubTerm(eq) => self.get_term(eq.rhs.clone()),
                }));
                Term::Expr {
                    func_id: *func_id,
                    func: func.clone(),
                    subterms: new_subterms,
                }
            }
            TermProof::Fiat {
                func_id, func, row, ..
            } => Term::Expr {
                func_id: *func_id,
                func: func.clone(),
                subterms: row
                    .iter()
                    .map(|t| match t {
                        TermValue::Prim(p) => Rc::new(Term::Prim(p.clone())),
                        TermValue::SubTerm(rc) => self.get_term(rc.clone()),
                    })
                    .collect(),
            },
            TermProof::FromRule {
                func_id, func, row, ..
            } => Term::Expr {
                func_id: func_id.func_id(),
                func: func.clone(),
                subterms: row
                    .iter()
                    .map(|t| match t {
                        TermValue::Prim(p) => Rc::new(Term::Prim(p.clone())),
                        TermValue::SubTerm(rc) => self.get_term(rc.clone()),
                    })
                    .collect(),
            },
        };
        let res = Rc::new(term);
        self.terms.insert(by_ptr, res.clone());
        res
    }

    fn get_term_from_val(&mut self, tv: &TermValue<TermProof>) -> Rc<Term> {
        match tv {
            TermValue::Prim(p) => Rc::new(Term::Prim(p.clone())),
            TermValue::SubTerm(rc) => self.get_term(rc.clone()),
        }
    }

    fn start_check<T>(&mut self, elt: &T) -> Result<bool> {
        let num = elt as *const T as *const () as usize;
        match self.check_cache.entry(num) {
            Entry::Occupied(o) => {
                if let CacheState::Marked = o.get() {
                    return Err(ProofCheckError::CyclicDependency.into());
                }
                Ok(true)
            }
            Entry::Vacant(v) => {
                v.insert(CacheState::Marked);
                Ok(false)
            }
        }
    }

    fn finish_check<T>(&mut self, elt: &T) {
        let num = elt as *const T as *const () as usize;
        self.check_cache.insert(num, CacheState::Checked);
    }
}

/// A primitive constant in a term.
#[derive(Clone, Eq, PartialEq, Debug, PartialOrd, Ord)]
pub struct PrimitiveConstant {
    /// The type of primitive.
    pub(crate) ty: PrimitiveId,
    /// The underlying (interned) value of the primitive, to be looked up in the
    /// primitives associated with an egraph.
    pub(crate) interned: Value,
    /// The string representation of the primitive.
    pub(crate) rendered: Arc<str>,
}

impl fmt::Display for PrimitiveConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rendered)
    }
}

/// A pointer-based representation of a term.
///
/// This is mostly used in printing routines. It is a less efficient
/// representation than something like Egg's RecExpr.
#[derive(Debug, PartialEq, Eq)]
// NB: my read of the standard library is that `Rc::eq` uses `ptr_eq` to
// short-circuit. When there is a lot of sharing between two terms, checking
// equality should be pretty quick.
pub enum Term {
    Prim(PrimitiveConstant),
    Expr {
        func_id: FunctionId,
        func: Arc<str>,
        subterms: Vec<Rc<Term>>,
    },
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Prim(c) => write!(f, "{c}"),
            Term::Expr { func, subterms, .. } => {
                write!(f, "({func}")?;
                for subterm in subterms {
                    write!(f, " {subterm}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl EGraph {
    /// Check the validity of the given term proof.
    ///
    // The need to pass an `Rc` here is a bit annoying. It's downstream of our
    // memoization strategy, which holds onto references for keys (via ByPtr
    // instead of *const or usize) for added safety when recyclic term
    // pointers.
    pub fn check_term_proof(&mut self, proof: Rc<TermProof>) -> Result<()> {
        self.check_term_proof_impl(proof, &mut Default::default())
    }

    /// Check the validity of the given equality proof.
    pub fn check_eq_proof(&mut self, proof: &EqProof) -> Result<()> {
        self.check_eq_proof_impl(proof, &mut Default::default())
    }

    fn check_term_proof_impl(
        &mut self,
        proof: Rc<TermProof>,
        term_env: &mut TermEnv,
    ) -> Result<()> {
        if term_env.start_check(&proof)? {
            return Ok(());
        }
        match proof.as_ref() {
            TermProof::Cong {
                old_term,
                pairwise_eq,
                ..
            } => {
                self.check_term_proof_impl(old_term.clone(), term_env)?;
                pairwise_eq
                    .iter()
                    .filter_map(TermValue::get_subterm)
                    .try_for_each(|eq| self.check_eq_proof_impl(eq, term_env))
            }
            TermProof::Fiat { .. } => {
                // NB: we should allow users to validate any instances of
                // `Fiat`; otherwise it's a way to cheat.
                Ok(())
            }
            TermProof::FromRule {
                rule_id,
                func,
                func_id,
                row,
                premises_eq,
                premises,
                lhs_atoms,
                rhs_atoms,
                ..
            } => {
                // First: Check the eq proofs and auxiliary proofs.
                // NB: do we not need premises anymore? premises_eq seems to
                // have everything we need.
                premises_eq
                    .iter()
                    .try_for_each(|eq| self.check_eq_proof_impl(eq, term_env))?;
                let rule_info = &self.rules[*rule_id];
                let syntax = rule_info.syntax.clone();
                let mut sub = Substitution {
                    mapping: Default::default(),
                    egraph: self,
                };

                // Contains a mapping from the canonical representative of a
                // variable back to the terms that we proved equal to it in the
                // grounding process. See the `RuleTarget::Union` branch below
                // for what this is about.
                let mut reverse_canon_mapping = BTreeMap::<Rc<Term>, Vec<Rc<Term>>>::new();
                // Process the LHS and RHS atoms
                for (binding, prf) in syntax.lhs_bindings[0..*lhs_atoms]
                    .iter()
                    .chain(&syntax.rhs_bindings[0..*rhs_atoms])
                    .zip(premises_eq.iter())
                {
                    let lhs_term = term_env.get_term(prf.lhs.clone());
                    let rhs_term = term_env.get_term(prf.rhs.clone());
                    reverse_canon_mapping
                        .entry(lhs_term.clone())
                        .or_default()
                        .push(rhs_term.clone());
                    sub.process_binding(&lhs_term, &rhs_term, binding)
                        .with_context(|| {
                            anyhow::format_err!(
                                "rule_id={rule_id:?}\nlhs-bindings={lbdgs:?}\nrhs-bindings={rbdgs:?}\ncur={binding:?}\nsub=\n\t{sub}: ",
                                lbdgs=&syntax.lhs_bindings[0..*lhs_atoms],
                                rbdgs=&syntax.rhs_bindings[0..*rhs_atoms],
                                sub=DisplayList(
                                    sub.mapping
                                        .iter()
                                        .map(|(var, term)| format!("{var:?} => {term}"))
                                        .collect(),
                                    "\n\t"
                                )
                            )
                        })?;
                }

                // Now we're ready to try reconstructing the term. The next step
                // here depends on what kind of insertion we're doing.
                match func_id {
                    RuleTarget::Union => {
                        let mut found = false;
                        let expected_l = term_env.get_term_from_val(&row[0]);
                        let expected_r = term_env.get_term_from_val(&row[1]);
                        for stmt in &syntax.statements {
                            sub.run_stmt(stmt, |l, r| {
                                // The union-find proofs behave a bit like row
                                // insertions into the database (hence why much
                                // of the preceeding code was shared).
                                //
                                // There is a difference, though. UF proofs
                                // operate on term ids rather canonical ids. As
                                // such, we need to do a search over the terms
                                // bound to a variable when checking the LHS and
                                // RHS of the union.
                                let Some(other_l) = reverse_canon_mapping.get(l) else {
                                    return;
                                };
                                let Some(other_r) = reverse_canon_mapping.get(r) else {
                                    return;
                                };
                                for got_l in other_l {
                                    for got_r in other_r {
                                        found |= got_l == &expected_l && got_r == &expected_r;
                                    }
                                }
                            })?;
                        }
                        if !found {
                            return Err(anyhow::Error::from(ProofCheckError::TermsNotUnioned {
                                lhs: format!("{expected_l}"),
                                rhs: format!("{expected_r}"),
                            }))
                            .with_context(|| {
                                // Big error here for now. This is where we can have a lot of trouble.
                                anyhow::format_err!(
                                    "rule_id={rule_id:?}, row=({func} {})\n\t=premises=\n\t{}\n\t=eqs=\n\t{}\n\tsubst=\n\t{}",
                                    DisplayList(
                                        row.iter()
                                            .map(|x| term_env.get_term_from_val(x).to_string())
                                            .collect(),
                                        " "
                                    ),
                                    DisplayList(
                                        premises
                                            .iter()
                                            .map(|x| term_env.get_term(x.clone()).to_string())
                                            .collect(),
                                        "\n\t"
                                    ),
                                    DisplayList(
                                        premises_eq
                                            .iter()
                                            .map(|x| {
                                                format!(
                                                    "{} = {}",
                                                    term_env.get_term(x.lhs.clone()),
                                                    term_env.get_term(x.rhs.clone())
                                                )
                                            })
                                            .collect(),
                                        "\n\t"
                                    ),
                                    DisplayList(
                                        sub.mapping
                                            .iter()
                                            .map(|(var, term)| format!("{var:?} => {term}"))
                                            .collect(),
                                        "\n\t"
                                    )
                                )
                            });
                        }
                    }
                    RuleTarget::Row(_, fragment) => {
                        let expected = term_env.get_term(proof.clone());
                        let mut found = false;
                        sub.construct_term(fragment, &mut |term| found |= term == &*expected)?;
                        if !found {
                            return Err(ProofCheckError::TermNotConstructed {
                                term: format!("{expected}"),
                            }
                            .into());
                        }
                    }
                }
                Ok(())
            }
        }?;
        term_env.finish_check(&proof);
        Ok(())
    }

    fn check_eq_proof_impl(&mut self, eq_proof: &EqProof, term_env: &mut TermEnv) -> Result<()> {
        if term_env.start_check(eq_proof)? {
            return Ok(());
        }
        let EqProof { lhs, rhs, reason } = eq_proof;
        match reason {
            EqReason::Id(term) => {
                self.check_term_proof_impl(term.clone(), term_env)?;
                let got = term_env.get_term(term.clone());
                let lhs = term_env.get_term(lhs.clone());
                let rhs = term_env.get_term(rhs.clone());
                if got != lhs || got != rhs || rhs != lhs {
                    return Err(ProofCheckError::IdProofTermDisagreement {
                        proof: format!("{got:?}"),
                        lhs: format!("{lhs:?}"),
                        rhs: format!("{rhs:?}"),
                    }
                    .into());
                }
            }
            EqReason::Base(proof) => {
                match proof.as_ref() {
                    TermProof::FromRule {
                        func_id: RuleTarget::Union,
                        row,
                        ..
                    } => {
                        // Ensure that the terms in the union are equal to the endpoints.
                        let l_expected = term_env.get_term_from_val(&row[0]);
                        let r_expected = term_env.get_term_from_val(&row[1]);
                        let lhs_term = term_env.get_term(lhs.clone());
                        let rhs_term = term_env.get_term(rhs.clone());
                        if l_expected != lhs_term || r_expected != rhs_term {
                            return Err(ProofCheckError::MisalignedUnion {
                                lhs_expected: format!("{l_expected}"),
                                lhs_actual: format!("{lhs_term}"),
                                rhs_expected: format!("{r_expected}"),
                                rhs_actual: format!("{rhs_term}"),
                            }
                            .into());
                        }
                        // Then check the underlying proof.
                        self.check_term_proof_impl(proof.clone(), term_env)?;
                    }
                    TermProof::Cong {
                        old_term,
                        pairwise_eq,
                        ..
                    } => {
                        // First, check all the pairwise equality proofs.
                        pairwise_eq.iter().try_for_each(|eq| {
                            let TermValue::SubTerm(eq) = eq else {
                                return Ok(());
                            };
                            self.check_eq_proof_impl(eq, term_env)
                        })?;
                        let old = term_env.get_term(old_term.clone());
                        let new = term_env.get_term(proof.clone());
                        let lhs = term_env.get_term(lhs.clone());
                        let rhs = term_env.get_term(rhs.clone());
                        if lhs != old || rhs != new {
                            return Err(anyhow::Error::from(ProofCheckError::MisalignedUnion {
                                lhs_expected: format!("{lhs}"),
                                lhs_actual: format!("{old}"),
                                rhs_expected: format!("{rhs}"),
                                rhs_actual: format!("{new}"),
                            }))
                            .with_context(|| {
                                anyhow::format_err!(
                                    "during congruence rule {proof:?}, {pairwise_eq:?}"
                                )
                            });
                        }
                    }
                    _ => {
                        return Err(ProofCheckError::NonUnionProofOfEquality {
                            rule: format!("{proof:?}"),
                        }
                        .into())
                    }
                }
            }
            EqReason::Backwards(eq) => {
                self.check_eq_proof_impl(eq, term_env)?;
                let outer_lhs = term_env.get_term(lhs.clone());
                let inner_lhs = term_env.get_term(eq.lhs.clone());
                let outer_rhs = term_env.get_term(rhs.clone());
                let inner_rhs = term_env.get_term(eq.rhs.clone());
                if outer_lhs != inner_rhs {
                    return Err(ProofCheckError::BackwardEndpointMismatch {
                        x: format!("{outer_lhs:?}"),
                        y: format!("{inner_rhs:?}"),
                    }
                    .into());
                }

                if outer_rhs != inner_lhs {
                    return Err(ProofCheckError::BackwardEndpointMismatch {
                        x: format!("{outer_rhs:?}"),
                        y: format!("{inner_lhs:?}"),
                    }
                    .into());
                }
            }
            EqReason::Trans(vec) => {
                // Check each component eq proof.
                vec.iter()
                    .try_for_each(|eq| self.check_eq_proof_impl(eq, term_env))?;
                for (x, y) in vec.iter().zip(vec.iter().skip(1)) {
                    let lhs_term = term_env.get_term(x.rhs.clone());
                    let rhs_term = term_env.get_term(y.lhs.clone());
                    if lhs_term != rhs_term {
                        return Err(anyhow::Error::from(
                            ProofCheckError::TransitiveEndpointMismatch {
                                x: format!("{lhs_term}"),
                                y: format!("{rhs_term}"),
                            },
                        ))
                        .with_context(|| anyhow::format_err!("{x:?} vs.\n{y:?}"));
                    }
                }
            }
        };
        term_env.finish_check(eq_proof);
        Ok(())
    }
}

struct Substitution<'a> {
    // TODO: we should be able to reuse the allocations here for really large proofs.
    mapping: HashMap<Variable, Rc<Term>>,
    egraph: &'a mut EGraph,
}

impl Substitution<'_> {
    fn process_binding(
        &mut self,
        canon: &Rc<Term>,
        term: &Rc<Term>,
        binding: &Binding,
    ) -> Result<()> {
        self.bind_variables_to_entry(canon, &SyntaxEntry::Placeholder(binding.var))?;
        self.bind_variables(term, &binding.syntax)
    }

    /// Find terms mapped to by the given TermFragment pattern, if possible.
    fn bind_variables(&mut self, term: &Rc<Term>, pat: &TermFragment<Variable>) -> Result<()> {
        match (term.as_ref(), pat) {
            (
                Term::Expr {
                    func_id,
                    func,
                    subterms,
                },
                TermFragment::App(syntax_func, vec),
            ) => {
                if func_id != syntax_func {
                    return Err(ProofCheckError::PatternFunctionMismatch {
                        func1: func.as_ref().into(),
                        func2: self.egraph.funcs[*syntax_func].name.as_ref().into(),
                    }
                    .into());
                }
                // Low-level invariants. If these assertiosn fail then something
                // has gone very wrong. If this is a salient error that comes up
                // a lot, we could upgrade it to a ProofCheckError.
                assert_eq!(subterms.len(), vec.len());
                assert_eq!(subterms.len(), self.egraph.funcs[*func_id].schema.len() - 1);
                subterms
                    .iter()
                    .zip(vec.iter())
                    .try_for_each(|(t, p)| self.bind_variables_to_entry(t, p))?;
            }

            _ => {
                return Err(ProofCheckError::PatternVariantMismatch {
                    pat: format!("{pat:?}"),
                    term: format!("{term}"),
                }
                .into())
            }
        };
        Ok(())
    }

    fn bind_variables_to_entry(
        &mut self,
        term: &Rc<Term>,
        entry: &SyntaxEntry<Variable>,
    ) -> Result<()> {
        match (term.as_ref(), entry) {
            (Term::Prim(p1), SyntaxEntry::Const(p2)) => {
                if p1 != p2 {
                    return Err(ProofCheckError::MismatchedConstants {
                        p1: p1.rendered.as_ref().into(),
                        p2: p2.rendered.as_ref().into(),
                    }
                    .into());
                }
            }
            (_, SyntaxEntry::Placeholder(var)) => {
                if let Some(prev) = self.mapping.insert(*var, term.clone()) {
                    if &prev != term {
                        return Err(ProofCheckError::UnificationFailure {
                            var: *var,
                            t1: format!("{prev}"),
                            t2: format!("{term}"),
                        }
                        .into());
                    }
                }
            }
            (_, SyntaxEntry::Const(..)) => {
                return Err(ProofCheckError::PatternVariantMismatch {
                    pat: format!("{entry:?}"),
                    term: format!("{term}"),
                }
                .into())
            }
        };
        Ok(())
    }

    fn run_stmt(
        &mut self,
        stmt: &Statement<Variable>,
        mut on_union: impl FnMut(&Term, &Term),
    ) -> Result<()> {
        match stmt {
            Statement::AssertEq(l, r) => {
                let l = self.construct_term_from_entry(l)?;
                let r = self.construct_term_from_entry(r)?;
                if l != r {
                    Err(ProofCheckError::AssertEqFailure {
                        lhs: format!("{l}"),
                        rhs: format!("{r}"),
                    }
                    .into())
                } else {
                    Ok(())
                }
            }
            Statement::Union(l, r) => {
                let l = self.construct_term_from_entry(l)?;
                let r = self.construct_term_from_entry(r)?;
                on_union(&l, &r);
                Ok(())
            }
        }
    }

    fn construct_term_from_entry(&mut self, entry: &SyntaxEntry<Variable>) -> Result<Rc<Term>> {
        match entry {
            SyntaxEntry::Placeholder(v) => self.lookup_var(*v),
            SyntaxEntry::Const(c) => Ok(Rc::new(Term::Prim(c.clone()))),
        }
    }

    fn construct_term(
        &mut self,
        rule: &TermFragment<Variable>,
        on_term: &mut impl FnMut(&Term),
    ) -> Result<Rc<Term>> {
        let result: Rc<Term> = match rule {
            TermFragment::Prim(func, args, ty) => {
                let ColumnTy::Primitive(ty) = *ty else {
                    panic!("expected primitive type, found {ty:?}");
                };
                // This is the hardest case but still fairly straight-forwad: we
                // need to extract primitives from `args`, then apply `func` to
                // them.
                let args = args
                    .iter()
                    .map(|p| {
                        let term = self.construct_term_from_entry(p)?;
                        let Term::Prim(pc) = term.as_ref() else {
                            return Err(ProofCheckError::NonPrimitiveArg {
                                arg: format!("{term}"),
                            }
                            .into());
                        };
                        Ok(pc.interned)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let result = self
                    .egraph
                    .db
                    .with_execution_state(|exec_state| exec_state.call_external_func(*func, &args))
                    // This should be a pretty rare error, but if we see it
                    // often we can upgrade it to a full ProofCheckError.
                    //
                    // When primitives return None, they intend to halt
                    // execution and not match a rule.
                    .expect("primitive functions should return a value");

                let rendered = format!(
                    "{:?}",
                    PrimitivePrinter {
                        prim: self.egraph.db.primitives_mut(),
                        ty,
                        val: result,
                    }
                );
                Rc::new(Term::Prim(PrimitiveConstant {
                    ty,
                    interned: result,
                    rendered: rendered.into(),
                }))
            }
            TermFragment::App(func, args) => {
                let args = args
                    .iter()
                    .map(|p| self.construct_term_from_entry(p))
                    .collect::<Result<Vec<_>>>()?;
                Rc::new(Term::Expr {
                    func_id: *func,
                    func: self.egraph.funcs[*func].name.clone(),
                    subterms: args.clone(),
                })
            }
        };
        on_term(&result);
        Ok(result)
    }

    fn lookup_var(&self, var: Variable) -> Result<Rc<Term>> {
        self.mapping
            .get(&var)
            .cloned()
            .ok_or(ProofCheckError::UnboundVariable { var }.into())
    }
}

// NB: why strings?
// `Error` needs to be Send, but Rc (which we use for deduping) is not. So we
// format the terms before returning an error.

#[derive(Debug, Error)]
pub(crate) enum ProofCheckError {
    #[error("Ran into a cyclic dependency while checking proof")]
    CyclicDependency,
    #[error("Mismatched endpoints in backward equality proof {x} != {y}")]
    BackwardEndpointMismatch { x: String, y: String },
    #[error("Mismatched endpoints in transitive equality proof: {x} != {y}")]
    TransitiveEndpointMismatch { x: String, y: String },
    #[error("Expected endpoints for an id proof don't match the term proof={proof}, lhs={lhs}, rhs={rhs}")]
    IdProofTermDisagreement {
        proof: String,
        lhs: String,
        rhs: String,
    },

    #[error("Failed to build a consistent substitution {var:?} used in both {t1} and {t2}")]
    UnificationFailure {
        var: Variable,
        t1: String,
        t2: String,
    },

    #[error("Found mismatched constants {p1} and {p2}")]
    MismatchedConstants { p1: String, p2: String },

    #[error("Found mismatched syntax variants {pat} vs. {term}")]
    PatternVariantMismatch { pat: String, term: String },

    #[error("Found mismatch between functions {func1} vs. {func2}")]
    PatternFunctionMismatch { func1: String, func2: String },

    #[error("Unbound variable {var:?}")]
    UnboundVariable { var: Variable },

    #[error("Expected terms {lhs} and {rhs} to be equal")]
    AssertEqFailure { lhs: String, rhs: String },

    #[error("Non-primitive argument to primitive function {arg}")]
    NonPrimitiveArg { arg: String },

    #[error("Failed to construct term {term}")]
    TermNotConstructed { term: String },

    #[error("Failed to prove equality {lhs} = {rhs}")]
    TermsNotUnioned { lhs: String, rhs: String },

    #[error("Proof of equality from a rule unioning {lhs_actual} and {rhs_actual} intended to prove {lhs_expected} = {rhs_expected}")]
    MisalignedUnion {
        lhs_expected: String,
        lhs_actual: String,
        rhs_expected: String,
        rhs_actual: String,
    },

    #[error(
        "Rule {rule} is being used as a proof of equality when `union` is not the  target function"
    )]
    NonUnionProofOfEquality { rule: String },
}

// Cleanups:
// * use named variants of FunctionId and Variable

// These custom impls here are to provide fairly fast map lookups on terms
// without implementing a full hashcons for them. This is a bit of a hack /
// half-measure: but this part of proof checking probably isn't going to be the
// most performance-sensitive either. Worth revisiting if we notice an issue.

impl PartialOrd for Term {
    fn le(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        match (self, other) {
            (Term::Prim(p1), Term::Prim(p2)) => p1 <= p2,
            (
                Term::Expr {
                    func_id: f1,
                    subterms: s1,
                    ..
                },
                Term::Expr {
                    func_id: f2,
                    subterms: s2,
                    ..
                },
            ) => (f1, s1) <= (f2, s2),
            (Term::Prim(_), Term::Expr { .. }) => true,
            (Term::Expr { .. }, Term::Prim(_)) => false,
        }
    }

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Term {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self == other {
            return std::cmp::Ordering::Equal;
        }
        match (self, other) {
            (Term::Prim(p1), Term::Prim(p2)) => p1.cmp(p2),
            (
                Term::Expr {
                    func_id: f1,
                    subterms: s1,
                    ..
                },
                Term::Expr {
                    func_id: f2,
                    subterms: s2,
                    ..
                },
            ) => (f1, s1).cmp(&(f2, s2)),
            (Term::Prim(_), Term::Expr { .. }) => std::cmp::Ordering::Less,
            (Term::Expr { .. }, Term::Prim(_)) => std::cmp::Ordering::Greater,
        }
    }
}
