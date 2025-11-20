//! A proof format for egglog programs, based on the Rocq format and checker from Tia Vu, Ryan
//! Doegens, and Oliver Flatt.
use std::{hash::Hash, io, rc::Rc, sync::Arc};

use indexmap::IndexSet;

use crate::ColumnTy;

define_id!(pub TermProofId, u32, "an id identifying proofs of terms within a [`ProofStore`]");
define_id!(pub EqProofId, u32, "an id identifying proofs of equality between two terms within a [`ProofStore`]");

#[derive(Clone, Debug)]
struct HashCons<K, T> {
    data: IndexSet<T>,
    _marker: std::marker::PhantomData<K>,
}

impl<K, T> Default for HashCons<K, T> {
    fn default() -> Self {
        HashCons {
            data: IndexSet::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K: NumericId, T: Clone + Eq + Hash> HashCons<K, T> {
    fn get_or_insert(&mut self, value: &T) -> K {
        if let Some((index, _)) = self.data.get_full(value) {
            K::from_usize(index)
        } else {
            let id = K::from_usize(self.data.len());
            self.data.insert(value.clone());
            id
        }
    }

    fn lookup(&self, id: K) -> Option<&T> {
        self.data.get_index(id.index())
    }
}

/// A hash-cons store for proofs and terms related to an egglog program.
#[derive(Clone, Default)]
pub struct ProofStore {
    eq_memo: HashCons<EqProofId, EqProof>,
    term_memo: HashCons<TermProofId, TermProof>,
    pub(crate) termdag: TermDag,
}

impl ProofStore {
    /// Create a new ProofStore
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a term proof by ID
    pub fn term_proof(&self, id: TermProofId) -> Option<&TermProof> {
        self.term_memo.lookup(id)
    }

    /// Get the termdag
    pub fn termdag(&self) -> &TermDag {
        &self.termdag
    }

    /// Get the termdag mutably
    pub fn termdag_mut(&mut self) -> &mut TermDag {
        &mut self.termdag
    }

    /// Get an equality proof by ID
    pub fn eq_proof(&self, id: EqProofId) -> Option<&EqProof> {
        self.eq_memo.lookup(id)
    }

    /// Get the children of a term by its ID
    pub fn get_term_children(&self, id: TermId) -> Vec<TermId> {
        self.termdag.get_children(id)
    }

    /// Create a literal term in the termdag
    pub fn make_lit(&mut self, lit: egglog_ast::generic_ast::Literal) -> TermId {
        self.termdag.lit_id(lit)
    }

    /// Create an application term in the termdag
    pub fn make_app(&mut self, func: String, args: Vec<TermId>) -> TermId {
        self.termdag.app_id(func, args)
    }

    /// Intern a term proof and get its ID
    pub fn add_term_proof(&mut self, proof: &TermProof) -> TermProofId {
        self.intern_term(proof)
    }

    /// Intern an equality proof and get its ID
    pub fn add_eq_proof(&mut self, proof: &EqProof) -> EqProofId {
        self.intern_eq(proof)
    }

    /// Print a term proof with pretty-printing configuration.
    pub fn print_term_proof_pretty(
        &self,
        term_pf: TermProofId,
        config: &PrettyPrintConfig,
        writer: &mut impl io::Write,
    ) -> io::Result<()> {
        let mut printer = PrettyPrinter::new(writer, config);
        self.print_term_proof_with_printer(term_pf, &mut printer)
    }

    /// Print an equality proof with pretty-printing configuration.
    pub fn print_eq_proof_pretty(
        &self,
        eq_pf: EqProofId,
        config: &PrettyPrintConfig,
        writer: &mut impl io::Write,
    ) -> io::Result<()> {
        let mut printer = PrettyPrinter::new(writer, config);
        self.print_eq_proof_with_printer(eq_pf, &mut printer)
    }

    fn print_cong_with_printer<W: io::Write>(
        &self,
        cong_pf: &CongProof,
        printer: &mut PrettyPrinter<W>,
    ) -> io::Result<()> {
        let CongProof {
            pf_args_eq,
            pf_f_args_ok,
            old_term,
            new_term,
            func,
        } = cong_pf;
        printer.write_str(&format!("Cong({func}, "))?;
        self.print_term(*old_term, printer)?;
        printer.write_str(" => ")?;
        self.print_term(*new_term, printer)?;
        printer.write_str(" by (")?;
        printer.increase_indent();
        for (i, pf) in pf_args_eq.iter().enumerate() {
            if i > 0 {
                printer.write_str(",")?;
            }
            printer.write_with_break(" ")?;
            self.print_eq_proof_with_printer(*pf, printer)?;
        }
        printer.write_with_break(") , old term exists by: ")?;
        printer.increase_indent();
        printer.newline()?;
        self.print_term_proof_with_printer(*pf_f_args_ok, printer)?;
        printer.decrease_indent();
        printer.write_str(")")?;
        printer.decrease_indent();
        Ok(())
    }

    /// Print the equality proof in a human-readable format to the given writer.
    pub fn print_eq_proof(&self, eq_pf: EqProofId, writer: &mut impl io::Write) -> io::Result<()> {
        self.print_eq_proof_pretty(eq_pf, &PrettyPrintConfig::default(), writer)
    }

    fn print_eq_proof_with_printer<W: io::Write>(
        &self,
        eq_pf: EqProofId,
        printer: &mut PrettyPrinter<W>,
    ) -> io::Result<()> {
        let eq_pf = self.eq_memo.lookup(eq_pf).unwrap();
        match eq_pf {
            EqProof::PRule {
                rule_name,
                subst,
                body_pfs,
                result_lhs,
                result_rhs,
            } => {
                printer.write_str(&format!("PRule[Equality]({rule_name:?}, Subst {{"))?;
                printer.increase_indent();
                printer.newline()?;
                for (i, binding) in subst.iter().enumerate() {
                    if i > 0 {
                        printer.write_str(",")?;
                    }
                    printer.write_with_break(" ")?;
                    printer.write_str(&format!("{}: {:?} => ", binding.name, binding.ty))?;
                    self.print_term(binding.term, printer)?;
                    printer.newline()?;
                }
                printer.newline()?;
                printer.write_with_break("},")?;
                printer.newline()?;
                printer.write_with_break("Body Pfs: [")?;
                printer.increase_indent();
                for (i, pf) in body_pfs.iter().enumerate() {
                    if i > 0 {
                        printer.write_str(",")?;
                    }
                    printer.write_with_break(" ")?;
                    match pf {
                        Premise::TermOk(term_pf) => {
                            printer.write_str("TermOk(")?;
                            self.print_term_proof_with_printer(*term_pf, printer)?;
                            printer.write_str(")")?;
                        }
                        Premise::Eq(eq_pf) => {
                            printer.write_str("Eq(")?;
                            self.print_eq_proof_with_printer(*eq_pf, printer)?;
                            printer.write_str(")")?;
                        }
                    }
                }
                printer.decrease_indent();
                printer.write_with_break("], ")?;
                printer.newline()?;
                printer.write_with_break(" Result: ")?;
                self.print_term(*result_lhs, printer)?;
                printer.write_str(" = ")?;
                self.print_term(*result_rhs, printer)?;
                printer.write_str(")")?;
                printer.decrease_indent();
            }
            EqProof::PRefl { t_ok_pf, t } => {
                printer.write_str("PRefl(")?;
                self.print_term_proof_with_printer(*t_ok_pf, printer)?;
                printer.write_str(", (term= ")?;
                self.termdag
                    .print_term_with_printer(self.termdag.get(*t), printer)?;
                printer.write_str("))")?
            }
            EqProof::PSym { eq_pf } => {
                printer.write_str("PSym(")?;
                self.print_eq_proof_with_printer(*eq_pf, printer)?;
                printer.write_str(")")?
            }
            EqProof::PTrans { pfxy, pfyz } => {
                printer.write_str("PTrans(")?;
                printer.increase_indent();
                printer.increase_indent();
                printer.newline()?;
                self.print_eq_proof_with_printer(*pfxy, printer)?;
                printer.decrease_indent();
                printer.newline()?;
                printer.write_with_break(" ... and then ... ")?;
                printer.increase_indent();
                printer.newline()?;
                self.print_eq_proof_with_printer(*pfyz, printer)?;
                printer.decrease_indent();
                printer.decrease_indent();
                printer.newline()?;
                printer.write_str(")")?
            }
            EqProof::PCong(cong_pf) => {
                printer.write_str("PCong[Equality](")?;
                self.print_cong_with_printer(cong_pf, printer)?;
                printer.write_str(")")?
            }
            EqProof::PFiat { desc, lhs, rhs } => {
                printer.write_str(&format!("PFiat({desc:?}, "))?;
                self.print_term(*lhs, printer)?;
                printer.write_str(" = ")?;
                self.print_term(*rhs, printer)?;
                printer.write_str(")")?
            }
        }
        printer.newline()
    }

    /// Print the term proof in a human-readable format to the given writer.
    pub fn print_term_proof(
        &self,
        term_pf: TermProofId,
        writer: &mut impl io::Write,
    ) -> io::Result<()> {
        self.print_term_proof_pretty(term_pf, &PrettyPrintConfig::default(), writer)
    }

    fn print_term_proof_with_printer<W: io::Write>(
        &self,
        term_pf: TermProofId,
        printer: &mut PrettyPrinter<W>,
    ) -> io::Result<()> {
        let term_pf = self.term_memo.lookup(term_pf).unwrap();
        match term_pf {
            TermProof::PRule {
                rule_name,
                subst,
                body_pfs,
                result,
            } => {
                printer.write_str(&format!("PRule[Existence]({rule_name:?}, Subst {{"))?;
                printer.increase_indent();
                printer.newline()?;
                for (i, binding) in subst.iter().enumerate() {
                    if i > 0 {
                        printer.write_str(",")?;
                    }
                    printer.write_with_break(" ")?;
                    printer.write_str(&format!("{}: {:?} => ", binding.name, binding.ty))?;
                    self.print_term(binding.term, printer)?;
                    printer.newline()?;
                }
                printer.newline()?;
                printer.write_with_break("},")?;
                printer.newline()?;
                printer.write_with_break("Body Pfs: [")?;
                printer.increase_indent();
                for (i, pf) in body_pfs.iter().enumerate() {
                    if i > 0 {
                        printer.write_str(",")?;
                    }
                    printer.write_with_break(" ")?;
                    match pf {
                        Premise::TermOk(term_pf) => {
                            printer.write_str("TermOk(")?;
                            self.print_term_proof_with_printer(*term_pf, printer)?;
                            printer.write_str(")")?;
                        }
                        Premise::Eq(eq_pf) => {
                            printer.write_str("Eq(")?;
                            self.print_eq_proof_with_printer(*eq_pf, printer)?;
                            printer.write_str(")")?;
                        }
                    }
                }
                printer.decrease_indent();
                printer.write_with_break("], Result: ")?;
                self.print_term(*result, printer)?;
                printer.write_str(")")
            }
            TermProof::PProj {
                pf_f_args_ok,
                arg_idx,
            } => {
                printer.write_str("PProj(")?;
                self.print_term_proof_with_printer(*pf_f_args_ok, printer)?;
                printer.write_str(&format!(", {arg_idx})"))
            }
            TermProof::PCong(cong_pf) => {
                printer.write_str("PCong[Exists](")?;
                self.print_cong_with_printer(cong_pf, printer)?;
                printer.write_str(")")
            }
            TermProof::PFiat { desc, term } => {
                printer.write_str(&format!("PFiat({desc:?}"))?;
                printer.write_str(", ")?;
                self.print_term(*term, printer)?;
                printer.write_str(")")
            }
        }
    }
    fn print_term<W: io::Write>(
        &self,
        term: TermId,
        printer: &mut PrettyPrinter<W>,
    ) -> io::Result<()> {
        self.termdag
            .print_term_with_printer(self.termdag.get(term), printer)
    }

    pub(crate) fn intern_term(&mut self, prf: &TermProof) -> TermProofId {
        self.term_memo.get_or_insert(prf)
    }
    pub(crate) fn intern_eq(&mut self, prf: &EqProof) -> EqProofId {
        self.eq_memo.get_or_insert(prf)
    }

    pub(crate) fn refl(&mut self, proof: TermProofId, term: TermId) -> EqProofId {
        self.intern_eq(&EqProof::PRefl {
            t_ok_pf: proof,
            t: term,
        })
    }

    pub(crate) fn sym(&mut self, proof: EqProofId) -> EqProofId {
        self.intern_eq(&EqProof::PSym { eq_pf: proof })
    }

    pub(crate) fn trans(&mut self, pfxy: EqProofId, pfyz: EqProofId) -> EqProofId {
        self.intern_eq(&EqProof::PTrans { pfxy, pfyz })
    }

    pub(crate) fn sequence_proofs(&mut self, pfs: &[EqProofId]) -> EqProofId {
        match pfs {
            [] => panic!("Cannot sequence an empty list of proofs"),
            [pf] => *pf,
            [pf1, rest @ ..] => {
                let mut cur = *pf1;
                for pf in rest {
                    cur = self.trans(cur, *pf);
                }
                cur
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Premise {
    TermOk(TermProofId),
    Eq(EqProofId),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CongProof {
    pub pf_args_eq: Vec<EqProofId>,
    pub pf_f_args_ok: TermProofId,
    pub old_term: TermId,
    pub new_term: TermId,
    pub func: Arc<str>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RuleVarBinding {
    pub name: Arc<str>,
    pub ty: ColumnTy,
    pub term: TermId,
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TermProof {
    /// proves a Proposition based on a rule application
    /// the subsitution gives the mapping from variables to terms
    /// the body_pfs gives proofs for each of the conditions in the query of the rule
    /// the act_pf gives a location in the action of the proposition
    PRule {
        rule_name: Rc<str>,
        subst: Vec<RuleVarBinding>,
        body_pfs: Vec<Premise>,
        result: TermId,
    },
    /// get a proof for the child of a term given a proof of a term
    PProj {
        pf_f_args_ok: TermProofId,
        arg_idx: usize,
    },
    PCong(CongProof),
    PFiat {
        desc: Rc<str>,
        term: TermId,
    },
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum EqProof {
    /// proves a Proposition based on a rule application
    /// the subsitution gives the mapping from variables to terms
    /// the body_pfs gives proofs for each of the conditions in the query of the rule
    /// the act_pf gives a location in the action of the proposition
    PRule {
        rule_name: Rc<str>,
        subst: Vec<RuleVarBinding>,
        body_pfs: Vec<Premise>,
        result_lhs: TermId,
        result_rhs: TermId,
    },
    /// A term is equal to itself- proves the proposition t = t
    PRefl {
        t_ok_pf: TermProofId,
        t: TermId,
    },
    /// The symmetric equality of eq_pf
    PSym {
        eq_pf: EqProofId,
    },
    PTrans {
        pfxy: EqProofId,
        pfyz: EqProofId,
    },
    /// Proves f(x1, y1, ...) = f(x2, y2, ...) where f is fun_sym
    /// A proof via congruence- one proof for each child of the term
    /// pf_f_args_ok is a proof that the term with the lhs children is valid
    PCong(CongProof),
    /// A fiat proof (assumed without further justification)
    PFiat {
        desc: Rc<str>,
        lhs: TermId,
        rhs: TermId,
    },
}
