//! Utilities for representing syntax.
//!
//! This crate provides a Rust-first API for constructing and running egglog
//! programs. Nevertheless, it's helpful at several points to be able to talk
//! about how the Rust API is being used in terms of an egglog-like language:
//!
//! * Syntax can be helpful in printing debug information about a rule.
//! * Syntax is needed on some level when representing proofs, particularly when
//!   those proofs need to be serialized to an external checker.

use std::{fmt, sync::Arc};

use crate::{
    rule::Variable, term_proof_dag::BaseValueConstant, ColumnTy, ExternalFunctionId, FunctionId,
};

/// A syntactic representation of a rule.
#[derive(Debug, Default, Clone)]
pub struct RuleRepresentation {
    /// Bindings on the left-hand side of a rule.
    pub lhs_bindings: Vec<Binding>,
    /// Bindings on the right-hand side of a rule.
    pub rhs_bindings: Vec<Binding>,
    /// Statements (unions and assertions) made on the right-hand side of a
    /// rule.
    pub statements: Vec<Statement<Variable>>,
}

/// A top-level binding on the left or right-hand side of a rule.
#[derive(Debug, Clone)]
pub(crate) struct Binding {
    /// The variable being equated with `syntax`.
    pub var: Variable,
    /// A syntactic egglog atom (e.g. the head of a term (F x_0 ... x_n) where
    /// x_i are variables and F is either an egglog function or a primitive).
    pub syntax: Arc<TermFragment<Variable>>,
}

/// An atomic entry in an atom.
#[derive(Clone)]
pub enum Entry<T> {
    /// Placeholders generally represent either a variable or a term substituted
    /// for that variable.
    Placeholder(T),
    /// A constant for some base value type.
    Const(BaseValueConstant),
}

#[derive(Clone)]
pub enum Statement<T> {
    /// A low-level assert-eq statement in the right-hand side of a rule. Egglog
    /// itself does not have this, but it is used to implement some egglog
    /// functionality (such as binding base value variables on the LHS of a
    /// rule).
    AssertEq(Entry<T>, Entry<T>),

    /// Merge the e-classes associated with these two entries.
    Union(Entry<T>, Entry<T>),
}

pub enum TermFragment<T> {
    /// Apply a given primitive function to the given arguments.
    Prim(ExternalFunctionId, Vec<Entry<T>>, ColumnTy),

    /// Apply the function to the given arguments.
    App(FunctionId, Vec<Entry<T>>),
}

impl<T: fmt::Debug> fmt::Debug for Entry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Placeholder(p) => write!(f, "{p:?}"),
            Entry::Const(constant) => write!(f, "{constant:?}"),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Statement<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::AssertEq(l, r) => write!(f, "(assert-eq {l:?} {r:?})"),
            Statement::Union(l, r) => write!(f, "(union {l:?} {r:?})"),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for TermFragment<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermFragment::Prim(p, args, _) => {
                write!(f, "({p:?} ")?;
                for (ix, arg) in args.iter().enumerate() {
                    if ix > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{arg:?}")?;
                }
                write!(f, ")")
            }
            TermFragment::App(func, args) => {
                write!(f, "({func:?} ")?;
                for (ix, arg) in args.iter().enumerate() {
                    if ix > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{arg:?}")?;
                }
                write!(f, ")")
            }
        }
    }
}
