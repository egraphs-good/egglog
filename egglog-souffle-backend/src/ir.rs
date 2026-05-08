//! IR for a Souffle program. Deliberately minimal — captures only what we
//! need to emit the souffle_compat-encoded form of an egglog program.

/// A complete Souffle program.
#[derive(Debug, Clone, Default)]
pub struct Program {
    /// Top-level pragmas (key, value).
    pub pragmas: Vec<(String, String)>,
    /// Type declarations (records / unions).
    pub types: Vec<TypeDecl>,
    /// Relation declarations.
    pub relations: Vec<RelationDecl>,
    /// Rules and facts.
    pub clauses: Vec<Clause>,
    /// `.printsize`, `.output`, `.limititerations`, `.snapshot`, etc.
    pub directives: Vec<Directive>,
}

/// A Souffle type. v0 supports atomic numeric types and recursive records.
#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: String,
    pub kind: TypeKind,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// `.type T = [field1: T1, field2: T2, ...]` — recursive record. Used to
    /// represent term IDs in our Skolemization-via-records scheme.
    Record(Vec<(String, String)>),
}

/// A relation declaration.
#[derive(Debug, Clone)]
pub struct RelationDecl {
    pub name: String,
    /// (column name, column type)
    pub columns: Vec<(String, String)>,
}

/// A clause: a fact or a rule.
#[derive(Debug, Clone)]
pub struct Clause {
    pub head: Atom,
    pub body: Vec<Literal>,
}

impl Clause {
    pub fn fact(head: Atom) -> Self {
        Self { head, body: vec![] }
    }
    pub fn rule(head: Atom, body: Vec<Literal>) -> Self {
        Self { head, body }
    }
}

/// An atom: relation name applied to a list of expressions.
#[derive(Debug, Clone)]
pub struct Atom {
    pub relation: String,
    pub args: Vec<Expr>,
}

/// Body literal: an atom, a binary constraint, or a built-in predicate.
#[derive(Debug, Clone)]
pub enum Literal {
    Atom(Atom),
    Constraint(BinaryOp, Expr, Expr),
    /// Negation: `!Atom`.
    Neg(Atom),
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// An expression in head/body. We keep this small — the encoded program's
/// expressions are mostly literals, variables, record constructors, and
/// `ord()` calls (to replace egglog's `ordering-min/max`).
#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    /// Wildcard `_`.
    Wildcard,
    Number(i64),
    Symbol(String),
    /// Nil — for record-typed positions where we want the empty record.
    Nil,
    /// `[e1, e2, ...]` — record constructor.
    Record(Vec<Expr>),
    /// `ord(e)` — Souffle built-in for ordinal of a record value.
    Ord(Box<Expr>),
}

/// Top-level directives that apply to a relation.
#[derive(Debug, Clone)]
pub enum Directive {
    PrintSize(String),
    Output { relation: String, params: Vec<(String, String)> },
    /// `.limititerations R(n=N)`
    LimitIterations { relation: String, n: u64 },
    /// `.snapshot R_snap(of = "R")`
    Snapshot { snap: String, source: String },
}
