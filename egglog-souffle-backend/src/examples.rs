//! Hand-built example programs in the IR. These are stand-ins for what the
//! eventual translator (egglog encoded form → IR) will produce. They serve
//! as integration tests for the `emit` step and the Souffle fork.

use crate::ir::*;

/// A tiny program demonstrating the buffer/canon/snapshot pattern.
///
/// Story: numbers and their canonical leaders. The "user" rule sees the
/// snapshot of canonical leaders and proposes them in a buffer. The
/// "rebuild" rule canonicalizes by making each buffered term its own leader.
///
/// This mirrors `experiments/souffle-compat/q7-snapshot.dl`.
pub fn buffer_canon_snap() -> Program {
    let mut p = Program::default();
    p.pragmas.push(("outer-saturate".into(), "5".into()));
    p.relations.push(RelationDecl {
        name: "Canonical".into(),
        columns: vec![("t".into(), "number".into()), ("leader".into(), "number".into())],
    });
    p.relations.push(RelationDecl {
        name: "CanonicalSnap".into(),
        columns: vec![("t".into(), "number".into()), ("leader".into(), "number".into())],
    });
    p.relations.push(RelationDecl {
        name: "Buffer".into(),
        columns: vec![("t".into(), "number".into())],
    });
    p.directives.push(Directive::Snapshot {
        snap: "CanonicalSnap".into(),
        source: "Canonical".into(),
    });

    // Initial state.
    for (t, l) in [(1, 1), (2, 1), (3, 1)] {
        p.clauses.push(Clause::fact(Atom {
            relation: "Canonical".into(),
            args: vec![Expr::Number(t), Expr::Number(l)],
        }));
    }

    // User rule: Buffer(l) :- CanonicalSnap(_, l).
    p.clauses.push(Clause::rule(
        Atom { relation: "Buffer".into(), args: vec![Expr::Var("l".into())] },
        vec![Literal::Atom(Atom {
            relation: "CanonicalSnap".into(),
            args: vec![Expr::Wildcard, Expr::Var("l".into())],
        })],
    ));

    // Rebuild rule: Canonical(t, t) :- Buffer(t).
    p.clauses.push(Clause::rule(
        Atom {
            relation: "Canonical".into(),
            args: vec![Expr::Var("t".into()), Expr::Var("t".into())],
        },
        vec![Literal::Atom(Atom {
            relation: "Buffer".into(),
            args: vec![Expr::Var("t".into())],
        })],
    ));

    p.directives.push(Directive::PrintSize("Canonical".into()));
    p.directives.push(Directive::PrintSize("CanonicalSnap".into()));
    p.directives.push(Directive::PrintSize("Buffer".into()));
    p.directives.push(Directive::Output {
        relation: "Canonical".into(),
        params: vec![("IO".into(), "stdout".into())],
    });
    p
}

/// A tiny e-graph: prove (Add 1 2) ~ (Add 2 1) using records as term IDs.
/// Mirrors `experiments/souffle-compat/q4-fresh-ids-4-egraph.dl`. No outer
/// saturate — the rules saturate in one SCC (this example doesn't yet
/// exercise the buffer/canon split; it proves the IR can express the basic
/// e-graph pattern).
pub fn add_commutativity_egraph() -> Program {
    let mut p = Program::default();

    p.types.push(TypeDecl {
        name: "Math".into(),
        kind: TypeKind::Record(vec![
            ("tag".into(), "number".into()),
            ("a".into(), "Math".into()),
            ("b".into(), "Math".into()),
            ("n".into(), "number".into()),
        ]),
    });

    let math = "Math".to_string();
    p.relations.push(RelationDecl {
        name: "Term".into(),
        columns: vec![("t".into(), math.clone())],
    });
    p.relations.push(RelationDecl {
        name: "AddView".into(),
        columns: vec![
            ("a".into(), math.clone()),
            ("b".into(), math.clone()),
            ("leader".into(), math.clone()),
        ],
    });
    p.relations.push(RelationDecl {
        name: "UF".into(),
        columns: vec![("child".into(), math.clone()), ("parent".into(), math.clone())],
    });

    // Helper: Lit(n) record `[0, nil, nil, n]`.
    let lit = |n: i64| {
        Expr::Record(vec![Expr::Number(0), Expr::Nil, Expr::Nil, Expr::Number(n)])
    };
    // Helper: Add(a, b) record `[1, a, b, 0]`.
    let add = |a: Expr, b: Expr| Expr::Record(vec![Expr::Number(1), a, b, Expr::Number(0)]);

    // Initial state.
    let l1 = lit(1);
    let l2 = lit(2);
    let add12 = add(l1.clone(), l2.clone());
    p.clauses.push(Clause::fact(Atom { relation: "Term".into(), args: vec![l1.clone()] }));
    p.clauses.push(Clause::fact(Atom { relation: "Term".into(), args: vec![l2.clone()] }));
    p.clauses.push(Clause::fact(Atom { relation: "Term".into(), args: vec![add12.clone()] }));
    // UF self-loops.
    p.clauses.push(Clause::rule(
        Atom { relation: "UF".into(), args: vec![Expr::Var("t".into()), Expr::Var("t".into())] },
        vec![Literal::Atom(Atom {
            relation: "Term".into(),
            args: vec![Expr::Var("t".into())],
        })],
    ));
    // Initial AddView entry.
    p.clauses.push(Clause::fact(Atom {
        relation: "AddView".into(),
        args: vec![l1.clone(), l2.clone(), add12.clone()],
    }));

    // Commutativity action — inline records for new terms.
    let av_body = Literal::Atom(Atom {
        relation: "AddView".into(),
        args: vec![Expr::Var("a".into()), Expr::Var("b".into()), Expr::Wildcard],
    });
    // Term(Add(a, b))
    p.clauses.push(Clause::rule(
        Atom {
            relation: "Term".into(),
            args: vec![add(Expr::Var("a".into()), Expr::Var("b".into()))],
        },
        vec![av_body.clone()],
    ));
    // Term(Add(b, a))
    p.clauses.push(Clause::rule(
        Atom {
            relation: "Term".into(),
            args: vec![add(Expr::Var("b".into()), Expr::Var("a".into()))],
        },
        vec![av_body.clone()],
    ));
    // AddView(b, a, Add(b, a))
    p.clauses.push(Clause::rule(
        Atom {
            relation: "AddView".into(),
            args: vec![
                Expr::Var("b".into()),
                Expr::Var("a".into()),
                add(Expr::Var("b".into()), Expr::Var("a".into())),
            ],
        },
        vec![av_body.clone()],
    ));
    // Self-loops for new Add terms.
    let add_ab = add(Expr::Var("a".into()), Expr::Var("b".into()));
    let add_ba = add(Expr::Var("b".into()), Expr::Var("a".into()));
    p.clauses.push(Clause::rule(
        Atom { relation: "UF".into(), args: vec![add_ab.clone(), add_ab.clone()] },
        vec![av_body.clone()],
    ));
    p.clauses.push(Clause::rule(
        Atom { relation: "UF".into(), args: vec![add_ba.clone(), add_ba.clone()] },
        vec![av_body.clone()],
    ));
    // Direct union with deterministic direction via ord(): bigger ord becomes child.
    p.clauses.push(Clause::rule(
        Atom { relation: "UF".into(), args: vec![Expr::Var("t1".into()), Expr::Var("t2".into())] },
        vec![
            av_body.clone(),
            Literal::Constraint(BinaryOp::Eq, Expr::Var("t1".into()), add_ab.clone()),
            Literal::Constraint(BinaryOp::Eq, Expr::Var("t2".into()), add_ba.clone()),
            Literal::Constraint(
                BinaryOp::Gt,
                Expr::Ord(Box::new(Expr::Var("t1".into()))),
                Expr::Ord(Box::new(Expr::Var("t2".into()))),
            ),
        ],
    ));
    p.clauses.push(Clause::rule(
        Atom { relation: "UF".into(), args: vec![Expr::Var("t2".into()), Expr::Var("t1".into())] },
        vec![
            av_body,
            Literal::Constraint(BinaryOp::Eq, Expr::Var("t1".into()), add_ab),
            Literal::Constraint(BinaryOp::Eq, Expr::Var("t2".into()), add_ba),
            Literal::Constraint(
                BinaryOp::Gt,
                Expr::Ord(Box::new(Expr::Var("t2".into()))),
                Expr::Ord(Box::new(Expr::Var("t1".into()))),
            ),
        ],
    ));
    // Path compression as subsumption — represented separately. (TODO: add a
    // SubsumeRule variant in IR; for now we approximate with a regular rule
    // that won't actually delete, but the test below avoids cases where path
    // compression matters.)

    // Output.
    p.directives.push(Directive::PrintSize("UF".into()));
    p.directives.push(Directive::PrintSize("AddView".into()));
    p.directives.push(Directive::Output {
        relation: "UF".into(),
        params: vec![("IO".into(), "stdout".into())],
    });
    p
}
