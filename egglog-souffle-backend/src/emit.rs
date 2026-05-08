//! Pretty-print a [`Program`] as Souffle `.dl` source.

use std::fmt::Write;

use crate::ir::*;

/// Emit `program` as Souffle source.
pub fn emit(program: &Program) -> String {
    let mut out = String::new();
    let _ = emit_into(&mut out, program);
    out
}

fn emit_into(out: &mut String, program: &Program) -> std::fmt::Result {
    // Pragmas first — they affect global config.
    for (k, v) in &program.pragmas {
        writeln!(out, ".pragma \"{k}\" \"{v}\"")?;
    }
    if !program.pragmas.is_empty() {
        writeln!(out)?;
    }

    // Type declarations.
    for ty in &program.types {
        emit_type(out, ty)?;
    }
    if !program.types.is_empty() {
        writeln!(out)?;
    }

    // Relation declarations.
    for rel in &program.relations {
        emit_relation_decl(out, rel)?;
    }
    if !program.relations.is_empty() {
        writeln!(out)?;
    }

    // Directives (printsize, output, limititerations, snapshot).
    for d in &program.directives {
        emit_directive(out, d)?;
    }
    if !program.directives.is_empty() {
        writeln!(out)?;
    }

    // Clauses (facts and rules).
    for c in &program.clauses {
        emit_clause(out, c)?;
    }
    Ok(())
}

fn emit_type(out: &mut String, ty: &TypeDecl) -> std::fmt::Result {
    match &ty.kind {
        TypeKind::Record(fields) => {
            write!(out, ".type {} = [", ty.name)?;
            for (i, (n, t)) in fields.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{n}: {t}")?;
            }
            writeln!(out, "]")?;
        }
    }
    Ok(())
}

fn emit_relation_decl(out: &mut String, rel: &RelationDecl) -> std::fmt::Result {
    write!(out, ".decl {}(", rel.name)?;
    for (i, (n, t)) in rel.columns.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        write!(out, "{n}: {t}")?;
    }
    writeln!(out, ")")?;
    Ok(())
}

fn emit_directive(out: &mut String, d: &Directive) -> std::fmt::Result {
    match d {
        Directive::PrintSize(r) => writeln!(out, ".printsize {r}"),
        Directive::Output { relation, params } => {
            write!(out, ".output {relation}")?;
            if !params.is_empty() {
                write!(out, "(")?;
                for (i, (k, v)) in params.iter().enumerate() {
                    if i > 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{k}={v}")?;
                }
                write!(out, ")")?;
            }
            writeln!(out)
        }
        Directive::LimitIterations { relation, n } => {
            writeln!(out, ".limititerations {relation}(n={n})")
        }
        Directive::Snapshot { snap, source } => {
            writeln!(out, ".snapshot {snap}(of = \"{source}\")")
        }
    }
}

fn emit_clause(out: &mut String, c: &Clause) -> std::fmt::Result {
    emit_atom(out, &c.head)?;
    if c.body.is_empty() {
        writeln!(out, ".")?;
    } else {
        writeln!(out, " :-")?;
        for (i, lit) in c.body.iter().enumerate() {
            write!(out, "  ")?;
            emit_literal(out, lit)?;
            if i + 1 < c.body.len() {
                writeln!(out, ",")?;
            } else {
                writeln!(out, ".")?;
            }
        }
    }
    Ok(())
}

fn emit_atom(out: &mut String, atom: &Atom) -> std::fmt::Result {
    write!(out, "{}(", atom.relation)?;
    for (i, e) in atom.args.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        emit_expr(out, e)?;
    }
    write!(out, ")")
}

fn emit_literal(out: &mut String, lit: &Literal) -> std::fmt::Result {
    match lit {
        Literal::Atom(a) => emit_atom(out, a),
        Literal::Neg(a) => {
            write!(out, "!")?;
            emit_atom(out, a)
        }
        Literal::Constraint(op, l, r) => {
            emit_expr(out, l)?;
            let s = match op {
                BinaryOp::Eq => "=",
                BinaryOp::Ne => "!=",
                BinaryOp::Lt => "<",
                BinaryOp::Le => "<=",
                BinaryOp::Gt => ">",
                BinaryOp::Ge => ">=",
            };
            write!(out, " {s} ")?;
            emit_expr(out, r)
        }
    }
}

fn emit_expr(out: &mut String, e: &Expr) -> std::fmt::Result {
    match e {
        Expr::Var(s) => write!(out, "{s}"),
        Expr::Wildcard => write!(out, "_"),
        Expr::Number(n) => write!(out, "{n}"),
        Expr::Symbol(s) => write!(out, "\"{s}\""),
        Expr::Nil => write!(out, "nil"),
        Expr::Record(fs) => {
            write!(out, "[")?;
            for (i, f) in fs.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                emit_expr(out, f)?;
            }
            write!(out, "]")
        }
        Expr::Ord(inner) => {
            write!(out, "ord(")?;
            emit_expr(out, inner)?;
            write!(out, ")")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_minimal_program() {
        let mut p = Program::default();
        p.pragmas.push(("outer-saturate".into(), "5".into()));
        p.relations.push(RelationDecl {
            name: "P".into(),
            columns: vec![("x".into(), "number".into())],
        });
        // Fact: P(1).
        p.clauses
            .push(Clause::fact(Atom { relation: "P".into(), args: vec![Expr::Number(1)] }));
        // Rule: Q(x) :- P(x).
        p.relations.push(RelationDecl {
            name: "Q".into(),
            columns: vec![("x".into(), "number".into())],
        });
        p.clauses.push(Clause::rule(
            Atom { relation: "Q".into(), args: vec![Expr::Var("x".into())] },
            vec![Literal::Atom(Atom {
                relation: "P".into(),
                args: vec![Expr::Var("x".into())],
            })],
        ));
        p.directives.push(Directive::PrintSize("Q".into()));

        let s = emit(&p);
        assert!(s.contains(".pragma \"outer-saturate\" \"5\""));
        assert!(s.contains(".decl P(x: number)"));
        assert!(s.contains("P(1)."));
        assert!(s.contains("Q(x) :-"));
        assert!(s.contains("  P(x)."));
        assert!(s.contains(".printsize Q"));
    }

    #[test]
    fn emits_record_type_and_construction() {
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
        p.relations.push(RelationDecl {
            name: "Term".into(),
            columns: vec![("t".into(), "Math".into())],
        });
        // Term([1, [0, nil, nil, 1], [0, nil, nil, 2], 0]).
        let lit1 = Expr::Record(vec![
            Expr::Number(0),
            Expr::Nil,
            Expr::Nil,
            Expr::Number(1),
        ]);
        let lit2 = Expr::Record(vec![
            Expr::Number(0),
            Expr::Nil,
            Expr::Nil,
            Expr::Number(2),
        ]);
        let add = Expr::Record(vec![Expr::Number(1), lit1, lit2, Expr::Number(0)]);
        p.clauses.push(Clause::fact(Atom { relation: "Term".into(), args: vec![add] }));

        let s = emit(&p);
        assert!(s.contains(".type Math = [tag: number, a: Math, b: Math, n: number]"));
        assert!(s.contains("Term([1, [0, nil, nil, 1], [0, nil, nil, 2], 0])."));
    }
}
