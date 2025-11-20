use super::*;
use egglog_add_primitive::add_literal_prim;

#[derive(Debug)]
pub struct BoolSort;

impl BaseSort for BoolSort {
    type Base = bool;

    fn name(&self) -> &str {
        "bool"
    }

    #[rustfmt::skip]
    fn register_primitives(&self, eg: &mut EGraph) {
        add_literal_prim!(eg, "not" = |a: bool| -> bool { !a });
        eg.add_primitive_validator("not", std::sync::Arc::new(|termdag, lhs| {
            use egglog_bridge::termdag::Term;
            use egglog_ast::generic_ast::Literal;
            if let Term::App(_, args) = termdag.get(lhs) {
                if args.len() == 1 {
                    if let Term::Lit(Literal::Bool(a)) = termdag.get(args[0]) {
                        return Some(Literal::Bool(!a));
                    }
                }
            }
            None
        }));

        add_literal_prim!(eg, "and" = |a: bool, b: bool| -> bool { a && b });
        eg.add_primitive_validator("and", std::sync::Arc::new(|termdag, lhs| {
            use egglog_bridge::termdag::Term;
            use egglog_ast::generic_ast::Literal;
            if let Term::App(_, args) = termdag.get(lhs) {
                if args.len() == 2 {
                    if let (Term::Lit(Literal::Bool(a)), Term::Lit(Literal::Bool(b))) =
                        (termdag.get(args[0]), termdag.get(args[1]))
                    {
                        return Some(Literal::Bool(*a && *b));
                    }
                }
            }
            None
        }));

        add_literal_prim!(eg, "or" = |a: bool, b: bool| -> bool { a || b });
        eg.add_primitive_validator("or", std::sync::Arc::new(|termdag, lhs| {
            use egglog_bridge::termdag::Term;
            use egglog_ast::generic_ast::Literal;
            if let Term::App(_, args) = termdag.get(lhs) {
                if args.len() == 2 {
                    if let (Term::Lit(Literal::Bool(a)), Term::Lit(Literal::Bool(b))) =
                        (termdag.get(args[0]), termdag.get(args[1]))
                    {
                        return Some(Literal::Bool(*a || *b));
                    }
                }
            }
            None
        }));

        add_primitive!(eg, "xor" = |a: bool, b: bool| -> bool { a ^ b });
        eg.add_primitive_validator("xor", std::sync::Arc::new(|termdag, lhs| {
            use egglog_bridge::termdag::Term;
            use egglog_ast::generic_ast::Literal;
            if let Term::App(_, args) = termdag.get(lhs) {
                if args.len() == 2 {
                    if let (Term::Lit(Literal::Bool(a)), Term::Lit(Literal::Bool(b))) =
                        (termdag.get(args[0]), termdag.get(args[1]))
                    {
                        return Some(Literal::Bool(*a ^ *b));
                    }
                }
            }
            None
        }));

        add_primitive!(eg, "=>" = |a: bool, b: bool| -> bool { !a || b });
        eg.add_primitive_validator("=>", std::sync::Arc::new(|termdag, lhs| {
            use egglog_bridge::termdag::Term;
            use egglog_ast::generic_ast::Literal;
            if let Term::App(_, args) = termdag.get(lhs) {
                if args.len() == 2 {
                    if let (Term::Lit(Literal::Bool(a)), Term::Lit(Literal::Bool(b))) =
                        (termdag.get(args[0]), termdag.get(args[1]))
                    {
                        return Some(Literal::Bool(!a || *b));
                    }
                }
            }
            None
        }));
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        use egglog_ast::generic_ast::Literal;

        let b = base_values.unwrap::<bool>(value);
        termdag.lit(Literal::Bool(b))
    }
}
