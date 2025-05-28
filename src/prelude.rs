use crate::*;

pub mod expr {
    use super::*;

    pub fn var(name: &str) -> Expr {
        Expr::Var(span!(), name.into())
    }

    pub fn int(value: i64) -> Expr {
        Expr::Lit(span!(), Literal::Int(value))
    }

    pub fn call(f: &str, xs: Vec<Expr>) -> Expr {
        Expr::Call(span!(), f.into(), xs)
    }
}

pub mod fact {
    use super::*;

    pub fn equals(x: Expr, y: Expr) -> Fact {
        Fact::Eq(span!(), x, y)
    }
}

pub mod sort {
    use super::*;

    pub fn int() -> ArcSort {
        Arc::new(I64Sort)
    }
}

#[macro_export]
macro_rules! expr {
    ((unquote $unquoted:expr)) => { $unquoted };
    (($func:tt $($arg:tt)*)) => { expr::call(stringify!($func), vec![$(expr!($arg)),*]) };
    // TODO: this matches ALL literals as ints
    ($value:literal) => { expr::int($value) };
    ($quoted:tt) => { expr::var(stringify!($quoted)) };
}

#[macro_export]
macro_rules! fact {
    ((= $($arg:tt)*)) => { fact::equals($(expr!($arg)),*) };
    ($a:tt) => { Fact::Fact(expr!($a)) };
}

#[macro_export]
macro_rules! facts {
    ($($tree:tt)*) => { Facts(vec![$(fact!($tree)),*]) };
}

#[derive(Clone)]
struct RustRuleRhs<F: Fn(&mut ExecutionState, &[Value]) -> Option<()>> {
    name: Symbol,
    // TODO: just store the TypeConstraint here
    input: Vec<ArcSort>,
    func: F,
}

impl<F: Fn(&mut ExecutionState, &[Value]) -> Option<()>> Primitive for RustRuleRhs<F> {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let sorts: Vec<_> = self
            .input
            .iter()
            .chain(once(&(Arc::new(UnitSort) as Arc<dyn Sort>)))
            .cloned()
            .collect();
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, values: &[Value]) -> Option<Value> {
        (self.func)(exec_state, values)?;
        Some(exec_state.prims().get(()))
    }
}

/// Add a rule to the e-graph. Returns the ruleset name.
pub fn rule(
    egraph: &mut EGraph,
    vars: &[(&str, ArcSort)],
    facts: Facts<Symbol, Symbol>,
    func: impl Fn(&mut ExecutionState, &[Value]) -> Option<()> + Clone + Send + Sync + 'static,
) -> Result<Symbol, Error> {
    let prim_name = egraph
        .parser
        .symbol_gen
        .fresh(&Symbol::from("rust_rule_prim"));
    egraph.add_primitive(RustRuleRhs {
        name: prim_name,
        input: vars.iter().map(|(_, s)| s.clone()).collect(),
        func,
    });

    let rule = Rule {
        span: span!(),
        head: GenericActions(vec![GenericAction::Expr(
            span!(),
            expr::call(
                prim_name.into(),
                vars.iter().map(|(v, _)| expr::var(v)).collect(),
            ),
        )]),
        body: facts.0,
    };

    let rule_name = egraph.parser.symbol_gen.fresh(&"rust_rule".into());
    let ruleset = egraph.parser.symbol_gen.fresh(&"rust_rule_ruleset".into());
    egraph.run_program(vec![
        Command::AddRuleset(span!(), ruleset),
        Command::Rule {
            name: rule_name,
            ruleset,
            rule,
        },
    ])?;

    Ok(ruleset)
}

pub fn run_ruleset(egraph: &mut EGraph, ruleset: Symbol) -> Result<(), Error> {
    egraph.run_program(vec![Command::RunSchedule(Schedule::Run(
        span!(),
        RunConfig {
            ruleset,
            until: None,
        },
    ))])?;
    Ok(())
}

pub fn query(
    egraph: &mut EGraph,
    vars: &[(&str, ArcSort)],
    facts: Facts<Symbol, Symbol>,
) -> Result<Vec<Vec<Value>>, Error> {
    use std::sync::{Arc, Mutex};

    let results = Arc::new(Mutex::new(Vec::new()));
    let results_weak = Arc::downgrade(&results);

    let ruleset = rule(egraph, vars, facts, move |_, values| {
        match results_weak.upgrade() {
            None => panic!("one-shot rule was called twice"),
            Some(arc) => {
                arc.lock().unwrap().push(values.to_vec());
                Some(())
            }
        }
    })?;
    run_ruleset(egraph, ruleset)?;

    match Arc::into_inner(results) {
        Some(mutex) => Ok(mutex.into_inner().unwrap()),
        None => panic!("results_weak.upgrade() was not dropped"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_database() -> Result<EGraph, Error> {
        let mut egraph = EGraph::default();
        egraph.parse_and_run_program(
            None,
            "
(function fib (i64) i64 :no-merge)
(set (fib 0) 0)
(set (fib 1) 1)
(rule (
    (= f0 (fib x))
    (= f1 (fib (+ x 1)))
) (
    (set (fib (+ x 2)) (+ f0 f1))
))
(run 10)
        ",
        )?;
        Ok(egraph)
    }

    #[test]
    fn rust_api_query() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        let results = query(
            &mut egraph,
            &[("x", sort::int()), ("y", sort::int())],
            facts![
                (= (fib x) y)
                (= y 13)
            ],
        )?;

        let x = egraph.backend.primitives().get::<i64>(7);
        let y = egraph.backend.primitives().get::<i64>(13);
        assert_eq!(results, [[x, y]]);

        Ok(())
    }

    #[test]
    fn rust_api_rule() -> Result<(), Error> {
        let mut egraph = build_test_database()?;

        let big_number = 20;

        // check that `fib` does not contain `20`
        let results = query(
            &mut egraph,
            &[("f", sort::int())],
            facts![(= (fib (unquote expr::int(big_number))) f)],
        )?;

        assert!(results.is_empty());

        let fib = egglog_bridge::TableAction::new(
            &egraph.backend,
            egraph.functions[&Symbol::from("fib")].backend_id,
        );

        // add the rule from `build_test_database` to the egraph with a handle
        let ruleset = rule(
            &mut egraph,
            &[("x", sort::int()), ("f0", sort::int()), ("f1", sort::int())],
            facts![
                (= f0 (fib x))
                (= f1 (fib (+ x 1)))
            ],
            move |exec_state, values| {
                let [x, f0, f1] = values else { unreachable!() };
                let x = exec_state.prims().unwrap::<i64>(*x);
                let f0 = exec_state.prims().unwrap::<i64>(*f0);
                let f1 = exec_state.prims().unwrap::<i64>(*f1);

                let y = exec_state.prims().get::<i64>(x + 2);
                let f2 = exec_state.prims().get::<i64>(f0 + f1);
                fib.insert(exec_state, vec![y, f2]);

                Some(())
            },
        )?;

        // run that rule 10 times
        for _ in 0..10 {
            run_ruleset(&mut egraph, ruleset)?;
        }

        // check that `fib` now contains `20`
        let results = query(
            &mut egraph,
            &[("f", sort::int())],
            facts![(= (fib (unquote expr::int(big_number))) f)],
        )?;

        let y = egraph.backend.primitives().get::<i64>(6765);
        assert_eq!(results, [[y]]);

        Ok(())
    }
}
