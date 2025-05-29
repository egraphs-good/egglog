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

pub mod ruleset {
    use super::*;

    /// Create a new ruleset.
    pub fn add(egraph: &mut EGraph, ruleset: Symbol) -> Result<(), Error> {
        egraph.run_program(vec![Command::AddRuleset(span!(), ruleset)])?;
        Ok(())
    }

    /// Run one iteration of a ruleset.
    pub fn run(egraph: &mut EGraph, ruleset: Symbol) -> Result<(), Error> {
        egraph.run_program(vec![Command::RunSchedule(Schedule::Run(
            span!(),
            RunConfig {
                ruleset,
                until: None,
            },
        ))])?;
        Ok(())
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

// TODO: actions macro

// TODO: rule vs rust_rule

/// A wrapper around an `ExecutionState` for rules that are written in Rust.
pub struct Context<'a, 'b> {
    exec_state: &'a mut ExecutionState<'b>,
    union_action: egglog_bridge::UnionAction,
    table_actions: HashMap<Symbol, egglog_bridge::TableAction>,
}

impl Context<'_, '_> {
    /// Convert from an egglog value to a Rust type.
    pub fn value_to_rust<T: core_relations::Primitive>(&self, x: Value) -> T {
        self.exec_state.prims().unwrap::<T>(x)
    }

    /// Convert from a Rust type to an egglog value.
    pub fn rust_to_value<T: core_relations::Primitive>(&self, x: T) -> Value {
        self.exec_state.prims().get::<T>(x)
    }

    fn get_table_action(&self, table: &str) -> egglog_bridge::TableAction {
        self.table_actions[&Symbol::from(table)]
    }

    /// Do a table lookup. This is potentially a mutable operation!
    /// For more information, see `egglog_bridge::TableAction::lookup`.
    pub fn lookup(&mut self, table: &str, key: Vec<Value>) -> Option<Value> {
        self.get_table_action(table).lookup(self.exec_state, &key)
    }

    /// Union two values in the e-graph.
    /// For more information, see `egglog_bridge::UnionAction::union`.
    pub fn union(&mut self, x: Value, y: Value) {
        self.union_action.union(self.exec_state, x, y)
    }

    /// Insert a row into a table.
    /// For more information, see `egglog_bridge::TableAction::insert`.
    pub fn insert(&mut self, table: &str, row: Vec<Value>) {
        self.get_table_action(table).insert(self.exec_state, row)
    }

    /// Remove a row from a table.
    /// For more information, see `egglog_bridge::TableAction::remove`.
    pub fn remove(&mut self, table: &str, key: &[Value]) {
        self.get_table_action(table).remove(self.exec_state, key)
    }

    /// Subsume a row in a table.
    /// For more information, see `egglog_bridge::TableAction::subsume`.
    pub fn subsume(&mut self, table: &str, key: &[Value]) {
        self.get_table_action(table).subsume(self.exec_state, key)
    }
}

#[derive(Clone)]
struct RustRuleRhs<F: Fn(&mut Context, &[Value]) -> Option<()>> {
    name: Symbol,
    inputs: Vec<ArcSort>,
    union_action: egglog_bridge::UnionAction,
    table_actions: HashMap<Symbol, egglog_bridge::TableAction>,
    func: F,
}

impl<F: Fn(&mut Context, &[Value]) -> Option<()>> Primitive for RustRuleRhs<F> {
    fn name(&self) -> Symbol {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let sorts: Vec<_> = self
            .inputs
            .iter()
            .chain(once(&(Arc::new(UnitSort) as Arc<dyn Sort>)))
            .cloned()
            .collect();
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }

    fn apply(&self, exec_state: &mut ExecutionState, values: &[Value]) -> Option<Value> {
        let mut context = Context {
            exec_state,
            union_action: self.union_action,
            table_actions: self.table_actions.clone(),
        };
        (self.func)(&mut context, values)?;
        Some(exec_state.prims().get(()))
    }
}

/// Add a rule to the e-graph in a new ruleset. Returns the ruleset name.
pub fn rule(
    egraph: &mut EGraph,
    ruleset: Symbol,
    vars: &[(&str, ArcSort)],
    facts: Facts<Symbol, Symbol>,
    func: impl Fn(&mut Context, &[Value]) -> Option<()> + Clone + Send + Sync + 'static,
) -> Result<(), Error> {
    let prim_name = egraph
        .parser
        .symbol_gen
        .fresh(&Symbol::from("rust_rule_prim"));

    egraph.add_primitive(RustRuleRhs {
        name: prim_name,
        inputs: vars.iter().map(|(_, s)| s.clone()).collect(),
        union_action: egglog_bridge::UnionAction::new(&egraph.backend),
        table_actions: egraph
            .functions
            .iter()
            .map(|(k, v)| {
                (
                    *k,
                    egglog_bridge::TableAction::new(&egraph.backend, v.backend_id),
                )
            })
            .collect(),
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
    egraph.run_program(vec![Command::Rule {
        name: rule_name,
        ruleset,
        rule,
    }])?;

    Ok(())
}

/// Run a query over the database. Each match is returned as a `Vec<Value>`
/// whose order is the order of the `vars`.
/// TODO: return just one wrapped vec and expose getting rows
/// TODO: add vars macro
pub fn query(
    egraph: &mut EGraph,
    vars: &[(&str, ArcSort)],
    facts: Facts<Symbol, Symbol>,
) -> Result<Vec<Vec<Value>>, Error> {
    use std::sync::{Arc, Mutex};

    let results = Arc::new(Mutex::new(Vec::new()));
    let results_weak = Arc::downgrade(&results);

    let ruleset = egraph
        .parser
        .symbol_gen
        .fresh(&Symbol::from("query_ruleset"));
    ruleset::add(egraph, ruleset)?;

    rule(
        egraph,
        ruleset,
        vars,
        facts,
        move |_, values| match results_weak.upgrade() {
            None => panic!("one-shot rule was called twice"),
            Some(arc) => {
                arc.lock().unwrap().push(values.to_vec());
                Some(())
            }
        },
    )?;

    ruleset::run(egraph, ruleset)?;

    let ruleset = egraph.rulesets.swap_remove(&ruleset).unwrap();

    let Ruleset::Rules(rules) = ruleset else {
        unreachable!()
    };
    assert_eq!(rules.len(), 1);
    let rule = rules.into_iter().next().unwrap().1;
    egraph.backend.free_rule(rule);

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

        let ruleset = Symbol::from("custom_ruleset");
        ruleset::add(&mut egraph, ruleset)?;

        // add the rule from `build_test_database` to the egraph
        rule(
            &mut egraph,
            ruleset,
            &[("x", sort::int()), ("f0", sort::int()), ("f1", sort::int())],
            facts![
                (= f0 (fib x))
                (= f1 (fib (+ x 1)))
            ],
            move |ctx, values| {
                let [x, f0, f1] = values else { unreachable!() };
                let x = ctx.value_to_rust::<i64>(*x);
                let f0 = ctx.value_to_rust::<i64>(*f0);
                let f1 = ctx.value_to_rust::<i64>(*f1);

                let y = ctx.rust_to_value::<i64>(x + 2);
                let f2 = ctx.rust_to_value::<i64>(f0 + f1);
                ctx.insert("fib", vec![y, f2]);

                Some(())
            },
        )?;

        // run that rule 10 times
        for _ in 0..10 {
            ruleset::run(&mut egraph, ruleset)?;
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
