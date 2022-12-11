use crate::*;
impl EGraph {
    fn calc_helper(
        &mut self,
        idents: Vec<IdentSort>,
        exprs: Vec<Expr>,
        depth: &mut i64,
    ) -> Result<(), Error> {
        self.push();
        *depth += 1;
        // Insert fresh symbols for locally universally quantified reasoning.
        for IdentSort { name, sort } in idents {
            let sort = self.sorts.get(&sort).unwrap().clone();
            self.declare_const(name, &sort)?;
        }
        // Insert each expression pair and run until they match.
        for ab in exprs.windows(2) {
            let a = &ab[0];
            let b = &ab[1];
            self.push();
            *depth += 1;
            self.eval_expr(a, None, true)?;
            self.eval_expr(b, None, true)?;
            let cond = Fact::Eq(vec![a.clone(), b.clone()]);
            self.run_command(
                Command::Run(RunConfig {
                    limit: 100000,
                    until: Some(cond.clone()),
                }),
                true,
            )?;
            self.run_command(Command::Check(cond), true)?;
            self.pop().unwrap();
            *depth -= 1;
        }
        self.pop().unwrap();
        *depth -= 1;
        Ok(())
    }

    // Prove a sequence of equalities universally quantified over idents
    pub fn calc(&mut self, idents: Vec<IdentSort>, exprs: Vec<Expr>) -> Result<(), Error> {
        if exprs.len() < 2 {
            Ok(())
        } else {
            let mut depth = 0;
            let res = self.calc_helper(idents, exprs, &mut depth);
            if res.is_err() {
                // pop egraph back to original state if error
                for _ in 0..depth {
                    self.pop()?;
                }
            } else {
                assert!(depth == 0);
            }
            res
        }
    }

    fn run_query_formula(&mut self, goal: Query) -> Result<(), Error> {
        match goal {
            Query::Atom(fact) => {
                if self.check_fact(&fact, true).is_err() {
                    println!("{}", self.summary());
                    // I should actually have run check until first if it doesn't already
                    self.run_command(
                        Command::Run(RunConfig {
                            limit: 100000,
                            until: Some(fact.clone()),
                        }),
                        true,
                    )?;
                    self.check_fact(&fact, true)
                } else {
                    Ok(())
                }
            }
            Query::And(goals) => {
                for goal in goals {
                    self.run_query_formula(goal)?;
                }
                Ok(())
            }
        }
    }
    fn assert_prog_helper(&mut self, body: &mut Vec<Fact>, prog: Prog) -> Result<(), Error> {
        match prog {
            Prog::Atom(action) => {
                if body.is_empty() {
                    self.eval_actions(&[action])
                } else {
                    let _ = self.add_rule(ast::Rule {
                        body: body.clone(),
                        head: vec![action],
                    })?; // Hmm duplicate rules error?
                    Ok(())
                }
            }
            Prog::And(progs) => {
                for prog in progs {
                    self.assert_prog_helper(&mut body.clone(), prog)?;
                }
                Ok(())
            }
            Prog::ForAll(idents, prog) => {
                for ident in idents {
                    // This is fishy
                    // I should actually track what is in the context.
                    assert!(!self.functions.contains_key(&ident.name));
                }
                self.assert_prog_helper(body, *prog)
            }
            Prog::Implies(goal, prog) => {
                EGraph::body_from_query(body, goal);
                self.assert_prog_helper(body, *prog)
            }
        }
    }

    pub fn assert_prog(&mut self, prog: Prog) -> Result<(), Error> {
        self.assert_prog_helper(&mut vec![], prog)
    }
    fn body_from_query(body: &mut Vec<Fact>, g: Query) {
        match g {
            Query::Atom(f) => body.push(f),
            Query::And(goals) => {
                for goal in goals {
                    EGraph::body_from_query(body, goal);
                }
            }
        }
    }
    fn reduce_goal(&mut self, goal: Goal) -> Result<(), Error> {
        match goal {
            Goal::ForAll(idents, goal) => {
                for IdentSort { name, sort } in idents {
                    let sort = self.sorts.get(&sort).unwrap().clone();
                    self.declare_const(name, &sort)?;
                }
                self.reduce_goal(*goal)
            }
            Goal::Query(goal) => self.run_query_formula(goal),
            Goal::Implies(prog, goal) => {
                self.assert_prog(prog)?;
                self.reduce_goal(*goal)
            }
        }
    }
    pub fn prove_goal(&mut self, goal: Goal) -> Result<(), Error> {
        self.push();
        let res = self.reduce_goal(goal);
        self.pop().unwrap();
        res
    }
}
