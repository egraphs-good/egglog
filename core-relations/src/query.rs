//! APIs for building a query of a database.

use std::iter::once;

use numeric_id::{DenseIdMap, NumericId};
use thiserror::Error;

use crate::{
    action::{Instr, QueryEntry, WriteVal},
    common::HashMap,
    free_join::{
        plan::{Plan, PlanStrategy},
        ActionId, AtomId, Database, ProcessedConstraints, SubAtom, TableId, TableInfo, VarInfo,
        Variable,
    },
    pool::{with_pool_set, Pooled},
    primitives::PrimitiveId,
    table_spec::{ColumnId, Constraint},
    ExternalFunctionId, PoolSet,
};

/// A set of rules to run against a [`Database`].
///
/// See [`Database::new_rule_set`] for more information.
#[derive(Default)]
pub struct RuleSet {
    pub(crate) plans: Vec<(Plan, String /* description */)>,
    pub(crate) actions: DenseIdMap<ActionId, Pooled<Vec<Instr>>>,
}

/// Builder for a [`RuleSet`].
///
/// See [`Database::new_rule_set`] for more information.
pub struct RuleSetBuilder<'outer> {
    rule_set: RuleSet,
    db: &'outer mut Database,
}

impl<'outer> RuleSetBuilder<'outer> {
    pub fn new(db: &'outer mut Database) -> Self {
        Self {
            rule_set: Default::default(),
            db,
        }
    }

    /// Estimate the size of the subset of the table matching the given
    /// constraint.
    ///
    /// This is a wrapper around the [`Database::estimate_size`] method.
    pub fn estimate_size(&self, table: TableId, c: Option<Constraint>) -> usize {
        self.db.estimate_size(table, c)
    }

    /// Add a rule to this rule set.
    pub fn new_rule<'a>(&'a mut self) -> QueryBuilder<'outer, 'a> {
        let instrs = with_pool_set(PoolSet::get);
        QueryBuilder {
            rsb: self,
            instrs,
            query: Query {
                var_info: Default::default(),
                atoms: Default::default(),
                // start with an invalid ActionId
                action: ActionId::new(u32::MAX),
                plan_strategy: Default::default(),
            },
        }
    }

    /// Build the ruleset.
    pub fn build(self) -> RuleSet {
        self.rule_set
    }
}

/// Builder for the "query" portion of the rule.
///
/// Queries specify scans or joins over the database that bind variables that
/// are accessible to rules.
pub struct QueryBuilder<'outer, 'a> {
    rsb: &'a mut RuleSetBuilder<'outer>,
    query: Query,
    instrs: Pooled<Vec<Instr>>,
}

impl<'outer, 'a> QueryBuilder<'outer, 'a> {
    /// Finish the query and start building the right-hand side of the rule.
    pub fn build(self) -> RuleBuilder<'outer, 'a> {
        RuleBuilder { qb: self }
    }

    /// Set the target plan strategy to use to execute this query.
    pub fn set_plan_strategy(&mut self, strategy: PlanStrategy) {
        self.query.plan_strategy = strategy;
    }

    /// Create a new variable of the given type.
    pub fn new_var(&mut self) -> Variable {
        self.query.var_info.push(VarInfo {
            occurrences: Default::default(),
            used_in_rhs: false,
        })
    }

    fn mark_used<'b>(&mut self, entries: impl IntoIterator<Item = &'b QueryEntry>) {
        for entry in entries {
            if let QueryEntry::Var(v) = entry {
                self.query.var_info[*v].used_in_rhs = true;
            }
        }
    }

    /// Add the given atom to the query, with the given variables and constraints.
    ///
    /// NB: it is possible to constrain two non-equal variables to be equal
    /// given this setup. Doing this will not cause any problems but
    /// nevertheless is not recommended.
    ///
    /// # Panics
    /// Like most methods that take a [`TableId`], this method will panic if the
    /// given table is not declared in the corresponding database.
    pub fn add_atom<'b>(
        &mut self,
        table_id: TableId,
        vars: &[QueryEntry],
        cs: impl IntoIterator<Item = &'b Constraint>,
    ) -> Result<(), QueryError> {
        let info = &self.rsb.db.tables[table_id];
        let arity = info.spec.arity();
        let check_constraint = |c: &Constraint| {
            let process_col = |col: &ColumnId| -> Result<(), QueryError> {
                if col.index() >= arity {
                    Err(QueryError::InvalidConstraint {
                        constraint: c.clone(),
                        column: col.index(),
                        table: table_id,
                        arity,
                    })
                } else {
                    Ok(())
                }
            };
            match c {
                Constraint::Eq { l_col, r_col } => {
                    process_col(l_col)?;
                    process_col(r_col)
                }
                Constraint::EqConst { col, .. }
                | Constraint::LtConst { col, .. }
                | Constraint::GtConst { col, .. }
                | Constraint::LeConst { col, .. }
                | Constraint::GeConst { col, .. } => process_col(col),
            }
        };
        if arity != vars.len() {
            return Err(QueryError::BadArity {
                table: table_id,
                expected: arity,
                got: vars.len(),
            });
        }
        let cs = Vec::from_iter(
            cs.into_iter()
                .cloned()
                .chain(vars.iter().enumerate().filter_map(|(i, qe)| match qe {
                    QueryEntry::Var(_) => None,
                    QueryEntry::Const(c) => Some(Constraint::EqConst {
                        col: ColumnId::from_usize(i),
                        val: *c,
                    }),
                })),
        );
        cs.iter().try_fold((), |_, c| check_constraint(c))?;
        let processed = self.rsb.db.process_constraints(table_id, &cs);
        let mut atom = Atom {
            table: table_id,
            var_to_column: Default::default(),
            column_to_var: Default::default(),
            constraints: processed,
        };
        let next_atom = AtomId::from_usize(self.query.atoms.n_ids());
        let mut subatoms = HashMap::<Variable, SubAtom>::default();
        for (i, qe) in vars.iter().enumerate() {
            let var = match qe {
                QueryEntry::Var(var) => *var,
                QueryEntry::Const(_) => {
                    continue;
                }
            };
            if var == Variable::placeholder() {
                continue;
            }
            let col = ColumnId::from_usize(i);
            if let Some(prev) = atom.var_to_column.insert(var, col) {
                atom.constraints.slow.push(Constraint::Eq {
                    l_col: col,
                    r_col: prev,
                })
            };
            atom.column_to_var.insert(col, var);
            subatoms
                .entry(var)
                .or_insert_with(|| SubAtom::new(next_atom))
                .vars
                .push(col);
        }
        for (var, subatom) in subatoms {
            self.query
                .var_info
                .get_mut(var)
                .expect("all variables must be bound in current query")
                .occurrences
                .push(subatom);
        }
        self.query.atoms.push(atom);
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("table {table:?} has {expected:?} keys but got {got:?}")]
    KeyArityMismatch {
        table: TableId,
        expected: usize,
        got: usize,
    },
    #[error("table {table:?} has {expected:?} columns but got {got:?}")]
    TableArityMismatch {
        table: TableId,
        expected: usize,
        got: usize,
    },

    #[error("counter used in primitive column {column_id:?} of table {table:?}, which is declared as a primitive")]
    CounterUsedInPrimitiveColumn {
        table: TableId,
        column_id: ColumnId,
        prim: PrimitiveId,
    },

    #[error("attempt to compare two groups of values, one of length {l}, another of length {r}")]
    MultiComparisonMismatch { l: usize, r: usize },

    #[error("table {table:?} expected {expected:?} columns but got {got:?}")]
    BadArity {
        table: TableId,
        expected: usize,
        got: usize,
    },

    #[error("expected {expected:?} columns in schema but got {got:?}")]
    InvalidSchema { expected: usize, got: usize },

    #[error("constraint {constraint:?} on table {table:?} references column {column:?}, but the table has arity {arity:?}")]
    InvalidConstraint {
        constraint: Constraint,
        column: usize,
        table: TableId,
        arity: usize,
    },
}

/// Builder for the "action" portion of the rule.
///
/// Rules can refer to the variables bound in their query to modify the database.
pub struct RuleBuilder<'outer, 'a> {
    qb: QueryBuilder<'outer, 'a>,
}

impl RuleBuilder<'_, '_> {
    /// Build the finished query.
    pub fn build(self) {
        self.build_with_description("")
    }
    pub fn build_with_description(mut self, desc: impl Into<String>) {
        // Generate an id for our actions and slot them in.
        let action_id = self.qb.rsb.rule_set.actions.push(self.qb.instrs);
        self.qb.query.action = action_id;
        // Plan the query
        let plan = self.qb.rsb.db.plan_query(self.qb.query);
        // Add it to the ruleset.
        self.qb.rsb.rule_set.plans.push((plan, desc.into()));
    }

    /// Return a variable containing the result of looking up the specified
    /// column from the row corresponding to given keys in the given
    /// table.
    ///
    /// If the key does not currently have a mapping in the table, the values
    /// specified by `default_vals` will be inserted.
    pub fn lookup_or_insert(
        &mut self,
        table: TableId,
        args: &[QueryEntry],
        default_vals: &[WriteVal],
        dst_col: ColumnId,
    ) -> Result<Variable, QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_keys(table, table_info, args)?;
        self.validate_vals(table, table_info, default_vals.iter())?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::LookupOrInsertDefault {
            table,
            args: args.to_vec(),
            default: default_vals.to_vec(),
            dst_col,
            dst_var: res,
        });
        self.qb.mark_used(args);
        self.qb
            .mark_used(default_vals.iter().filter_map(|x| match x {
                WriteVal::QueryEntry(qe) => Some(qe),
                WriteVal::IncCounter(_) | WriteVal::CurrentVal(_) => None,
            }));
        Ok(res)
    }

    /// Return a variable containing the result of looking up the specified
    /// column from the row corresponding to given keys in the given
    /// table.
    ///
    /// If the key does not currently have a mapping in the table, the variable
    /// takes the value of `default`.
    pub fn lookup_with_default(
        &mut self,
        table: TableId,
        args: &[QueryEntry],
        default: QueryEntry,
        dst_col: ColumnId,
    ) -> Result<Variable, QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_keys(table, table_info, args)?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::LookupWithDefault {
            table,
            args: args.to_vec(),
            dst_col,
            dst_var: res,
            default,
        });
        self.qb.mark_used(args);
        self.qb.mark_used(&[default]);
        Ok(res)
    }

    /// Return a variable containing the result of looking up the specified
    /// column from the row corresponding to given keys in the given
    /// table.
    ///
    /// If the key does not currently have a mapping in the table, execution of
    /// the rule is halted.
    pub fn lookup(
        &mut self,
        table: TableId,
        args: &[QueryEntry],
        dst_col: ColumnId,
    ) -> Result<Variable, QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_keys(table, table_info, args)?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::Lookup {
            table,
            args: args.to_vec(),
            dst_col,
            dst_var: res,
        });
        self.qb.mark_used(args);
        Ok(res)
    }

    /// Insert the specified values into the given table.
    pub fn insert(&mut self, table: TableId, vals: &[QueryEntry]) -> Result<(), QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_row(table, table_info, vals)?;
        self.qb.instrs.push(Instr::Insert {
            table,
            vals: vals.to_vec(),
        });
        self.qb.mark_used(vals);
        Ok(())
    }

    /// Insert the specified values into the given table if `l` and `r` are equal.
    pub fn insert_if_eq(
        &mut self,
        table: TableId,
        l: QueryEntry,
        r: QueryEntry,
        vals: &[QueryEntry],
    ) -> Result<(), QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_row(table, table_info, vals)?;
        self.qb.instrs.push(Instr::InsertIfEq {
            table,
            l,
            r,
            vals: vals.to_vec(),
        });
        self.qb
            .mark_used(vals.iter().chain(once(&l)).chain(once(&r)));
        Ok(())
    }

    /// Remove the specified entry from the given table, if it is there.
    pub fn remove(&mut self, table: TableId, args: &[QueryEntry]) -> Result<(), QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_keys(table, table_info, args)?;
        self.qb.instrs.push(Instr::Remove {
            table,
            args: args.to_vec(),
        });
        self.qb.mark_used(args);
        Ok(())
    }

    /// Apply the given external function to the specified arguments.
    pub fn call_external(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
    ) -> Result<Variable, QueryError> {
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::External {
            func,
            args: args.to_vec(),
            dst: res,
        });
        self.qb.mark_used(args);
        Ok(res)
    }

    /// Look up the given key in the given table. If the lookup fails, then call the given external
    /// function with the given arguments. Bind the result to the returned variable. If the
    /// external function returns None (and the lookup fails) then the execution of the rule halts.
    pub fn lookup_with_fallback(
        &mut self,
        table: TableId,
        key: &[QueryEntry],
        dst_col: ColumnId,
        func: ExternalFunctionId,
        func_args: &[QueryEntry],
    ) -> Result<Variable, QueryError> {
        let table_info = self
            .qb
            .rsb
            .db
            .tables
            .get(table)
            .expect("table must be declared in the current database");
        self.validate_keys(table, table_info, key)?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::LookupWithFallback {
            table,
            table_key: key.to_vec(),
            func,
            func_args: func_args.to_vec(),
            dst_var: res,
            dst_col,
        });
        self.qb.mark_used(key);
        self.qb.mark_used(func_args);
        Ok(res)
    }

    pub fn call_external_with_fallback(
        &mut self,
        f1: ExternalFunctionId,
        args1: &[QueryEntry],
        f2: ExternalFunctionId,
        args2: &[QueryEntry],
    ) -> Result<Variable, QueryError> {
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::ExternalWithFallback {
            f1,
            args1: args1.to_vec(),
            f2,
            args2: args2.to_vec(),
            dst: res,
        });
        self.qb.mark_used(args1);
        self.qb.mark_used(args2);
        Ok(res)
    }

    /// Continue execution iff the two arguments are equal.
    pub fn assert_eq(&mut self, l: QueryEntry, r: QueryEntry) {
        self.qb.instrs.push(Instr::AssertEq(l, r));
        self.qb.mark_used(&[l, r]);
    }

    /// Continue execution iff the two arguments are not equal.
    pub fn assert_ne(&mut self, l: QueryEntry, r: QueryEntry) -> Result<(), QueryError> {
        self.qb.instrs.push(Instr::AssertNe(l, r));
        self.qb.mark_used(&[l, r]);
        Ok(())
    }

    /// Continue execution iff there is some `i` such that `l[i] != r[i]`.
    ///
    /// This is useful when doing egglog-style rebuilding.
    pub fn assert_any_ne(&mut self, l: &[QueryEntry], r: &[QueryEntry]) -> Result<(), QueryError> {
        if l.len() != r.len() {
            return Err(QueryError::MultiComparisonMismatch {
                l: l.len(),
                r: r.len(),
            });
        }

        let mut ops = Vec::with_capacity(l.len() + r.len());
        ops.extend_from_slice(l);
        ops.extend_from_slice(r);
        self.qb.instrs.push(Instr::AssertAnyNe {
            ops,
            divider: l.len(),
        });
        self.qb.mark_used(l);
        self.qb.mark_used(r);
        Ok(())
    }

    fn validate_row(
        &self,
        table: TableId,
        info: &TableInfo,
        vals: &[QueryEntry],
    ) -> Result<(), QueryError> {
        if vals.len() != info.spec.arity() {
            Err(QueryError::TableArityMismatch {
                table,
                expected: info.spec.arity(),
                got: vals.len(),
            })
        } else {
            Ok(())
        }
    }

    fn validate_keys(
        &self,
        table: TableId,
        info: &TableInfo,
        keys: &[QueryEntry],
    ) -> Result<(), QueryError> {
        if keys.len() != info.spec.n_keys {
            Err(QueryError::KeyArityMismatch {
                table,
                expected: info.spec.n_keys,
                got: keys.len(),
            })
        } else {
            Ok(())
        }
    }

    fn validate_vals<'b>(
        &self,
        table: TableId,
        info: &TableInfo,
        vals: impl Iterator<Item = &'b WriteVal>,
    ) -> Result<(), QueryError> {
        for (i, _) in vals.enumerate() {
            let col = i + info.spec.n_keys;
            if col >= info.spec.arity() {
                return Err(QueryError::TableArityMismatch {
                    table,
                    expected: info.spec.arity(),
                    got: col,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Atom {
    pub(crate) table: TableId,
    pub(crate) var_to_column: HashMap<Variable, ColumnId>,
    pub(crate) column_to_var: DenseIdMap<ColumnId, Variable>,
    pub(crate) constraints: ProcessedConstraints,
}

pub(crate) struct Query {
    pub(crate) var_info: DenseIdMap<Variable, VarInfo>,
    pub(crate) atoms: DenseIdMap<AtomId, Atom>,
    pub(crate) action: ActionId,
    pub(crate) plan_strategy: PlanStrategy,
}
