use crate::free_var_set::{FreeVarSetExternalFns, register_free_var_set_functions};
use egglog_bridge::{
    ColumnTy, DefaultVal, EGraph, FunctionConfig, FunctionId, MergeFn, QueryEntry, RuleId,
};
use egglog_core_relations::{BaseValueId, ExternalFunctionId, Value, make_external_func};
#[cfg(test)]
use std::sync::Once;

#[cfg(test)]
use egglog_bridge::TableAction;

mod church;
mod sexp;

const GUARD_TRUE: Value = Value::new_const(1);

#[cfg(test)]
static INIT_LOGGING: Once = Once::new();

#[cfg(test)]
fn init_logging() {
    INIT_LOGGING.call_once(|| {
        let mut builder = env_logger::Builder::from_default_env();
        builder.is_test(true);
        let _ = builder.try_init();
    });
}

struct LambdaTables {
    lam: FunctionId,
    app: FunctionId,
    var: FunctionId,
    subst: FunctionId,
    free_vars: FunctionId,
    num: FunctionId,
    add: FunctionId,
}

struct LambdaGuards {
    neq: ExternalFunctionId,
}

struct LambdaRules {
    #[allow(dead_code)]
    free_vars: Vec<RuleId>,
    subst: Vec<RuleId>,
    arith: Vec<RuleId>,
    beta: Vec<RuleId>,
}

pub(crate) struct LambdaEnv {
    tables: LambdaTables,
    rules: LambdaRules,
}

fn register_guards(egraph: &mut EGraph) -> LambdaGuards {
    let neq = egraph.register_external_func(Box::new(make_external_func(|_, vals| {
        let [left, right] = vals else {
            panic!("[neq-guard] expected 2 values, got {vals:?}");
        };
        if left != right {
            Some(GUARD_TRUE)
        } else {
            None
        }
    })));
    LambdaGuards { neq }
}

fn add_lambda_tables(
    egraph: &mut EGraph,
    free_var_funcs: &FreeVarSetExternalFns,
    int_base: BaseValueId,
) -> LambdaTables {
    let lam = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "lam".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let app = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "app".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let var = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "var".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let subst = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "subst".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let free_vars = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::Fail,
        merge: MergeFn::Primitive(free_var_funcs.union, vec![MergeFn::Old, MergeFn::New]),
        name: "free-vars".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let num = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "num".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let add = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".to_string(),
        can_subsume: true,
        row_id: false,
    });
    LambdaTables {
        lam,
        app,
        var,
        subst,
        free_vars,
        num,
        add,
    }
}

fn build_free_var_rules(
    egraph: &mut EGraph,
    tables: &LambdaTables,
    free_var_funcs: &FreeVarSetExternalFns,
    int_base: BaseValueId,
) -> Vec<RuleId> {
    let mut rules = Vec::new();

    let var_rule = {
        let mut rb = egraph.new_rule("free-vars-var", true);
        let binder: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let var_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.var, &[binder.clone(), var_id.clone()], None)
            .unwrap();
        let set: QueryEntry = rb
            .call_external_func(
                free_var_funcs.singleton,
                &[binder.clone()],
                ColumnTy::Id,
                || "free-vars-var".to_string(),
            )
            .into();
        rb.set(tables.free_vars, &[var_id, set]);
        rb.build()
    };
    rules.push(var_rule);

    let app_rule = {
        let mut rb = egraph.new_rule("free-vars-app", true);
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let g: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let app_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[f.clone(), g.clone(), app_id.clone()], None)
            .unwrap();
        let fv_f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let fv_g: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.free_vars, &[f, fv_f.clone()], None)
            .unwrap();
        rb.query_table(tables.free_vars, &[g, fv_g.clone()], None)
            .unwrap();
        // Set does a union
        rb.set(tables.free_vars, &[app_id.clone(), fv_f]);
        rb.set(tables.free_vars, &[app_id.clone(), fv_g]);
        rb.build()
    };
    rules.push(app_rule);

    let lam_rule = {
        let mut rb = egraph.new_rule("free-vars-lam", true);
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let lam_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.lam, &[body.clone(), lam_id.clone()], None)
            .unwrap();
        let fv_body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.free_vars, &[body, fv_body.clone()], None)
            .unwrap();
        let removed: QueryEntry = rb
            .call_external_func(
                free_var_funcs.remove,
                &[fv_body, lam_id.clone()],
                ColumnTy::Id,
                || "free-vars-lam".to_string(),
            )
            .into();
        rb.set(tables.free_vars, &[lam_id, removed]);
        rb.build()
    };
    rules.push(lam_rule);

    let num_rule = {
        let mut rb = egraph.new_rule("free-vars-num", true);
        let raw: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let num_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.num, &[raw, num_id.clone()], None)
            .unwrap();
        let empty: QueryEntry = rb
            .call_external_func(free_var_funcs.empty, &[], ColumnTy::Id, || {
                "free-vars-num".to_string()
            })
            .into();
        rb.set(tables.free_vars, &[num_id, empty]);
        rb.build()
    };
    rules.push(num_rule);

    let add_rule = {
        let mut rb = egraph.new_rule("free-vars-add", true);
        let lhs: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let rhs: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let add_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(
            tables.add,
            &[lhs.clone(), rhs.clone(), add_id.clone()],
            None,
        )
        .unwrap();
        let fv_lhs: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let fv_rhs: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.free_vars, &[lhs, fv_lhs.clone()], None)
            .unwrap();
        rb.query_table(tables.free_vars, &[rhs, fv_rhs.clone()], None)
            .unwrap();
        rb.set(tables.free_vars, &[add_id.clone(), fv_lhs]);
        rb.set(tables.free_vars, &[add_id, fv_rhs]);
        rb.build()
    };
    rules.push(add_rule);

    rules
}

fn build_subst_rules(
    egraph: &mut EGraph,
    tables: &LambdaTables,
    guards: &LambdaGuards,
) -> Vec<RuleId> {
    let mut rules = Vec::new();

    let subst_var_eq = {
        let mut rb = egraph.new_rule("subst-var-eq", true);
        let b: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let x: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let e: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let cur: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.subst, &[b.clone(), x.clone(), e.clone(), cur], None)
            .unwrap();
        rb.query_table(tables.var, &[x.clone(), b.clone()], None)
            .unwrap();
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        rb.set(tables.subst, &[b, x, e.clone(), e]);
        rb.build()
    };
    rules.push(subst_var_eq);

    let subst_var_neq = {
        let mut rb = egraph.new_rule("subst-var-neq", true);
        let b: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let x: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let e: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let cur: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.subst, &[b.clone(), x.clone(), e.clone(), cur], None)
            .unwrap();
        let y: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.var, &[y.clone(), b.clone()], None)
            .unwrap();
        rb.query_prim(
            guards.neq,
            &[
                y,
                x.clone(),
                QueryEntry::Const {
                    val: GUARD_TRUE,
                    ty: ColumnTy::Id,
                },
            ],
            ColumnTy::Id,
        )
        .unwrap();
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        rb.set(tables.subst, &[b.clone(), x, e, b]);
        rb.build()
    };
    rules.push(subst_var_neq);

    let subst_app = {
        let mut rb = egraph.new_rule("subst-app", true);
        let b: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let x: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let e: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let cur: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.subst, &[b.clone(), x.clone(), e.clone(), cur], None)
            .unwrap();
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let g: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[f.clone(), g.clone(), b.clone()], None)
            .unwrap();
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        let f_sub: QueryEntry = rb
            .lookup(tables.subst, &[f, x.clone(), e.clone()], || {
                "subst-app-f".to_string()
            })
            .into();
        let g_sub: QueryEntry = rb
            .lookup(tables.subst, &[g, x.clone(), e.clone()], || {
                "subst-app-g".to_string()
            })
            .into();
        let app_sub: QueryEntry = rb
            .lookup(tables.app, &[f_sub, g_sub], || {
                "subst-app-build".to_string()
            })
            .into();
        rb.set(tables.subst, &[b, x, e, app_sub]);
        rb.build()
    };
    rules.push(subst_app);
    let subst_lam_neq = {
        let mut rb = egraph.new_rule("subst-lam-neq", true);
        let b: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let x: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let e: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let cur: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.subst, &[b.clone(), x.clone(), e.clone(), cur], None)
            .unwrap();
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.lam, &[body.clone(), b.clone()], None)
            .unwrap();
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        let body_sub: QueryEntry = rb
            .lookup(tables.subst, &[body, x.clone(), e.clone()], || {
                "subst-lam-body".to_string()
            })
            .into();
        let lam_sub: QueryEntry = rb
            .lookup(tables.lam, &[body_sub], || "subst-lam-build".to_string())
            .into();
        rb.set(tables.subst, &[b, x, e, lam_sub]);
        rb.build()
    };
    rules.push(subst_lam_neq);

    rules
}

fn register_int_add(egraph: &mut EGraph) -> ExternalFunctionId {
    egraph.register_external_func(Box::new(make_external_func(|state, vals| {
        let [left, right] = vals else {
            panic!("[int-add] expected 2 values, got {vals:?}");
        };
        let lhs = state.base_values().unwrap::<i64>(*left);
        let rhs = state.base_values().unwrap::<i64>(*right);
        Some(state.base_values().get(lhs + rhs))
    })))
}

fn build_arith_rules(
    egraph: &mut EGraph,
    tables: &LambdaTables,
    int_base: BaseValueId,
    int_add: ExternalFunctionId,
) -> Vec<RuleId> {
    let mut rules = Vec::new();

    let add_ints = {
        let mut rb = egraph.new_rule("add-ints", true);
        let lhs_raw: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let rhs_raw: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let lhs_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let rhs_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let add_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.num, &[lhs_raw.clone(), lhs_id.clone()], None)
            .unwrap();
        rb.query_table(tables.num, &[rhs_raw.clone(), rhs_id.clone()], None)
            .unwrap();
        rb.query_table(tables.add, &[lhs_id.clone(), rhs_id.clone(), add_id], None)
            .unwrap();
        let sum_raw: QueryEntry = rb
            .call_external_func(
                int_add,
                &[lhs_raw, rhs_raw],
                ColumnTy::Base(int_base),
                || "add-ints".to_string(),
            )
            .into();
        let sum_id: QueryEntry = rb
            .lookup(tables.num, &[sum_raw], || "add-ints-num".to_string())
            .into();
        rb.set(tables.add, &[lhs_id, rhs_id, sum_id]);
        rb.build()
    };
    rules.push(add_ints);

    rules
}

fn build_beta_rules(egraph: &mut EGraph, tables: &LambdaTables) -> Vec<RuleId> {
    let mut rules = Vec::new();

    let beta = {
        let mut rb = egraph.new_rule("beta", true);
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let arg: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let app_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[f.clone(), arg.clone(), app_id.clone()], None)
            .unwrap();
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.lam, &[body.clone(), f.clone()], None)
            .unwrap();
        let subst_res: QueryEntry = rb
            .lookup(tables.subst, &[body, f, arg], || "beta-subst".to_string())
            .into();
        rb.union(app_id, subst_res);
        rb.build()
    };
    rules.push(beta);

    rules
}

pub(crate) fn setup_lambda(egraph: &mut EGraph) -> LambdaEnv {
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let free_var_funcs = register_free_var_set_functions(egraph);
    let guards = register_guards(egraph);
    let tables = add_lambda_tables(egraph, &free_var_funcs, int_base);
    let free_vars = build_free_var_rules(egraph, &tables, &free_var_funcs, int_base);
    let subst = build_subst_rules(egraph, &tables, &guards);
    let int_add = register_int_add(egraph);
    let arith = build_arith_rules(egraph, &tables, int_base, int_add);
    let beta = build_beta_rules(egraph, &tables);
    let rules = LambdaRules {
        free_vars,
        subst,
        arith,
        beta,
    };
    LambdaEnv { tables, rules }
}

fn run_rules(
    egraph: &mut EGraph,
    rules: &[RuleId],
    run_partition_refinement: bool,
    max_iterations: Option<usize>,
) -> usize {
    let mut iterations = 0;
    let mut changed = true;
    while changed {
        changed = false;
        if !rules.is_empty() {
            let report = egraph.run_rules(rules).unwrap();
            changed |= report.changed();
        }
        if run_partition_refinement {
            let refined = egraph.run_hash_partition_refinement().unwrap();
            changed |= refined;
        }
        iterations += 1;
        if let Some(max_iterations) = max_iterations {
            if max_iterations <= iterations {
                break;
            }
        }
    }
    iterations
}

fn run_to_fixpoint(egraph: &mut EGraph, rules: &[RuleId], run_partition_refinement: bool) -> usize {
    run_rules(egraph, rules, run_partition_refinement, None)
}

#[cfg(test)]
fn insert_identity_lam(egraph: &mut EGraph, tables: &LambdaTables) -> (Value, Value) {
    let lam_id = egraph.fresh_id();
    let var_id = egraph.fresh_id();
    let mut lam_action = TableAction::new(egraph, tables.lam);
    let mut var_action = TableAction::new(egraph, tables.var);
    egraph.with_execution_state(|state| {
        var_action.insert(state, [lam_id, var_id].into_iter());
        lam_action.insert(state, [var_id, lam_id].into_iter());
    });
    egraph.flush_updates();
    (lam_id, var_id)
}

#[cfg(test)]
fn insert_var(egraph: &mut EGraph, tables: &LambdaTables, binder: Value) -> Value {
    let var_id = egraph.fresh_id();
    let mut var_action = TableAction::new(egraph, tables.var);
    egraph.with_execution_state(|state| {
        var_action.insert(state, [binder, var_id].into_iter());
    });
    egraph.flush_updates();
    var_id
}

#[cfg(test)]
fn insert_named_lam_example(egraph: &mut EGraph, tables: &LambdaTables) -> Value {
    let lam_x = egraph.fresh_id();
    let lam_y = egraph.fresh_id();
    let lam_z = egraph.fresh_id();
    let mut lam_action = TableAction::new(egraph, tables.lam);
    let var_action = TableAction::new(egraph, tables.var);
    let app_action = TableAction::new(egraph, tables.app);
    egraph.with_execution_state(|state| {
        let var_x = var_action
            .lookup(state, &[lam_x])
            .expect("var lookup for x");
        let var_y = var_action
            .lookup(state, &[lam_y])
            .expect("var lookup for y");
        let var_z = var_action
            .lookup(state, &[lam_z])
            .expect("var lookup for z");
        let app_xz = app_action
            .lookup(state, &[var_x, var_z])
            .expect("app lookup for xz");
        let app_xz_y = app_action
            .lookup(state, &[app_xz, var_y])
            .expect("app lookup for xz y");
        lam_action.insert(state, [app_xz_y, lam_z].into_iter());
        lam_action.insert(state, [lam_z, lam_y].into_iter());
        lam_action.insert(state, [lam_y, lam_x].into_iter());
    });
    egraph.flush_updates();
    lam_x
}

#[cfg(test)]
fn seed_subst(egraph: &mut EGraph, tables: &LambdaTables, b: Value, x: Value, e: Value) {
    let subst_action = TableAction::new(egraph, tables.subst);
    egraph.with_execution_state(|state| {
        subst_action.lookup(state, &[b, x, e]);
    });
    egraph.flush_updates();
}

pub fn run_demo() {
    church::run_church_demo(3, false);
    church::run_church_demo(3, true);
}

#[cfg(test)]
mod tests {
    use super::sexp::{ParseError, add_expr_from_sexp};
    use super::*;
    use crate::free_var_set::FreeVarSet;
    use egglog_core_relations::ContainerValue;
    use egglog_numeric_id::NumericId;

    fn lookup_free_vars(egraph: &EGraph, tables: &LambdaTables, expr: Value) -> Vec<Value> {
        let free_vars_action = TableAction::new(egraph, tables.free_vars);
        let set_id = egraph
            .with_execution_state(|state| free_vars_action.lookup(state, &[expr]))
            .expect("missing free-vars row");
        let set = egraph
            .container_values()
            .get_val::<FreeVarSet>(set_id)
            .expect("missing free-vars container");
        set.iter().collect()
    }

    fn lookup_subst(egraph: &EGraph, tables: &LambdaTables, b: Value, x: Value, e: Value) -> Value {
        let mut out = None;
        egraph.for_each_while(tables.subst, |row| {
            if row.vals[0] == b && row.vals[1] == x && row.vals[2] == e {
                out = Some(row.vals[3]);
                false
            } else {
                true
            }
        });
        out.expect("missing subst row")
    }

    fn lookup_add(egraph: &EGraph, tables: &LambdaTables, lhs: Value, rhs: Value) -> Value {
        let mut out = None;
        egraph.for_each_while(tables.add, |row| {
            if row.vals[0] == lhs && row.vals[1] == rhs {
                out = Some(row.vals[2]);
                false
            } else {
                true
            }
        });
        out.expect("missing add row")
    }

    #[allow(unused)]
    fn dump_table(egraph: &EGraph, label: &str, table: FunctionId) {
        init_logging();
        log::info!("{label}:");
        egraph.for_each_while(table, |row| {
            log::info!("{label} {:?}", row.vals);
            true
        });
    }

    #[test]
    fn free_vars_identity_cycle() {
        let mut egraph = EGraph::default();
        let env = setup_lambda(&mut egraph);
        let (lam_id, var_id) = insert_identity_lam(&mut egraph, &env.tables);
        run_to_fixpoint(&mut egraph, &env.rules.free_vars, false);

        let lam_vars = lookup_free_vars(&egraph, &env.tables, lam_id);
        let var_vars = lookup_free_vars(&egraph, &env.tables, var_id);

        assert!(lam_vars.is_empty());
        assert_eq!(var_vars, vec![lam_id]);
    }

    #[test]
    fn subst_var_equal() {
        let mut egraph = EGraph::default();
        let env = setup_lambda(&mut egraph);
        let (lam_id, var_id) = insert_identity_lam(&mut egraph, &env.tables);
        let other_binder = Value::new(99);
        let other_var = insert_var(&mut egraph, &env.tables, other_binder);

        seed_subst(&mut egraph, &env.tables, var_id, lam_id, other_var);
        run_to_fixpoint(&mut egraph, &env.rules.subst, false);

        let res = lookup_subst(&egraph, &env.tables, var_id, lam_id, other_var);
        assert_eq!(
            egraph.get_canon_repr(res, ColumnTy::Id),
            egraph.get_canon_repr(other_var, ColumnTy::Id)
        );
    }

    #[test]
    fn subst_var_neq() {
        let mut egraph = EGraph::default();
        let env = setup_lambda(&mut egraph);
        let (_lam_id, var_id) = insert_identity_lam(&mut egraph, &env.tables);
        let other_binder = Value::new(77);
        let other_var = insert_var(&mut egraph, &env.tables, other_binder);

        seed_subst(&mut egraph, &env.tables, var_id, other_binder, other_var);
        run_to_fixpoint(&mut egraph, &env.rules.subst, false);

        let res = lookup_subst(&egraph, &env.tables, var_id, other_binder, other_var);
        assert_eq!(
            egraph.get_canon_repr(res, ColumnTy::Id),
            egraph.get_canon_repr(var_id, ColumnTy::Id)
        );
    }

    #[test]
    fn subst_app_recurses() {
        let mut egraph = EGraph::default();
        let env = setup_lambda(&mut egraph);
        let (lam_id, var_id) = insert_identity_lam(&mut egraph, &env.tables);
        let other_binder = Value::new(55);
        let other_var = insert_var(&mut egraph, &env.tables, other_binder);
        let app_id = egraph.add_term(env.tables.app, &[var_id, other_var], "app");

        seed_subst(&mut egraph, &env.tables, app_id, lam_id, other_var);
        run_to_fixpoint(&mut egraph, &env.rules.subst, false);

        let res = lookup_subst(&egraph, &env.tables, app_id, lam_id, other_var);
        let expected = egraph.add_term(env.tables.app, &[other_var, other_var], "app");
        assert_eq!(
            egraph.get_canon_repr(res, ColumnTy::Id),
            egraph.get_canon_repr(expected, ColumnTy::Id)
        );
    }

    #[test]
    fn add_base_integers() {
        let mut egraph = EGraph::default();
        let env = setup_lambda(&mut egraph);

        let base_two = egraph.base_values().get(2i64);
        let base_three = egraph.base_values().get(3i64);
        let num_action = TableAction::new(&egraph, env.tables.num);
        let add_action = TableAction::new(&egraph, env.tables.add);
        let id_two = egraph
            .with_execution_state(|state| num_action.lookup(state, &[base_two]))
            .expect("missing num row for 2");
        let id_three = egraph
            .with_execution_state(|state| num_action.lookup(state, &[base_three]))
            .expect("missing num row for 3");
        let _add_id = egraph
            .with_execution_state(|state| add_action.lookup(state, &[id_two, id_three]))
            .expect("missing add row");
        egraph.flush_updates();

        run_to_fixpoint(&mut egraph, &env.rules.arith, false);

        let result = lookup_add(&egraph, &env.tables, id_two, id_three);
        let base_five = egraph.base_values().get(5i64);
        let id_five = egraph
            .with_execution_state(|state| num_action.lookup(state, &[base_five]))
            .expect("missing num row for 5");
        egraph.flush_updates();

        assert_eq!(
            egraph.get_canon_repr(result, ColumnTy::Id),
            egraph.get_canon_repr(id_five, ColumnTy::Id)
        );
    }

    #[test]
    fn partition_refinement_merges_identity_cycles() {
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_lambda(&mut egraph);
        let (lam_a, _var_a) = insert_identity_lam(&mut egraph, &env.tables);
        let (lam_b, _var_b) = insert_identity_lam(&mut egraph, &env.tables);

        run_to_fixpoint(&mut egraph, &[], true);

        assert_eq!(
            egraph.get_canon_repr(lam_a, ColumnTy::Id),
            egraph.get_canon_repr(lam_b, ColumnTy::Id)
        );
    }

    #[test]
    fn partition_refinement_skipped_keeps_identity_cycles_distinct() {
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_lambda(&mut egraph);
        let (lam_a, _var_a) = insert_identity_lam(&mut egraph, &env.tables);
        let (lam_b, _var_b) = insert_identity_lam(&mut egraph, &env.tables);

        run_to_fixpoint(&mut egraph, &[], false);

        assert_ne!(
            egraph.get_canon_repr(lam_a, ColumnTy::Id),
            egraph.get_canon_repr(lam_b, ColumnTy::Id)
        );
    }

    #[test]
    fn partition_refinement_splits_nested_lambda_cycle() {
        // Two terms:
        // - (lam x (lam y ((lam z z) (y x))))
        // - (lam w w)
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_lambda(&mut egraph);

        let outer_lam = egraph.fresh_id();
        let inner_lam = egraph.fresh_id();
        let var_x = egraph.fresh_id();
        let var_y = egraph.fresh_id();
        let app_yx = egraph.fresh_id();
        let app_id_appyx = egraph.fresh_id();

        let mut lam_action = TableAction::new(&egraph, env.tables.lam);
        let mut var_action = TableAction::new(&egraph, env.tables.var);
        let mut app_action = TableAction::new(&egraph, env.tables.app);
        let id_lam = egraph.fresh_id();
        let var_id = egraph.fresh_id();
        let id2_lam = egraph.fresh_id();
        let var_id2 = egraph.fresh_id();
        egraph.with_execution_state(|state| {
            var_action.insert(state, [id2_lam, var_id2].into_iter());
            lam_action.insert(state, [var_id2, id2_lam].into_iter());
            var_action.insert(state, [id_lam, var_id].into_iter());
            lam_action.insert(state, [var_id, id_lam].into_iter());

            app_action.insert(state, [var_y, var_x, app_yx].into_iter());
            app_action.insert(state, [id_lam, app_yx, app_id_appyx].into_iter());

            var_action.insert(state, [outer_lam, var_x].into_iter());
            lam_action.insert(state, [app_id_appyx, inner_lam].into_iter());
            var_action.insert(state, [inner_lam, var_y].into_iter());
            lam_action.insert(state, [inner_lam, outer_lam].into_iter());
        });
        egraph.flush_updates();

        let mut rules = Vec::new();
        rules.extend(env.rules.subst.iter().copied());
        rules.extend(env.rules.beta.iter().copied());
        run_to_fixpoint(&mut egraph, &rules, true);

        assert_ne!(
            egraph.get_canon_repr(outer_lam, ColumnTy::Id),
            egraph.get_canon_repr(inner_lam, ColumnTy::Id)
        );
        // We should have proved that (id (y x)) = (y x)
        assert_eq!(
            egraph.get_canon_repr(app_yx, ColumnTy::Id),
            egraph.get_canon_repr(app_id_appyx, ColumnTy::Id)
        );

        // We should have merged the two identify functions
        assert_eq!(
            egraph.get_canon_repr(id_lam, ColumnTy::Id),
            egraph.get_canon_repr(id2_lam, ColumnTy::Id)
        );
    }

    #[test]
    fn church_n4_runs() {
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_lambda(&mut egraph);
        let _expr = church::add_church_add_application(&mut egraph, &env, 4)
            .expect("church expression parse failed");
        assert!(egraph.table_size(env.tables.lam) > 1);
        let mut rules = Vec::new();
        rules.extend(env.rules.subst.iter().copied());
        rules.extend(env.rules.beta.iter().copied());
        run_rules(&mut egraph, &rules, false, None);
        let lam_before = egraph.table_size(env.tables.lam);
        let app_before = egraph.table_size(env.tables.app);
        let var_before = egraph.table_size(env.tables.var);
        assert!(lam_before > 1);
        assert!(app_before > 1);
        assert!(var_before > 1);
        run_rules(&mut egraph, &[], true, None);
        let lam_after = egraph.table_size(env.tables.lam);
        let app_after = egraph.table_size(env.tables.app);
        let var_after = egraph.table_size(env.tables.var);
        assert!(lam_after > 1);
        assert!(app_after > 1);
        assert!(var_after > 1);
        assert!(lam_after < lam_before);
        assert!(app_after < app_before);
        assert!(var_after < var_before);
    }

    #[test]
    fn sexp_matches_explicit_lambda() {
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_lambda(&mut egraph);
        let explicit = insert_named_lam_example(&mut egraph, &env.tables);
        let parsed = add_expr_from_sexp(&mut egraph, &env, "(lam x (lam y (lam z ((x z) y))))")
            .expect("failed to parse sexp");
        run_to_fixpoint(&mut egraph, &[], true);
        assert_eq!(
            egraph.get_canon_repr(explicit, ColumnTy::Id),
            egraph.get_canon_repr(parsed, ColumnTy::Id)
        );
    }

    #[test]
    fn sexp_rejects_unbound_variable() {
        let mut egraph = EGraph::default();
        let env = setup_lambda(&mut egraph);
        let err = add_expr_from_sexp(&mut egraph, &env, "y").unwrap_err();
        assert!(matches!(err, ParseError::UnboundVariable(_)));
    }
}
