//! Rise-inspired rewrites over a cyclic lambda encoding.
//!
//! Design notes:
//! - We encode binders as cyclic lambdas (a `var` points back to its binder id).
//! - `rise-fresh` is a memo table: `(row_id) -> fresh binder id` for a specific rule match.
//!   Row ids are stable for a given table row, so they give us a deterministic key per match.
//!   Without this, each re-fire would allocate a new binder id and blow up the search space.
//! - `rise-fresh` is *not* part of term structure, so we opt it out of partition refinement.
//!   If it participated, fingerprints would depend on arbitrary row ids.

use crate::free_var_set::{FreeVarSetExternalFns, register_free_var_set_functions};
use clap::Parser;
use egglog_bridge::{
    ColumnTy, DefaultVal, EGraph, FunctionConfig, FunctionId, MergeFn, QueryEntry,
    RefinementInput, RuleId, TableAction,
};
use egglog_core_relations::{BaseValueId, ExternalFunctionId, Value, make_external_func};
use hashbrown::HashMap;
use log::info;
use thiserror::Error;

const GUARD_TRUE: Value = Value::new_const(1);

pub(crate) struct RiseTables {
    lam: FunctionId,
    app: FunctionId,
    var: FunctionId,
    subst: FunctionId,
    free_vars: FunctionId,
    num: FunctionId,
    sym: FunctionId,
    fresh: FunctionId,
}

struct RiseGuards {
    neq: ExternalFunctionId,
}

struct RiseRules {
    free_vars: Vec<RuleId>,
    subst: Vec<RuleId>,
    beta: Vec<RuleId>,
    rise: Vec<RuleId>,
}

pub(crate) struct RiseEnv {
    tables: RiseTables,
    rules: RiseRules,
}

fn register_guards(egraph: &mut EGraph) -> RiseGuards {
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
    RiseGuards { neq }
}

fn add_rise_tables(
    egraph: &mut EGraph,
    free_var_funcs: &FreeVarSetExternalFns,
    int_base: BaseValueId,
    sym_base: BaseValueId,
) -> RiseTables {
    let lam = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-lam".to_string(),
        can_subsume: true,
        // Row ids are required for rules that need a stable per-row key
        // (e.g., for deterministic fresh binders).
        row_id: true,
    });
    let app = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-app".to_string(),
        can_subsume: true,
        // Row ids are required for rules that need a stable per-row key
        // (e.g., for deterministic fresh binders).
        row_id: true,
    });
    let var = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-var".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let subst = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-subst".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let free_vars = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: Some(vec![RefinementInput::Block, RefinementInput::Raw]),
        default: DefaultVal::Fail,
        merge: MergeFn::Primitive(free_var_funcs.union, vec![MergeFn::Old, MergeFn::New]),
        name: "rise-free-vars".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let num = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Base(int_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-num".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let sym = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: true,
        schema: vec![ColumnTy::Base(sym_base), ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-sym".to_string(),
        can_subsume: true,
        row_id: false,
    });
    let fresh = egraph.add_table(FunctionConfig {
        participate_in_partition_refinement: false,
        // `rise-fresh` is an auxiliary table keyed by *row id*, not an e-class id.
        // Including it in partition refinement would mix row ids into fingerprints.
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        refinement_inputs: None,
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "rise-fresh".to_string(),
        can_subsume: false,
        row_id: false,
    });
    RiseTables {
        lam,
        app,
        var,
        subst,
        free_vars,
        num,
        sym,
        fresh,
    }
}

fn add_free_in_guard(
    rb: &mut egglog_bridge::RuleBuilder<'_>,
    tables: &RiseTables,
    free_var_funcs: &FreeVarSetExternalFns,
    expr: &QueryEntry,
    binder: &QueryEntry,
) {
    let fv: QueryEntry = rb.new_var(ColumnTy::Id).into();
    rb.query_table(tables.free_vars, &[expr.clone(), fv.clone()], None)
        .unwrap();
    rb.query_prim(
        free_var_funcs.contains,
        &[
            fv,
            binder.clone(),
            QueryEntry::Const {
                val: GUARD_TRUE,
                ty: ColumnTy::Id,
            },
        ],
        ColumnTy::Id,
    )
    .unwrap();
}

fn add_not_free_in_guard(
    rb: &mut egglog_bridge::RuleBuilder<'_>,
    tables: &RiseTables,
    free_var_funcs: &FreeVarSetExternalFns,
    expr: &QueryEntry,
    binder: &QueryEntry,
) {
    let fv: QueryEntry = rb.new_var(ColumnTy::Id).into();
    rb.query_table(tables.free_vars, &[expr.clone(), fv.clone()], None)
        .unwrap();
    rb.query_prim(
        free_var_funcs.not_contains,
        &[
            fv,
            binder.clone(),
            QueryEntry::Const {
                val: GUARD_TRUE,
                ty: ColumnTy::Id,
            },
        ],
        ColumnTy::Id,
    )
    .unwrap();
}

fn build_free_var_rules(
    egraph: &mut EGraph,
    tables: &RiseTables,
    free_var_funcs: &FreeVarSetExternalFns,
    int_base: BaseValueId,
    sym_base: BaseValueId,
) -> Vec<RuleId> {
    let mut rules = Vec::new();

    let var_rule = {
        let mut rb = egraph.new_rule("rise-free-vars-var", true);
        let binder: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let var_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.var, &[binder.clone(), var_id.clone()], None)
            .unwrap();
        let set: QueryEntry = rb
            .call_external_func(
                free_var_funcs.singleton,
                &[binder.clone()],
                ColumnTy::Id,
                || "rise-free-vars-var".to_string(),
            )
            .into();
        rb.set(tables.free_vars, &[var_id, set]);
        rb.build()
    };
    rules.push(var_rule);

    let app_rule = {
        let mut rb = egraph.new_rule("rise-free-vars-app", true);
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
        rb.set(tables.free_vars, &[app_id.clone(), fv_f]);
        rb.set(tables.free_vars, &[app_id, fv_g]);
        rb.build()
    };
    rules.push(app_rule);

    let lam_rule = {
        let mut rb = egraph.new_rule("rise-free-vars-lam", true);
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
                || "rise-free-vars-lam".to_string(),
            )
            .into();
        rb.set(tables.free_vars, &[lam_id, removed]);
        rb.build()
    };
    rules.push(lam_rule);

    let num_rule = {
        let mut rb = egraph.new_rule("rise-free-vars-num", true);
        let raw: QueryEntry = rb.new_var(ColumnTy::Base(int_base)).into();
        let num_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.num, &[raw, num_id.clone()], None)
            .unwrap();
        let empty: QueryEntry = rb
            .call_external_func(free_var_funcs.empty, &[], ColumnTy::Id, || {
                "rise-free-vars-num".to_string()
            })
            .into();
        rb.set(tables.free_vars, &[num_id, empty]);
        rb.build()
    };
    rules.push(num_rule);

    let sym_rule = {
        let mut rb = egraph.new_rule("rise-free-vars-sym", true);
        let raw: QueryEntry = rb.new_var(ColumnTy::Base(sym_base)).into();
        let sym_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.sym, &[raw, sym_id.clone()], None)
            .unwrap();
        let empty: QueryEntry = rb
            .call_external_func(free_var_funcs.empty, &[], ColumnTy::Id, || {
                "rise-free-vars-sym".to_string()
            })
            .into();
        rb.set(tables.free_vars, &[sym_id, empty]);
        rb.build()
    };
    rules.push(sym_rule);

    rules
}

fn build_subst_rules(
    egraph: &mut EGraph,
    tables: &RiseTables,
    guards: &RiseGuards,
    free_var_funcs: &FreeVarSetExternalFns,
) -> Vec<RuleId> {
    let mut rules = Vec::new();

    let subst_var_eq = {
        let mut rb = egraph.new_rule("rise-subst-var-eq", true);
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
        let mut rb = egraph.new_rule("rise-subst-var-neq", true);
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
        rb.set(tables.subst, &[b.clone(), x, e, b.clone()]);
        rb.build()
    };
    rules.push(subst_var_neq);

    let subst_app_left = {
        let mut rb = egraph.new_rule("rise-subst-app-left", true);
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
        add_free_in_guard(&mut rb, tables, free_var_funcs, &f, &x);
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        let f_sub: QueryEntry = rb
            .lookup(tables.subst, &[f, x.clone(), e.clone()], || {
                "rise-subst-app-f".to_string()
            })
            .into();
        let g_sub: QueryEntry = rb
            .lookup(tables.subst, &[g, x.clone(), e.clone()], || {
                "rise-subst-app-g".to_string()
            })
            .into();
        let app_sub: QueryEntry = rb
            .lookup(tables.app, &[f_sub, g_sub], || {
                "rise-subst-app-build".to_string()
            })
            .into();
        rb.set(tables.subst, &[b, x, e, app_sub]);
        rb.build()
    };
    rules.push(subst_app_left);

    let subst_app_right = {
        let mut rb = egraph.new_rule("rise-subst-app-right", true);
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
        add_free_in_guard(&mut rb, tables, free_var_funcs, &g, &x);
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        let f_sub: QueryEntry = rb
            .lookup(tables.subst, &[f, x.clone(), e.clone()], || {
                "rise-subst-app-f".to_string()
            })
            .into();
        let g_sub: QueryEntry = rb
            .lookup(tables.subst, &[g, x.clone(), e.clone()], || {
                "rise-subst-app-g".to_string()
            })
            .into();
        let app_sub: QueryEntry = rb
            .lookup(tables.app, &[f_sub, g_sub], || {
                "rise-subst-app-build".to_string()
            })
            .into();
        rb.set(tables.subst, &[b, x, e, app_sub]);
        rb.build()
    };
    rules.push(subst_app_right);

    let subst_lam_neq = {
        let mut rb = egraph.new_rule("rise-subst-lam-neq", true);
        let b: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let x: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let e: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let cur: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.subst, &[b.clone(), x.clone(), e.clone(), cur], None)
            .unwrap();
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.lam, &[body.clone(), b.clone()], None)
            .unwrap();
        rb.query_prim(
            guards.neq,
            &[
                b.clone(),
                x.clone(),
                QueryEntry::Const {
                    val: GUARD_TRUE,
                    ty: ColumnTy::Id,
                },
            ],
            ColumnTy::Id,
        )
        .unwrap();
        add_free_in_guard(&mut rb, tables, free_var_funcs, &body, &x);
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        let body_sub: QueryEntry = rb
            .lookup(tables.subst, &[body, x.clone(), e.clone()], || {
                "rise-subst-lam-body".to_string()
            })
            .into();
        rb.set(tables.lam, &[body_sub.clone(), b.clone()]);
        rb.set(tables.subst, &[b.clone(), x, e, b]);
        rb.build()
    };
    rules.push(subst_lam_neq);

    let subst_unused = {
        let mut rb = egraph.new_rule("rise-subst-unused", true);
        let b: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let x: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let e: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let cur: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.subst, &[b.clone(), x.clone(), e.clone(), cur], None)
            .unwrap();
        add_not_free_in_guard(&mut rb, tables, free_var_funcs, &b, &x);
        rb.subsume(tables.subst, &[b.clone(), x.clone(), e.clone()]);
        rb.set(tables.subst, &[b.clone(), x, e, b]);
        rb.build()
    };
    rules.push(subst_unused);

    rules
}

fn build_beta_rules(egraph: &mut EGraph, tables: &RiseTables) -> Vec<RuleId> {
    let mut rules = Vec::new();
    let beta = {
        let mut rb = egraph.new_rule("rise-beta", true);
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let arg: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let app_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[f.clone(), arg.clone(), app_id.clone()], None)
            .unwrap();
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.lam, &[body.clone(), f.clone()], None)
            .unwrap();
        let subst_res: QueryEntry = rb
            .lookup(tables.subst, &[body, f, arg], || "rise-beta-subst".to_string())
            .into();
        rb.union(app_id, subst_res);
        rb.build()
    };
    rules.push(beta);
    rules
}

fn build_rise_rules(
    egraph: &mut EGraph,
    tables: &RiseTables,
    _guards: &RiseGuards,
    free_var_funcs: &FreeVarSetExternalFns,
    map_sym: Value,
) -> Vec<RuleId> {
    let mut rules = Vec::new();
    let map_entry = QueryEntry::Const {
        val: map_sym,
        ty: ColumnTy::Id,
    };

    let eta = {
        let mut rb = egraph.new_rule("rise-eta", true);
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let lam_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.lam, &[body.clone(), lam_id.clone()], None)
            .unwrap();
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let v: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[f.clone(), v.clone(), body], None)
            .unwrap();
        rb.query_table(tables.var, &[lam_id.clone(), v], None)
            .unwrap();
        add_not_free_in_guard(&mut rb, tables, free_var_funcs, &f, &lam_id);
        rb.union(lam_id, f);
        rb.build()
    };
    rules.push(eta);

        let map_fusion = {
            let mut rb = egraph.new_rule("rise-map-fusion", true);
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let g: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let arg: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let map_f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(
            tables.app,
            &[map_entry.clone(), f.clone(), map_f.clone()],
            None,
        )
        .unwrap();
        let map_g: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(
            tables.app,
            &[map_entry.clone(), g.clone(), map_g.clone()],
            None,
        )
        .unwrap();
        let map_g_arg: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[map_g.clone(), arg.clone(), map_g_arg.clone()], None)
            .unwrap();
        let top: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let (_atom, row_id) = rb
            .query_table_with_row_id(tables.app, &[map_f.clone(), map_g_arg.clone(), top.clone()])
            .expect("map-fusion missing row id");

        let binder: QueryEntry = rb
            .lookup(tables.fresh, &[row_id], || "rise-map-fusion-binder".to_string())
            .into();
        let var_id: QueryEntry = rb
            .lookup(tables.var, &[binder.clone()], || "rise-map-fusion-var".to_string())
            .into();
        let g_app: QueryEntry = rb
            .lookup(tables.app, &[g, var_id], || "rise-map-fusion-g-app".to_string())
            .into();
        let f_app: QueryEntry = rb
            .lookup(tables.app, &[f, g_app], || "rise-map-fusion-f-app".to_string())
            .into();
        rb.set(tables.lam, &[f_app.clone(), binder.clone()]);
        let map_lam: QueryEntry = rb
            .lookup(tables.app, &[map_entry.clone(), binder.clone()], || {
                "rise-map-fusion-map-lam".to_string()
            })
            .into();
        let out: QueryEntry = rb
            .lookup(tables.app, &[map_lam, arg], || "rise-map-fusion-out".to_string())
            .into();
        rb.union(top, out);
        rb.build()
    };
    rules.push(map_fusion);

        let map_fission = {
            let mut rb = egraph.new_rule("rise-map-fission", true);
        let f: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let gx: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let body: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.query_table(tables.app, &[f.clone(), gx.clone(), body.clone()], None)
            .unwrap();
        let lam_id: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let (_lam_atom, lam_row_id) = rb
            .query_table_with_row_id(tables.lam, &[body, lam_id.clone()])
            .expect("map-fission missing lam row id");
        let map_lam: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let (_map_atom, map_row_id) = rb
            .query_table_with_row_id(tables.app, &[map_entry.clone(), lam_id.clone(), map_lam.clone()])
            .expect("map-fission missing map row id");
        add_not_free_in_guard(&mut rb, tables, free_var_funcs, &f, &lam_id);

        // Fresh binders are memoized by row id to avoid repeated allocations
        // when the same rule match re-fires.
        let inner_binder: QueryEntry = rb
            .lookup(tables.fresh, &[lam_row_id], || "rise-map-fission-inner".to_string())
            .into();
        let inner_var: QueryEntry = rb
            .lookup(tables.var, &[inner_binder.clone()], || "rise-map-fission-inner-var".to_string())
            .into();
        let gx_renamed: QueryEntry = rb
            .lookup(
                tables.subst,
                &[gx.clone(), lam_id.clone(), inner_var.clone()],
                || "rise-map-fission-rename".to_string(),
            )
            .into();
        rb.set(tables.lam, &[gx_renamed.clone(), inner_binder.clone()]);

        let outer_binder: QueryEntry = rb
            .lookup(tables.fresh, &[map_row_id], || "rise-map-fission-outer".to_string())
            .into();
        let outer_var: QueryEntry = rb
            .lookup(tables.var, &[outer_binder.clone()], || "rise-map-fission-outer-var".to_string())
            .into();
        let map_f: QueryEntry = rb
            .lookup(tables.app, &[map_entry.clone(), f], || "rise-map-fission-map-f".to_string())
            .into();
        let map_inner: QueryEntry = rb
            .lookup(
                tables.app,
                &[map_entry.clone(), inner_binder.clone()],
                || "rise-map-fission-map-inner".to_string(),
            )
            .into();
        let map_inner_outer: QueryEntry = rb
            .lookup(
                tables.app,
                &[map_inner, outer_var],
                || "rise-map-fission-map-inner-outer".to_string(),
            )
            .into();
        let body_out: QueryEntry = rb
            .lookup(
                tables.app,
                &[map_f, map_inner_outer],
                || "rise-map-fission-body".to_string(),
            )
            .into();
        rb.set(tables.lam, &[body_out, outer_binder.clone()]);
        rb.union(map_lam, outer_binder);
        rb.build()
    };
    rules.push(map_fission);

    rules
}

pub(crate) fn setup_rise(egraph: &mut EGraph) -> RiseEnv {
    let int_base = egraph.base_values_mut().register_type::<i64>();
    let sym_base = egraph.base_values_mut().register_type::<String>();
    let free_var_funcs = register_free_var_set_functions(egraph);
    let guards = register_guards(egraph);
    let tables = add_rise_tables(egraph, &free_var_funcs, int_base, sym_base);
    let map_sym = lookup_symbol(egraph, &tables, "map".to_string());
    let free_vars = build_free_var_rules(egraph, &tables, &free_var_funcs, int_base, sym_base);
    let subst = build_subst_rules(egraph, &tables, &guards, &free_var_funcs);
    let beta = build_beta_rules(egraph, &tables);
    let rise = build_rise_rules(egraph, &tables, &guards, &free_var_funcs, map_sym);
    RiseEnv {
        tables,
        rules: RiseRules {
            free_vars,
            subst,
            beta,
            rise,
        },
    }
}

fn lookup_symbol(egraph: &mut EGraph, tables: &RiseTables, name: String) -> Value {
    let sym_action = TableAction::new(egraph, tables.sym);
    let base_val = egraph.base_values().get(name);
    egraph
        .with_execution_state(|state| sym_action.lookup(state, &[base_val]))
        .expect("symbol lookup failed")
}

#[derive(Debug, Error)]
pub(crate) enum ParseError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unexpected token `{0}`")]
    UnexpectedToken(String),
    #[error("expected symbol, found `{0}`")]
    ExpectedSymbol(String),
    #[error("invalid integer literal `{0}`")]
    InvalidInteger(String),
    #[error("invalid form: {0}")]
    InvalidForm(String),
    #[error("unbound variable `{0}`")]
    UnboundVariable(String),
}

#[derive(Clone, Debug)]
enum Token {
    LParen,
    RParen,
    Atom(String),
}

#[derive(Clone, Debug)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

pub(crate) fn add_expr_from_sexp(
    egraph: &mut EGraph,
    env: &RiseEnv,
    input: &str,
) -> Result<Value, ParseError> {
    let sexp = parse_sexp(input)?;
    let mut builder = Builder::new(egraph, &env.tables);
    let expr = builder.eval(&sexp)?;
    egraph.flush_updates();
    Ok(expr)
}

fn parse_sexp(input: &str) -> Result<Sexp, ParseError> {
    let tokens = tokenize(input)?;
    let mut idx = 0;
    let expr = parse_expr(&tokens, &mut idx)?;
    if idx != tokens.len() {
        return Err(ParseError::UnexpectedToken(format!("{:?}", tokens[idx])));
    }
    Ok(expr)
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut atom = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' {
                        break;
                    }
                    atom.push(c);
                    chars.next();
                }
                if atom.is_empty() {
                    return Err(ParseError::UnexpectedToken(ch.to_string()));
                }
                tokens.push(Token::Atom(atom));
            }
        }
    }
    Ok(tokens)
}

fn parse_expr(tokens: &[Token], idx: &mut usize) -> Result<Sexp, ParseError> {
    let Some(token) = tokens.get(*idx) else {
        return Err(ParseError::UnexpectedEof);
    };
    match token {
        Token::LParen => {
            *idx += 1;
            let mut elems = Vec::new();
            loop {
                let Some(token) = tokens.get(*idx) else {
                    return Err(ParseError::UnexpectedEof);
                };
                match token {
                    Token::RParen => {
                        *idx += 1;
                        break;
                    }
                    _ => elems.push(parse_expr(tokens, idx)?),
                }
            }
            Ok(Sexp::List(elems))
        }
        Token::RParen => Err(ParseError::UnexpectedToken(")".to_string())),
        Token::Atom(atom) => {
            *idx += 1;
            Ok(Sexp::Atom(atom.clone()))
        }
    }
}

struct Builder<'a> {
    egraph: &'a mut EGraph,
    tables: &'a RiseTables,
    scopes: Vec<HashMap<String, Value>>,
}

impl<'a> Builder<'a> {
    fn new(egraph: &'a mut EGraph, tables: &'a RiseTables) -> Self {
        Self {
            egraph,
            tables,
            scopes: Vec::new(),
        }
    }

    fn eval(&mut self, expr: &Sexp) -> Result<Value, ParseError> {
        match expr {
            Sexp::Atom(atom) => self.eval_atom(atom),
            Sexp::List(list) => self.eval_list(list),
        }
    }

    fn eval_atom(&mut self, atom: &str) -> Result<Value, ParseError> {
        if let Some(val) = self.lookup_var(atom) {
            return self.lookup_var_expr(val);
        }
        if let Ok(value) = atom.parse::<i64>() {
            return self.lookup_num_expr(value);
        }
        self.lookup_sym_expr(atom.to_string())
    }

    fn eval_list(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        if list.is_empty() {
            return Err(ParseError::InvalidForm("empty list".to_string()));
        }
        if let Sexp::Atom(head) = &list[0] {
            if head == "lam" {
                return self.eval_lam(list);
            }
            if head == "var" {
                return self.eval_var(list);
            }
            if head == "num" {
                return self.eval_num(list);
            }
            if head == "app" {
                return self.eval_app(list);
            }
        }

        let mut iter = list.iter();
        let mut expr = self.eval(
            iter.next()
                .expect("non-empty list should have a head"),
        )?;
        for arg in iter {
            let arg_id = self.eval(arg)?;
            expr = self.lookup_app_expr(expr, arg_id)?;
        }
        Ok(expr)
    }

    fn eval_lam(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        let [_, name, body] = list else {
            return Err(ParseError::InvalidForm(
                "lam expects (lam <name> <body>)".to_string(),
            ));
        };
        let name = match name {
            Sexp::Atom(name) => name.as_str(),
            other => {
                return Err(ParseError::ExpectedSymbol(format!("{other:?}")));
            }
        };
        let lam_id = self.egraph.fresh_id();
        self.scopes.push(HashMap::from([(name.to_string(), lam_id)]));
        let body_id = self.eval(body)?;
        self.scopes.pop();

        let mut lam_action = TableAction::new(self.egraph, self.tables.lam);
        let lam_id_copy = lam_id;
        self.egraph.with_execution_state(|state| {
            lam_action.insert(state, [body_id, lam_id_copy].into_iter());
        });

        Ok(lam_id)
    }

    fn eval_var(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        let [_, name] = list else {
            return Err(ParseError::InvalidForm(
                "var expects (var <name>)".to_string(),
            ));
        };
        let name = match name {
            Sexp::Atom(name) => name.as_str(),
            other => {
                return Err(ParseError::ExpectedSymbol(format!("{other:?}")));
            }
        };
        let Some(binder) = self.lookup_var(name) else {
            return Err(ParseError::UnboundVariable(name.to_string()));
        };
        self.lookup_var_expr(binder)
    }

    fn eval_num(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        let [_, value] = list else {
            return Err(ParseError::InvalidForm(
                "num expects (num <int>)".to_string(),
            ));
        };
        let value = match value {
            Sexp::Atom(atom) => atom
                .parse::<i64>()
                .map_err(|_| ParseError::InvalidInteger(atom.to_string()))?,
            other => {
                return Err(ParseError::ExpectedSymbol(format!("{other:?}")));
            }
        };
        self.lookup_num_expr(value)
    }

    fn eval_app(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        let [_, f, arg] = list else {
            return Err(ParseError::InvalidForm(
                "app expects (app <f> <arg>)".to_string(),
            ));
        };
        let f_id = self.eval(f)?;
        let arg_id = self.eval(arg)?;
        self.lookup_app_expr(f_id, arg_id)
    }

    fn lookup_var(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(*val);
            }
        }
        None
    }

    fn lookup_var_expr(&mut self, binder: Value) -> Result<Value, ParseError> {
        let var_action = TableAction::new(self.egraph, self.tables.var);
        let var_id = self
            .egraph
            .with_execution_state(|state| var_action.lookup(state, &[binder]));
        var_id.ok_or_else(|| ParseError::InvalidForm("var lookup failed".to_string()))
    }

    fn lookup_num_expr(&mut self, value: i64) -> Result<Value, ParseError> {
        let num_action = TableAction::new(self.egraph, self.tables.num);
        let base_val = self.egraph.base_values().get(value);
        let num_id = self
            .egraph
            .with_execution_state(|state| num_action.lookup(state, &[base_val]));
        num_id.ok_or_else(|| ParseError::InvalidForm("num lookup failed".to_string()))
    }

    fn lookup_sym_expr(&mut self, name: String) -> Result<Value, ParseError> {
        let sym_action = TableAction::new(self.egraph, self.tables.sym);
        let base_val = self.egraph.base_values().get(name);
        let sym_id = self
            .egraph
            .with_execution_state(|state| sym_action.lookup(state, &[base_val]));
        sym_id.ok_or_else(|| ParseError::InvalidForm("sym lookup failed".to_string()))
    }

    fn lookup_app_expr(&mut self, fun: Value, arg: Value) -> Result<Value, ParseError> {
        let app_action = TableAction::new(self.egraph, self.tables.app);
        let app_id = self
            .egraph
            .with_execution_state(|state| app_action.lookup(state, &[fun, arg]));
        app_id.ok_or_else(|| ParseError::InvalidForm("app lookup failed".to_string()))
    }
}

#[derive(Parser, Debug)]
#[command(name = "binder-experiments")]
#[command(about = "Rise benchmark runner (cyclic lambdas + partition refinement)")]
struct RiseArgs {
    /// N: number of nested maps
    #[arg(long, default_value_t = 2)]
    n: usize,
    /// M: half the number of chained functions
    #[arg(long, default_value_t = 2)]
    m: usize,
    /// O: number of function parameters
    #[arg(long, default_value_t = 2)]
    o: usize,
    /// Whether to wrap bound variables with (var ...)
    #[arg(long, default_value_t = true)]
    vars: bool,
    /// Max number of iterations
    #[arg(long, default_value_t = 40)]
    iters: usize,
    /// Run partition refinement every N iterations (0 = never)
    #[arg(long, default_value_t = 1)]
    pr_every: usize,
    /// Enable partition refinement
    #[arg(long, default_value_t = true)]
    pr: bool,
    /// Keep iterating until lhs == rhs (or max iters), even if no changes occur
    #[arg(long, default_value_t = false)]
    until_equal: bool,
    /// Comma-separated list of rule groups to run (free_vars,subst,beta,rise)
    #[arg(long, value_delimiter = ',', default_value = "free_vars,subst,beta,rise")]
    rules: Vec<String>,
}

#[derive(Debug)]
struct RiseBenchConfig {
    n: usize,
    m: usize,
    o: usize,
    vars: bool,
    iters: usize,
    pr_every: usize,
    use_partition_refinement: bool,
    until_equal: bool,
    run_free_vars: bool,
    run_subst: bool,
    run_beta: bool,
    run_rise: bool,
}

pub(crate) fn run_rise_benchmark() {
    let args = RiseArgs::parse();
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let mut rules = RuleToggles::default();
    for name in &args.rules {
        rules.enable(name);
    }
    let config = RiseBenchConfig {
        n: args.n,
        m: args.m,
        o: args.o,
        vars: args.vars,
        iters: args.iters,
        pr_every: args.pr_every,
        use_partition_refinement: args.pr,
        until_equal: args.until_equal,
        run_free_vars: rules.free_vars,
        run_subst: rules.subst,
        run_beta: rules.beta,
        run_rise: rules.rise,
    };
    run_rise_benchmark_with_config(&config);
}

#[derive(Default)]
struct RuleToggles {
    free_vars: bool,
    subst: bool,
    beta: bool,
    rise: bool,
}

impl RuleToggles {
    fn enable(&mut self, name: &str) {
        match name {
            "free_vars" => self.free_vars = true,
            "subst" => self.subst = true,
            "beta" => self.beta = true,
            "rise" => self.rise = true,
            _ => {}
        }
    }
}

fn run_rise_benchmark_with_config(config: &RiseBenchConfig) {
    let mut egraph = if config.use_partition_refinement {
        EGraph::with_partition_refinement()
    } else {
        EGraph::default()
    };
    let env = setup_rise(&mut egraph);
    let (lhs, rhs) = bench_rise_terms(config.n, config.m, config.o, config.vars);
    let lhs_id = add_expr_from_sexp(&mut egraph, &env, &lhs).expect("lhs parse failed");
    let rhs_id = add_expr_from_sexp(&mut egraph, &env, &rhs).expect("rhs parse failed");

    info!("Rise benchmark (bench_rise):");
    info!("  n={}, m={}, o={}, vars={}", config.n, config.m, config.o, config.vars);
    info!(
        "  rules: free_vars={} subst={} beta={} rise={}",
        config.run_free_vars, config.run_subst, config.run_beta, config.run_rise
    );
    info!(
        "  pr: enabled={} pr_every={}",
        config.use_partition_refinement, config.pr_every
    );
    info!("  iters={} until_equal={}", config.iters, config.until_equal);

    let start = std::time::Instant::now();
    let mut iterations = 0usize;
    let mut found = false;

    while iterations < config.iters {
        let mut changed = false;
        if config.run_free_vars {
            changed |= run_group(&mut egraph, &env.rules.free_vars);
        }
        if config.run_subst {
            changed |= run_group(&mut egraph, &env.rules.subst);
        }
        if config.run_beta {
            changed |= run_group(&mut egraph, &env.rules.beta);
        }
        if config.run_rise {
            changed |= run_group(&mut egraph, &env.rules.rise);
        }
        if config.use_partition_refinement
            && config.pr_every > 0
            && (iterations + 1) % config.pr_every == 0
        {
            let pr_changed = egraph
                .run_hash_partition_refinement()
                .expect("partition refinement failed");
            changed |= pr_changed;
        }
        iterations += 1;
        if egraph.get_canon_repr(lhs_id, ColumnTy::Id)
            == egraph.get_canon_repr(rhs_id, ColumnTy::Id)
        {
            found = true;
            break;
        }
        if !config.until_equal && !changed {
            break;
        }
    }

    let elapsed = start.elapsed();
    let sizes = table_sizes(&egraph, &env.tables);
    let total_enodes = sizes
        .iter()
        .filter(|(name, _)| matches!(*name, "lam" | "app" | "var" | "num" | "sym"))
        .map(|(_, size)| *size)
        .sum::<usize>();

    info!("Results:");
    info!("  iterations={} found={}", iterations, found);
    info!("  elapsed_ms={}", elapsed.as_millis());
    info!("  total_enodes={}", total_enodes);
    info!("  table_sizes:");
    for (name, size) in sizes {
        info!("    {}={}", name, size);
    }
    info!(
        "Summary: n={} m={} o={} iters={} found={} pr={} pr_every={} until_equal={} total_enodes={} elapsed_ms={}",
        config.n,
        config.m,
        config.o,
        iterations,
        found,
        config.use_partition_refinement,
        config.pr_every,
        config.until_equal,
        total_enodes,
        elapsed.as_millis()
    );
}

fn run_group(egraph: &mut EGraph, rules: &[RuleId]) -> bool {
    if rules.is_empty() {
        return false;
    }
    let report = egraph.run_rules(rules).expect("rule execution failed");
    report.changed()
}

fn table_sizes(egraph: &EGraph, tables: &RiseTables) -> Vec<(&'static str, usize)> {
    vec![
        ("lam", egraph.table_size(tables.lam)),
        ("app", egraph.table_size(tables.app)),
        ("var", egraph.table_size(tables.var)),
        ("subst", egraph.table_size(tables.subst)),
        ("free_vars", egraph.table_size(tables.free_vars)),
        ("num", egraph.table_size(tables.num)),
        ("sym", egraph.table_size(tables.sym)),
        ("fresh", egraph.table_size(tables.fresh)),
    ]
}

fn bench_rise_terms(n: usize, m: usize, o: usize, vars: bool) -> (String, String) {
    assert!(n >= 1);
    assert!(m >= 1);
    let mut fresh = 0usize;

    let var_wrapper = |x: &str| {
        if vars {
            // Mirror slotted's generator: wrap with `$` prefix when vars are enabled.
            format!("(var ${x})")
        } else {
            x.to_string()
        }
    };

    let mut fresh_slot = || {
        fresh += 1;
        format!("${}", fresh)
    };

    let comp = |a: String, b: String, fresh_slot: &mut dyn FnMut() -> String| {
        let x = fresh_slot();
        format!("(lam {x} (app {a} (app {b} (var {x}))))")
    };

    let map_ = |x: String| format!("(app map {x})");

    let fn_with_args = |f: String, o: usize, var_wrapper: &dyn Fn(&str) -> String| {
        let mut out = f;
        for i in 1..=o {
            let v = var_wrapper(&format!("p{i}"));
            out = format!("(app {out} {v})");
        }
        out
    };

    let chained_fns = |range: std::ops::RangeInclusive<usize>,
                       o: usize,
                       fresh_slot: &mut dyn FnMut() -> String,
                       var_wrapper: &dyn Fn(&str) -> String| {
        let mut fns: Vec<String> = range
            .map(|i| fn_with_args(var_wrapper(&format!("fn{i}")), o, var_wrapper))
            .collect();
        if fns.len() == 1 {
            return fns.pop().expect("non-empty");
        }
        let fresh = fresh_slot();
        let mut out = format!("(var {fresh})");
        for f in fns {
            out = format!("(app {f} {out})");
        }
        format!("(lam {fresh} {out})")
    };

    let nested_maps = |n: usize, mut arg: String| {
        for _ in 0..n {
            arg = map_(arg);
        }
        arg
    };

    let nest_lams = |mut arg: String, m: usize, o: usize| {
        if vars {
            for i in (1..=o).rev() {
                arg = format!("(lam $p{i} {arg})");
            }
            for i in (1..=2 * m).rev() {
                arg = format!("(lam $fn{i} {arg})");
            }
        }
        arg
    };

    let lhs_base = chained_fns(1..=2 * m, o, &mut fresh_slot, &var_wrapper);
    let lhs = nest_lams(nested_maps(n, lhs_base), m, o);

    let lhs_funcs = chained_fns(m + 1..=2 * m, o, &mut fresh_slot, &var_wrapper);
    let rhs_left = nested_maps(n, lhs_funcs);
    let rhs_right = nested_maps(
        n,
        chained_fns(1..=m, o, &mut fresh_slot, &var_wrapper),
    );
    let rhs = nest_lams(comp(rhs_left, rhs_right, &mut fresh_slot), m, o);

    (lhs, rhs)
}

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rise_map_fission_basic() {
        // This mirrors the fission example from the slotted egraphs artifact.
        // We rely on `rise-fresh` to pick deterministic binders per matched row,
        // avoiding infinite alpha-renaming during saturation.
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_rise(&mut egraph);
        let lhs = "(app map (lam $42 (app f5 (app f4 (app f3 (app f2 (app f1 (var $42))))))))";
        let rhs = "(lam $1 (app (app map (lam $42 (app f5 (app f4 (app f3 (var $42)))))) (app (app map (lam $42 (app f2 (app f1 (var $42))))) (var $1))))";
        let lhs_id = add_expr_from_sexp(&mut egraph, &env, lhs).expect("lhs parse failed");
        let rhs_id = add_expr_from_sexp(&mut egraph, &env, rhs).expect("rhs parse failed");

        let mut rules = Vec::new();
        rules.extend(env.rules.free_vars.iter().copied());
        rules.extend(env.rules.subst.iter().copied());
        rules.extend(env.rules.beta.iter().copied());
        rules.extend(env.rules.rise.iter().copied());

        for _ in 0..25 {
            run_rules(&mut egraph, &rules, false, Some(1));
            if egraph.get_canon_repr(lhs_id, ColumnTy::Id)
                == egraph.get_canon_repr(rhs_id, ColumnTy::Id)
            {
                return;
            }
        }

        assert_eq!(
            egraph.get_canon_repr(lhs_id, ColumnTy::Id),
            egraph.get_canon_repr(rhs_id, ColumnTy::Id)
        );
    }

    #[test]
    fn rise_map_fusion_basic() {
        // Fusion creates a new lambda with a fresh binder, so we still need
        // partition refinement to merge alpha-equivalent results.
        let mut egraph = EGraph::with_partition_refinement();
        let env = setup_rise(&mut egraph);
        let lhs = "(app (app map f) (app (app map g) xs))";
        let rhs = "(app (app map (lam x (app f (app g (var x))))) xs)";
        let lhs_id = add_expr_from_sexp(&mut egraph, &env, lhs).expect("lhs parse failed");
        let rhs_id = add_expr_from_sexp(&mut egraph, &env, rhs).expect("rhs parse failed");

        // Only run the fusion rule to keep this test focused.
        let rules = [env.rules.rise[1]];

        for _ in 0..25 {
            // Fusion introduces a fresh binder, so we need partition refinement to
            // merge alpha-equivalent lambdas.
            run_rules(&mut egraph, &rules, true, Some(1));
            if egraph.get_canon_repr(lhs_id, ColumnTy::Id)
                == egraph.get_canon_repr(rhs_id, ColumnTy::Id)
            {
                return;
            }
        }

        assert_eq!(
            egraph.get_canon_repr(lhs_id, ColumnTy::Id),
            egraph.get_canon_repr(rhs_id, ColumnTy::Id)
        );
    }
}
