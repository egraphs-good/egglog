//! DuckDB-backed executor for a small subset of egglog's resolved IR.
//!
//! Phase 1.1 scope: relations + functions with outputs and merge
//! modes (`:merge old`, `:merge new`). No term encoding, no UF, no
//! primitives. See `../duckdb-backend-plan.md` for the full plan.
//!
//! Design notes:
//! - One DuckDB table per registered relation/function.
//! - Every table carries a `ts BIGINT NOT NULL` column for seminaive.
//! - `next_ts` and `last_run_at_<rule>` live in Rust state; they are
//!   bind parameters in generated SQL, never database rows.
//! - Each rule with N function-table atoms compiles to N seminaive
//!   variants (one per focused atom), emitted as separate prepared
//!   `INSERT INTO target SELECT ...` statements with appropriate
//!   `ON CONFLICT` clauses depending on merge mode.

use anyhow::{Result, anyhow};
use duckdb::{Connection, ToSql};
use std::collections::HashMap;

mod compile;

/// Quote a SQL identifier with double quotes, escaping any embedded
/// double quote. Necessary because egglog identifiers can contain
/// `@`, `$`, etc., which DuckDB rejects in unquoted form.
///
/// DuckDB compares quoted identifiers case-insensitively (so
/// `"AConst"` and `"aConst"` collide as table names). We encode the
/// case bits with a two-step escape so distinct case-shapes always
/// produce distinct lowered identifiers:
///   1. `_` → `__` (escape literal underscore so it can't be
///      confused with the uppercase marker).
///   2. ASCII uppercase `X` → `_<lowercase x>`.
pub fn q(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for c in name.chars() {
        if c == '_' {
            out.push('_');
            out.push('_');
        } else if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    format!("\"{}\"", out.replace('"', "\"\""))
}

/// Format a list of comma-separated SQL fragments as a prefix that
/// is followed by something else, with a trailing comma when the
/// list is non-empty and an empty string otherwise. Used to safely
/// concatenate column lists with the always-present trailing `ts`
/// column without producing `(, ts)` when there are no other cols.
/// Bind a rule template by substituting `?1` -> `last`,
/// `?2` -> `cur`, and execute it. Values are i64s, so direct
/// inlining is safe.
///
/// We tried `prepare_cached` with proper positional binds, but
/// duckdb-rust 1.5 didn't actually keep the planned representation
/// across calls in our setup — overall wall time roughly doubled.
/// Plain `execute` with string substitution beats it.
pub(crate) fn exec_bound(
    conn: &Connection,
    sql: &str,
    last: i64,
    cur: i64,
) -> Result<usize> {
    let bound = sql
        .replace("?1", &last.to_string())
        .replace("?2", &cur.to_string());
    Ok(conn.execute(&bound, [])?)
}

/// Execute all variants of a single rule for one seminaive
/// iteration. Pulled out of `run_iteration_in*` so both entry points
/// share the same shape and so per-rule timing accounting lives in
/// one place. Borrows `self.rules` (read-only) and the timing fields
/// disjointly — the caller passes those as `&mut` references.
#[allow(clippy::too_many_arguments)]
fn run_rule_variants(
    rule: &CompiledRule,
    last: i64,
    cur: i64,
    conn: &Connection,
    time_mat_ns: &mut u64,
    time_mat_act_ns: &mut u64,
    time_act_ns: &mut u64,
    rules_affected: &mut u64,
    rule_perf_ns: &mut HashMap<String, (u64, u64)>,
    table_watermarks: &mut HashMap<String, i64>,
    rules_skipped: &mut u64,
) -> Result<usize> {
    // Watermark gate: if no body table has had inserts since this
    // rule last ran, every variant's seminaive predicate is empty.
    // Skip the whole rule. Falls through for rules with no body
    // (shouldn't happen, but cheap to handle).
    if !rule.body_tables.is_empty() {
        let mut any_fresh = false;
        for t in &rule.body_tables {
            let wm = table_watermarks.get(t).copied().unwrap_or(0);
            if wm >= last {
                any_fresh = true;
                break;
            }
        }
        if !any_fresh {
            *rules_skipped = rules_skipped.wrapping_add(1);
            return Ok(0);
        }
    }

    let mut total: usize = 0;
    let mut rule_mat_ns: u64 = 0;
    let mut rule_act_ns: u64 = 0;
    let trace = std::env::var("DUCK_TRACE_RULE_AFFECTED").is_ok();
    let trace_sql = std::env::var("DUCK_TRACE_SQL").is_ok();
    for variant in &rule.variants {
        if let Some(mat_sql_template) = &variant.materialize {
            if trace_sql {
                eprintln!("[duck/mat] {mat_sql_template}");
            }
            let t0 = std::time::Instant::now();
            exec_bound(conn, mat_sql_template, last, cur)?;
            let dt = t0.elapsed().as_nanos() as u64;
            *time_mat_ns = time_mat_ns.wrapping_add(dt);
            rule_mat_ns = rule_mat_ns.wrapping_add(dt);
        }
        for act in &variant.actions {
            if trace_sql {
                eprintln!(
                    "[duck/{}] {}",
                    if variant.materialize.is_some() { "mat-act" } else { "act" },
                    act.sql,
                );
            }
            let t0 = std::time::Instant::now();
            let n = exec_bound(conn, &act.sql, last, cur)?;
            let dt = t0.elapsed().as_nanos() as u64;
            if variant.materialize.is_some() {
                *time_mat_act_ns = time_mat_act_ns.wrapping_add(dt);
            } else {
                *time_act_ns = time_act_ns.wrapping_add(dt);
            }
            rule_act_ns = rule_act_ns.wrapping_add(dt);
            *rules_affected = rules_affected.wrapping_add(n as u64);
            total += n;
            if n > 0 {
                if let Some(target) = &act.target {
                    let e = table_watermarks.entry(target.clone()).or_insert(0);
                    if cur > *e {
                        *e = cur;
                    }
                }
                if trace {
                    eprintln!("[duck/rule_n] {} +{n}", rule.name);
                }
            }
        }
    }
    if rule_mat_ns != 0 || rule_act_ns != 0 {
        let e = rule_perf_ns.entry(rule.name.clone()).or_insert((0, 0));
        e.0 = e.0.wrapping_add(rule_mat_ns);
        e.1 = e.1.wrapping_add(rule_act_ns);
    }
    Ok(total)
}

pub(crate) fn prefix_with_comma(parts: &[String]) -> String {
    if parts.is_empty() {
        String::new()
    } else {
        format!("{}, ", parts.join(", "))
    }
}

/// The (very small) set of column types we currently understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnTy {
    I64,
    Bool,
    F64,
    Str,
}

impl ColumnTy {
    fn sql(self) -> &'static str {
        match self {
            ColumnTy::I64 => "BIGINT",
            ColumnTy::Bool => "BOOLEAN",
            ColumnTy::F64 => "DOUBLE",
            ColumnTy::Str => "VARCHAR",
        }
    }
}

/// Merge mode for functions with outputs. Mirrors egglog's
/// `:merge old` / `:merge new` keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    /// First-set wins. `ON CONFLICT DO NOTHING`.
    Old,
    /// Latest-set wins. `ON CONFLICT DO UPDATE` of the output and ts.
    New,
}

/// A literal value usable in seed inserts and `check`/`lookup`.
#[derive(Debug, Clone)]
pub enum Literal {
    I64(i64),
    Bool(bool),
    F64(f64),
    Str(String),
}

impl ToSql for Literal {
    fn to_sql(&self) -> duckdb::Result<duckdb::types::ToSqlOutput<'_>> {
        match self {
            Literal::I64(i) => i.to_sql(),
            Literal::Bool(b) => b.to_sql(),
            Literal::F64(f) => f.to_sql(),
            Literal::Str(s) => s.as_str().to_sql(),
        }
    }
}

/// A term in a rule body or action.
#[derive(Debug, Clone)]
pub enum Term {
    Var(String),
    Lit(Literal),
    /// A primitive expression: arithmetic (`+`, `-`, `*`, `/`),
    /// comparison (`<`, `<=`, `>`, `>=`, `=`, `!=`), or boolean
    /// (`and`, `or`, `not`). The op name is mapped to a SQL
    /// operator at codegen; see `compile.rs::prim_sql`.
    Prim(String, Vec<Term>),
    /// Read a function's output value as an expression. Compiles to
    /// `(SELECT c<out> FROM <name> WHERE c0 = arg0 AND … LIMIT 1)`.
    /// Used for term-encoding's globals: `(let v (foo 1 2))` becomes
    /// a synthetic `(function v () Sort :no-merge)` plus `(set (v)
    /// ...)`, and later references `(v)` are reads of this function.
    /// Cannot be used on relations (no output column).
    FuncCall { name: String, args: Vec<Term> },
}

impl Term {
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }
    pub fn i64(v: i64) -> Self {
        Term::Lit(Literal::I64(v))
    }
    pub fn prim(op: impl Into<String>, args: Vec<Term>) -> Self {
        Term::Prim(op.into(), args)
    }
}

/// A body atom of a rule.
#[derive(Debug, Clone)]
pub enum Atom {
    /// A function-table atom. `args.len()` must match the function's
    /// full arity (inputs for relations; inputs + 1 for functions
    /// with outputs).
    Func { name: String, args: Vec<Term> },
    /// A pure-primitive constraint: the term must evaluate to true.
    /// Examples: `(< x 5)`, `(!= a b)`. Compiles to a SQL WHERE
    /// constraint.
    Filter(Term),
    /// Bind `var` to the value of `expr` for the rest of the body
    /// and any subsequent actions. Used for `(= var (primitive ...))`
    /// patterns in egglog rule bodies. Compiles to nothing on its
    /// own — it just extends the body's variable→SQL binding.
    Bind { var: String, expr: Term },
}

/// A rule action.
#[derive(Debug, Clone)]
pub enum Action {
    /// Insert a row. The trailing arg is the output value for
    /// functions; for relations all args are key columns.
    Insert { name: String, args: Vec<Term> },
    /// Delete a row matched by its key columns. For functions, the
    /// args are the input columns only (output is ignored). For
    /// relations, args are all the columns.
    Delete { name: String, key_args: Vec<Term> },
    /// Allocate a fresh EqSort ID via the constructor table `name`,
    /// inserting `(args..., fresh_id)`, and bind `var` to the
    /// allocated ID for use in subsequent actions of the same rule.
    /// Compiles into a `nextval('seq')` column in the rule's
    /// materialized match table; subsequent actions reference `var`
    /// as a regular term.
    LetCtor {
        var: String,
        name: String,
        args: Vec<Term>,
    },
    /// Bind `var` to a pure expression (no constructor allocation,
    /// no insert). Compiles into a non-side-effecting column of the
    /// rule's materialized match table. Subsequent actions reference
    /// `var` as a regular term.
    LetExpr { var: String, expr: Term },
    /// `(panic msg)` action: raise a runtime error if the rule body
    /// matches at all. Compiles to `SELECT error('msg') FROM <body>
    /// LIMIT 1` — DuckDB only evaluates `error()` on returned rows, so
    /// no match = no error.
    Panic { msg: String },
}

/// A whole rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    /// Ruleset the rule lives in. Empty string for the default
    /// ruleset. `run_iteration_for(ruleset)` runs only rules whose
    /// ruleset matches.
    pub ruleset: String,
    pub body: Vec<Atom>,
    pub actions: Vec<Action>,
}

/// Internal: schema for a single registered function. Public so
/// frontend diagnostic dumps can read it; not part of the stable
/// API.
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// All schema columns (inputs followed by output/ID, if any).
    pub cols: Vec<ColumnTy>,
    /// Number of "input" columns from the user perspective. For
    /// relations this is `cols.len()`; for functions and EqSort
    /// constructors it's `cols.len() - 1`.
    pub inputs_len: usize,
    /// `Some` if this is a function with an output column; `None`
    /// for relations and EqSort constructors.
    pub merge: Option<MergeMode>,
    /// True iff this is an EqSort constructor: PK covers all cols
    /// (so multiple distinct IDs per input set are allowed) and
    /// `allocate_and_insert` is the intended insertion path.
    pub eq_sort_ctor: bool,
}

impl FunctionInfo {
    pub(crate) fn arity(&self) -> usize {
        self.cols.len()
    }
    pub(crate) fn has_output(&self) -> bool {
        self.merge.is_some()
    }
}

/// The executor.
pub struct EGraph {
    conn: Connection,
    pub functions: HashMap<String, FunctionInfo>,
    rules: Vec<CompiledRule>,
    next_ts: i64,
    /// Per source-rule "last run at" — the ts at which it last ran.
    last_run_at: HashMap<String, i64>,
    /// Cumulative count of rows affected by every rule action's SQL
    /// across all iterations. The frontend uses snapshots of this
    /// counter to detect saturation precisely (vs total tuple count,
    /// which can balance to zero when deletes match inserts).
    rules_affected: u64,
    /// Per-category nanosecond counters for SQL `execute` calls.
    /// Exposed by `DUCK_PERF_DUMP=1` so we can see where the per-
    /// iteration wall time goes. Not part of the public API; debug
    /// hook only.
    time_mat_ns: u64,
    time_mat_act_ns: u64,
    time_act_ns: u64,
    /// (mat_ns, act_ns) per rule name. Populated alongside the
    /// global accumulators. Used by `DUCK_PERF_DUMP` to surface
    /// which individual rules dominate.
    rule_perf_ns: HashMap<String, (u64, u64)>,
    /// Ruleset of each rule, mirrored so `DUCK_PERF_DUMP` can roll
    /// per-rule timings up to per-ruleset.
    rule_to_ruleset: HashMap<String, String>,
    /// Per-table "max `ts` of any insert" watermark. Bumped on
    /// every successful row-affecting INSERT (top-level seeds and
    /// rule actions alike). The seminaive predicate fires on rows
    /// with `ts >= last_run_at`, so a rule whose every body table
    /// has `watermark < last_run_at` cannot match anything — we
    /// skip it entirely.
    table_watermarks: HashMap<String, i64>,
    /// Count of rule firings short-circuited by the watermark gate.
    /// Surfaced by `DUCK_PERF_DUMP`.
    rules_skipped: u64,
}

struct CompiledRule {
    name: String,
    ruleset: String,
    variants: Vec<CompiledVariant>,
    /// Distinct function-table names appearing in the rule body. The
    /// runner skips the whole rule when none of these tables has had
    /// rows inserted since the rule's `last_run_at` — there's nothing
    /// for any variant to match. Pure functional / filter atoms
    /// don't appear here.
    body_tables: Vec<String>,
}

/// One seminaive variant of a rule, ready to execute.
pub(crate) struct CompiledVariant {
    /// Optional CREATE TEMP TABLE SQL that materializes the body
    /// matches (with any `nextval()`-allocated ids per LetCtor).
    /// `None` for variants whose actions are all simple inserts/
    /// deletes — those just run directly.
    pub(crate) materialize: Option<String>,
    /// Action SQLs to run in order. When `materialize` is Some, each
    /// action SELECTs from the temp table; otherwise from the body.
    pub(crate) actions: Vec<CompiledAction>,
}

/// One rule action paired with the table it writes to (so the
/// watermark tracker can bump that table's high-water-mark after a
/// successful row-affecting execute). `target` is `None` for no-op
/// LetCtor/LetExpr placeholders and for Panic.
pub(crate) struct CompiledAction {
    pub(crate) sql: String,
    pub(crate) target: Option<String>,
}

impl EGraph {
    pub fn new() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        // Tuning: DuckDB defaults to row-order preservation, which
        // uses extra memory to track insertion order through joins.
        // We don't care about row order — egglog is set-semantics.
        // Disabling drops a sizeable chunk of per-iteration overhead
        // on workloads that do many INSERT…SELECT.
        conn.execute("SET preserve_insertion_order = false", [])?;
        // Determinism: parallel execution can produce rows from a
        // SELECT in any order. With hash-cons (or any rule action
        // that side-effects via nextval()), per-row evaluation order
        // determines which sequence values get burned for which
        // matches, which affects unification chains and thus tuple
        // counts at bounded iteration. Force single-threaded so our
        // output is reproducible run-to-run.
        conn.execute("SET threads = 1", [])?;
        // Sequence for fresh EqSort IDs. Term-encoded constructors
        // call `nextval` once per allocation; collisions across
        // rows with the same inputs are intentional — congruence
        // rules will unify them later.
        conn.execute("CREATE SEQUENCE __egglog_eqsort_seq START 1", [])?;
        Ok(Self {
            conn,
            functions: HashMap::new(),
            rules: Vec::new(),
            next_ts: 0,
            last_run_at: HashMap::new(),
            rules_affected: 0,
            time_mat_ns: 0,
            time_mat_act_ns: 0,
            time_act_ns: 0,
            rule_perf_ns: HashMap::new(),
            rule_to_ruleset: HashMap::new(),
            table_watermarks: HashMap::new(),
            rules_skipped: 0,
        })
    }

    /// Bump `table`'s watermark to `ts` if `ts` is newer than the
    /// current value. Cheap: called from every insert path.
    fn bump_watermark(&mut self, table: &str, ts: i64) {
        let e = self.table_watermarks.entry(table.to_string()).or_insert(0);
        if ts > *e {
            *e = ts;
        }
    }

    /// Total rule firings short-circuited by the watermark gate
    /// since this `EGraph` was created.
    pub fn rules_skipped(&self) -> u64 {
        self.rules_skipped
    }

    /// Per-category nanosecond accumulators for SQL `execute` calls.
    /// Returns `(materialize, materialized_action, simple_action)`.
    /// Read after `run_program` to see where time goes.
    pub fn perf_timings_ns(&self) -> (u64, u64, u64) {
        (
            self.time_mat_ns,
            self.time_mat_act_ns,
            self.time_act_ns,
        )
    }

    /// `(rule_name, ruleset, mat_ns, act_ns)` rows sorted by total
    /// time descending. Empty if the program hasn't run yet.
    pub fn perf_per_rule(&self) -> Vec<(String, String, u64, u64)> {
        let mut rows: Vec<(String, String, u64, u64)> = self
            .rule_perf_ns
            .iter()
            .map(|(rn, &(m, a))| {
                let rs = self
                    .rule_to_ruleset
                    .get(rn)
                    .cloned()
                    .unwrap_or_default();
                (rn.clone(), rs, m, a)
            })
            .collect();
        rows.sort_by(|x, y| (y.2 + y.3).cmp(&(x.2 + x.3)));
        rows
    }

    /// Cumulative count of rows affected by rule action SQLs across
    /// all iterations so far. Frontend snapshots this around its
    /// schedule loops to detect saturation precisely.
    pub fn rules_affected_total(&self) -> u64 {
        self.rules_affected
    }

    /// Diagnostic-only access to the underlying DuckDB connection,
    /// used by the frontend's `dump_tables`. Not for general use.
    pub fn conn_for_dump(&self) -> &Connection {
        &self.conn
    }

    fn debug_sql(&self, label: &str, sql: &str) {
        if std::env::var("DUCK_TRACE_SQL").is_ok() {
            eprintln!("[duck/{label}] {sql}");
        }
    }

    /// Allocate a fresh EqSort ID, insert a row into a constructor
    /// table with that ID, and return the ID. Used by term encoding
    /// for `(let v (C a b))` patterns at top level.  May produce
    /// rows with duplicate input keys but distinct IDs; the
    /// term-encoding-generated congruence rule will unify those
    /// IDs in UF and the rebuild rule will clean up.
    pub fn allocate_and_insert(&mut self, name: &str, inputs: &[Literal]) -> Result<i64> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if inputs.len() != info.inputs_len {
            return Err(anyhow!(
                "wrong input arity for `{name}`: got {}, expected {}",
                inputs.len(),
                info.inputs_len
            ));
        }
        if !info.eq_sort_ctor {
            return Err(anyhow!(
                "`{name}` is not registered as an EqSort constructor"
            ));
        }
        // Allocate a fresh ID. The raw constructor table is never
        // queried (term encoding routes all reads through
        // `@<name>View`) so we skip the INSERT and just take the next
        // sequence value. The caller's subsequent `(set @<name>View
        // args fresh_id) ()` writes the canonical row.
        let _ = inputs; // arity validated above; values flow through the view.
        let id: i64 = self
            .conn
            .query_row("SELECT nextval('__egglog_eqsort_seq')", [], |r| r.get(0))?;
        Ok(id)
    }

    /// Register a relation (no output column). Inserts use
    /// `ON CONFLICT DO NOTHING` — duplicates are silently dropped.
    pub fn add_relation(&mut self, name: &str, inputs: &[ColumnTy]) -> Result<()> {
        self.declare(
            name,
            FunctionInfo {
                cols: inputs.to_vec(),
                inputs_len: inputs.len(),
                merge: None,
                eq_sort_ctor: false,
            },
        )
    }

    /// Register a function with an output column and merge mode.
    /// PRIMARY KEY covers only the input columns; the output column
    /// is updated or kept on conflict according to `merge`.
    pub fn add_function(
        &mut self,
        name: &str,
        inputs: &[ColumnTy],
        output: ColumnTy,
        merge: MergeMode,
    ) -> Result<()> {
        let mut cols = inputs.to_vec();
        cols.push(output);
        self.declare(
            name,
            FunctionInfo {
                cols,
                inputs_len: inputs.len(),
                merge: Some(merge),
                eq_sort_ctor: false,
            },
        )
    }

    /// Register an EqSort constructor — a table whose last column is
    /// an EqSort ID allocated by `allocate_and_insert` from a global
    /// sequence. The PRIMARY KEY covers ALL columns (including the
    /// ID), so calling the constructor with the same inputs but a
    /// fresh ID never conflicts. Multiple rows per input key are the
    /// expected, intentional state — congruence rules emitted by
    /// term encoding unify the resulting IDs in UF later.
    pub fn add_eq_sort_constructor(
        &mut self,
        name: &str,
        inputs: &[ColumnTy],
    ) -> Result<()> {
        let mut cols = inputs.to_vec();
        cols.push(ColumnTy::I64); // the EqSort ID column
        self.declare(
            name,
            FunctionInfo {
                cols,
                inputs_len: inputs.len(),
                merge: None,
                eq_sort_ctor: true,
            },
        )?;
        Ok(())
    }

    fn declare(&mut self, name: &str, info: FunctionInfo) -> Result<()> {
        if self.functions.contains_key(name) {
            return Err(anyhow!("function {name} already registered"));
        }
        let col_decls: Vec<String> = info
            .cols
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("c{i} {} NOT NULL", ty.sql()))
            .collect();
        // PK width:
        // - relations and eq-sort constructors: cover ALL columns
        //   (eq-sort ctors expect duplicate input keys with distinct
        //   IDs; relations have no output to exclude).
        // - functions with merge mode: cover input columns only, so
        //   ON CONFLICT can update the output.
        let pk_width = match (&info.merge, info.eq_sort_ctor) {
            (Some(_), false) => info.inputs_len,
            _ => info.cols.len(),
        };
        let pk: Vec<String> = (0..pk_width).map(|i| format!("c{i}")).collect();
        let pk_clause = if pk.is_empty() {
            String::new()
        } else {
            format!(", PRIMARY KEY ({})", pk.join(", "))
        };
        // For 0-column tables (nullary relations / nullary
        // constructor helpers from term encoding), DuckDB rejects
        // an empty leading list — skip the leading comma.
        let col_list = if col_decls.is_empty() {
            "ts BIGINT NOT NULL".to_string()
        } else {
            format!("{}, ts BIGINT NOT NULL", col_decls.join(", "))
        };
        let sql = format!("CREATE TABLE {} ({col_list}{pk_clause})", q(name));
        self.debug_sql("create", &sql);
        self.conn.execute(&sql, [])?;
        // We tried two flavors of auxiliary indexes here and both
        // were a net loss on math-microbenchmark:
        //  - Secondary B-tree indexes on each non-leading input
        //    column (intended to accelerate rebuild variants that
        //    join `view.c_i = uf.c0` for i > 0). DuckDB's planner
        //    correctly chose hash joins over index seeks (the NEW
        //    side is small, the build cost is amortized), so the
        //    indexes went unused while costing per-insert
        //    maintenance. mm-microbenchmark slowed by ~26%.
        //  - Index on the `ts` column (intended to speed up the
        //    seminaive `ts >= ?1 AND ts < ?2` range filter). Lost
        //    to DuckDB's built-in zone maps for monotonically-
        //    inserted ts; insert maintenance dominated. Slowed by
        //    ~14%.
        // Insert-heavy analytical workloads on DuckDB don't want
        // OLTP-style auxiliary indexes — the columnar storage and
        // zone maps already cover the access patterns we use.
        self.functions.insert(name.to_string(), info);
        Ok(())
    }

    /// Compile and store a rule. Compilation produces one SQL
    /// statement per (variant × action).
    pub fn add_rule(&mut self, rule: Rule) -> Result<()> {
        let mut compiled = compile::compile_rule(&rule, &self.functions)?;
        // Body tables = distinct Atom::Func names. The watermark gate
        // reads this set to decide whether the rule has anything new
        // to look at on a given iteration.
        let mut bt: Vec<String> = Vec::new();
        for atom in &rule.body {
            if let Atom::Func { name, .. } = atom {
                if !bt.iter().any(|n| n == name) {
                    bt.push(name.clone());
                }
            }
        }
        compiled.body_tables = bt;
        self.last_run_at.insert(rule.name.clone(), 0);
        self.rule_to_ruleset
            .insert(rule.name.clone(), rule.ruleset.clone());
        self.rules.push(compiled);
        Ok(())
    }

    /// Seed an initial fact at `ts = 0`. With seminaive predicate
    /// `focused.ts >= last_run_at` and `last_run_at` starting at 0,
    /// the first iteration will see all seeded rows.
    pub fn insert(&mut self, name: &str, args: &[Literal]) -> Result<()> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.arity() {
            return Err(anyhow!(
                "wrong arity for {name}: got {}, expected {}",
                args.len(),
                info.arity()
            ));
        }
        // Inline literal values directly; `?N`-style binding through
        // `&[&dyn ToSql]` slices has been flaky in our context. All
        // values are i64/bool/f64/string literals with safe SQL
        // representations.
        let cols: Vec<String> = (0..args.len()).map(|i| format!("c{i}")).collect();
        let conflict = conflict_clause(info);
        let cols_prefix = prefix_with_comma(&cols);
        let arg_sqls: Vec<String> = args.iter().map(crate::compile::lit_sql_pub).collect();
        let arg_prefix = prefix_with_comma(&arg_sqls);
        let cur_ts = self.next_ts;
        let sql_unfiltered = format!(
            "INSERT INTO {} ({cols_prefix}ts) VALUES ({arg_prefix}{cur_ts}) {conflict}",
            q(name),
        );
        let sql = sql_unfiltered.trim_end();
        self.debug_sql("insert", sql);
        let n = self.conn.execute(sql, [])?;
        if n > 0 {
            self.bump_watermark(name, cur_ts);
        }
        Ok(())
    }

    /// Insert at `ts = 0` with arbitrary `Term` values (literals,
    /// primitive expressions, or subquery reads of other functions).
    /// Used for term-encoded top-level sets where the value column
    /// references another global table.
    pub fn insert_terms(&mut self, name: &str, args: &[Term]) -> Result<()> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.arity() {
            return Err(anyhow!(
                "wrong arity for {name}: got {}, expected {}",
                args.len(),
                info.arity()
            ));
        }
        let cols: Vec<String> = (0..args.len()).map(|i| format!("c{i}")).collect();
        let arg_sqls: Vec<String> = args
            .iter()
            .map(|t| compile::term_sql_no_binding(t, "<top-level>"))
            .collect::<Result<_>>()?;
        let conflict = conflict_clause(info);
        let cols_prefix = prefix_with_comma(&cols);
        let arg_prefix = prefix_with_comma(&arg_sqls);
        let cur_ts = self.next_ts;
        let sql = format!(
            "INSERT INTO {} ({cols_prefix}ts) SELECT {arg_prefix}{cur_ts} {conflict}",
            q(name),
        );
        self.debug_sql("insert_terms", &sql);
        let n = self.conn.execute(&sql, [])?;
        if n > 0 {
            self.bump_watermark(name, cur_ts);
        }
        Ok(())
    }

    /// Run rules whose ruleset is in `allowed` once. Empty set means
    /// "run all rules". Returns total rows inserted.
    pub fn run_iteration_in_set(&mut self, allowed: &[&str]) -> Result<usize> {
        let allow_all = allowed.is_empty();
        // Build the iteration with the existing single-ruleset path
        // by using a closure on the rule list. Mirrors run_iteration_in
        // but checks set membership.
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total: usize = 0;
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        for rule in &self.rules {
            if !allow_all && !allowed.iter().any(|rs| rule.ruleset == *rs) {
                continue;
            }
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            total += run_rule_variants(
                rule,
                last,
                cur,
                &self.conn,
                &mut self.time_mat_ns,
                &mut self.time_mat_act_ns,
                &mut self.time_act_ns,
                &mut self.rules_affected,
                &mut self.rule_perf_ns,
                &mut self.table_watermarks,
                &mut self.rules_skipped,
            )?;
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        Ok(total)
    }

    /// Run rules in `ruleset` once (or all rules when `ruleset` is
    /// `None`). Returns total rows inserted across rules and
    /// variants.
    pub fn run_iteration_in(&mut self, ruleset: Option<&str>) -> Result<usize> {
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total: usize = 0;
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        for rule in &self.rules {
            if let Some(rs) = ruleset
                && rule.ruleset != rs
            {
                continue;
            }
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            total += run_rule_variants(
                rule,
                last,
                cur,
                &self.conn,
                &mut self.time_mat_ns,
                &mut self.time_mat_act_ns,
                &mut self.time_act_ns,
                &mut self.rules_affected,
                &mut self.rule_perf_ns,
                &mut self.table_watermarks,
                &mut self.rules_skipped,
            )?;
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        Ok(total)
    }

    /// Run all rules (any ruleset) once. Convenience over
    /// `run_iteration_in(None)`.
    pub fn run_iteration(&mut self) -> Result<usize> {
        self.run_iteration_in(None)
    }

    /// Run rules in `ruleset` until no iteration adds any rows.
    pub fn run_to_saturation_in(&mut self, ruleset: Option<&str>) -> Result<(usize, i64)> {
        let mut iters = 0;
        loop {
            iters += 1;
            if self.run_iteration_in(ruleset)? == 0 {
                return Ok((iters, self.next_ts));
            }
        }
    }

    /// Run all rules to saturation. Convenience over
    /// `run_to_saturation_in(None)`.
    pub fn run_to_saturation(&mut self) -> Result<(usize, i64)> {
        self.run_to_saturation_in(None)
    }

    /// Whether a row matching the given args exists. For functions
    /// with outputs, you may pass either input-only args (asks "is
    /// this key present?") or full args including output (asks "is
    /// this exact row present?").
    pub fn check_exists(&self, name: &str, args: &[Literal]) -> Result<bool> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.arity() && args.len() != info.inputs_len {
            return Err(anyhow!(
                "check_exists arity for {name}: got {}, want {} or {}",
                args.len(),
                info.inputs_len,
                info.arity(),
            ));
        }
        let where_parts: Vec<String> = (0..args.len())
            .map(|i| format!("c{i} = ?{}", i + 1))
            .collect();
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!("SELECT COUNT(*) FROM {}{where_clause}", q(name));
        let params: Vec<&dyn ToSql> = args.iter().map(|a| a as &dyn ToSql).collect();
        let n: i64 = self.conn.query_row(&sql, params.as_slice(), |r| r.get(0))?;
        Ok(n > 0)
    }

    /// Look up the output value of a function for given inputs.
    /// Returns `None` for missing rows; errors if called on a relation.
    pub fn lookup_i64(&self, name: &str, inputs: &[Literal]) -> Result<Option<i64>> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if !info.has_output() {
            return Err(anyhow!("{name} is a relation; lookup_i64 not applicable"));
        }
        if inputs.len() != info.inputs_len {
            return Err(anyhow!(
                "lookup_i64 arity for {name}: got {}, expected {}",
                inputs.len(),
                info.inputs_len,
            ));
        }
        if info.cols[info.inputs_len] != ColumnTy::I64 {
            return Err(anyhow!("{name}'s output is not i64"));
        }
        let where_parts: Vec<String> = (0..inputs.len())
            .map(|i| format!("c{i} = ?{}", i + 1))
            .collect();
        let out_col = info.inputs_len;
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!(
            "SELECT c{out_col} FROM {}{where_clause}",
            q(name),
        );
        let params: Vec<&dyn ToSql> = inputs.iter().map(|a| a as &dyn ToSql).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params.as_slice())?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    /// The number of columns of a registered table, or `None` if
    /// the name isn't registered.
    pub fn function_arity(&self, name: &str) -> Option<usize> {
        self.functions.get(name).map(|f| f.cols.len())
    }

    /// Whether at least one row matches the given body atoms,
    /// interpreted as a conjunctive query. This is the same query
    /// machinery rules use, minus the seminaive focus predicate —
    /// the check passes iff the body would have any match.
    pub fn body_exists(&self, atoms: &[Atom]) -> Result<bool> {
        let sql = compile::compile_body_select(atoms, &self.functions)?;
        let n: i64 = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(n > 0)
    }

    /// Whether any row in `name` matches the given pre-built WHERE
    /// fragments. Each fragment is a `cN = literal` constraint;
    /// the caller must have already validated arity. Used for
    /// existential checks on relations.
    pub fn relation_exists_raw(&self, name: &str, where_parts: &[String]) -> Result<bool> {
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!("SELECT COUNT(*) FROM {}{}", q(name), where_clause);
        let n: i64 = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(n > 0)
    }

    pub fn count(&self, name: &str) -> Result<i64> {
        Ok(self.conn.query_row(
            &format!("SELECT COUNT(*) FROM {}", q(name)),
            [],
            |r| r.get(0),
        )?)
    }
}

/// Build the `ON CONFLICT ...` clause for an INSERT into the given
/// table. Used by both seeding (`EGraph::insert`) and rule-action
/// codegen (`compile.rs`).
pub(crate) fn conflict_clause(info: &FunctionInfo) -> String {
    // `declare` emits an empty PK clause iff the PK width is 0.
    // Without a PK, DuckDB rejects `ON CONFLICT` entirely. For
    // these tables (nullary functions with output) we just emit
    // plain INSERTs; second-write semantics are then "duplicate
    // rows accumulate", which matches `:no-merge` if the user
    // never re-sets, and is wrong otherwise. Term encoding's use
    // of nullary `:no-merge` functions is for `let`-binding
    // globals, and those are set exactly once at declaration time,
    // so this is safe in practice.
    let pk_width = match (&info.merge, info.eq_sort_ctor) {
        (Some(_), false) => info.inputs_len,
        _ => info.cols.len(),
    };
    if pk_width == 0 {
        return String::new();
    }
    if info.eq_sort_ctor {
        // EqSort constructor inserts come with a freshly-allocated
        // ID column, so collisions on the all-cols PK shouldn't
        // happen in practice. If they do (caller passed a literal
        // ID), drop quietly.
        return "ON CONFLICT DO NOTHING".to_string();
    }
    match info.merge {
        None => "ON CONFLICT DO NOTHING".to_string(),
        Some(MergeMode::Old) => "ON CONFLICT DO NOTHING".to_string(),
        Some(MergeMode::New) => {
            let out_col = info.inputs_len;
            format!(
                "ON CONFLICT DO UPDATE SET c{out_col} = EXCLUDED.c{out_col}, ts = EXCLUDED.ts"
            )
        }
    }
}
