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
pub(crate) fn q(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The (very small) set of column types we currently understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnTy {
    I64,
    Bool,
}

impl ColumnTy {
    fn sql(self) -> &'static str {
        match self {
            ColumnTy::I64 => "BIGINT",
            ColumnTy::Bool => "BOOLEAN",
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
}

impl ToSql for Literal {
    fn to_sql(&self) -> duckdb::Result<duckdb::types::ToSqlOutput<'_>> {
        match self {
            Literal::I64(i) => i.to_sql(),
            Literal::Bool(b) => b.to_sql(),
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
    /// Examples: `(< x 5)`, `(!= a b)`, `(= z (+ x y))`. The last
    /// form is treated as a filter, not a binding — we don't yet
    /// support computed-variable bindings in the body.
    Filter(Term),
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
}

/// A whole rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub body: Vec<Atom>,
    pub actions: Vec<Action>,
}

/// Internal: schema for a single registered function.
#[derive(Debug, Clone)]
pub(crate) struct FunctionInfo {
    /// All schema columns (inputs followed by output/ID, if any).
    pub(crate) cols: Vec<ColumnTy>,
    /// Number of "input" columns from the user perspective. For
    /// relations this is `cols.len()`; for functions and EqSort
    /// constructors it's `cols.len() - 1`.
    pub(crate) inputs_len: usize,
    /// `Some` if this is a function with an output column; `None`
    /// for relations and EqSort constructors.
    pub(crate) merge: Option<MergeMode>,
    /// True iff this is an EqSort constructor: PK covers all cols
    /// (so multiple distinct IDs per input set are allowed) and
    /// `allocate_and_insert` is the intended insertion path.
    pub(crate) eq_sort_ctor: bool,
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
    pub(crate) functions: HashMap<String, FunctionInfo>,
    rules: Vec<CompiledRule>,
    next_ts: i64,
    /// Per source-rule "last run at" — the ts at which it last ran.
    last_run_at: HashMap<String, i64>,
}

struct CompiledRule {
    name: String,
    /// Each variant is one or more SQL statements (one per action).
    /// Inner Vec lets multi-action rules execute their actions in
    /// order from a single firing of the variant.
    variants: Vec<Vec<String>>,
}

impl EGraph {
    pub fn new() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
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
        })
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
        let placeholders: Vec<String> = (1..=inputs.len()).map(|i| format!("?{i}")).collect();
        let in_cols: Vec<String> = (0..info.inputs_len).map(|i| format!("c{i}")).collect();
        let out_col = info.inputs_len;
        let sql = format!(
            "INSERT INTO {} ({}, c{out_col}, ts) VALUES ({}, nextval('__egglog_eqsort_seq'), 0) RETURNING c{out_col}",
            q(name),
            in_cols.join(", "),
            placeholders.join(", "),
        );
        let params: Vec<&dyn ToSql> = inputs.iter().map(|a| a as &dyn ToSql).collect();
        let id: i64 = self.conn.query_row(&sql, params.as_slice(), |r| r.get(0))?;
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
        let sql = format!(
            "CREATE TABLE {} ({}, ts BIGINT NOT NULL{pk_clause})",
            q(name),
            col_decls.join(", "),
        );
        self.conn.execute(&sql, [])?;
        self.functions.insert(name.to_string(), info);
        Ok(())
    }

    /// Compile and store a rule. Compilation produces one SQL
    /// statement per (variant × action).
    pub fn add_rule(&mut self, rule: Rule) -> Result<()> {
        let compiled = compile::compile_rule(&rule, &self.functions)?;
        self.last_run_at.insert(rule.name.clone(), 0);
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
        let placeholders: Vec<String> = (1..=args.len()).map(|i| format!("?{i}")).collect();
        let cols: Vec<String> = (0..args.len()).map(|i| format!("c{i}")).collect();
        let conflict = conflict_clause(info);
        let sql = format!(
            "INSERT INTO {} ({}, ts) VALUES ({}, 0) {conflict}",
            q(name),
            cols.join(", "),
            placeholders.join(", "),
        );
        let params: Vec<&dyn ToSql> = args.iter().map(|a| a as &dyn ToSql).collect();
        self.conn.execute(&sql, params.as_slice())?;
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
        let sql = format!(
            "INSERT INTO {} ({}, ts) SELECT {}, 0 {conflict}",
            q(name),
            cols.join(", "),
            arg_sqls.join(", "),
        );
        self.conn.execute(&sql, [])?;
        Ok(())
    }

    /// Run all rules once. Returns total rows inserted across rules
    /// and variants.
    pub fn run_iteration(&mut self) -> Result<usize> {
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total: usize = 0;
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        for rule in &self.rules {
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            for variant in &rule.variants {
                for sql in variant {
                    let mut stmt = self.conn.prepare_cached(sql)?;
                    total += stmt.execute(duckdb::params![last, cur])?;
                }
            }
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        Ok(total)
    }

    /// Run iterations until no rule adds any rows.
    pub fn run_to_saturation(&mut self) -> Result<(usize, i64)> {
        let mut iters = 0;
        loop {
            iters += 1;
            if self.run_iteration()? == 0 {
                return Ok((iters, self.next_ts));
            }
        }
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
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE {}",
            q(name),
            where_parts.join(" AND "),
        );
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
        let sql = format!(
            "SELECT c{out_col} FROM {} WHERE {}",
            q(name),
            where_parts.join(" AND "),
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
