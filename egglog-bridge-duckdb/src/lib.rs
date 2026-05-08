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

/// A rule action: insert a row into a function/relation table.
#[derive(Debug, Clone)]
pub enum Action {
    Insert { name: String, args: Vec<Term> },
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
    /// All schema columns (inputs followed by output, if any).
    pub(crate) cols: Vec<ColumnTy>,
    /// Number of input columns. Output column, if present, is at
    /// index `inputs_len`.
    pub(crate) inputs_len: usize,
    /// `Some` if this is a function with an output column; `None`
    /// for relations.
    pub(crate) merge: Option<MergeMode>,
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
        Ok(Self {
            conn: Connection::open_in_memory()?,
            functions: HashMap::new(),
            rules: Vec::new(),
            next_ts: 0,
            last_run_at: HashMap::new(),
        })
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
            },
        )
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
        let pk: Vec<String> = (0..info.inputs_len).map(|i| format!("c{i}")).collect();
        let pk_clause = if pk.is_empty() {
            // 0-input nullary function: PRIMARY KEY () isn't a thing
            // in DuckDB. Skip the clause; we'll enforce via UNIQUE
            // not-supported workaround later. For now, accept duplicates
            // for nullary tables (which we don't yet generate).
            String::new()
        } else {
            format!(", PRIMARY KEY ({})", pk.join(", "))
        };
        let sql = format!(
            "CREATE TABLE {name} ({}, ts BIGINT NOT NULL{pk_clause})",
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
            "INSERT INTO {name} ({}, ts) VALUES ({}, 0) {conflict}",
            cols.join(", "),
            placeholders.join(", "),
        );
        let params: Vec<&dyn ToSql> = args.iter().map(|a| a as &dyn ToSql).collect();
        self.conn.execute(&sql, params.as_slice())?;
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
            "SELECT COUNT(*) FROM {name} WHERE {}",
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
            "SELECT c{out_col} FROM {name} WHERE {}",
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
        Ok(self
            .conn
            .query_row(&format!("SELECT COUNT(*) FROM {name}"), [], |r| r.get(0))?)
    }
}

/// Build the `ON CONFLICT ...` clause for an INSERT into the given
/// table. Used by both seeding (`EGraph::insert`) and rule-action
/// codegen (`compile.rs`).
pub(crate) fn conflict_clause(info: &FunctionInfo) -> String {
    match info.merge {
        // Relation: presence-only, conflicts are no-ops.
        None => "ON CONFLICT DO NOTHING".to_string(),
        Some(MergeMode::Old) => "ON CONFLICT DO NOTHING".to_string(),
        Some(MergeMode::New) => {
            // Update the output and ts on conflict.
            let out_col = info.inputs_len;
            format!(
                "ON CONFLICT DO UPDATE SET c{out_col} = EXCLUDED.c{out_col}, ts = EXCLUDED.ts"
            )
        }
    }
}
