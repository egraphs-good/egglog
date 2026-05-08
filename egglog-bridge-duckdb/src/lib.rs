//! DuckDB-backed executor for a small subset of egglog's resolved IR.
//!
//! Phase 1.0 scope: enough to run pure Datalog (no term encoding, no
//! UF, no merges, no primitives beyond what naturally falls out of
//! variable equality). See `../duckdb-backend-plan.md` for the full
//! plan and where this fits.
//!
//! Design notes:
//! - One DuckDB table per registered function.
//! - Every table carries a `ts BIGINT NOT NULL` column for seminaive.
//! - `next_ts` and `last_run_at_<rule>` live in Rust state; they are
//!   bind parameters in the generated SQL, never database rows.
//! - Each rule with N function-table atoms compiles to N seminaive
//!   variants (one per focused atom), emitted as separate prepared
//!   `INSERT INTO target SELECT ... WHERE focused.ts >= :last`
//!   statements.

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

/// A term in a rule body or action: either a free variable (bound by
/// matching) or a literal. We currently don't carry types on terms;
/// the schema does the type checking implicitly.
#[derive(Debug, Clone)]
pub enum Term {
    Var(String),
    Lit(Literal),
}

impl Term {
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }
    pub fn i64(v: i64) -> Self {
        Term::Lit(Literal::I64(v))
    }
}

/// A body atom of a rule.
#[derive(Debug, Clone)]
pub enum Atom {
    /// A function-table atom: `(f arg0 arg1 ...)`.  All non-output
    /// columns are matched/bound; functions with logical "outputs"
    /// would be handled by a future variant.
    Func { name: String, args: Vec<Term> },
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

#[derive(Debug, Clone)]
struct FunctionInfo {
    schema: Vec<ColumnTy>,
}

/// The executor.
pub struct EGraph {
    conn: Connection,
    functions: HashMap<String, FunctionInfo>,
    rules: Vec<CompiledRule>,
    next_ts: i64,
    /// Per source-rule "last run at" — the ts at which it last ran,
    /// used to bound the focused atom in seminaive variants.
    last_run_at: HashMap<String, i64>,
}

struct CompiledRule {
    name: String,
    /// Each variant is one prepared SQL statement.
    variants: Vec<String>,
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

    /// Register a function (egglog `relation`/`function`) with the
    /// given column types. The DuckDB table is created with all
    /// schema columns plus an extra `ts BIGINT NOT NULL`. PRIMARY
    /// KEY covers all schema columns (relation semantics; functions
    /// would key on a prefix).
    pub fn add_function(&mut self, name: &str, schema: &[ColumnTy]) -> Result<()> {
        if self.functions.contains_key(name) {
            return Err(anyhow!("function {name} already registered"));
        }
        let cols: Vec<String> = schema
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("c{i} {} NOT NULL", ty.sql()))
            .collect();
        let pk: Vec<String> = (0..schema.len()).map(|i| format!("c{i}")).collect();
        let sql = format!(
            "CREATE TABLE {name} ({}, ts BIGINT NOT NULL, PRIMARY KEY ({}))",
            cols.join(", "),
            pk.join(", "),
        );
        self.conn.execute(&sql, [])?;
        self.functions.insert(
            name.to_string(),
            FunctionInfo {
                schema: schema.to_vec(),
            },
        );
        Ok(())
    }

    /// Compile and store a rule. Compilation produces one SQL
    /// statement per seminaive variant.
    pub fn add_rule(&mut self, rule: Rule) -> Result<()> {
        let compiled = compile::compile_rule(&rule, &self.functions)?;
        self.last_run_at.insert(rule.name.clone(), 0);
        self.rules.push(compiled);
        Ok(())
    }

    /// Seed an initial fact. Picks `ts = 0` (everything seeded is
    /// "from before any iteration"). With the seminaive predicate
    /// `focused.ts >= last_run_at` and `last_run_at` starting at 0,
    /// the first iteration will see all seeded rows.
    pub fn insert(&mut self, name: &str, args: &[Literal]) -> Result<()> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.schema.len() {
            return Err(anyhow!(
                "wrong arity for {name}: got {}, expected {}",
                args.len(),
                info.schema.len()
            ));
        }
        let placeholders: Vec<String> = (1..=args.len()).map(|i| format!("?{i}")).collect();
        let cols: Vec<String> = (0..args.len()).map(|i| format!("c{i}")).collect();
        let sql = format!(
            "INSERT INTO {name} ({}, ts) VALUES ({}, 0) ON CONFLICT DO NOTHING",
            cols.join(", "),
            placeholders.join(", "),
        );
        // Bind args.
        let params: Vec<&dyn ToSql> = args.iter().map(|a| a as &dyn ToSql).collect();
        self.conn.execute(&sql, params.as_slice())?;
        Ok(())
    }

    /// Run all rules once, then advance bookkeeping. Returns the
    /// total number of rows inserted across all rules / variants.
    pub fn run_iteration(&mut self) -> Result<usize> {
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total_inserted: usize = 0;
        // Collect (name, last_run_at) so we don't borrow self
        // mutably across the rule loop.
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        for rule in &self.rules {
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            for sql in &rule.variants {
                // `prepare_cached` reuses an LRU-cached plan keyed
                // by SQL string, so we don't re-plan per iteration.
                let mut stmt = self.conn.prepare_cached(sql)?;
                let n = stmt.execute(duckdb::params![last, cur])?;
                total_inserted += n;
            }
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        Ok(total_inserted)
    }

    /// Run iterations until none of them add any rows. Returns
    /// `(iterations, final_ts)`.
    pub fn run_to_saturation(&mut self) -> Result<(usize, i64)> {
        let mut iters = 0;
        loop {
            iters += 1;
            let added = self.run_iteration()?;
            if added == 0 {
                return Ok((iters, self.next_ts));
            }
        }
    }

    /// Return whether a row exists in the named function.  The check
    /// matches the row by the supplied arg values; ts is ignored.
    pub fn check_exists(&self, name: &str, args: &[Literal]) -> Result<bool> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.schema.len() {
            return Err(anyhow!("wrong arity for {name}"));
        }
        let where_parts: Vec<String> = (0..args.len()).map(|i| format!("c{i} = ?{}", i + 1)).collect();
        let sql = format!(
            "SELECT COUNT(*) FROM {name} WHERE {}",
            where_parts.join(" AND "),
        );
        let params: Vec<&dyn ToSql> = args.iter().map(|a| a as &dyn ToSql).collect();
        let n: i64 = self
            .conn
            .query_row(&sql, params.as_slice(), |r| r.get(0))?;
        Ok(n > 0)
    }

    /// Return the total number of rows in the named function.
    pub fn count(&self, name: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {name}");
        Ok(self.conn.query_row(&sql, [], |r| r.get(0))?)
    }
}
