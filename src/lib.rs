pub mod ast;
mod extract;
mod gj;
mod typecheck;
mod unionfind;
mod util;
mod value;

use hashbrown::hash_map::Entry;
use thiserror::Error;

use ast::*;
use std::fmt::Debug;
use std::hash::Hash;

pub use value::*;

use gj::*;
use num_rational::BigRational;
use unionfind::*;
use util::*;

use crate::typecheck::TypeError;

#[derive(Clone)]
pub struct Function {
    schema: Schema,
    nodes: HashMap<Vec<Value>, Value>,
    updates: usize,
}

impl Function {
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            nodes: Default::default(),
            updates: 0,
        }
    }

    pub fn get(&mut self, uf: &mut UnionFind, args: &[Value]) -> Option<Value> {
        // TODO typecheck?
        if self.schema.output.is_sort() {
            let id = self
                .nodes
                .entry(args.into())
                .or_insert_with(|| uf.make_set().into());
            Some(id.clone())
        } else {
            self.nodes.get(args).cloned()
        }
    }

    pub fn set(&mut self, uf: &mut UnionFind, args: &[Value], value: Value) -> Value {
        // TODO typecheck?
        match self.nodes.entry(args.into()) {
            Entry::Occupied(mut e) => {
                let old = e.get().clone();
                if old == value {
                    value
                } else {
                    self.updates += 1;
                    let new_value = match self.schema.output {
                        OutputType::Type(InputType::Sort(_)) => uf.union_values(old, value),
                        OutputType::Type(_) => {
                            assert_ne!(old, value);
                            panic!("bad merge")
                        }
                        OutputType::Max(NumType::I64) => i64::max(old.into(), value.into()).into(),
                        // OutputType::Max(NumType::F64) => f64::max(old.into(), value.into()).into(),
                        OutputType::Min(NumType::I64) => i64::min(old.into(), value.into()).into(),
                        // we use 0 for unit
                        OutputType::Unit => 0.into(),
                        // we use 0 for unit
                        // OutputType::Min(NumType::F64) => f64::min(old.into(), value.into()).into(),
                        OutputType::Max(NumType::Rational) => {
                            std::cmp::max(BigRational::from(old), BigRational::from(value)).into()
                        }
                        OutputType::Min(NumType::Rational) => {
                            std::cmp::min(BigRational::from(old), BigRational::from(value)).into()
                        }
                    };
                    // if self.schema.output.is_sort() {
                    //     uf.union_values(old, value)
                    // } else if let Some(merge) = &self.merge_fn {
                    //     match merge.expr {
                    //         Expr::Node(s, _) if s.as_str() == "max" => {
                    //             i64::max(old.into(), value.into()).into()
                    //         }
                    //         Expr::Node(s, _) if s.as_str() == "min" => {
                    //             i64::min(old.into(), value.into()).into()
                    //         }
                    //         _ => panic!("Unsupported merge for now"),
                    //     }
                    // } else {
                    //     panic!("no merge!")
                    // };

                    e.insert(new_value.clone());
                    new_value
                }
            }
            Entry::Vacant(e) => {
                self.updates += 1;
                e.insert(value).clone()
            }
        }
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind) -> usize {
        // FIXME this doesn't compute updates properly
        let n_unions = uf.n_unions();
        let old_nodes = std::mem::take(&mut self.nodes);
        for (mut args, value) in old_nodes {
            for (a, ty) in args.iter_mut().zip(&self.schema.input) {
                if ty.is_sort() {
                    *a = uf.find_mut_value(a.clone())
                }
            }
            let _new_value = if self.schema.output.is_sort() {
                self.nodes
                    .entry(args)
                    .and_modify(|value2| *value2 = uf.union_values(value.clone(), value2.clone()))
                    .or_insert_with(|| uf.find_mut_value(value))
            } else {
                self.nodes
                    .entry(args)
                    // .and_modify(|value2| *value2 = uf.union_values(value.clone(), value2.clone()))
                    .or_insert(value)
            };
        }
        uf.n_unions() - n_unions + std::mem::take(&mut self.updates)
    }
}

pub type Subst = IndexMap<Symbol, Value>;

#[derive(Clone)]
#[allow(dead_code)]
pub struct Primitive {
    input: Vec<NumType>,
    output: NumType,
    f: fn(&[Value]) -> Value,
}

impl Primitive {
    pub fn accept(&self, values: &[Value]) -> bool {
        self.input
            .iter()
            .zip(values.iter())
            .all(|(t, v)| matches!(v.get_type(), InputType::NumType(t1) if &t1 == t))
    }

    pub fn apply(&self, values: &[Value]) -> Value {
        (self.f)(values)
    }
}

fn default_primitives() -> HashMap<Symbol, Vec<Primitive>> {
    macro_rules! prim {
        (@type I64) => { i64 };
        (@type Rational) => { BigRational };
        (|$($param:ident : $t:ident),*| -> $output:ident { $body:expr }) => {
            Primitive {
                input: vec![$(NumType::$t),*],
                output: NumType::$output,
                f: |values: &[Value]| -> Value {
                    let mut values = values.iter();
                    $(
                        let $param: prim!(@type $t) = values.next().unwrap().clone().into();
                    )*
                    Value::from($body)
                }
            }
        };
    }

    [
        (
            "+",
            vec![
                prim!(|a: I64, b: I64| -> I64 { a + b }),
                prim!(|a: Rational, b: Rational| -> Rational { a + b }),
            ],
        ),
        (
            "-",
            vec![
                prim!(|a: I64, b: I64| -> I64 { a - b }),
                prim!(|a: Rational, b: Rational| -> Rational { a - b }),
            ],
        ),
        (
            "*",
            vec![
                prim!(|a: I64, b: I64| -> I64 { a * b }),
                prim!(|a: Rational, b: Rational| -> Rational { a * b }),
            ],
        ),
        (
            "max",
            vec![
                prim!(|a: I64, b: I64| -> I64 { a.max(b) }),
                prim!(|a: Rational, b: Rational| -> Rational { a.max(b) }),
            ],
        ),
        (
            "min",
            vec![
                prim!(|a: I64, b: I64| -> I64 { a.min(b) }),
                prim!(|a: Rational, b: Rational| -> Rational { a.min(b) }),
            ],
        ),
    ]
    .into_iter()
    .map(|(k, v)| (Symbol::from(k), v))
    .collect()
}

#[derive(Clone)]
pub struct EGraph {
    unionfind: UnionFind,
    sorts: HashMap<Symbol, Vec<Symbol>>,
    primitives: HashMap<Symbol, Vec<Primitive>>,
    functions: HashMap<Symbol, Function>,
    rules: HashMap<Symbol, Rule>,
    globals: HashMap<Symbol, Value>,
}

#[derive(Clone, Debug)]
struct Rule {
    query: Query,
    head: Vec<Fact>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self {
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            globals: Default::default(),
            primitives: default_primitives(),
        }
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(Expr);

impl EGraph {
    pub fn union(&mut self, id1: Id, id2: Id) -> Id {
        self.unionfind.union(id1, id2)
    }

    #[track_caller]
    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        for (name, function) in self.functions.iter() {
            for (inputs, output) in function.nodes.iter() {
                for input in inputs {
                    assert_eq!(
                        input,
                        &self.bad_find_value(input.clone()),
                        "{name}({inputs:?}) = {output}\n{:?}",
                        function.schema,
                    )
                }
                assert_eq!(
                    output,
                    &self.bad_find_value(output.clone()),
                    "{name}({inputs:?}) = {output}\n{:?}",
                    function.schema,
                )
            }
        }
    }

    pub fn union_exprs(&mut self, ctx: &Subst, exprs: &[Expr]) -> Result<Value, NotFoundError> {
        let mut exprs = exprs.iter();
        let e = exprs.next().expect("shouldn't be empty");
        let mut val = self.eval_expr(ctx, e)?;
        for e2 in exprs {
            let val2 = self.eval_expr(ctx, e2)?;
            val = self.unionfind.union_values(val, val2);
        }
        Ok(val)
    }

    pub fn assert_with(&mut self, ctx: &Subst, fact: &Fact) -> Result<(), Error> {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() > 1);
                let mut should_union = true;
                if let Expr::Call(sym, args) = &exprs[0] {
                    if !self.functions[sym].schema.output.is_sort() {
                        assert_eq!(exprs.len(), 2);
                        let arg_values: Vec<Value> = args
                            .iter()
                            .map(|e| self.eval_expr(ctx, e))
                            .collect::<Result<_, _>>()?;
                        let value = self.eval_expr(ctx, &exprs[1])?;
                        let f = self
                            .functions
                            .get_mut(sym)
                            .expect("FIXME add error message");
                        f.set(&mut self.unionfind, &arg_values, value);
                        should_union = false;
                    }
                }

                if should_union {
                    self.union_exprs(ctx, exprs)?;
                }
            }
            Fact::Fact(expr) => match expr {
                Expr::Lit(_) => panic!("can't assert a literal"),
                Expr::Var(_) => panic!("can't assert a var"),
                Expr::Call(sym, args) => {
                    let values: Vec<Value> = args
                        .iter()
                        .map(|e| self.eval_expr(ctx, e))
                        .collect::<Result<_, _>>()?;
                    let f = self
                        .functions
                        .get_mut(sym)
                        .expect("FIXME add error message");
                    // FIXME We don't have a unit value
                    f.set(&mut self.unionfind, &values, true.into());
                    assert_eq!(f.schema.output, OutputType::Unit);
                }
            },
        }
        Ok(())
    }

    pub fn check_with(&mut self, ctx: &Subst, fact: &Fact) -> Result<(), Error> {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() > 1);
                let values: Vec<Value> = exprs
                    .iter()
                    .map(|e| self.eval_expr(ctx, e).map(|v| self.bad_find_value(v)))
                    .collect::<Result<_, _>>()?;
                for v in &values[1..] {
                    if &values[0] != v {
                        return Err(Error::CheckError(values[0].clone(), v.clone()));
                    }
                }
                // let mut should_union = true;
                // if let Expr::Node(sym, args) = &exprs[0] {
                //     if !self.functions[sym].schema.output.is_sort() {
                //         assert_eq!(exprs.len(), 2);
                //         let arg_values: Vec<Value> = args
                //             .iter()
                //             .map(|e| self.eval_expr(ctx, e))
                //             .collect::<Result<_, _>>()?;
                //         let value = self.eval_expr(ctx, &exprs[1])?;
                //         let f = self
                //             .functions
                //             .get_mut(sym)
                //             .expect("FIXME add error message");
                //         assert_eq!(f.get(&mut self.unionfind, &arg_values).unwrap(), value);
                //         should_union = false;
                //     }
                // }

                // if should_union {
                //     self.union_exprs(ctx, exprs)?;
                // }
            }
            Fact::Fact(expr) => match expr {
                Expr::Lit(_) => panic!("can't assert a literal"),
                Expr::Var(_) => panic!("can't assert a var"),
                Expr::Call(sym, args) => {
                    let values: Vec<Value> = args
                        .iter()
                        .map(|e| self.eval_expr(ctx, e))
                        .collect::<Result<_, _>>()?;
                    let f = self
                        .functions
                        .get_mut(sym)
                        .expect("FIXME add error message");
                    // FIXME We don't have a unit value
                    f.get(&mut self.unionfind, &values)
                        .ok_or_else(|| NotFoundError(expr.clone()))?;
                    assert_eq!(f.schema.output, OutputType::Unit);
                }
            },
        }
        Ok(())
    }

    pub fn assert(&mut self, fact: &Fact) -> Result<(), Error> {
        self.assert_with(&Default::default(), fact)
    }

    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    pub fn rebuild(&mut self) -> usize {
        let mut updates = 0;
        loop {
            let new = self.rebuild_one();
            log::debug!("{new} rebuilds?");
            updates += new;
            if new == 0 {
                break;
            }
        }
        self.debug_assert_invariants();
        updates
    }

    fn rebuild_one(&mut self) -> usize {
        let mut new_unions = 0;
        for function in self.functions.values_mut() {
            new_unions += function.rebuild(&mut self.unionfind);
        }
        new_unions
    }

    pub fn declare_sort(&mut self, name: impl Into<Symbol>) -> Result<(), Error> {
        let name = name.into();
        match self.sorts.entry(name) {
            Entry::Occupied(_) => Err(Error::SortAlreadyBound(name)),
            Entry::Vacant(e) => {
                e.insert(vec![]);
                Ok(())
            }
        }
    }

    pub fn declare_function(&mut self, decl: &FunctionDecl) -> Result<(), Error> {
        for ty in &decl.schema.input {
            if let InputType::Sort(sort) = ty {
                if !self.sorts.contains_key(sort) {
                    return Err(TypeError::UndefinedSort(*sort).into());
                }
            }
        }

        if let OutputType::Type(InputType::Sort(sort)) = &decl.schema.output {
            if !self.sorts.contains_key(sort) {
                return Err(TypeError::UndefinedSort(*sort).into());
            }
        }

        let old = self
            .functions
            .insert(decl.name, Function::new(decl.schema.clone()));
        if old.is_some() {
            return Err(TypeError::FunctionAlreadyBound(decl.name).into());
        }

        Ok(())
    }

    pub fn declare_constructor(
        &mut self,
        name: impl Into<Symbol>,
        types: Vec<InputType>,
        sort: impl Into<Symbol>,
    ) -> Result<(), Error> {
        let name = name.into();
        let sort = sort.into();
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: types,
                output: OutputType::Type(InputType::Sort(sort)),
            },
        })?;
        if let Some(ctors) = self.sorts.get_mut(&sort) {
            ctors.push(name);
        }
        Ok(())
    }

    // this must be &mut because it'll call "make_set",
    // but it'd be nice if that didn't have to happen
    pub fn eval_expr(&mut self, ctx: &Subst, expr: &Expr) -> Result<Value, NotFoundError> {
        match expr {
            // TODO should we canonicalize here?
            Expr::Var(var) => Ok(ctx
                .get(var)
                .or_else(|| self.globals.get(var))
                .cloned()
                .unwrap_or_else(|| panic!("Couldn't find variable '{var}'"))),
            Expr::Lit(lit) => Ok(lit.to_value()),
            Expr::Call(op, args) => {
                let values: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(ctx, a))
                    .collect::<Result<_, _>>()?;
                if let Some(rel) = self.functions.get_mut(op) {
                    rel.get(&mut self.unionfind, &values)
                        .ok_or_else(|| NotFoundError(expr.clone()))
                } else if let Some(prims) = self.primitives.get(op) {
                    let mut res = None;
                    for prim in prims.iter() {
                        if prim.accept(&values) {
                            if res.is_none() {
                                res = Some(prim.apply(&values));
                            } else {
                                panic!("multiple implementation matches primitives {op}");
                            }
                        }
                    }
                    res.ok_or_else(|| NotFoundError(expr.clone()))
                } else {
                    panic!("Couldn't find function/primitive: {op}")
                }
            }
        }
    }

    pub fn eval_closed_expr(&mut self, expr: &Expr) -> Result<Value, NotFoundError> {
        self.eval_expr(&Default::default(), expr)
    }

    pub fn set_expr(
        &mut self,
        ctx: &Subst,
        expr: &Expr,
        values: &[Value],
    ) -> Result<Value, NotFoundError> {
        assert!(!values.is_empty());
        match expr {
            Expr::Var(var) => Ok(ctx[var].clone()),
            Expr::Lit(lit) => Ok(lit.to_value()),
            Expr::Call(op, args) => {
                let args: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(ctx, a))
                    .collect::<Result<_, _>>()?;
                let func = self.functions.get_mut(op).unwrap();
                Ok(values
                    .iter()
                    .map(|v| func.set(&mut self.unionfind, &args, v.clone()))
                    .last()
                    .unwrap())
            }
        }
    }

    fn query(&self, query: &Query, callback: impl FnMut(&[Value])) {
        let compiled_query = self.compile_gj_query(&query.atoms);
        self.run_query(&compiled_query, callback)
    }

    // fn apply(&mut self, actions: &[Action], subst: &Subst) -> Result<(), NotFoundError> {
    //     let mut subst = subst.clone(); // This is slow
    //     for action in actions {
    //         match action {
    //             Action::Define(v, e) => {
    //                 let value = self.eval_expr(&subst, e)?;
    //                 subst
    //                     .entry(*v)
    //                     .and_modify(|old| {
    //                         *old = self.unionfind.union_values(value.clone(), old.clone())
    //                     })
    //                     .or_insert(value);
    //             }
    //             Action::Union(exprs) => {
    //                 self.union_exprs(&subst, exprs)?;
    //             }
    //             Action::Assert(exprs) => {
    //                 self.assert_exprs(&subst, exprs)?;
    //             }
    //             Action::Set(dst, srcs) => {
    //                 self.set_exprs(&subst, dst, srcs)?;
    //             }
    //         }
    //     }
    //     Ok(())
    // }

    pub fn run_rules(&mut self, limit: usize) {
        for _ in 0..limit {
            self.step_rules();
            let updates = self.rebuild();
            log::debug!("Made {updates} updates",);
            // if updates == 0 {
            //     log::debug!("Breaking early!");
            //     break;
            // }
        }

        // TODO detect functions
        for (name, r) in &self.functions {
            log::debug!("{name}:");
            for (args, val) in &r.nodes {
                log::debug!("  {args:?} = {val}");
            }
        }
    }

    fn step_rules(&mut self) {
        let searched: Vec<_> = self
            .rules
            .values()
            .map(|rule| {
                let mut substs = Vec::<Subst>::new();
                self.query(&rule.query, |values| {
                    let get = |a: &AtomTerm| -> Value {
                        match a {
                            AtomTerm::Var(i) => values[*i].clone(),
                            AtomTerm::Value(val) => val.clone(),
                        }
                    };
                    substs.push(
                        rule.query
                            .bindings
                            .iter()
                            .map(|(sym, a)| (*sym, get(a)))
                            .collect(),
                    )
                });
                substs
            })
            .collect();

        let rules = std::mem::take(&mut self.rules);
        for (rule, substs) in rules.values().zip(searched) {
            for subst in substs {
                // we ignore the result here because rule applications are best effort
                for fact in &rule.head {
                    let _result: Result<_, _> = self.assert_with(&subst, fact);
                }
            }
        }
        self.rules = rules;
    }

    fn add_rule_with_name(&mut self, name: String, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let compiled_rule = Rule {
            query: self.compile_query(rule.body)?,
            head: rule.head,
        };
        match self.rules.entry(name) {
            Entry::Occupied(_) => panic!("Rule '{name}' was already present"),
            Entry::Vacant(e) => e.insert(compiled_rule),
        };
        Ok(name)
    }

    pub fn add_rule(&mut self, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = format!("{}", rule);
        self.add_rule_with_name(name, rule)
    }

    pub fn clear_rules(&mut self) {
        self.rules = Default::default();
    }

    pub fn add_rewrite(&mut self, rewrite: ast::Rewrite) -> Result<Symbol, Error> {
        let name = format!("{} -> {}", rewrite.lhs, rewrite.rhs);
        let var = Symbol::from("__rewrite_var");
        let rule = ast::Rule {
            body: vec![Fact::Eq(vec![Expr::Var(var), rewrite.lhs])],
            head: vec![Fact::Eq(vec![Expr::Var(var), rewrite.rhs])],
        };
        self.add_rule_with_name(name, rule)
    }

    fn for_each_canonicalized(&self, name: Symbol, mut cb: impl FnMut(&[Value])) {
        let mut ids = vec![];
        let f = self
            .functions
            .get(&name)
            .unwrap_or_else(|| panic!("No function {name}"));
        for (children, value) in &f.nodes {
            ids.clear();
            // FIXME canonicalize, do we need to with rebuilding?
            // ids.extend(children.iter().map(|id| self.find(value)));
            ids.extend(children.iter().cloned());
            ids.push(value.clone());
            cb(&ids);
        }
    }

    fn run_command(&mut self, command: Command, should_run: bool) -> Result<String, Error> {
        Ok(match command {
            Command::Datatype { name, variants } => {
                self.declare_sort(name)?;
                for variant in variants {
                    self.declare_constructor(variant.name, variant.types, name)?;
                }
                format!("Declared datatype {name}.")
            }
            Command::Function(fdecl) => {
                self.declare_function(&fdecl)?;
                format!("Declared function {}.", fdecl.name)
            }
            Command::Rule(rule) => {
                let name = self.add_rule(rule)?;
                format!("Declared rule {name}.")
            }
            Command::Rewrite(rewrite) => {
                let name = self.add_rewrite(rewrite)?;
                format!("Declared rw {name}.")
            }
            Command::Run(limit) => {
                if should_run {
                    self.run_rules(limit);
                    format!("Ran {limit}.")
                } else {
                    log::info!("Skipping running!");
                    format!("Skipped run {limit}.")
                }
            }
            Command::Extract(e) => {
                if should_run {
                    // TODO typecheck
                    let value = self.eval_closed_expr(&e)?;
                    self.rebuild();
                    let id = Id::from(value);
                    log::info!("Extracting {e} at {id}");
                    let (cost, expr) = self.extract(id);
                    format!("Extracted with cost {cost}: {expr}")
                } else {
                    "Skipping extraction.".into()
                }
            }
            Command::Check(fact) => {
                if should_run {
                    self.check_with(&Default::default(), &fact)?;
                    "Checked.".into()
                } else {
                    "Skipping check.".into()
                }
            }
            Command::Fact(fact) => {
                if should_run {
                    self.assert(&fact)?;
                    format!("Asserted {fact}.")
                } else {
                    format!("Skipping assert {fact}.")
                }
            }
            Command::Define(name, expr) => {
                if should_run {
                    let value = self.eval_closed_expr(&expr)?;
                    let old = self.globals.insert(name, value);
                    assert!(old.is_none());
                    format!("Defined {name}")
                } else {
                    format!("Skipping define {name}")
                }
            }
            Command::ClearRules => {
                self.clear_rules();
                "Clearing rules.".into()
            }
            Command::Query(q) => {
                let qsexp = sexp::Sexp::List(
                    q.iter()
                        .map(|fact| sexp::parse(&fact.to_string()).unwrap())
                        .collect(),
                );
                let qcomp = self
                    .compile_query(q)
                    .unwrap_or_else(|_| panic!("Could not compile query"));
                let mut res = vec![];
                self.query(&qcomp, |v| {
                    res.push(sexp::Sexp::List(
                        v.iter()
                            .map(|val| sexp::Sexp::Atom(sexp::Atom::S(format!("{}", val))))
                            .collect(),
                    ));
                });
                format!(
                    "Query: {}\n  Bindings: {:?}\n  Results: {}",
                    qsexp,
                    qcomp,
                    sexp::Sexp::List(res)
                )
            }
        })
    }

    fn run_program(&mut self, program: Vec<Command>) -> Result<Vec<String>, Error> {
        let mut msgs = vec![];
        let should_run = true;

        for command in program {
            let msg = self.run_command(command, should_run)?;
            log::info!("{}", msg);
            msgs.push(msg);
        }

        Ok(msgs)
    }

    // this is bad because we shouldn't inspect values like this, we should use type information
    fn bad_find_value(&self, value: Value) -> Value {
        match &value.0 {
            ValueInner::Id(id) => self.unionfind.find(*id).into(),
            _ => value,
        }
    }

    pub fn parse_and_run_program(&mut self, input: &str) -> Result<Vec<String>, Error> {
        let parser = ast::parse::ProgramParser::new();
        let program = parser
            .parse(input)
            .map_err(|e| e.map_token(|tok| tok.to_string()))?;
        self.run_program(program)
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] lalrpop_util::ParseError<usize, String, String>),
    #[error(transparent)]
    NotFoundError(#[from] NotFoundError),
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error("{}", ListDisplay(.0, "\n"))]
    TypeErrors(Vec<TypeError>),
    #[error("Check failed: {0} != {1}")]
    CheckError(Value, Value),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
}

pub type Pattern = Expr;

#[derive(Default, Clone, Debug)]
pub struct Query {
    bindings: HashMap<Symbol, AtomTerm>,
    atoms: Vec<Atom>,
}

impl Query {
    pub fn from_facts(facts: Vec<Fact>) -> Self {
        #[derive(PartialEq, Eq, Hash, Clone, Debug)]
        enum VarOrValue {
            Var(Symbol),
            Value(Value),
        }

        let mut aux_counter = 0;
        let mut uf = SparseUnionFind::<VarOrValue, ()>::default();
        let mut pre_atoms: Vec<(Symbol, Vec<VarOrValue>)> = vec![];

        for (i, fact) in facts.into_iter().enumerate() {
            let group_var = VarOrValue::Var(Symbol::from(format!("__group_{i}")));
            uf.insert(group_var.clone(), ());
            let group: Vec<Expr> = match fact {
                Fact::Eq(exprs) => exprs,
                Fact::Fact(expr) => vec![expr],
            };
            for expr in group {
                let vv = expr.fold(&mut |expr, mut child_pre_atoms| -> VarOrValue {
                    let vv = match expr {
                        Expr::Lit(lit) => VarOrValue::Value(lit.to_value()),
                        Expr::Var(var) => VarOrValue::Var(*var),
                        Expr::Call(op, _) => {
                            let aux = VarOrValue::Var(format!("_aux_{}", aux_counter).into());
                            aux_counter += 1;
                            child_pre_atoms.push(aux.clone());
                            pre_atoms.push((*op, child_pre_atoms));
                            aux
                        }
                    };
                    uf.insert(vv.clone(), ());
                    vv
                });
                uf.union(group_var.clone(), vv);
            }
        }

        let mut next_var_index = 0;
        let mut bindings = HashMap::default();

        for set in uf.sets() {
            let mut values: Vec<Value> = set
                .iter()
                .filter_map(|vv| match vv {
                    VarOrValue::Var(_) => None,
                    VarOrValue::Value(val) => Some(val.clone()),
                })
                .collect();

            if values.len() > 1 {
                panic!("too many values")
            }

            let atom_term = if let Some(value) = values.pop() {
                AtomTerm::Value(value)
            } else {
                debug_assert!(set.iter().all(|vv| matches!(vv, VarOrValue::Var(_))));
                let a = AtomTerm::Var(next_var_index);
                next_var_index += 1;
                a
            };

            assert!(values.is_empty());
            for vv in set {
                if let VarOrValue::Var(var) = vv {
                    let old = bindings.insert(var, atom_term.clone());
                    assert!(old.is_none());
                }
            }
        }

        let vv_to_atomterm = |vv: VarOrValue| match vv {
            VarOrValue::Var(v) => bindings[&v].clone(),
            VarOrValue::Value(val) => AtomTerm::Value(val),
        };
        let atoms = pre_atoms
            .into_iter()
            .map(|(sym, vvs)| Atom(sym, vvs.into_iter().map(vv_to_atomterm).collect()))
            .collect();

        log::debug!("atoms: {:?}", atoms);
        Self { bindings, atoms }
    }
}
