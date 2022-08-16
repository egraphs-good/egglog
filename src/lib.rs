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
use std::fmt::Write;
use std::hash::Hash;
use std::ops::Deref;
use std::{fmt::Debug, sync::Arc};
use typecheck::{AtomTerm, Bindings};

pub use value::*;

use gj::*;
use num_rational::BigRational;
use unionfind::*;
use util::*;

use crate::typecheck::TypeError;

#[derive(Clone)]
pub struct Function {
    decl: FunctionDecl,
    nodes: HashMap<Vec<Value>, Value>,
    updates: usize,
}

impl Function {
    pub fn new(decl: FunctionDecl) -> Self {
        Self {
            decl,
            nodes: Default::default(),
            updates: 0,
        }
    }

    pub fn rebuild(&mut self, uf: &mut UnionFind) -> usize {
        // FIXME this doesn't compute updates properly
        let n_unions = uf.n_unions();
        let old_nodes = std::mem::take(&mut self.nodes);
        for (mut args, value) in old_nodes {
            for (a, ty) in args.iter_mut().zip(&self.decl.schema.input) {
                if ty.is_sort() {
                    *a = uf.find_mut_value(a.clone())
                }
            }
            let _new_value = if self.decl.schema.output.is_sort() {
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

pub trait PrimitiveLike {
    fn name(&self) -> Symbol;
    fn accept(&self, types: &[Type]) -> Option<Type>;
    fn apply(&self, values: &[Value]) -> Option<Value>;
}

#[derive(Clone)]
pub struct Primitive(Arc<dyn PrimitiveLike>);

impl Deref for Primitive {
    type Target = dyn PrimitiveLike;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl Hash for Primitive {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Eq for Primitive {}
impl PartialEq for Primitive {
    fn eq(&self, other: &Self) -> bool {
        // this is a bit of a hack, but clippy says we don't want to compare the
        // vtables, just the data pointers
        std::ptr::eq(
            Arc::as_ptr(&self.0) as *const u8,
            Arc::as_ptr(&other.0) as *const u8,
        )
    }
}

impl Debug for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prim({})", self.0.name())
    }
}

impl<T: PrimitiveLike + 'static> From<T> for Primitive {
    fn from(p: T) -> Self {
        Self(Arc::new(p))
    }
}

pub struct SimplePrimitive {
    name: Symbol,
    input: Vec<Type>,
    output: Type,
    f: fn(&[Value]) -> Option<Value>,
}

impl PrimitiveLike for SimplePrimitive {
    fn name(&self) -> Symbol {
        self.name
    }
    fn accept(&self, types: &[Type]) -> Option<Type> {
        (types == self.input).then(|| self.output.clone())
    }
    fn apply(&self, values: &[Value]) -> Option<Value> {
        (self.f)(values)
    }
}

pub struct NotEqualPrimitive;

impl PrimitiveLike for NotEqualPrimitive {
    fn name(&self) -> Symbol {
        "!=".into()
    }

    fn accept(&self, types: &[Type]) -> Option<Type> {
        match types {
            [a, b] if a == b => Some(Type::Unit),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        (values[0] != values[1]).then(|| Value(ValueInner::Unit))
    }
}

fn default_primitives() -> Vec<Primitive> {
    macro_rules! prim {
        (@type ()) => { Type::Unit };
        (@type i64) => { Type::NumType(NumType::I64) };
        (@type BigRational) => { Type::NumType(NumType::Rational) };
        ($name: expr, |$($param:ident : $t:tt),*| -> $output:tt { $body:expr }) => {{
            Primitive::from(SimplePrimitive {
                name: Symbol::from($name),
                input: vec![$(prim!(@type $t)),*],
                output: prim!(@type $output),
                f: |values| {
                    let mut values = values.iter();
                    $(
                        let $param: $t = values.next().unwrap().clone().into();
                    )*
                    let opt: Option<_> = $body;
                    opt.map(Value::from)
                }
            })
        }};
    }

    #[rustfmt::skip]
    let vec = vec![
        NotEqualPrimitive.into(),
        prim!("<", |a: i64, b: i64| -> () { (a < b).then(|| ()) }),
        prim!("<=", |a: i64, b: i64| -> () { (a <= b).then(|| ()) }),
        prim!(">", |a: i64, b: i64| -> () { (a > b).then(|| ()) }),
        prim!(">=", |a: i64, b: i64| -> () { (a >= b).then(|| ()) }),
        prim!("+", |a: i64, b: i64| -> i64 { Some(a + b) }),
        prim!("-", |a: i64, b: i64| -> i64 { Some(a - b) }),
        prim!("*", |a: i64, b: i64| -> i64 { Some(a * b) }),
        prim!("/", |a: i64, b: i64| -> i64 { (b != 0).then(|| a / b ) }),
        prim!("%", |a: i64, b: i64| -> i64 { (b != 0).then(|| a % b ) }),
        prim!("&", |a: i64, b: i64| -> i64 { Some(a & b) }),
        prim!("|", |a: i64, b: i64| -> i64 { Some(a | b) }),
        prim!("^", |a: i64, b: i64| -> i64 { Some(a ^ b) }),
        prim!(">>", |a: i64, b: i64| -> i64 { Some(a >> b) }),
        prim!("<<", |a: i64, b: i64| -> i64 { Some(a << b) }),
        prim!("not-i64", |a: i64| -> i64 { Some(!a) }),
        prim!("max", |a: i64, b: i64| -> i64 { Some(a.max(b)) }),
        prim!("min", |a: i64, b: i64| -> i64 { Some(a.min(b)) }),
        prim!("<", |a: BigRational, b: BigRational| -> () { (a < b).then(|| ()) }),
        prim!("<=", |a: BigRational, b: BigRational| -> () { (a <= b).then(|| ()) }),
        prim!(">", |a: BigRational, b: BigRational| -> () { (a > b).then(|| ()) }),
        prim!(">=", |a: BigRational, b: BigRational| -> () { (a >= b).then(|| ()) }),
        prim!("+", |a: BigRational, b: BigRational| -> BigRational { Some(a + b) }),
        prim!("-", |a: BigRational, b: BigRational| -> BigRational { Some(a - b) }),
        prim!("*", |a: BigRational, b: BigRational| -> BigRational { Some(a * b) }),
        prim!("max", |a: BigRational, b: BigRational| -> BigRational { Some(a.max(b)) }),
        prim!("min", |a: BigRational, b: BigRational| -> BigRational { Some(a.min(b)) }),
    ];
    vec
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
    query: CompiledQuery,
    bindings: Bindings,
    head: Vec<Action>,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut egraph = Self {
            unionfind: Default::default(),
            sorts: Default::default(),
            functions: Default::default(),
            rules: Default::default(),
            globals: Default::default(),
            primitives: Default::default(),
        };
        for prim in default_primitives() {
            egraph.add_primitive(prim);
        }
        egraph
    }
}

#[derive(Debug, Error)]
#[error("Not found: {0}")]
pub struct NotFoundError(Expr);

impl EGraph {
    fn add_primitive(&mut self, prim: Primitive) {
        self.primitives.entry(prim.name()).or_default().push(prim);
    }

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
                        function.decl.schema,
                    )
                }
                assert_eq!(
                    output,
                    &self.bad_find_value(output.clone()),
                    "{name}({inputs:?}) = {output}\n{:?}",
                    function.decl.schema,
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

    pub fn eval_actions(
        &mut self,
        mut ctx: Option<Subst>,
        actions: &[Action],
    ) -> Result<(), Error> {
        let default = Subst::default();
        for action in actions {
            match action {
                Action::Panic(msg) => {
                    panic!("panic {} {:?}", msg, ctx.as_ref().unwrap_or(&default))
                }
                Action::Expr(e) => {
                    self.eval_expr(ctx.as_ref().unwrap_or(&default), e)?;
                }
                Action::Define(x, e) => {
                    if let Some(ctx) = ctx.as_mut() {
                        let value = self.eval_expr(ctx, e)?;
                        ctx.insert(*x, value);
                    } else {
                        let value = self.eval_expr(&default, e)?;
                        self.globals.insert(*x, value);
                    }
                }
                Action::Set(f, args, e) => {
                    let ctx = ctx.as_ref().unwrap_or(&default);
                    let values = args
                        .iter()
                        .map(|a| self.eval_expr(ctx, a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = self.eval_expr(ctx, e)?;
                    let function = self
                        .functions
                        .get_mut(f)
                        .ok_or_else(|| NotFoundError(e.clone()))?;
                    let old_value = function.nodes.insert(values.clone(), value.clone());

                    if let Some(old_value) = old_value {
                        if value != old_value {
                            match (function.decl.merge.as_ref(), &function.decl.schema.output) {
                                (None, Type::Unit) => (),
                                (None, Type::Sort(_)) => {
                                    self.unionfind.union_values(old_value, value);
                                }
                                (Some(expr), _) => {
                                    let mut ctx = Subst::default();
                                    ctx.insert("old".into(), old_value);
                                    ctx.insert("new".into(), value);
                                    let expr = expr.clone(); // break the borrow of `function`
                                    let new_value = self.eval_expr(&ctx, &expr)?;
                                    self.functions
                                        .get_mut(f)
                                        .unwrap()
                                        .nodes
                                        .insert(values, new_value);
                                }
                                _ => panic!("invalid merge function"),
                            }
                        }
                    }
                }
                Action::Union(a, b) => {
                    let ctx = ctx.as_ref().unwrap_or(&default);
                    let a = self.eval_expr(ctx, a)?;
                    let b = self.eval_expr(ctx, b)?;
                    self.unionfind.union_values(a, b);
                }
            }
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
                        // the check failed, so print out some useful info
                        for value in &values {
                            if let Value(ValueInner::Id(id)) = value {
                                let best = self.extract(*id).1;
                                println!("{}: {}", id, best);
                            }
                        }
                        return Err(Error::CheckError(values[0].clone(), v.clone()));
                    }
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
                    if let Some(f) = self.functions.get_mut(sym) {
                        // FIXME We don't have a unit value
                        assert_eq!(f.decl.schema.output, Type::Unit);
                        f.nodes
                            .get(&values)
                            .ok_or_else(|| NotFoundError(expr.clone()))?;
                    } else if self.primitives.contains_key(sym) {
                        // HACK
                        // we didn't typecheck so we don't know which prim to call
                        let val = self.eval_expr(ctx, expr)?;
                        assert_eq!(val, ().into());
                    } else {
                        return Err(Error::TypeError(TypeError::Unbound(*sym)));
                    }
                }
            },
        }
        Ok(())
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
            if let Type::Sort(sort) = ty {
                if !self.sorts.contains_key(sort) {
                    return Err(TypeError::UndefinedSort(*sort).into());
                }
            }
        }

        if let Type::Sort(sort) = &decl.schema.output {
            if !self.sorts.contains_key(sort) {
                return Err(TypeError::UndefinedSort(*sort).into());
            }
        }

        let old = self
            .functions
            .insert(decl.name, Function::new(decl.clone()));
        if old.is_some() {
            return Err(TypeError::FunctionAlreadyBound(decl.name).into());
        }

        Ok(())
    }

    pub fn declare_constructor(
        &mut self,
        name: impl Into<Symbol>,
        types: Vec<Type>,
        sort: impl Into<Symbol>,
    ) -> Result<(), Error> {
        let name = name.into();
        let sort = sort.into();
        self.declare_function(&FunctionDecl {
            name,
            schema: Schema {
                input: types,
                output: Type::Sort(sort),
            },
            merge: None,
            default: None,
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
                if let Some(function) = self.functions.get_mut(op) {
                    if let Some(value) = function.nodes.get(&values) {
                        Ok(value.clone())
                    } else {
                        match (function.decl.default.as_ref(), &function.decl.schema.output) {
                            (None, Type::Unit) => {
                                function.nodes.insert(values, Value(ValueInner::Unit));
                                Ok(Value(ValueInner::Unit))
                            }
                            (None, Type::Sort(_)) => {
                                let id = self.unionfind.make_set();
                                function.nodes.insert(values, Value(ValueInner::Id(id)));
                                Ok(Value(ValueInner::Id(id)))
                            }
                            (Some(default), _) => {
                                let default = default.clone(); // break the borrow
                                let value = self.eval_expr(ctx, &default)?;
                                let function = self.functions.get_mut(op).unwrap();
                                function.nodes.insert(values, value.clone());
                                Ok(value)
                            }
                            _ => panic!("invalid default"),
                        }
                    }
                } else if let Some(prims) = self.primitives.get(op) {
                    let mut res = None;
                    for prim in prims.iter() {
                        // HACK
                        let types = values.iter().map(|v| v.get_type()).collect::<Vec<_>>();
                        if prim.accept(&types).is_some() {
                            if res.is_none() {
                                res = prim.apply(&values);
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
                // dbg!(&rule);
                let mut substs = Vec::<Subst>::new();
                self.run_query(&rule.query, |values| {
                    substs.push(
                        rule.bindings
                            .iter()
                            .map(|(k, t)| {
                                let val = match t {
                                    AtomTerm::Var(sym) => {
                                        let i = rule.query.vars.get_index_of(sym).unwrap_or_else(
                                            || panic!("Couldn't find variable '{sym}'"),
                                        );
                                        values[i].clone()
                                    }
                                    AtomTerm::Value(val) => val.clone(),
                                };
                                (*k, val)
                            })
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
                let _result: Result<_, _> = self.eval_actions(Some(subst), &rule.head);
            }
        }
        self.rules = rules;
    }

    fn add_rule_with_name(&mut self, name: String, rule: ast::Rule) -> Result<Symbol, Error> {
        let name = Symbol::from(name);
        let mut ctx = typecheck::Context::new(self);
        let (atoms, bindings) = ctx.typecheck_query(&rule.body).map_err(Error::TypeErrors)?;
        let query = self.compile_gj_query(atoms, ctx.types);
        let compiled_rule = Rule {
            query,
            bindings,
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
            body: [Fact::Eq(vec![Expr::Var(var), rewrite.lhs])]
                .into_iter()
                .chain(rewrite.conditions)
                .collect(),
            head: vec![Action::Union(Expr::Var(var), rewrite.rhs)],
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
            Command::Extract { e, variants } => {
                if should_run {
                    // TODO typecheck
                    let value = self.eval_closed_expr(&e)?;
                    self.rebuild();
                    let id = Id::from(value);
                    log::info!("Extracting {e} at {id}");
                    let (cost, expr) = self.extract(id);
                    let mut msg = format!("Extracted with cost {cost}: {expr}");
                    if variants > 0 {
                        let exprs = self.extract_variants(id, variants);
                        let line = "\n    ";
                        let varnts = ListDisplay(&exprs, line);
                        write!(msg, "\nVariants of {expr}:{line}{varnts}").unwrap();
                    }
                    msg
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
            Command::Action(action) => {
                if should_run {
                    self.eval_actions(None, std::slice::from_ref(&action))?;
                    format!("Run {action}.")
                } else {
                    format!("Skipping running {action}.")
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
            Command::Query(_q) => {
                // let qsexp = sexp::Sexp::List(
                //     q.iter()
                //         .map(|fact| sexp::parse(&fact.to_string()).unwrap())
                //         .collect(),
                // );
                // let qcomp = self
                //     .compile_query(q)
                //     .unwrap_or_else(|_| panic!("Could not compile query"));
                // let mut res = vec![];
                // self.query(&qcomp, |v| {
                //     res.push(sexp::Sexp::List(
                //         v.iter()
                //             .map(|val| sexp::Sexp::Atom(sexp::Atom::S(format!("{}", val))))
                //             .collect(),
                //     ));
                // });
                // format!(
                //     "Query: {}\n  Bindings: {:?}\n  Results: {}",
                //     qsexp,
                //     qcomp,
                //     sexp::Sexp::List(res)
                // )
                todo!()
            }
            Command::Clear => {
                for f in self.functions.values_mut() {
                    f.nodes.clear();
                }
                "Cleared.".into()
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
    #[error("Errors:\n{}", ListDisplay(.0, "\n"))]
    TypeErrors(Vec<TypeError>),
    #[error("Check failed: {0} != {1}")]
    CheckError(Value, Value),
    #[error("Sort {0} already declared.")]
    SortAlreadyBound(Symbol),
}
