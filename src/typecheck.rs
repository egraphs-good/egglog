use crate::*;
use indexmap::map::Entry as IEntry;

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("Arity mismatch, expected {expected} args: {expr}")]
    Arity { expr: Expr, expected: usize },
    #[error(
        "Type mismatch: expr = {expr}, expected = {}, actual = {}, reason: {reason}", 
        .expected.name(), .actual.name(),
    )]
    Mismatch {
        expr: Expr,
        expected: ArcSort,
        actual: ArcSort,
        reason: String,
    },
    #[error("Tried to unify too many literals: {}", ListDisplay(.0, "\n"))]
    TooManyLiterals(Vec<Literal>),
    #[error("Unbound symbol {0}")]
    Unbound(Symbol),
    #[error("Undefined sort {0}")]
    UndefinedSort(Symbol),
    #[error("Function already bound {0}")]
    FunctionAlreadyBound(Symbol),
    #[error("Cannot type a variable as unit: {0}")]
    UnitVar(Symbol),
    #[error("Failed to infer a type for: {0}")]
    InferenceFailure(Expr),
    #[error("No matching primitive for: ({op} {})", ListDisplay(.inputs, " "))]
    NoMatchingPrimitive { op: Symbol, inputs: Vec<Symbol> },
    #[error("Variable {0} was already defined")]
    AlreadyDefined(Symbol),
}

pub struct Context<'a> {
    pub egraph: &'a EGraph,
    pub types: IndexMap<Symbol, ArcSort>,
    unit: ArcSort,
    errors: Vec<TypeError>,
    unionfind: UnionFind,
    nodes: HashMap<ENode, Id>,
}

#[derive(Hash, Eq, PartialEq)]
enum ENode {
    Func(Symbol, Vec<Id>),
    Prim(Primitive, Vec<Id>),
    Literal(Literal),
    Var(Symbol),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomTerm {
    Var(Symbol),
    Value(Value),
}

impl std::fmt::Display for AtomTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomTerm::Var(v) => write!(f, "{}", v),
            AtomTerm::Value(_) => write!(f, "<value>"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom<T> {
    pub head: T,
    pub args: Vec<AtomTerm>,
}

impl<T: std::fmt::Display> std::fmt::Display for Atom<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {}) ", self.head, ListDisplay(&self.args, " "))
    }
}

#[derive(Default, Debug, Clone)]
pub struct Query {
    pub atoms: Vec<Atom<Symbol>>,
    pub filters: Vec<Atom<Primitive>>,
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for atom in &self.atoms {
            write!(f, "{atom}")?;
        }
        if !self.filters.is_empty() {
            write!(f, "where ")?;
            for filter in &self.filters {
                write!(
                    f,
                    "({} {}) ",
                    filter.head.name(),
                    ListDisplay(&filter.args, " ")
                )?;
            }
        }
        Ok(())
    }
}

impl<T> Atom<T> {
    pub fn vars(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.args.iter().filter_map(|t| match t {
            AtomTerm::Var(v) => Some(*v),
            AtomTerm::Value(_) => None,
        })
    }
}
impl<'a> Context<'a> {
    pub fn new(egraph: &'a EGraph) -> Self {
        Self {
            egraph,
            unit: egraph.sorts[&"Unit".into()].clone(),
            types: Default::default(),
            errors: Vec::default(),
            unionfind: UnionFind::default(),
            nodes: HashMap::default(),
        }
    }

    fn add_node(&mut self, node: ENode) -> Id {
        let entry = self.nodes.entry(node);
        *entry.or_insert_with(|| self.unionfind.make_set())
    }

    pub fn typecheck_query(&mut self, facts: &'a [Fact]) -> Result<Query, Vec<TypeError>> {
        for fact in facts {
            self.typecheck_fact(fact);
        }

        // congruence isn't strictly necessary, but it can eliminate some redundant atoms
        self.rebuild();

        // First find the canoncial version of each leaf
        let mut leaves = HashMap::<Id, AtomTerm>::default();
        for (node, &id) in &self.nodes {
            debug_assert_eq!(id, self.unionfind.find(id));
            match node {
                ENode::Literal(lit) => {
                    let old = leaves.insert(id, AtomTerm::Value(self.egraph.eval_lit(lit)));
                    if let Some(AtomTerm::Value(old)) = old {
                        panic!("Duplicate literal: {:?} {:?}", old, lit);
                    }
                }
                ENode::Var(var) => {
                    leaves.entry(id).or_insert_with(|| AtomTerm::Var(*var));
                }
                _ => continue,
            }
        }

        let get_leaf = |id: &Id| -> AtomTerm {
            let mk = || AtomTerm::Var(Symbol::from(format!("?__{}", id)));
            leaves.get(id).cloned().unwrap_or_else(mk)
        };

        let mut query = Query::default();
        // Now we can fill in the nodes with the canonical leaves
        for (node, id) in &self.nodes {
            match node {
                ENode::Func(f, ids) => {
                    let args = ids.iter().chain([id]).map(get_leaf).collect();
                    query.atoms.push(Atom { head: *f, args });
                }
                ENode::Prim(p, ids) => {
                    let args = ids.iter().chain([id]).map(get_leaf).collect();
                    query.filters.push(Atom {
                        head: p.clone(),
                        args,
                    });
                }
                _ => {}
            }
        }

        if self.errors.is_empty() {
            Ok(query)
        } else {
            Err(self.errors.clone())
        }
    }

    fn rebuild(&mut self) {
        let mut keep_going = true;
        while keep_going {
            keep_going = false;
            let nodes = std::mem::take(&mut self.nodes);
            for (mut node, id) in nodes {
                // canonicalize
                let id = self.unionfind.find_mut(id);
                if let ENode::Func(_, children) | ENode::Prim(_, children) = &mut node {
                    for child in children {
                        *child = self.unionfind.find_mut(*child);
                    }
                }

                // reinsert and handle hit
                if let Some(old) = self.nodes.insert(node, id) {
                    keep_going = true;
                    self.unionfind.union(old, id);
                }
            }
        }
    }

    fn typecheck_fact(&mut self, fact: &Fact) {
        match fact {
            Fact::Eq(exprs) => {
                let mut later = vec![];
                let mut ty: Option<ArcSort> = None;
                let mut ids = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    match (expr, &ty) {
                        (_, Some(expected)) => {
                            ids.push(self.check_query_expr(expr, expected.clone()))
                        }
                        // This is a variable the we couldn't infer the type of,
                        // so we'll try again later when we can check its type
                        (Expr::Var(v), None)
                            if !self.types.contains_key(v)
                                && !self.egraph.functions.contains_key(v) =>
                        {
                            later.push(expr)
                        }
                        (_, None) => match self.infer_query_expr(expr) {
                            (_, None) => (),
                            (id, Some(t)) => {
                                ty = Some(t);
                                ids.push(id);
                            }
                        },
                    }
                }

                if let Some(ty) = ty {
                    for e in later {
                        ids.push(self.check_query_expr(e, ty.clone()));
                    }
                } else {
                    for e in later {
                        self.errors.push(TypeError::InferenceFailure(e.clone()));
                    }
                }

                ids.into_iter().reduce(|a, b| self.unionfind.union(a, b));
            }
            Fact::Fact(e) => {
                self.check_query_expr(e, self.unit.clone());
            }
        }
    }

    fn check_query_expr(&mut self, expr: &Expr, expected: ArcSort) -> Id {
        match expr {
            Expr::Var(sym) if !self.egraph.functions.contains_key(sym) => {
                match self.types.entry(*sym) {
                    IEntry::Occupied(ty) => {
                        // TODO name comparison??
                        if ty.get().name() != expected.name() {
                            self.errors.push(TypeError::Mismatch {
                                expr: expr.clone(),
                                expected,
                                actual: ty.get().clone(),
                                reason: "mismatch".into(),
                            })
                        }
                    }
                    // we can actually bind the variable here
                    IEntry::Vacant(entry) => {
                        entry.insert(expected);
                    }
                }
                self.add_node(ENode::Var(*sym))
            }
            _ => {
                let (id, actual) = self.infer_query_expr(expr);
                if let Some(actual) = actual {
                    if actual.name() != expected.name() {
                        self.errors.push(TypeError::Mismatch {
                            expr: expr.clone(),
                            expected,
                            actual,
                            reason: "mismatch".into(),
                        })
                    }
                }
                id
            }
        }
    }

    fn infer_query_expr(&mut self, expr: &Expr) -> (Id, Option<ArcSort>) {
        match expr {
            Expr::Var(sym) => {
                if self.egraph.functions.contains_key(sym) {
                    return self.infer_query_expr(&Expr::call(*sym, []));
                }
                let ty = if let Some(ty) = self.types.get(sym) {
                    Some(ty.clone())
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    None
                };
                (self.add_node(ENode::Var(*sym)), ty)
            }
            Expr::Lit(lit) => {
                let t = self.egraph.infer_literal(lit);
                (self.add_node(ENode::Literal(lit.clone())), Some(t))
            }
            Expr::Call(sym, args) => {
                if let Some(f) = self.egraph.functions.get(sym) {
                    if f.schema.input.len() != args.len() {
                        self.errors.push(TypeError::Arity {
                            expr: expr.clone(),
                            expected: f.schema.input.len(),
                        });
                    }

                    let ids: Vec<Id> = args
                        .iter()
                        .zip(&f.schema.input)
                        .map(|(arg, ty)| self.check_query_expr(arg, ty.clone()))
                        .collect();
                    let t = f.schema.output.clone();
                    (self.add_node(ENode::Func(*sym, ids)), Some(t))
                } else if let Some(prims) = self.egraph.primitives.get(sym) {
                    let (ids, arg_tys): (Vec<Id>, Vec<Option<ArcSort>>) =
                        args.iter().map(|arg| self.infer_query_expr(arg)).unzip();

                    if let Some(arg_tys) = arg_tys.iter().cloned().collect::<Option<Vec<ArcSort>>>()
                    {
                        for prim in prims {
                            if let Some(output_type) = prim.accept(&arg_tys) {
                                let id = self.add_node(ENode::Prim(prim.clone(), ids));
                                return (id, Some(output_type));
                            }
                        }
                        self.errors.push(TypeError::NoMatchingPrimitive {
                            op: *sym,
                            inputs: arg_tys.iter().map(|t| t.name()).collect(),
                        });
                    }

                    (self.unionfind.make_set(), None)
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    (self.unionfind.make_set(), None)
                }
            }
        }
    }
}

struct ActionChecker<'a> {
    egraph: &'a EGraph,
    types: &'a IndexMap<Symbol, ArcSort>,
    locals: IndexMap<Symbol, ArcSort>,
    instructions: Vec<Instruction>,
}

impl<'a> ActionChecker<'a> {
    fn check_action(&mut self, action: &Action) -> Result<(), TypeError> {
        match action {
            Action::Define(v, e) => {
                if self.types.contains_key(v) || self.locals.contains_key(v) {
                    return Err(TypeError::AlreadyDefined(*v));
                }
                let (_, ty) = self.infer_expr(e)?;
                self.locals.insert(*v, ty);
                Ok(())
            }
            Action::Set(f, args, val) => {
                let fake_call = Expr::Call(*f, args.clone());
                let (_, ty) = self.infer_expr(&fake_call)?;
                let fake_instr = self.instructions.pop().unwrap();
                assert!(matches!(fake_instr, Instruction::CallFunction(..)));
                self.check_expr(val, ty)?;
                self.instructions.push(Instruction::Set(*f));
                Ok(())
            }
            Action::Delete(f, args) => {
                let fake_call = Expr::Call(*f, args.clone());
                let (_, _ty) = self.infer_expr(&fake_call)?;
                let fake_instr = self.instructions.pop().unwrap();
                assert!(matches!(fake_instr, Instruction::CallFunction(..)));
                self.instructions.push(Instruction::DeleteRow(*f));
                Ok(())
            }
            Action::Union(a, b) => {
                let (_, ty) = self.infer_expr(a)?;
                if !ty.is_eq_sort() {
                    panic!("no error for this yet")
                }
                self.check_expr(b, ty)?;
                self.instructions.push(Instruction::Union(2));
                Ok(())
            }
            Action::Panic(msg) => {
                self.instructions.push(Instruction::Panic(msg.clone()));
                Ok(())
            }
            Action::Expr(expr) => {
                self.infer_expr(expr)?;
                self.instructions.push(Instruction::Pop);
                Ok(())
            }
        }
    }
}

impl<'a> ExprChecker<'a> for ActionChecker<'a> {
    type T = ();

    fn egraph(&self) -> &'a EGraph {
        self.egraph
    }

    fn do_lit(&mut self, lit: &Literal) -> Self::T {
        self.instructions.push(Instruction::Literal(lit.clone()));
    }

    fn infer_var(&mut self, sym: Symbol) -> Result<(Self::T, ArcSort), TypeError> {
        if let Some((i, _, ty)) = self.locals.get_full(&sym) {
            self.instructions.push(Instruction::Load(Load::Stack(i)));
            Ok(((), ty.clone()))
        } else if let Some((i, _, ty)) = self.types.get_full(&sym) {
            self.instructions.push(Instruction::Load(Load::Subst(i)));
            Ok(((), ty.clone()))
        } else {
            Err(TypeError::Unbound(sym))
        }
    }

    fn do_function(&mut self, f: Symbol, _args: Vec<Self::T>) -> Self::T {
        self.instructions.push(Instruction::CallFunction(f));
    }

    fn do_prim(&mut self, prim: Primitive, args: Vec<Self::T>) -> Self::T {
        self.instructions
            .push(Instruction::CallPrimitive(prim, args.len()));
    }
}

trait ExprChecker<'a> {
    type T;
    fn egraph(&self) -> &'a EGraph;
    fn do_lit(&mut self, lit: &Literal) -> Self::T;
    fn do_function(&mut self, f: Symbol, args: Vec<Self::T>) -> Self::T;
    fn do_prim(&mut self, prim: Primitive, args: Vec<Self::T>) -> Self::T;

    fn infer_var(&mut self, var: Symbol) -> Result<(Self::T, ArcSort), TypeError>;
    fn check_var(&mut self, var: Symbol, ty: ArcSort) -> Result<Self::T, TypeError> {
        let (t, actual) = self.infer_var(var)?;
        if actual.name() != ty.name() {
            Err(TypeError::Mismatch {
                expr: Expr::Var(var),
                expected: ty,
                actual,
                reason: "mismatch".into(),
            })
        } else {
            Ok(t)
        }
    }

    fn check_expr(&mut self, expr: &Expr, ty: ArcSort) -> Result<Self::T, TypeError> {
        match expr {
            Expr::Var(v) if !self.egraph().functions.contains_key(v) => self.check_var(*v, ty),
            _ => {
                let (t, actual) = self.infer_expr(expr)?;
                if actual.name() != ty.name() {
                    Err(TypeError::Mismatch {
                        expr: expr.clone(),
                        expected: ty,
                        actual,
                        reason: "mismatch".into(),
                    })
                } else {
                    Ok(t)
                }
            }
        }
    }

    fn infer_expr(&mut self, expr: &Expr) -> Result<(Self::T, ArcSort), TypeError> {
        match expr {
            Expr::Lit(lit) => {
                let t = self.do_lit(lit);
                Ok((t, self.egraph().infer_literal(lit)))
            }
            Expr::Var(sym) => {
                if self.egraph().functions.contains_key(sym) {
                    return self.infer_expr(&Expr::call(*sym, []));
                }
                self.infer_var(*sym)
            }
            Expr::Call(sym, args) => {
                if let Some(f) = self.egraph().functions.get(sym) {
                    if f.schema.input.len() != args.len() {
                        return Err(TypeError::Arity {
                            expr: expr.clone(),
                            expected: f.schema.input.len(),
                        });
                    }

                    let mut ts = vec![];
                    for (expected, arg) in f.schema.input.iter().zip(args) {
                        ts.push(self.check_expr(arg, expected.clone())?);
                    }

                    let t = self.do_function(*sym, ts);
                    Ok((t, f.schema.output.clone()))
                } else if let Some(prims) = self.egraph().primitives.get(sym) {
                    let mut ts = Vec::with_capacity(args.len());
                    let mut tys = Vec::with_capacity(args.len());
                    for arg in args {
                        let (t, ty) = self.infer_expr(arg)?;
                        ts.push(t);
                        tys.push(ty);
                    }

                    for prim in prims {
                        if let Some(output_type) = prim.accept(&tys) {
                            let t = self.do_prim(prim.clone(), ts);
                            return Ok((t, output_type));
                        }
                    }

                    Err(TypeError::NoMatchingPrimitive {
                        op: *sym,
                        inputs: tys.iter().map(|ty| ty.name()).collect(),
                    })
                } else {
                    Err(TypeError::Unbound(*sym))
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
enum Load {
    Stack(usize),
    Subst(usize),
}

#[derive(Clone, Debug)]
enum Instruction {
    Literal(Literal),
    Load(Load),
    CallFunction(Symbol),
    CallPrimitive(Primitive, usize),
    DeleteRow(Symbol),
    Set(Symbol),
    Union(usize),
    Panic(String),
    Pop,
}

#[derive(Clone, Debug)]
pub struct Program(Vec<Instruction>);

impl EGraph {
    fn infer_literal(&self, lit: &Literal) -> ArcSort {
        match lit {
            Literal::Int(_) => self.sorts.get(&"i64".into()),
            Literal::String(_) => self.sorts.get(&"String".into()),
            Literal::Unit => self.sorts.get(&"Unit".into()),
        }
        .unwrap()
        .clone()
    }

    pub fn compile_actions(
        &self,
        types: &IndexMap<Symbol, ArcSort>,
        actions: &[Action],
    ) -> Result<Program, Vec<TypeError>> {
        let mut checker = ActionChecker {
            egraph: self,
            types,
            locals: IndexMap::default(),
            instructions: Vec::new(),
        };

        let mut errors = vec![];
        for a in actions {
            if let Err(err) = checker.check_action(a) {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            for _ in 0..checker.locals.len() {
                checker.instructions.push(Instruction::Pop);
            }
            Ok(Program(checker.instructions))
        } else {
            Err(errors)
        }
    }

    pub fn compile_expr(
        &self,
        types: &IndexMap<Symbol, ArcSort>,
        expr: &Expr,
        expected_type: Option<ArcSort>,
    ) -> Result<(ArcSort, Program), Vec<TypeError>> {
        let mut checker = ActionChecker {
            egraph: self,
            types,
            locals: IndexMap::default(),
            instructions: Vec::new(),
        };

        let t: ArcSort = if let Some(expected) = expected_type {
            checker
                .check_expr(expr, expected.clone())
                .map_err(|err| vec![err])?;
            expected
        } else {
            checker.infer_expr(expr).map_err(|err| vec![err])?.1
        };

        Ok((t, Program(checker.instructions)))
    }

    pub fn run_actions(
        &mut self,
        stack: &mut Vec<Value>,
        subst: &[Value],
        program: &Program,
        make_defaults: bool,
    ) -> Result<(), Error> {
        // println!("{:?}", program);
        for instr in &program.0 {
            match instr {
                Instruction::Load(load) => match load {
                    Load::Stack(idx) => stack.push(stack[*idx]),
                    Load::Subst(idx) => stack.push(subst[*idx]),
                },
                Instruction::CallFunction(f) => {
                    let function = self.functions.get_mut(f).unwrap();
                    let new_len = stack.len() - function.schema.input.len();
                    let values = &stack[new_len..];

                    if cfg!(debug_assertions) {
                        for (ty, val) in function.schema.input.iter().zip(values) {
                            assert_eq!(ty.name(), val.tag);
                        }
                    }

                    let value = if let Some(out) = function.nodes.get(values) {
                        out.value
                    } else if make_defaults {
                        let ts = self.timestamp;
                        self.saturated = false;
                        let out = &function.schema.output;
                        match function.decl.default.as_ref() {
                            None if out.name() == "Unit".into() => {
                                function.insert(values.into(), Value::unit(), ts);
                                Value::unit()
                            }
                            None if out.is_eq_sort() => {
                                let id = self.unionfind.make_set();
                                let value = Value::from_id(out.name(), id);
                                function.insert(values.into(), value, ts);
                                value
                            }
                            Some(_default) => {
                                todo!("Handle default expr")
                                // let default = default.clone(); // break the borrow
                                // let value = self.eval_expr(ctx, &default)?;
                                // let function = self.functions.get_mut(f).unwrap();
                                // function.insert(values.to_vec(), value, ts);
                                // Ok(value)
                            }
                            _ => panic!("invalid default for {:?}", function.decl.name),
                        }
                    } else {
                        return Err(Error::NotFoundError(NotFoundError(Expr::Var(
                            format!("fake expression {f} {:?}", values).into(),
                        ))));
                    };
                    stack.truncate(new_len);
                    stack.push(value);
                }
                Instruction::CallPrimitive(p, arity) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    if let Some(value) = p.apply(values) {
                        stack.truncate(new_len);
                        stack.push(value);
                    } else {
                        panic!("prim was partial... do we allow this?");
                        // return;
                    }
                }
                Instruction::Set(f) => {
                    assert!(make_defaults);
                    let function = self.functions.get_mut(f).unwrap();
                    let new_value = stack.pop().unwrap();
                    let new_len = stack.len() - function.schema.input.len();
                    let args = &stack[new_len..];
                    let old_value = function.insert(args.into(), new_value, self.timestamp);

                    // if the value does not exist or the two values differ
                    if old_value.is_none() || old_value != Some(new_value) {
                        self.saturated = false;
                    }

                    if let Some(old_value) = old_value {
                        if new_value != old_value {
                            self.saturated = false;
                            let merged: Value = match function.merge.clone() {
                                MergeFn::AssertEq => panic!("No error for this yet"),
                                MergeFn::Union => self.unionfind.union_values(old_value, new_value),
                                MergeFn::Expr(merge_prog) => {
                                    let values = [old_value, new_value];
                                    let old_len = stack.len();
                                    self.run_actions(stack, &values, &merge_prog, true)?;
                                    assert_eq!(stack.len(), old_len + 1);
                                    stack.pop().unwrap()
                                }
                            };
                            // re-borrow
                            let args = &stack[new_len..];
                            let function = self.functions.get_mut(f).unwrap();
                            function.insert(args.into(), merged, self.timestamp);
                        }
                    }
                    stack.truncate(new_len)
                }
                Instruction::Union(arity) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    let first = self.unionfind.find(Id::from(values[0].bits as usize));
                    values[1..].iter().fold(first, |a, b| {
                        let b = self.unionfind.find(Id::from(b.bits as usize));
                        if a != b {
                            self.saturated = false;
                        }
                        self.unionfind.union(a, b)
                    });
                    stack.truncate(new_len);
                }
                Instruction::Panic(msg) => panic!("Panic: {}", msg),
                Instruction::Literal(lit) => match lit {
                    Literal::Int(i) => stack.push(Value::from(*i)),
                    Literal::String(s) => stack.push(Value::from(*s)),
                    Literal::Unit => stack.push(Value::unit()),
                },
                Instruction::Pop => {
                    stack.pop().unwrap();
                }
                Instruction::DeleteRow(f) => {
                    let function = self.functions.get_mut(f).unwrap();
                    let new_len = stack.len() - function.schema.input.len();
                    let args = &stack[new_len..];
                    let old_value = function.nodes.remove(args);
                    if old_value.is_some() {
                        self.saturated = false;
                    }
                    stack.truncate(new_len);
                }
            }
        }
        Ok(())
    }
}
