use crate::*;
use indexmap::map::Entry as IEntry;
use typechecking::TypeError;

pub struct Context<'a> {
    pub egraph: &'a EGraph,
    pub types: IndexMap<Symbol, ArcSort>,
    unit: ArcSort,
    errors: Vec<TypeError>,
    unionfind: UnionFind,
    nodes: HashMap<ENode, Id>,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
enum ENode {
    Func(Symbol, Vec<Id>),
    Prim(Primitive, Vec<Id>),
    ComputeFunc(Symbol, Vec<Id>),
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
    pub function_filters: Vec<Atom<Symbol>>,
    pub original_facts: Vec<Fact>,
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

pub(crate) struct ValueEq {}

impl PrimitiveLike for ValueEq {
    fn name(&self) -> Symbol {
        "value-eq".into()
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [a, b] if a.name() == b.name() => Some(a.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        assert_eq!(values.len(), 2);
        if values[0] == values[1] {
            Some(values[0])
        } else {
            None
        }
    }
}

impl<'a> Context<'a> {
    pub fn new(egraph: &'a EGraph) -> Self {
        Self {
            egraph,
            unit: egraph.proof_state.type_info.sorts[&Symbol::from(UNIT_SYM)].clone(),
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

    pub fn typecheck_query(
        &mut self,
        facts: &'a [Fact],
        actions: &'a [Action],
    ) -> Result<(Query, Vec<Action>), Vec<TypeError>> {
        eprintln!("typechecking query: {}", ListDisplay(facts, " "));
        for fact in facts {
            self.typecheck_fact(fact);
        }

        // congruence isn't strictly necessary, but it can eliminate some redundant atoms
        self.rebuild();

        // First find the canoncial version of each leaf
        let mut leaves = HashMap::<Id, Expr>::default();
        let mut canon = HashMap::<Symbol, Expr>::default();

        // Do literals first
        for (node, &id) in &self.nodes {
            match node {
                ENode::Literal(lit) => {
                    let old = leaves.insert(id, Expr::Lit(lit.clone()));
                    if let Some(Expr::Lit(old_lit)) = old {
                        panic!("Duplicate literal: {:?} {:?}", old_lit, lit);
                    }
                }
                _ => continue,
            }
        }
        // Globally bound variables next
        for (node, &id) in &self.nodes {
            match node {
                ENode::Var(var) => {
                    if self.egraph.global_bindings.get(var).is_some() {
                        match leaves.entry(id) {
                            Entry::Occupied(existing) => {
                                canon.insert(*var, existing.get().clone());
                            }
                            Entry::Vacant(v) => {
                                v.insert(Expr::Var(*var));
                            }
                        }
                    }
                }
                _ => continue,
            }
        }

        // Now do variables
        for (node, &id) in &self.nodes {
            debug_assert_eq!(id, self.unionfind.find(id));
            match node {
                ENode::Var(var) => match leaves.entry(id) {
                    Entry::Occupied(existing) => {
                        canon.insert(*var, existing.get().clone());
                    }
                    Entry::Vacant(v) => {
                        v.insert(Expr::Var(*var));
                    }
                },
                _ => continue,
            }
        }

        // replace canonical things in the actions
        let res_actions = actions.iter().map(|a| a.replace_canon(&canon)).collect();
        for (var, _expr) in canon {
            self.types.remove(&var);
        }

        let get_leaf = |id: &Id| -> AtomTerm {
            assert!(*id == self.unionfind.find(*id));
            let mk = || AtomTerm::Var(Symbol::from(format!("?__{}", id)));
            match leaves.get(id) {
                Some(Expr::Var(v)) => {
                    if let Some((_ty, value)) = self.egraph.global_bindings.get(v) {
                        AtomTerm::Value(*value)
                    } else {
                        AtomTerm::Var(*v)
                    }
                }
                Some(Expr::Lit(l)) => AtomTerm::Value(self.egraph.eval_lit(l)),
                _ => mk(),
            }
        };

        let mut query = Query {
            original_facts: facts.iter().cloned().collect(),
            atoms: vec![],
            filters: vec![],
            function_filters: vec![],
        };
        let mut query_eclasses = HashSet::<Id>::default();
        // Now we can fill in the nodes with the canonical leaves
        for (node, id) in &self.nodes {
            match node {
                ENode::Func(f, ids) => {
                    let args = ids.iter().chain([id]).map(get_leaf).collect();
                    for id in ids {
                        query_eclasses.insert(*id);
                    }
                    query.atoms.push(Atom { head: *f, args });
                }
                ENode::Prim(p, ids) => {
                    let mut args = vec![];
                    for child in ids {
                        let leaf = get_leaf(child);
                        if let AtomTerm::Var(v) = leaf {
                            if self.egraph.global_bindings.contains_key(&v) {
                                args.push(AtomTerm::Value(self.egraph.global_bindings[&v].1));
                                continue;
                            }
                        }
                        args.push(get_leaf(child));
                        query_eclasses.insert(*child);
                    }
                    args.push(get_leaf(id));
                    query.filters.push(Atom {
                        head: p.clone(),
                        args,
                    });
                }
                ENode::ComputeFunc(f, ids) => {
                    let args = ids.iter().chain([id]).map(get_leaf).collect();
                    for id in ids {
                        query_eclasses.insert(*id);
                    }
                    query.function_filters.push(Atom { head: *f, args });
                }
                _ => {}
            }
        }

        // filter for global variables
        for node in &self.nodes {
            if let ENode::Var(var) = node.0 {
                if let Some((_sort, value)) = self.egraph.global_bindings.get(var) {
                    let canon = get_leaf(node.1);

                    // canon is either a global variable or a literal
                    let canon_value = match canon {
                        AtomTerm::Var(v) => self.egraph.global_bindings[&v].1,
                        AtomTerm::Value(v) => v,
                    };
                    // we actually know the query won't fire
                    if canon_value != *value {
                        query.filters.push(Atom {
                            head: Primitive(Arc::new(ValueEq {})),
                            args: vec![
                                AtomTerm::Value(canon_value),
                                AtomTerm::Value(*value),
                                AtomTerm::Value(*value),
                            ],
                        })
                    }
                }
            }
        }

        eprintln!(
            "output filters: 
            {}",
            ListDisplay(&query.function_filters, "\n")
        );

        if self.errors.is_empty() {
            Ok((query, res_actions))
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
                let id = self.unionfind.find(id);
                match &mut node {
                    ENode::Func(_, children)
                    | ENode::Prim(_, children)
                    | ENode::ComputeFunc(_, children) => {
                        for child in children {
                            *child = self.unionfind.find(*child);
                        }
                    }
                    ENode::Var(_) | ENode::Literal(_) => {}
                }

                // reinsert and handle hit
                if let Some(old) = self.nodes.insert(node.clone(), id) {
                    keep_going = true;
                    self.unionfind.union_raw(old, id);
                }
            }
        }
    }

    fn typecheck_fact(&mut self, fact: &Fact) {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() == 2);
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
                                && !self.egraph.global_bindings.contains_key(v) =>
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

                ids.into_iter()
                    .reduce(|a, b| self.unionfind.union_raw(a, b));
            }
            Fact::Fact(e) => {
                self.check_query_expr(e, self.unit.clone());
            }
        }
    }

    fn check_query_expr(&mut self, expr: &Expr, expected: ArcSort) -> Id {
        match expr {
            Expr::Var(sym) => {
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
                } else if let Some(ty) = self.egraph.global_bindings.get(sym) {
                    Some(ty.0.clone())
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    None
                };
                (self.add_node(ENode::Var(*sym)), ty)
            }
            Expr::Lit(lit) => {
                let t = self.egraph.proof_state.type_info.infer_literal(lit);
                (self.add_node(ENode::Literal(lit.clone())), Some(t))
            }
            Expr::Compute(sym, args) => {
                if let Some(prims) = self.egraph.proof_state.type_info.primitives.get(sym) {
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
                } else if let Some(f) = self.egraph.functions.get(sym) {
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
                    (self.add_node(ENode::ComputeFunc(*sym, ids)), Some(t))
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    (self.unionfind.make_set(), None)
                }
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
                } else if let Some(_prims) = self.egraph.proof_state.type_info.primitives.get(sym) {
                    panic!("{sym} should have been desugared to compute");
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
            Action::Let(v, e) => {
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
            Action::Extract(variable, variants) => {
                let (_, _ty) = self.infer_expr(variable)?;
                let (_, _ty2) = self.infer_expr(variants)?;
                self.instructions.push(Instruction::Extract(2));
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
                    panic!("Base types cannot be unioned")
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
        if sym.to_string() == "iteration" {
            self.instructions.push(Instruction::Iteration);
            Ok((
                (),
                self.egraph
                    .proof_state
                    .type_info
                    .reserved_type("iteration".into())
                    .unwrap(),
            ))
        } else if let Some((sort, v)) = self.egraph().global_bindings.get(&sym) {
            self.instructions.push(Instruction::Value(v.clone()));
            Ok(((), sort.clone()))
        } else if let Some((i, _, ty)) = self.locals.get_full(&sym) {
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
        let func_type = self
            .egraph
            .proof_state
            .type_info
            .func_types
            .get(&f)
            .unwrap();
        self.instructions.push(Instruction::CallFunction(
            f,
            func_type.has_default || !func_type.has_merge,
        ));
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
            Expr::Var(v) if !self.is_variable(*v) => self.check_var(*v, ty),
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

    fn is_variable(&self, sym: Symbol) -> bool {
        self.egraph().global_bindings.contains_key(&sym)
    }

    fn infer_expr(&mut self, expr: &Expr) -> Result<(Self::T, ArcSort), TypeError> {
        match expr {
            Expr::Lit(lit) => {
                let t = self.do_lit(lit);
                Ok((t, self.egraph().proof_state.type_info.infer_literal(lit)))
            }
            Expr::Var(sym) => self.infer_var(*sym),
            Expr::Call(sym, args) | Expr::Compute(sym, args) => {
                if let Some(functype) = self.egraph().proof_state.type_info.func_types.get(sym) {
                    assert!(functype.input.len() == args.len());

                    let mut ts = vec![];
                    for (expected, arg) in functype.input.iter().zip(args) {
                        ts.push(self.check_expr(arg, expected.clone())?);
                    }

                    let t = self.do_function(*sym, ts);
                    Ok((t, functype.output.clone()))
                } else if let Some(prims) = self.egraph().proof_state.type_info.primitives.get(sym)
                {
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
                        inputs: tys.into_iter().map(|t| t.name()).collect(),
                    })
                } else {
                    panic!("Unbound function {}", sym);
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
    Iteration,
    Literal(Literal),
    Load(Load),
    Value(Value),
    // function to call, and whether to make defaults
    CallFunction(Symbol, bool),
    CallPrimitive(Primitive, usize),
    DeleteRow(Symbol),
    Set(Symbol),
    Union(usize),
    Extract(usize),
    Panic(String),
    Pop,
}

#[derive(Clone, Debug)]
pub struct Program(Vec<Instruction>);

impl EGraph {
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
        for instr in &program.0 {
            match instr {
                Instruction::Iteration => stack.push(self.iteration.into()),
                Instruction::Value(v) => stack.push(v.clone()),
                Instruction::Load(load) => match load {
                    Load::Stack(idx) => stack.push(stack[*idx]),
                    Load::Subst(idx) => stack.push(subst[*idx]),
                },
                Instruction::CallFunction(f, make_defaults_func) => {
                    let make_defaults = make_defaults && *make_defaults_func;
                    let function = self.functions.get_mut(f).unwrap();
                    let output_tag = function.schema.output.name();
                    let new_len = stack.len() - function.schema.input.len();
                    let values = &stack[new_len..];

                    if cfg!(debug_assertions) {
                        for (ty, val) in function.schema.input.iter().zip(values) {
                            assert_eq!(ty.name(), val.tag,);
                        }
                    }

                    let value = if let Some(out) = function.nodes.get(values) {
                        out.value
                    } else if make_defaults {
                        if function.merge.on_merge.is_some() {
                            panic!("No value found for function {} with values {:?}", f, values);
                        }
                        let ts = self.timestamp;
                        let out = &function.schema.output;
                        match function.decl.default.as_ref() {
                            None if out.name() == UNIT_SYM.into() => {
                                function.insert(values, Value::unit(), ts);
                                Value::unit()
                            }
                            None if out.is_eq_sort() => {
                                let id = self.unionfind.make_set();
                                let value = Value::from_id(out.name(), id);
                                function.insert(values, value, ts);
                                value
                            }
                            Some(default) => {
                                // TODO: this is not efficient due to cloning
                                let out = out.clone();
                                let default = default.clone();
                                let (_, value) = self.eval_expr(&default, Some(out), true)?;
                                self.functions.get_mut(f).unwrap().insert(values, value, ts);
                                value
                            }
                            _ => {
                                return Err(Error::NotFoundError(NotFoundError(Expr::Var(
                                    format!("No value found for {f} {:?}", values).into(),
                                ))))
                            }
                        }
                    } else {
                        return Err(Error::NotFoundError(NotFoundError(Expr::Var(
                            format!("No value found for {f} {:?}", values).into(),
                        ))));
                    };

                    debug_assert_eq!(output_tag, value.tag);
                    stack.truncate(new_len);
                    stack.push(value);
                }
                Instruction::CallPrimitive(p, arity) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    if let Some(value) = p.apply(values, self) {
                        stack.truncate(new_len);
                        stack.push(value);
                    } else {
                        return Err(Error::PrimitiveError(p.clone(), values.to_vec()));
                    }
                }
                Instruction::Set(f) => {
                    assert!(make_defaults);
                    let function = self.functions.get_mut(f).unwrap();
                    // desugaring should have desugared
                    // set to union
                    // except for setting the parent relation
                    let new_value = stack.pop().unwrap();
                    let new_len = stack.len() - function.schema.input.len();
                    let args = &stack[new_len..];

                    // We should only have canonical values here: omit the canonicalization step
                    let old_value = function.get(args);

                    if let Some(old_value) = old_value {
                        if new_value != old_value {
                            let merged: Value = match function.merge.merge_vals.clone() {
                                MergeFn::AssertEq => {
                                    return Err(Error::MergeError(*f, new_value, old_value));
                                }
                                MergeFn::Union => {
                                    panic!("There should be no union merge functions after term encoding");
                                }
                                MergeFn::Expr(merge_prog) => {
                                    let values = [old_value, new_value];
                                    let old_len = stack.len();
                                    self.run_actions(stack, &values, &merge_prog, true)?;
                                    let result = stack.pop().unwrap();
                                    stack.truncate(old_len);
                                    result
                                }
                            };
                            let args = &stack[new_len..];
                            let function = self.functions.get_mut(f).unwrap();
                            function.insert(args, merged, self.timestamp);

                            if new_value != old_value {
                                // re-borrow
                                let function = self.functions.get_mut(f).unwrap();
                                if let Some(prog) = function.merge.on_merge.clone() {
                                    let values = [old_value, new_value];
                                    // XXX: we get an error if we pass the current
                                    // stack and then truncate it to the old length.
                                    // Why?
                                    self.run_actions(&mut Vec::new(), &values, &prog, true)?;
                                }
                            }
                        }
                    } else {
                        function.insert(args, new_value, self.timestamp);
                    }
                    stack.truncate(new_len)
                }
                Instruction::Union(_arity) => {
                    panic!("term encoding gets rid of union");
                }
                Instruction::Extract(arity) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    let new_len = stack.len() - arity;
                    let mut termdag = TermDag::default();
                    let num_sort = values[1].tag;
                    assert!(num_sort.to_string() == "i64");

                    let variants = values[1].bits as i64;
                    if variants == 0 {
                        let (cost, expr) = self.extract(
                            values[0],
                            &mut termdag,
                            self.proof_state
                                .type_info
                                .sorts
                                .get(&values[0].tag)
                                .unwrap(),
                        );
                        log::info!("extracted with cost {cost}: {}", termdag.to_string(&expr));
                    } else {
                        if variants < 0 {
                            panic!("Cannot extract negative number of variants");
                        }
                        let extracted =
                            self.extract_variants(values[0], variants as usize, &mut termdag);
                        log::info!("extracted variants:");
                        for expr in extracted {
                            log::info!("   {}", termdag.to_string(&expr));
                        }
                    }

                    stack.truncate(new_len);
                }
                Instruction::Panic(msg) => panic!("Panic: {}", msg),
                Instruction::Literal(lit) => match lit {
                    Literal::Int(i) => stack.push(Value::from(*i)),
                    Literal::F64(f) => stack.push(Value::from(*f)),
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
                    function.remove(args, self.timestamp);
                    stack.truncate(new_len);
                }
            }
        }
        Ok(())
    }
}
