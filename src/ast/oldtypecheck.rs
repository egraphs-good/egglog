use std::ops::AddAssign;

use crate::{constraint::AllEqualTypeConstraint, *};
use hashbrown::HashMap;
use typechecking::TypeError;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Actions(pub(crate) Vec<NormAction>);
impl Actions {
    fn subst(&mut self, subst: &HashMap<Symbol, AtomTerm>) {
        let actions = subst.iter().map(|(symbol, atom_term)| match atom_term {
            AtomTerm::Var(v) => NormAction::LetVar(*symbol, *v),
            AtomTerm::Literal(lit) => NormAction::LetLit(*symbol, lit.clone()),
            AtomTerm::Global(v) => NormAction::LetVar(*symbol, *v),
        });
        let existing_actions = std::mem::take(&mut self.0);
        self.0 = actions.chain(existing_actions).collect();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolOrEq {
    Symbol(Symbol),
    Eq,
}

impl From<Symbol> for SymbolOrEq {
    fn from(value: Symbol) -> Self {
        SymbolOrEq::Symbol(value)
    }
}

impl SymbolOrEq {
    pub fn is_eq(&self) -> bool {
        matches!(self, SymbolOrEq::Eq)
    }

    pub fn get_symbol(&self) -> Option<&Symbol> {
        if let SymbolOrEq::Symbol(symbol) = self {
            Some(symbol)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnresolvedCoreRule {
    pub body: Query<SymbolOrEq>,
    pub head: Actions,
}

#[derive(Debug, Clone)]
pub struct CanonicalizedCoreRule {
    pub body: Query<Symbol>,
    pub head: Actions,
}

#[derive(Debug, Clone, Default)]
pub struct ResolvedCoreRule {
    pub body: Query<ResolvedCall>,
    pub head: Actions,
}

#[derive(Debug, Clone)]
pub enum ResolvedCall {
    Func(Symbol),
    Primitive(Primitive),
}

impl Query<ResolvedCall> {
    pub fn filters(&self) -> impl Iterator<Item = Atom<Primitive>> + '_ {
        self.atoms.iter().filter_map(|atom| match &atom.head {
            ResolvedCall::Func(_) => None,
            ResolvedCall::Primitive(head) => Some(Atom {
                head: head.clone(),
                args: atom.args.clone(),
            }),
        })
    }

    pub fn funcs(&self) -> impl Iterator<Item = Atom<Symbol>> + '_ {
        self.atoms.iter().filter_map(|atom| match atom.head {
            ResolvedCall::Func(head) => Some(Atom {
                head,
                args: atom.args.clone(),
            }),
            ResolvedCall::Primitive(_) => None,
        })
    }
}

impl UnresolvedCoreRule {
    pub fn subst(&mut self, subst: &HashMap<Symbol, AtomTerm>) {
        for atom in &mut self.body.atoms {
            atom.subst(subst);
        }
        self.head.subst(subst);
    }
}

pub(crate) fn facts_to_query(body: &Vec<NormFact>, typeinfo: &TypeInfo) -> Query<SymbolOrEq> {
    fn to_atom_term(s: Symbol, typeinfo: &TypeInfo) -> AtomTerm {
        if typeinfo.is_global(s) {
            AtomTerm::Global(s)
        } else {
            AtomTerm::Var(s)
        }
    }
    let mut atoms = vec![];
    for fact in body {
        match fact {
            NormFact::Assign(symbol, NormExpr::Call(head, args))
            | NormFact::Compute(symbol, NormExpr::Call(head, args)) => {
                let args = args
                    .iter()
                    .chain(once(symbol))
                    .cloned()
                    .map(|s| to_atom_term(s, typeinfo))
                    .collect();
                let head = SymbolOrEq::Symbol(*head);
                atoms.push(Atom { head, args });
            }
            NormFact::AssignVar(lhs, rhs) => atoms.push(Atom {
                head: SymbolOrEq::Eq,
                args: vec![to_atom_term(*lhs, typeinfo), to_atom_term(*rhs, typeinfo)],
            }),
            NormFact::ConstrainEq(lhs, rhs) => atoms.push(Atom {
                head: SymbolOrEq::Eq,
                args: vec![to_atom_term(*lhs, typeinfo), to_atom_term(*rhs, typeinfo)],
            }),
            NormFact::AssignLit(symbol, lit) => atoms.push(Atom {
                head: SymbolOrEq::Eq,
                args: vec![
                    to_atom_term(*symbol, typeinfo),
                    AtomTerm::Literal(lit.clone()),
                ],
            }),
        }
    }
    Query { atoms }
}

impl NormRule {
    pub(crate) fn to_core_rule(&self, typeinfo: &TypeInfo) -> UnresolvedCoreRule {
        let NormRule { head, body } = self;
        UnresolvedCoreRule {
            body: facts_to_query(body, typeinfo),
            head: Actions(head.clone()),
        }
    }
}

pub struct Context<'a> {
    pub egraph: &'a mut EGraph,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomTerm {
    Var(Symbol),
    Literal(Literal),
    Global(Symbol),
}
impl AtomTerm {
    pub fn to_expr(&self) -> Expr {
        match self {
            AtomTerm::Var(v) => Expr::Var(*v),
            AtomTerm::Literal(l) => Expr::Lit(l.clone()),
            AtomTerm::Global(v) => Expr::Var(*v),
        }
    }
}

impl std::fmt::Display for AtomTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomTerm::Var(v) => write!(f, "{}", v),
            AtomTerm::Literal(lit) => write!(f, "{}", lit),
            AtomTerm::Global(g) => write!(f, "{}", g),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom<T> {
    pub head: T,
    pub args: Vec<AtomTerm>,
}

impl<T: std::fmt::Display> std::fmt::Display for Atom<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {}) ", self.head, ListDisplay(&self.args, " "))
    }
}

#[derive(Debug, Clone)]
pub struct Query<T> {
    pub atoms: Vec<Atom<T>>,
}

impl<T> Default for Query<T> {
    fn default() -> Self {
        Self {
            atoms: Default::default(),
        }
    }
}

impl Query<SymbolOrEq> {
    pub fn get_constraints(
        &self,
        type_info: &TypeInfo,
    ) -> Result<Vec<Constraint<AtomTerm, ArcSort>>, TypeError> {
        let mut constraints = vec![];
        for atom in self.atoms.iter() {
            constraints.extend(atom.get_constraints(type_info)?.into_iter());
        }
        Ok(constraints)
    }

    pub(crate) fn atom_terms(&self) -> HashSet<AtomTerm> {
        self.atoms
            .iter()
            .flat_map(|atom| atom.args.iter().cloned())
            .collect()
    }
}

impl<T> AddAssign for Query<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.atoms.extend(rhs.atoms);
    }
}

impl std::fmt::Display for Query<Symbol> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for atom in &self.atoms {
            writeln!(f, "{atom}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Query<ResolvedCall> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for atom in self.funcs() {
            writeln!(f, "{atom}")?;
        }
        let filters: Vec<_> = self.filters().collect();
        if !filters.is_empty() {
            writeln!(f, "where ")?;
            for filter in &filters {
                writeln!(
                    f,
                    "({} {})",
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
            AtomTerm::Literal(_) => None,
            AtomTerm::Global(_) => None,
        })
    }

    fn subst(&mut self, subst: &HashMap<Symbol, AtomTerm>) {
        for arg in self.args.iter_mut() {
            match arg {
                AtomTerm::Var(v) => {
                    if let Some(at) = subst.get(v) {
                        *arg = at.clone();
                    }
                }
                AtomTerm::Literal(_) => (),
                AtomTerm::Global(_) => (),
            }
        }
    }
}
impl Atom<Symbol> {
    pub(crate) fn to_expr(&self) -> Expr {
        let n = self.args.len();
        Expr::Call(
            self.head,
            self.args[0..n - 1]
                .iter()
                .map(|arg| arg.to_expr())
                .collect(),
        )
    }
}

pub(crate) struct ValueEq {
    pub unit: Arc<UnitSort>,
}

impl PrimitiveLike for ValueEq {
    fn name(&self) -> Symbol {
        "value-eq".into()
    }

    fn get_type_constraints(&self) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name())
            .with_exact_length(3)
            .with_output_sort(self.unit.clone())
            .into_box()
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        assert_eq!(values.len(), 2);
        if values[0] == values[1] {
            Some(Value::unit())
        } else {
            None
        }
    }
}

impl<'a> Context<'a> {
    pub fn new(egraph: &'a mut EGraph) -> Self {
        Self { egraph }
    }

    /// Transformed a UnresolvedCoreRule into a CanonicalizedCoreRule.
    /// In particular, it removes equality checks between variables and
    /// other arguments, and turns equality checks between non-variable arguments
    /// into a primitive equality check `value-eq`.
    pub fn canonicalize(&self, rule: UnresolvedCoreRule) -> CanonicalizedCoreRule {
        let mut result_rule = rule;
        loop {
            let mut to_subst = None;
            for atom in result_rule.body.atoms.iter() {
                if atom.head.is_eq() && atom.args[0] != atom.args[1] {
                    match &atom.args[..] {
                        [AtomTerm::Var(x), y] | [y, AtomTerm::Var(x)] => {
                            to_subst = Some((x, y));
                            break;
                        }
                        _ => (),
                    }
                }
            }
            if let Some((x, y)) = to_subst {
                result_rule.subst(&[(*x, y.clone())].into());
            } else {
                break;
            }
        }

        let atoms = result_rule
            .body
            .atoms
            .into_iter()
            .filter_map(|atom| match atom.head {
                SymbolOrEq::Eq => {
                    assert_eq!(atom.args.len(), 2);
                    match (&atom.args[0], &atom.args[1]) {
                        (AtomTerm::Var(v1), AtomTerm::Var(v2)) => {
                            assert_eq!(v1, v2);
                            None
                        }
                        (AtomTerm::Var(_), _) | (_, AtomTerm::Var(_)) => {
                            panic!("equalities between variable and non-variable arguments should have been canonicalized")
                        }
                        (at1, at2) => {
                            if at1 == at2 {
                                None
                            } else {
                                Some(Atom {
                                    head: "value-eq".into(),
                                    args: vec![
                                        atom.args[0].clone(),
                                        atom.args[1].clone(),
                                        AtomTerm::Literal(Literal::Unit),
                                    ],
                                })
                            }
                        },
                    }
                }
                SymbolOrEq::Symbol(symbol) => Some(Atom {
                    head: symbol,
                    args: atom.args,
                }),
            })
            .collect();
        CanonicalizedCoreRule {
            body: Query { atoms },
            head: result_rule.head,
        }
    }

    pub(crate) fn resolve_rule(
        &self,
        ctx: CommandId,
        rule: CanonicalizedCoreRule,
    ) -> Result<ResolvedCoreRule, TypeError> {
        let type_info = self.egraph.type_info();
        let local_types = type_info.local_types.get(&ctx).unwrap();
        let get_type = |arg: &AtomTerm| match arg {
            AtomTerm::Global(v) => type_info.lookup_global(v).unwrap().clone(),
            AtomTerm::Var(v) => local_types.get(v).unwrap().clone(),
            AtomTerm::Literal(lit) => type_info.infer_literal(lit),
        };
        let body_atoms: Vec<Atom<ResolvedCall>> = rule
            .body
            .atoms
            .into_iter()
            .map(|mut atom| {
                let symbol = atom.head;
                if let Some(ty) = type_info.func_types.get(&symbol) {
                    let expected = ty.input.iter().chain(once(&ty.output));
                    let expected = expected.map(|s| s.name());
                    let actual = atom.args.iter().map(get_type);
                    let actual = actual.map(|s| s.name());
                    if expected.eq(actual) {
                        return Atom {
                            head: ResolvedCall::Func(symbol),
                            args: std::mem::take(&mut atom.args),
                        };
                    }
                }
                if let Some(primitives) = type_info.primitives.get(&symbol) {
                    if primitives.len() == 1 {
                        Atom {
                            head: ResolvedCall::Primitive(primitives[0].clone()),
                            args: std::mem::take(&mut atom.args),
                        }
                    } else {
                        let tys: Vec<_> = atom
                            .args
                            // remove the output type since accept() only takes the input types
                            .split_last()
                            .unwrap()
                            .1
                            .iter()
                            .map(get_type)
                            .collect();
                        for primitive in primitives.iter() {
                            if primitive.accept(&tys).is_some() {
                                return Atom {
                                    head: ResolvedCall::Primitive(primitive.clone()),
                                    args: std::mem::take(&mut atom.args),
                                };
                            }
                        }
                        panic!("Impossible: there should be exactly one primitive that satisfy the type assignment")
                    }
                } else {
                    panic!("Impossible: atom symbols not bound anywhere")
                }
            })
            .collect();
        let body = Query { atoms: body_atoms };

        let result_rule = ResolvedCoreRule {
            body,
            head: rule.head,
        };
        Ok(result_rule)
    }

    pub(crate) fn compile_norm_rule(
        &mut self,
        ctx: CommandId,
        rule: &'a NormRule,
    ) -> Result<ResolvedCoreRule, Vec<TypeError>> {
        let rule = rule.to_core_rule(self.egraph.type_info());
        let rule = self.canonicalize(rule);
        let rule = self.resolve_rule(ctx, rule).map_err(|e| vec![e])?;
        Ok(rule)
    }
}

struct ActionChecker<'a> {
    egraph: &'a EGraph,
    types: &'a IndexMap<Symbol, ArcSort>,
    locals: IndexMap<Symbol, ArcSort>,
    instructions: Vec<Instruction>,
}

impl<'a> ActionChecker<'a> {
    // TODO: refactor this to get rid of "the type part" of the action checker
    fn check_action(&mut self, action: &Action) -> Result<(), TypeError> {
        match action {
            Action::Let(v, e) => {
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
        if let Some((sort, _v, _ts)) = self.egraph().global_bindings.get(&sym) {
            self.instructions.push(Instruction::Global(sym));
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
        let func_type = self.egraph.type_info().lookup_user_func(f).unwrap();
        self.instructions.push(Instruction::CallFunction(
            f,
            func_type.has_default || func_type.is_datatype,
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
                Ok((t, self.egraph().type_info().infer_literal(lit)))
            }
            Expr::Var(sym) => self.infer_var(*sym),
            Expr::Call(sym, args) => {
                if let Some(functype) = self.egraph().type_info().lookup_user_func(*sym) {
                    assert_eq!(
                        functype.input.len(),
                        args.len(),
                        "Got wrong number of arguments for function {}",
                        functype.name
                    );

                    let mut ts = vec![];
                    for (expected, arg) in functype.input.iter().zip(args) {
                        ts.push(self.check_expr(arg, expected.clone())?);
                    }

                    let t = self.do_function(*sym, ts);
                    Ok((t, functype.output.clone()))
                } else if let Some(prims) = self.egraph().type_info().primitives.get(sym) {
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
    Global(Symbol),
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

    fn perform_set(
        &mut self,
        table: Symbol,
        new_value: Value,
        stack: &mut Vec<Value>,
    ) -> Result<(), Error> {
        let function = self.functions.get_mut(&table).unwrap();

        let new_len = stack.len() - function.schema.input.len();
        // TODO would be nice to use slice here
        let args = &stack[new_len..];

        // We should only have canonical values here: omit the canonicalization step
        let old_value = function.get(args);

        if let Some(old_value) = old_value {
            if new_value != old_value {
                let merged: Value = match function.merge.merge_vals.clone() {
                    MergeFn::AssertEq => {
                        return Err(Error::MergeError(table, new_value, old_value));
                    }
                    MergeFn::Union => {
                        self.unionfind
                            .union_values(old_value, new_value, old_value.tag)
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
                if merged != old_value {
                    let args = &stack[new_len..];
                    let function = self.functions.get_mut(&table).unwrap();
                    function.insert(args, merged, self.timestamp);
                }
                // re-borrow
                let function = self.functions.get_mut(&table).unwrap();
                if let Some(prog) = function.merge.on_merge.clone() {
                    let values = [old_value, new_value];
                    // XXX: we get an error if we pass the current
                    // stack and then truncate it to the old length.
                    // Why?
                    self.run_actions(&mut Vec::new(), &values, &prog, true)?;
                }
            }
        } else {
            function.insert(args, new_value, self.timestamp);
        }
        Ok(())
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
                Instruction::Global(sym) => {
                    let (_ty, value, _ts) = self.global_bindings.get(sym).unwrap();
                    stack.push(*value);
                }
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

                    self.perform_set(*f, new_value, stack)?;
                    stack.truncate(new_len)
                }
                Instruction::Union(arity) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    let sort = values[0].tag;
                    let first = self.unionfind.find(Id::from(values[0].bits as usize));
                    values[1..].iter().fold(first, |a, b| {
                        let b = self.unionfind.find(Id::from(b.bits as usize));
                        self.unionfind.union(a, b, sort)
                    });
                    stack.truncate(new_len);
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
                        let (cost, term) = self.extract(
                            values[0],
                            &mut termdag,
                            self.type_info().sorts.get(&values[0].tag).unwrap(),
                        );
                        let extracted = termdag.to_string(&term);
                        log::info!("extracted with cost {cost}: {}", extracted);
                        self.print_msg(extracted);
                        self.extract_report = Some(ExtractReport::Best {
                            termdag,
                            cost,
                            term,
                        });
                    } else {
                        if variants < 0 {
                            panic!("Cannot extract negative number of variants");
                        }
                        let terms =
                            self.extract_variants(values[0], variants as usize, &mut termdag);
                        log::info!("extracted variants:");
                        let mut msg = String::default();
                        msg += "(\n";
                        assert!(!terms.is_empty());
                        for expr in &terms {
                            let str = termdag.to_string(expr);
                            log::info!("   {}", str);
                            msg += &format!("   {}\n", str);
                        }
                        msg += ")";
                        self.print_msg(msg);
                        self.extract_report = Some(ExtractReport::Variants { termdag, terms });
                    }

                    stack.truncate(new_len);
                }
                Instruction::Panic(msg) => panic!("Panic: {}", msg),
                Instruction::Literal(lit) => match lit {
                    Literal::Int(i) => stack.push(Value::from(*i)),
                    Literal::F64(f) => stack.push(Value::from(*f)),
                    Literal::String(s) => stack.push(Value::from(*s)),
                    Literal::Bool(b) => stack.push(Value::from(*b)),
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
