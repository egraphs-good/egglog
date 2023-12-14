use std::ops::AddAssign;

use crate::HashMap;
use crate::{constraint::AllEqualTypeConstraint, typechecking::FuncType, *};
use typechecking::TypeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HeadOrEq<Head> {
    Symbol(Head),
    Eq,
}

pub(crate) type SymbolOrEq = HeadOrEq<Symbol>;

impl From<Symbol> for SymbolOrEq {
    fn from(value: Symbol) -> Self {
        SymbolOrEq::Symbol(value)
    }
}

impl<Head> HeadOrEq<Head> {
    pub fn is_eq(&self) -> bool {
        matches!(self, HeadOrEq::Eq)
    }
}

#[derive(Debug, Clone)]
pub struct GenericCoreRule<BodyF, HeadF, Leaf> {
    pub body: Query<BodyF, Leaf>,
    pub head: CoreActions<HeadF, Leaf>,
}

pub(crate) type CoreRule = GenericCoreRule<SymbolOrEq, Symbol, Symbol>;
pub(crate) type ResolvedCoreRule = GenericCoreRule<ResolvedCall, ResolvedCall, ResolvedVar>;

#[derive(Debug, Clone)]
pub(crate) struct SpecializedPrimitive {
    pub(crate) primitive: Primitive,
    pub(crate) input: Vec<ArcSort>,
    pub(crate) output: ArcSort,
}

#[derive(Debug, Clone)]
pub(crate) enum ResolvedCall {
    Func(FuncType),
    Primitive(SpecializedPrimitive),
}

impl SymbolLike for ResolvedCall {
    fn to_symbol(&self) -> Symbol {
        match self {
            ResolvedCall::Func(f) => f.name,
            ResolvedCall::Primitive(prim) => prim.primitive.0.name(),
        }
    }
}

impl ResolvedCall {
    pub fn output(&self) -> &ArcSort {
        match self {
            ResolvedCall::Func(func) => &func.output,
            ResolvedCall::Primitive(prim) => &prim.output,
        }
    }

    // Different from `from_resolution`, this function only considers function types and not primitives.
    // As a result, it only requires input argument types, so types.len() == func.input.len(),
    // while for `from_resolution`, types.len() == func.input.len() + 1 to account for the output type
    pub fn from_resolution_func_types(
        head: &Symbol,
        types: &[ArcSort],
        typeinfo: &TypeInfo,
    ) -> Option<ResolvedCall> {
        if let Some(ty) = typeinfo.func_types.get(head) {
            // As long as input types match, a result is returned.
            let expected = ty.input.iter().map(|s| s.name());
            let actual = types.iter().map(|s| s.name());
            if expected.eq(actual) {
                return Some(ResolvedCall::Func(ty.clone()));
            }
        }
        None
    }

    pub fn from_resolution(head: &Symbol, types: &[ArcSort], typeinfo: &TypeInfo) -> ResolvedCall {
        let mut resolved_call = Vec::with_capacity(1);
        if let Some(ty) = typeinfo.func_types.get(head) {
            let expected = ty.input.iter().chain(once(&ty.output)).map(|s| s.name());
            let actual = types.iter().map(|s| s.name());
            if expected.eq(actual) {
                resolved_call.push(ResolvedCall::Func(ty.clone()));
            }
        }

        if let Some(primitives) = typeinfo.primitives.get(head) {
            for primitive in primitives {
                if primitive.accept(types) {
                    let (out, inp) = types.split_last().unwrap();
                    resolved_call.push(ResolvedCall::Primitive(SpecializedPrimitive {
                        primitive: primitive.clone(),
                        input: inp.to_vec(),
                        output: out.clone(),
                    }));
                }
            }
        }

        assert!(resolved_call.len() == 1);
        resolved_call.pop().unwrap()
    }
}

impl Display for ResolvedCall {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedCall::Func(func) => write!(f, "{}", func.name),
            ResolvedCall::Primitive(prim) => write!(f, "{}", prim.primitive.0.name()),
        }
    }
}

impl ToSexp for ResolvedCall {
    fn to_sexp(&self) -> Sexp {
        Sexp::Symbol(self.to_string())
    }
}

impl<Leaf: Clone> Query<ResolvedCall, Leaf> {
    pub fn filters(&self) -> impl Iterator<Item = GenericAtom<Primitive, Leaf>> + '_ {
        self.atoms.iter().filter_map(|atom| match &atom.head {
            ResolvedCall::Func(_) => None,
            ResolvedCall::Primitive(head) => Some(GenericAtom {
                head: head.primitive.clone(),
                args: atom.args.clone(),
            }),
        })
    }

    pub fn funcs(&self) -> impl Iterator<Item = GenericAtom<Symbol, Leaf>> + '_ {
        self.atoms.iter().filter_map(|atom| match &atom.head {
            ResolvedCall::Func(head) => Some(GenericAtom {
                head: head.name,
                args: atom.args.clone(),
            }),
            ResolvedCall::Primitive(_) => None,
        })
    }
}

impl<Head1, Head2, Leaf> GenericCoreRule<Head1, Head2, Leaf>
where
    Head1: Clone,
    Head2: Clone,
    Leaf: Clone + Eq + Hash,
{
    pub fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf>>) {
        for atom in &mut self.body.atoms {
            atom.subst(subst);
        }
        self.head.subst(subst);
    }
}

impl<Head, Leaf> GenericRule<Head, Leaf, ()>
where
    Leaf: Clone + Eq + Hash + Debug,
    Head: Clone,
{
    pub(crate) fn to_core_rule(
        &self,
        typeinfo: &TypeInfo,
        mut fresh_gen: impl FreshGen<Head, Leaf>,
    ) -> Result<GenericCoreRule<HeadOrEq<Head>, Head, Leaf>, TypeError>
    where
        Leaf: SymbolLike,
    {
        let GenericRule { head, body } = self;

        let (body, _correspondence) = Facts(body.clone()).to_query(typeinfo, &mut fresh_gen);
        let mut binding = body.get_vars();
        let (head, _correspondence) =
            Actions(head.clone()).to_norm_actions(typeinfo, &mut binding, &mut fresh_gen)?;
        Ok(GenericCoreRule {
            body,
            head: CoreActions(head),
        })
    }

    fn to_canonicalized_core_rule_impl(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: impl FreshGen<Head, Leaf>,
        value_eq: impl Fn(&GenericAtomTerm<Leaf>, &GenericAtomTerm<Leaf>) -> Head,
    ) -> Result<GenericCoreRule<Head, Head, Leaf>, TypeError>
    where
        Leaf: SymbolLike,
    {
        let rule = self.to_core_rule(typeinfo, fresh_gen)?;
        let rule = rule.canonicalize(value_eq);
        Ok(rule)
    }
}

impl ResolvedRule {
    pub(crate) fn to_canonicalized_core_rule(
        &self,
        typeinfo: &TypeInfo,
    ) -> Result<ResolvedCoreRule, TypeError> {
        let value_eq = &typeinfo.primitives.get(&Symbol::from("value-eq")).unwrap()[0];
        let unit = typeinfo.get_sort_nofail::<UnitSort>();
        self.to_canonicalized_core_rule_impl(typeinfo, ResolvedGen::new(), |at1, at2| {
            ResolvedCall::Primitive(SpecializedPrimitive {
                primitive: value_eq.clone(),
                input: vec![at1.output(typeinfo), at2.output(typeinfo)],
                output: unit.clone(),
            })
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericAtomTerm<Leaf> {
    Var(Leaf),
    Literal(Literal),
    Global(Leaf),
}

pub type AtomTerm = GenericAtomTerm<Symbol>;
pub type ResolvedAtomTerm = GenericAtomTerm<ResolvedVar>;

impl AtomTerm {
    pub fn to_expr(&self) -> Expr {
        match self {
            AtomTerm::Var(v) => Expr::Var((), *v),
            AtomTerm::Literal(l) => Expr::Lit((), l.clone()),
            AtomTerm::Global(v) => Expr::Var((), *v),
        }
    }
}

impl ResolvedAtomTerm {
    pub fn output(&self, typeinfo: &TypeInfo) -> ArcSort {
        match self {
            ResolvedAtomTerm::Var(v) => v.sort.clone(),
            ResolvedAtomTerm::Literal(l) => typeinfo.infer_literal(l),
            ResolvedAtomTerm::Global(v) => v.sort.clone(),
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
pub struct GenericAtom<Head, Leaf> {
    pub head: Head,
    pub args: Vec<GenericAtomTerm<Leaf>>,
}

pub type Atom<T> = GenericAtom<T, Symbol>;

impl<T: std::fmt::Display> std::fmt::Display for Atom<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {}) ", self.head, ListDisplay(&self.args, " "))
    }
}

#[derive(Debug, Clone)]
pub struct Query<Head, Leaf> {
    pub atoms: Vec<GenericAtom<Head, Leaf>>,
}

impl<Head, Leaf> Default for Query<Head, Leaf> {
    fn default() -> Self {
        Self {
            atoms: Default::default(),
        }
    }
}

impl Query<SymbolOrEq, Symbol> {
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

impl<Head, Leaf> Query<Head, Leaf>
where
    Leaf: Eq + Clone + Hash,
    Head: Clone,
{
    pub(crate) fn get_vars(&self) -> IndexSet<Leaf> {
        self.atoms
            .iter()
            .flat_map(|atom| atom.vars())
            .collect::<IndexSet<_>>()
    }
}

impl<Head, Leaf> AddAssign for Query<Head, Leaf> {
    fn add_assign(&mut self, rhs: Self) {
        self.atoms.extend(rhs.atoms);
    }
}

impl std::fmt::Display for Query<Symbol, Symbol> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for atom in &self.atoms {
            writeln!(f, "{atom}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Query<ResolvedCall, Symbol> {
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

impl<Head, Leaf> GenericAtom<Head, Leaf>
where
    Leaf: Clone + Eq + Hash,
    Head: Clone,
{
    pub fn vars(&self) -> impl Iterator<Item = Leaf> + '_ {
        self.args.iter().filter_map(|t| match t {
            GenericAtomTerm::Var(v) => Some(v.clone()),
            GenericAtomTerm::Literal(_) => None,
            GenericAtomTerm::Global(_) => None,
        })
    }

    fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf>>) {
        for arg in self.args.iter_mut() {
            match arg {
                GenericAtomTerm::Var(v) => {
                    if let Some(at) = subst.get(v) {
                        *arg = at.clone();
                    }
                }
                GenericAtomTerm::Literal(_) => (),
                GenericAtomTerm::Global(_) => (),
            }
        }
    }
}
impl Atom<Symbol> {
    pub(crate) fn to_expr(&self) -> Expr {
        let n = self.args.len();
        Expr::Call(
            (),
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

impl<Head, Leaf> GenericCoreRule<HeadOrEq<Head>, Head, Leaf>
where
    Leaf: Eq + Clone + Hash + Debug,
    Head: Clone,
{
    /// Transformed a UnresolvedCoreRule into a CanonicalizedCoreRule.
    /// In particular, it removes equality checks between variables and
    /// other arguments, and turns equality checks between non-variable arguments
    /// into a primitive equality check `value-eq`.
    pub(crate) fn canonicalize(
        self,
        // Users need to pass in a substitute for equality constraints.
        value_eq: impl Fn(&GenericAtomTerm<Leaf>, &GenericAtomTerm<Leaf>) -> Head,
    ) -> GenericCoreRule<Head, Head, Leaf> {
        let mut result_rule = self;
        loop {
            let mut to_subst = None;
            for atom in result_rule.body.atoms.iter() {
                if atom.head.is_eq() && atom.args[0] != atom.args[1] {
                    match &atom.args[..] {
                        [GenericAtomTerm::Var(x), y] | [y, GenericAtomTerm::Var(x)] => {
                            to_subst = Some((x, y));
                            break;
                        }
                        _ => (),
                    }
                }
            }
            if let Some((x, y)) = to_subst {
                let subst = HashMap::from_iter(vec![(x.clone(), y.clone())]);
                result_rule.subst(&subst);
            } else {
                break;
            }
        }

        let atoms = result_rule
            .body
            .atoms
            .into_iter()
            .filter_map(|atom| match atom.head {
                HeadOrEq::Eq => {
                    assert_eq!(atom.args.len(), 2);
                    match (&atom.args[0], &atom.args[1]) {
                        (GenericAtomTerm::Var(v1), GenericAtomTerm::Var(v2)) => {
                            assert_eq!(v1, v2);
                            None
                        }
                        (GenericAtomTerm::Var(_), _) | (_, GenericAtomTerm::Var(_)) => {
                            panic!("equalities between variable and non-variable arguments should have been canonicalized")
                        }
                        (at1, at2) => {
                            if at1 == at2 {
                                None
                            } else {
                                Some(GenericAtom {
                                    head: value_eq(&atom.args[0], &atom.args[1]),
                                    args: vec![
                                        atom.args[0].clone(),
                                        atom.args[1].clone(),
                                        GenericAtomTerm::Literal(Literal::Unit),
                                    ],
                                })
                            }
                        },
                    }
                }
                HeadOrEq::Symbol(symbol) => Some(GenericAtom {
                    head: symbol,
                    args: atom.args,
                }),
            })
            .collect();
        GenericCoreRule {
            body: Query { atoms },
            head: result_rule.head,
        }
    }
}

struct ActionCompiler<'a> {
    egraph: &'a EGraph,
    types: &'a IndexMap<Symbol, ArcSort>,
    locals: IndexSet<ResolvedVar>,
    instructions: Vec<Instruction>,
}

impl<'a> ActionCompiler<'a> {
    fn check_action(&mut self, action: &CoreAction<ResolvedCall, ResolvedVar>) {
        match action {
            CoreAction::Let(v, f, args) => {
                self.do_call(f, args);
                self.locals.insert(v.clone());
            }
            CoreAction::LetAtomTerm(v, at) => {
                self.do_atom_term(at);
                self.locals.insert(v.clone());
            }
            CoreAction::Extract(e, b) => {
                self.do_atom_term(e);
                self.do_atom_term(b);
                self.instructions.push(Instruction::Extract(2));
            }
            CoreAction::Set(f, args, e) => {
                let ResolvedCall::Func(func) = f else {
                    panic!("Cannot set primitive- should have been caught by typechecking!!!")
                };
                for arg in args {
                    self.do_atom_term(arg);
                }
                self.do_atom_term(e);
                self.instructions.push(Instruction::Set(func.name));
            }
            CoreAction::Delete(f, args) => {
                let ResolvedCall::Func(func) = f else {
                    panic!("Cannot delete primitive- should have been caught by typechecking!!!")
                };
                for arg in args {
                    self.do_atom_term(arg);
                }
                self.instructions.push(Instruction::DeleteRow(func.name));
            }
            CoreAction::Union(arg1, arg2) => {
                self.do_atom_term(arg1);
                self.do_atom_term(arg2);
                self.instructions.push(Instruction::Union(2));
            }
            CoreAction::Panic(msg) => {
                self.instructions.push(Instruction::Panic(msg.clone()));
            }
        }
    }

    fn egraph(&self) -> &'a EGraph {
        self.egraph
    }

    fn do_call(&mut self, f: &ResolvedCall, args: &[ResolvedAtomTerm]) {
        for arg in args {
            self.do_atom_term(arg);
        }
        match f {
            ResolvedCall::Func(f) => self.do_function(f),
            ResolvedCall::Primitive(p) => self.do_prim(p),
        }
    }

    fn do_atom_term(&mut self, at: &ResolvedAtomTerm) {
        match at {
            ResolvedAtomTerm::Var(var) => {
                if let Some((i, _ty)) = self.locals.get_full(var) {
                    self.instructions.push(Instruction::Load(Load::Stack(i)));
                } else {
                    let (i, _, _ty) = self.types.get_full(&var.name).unwrap();
                    self.instructions.push(Instruction::Load(Load::Subst(i)));
                }
            }
            ResolvedAtomTerm::Literal(lit) => {
                self.instructions.push(Instruction::Literal(lit.clone()));
            }
            ResolvedAtomTerm::Global(var) => {
                assert!(self.egraph().global_bindings.contains_key(&var.name));
                self.instructions.push(Instruction::Global(var.name));
            }
        }
    }

    fn do_function(&mut self, func_type: &FuncType) {
        self.instructions.push(Instruction::CallFunction(
            func_type.name,
            func_type.has_default || func_type.is_datatype,
        ));
    }

    fn do_prim(&mut self, prim: &SpecializedPrimitive) {
        self.instructions.push(Instruction::CallPrimitive(
            prim.primitive.clone(),
            prim.input.len(),
        ));
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
}

#[derive(Clone, Debug)]
pub struct Program(Vec<Instruction>);

impl EGraph {
    /// Takes a list of variables bound to `subst` (variables bound during matching),
    /// whose positions are captured by indices of the IndexSet, and a list of core actions.
    /// Returns a program compiled from core actions and a list of variables bound to `stack`
    /// (whose positions are described by IndexSet indices as well).
    pub(crate) fn compile_actions(
        &self,
        binding: &IndexSet<ResolvedVar>,
        actions: &[CoreAction<ResolvedCall, ResolvedVar>],
    ) -> Result<Program, Vec<TypeError>> {
        let mut types = IndexMap::default();
        for var in binding {
            types.insert(var.name, var.sort.clone());
        }
        let mut checker = ActionCompiler {
            egraph: self,
            types: &types,
            locals: IndexSet::default(),
            instructions: Vec::new(),
        };

        for a in actions {
            checker.check_action(a);
        }

        Ok(Program(checker.instructions))
    }

    // This is the ugly part of the code. CoreActions lowered from
    // expressions like `2` is an empty vector, because no action is taken.
    // So to explicitly obtain the return value of an expression, compile_expr
    // needs to also take a `target`.`
    pub(crate) fn compile_expr(
        &self,
        binding: &IndexSet<ResolvedVar>,
        actions: &[CoreAction<ResolvedCall, ResolvedVar>],
        target: &GenericAtomTerm<ResolvedVar>,
    ) -> Result<Program, Vec<TypeError>> {
        let mut types = IndexMap::default();
        for var in binding {
            types.insert(var.name, var.sort.clone());
        }
        let mut checker = ActionCompiler {
            egraph: self,
            types: &types,
            locals: IndexSet::default(),
            instructions: Vec::new(),
        };

        for a in actions {
            checker.check_action(a);
        }
        checker.do_atom_term(target);

        Ok(Program(checker.instructions))
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
                        let mut stack = vec![];
                        self.run_actions(&mut stack, &values, &merge_prog, true)?;
                        stack.pop().unwrap()
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
                    // We need to pass a new stack instead of reusing the old one
                    // because Load(Stack(idx)) use absolute index.
                    self.run_actions(&mut Vec::new(), &values, &prog, true)?;
                }
            }
        } else {
            function.insert(args, new_value, self.timestamp);
        }
        Ok(())
    }

    pub(crate) fn run_actions(
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
                                let default = default.clone();
                                let value = self.eval_resolved_expr(&default, true)?;
                                self.functions.get_mut(f).unwrap().insert(values, value, ts);
                                value
                            }
                            _ => {
                                return Err(Error::NotFoundError(NotFoundError(Expr::Var(
                                    (),
                                    format!("No value found for {f} {:?}", values).into(),
                                ))))
                            }
                        }
                    } else {
                        return Err(Error::NotFoundError(NotFoundError(Expr::Var(
                            (),
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
