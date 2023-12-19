/// This file implements the core IR of the language, which is called CoreRule.
/// CoreRule uses a conjunctive query-like IR for the body (queries) and a
/// SSA-like IR for the head (actions) based on the previous NormAction form.
/// Every construct has two forms: a standard (unresolved) form and a resolved form,
/// which differs in whether the head is a symbol or a resolved call.
/// Currently, CoreRule has several usages:
///   Typechecking is done over CoreRule format
///   Canonicalization is done over CoreRule format
///   ActionCompilers further compiles core actions to programs in a small VM
///   GJ compiler further compiler core queries to gj's CompiledQueries
///
/// Most compiler-time optimizations are expected to be done over CoreRule format.
use std::ops::AddAssign;

use crate::HashMap;
use crate::{typechecking::FuncType, *};
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoreAction<Head, Leaf> {
    Let(Leaf, Head, Vec<GenericAtomTerm<Leaf>>),
    LetAtomTerm(Leaf, GenericAtomTerm<Leaf>),
    Extract(GenericAtomTerm<Leaf>, GenericAtomTerm<Leaf>),
    Set(Head, Vec<GenericAtomTerm<Leaf>>, GenericAtomTerm<Leaf>),
    Delete(Head, Vec<GenericAtomTerm<Leaf>>),
    Union(GenericAtomTerm<Leaf>, GenericAtomTerm<Leaf>),
    Panic(String),
}

pub type NormAction = CoreAction<Symbol, Symbol>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoreActions<Head, Leaf>(pub(crate) Vec<CoreAction<Head, Leaf>>);
pub(crate) type NormActions = CoreActions<Symbol, Symbol>;
pub(crate) type ResolvedCoreActions = CoreActions<ResolvedCall, ResolvedVar>;

impl<Head, Leaf> Default for CoreActions<Head, Leaf> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<Head, Leaf> CoreActions<Head, Leaf>
where
    Leaf: Clone,
{
    pub(crate) fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf>>) {
        let actions = subst
            .iter()
            .map(|(symbol, atom_term)| CoreAction::LetAtomTerm(symbol.clone(), atom_term.clone()));
        let existing_actions = std::mem::take(&mut self.0);
        self.0 = actions.chain(existing_actions).collect();
    }

    fn new(actions: Vec<CoreAction<Head, Leaf>>) -> CoreActions<Head, Leaf> {
        Self(actions)
    }
}

#[allow(clippy::type_complexity)]
impl<Head, Leaf> GenericActions<Head, Leaf, ()>
where
    Head: Clone,
    Leaf: Clone + Hash + Eq + Clone,
{
    pub(crate) fn to_core_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut IndexSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<
        (
            CoreActions<Head, Leaf>,
            GenericActions<(Head, Leaf), Leaf, ()>,
        ),
        TypeError,
    >
    where
        Leaf: SymbolLike,
    {
        let mut norm_actions = vec![];
        let mut mapped_actions = vec![];

        // During the lowering, there are two important guaratees:
        //   Every used variable should be bound.
        //   Every introduced variable should be unbound before.
        for action in self.0.iter() {
            match action {
                GenericAction::Let(_ann, var, expr) => {
                    if binding.contains(var) {
                        return Err(TypeError::AlreadyDefined(var.to_symbol()));
                    }
                    let (actions, mapped_expr) =
                        expr.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    norm_actions.push(CoreAction::LetAtomTerm(
                        var.clone(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.push(GenericAction::Let((), var.clone(), mapped_expr));
                    binding.insert(var.clone());
                }
                GenericAction::Set(_ann, head, args, expr) => {
                    let mut mapped_args = vec![];
                    for arg in args {
                        let (actions, mapped_arg) =
                            arg.to_core_actions(typeinfo, binding, fresh_gen)?;
                        norm_actions.extend(actions.0);
                        mapped_args.push(mapped_arg);
                    }
                    let (actions, mapped_expr) =
                        expr.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    norm_actions.push(CoreAction::Set(
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    let v = fresh_gen.fresh(head);
                    mapped_actions.push(GenericAction::Set(
                        (),
                        (head.clone(), v),
                        mapped_args,
                        mapped_expr,
                    ));
                }
                GenericAction::Delete(_ann, head, args) => {
                    let mut mapped_args = vec![];
                    for arg in args {
                        let (actions, mapped_arg) =
                            arg.to_core_actions(typeinfo, binding, fresh_gen)?;
                        norm_actions.extend(actions.0);
                        mapped_args.push(mapped_arg);
                    }
                    norm_actions.push(CoreAction::Delete(
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                    ));
                    let v = fresh_gen.fresh(head);
                    mapped_actions.push(GenericAction::Delete((), (head.clone(), v), mapped_args));
                }
                GenericAction::Union(_ann, e1, e2) => {
                    let (actions1, mapped_e1) = e1.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions1.0);
                    let (actions2, mapped_e2) = e2.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions2.0);
                    norm_actions.push(CoreAction::Union(
                        mapped_e1.get_corresponding_var_or_lit(typeinfo),
                        mapped_e2.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.push(GenericAction::Union((), mapped_e1, mapped_e2));
                }
                GenericAction::Extract(_ann, e, n) => {
                    let (actions, mapped_e) = e.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    let (actions, mapped_n) = n.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    norm_actions.push(CoreAction::Extract(
                        mapped_e.get_corresponding_var_or_lit(typeinfo),
                        mapped_n.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.push(GenericAction::Extract((), mapped_e, mapped_n));
                }
                GenericAction::Panic(_ann, string) => {
                    norm_actions.push(CoreAction::Panic(string.clone()));
                    mapped_actions.push(GenericAction::Panic((), string.clone()));
                }
                GenericAction::Expr(_ann, expr) => {
                    let (actions, mapped_expr) =
                        expr.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    mapped_actions.push(GenericAction::Expr((), mapped_expr));
                }
            }
        }
        Ok((
            CoreActions::new(norm_actions),
            GenericActions::new(mapped_actions),
        ))
    }
}

#[allow(clippy::type_complexity)]
impl<Head: Clone, Leaf: Clone, Ann: Clone> GenericExpr<Head, Leaf, Ann> {
    pub(crate) fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> (
        Vec<GenericAtom<HeadOrEq<Head>, Leaf>>,
        GenericExpr<(Head, Leaf), Leaf, Ann>,
    )
    where
        Leaf: SymbolLike,
    {
        match self {
            GenericExpr::Lit(ann, lit) => (vec![], GenericExpr::Lit(ann.clone(), lit.clone())),
            GenericExpr::Var(ann, v) => (vec![], GenericExpr::Var(ann.clone(), v.clone())),
            GenericExpr::Call(ann, f, children) => {
                let fresh = fresh_gen.fresh(f);
                let mut new_children = vec![];
                let mut atoms = vec![];
                let mut child_exprs = vec![];
                for child in children {
                    let (child_atoms, child_expr) = child.to_query(typeinfo, fresh_gen);
                    let child_atomterm = child_expr.get_corresponding_var_or_lit(typeinfo);
                    new_children.push(child_atomterm);
                    atoms.extend(child_atoms);
                    child_exprs.push(child_expr);
                }
                let args = {
                    new_children.push(GenericAtomTerm::Var(fresh.clone()));
                    new_children
                };
                atoms.push(GenericAtom {
                    head: HeadOrEq::Symbol(f.clone()),
                    args,
                });
                (
                    atoms,
                    GenericExpr::Call(ann.clone(), (f.clone(), fresh), child_exprs),
                )
            }
        }
    }

    pub(crate) fn to_core_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut IndexSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<(CoreActions<Head, Leaf>, GenericExpr<(Head, Leaf), Leaf, ()>), TypeError>
    where
        Leaf: Hash + Eq + SymbolLike,
    {
        match self {
            GenericExpr::Lit(_ann, lit) => {
                Ok((CoreActions::default(), GenericExpr::Lit((), lit.clone())))
            }
            GenericExpr::Var(_ann, v) => {
                let sym = v.to_symbol();
                if binding.contains(v) || typeinfo.is_global(sym) {
                    Ok((CoreActions::default(), GenericExpr::Var((), v.clone())))
                } else {
                    Err(TypeError::Unbound(sym))
                }
            }
            GenericExpr::Call(_ann, f, args) => {
                let mut norm_actions = vec![];
                let mut norm_args = vec![];
                let mut mapped_args = vec![];
                for arg in args {
                    let (actions, mapped_arg) =
                        arg.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    norm_args.push(mapped_arg.get_corresponding_var_or_lit(typeinfo));
                    mapped_args.push(mapped_arg);
                }

                let var = fresh_gen.fresh(f);
                binding.insert(var.clone());

                norm_actions.push(CoreAction::Let(var.clone(), f.clone(), norm_args));
                Ok((
                    CoreActions::new(norm_actions),
                    GenericExpr::Call((), (f.clone(), var), mapped_args),
                ))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenericCoreRule<BodyF, HeadF, Leaf> {
    pub body: Query<BodyF, Leaf>,
    pub head: CoreActions<HeadF, Leaf>,
}

pub(crate) type CoreRule = GenericCoreRule<SymbolOrEq, Symbol, Symbol>;
pub(crate) type ResolvedCoreRule = GenericCoreRule<ResolvedCall, ResolvedCall, ResolvedVar>;

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
            head.to_core_actions(typeinfo, &mut binding, &mut fresh_gen)?;
        Ok(GenericCoreRule { body, head })
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
