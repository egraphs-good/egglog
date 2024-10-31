//! This file implements the core IR of the language, which is called CoreRule.
//! CoreRule uses a conjunctive query-like IR for the body (queries) and a
//! SSA-like IR for the head (actions) based on the previous CoreAction form.
//! Every construct has two forms: a standard (unresolved) form and a resolved form,
//! which differs in whether the head is a symbol or a resolved call.
//! Currently, CoreRule has several usages:
//!   Typechecking is done over CoreRule format
//!   Canonicalization is done over CoreRule format
//!   ActionCompilers further compiles core actions to programs in a small VM
//!   GJ compiler further compiler core queries to gj's CompiledQueries
//!
//! Most compiler-time optimizations are expected to be done over CoreRule format.
use std::hash::Hasher;
use std::ops::AddAssign;

use crate::{typechecking::FuncType, HashMap, *};
use typechecking::TypeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HeadOrEq<Head> {
    String(Head),
    Eq,
}

pub(crate) type StringOrEq = HeadOrEq<String>;

impl From<String> for StringOrEq {
    fn from(value: String) -> Self {
        StringOrEq::String(value)
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
        head: &String,
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

    pub fn from_resolution(head: &String, types: &[ArcSort], typeinfo: &TypeInfo) -> ResolvedCall {
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
                if primitive.accept(types, typeinfo) {
                    let (out, inp) = types.split_last().unwrap();
                    resolved_call.push(ResolvedCall::Primitive(SpecializedPrimitive {
                        primitive: primitive.clone(),
                        input: inp.to_vec(),
                        output: out.clone(),
                    }));
                }
            }
        }
        assert!(
            resolved_call.len() == 1,
            "Ambiguous resolution for {:?}",
            head,
        );
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
        Sexp::String(self.to_string())
    }
}

pub(crate) trait CreateUnit {
    fn unit() -> Self;
}

impl CreateUnit for Literal {
    fn unit() -> Self {
        Literal::Unit
    }
}

impl CreateUnit for ResolvedLiteral {
    fn unit() -> Self {
        ResolvedLiteral {
            literal: Literal::Unit,
            sort: Arc::new(UnitSort),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GenericAtomTerm<Leaf, Lit> {
    Var(Span, Leaf),
    Literal(Span, Lit),
    Global(Span, Leaf),
}

// Ignores annotations for equality and hasing
impl<Leaf, Lit> PartialEq for GenericAtomTerm<Leaf, Lit>
where
    Leaf: PartialEq,
    Lit: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GenericAtomTerm::Var(_, v1), GenericAtomTerm::Var(_, v2)) => v1 == v2,
            (GenericAtomTerm::Literal(_, l1), GenericAtomTerm::Literal(_, l2)) => l1 == l2,
            (GenericAtomTerm::Global(_, g1), GenericAtomTerm::Global(_, g2)) => g1 == g2,
            _ => false,
        }
    }
}

impl<Leaf, Lit> Eq for GenericAtomTerm<Leaf, Lit>
where
    Leaf: Eq,
    Lit: Eq,
{
}

impl<Leaf, Lit> Hash for GenericAtomTerm<Leaf, Lit>
where
    Leaf: Hash,
    Lit: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            GenericAtomTerm::Var(_, v) => v.hash(state),
            GenericAtomTerm::Literal(_, l) => l.hash(state),
            GenericAtomTerm::Global(_, g) => g.hash(state),
        }
    }
}

pub type AtomTerm = GenericAtomTerm<String, Literal>;
pub type ResolvedAtomTerm = GenericAtomTerm<ResolvedVar, ResolvedLiteral>;

impl<Leaf, Lit> GenericAtomTerm<Leaf, Lit> {
    pub fn span(&self) -> &Span {
        match self {
            GenericAtomTerm::Var(span, _) => span,
            GenericAtomTerm::Literal(span, _) => span,
            GenericAtomTerm::Global(span, _) => span,
        }
    }
}

impl<Leaf: Clone, Lit: Clone> GenericAtomTerm<Leaf, Lit> {
    pub fn to_expr<Head>(&self) -> GenericExpr<Head, Leaf, Lit> {
        match self {
            GenericAtomTerm::Var(span, v) => GenericExpr::Var(span.clone(), v.clone()),
            GenericAtomTerm::Literal(span, l) => GenericExpr::Lit(span.clone(), l.clone()),
            GenericAtomTerm::Global(span, v) => GenericExpr::Var(span.clone(), v.clone()),
        }
    }
}

impl ResolvedAtomTerm {
    pub fn output(&self) -> ArcSort {
        match self {
            ResolvedAtomTerm::Var(_, v) => v.sort.clone(),
            ResolvedAtomTerm::Literal(_, l) => l.sort.clone(),
            ResolvedAtomTerm::Global(_, v) => v.sort.clone(),
        }
    }
}

impl std::fmt::Display for AtomTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomTerm::Var(_, v) => write!(f, "{v}"),
            AtomTerm::Literal(_, lit) => write!(f, "{lit}"),
            AtomTerm::Global(_, g) => write!(f, "{g}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericAtom<Head, Leaf, Lit> {
    pub span: Span,
    pub head: Head,
    pub args: Vec<GenericAtomTerm<Leaf, Lit>>,
}

pub type Atom<T> = GenericAtom<T, String, Literal>;

impl<T: std::fmt::Display> std::fmt::Display for Atom<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {}) ", self.head, ListDisplay(&self.args, " "))
    }
}

impl<Head, Leaf, Lit> GenericAtom<Head, Leaf, Lit>
where
    Leaf: Clone + Eq + Hash,
    Head: Clone,
    Lit: Clone,
{
    pub fn vars(&self) -> impl Iterator<Item = Leaf> + '_ {
        self.args.iter().filter_map(|t| match t {
            GenericAtomTerm::Var(_, v) => Some(v.clone()),
            GenericAtomTerm::Literal(..) => None,
            GenericAtomTerm::Global(..) => None,
        })
    }

    fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf, Lit>>) {
        for arg in self.args.iter_mut() {
            match arg {
                GenericAtomTerm::Var(_, v) => {
                    if let Some(at) = subst.get(v) {
                        *arg = at.clone();
                    }
                }
                GenericAtomTerm::Literal(..) => (),
                GenericAtomTerm::Global(..) => (),
            }
        }
    }
}
impl Atom<String> {
    pub(crate) fn to_expr(&self) -> Expr {
        let n = self.args.len();
        Expr::Call(
            self.span.clone(),
            self.head.clone(),
            self.args[0..n - 1]
                .iter()
                .map(|arg| arg.to_expr())
                .collect(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct Query<Head, Leaf, Lit> {
    pub atoms: Vec<GenericAtom<Head, Leaf, Lit>>,
}

impl<Head, Leaf, Lit> Default for Query<Head, Leaf, Lit> {
    fn default() -> Self {
        Self {
            atoms: Default::default(),
        }
    }
}

impl Query<StringOrEq, String, Literal> {
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

impl<Head, Leaf, Lit> Query<Head, Leaf, Lit>
where
    Leaf: Eq + Clone + Hash,
    Head: Clone,
    Lit: Clone,
{
    pub(crate) fn get_vars(&self) -> IndexSet<Leaf> {
        self.atoms
            .iter()
            .flat_map(|atom| atom.vars())
            .collect::<IndexSet<_>>()
    }
}

impl<Head, Leaf, Lit> AddAssign for Query<Head, Leaf, Lit> {
    fn add_assign(&mut self, rhs: Self) {
        self.atoms.extend(rhs.atoms);
    }
}

impl std::fmt::Display for Query<String, String, Literal> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for atom in &self.atoms {
            writeln!(f, "{atom}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Query<ResolvedCall, String, Literal> {
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
                    filter.head.primitive.name(),
                    ListDisplay(&filter.args, " ")
                )?;
            }
        }
        Ok(())
    }
}

impl<Leaf: Clone, Lit: Clone> Query<ResolvedCall, Leaf, Lit> {
    pub fn filters(
        &self,
    ) -> impl Iterator<Item = GenericAtom<SpecializedPrimitive, Leaf, Lit>> + '_ {
        self.atoms.iter().filter_map(|atom| match &atom.head {
            ResolvedCall::Func(_) => None,
            ResolvedCall::Primitive(head) => Some(GenericAtom {
                span: atom.span.clone(),
                head: head.clone(),
                args: atom.args.clone(),
            }),
        })
    }

    pub fn funcs(&self) -> impl Iterator<Item = GenericAtom<String, Leaf, Lit>> + '_ {
        self.atoms.iter().filter_map(|atom| match &atom.head {
            ResolvedCall::Func(head) => Some(GenericAtom {
                span: atom.span.clone(),
                head: head.name.clone(),
                args: atom.args.clone(),
            }),
            ResolvedCall::Primitive(_) => None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericCoreAction<Head, Leaf, Lit> {
    Let(Span, Leaf, Head, Vec<GenericAtomTerm<Leaf, Lit>>),
    LetAtomTerm(Span, Leaf, GenericAtomTerm<Leaf, Lit>),
    Extract(Span, GenericAtomTerm<Leaf, Lit>, GenericAtomTerm<Leaf, Lit>),
    Set(
        Span,
        Head,
        Vec<GenericAtomTerm<Leaf, Lit>>,
        GenericAtomTerm<Leaf, Lit>,
    ),
    Change(Span, Change, Head, Vec<GenericAtomTerm<Leaf, Lit>>),
    Union(Span, GenericAtomTerm<Leaf, Lit>, GenericAtomTerm<Leaf, Lit>),
    Panic(Span, String),
}

pub type CoreAction = GenericCoreAction<String, String, Literal>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericCoreActions<Head, Leaf, Lit>(pub(crate) Vec<GenericCoreAction<Head, Leaf, Lit>>);
pub(crate) type ResolvedCoreActions =
    GenericCoreActions<ResolvedCall, ResolvedVar, ResolvedLiteral>;

impl<Head, Leaf, Lit> Default for GenericCoreActions<Head, Leaf, Lit> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<Head, Leaf, Lit> GenericCoreActions<Head, Leaf, Lit>
where
    Leaf: Clone,
    Lit: Clone,
{
    pub(crate) fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf, Lit>>) {
        let actions = subst.iter().map(|(symbol, atom_term)| {
            GenericCoreAction::LetAtomTerm(
                atom_term.span().clone(),
                symbol.clone(),
                atom_term.clone(),
            )
        });
        let existing_actions = std::mem::take(&mut self.0);
        self.0 = actions.chain(existing_actions).collect();
    }

    fn new(
        actions: Vec<GenericCoreAction<Head, Leaf, Lit>>,
    ) -> GenericCoreActions<Head, Leaf, Lit> {
        Self(actions)
    }
}

#[allow(clippy::type_complexity)]
impl<Head, Leaf, Lit> GenericActions<Head, Leaf, Lit>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
    Lit: Clone + Display,
{
    pub(crate) fn to_core_actions<FG: FreshGen<Head, Leaf, Lit>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut IndexSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<
        (
            GenericCoreActions<Head, Leaf, Lit>,
            MappedActions<Head, Leaf, Lit>,
        ),
        TypeError,
    >
    where
        Leaf: ToString,
    {
        let mut norm_actions = vec![];
        let mut mapped_actions: MappedActions<Head, Leaf, Lit> = GenericActions(vec![]);

        // During the lowering, there are two important guaratees:
        //   Every used variable should be bound.
        //   Every introduced variable should be unbound before.
        for action in self.0.iter() {
            match action {
                GenericAction::Let(span, var, expr) => {
                    if binding.contains(var) {
                        return Err(TypeError::AlreadyDefined(var.to_string(), span.clone()));
                    }
                    let (actions, mapped_expr) =
                        expr.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    norm_actions.push(GenericCoreAction::LetAtomTerm(
                        span.clone(),
                        var.clone(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions.0.push(GenericAction::Let(
                        span.clone(),
                        var.clone(),
                        mapped_expr,
                    ));
                    binding.insert(var.clone());
                }
                GenericAction::Set(span, head, args, expr) => {
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
                    norm_actions.push(GenericCoreAction::Set(
                        span.clone(),
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    let v = fresh_gen.fresh(head);
                    mapped_actions.0.push(GenericAction::Set(
                        span.clone(),
                        CorrespondingVar::new(head.clone(), v),
                        mapped_args,
                        mapped_expr,
                    ));
                }
                GenericAction::Change(span, change, head, args) => {
                    let mut mapped_args = vec![];
                    for arg in args {
                        let (actions, mapped_arg) =
                            arg.to_core_actions(typeinfo, binding, fresh_gen)?;
                        norm_actions.extend(actions.0);
                        mapped_args.push(mapped_arg);
                    }
                    norm_actions.push(GenericCoreAction::Change(
                        span.clone(),
                        *change,
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                    ));
                    let v = fresh_gen.fresh(head);
                    mapped_actions.0.push(GenericAction::Change(
                        span.clone(),
                        *change,
                        CorrespondingVar::new(head.clone(), v),
                        mapped_args,
                    ));
                }
                GenericAction::Union(span, e1, e2) => {
                    let (actions1, mapped_e1) = e1.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions1.0);
                    let (actions2, mapped_e2) = e2.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions2.0);
                    norm_actions.push(GenericCoreAction::Union(
                        span.clone(),
                        mapped_e1.get_corresponding_var_or_lit(typeinfo),
                        mapped_e2.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions
                        .0
                        .push(GenericAction::Union(span.clone(), mapped_e1, mapped_e2));
                }
                GenericAction::Extract(span, e, n) => {
                    let (actions, mapped_e) = e.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    let (actions, mapped_n) = n.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    norm_actions.push(GenericCoreAction::Extract(
                        span.clone(),
                        mapped_e.get_corresponding_var_or_lit(typeinfo),
                        mapped_n.get_corresponding_var_or_lit(typeinfo),
                    ));
                    mapped_actions
                        .0
                        .push(GenericAction::Extract(span.clone(), mapped_e, mapped_n));
                }
                GenericAction::Panic(span, string) => {
                    norm_actions.push(GenericCoreAction::Panic(span.clone(), string.clone()));
                    mapped_actions
                        .0
                        .push(GenericAction::Panic(span.clone(), string.clone()));
                }
                GenericAction::Expr(span, expr) => {
                    let (actions, mapped_expr) =
                        expr.to_core_actions(typeinfo, binding, fresh_gen)?;
                    norm_actions.extend(actions.0);
                    mapped_actions
                        .0
                        .push(GenericAction::Expr(span.clone(), mapped_expr));
                }
            }
        }
        Ok((GenericCoreActions::new(norm_actions), mapped_actions))
    }
}

impl<Head, Leaf, Lit> GenericExpr<Head, Leaf, Lit>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
    Lit: Clone + Display,
{
    pub(crate) fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf, Lit>,
    ) -> (
        Vec<GenericAtom<HeadOrEq<Head>, Leaf, Lit>>,
        MappedExpr<Head, Leaf, Lit>,
    )
    where
        Leaf: ToString,
    {
        match self {
            GenericExpr::Lit(span, lit) => (vec![], GenericExpr::Lit(span.clone(), lit.clone())),
            GenericExpr::Var(span, v) => (vec![], GenericExpr::Var(span.clone(), v.clone())),
            GenericExpr::Call(span, f, children) => {
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
                    new_children.push(GenericAtomTerm::Var(span.clone(), fresh.clone()));
                    new_children
                };
                atoms.push(GenericAtom {
                    span: span.clone(),
                    head: HeadOrEq::String(f.clone()),
                    args,
                });
                (
                    atoms,
                    GenericExpr::Call(
                        span.clone(),
                        CorrespondingVar::new(f.clone(), fresh),
                        child_exprs,
                    ),
                )
            }
        }
    }

    pub(crate) fn to_core_actions<FG: FreshGen<Head, Leaf, Lit>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut IndexSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<
        (
            GenericCoreActions<Head, Leaf, Lit>,
            MappedExpr<Head, Leaf, Lit>,
        ),
        TypeError,
    >
    where
        Leaf: Hash + Eq + ToString,
    {
        match self {
            GenericExpr::Lit(span, lit) => Ok((
                GenericCoreActions::default(),
                GenericExpr::Lit(span.clone(), lit.clone()),
            )),
            GenericExpr::Var(span, v) => {
                let sym = v.to_string();
                if binding.contains(v) || typeinfo.is_global(sym.clone()) {
                    Ok((
                        GenericCoreActions::default(),
                        GenericExpr::Var(span.clone(), v.clone()),
                    ))
                } else {
                    Err(TypeError::Unbound(sym, span.clone()))
                }
            }
            GenericExpr::Call(span, f, args) => {
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

                norm_actions.push(GenericCoreAction::Let(
                    span.clone(),
                    var.clone(),
                    f.clone(),
                    norm_args,
                ));
                Ok((
                    GenericCoreActions::new(norm_actions),
                    GenericExpr::Call(
                        span.clone(),
                        CorrespondingVar::new(f.clone(), var),
                        mapped_args,
                    ),
                ))
            }
        }
    }
}

/// A [`GenericCoreRule`] represents a generalization of lowered form of a rule.
/// Unlike other `Generic`-prefixed types, [`GenericCoreRule`] takes two `Head`
/// parameters instead of one. This is because the `Head` parameter of `body` and
/// `head` can be different. In particular, early in the compilation pipeline,
/// `body` can contain `Eq` atoms, which denotes equality constraints, so the `Head`
/// for `body` needs to be a `HeadOrEq<Head>`, while `head` does not have equality
/// constraints.
#[derive(Debug, Clone)]
pub struct GenericCoreRule<HeadQ, HeadA, Leaf, Lit> {
    pub span: Span,
    pub body: Query<HeadQ, Leaf, Lit>,
    pub head: GenericCoreActions<HeadA, Leaf, Lit>,
}

pub(crate) type CoreRule = GenericCoreRule<StringOrEq, String, String, Literal>;
pub(crate) type ResolvedCoreRule =
    GenericCoreRule<ResolvedCall, ResolvedCall, ResolvedVar, ResolvedLiteral>;

impl<Head1, Head2, Leaf, Lit> GenericCoreRule<Head1, Head2, Leaf, Lit>
where
    Head1: Clone,
    Head2: Clone,
    Leaf: Clone + Eq + Hash,
    Lit: Clone,
{
    pub fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf, Lit>>) {
        for atom in &mut self.body.atoms {
            atom.subst(subst);
        }
        self.head.subst(subst);
    }
}

impl<Head, Leaf, Lit> GenericCoreRule<HeadOrEq<Head>, Head, Leaf, Lit>
where
    Leaf: Eq + Clone + Hash + Debug,
    Head: Clone,
    Lit: Clone + Eq + CreateUnit,
{
    /// Transformed a UnresolvedCoreRule into a CanonicalizedCoreRule.
    /// In particular, it removes equality checks between variables and
    /// other arguments, and turns equality checks between non-variable arguments
    /// into a primitive equality check `value-eq`.
    pub(crate) fn canonicalize(
        self,
        // Users need to pass in a substitute for equality constraints.
        value_eq: impl Fn(&GenericAtomTerm<Leaf, Lit>, &GenericAtomTerm<Leaf, Lit>) -> Head,
    ) -> GenericCoreRule<Head, Head, Leaf, Lit> {
        let mut result_rule = self;
        loop {
            let mut to_subst = None;
            for atom in result_rule.body.atoms.iter() {
                if atom.head.is_eq() && atom.args[0] != atom.args[1] {
                    match &atom.args[..] {
                        [GenericAtomTerm::Var(_, x), y] | [y, GenericAtomTerm::Var(_, x)] => {
                            to_subst = Some((x, y));
                            break;
                        }
                        _ => (),
                    }
                }
            }
            if let Some((x, y)) = to_subst {
                let subst = HashMap::from_iter([(x.clone(), y.clone())]);
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
                        (GenericAtomTerm::Var(_, v1), GenericAtomTerm::Var(_, v2)) => {
                            assert_eq!(v1, v2);
                            None
                        }
                        (GenericAtomTerm::Var(..), _) | (_, GenericAtomTerm::Var(..)) => {
                            panic!("equalities between variable and non-variable arguments should have been canonicalized")
                        }
                        (at1, at2) => {
                            if at1 == at2 {
                                None
                            } else {
                                Some(GenericAtom {
                                    span: atom.span.clone(),
                                    head: value_eq(&atom.args[0], &atom.args[1]),
                                    args: vec![
                                        atom.args[0].clone(),
                                        atom.args[1].clone(),
                                        GenericAtomTerm::Literal(atom.span.clone(), Lit::unit()),
                                    ],
                                })
                            }
                        },
                    }
                }
                HeadOrEq::String(symbol) => Some(GenericAtom {
                    span: atom.span.clone(),
                    head: symbol,
                    args: atom.args,
                }),
            })
            .collect();
        GenericCoreRule {
            span: result_rule.span,
            body: Query { atoms },
            head: result_rule.head,
        }
    }
}

impl<Head, Leaf, Lit> GenericRule<Head, Leaf, Lit>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash + Debug,
    Lit: Clone + Display + Eq,
{
    pub(crate) fn to_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf, Lit>,
    ) -> Result<GenericCoreRule<HeadOrEq<Head>, Head, Leaf, Lit>, TypeError>
    where
        Leaf: ToString,
    {
        let GenericRule {
            span: _,
            head,
            body,
        } = self;

        let (body, _correspondence) = Facts(body.clone()).to_query(typeinfo, fresh_gen);
        let mut binding = body.get_vars();
        let (head, _correspondence) = head.to_core_actions(typeinfo, &mut binding, fresh_gen)?;
        Ok(GenericCoreRule {
            span: self.span.clone(),
            body,
            head,
        })
    }

    fn to_canonicalized_core_rule_impl(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf, Lit>,
        value_eq: impl Fn(&GenericAtomTerm<Leaf, Lit>, &GenericAtomTerm<Leaf, Lit>) -> Head,
    ) -> Result<GenericCoreRule<Head, Head, Leaf, Lit>, TypeError>
    where
        Leaf: ToString,
        Lit: CreateUnit,
    {
        let rule = self.to_core_rule(typeinfo, fresh_gen)?;
        Ok(rule.canonicalize(value_eq))
    }
}

impl ResolvedRule {
    pub(crate) fn to_canonicalized_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut StringGen,
    ) -> Result<ResolvedCoreRule, TypeError> {
        let value_eq = &typeinfo.primitives.get(&String::from("value-eq")).unwrap()[0];
        self.to_canonicalized_core_rule_impl(typeinfo, fresh_gen, |at1, at2| {
            ResolvedCall::Primitive(SpecializedPrimitive {
                primitive: value_eq.clone(),
                input: vec![at1.output(), at2.output()],
                output: Arc::new(UnitSort),
            })
        })
    }
}
