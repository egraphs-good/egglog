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

use crate::*;
use typechecking::{FuncType, PrimitiveWithId, TypeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HeadOrEq<Head> {
    Head(Head),
    Eq,
}

pub(crate) type StringOrEq = HeadOrEq<String>;

impl From<String> for StringOrEq {
    fn from(value: String) -> Self {
        StringOrEq::Head(value)
    }
}

impl<Head> HeadOrEq<Head> {
    pub fn is_eq(&self) -> bool {
        matches!(self, HeadOrEq::Eq)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SpecializedPrimitive {
    pub(crate) primitive: PrimitiveWithId,
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
        head: &str,
        types: &[ArcSort],
        typeinfo: &TypeInfo,
    ) -> Option<ResolvedCall> {
        if let Some(ty) = typeinfo.get_func_type(head) {
            // As long as input types match, a result is returned.
            let expected = ty.input.iter().map(|s| s.name());
            let actual = types.iter().map(|s| s.name());
            if expected.eq(actual) {
                return Some(ResolvedCall::Func(ty.clone()));
            }
        }
        None
    }

    pub fn from_resolution(head: &str, types: &[ArcSort], typeinfo: &TypeInfo) -> ResolvedCall {
        let mut resolved_call = Vec::with_capacity(1);
        if let Some(ty) = typeinfo.get_func_type(head) {
            let expected = ty.input.iter().chain(once(&ty.output)).map(|s| s.name());
            let actual = types.iter().map(|s| s.name());
            if expected.eq(actual) {
                resolved_call.push(ResolvedCall::Func(ty.clone()));
            }
        }

        if let Some(primitives) = typeinfo.get_prims(head) {
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

/// A trait encapsulating the ability to query a [`TypeInfo`] to determine
/// whether or not a symbol is bound as a function in the current egglog program.
///
/// Currently, we only use this trait to determine whether a symbol is a
/// [`FunctionSubtype::Constructor`].
pub trait IsFunc {
    fn is_constructor(&self, type_info: &TypeInfo) -> bool;
}

impl IsFunc for ResolvedCall {
    fn is_constructor(&self, type_info: &TypeInfo) -> bool {
        match self {
            ResolvedCall::Func(func) => type_info.is_constructor(&func.name),
            ResolvedCall::Primitive(_) => false,
        }
    }
}

impl IsFunc for String {
    fn is_constructor(&self, type_info: &TypeInfo) -> bool {
        type_info.is_constructor(self)
    }
}

#[derive(Debug, Clone)]
pub enum GenericAtomTerm<Leaf> {
    Var(Span, Leaf),
    Literal(Span, Literal),
    Global(Span, Leaf),
}

// Ignores annotations for equality and hasing
impl<Leaf> PartialEq for GenericAtomTerm<Leaf>
where
    Leaf: PartialEq,
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

impl<Leaf> Eq for GenericAtomTerm<Leaf> where Leaf: Eq {}

impl<Leaf> Hash for GenericAtomTerm<Leaf>
where
    Leaf: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            GenericAtomTerm::Var(_, v) => v.hash(state),
            GenericAtomTerm::Literal(_, l) => l.hash(state),
            GenericAtomTerm::Global(_, g) => g.hash(state),
        }
    }
}

pub type AtomTerm = GenericAtomTerm<String>;
pub type ResolvedAtomTerm = GenericAtomTerm<ResolvedVar>;

impl<Leaf> GenericAtomTerm<Leaf> {
    pub fn span(&self) -> &Span {
        match self {
            GenericAtomTerm::Var(span, _) => span,
            GenericAtomTerm::Literal(span, _) => span,
            GenericAtomTerm::Global(span, _) => span,
        }
    }
}

impl<Leaf: Clone> GenericAtomTerm<Leaf> {
    pub fn to_expr<Head>(&self) -> GenericExpr<Head, Leaf> {
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
            ResolvedAtomTerm::Literal(_, l) => literal_sort(l),
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
pub struct GenericAtom<Head, Leaf> {
    pub span: Span,
    pub head: Head,
    pub args: Vec<GenericAtomTerm<Leaf>>,
}

pub type Atom<T> = GenericAtom<T, String>;

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
            GenericAtomTerm::Var(_, v) => Some(v.clone()),
            GenericAtomTerm::Literal(..) => None,
            GenericAtomTerm::Global(..) => None,
        })
    }

    fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf>>) {
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

impl Query<StringOrEq, String> {
    pub fn get_constraints(
        &self,
        type_info: &TypeInfo,
    ) -> Result<Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>, TypeError> {
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

impl std::fmt::Display for Query<String, String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for atom in &self.atoms {
            writeln!(f, "{atom}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Query<ResolvedCall, String> {
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
                    filter.head.primitive.0.name(),
                    ListDisplay(&filter.args, " ")
                )?;
            }
        }
        Ok(())
    }
}

impl<Leaf: Clone> Query<ResolvedCall, Leaf> {
    pub fn filters(&self) -> impl Iterator<Item = GenericAtom<SpecializedPrimitive, Leaf>> + '_ {
        self.atoms.iter().filter_map(|atom| match &atom.head {
            ResolvedCall::Func(_) => None,
            ResolvedCall::Primitive(head) => Some(GenericAtom {
                span: atom.span.clone(),
                head: head.clone(),
                args: atom.args.clone(),
            }),
        })
    }

    pub fn funcs(&self) -> impl Iterator<Item = GenericAtom<String, Leaf>> + '_ {
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
pub enum GenericCoreAction<Head, Leaf> {
    Let(Span, Leaf, Head, Vec<GenericAtomTerm<Leaf>>),
    LetAtomTerm(Span, Leaf, GenericAtomTerm<Leaf>),
    Set(
        Span,
        Head,
        Vec<GenericAtomTerm<Leaf>>,
        GenericAtomTerm<Leaf>,
    ),
    Change(Span, Change, Head, Vec<GenericAtomTerm<Leaf>>),
    Union(Span, GenericAtomTerm<Leaf>, GenericAtomTerm<Leaf>),
    Panic(Span, String),
}

pub type CoreAction = GenericCoreAction<String, String>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericCoreActions<Head, Leaf>(pub(crate) Vec<GenericCoreAction<Head, Leaf>>);
pub(crate) type ResolvedCoreActions = GenericCoreActions<ResolvedCall, ResolvedVar>;

impl<Head, Leaf> Default for GenericCoreActions<Head, Leaf> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<Head, Leaf> GenericCoreActions<Head, Leaf>
where
    Leaf: Clone + Eq + Hash,
{
    pub(crate) fn subst(&mut self, subst: &HashMap<Leaf, GenericAtomTerm<Leaf>>) {
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

    fn new(actions: Vec<GenericCoreAction<Head, Leaf>>) -> GenericCoreActions<Head, Leaf> {
        Self(actions)
    }

    pub(crate) fn get_free_vars(&self) -> HashSet<Leaf> {
        let at_free_var = |at: &GenericAtomTerm<Leaf>| match at {
            GenericAtomTerm::Var(_, v) => Some(v.clone()),
            GenericAtomTerm::Literal(..) => None,
            GenericAtomTerm::Global(..) => None,
        };

        let add_from_atom = |free_vars: &mut HashSet<Leaf>, at: &GenericAtomTerm<Leaf>| {
            if let Some(v) = at_free_var(at) {
                free_vars.insert(v);
            }
        };

        let add_from_atoms = |free_vars: &mut HashSet<Leaf>, ats: &[GenericAtomTerm<Leaf>]| {
            ats.iter().flat_map(&at_free_var).for_each(|v| {
                free_vars.insert(v);
            });
        };

        let mut free_vars = HashSet::default();
        for action in self.0.iter().rev() {
            match action {
                GenericCoreAction::Let(_span, v, _, ats) => {
                    add_from_atoms(&mut free_vars, ats);
                    free_vars.swap_remove(v);
                }
                GenericCoreAction::LetAtomTerm(_span, v, at) => {
                    add_from_atom(&mut free_vars, at);
                    free_vars.swap_remove(v);
                }
                GenericCoreAction::Set(_span, _, ats, at) => {
                    add_from_atoms(&mut free_vars, ats);
                    add_from_atom(&mut free_vars, at);
                }
                GenericCoreAction::Change(_span, _change, _, ats) => {
                    add_from_atoms(&mut free_vars, ats);
                }
                GenericCoreAction::Union(_span, at, at1) => {
                    add_from_atom(&mut free_vars, at);
                    add_from_atom(&mut free_vars, at1);
                }
                GenericCoreAction::Panic(_span, _) => {}
            }
        }
        free_vars
    }
}

#[allow(clippy::type_complexity)]
impl<Head, Leaf> GenericActions<Head, Leaf>
where
    Head: Clone + Display + IsFunc,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn to_core_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut IndexSet<Leaf>,
        fresh_gen: &mut FG,
    ) -> Result<(GenericCoreActions<Head, Leaf>, MappedActions<Head, Leaf>), TypeError> {
        let mut norm_actions = vec![];
        let mut mapped_actions: MappedActions<Head, Leaf> = GenericActions(vec![]);

        // During the lowering, there are two important guaratees:
        //   Every used variable should be bound.
        //   Every introduced variable should be unbound before.
        for action in self.0.iter() {
            match action {
                GenericAction::Let(span, var, expr) => {
                    if binding.contains(var) {
                        return Err(TypeError::AlreadyDefined(var.to_string(), span.clone()));
                    }
                    let mapped_expr =
                        expr.to_core_actions(typeinfo, binding, fresh_gen, &mut norm_actions)?;
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
                        let mapped_arg =
                            arg.to_core_actions(typeinfo, binding, fresh_gen, &mut norm_actions)?;
                        mapped_args.push(mapped_arg);
                    }
                    let mapped_expr =
                        expr.to_core_actions(typeinfo, binding, fresh_gen, &mut norm_actions)?;
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
                        let mapped_arg =
                            arg.to_core_actions(typeinfo, binding, fresh_gen, &mut norm_actions)?;
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
                    match (e1, e2) {
                        (var @ GenericExpr::Var(..), GenericExpr::Call(_, f, args))
                        | (GenericExpr::Call(_, f, args), var @ GenericExpr::Var(..))
                            if f.is_constructor(typeinfo) =>
                        {
                            // This is a rewrite. Rewrites from a constructor or
                            // function to an id are much faster when we perform a
                            // `set` directly.
                            let head = f;
                            let expr = var;

                            let mut mapped_args = vec![];
                            for arg in args {
                                let mapped_arg = arg.to_core_actions(
                                    typeinfo,
                                    binding,
                                    fresh_gen,
                                    &mut norm_actions,
                                )?;
                                mapped_args.push(mapped_arg);
                            }
                            let mapped_expr = expr.to_core_actions(
                                typeinfo,
                                binding,
                                fresh_gen,
                                &mut norm_actions,
                            )?;
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
                        _ => {
                            let mapped_e1 = e1.to_core_actions(
                                typeinfo,
                                binding,
                                fresh_gen,
                                &mut norm_actions,
                            )?;
                            let mapped_e2 = e2.to_core_actions(
                                typeinfo,
                                binding,
                                fresh_gen,
                                &mut norm_actions,
                            )?;
                            norm_actions.push(GenericCoreAction::Union(
                                span.clone(),
                                mapped_e1.get_corresponding_var_or_lit(typeinfo),
                                mapped_e2.get_corresponding_var_or_lit(typeinfo),
                            ));
                            mapped_actions.0.push(GenericAction::Union(
                                span.clone(),
                                mapped_e1,
                                mapped_e2,
                            ));
                        }
                    };
                }
                GenericAction::Panic(span, string) => {
                    norm_actions.push(GenericCoreAction::Panic(span.clone(), string.clone()));
                    mapped_actions
                        .0
                        .push(GenericAction::Panic(span.clone(), string.clone()));
                }
                GenericAction::Expr(span, expr) => {
                    let mapped_expr =
                        expr.to_core_actions(typeinfo, binding, fresh_gen, &mut norm_actions)?;
                    mapped_actions
                        .0
                        .push(GenericAction::Expr(span.clone(), mapped_expr));
                }
            }
        }
        Ok((GenericCoreActions::new(norm_actions), mapped_actions))
    }
}

impl<Head, Leaf> GenericExpr<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    pub(crate) fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> (
        Vec<GenericAtom<HeadOrEq<Head>, Leaf>>,
        MappedExpr<Head, Leaf>,
    ) {
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
                    head: HeadOrEq::Head(f.clone()),
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

    pub(crate) fn to_core_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        typeinfo: &TypeInfo,
        binding: &mut IndexSet<Leaf>,
        fresh_gen: &mut FG,
        out_actions: &mut Vec<GenericCoreAction<Head, Leaf>>,
    ) -> Result<MappedExpr<Head, Leaf>, TypeError> {
        match self {
            GenericExpr::Lit(span, lit) => Ok(GenericExpr::Lit(span.clone(), lit.clone())),
            GenericExpr::Var(span, v) => {
                let sym = v.to_string();
                if binding.contains(v) || typeinfo.is_global(&sym) {
                    Ok(GenericExpr::Var(span.clone(), v.clone()))
                } else {
                    Err(TypeError::Unbound(sym, span.clone()))
                }
            }
            GenericExpr::Call(span, f, args) => {
                let mut norm_args = vec![];
                let mut mapped_args = vec![];
                for arg in args {
                    let mapped_arg =
                        arg.to_core_actions(typeinfo, binding, fresh_gen, out_actions)?;
                    norm_args.push(mapped_arg.get_corresponding_var_or_lit(typeinfo));
                    mapped_args.push(mapped_arg);
                }

                let var = fresh_gen.fresh(f);
                binding.insert(var.clone());

                out_actions.push(GenericCoreAction::Let(
                    span.clone(),
                    var.clone(),
                    f.clone(),
                    norm_args,
                ));

                Ok(GenericExpr::Call(
                    span.clone(),
                    CorrespondingVar::new(f.clone(), var),
                    mapped_args,
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
pub struct GenericCoreRule<HeadQ, HeadA, Leaf> {
    pub span: Span,
    pub body: Query<HeadQ, Leaf>,
    pub head: GenericCoreActions<HeadA, Leaf>,
}

pub(crate) type CoreRule = GenericCoreRule<StringOrEq, String, String>;
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
                                        GenericAtomTerm::Literal(atom.span.clone(), Literal::Unit),
                                    ],
                                })
                            }
                        },
                    }
                }
                HeadOrEq::Head(symbol) => Some(GenericAtom {
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

impl<Head, Leaf> GenericRule<Head, Leaf>
where
    Head: Clone + Display + IsFunc,
    Leaf: Clone + PartialEq + Eq + Display + Hash + Debug,
{
    pub(crate) fn to_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> Result<GenericCoreRule<HeadOrEq<Head>, Head, Leaf>, TypeError> {
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
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
        value_eq: impl Fn(&GenericAtomTerm<Leaf>, &GenericAtomTerm<Leaf>) -> Head,
    ) -> Result<GenericCoreRule<Head, Head, Leaf>, TypeError> {
        let rule = self.to_core_rule(typeinfo, fresh_gen)?;
        Ok(rule.canonicalize(value_eq))
    }
}

impl ResolvedRule {
    pub(crate) fn to_canonicalized_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut SymbolGen,
    ) -> Result<ResolvedCoreRule, TypeError> {
        let value_eq = &typeinfo.get_prims("value-eq").unwrap()[0];
        self.to_canonicalized_core_rule_impl(typeinfo, fresh_gen, |at1, at2| {
            ResolvedCall::Primitive(SpecializedPrimitive {
                primitive: value_eq.clone(),
                input: vec![at1.output(), at2.output()],
                output: UnitSort.to_arcsort(),
            })
        })
    }
}
