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
use std::marker::PhantomData;
use std::ops::AddAssign;

use crate::{constraint::grounded_check, *};
use egglog_ast::generic_ast::{Change, GenericAction, GenericActions, GenericExpr};
use egglog_ast::span::Span;
use egglog_ast::util::ListDisplay;
use typechecking::{FuncType, PrimitiveValidator, PrimitiveWithId, TypeError};

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
pub struct SpecializedPrimitive {
    /// Exact registration selected before nominal sorts are erased. Cloning
    /// retains its native implementation and per-context runtime identity.
    pub prim_with_id: PrimitiveWithId,
    input: Vec<ArcSort>,
    output: ArcSort,
}

impl SpecializedPrimitive {
    /// Get the name of this primitive
    pub fn name(&self) -> &str {
        self.prim_with_id.primitive.name()
    }

    /// Get the output sort of this primitive
    pub fn output(&self) -> &ArcSort {
        &self.output
    }

    /// Get the input sorts of this primitive
    pub fn input(&self) -> &[ArcSort] {
        &self.input
    }

    /// Get the external function ID of this primitive
    pub(crate) fn external_id(&self, ctx: crate::Context) -> ExternalFunctionId {
        self.prim_with_id.context_ids[ctx].unwrap_or_else(|| {
            panic!(
                "primitive {:?} is not valid in context {ctx:?}",
                self.prim_with_id.primitive.name()
            )
        })
    }

    /// Get the validator function of this primitive, if any
    pub fn validator(&self) -> Option<&PrimitiveValidator> {
        self.prim_with_id.validator.as_ref()
    }
}

impl PartialEq for SpecializedPrimitive {
    fn eq(&self, other: &Self) -> bool {
        // This is the key used when resolved atoms are deduplicated by
        // `(head, inputs)`. The context-id map identifies the primitive
        // registration, while the concrete input/output sorts identify the
        // specialization of generic primitives. The primitive name and
        // validator are registration metadata, so they are intentionally not
        // separate key fields.
        self.prim_with_id.context_ids == other.prim_with_id.context_ids
            && self.output.name() == other.output.name()
            && self.input.len() == other.input.len()
            && self
                .input
                .iter()
                .zip(&other.input)
                .all(|(a, b)| a.name() == b.name())
    }
}

impl Eq for SpecializedPrimitive {}

impl Hash for SpecializedPrimitive {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prim_with_id.context_ids.hash(state);
        self.output.name().hash(state);
        self.input.len().hash(state);
        for input in &self.input {
            input.name().hash(state);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolvedCall {
    Func(Arc<FuncType>),
    Primitive(SpecializedPrimitive),
}

impl ResolvedCall {
    pub fn name(&self) -> &str {
        match self {
            ResolvedCall::Func(func) => &func.name,
            ResolvedCall::Primitive(prim) => prim.name(),
        }
    }

    pub fn output(&self) -> &ArcSort {
        match self {
            ResolvedCall::Func(func) => &func.output,
            ResolvedCall::Primitive(prim) => prim.output(),
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

    pub fn from_resolution(
        head: &str,
        types: &[ArcSort],
        typeinfo: &TypeInfo,
        ctx: crate::Context,
        span: &Span,
    ) -> Result<ResolvedCall, TypeError> {
        if let Some(ty) = typeinfo.get_func_type(head) {
            let expected = ty.input.iter().chain(once(&ty.output)).map(|s| s.name());
            let actual = types.iter().map(|s| s.name());
            if expected.eq(actual) {
                return Ok(ResolvedCall::Func(ty.clone()));
            }
        }

        let mut primitives = typeinfo
            .get_prims(head)
            .into_iter()
            .flatten()
            .filter(|p| p.context_ids[ctx].is_some() && p.accept(types, typeinfo));
        if let Some(picked) = primitives.next() {
            // Type inference has already assigned concrete types, so a valid
            // overload (`+`, `map-empty`, container primitives) leaves exactly
            // one matching registration. A second match means two registrations
            // are indistinguishable for this signature and context.
            if primitives.next().is_some() {
                return Err(TypeError::AmbiguousPrimitive {
                    name: head.to_owned(),
                    ctx,
                    span: span.clone(),
                });
            }
            let (out, inp) = types.split_last().unwrap();
            return Ok(ResolvedCall::Primitive(SpecializedPrimitive {
                prim_with_id: picked.clone(),
                input: inp.to_vec(),
                output: out.clone(),
            }));
        }

        Err(TypeError::UnresolvedPrimitive {
            name: head.to_owned(),
            ctx,
            span: span.clone(),
        })
    }

    /// Whether this call is to a `function` table, as opposed to a constructor
    /// or a primitive.
    pub(crate) fn is_custom_func(&self) -> bool {
        matches!(self, ResolvedCall::Func(func) if func.subtype == FunctionSubtype::Custom)
    }
}

impl Display for ResolvedCall {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedCall::Func(func) => write!(f, "{}", func.name),
            ResolvedCall::Primitive(prim) => write!(f, "{}", prim.name()),
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
    // Alias substitution and repeated operands share concrete payloads. AST
    // projection is the explicit boundary that materializes an owned Literal.
    Literal(Span, Arc<Literal>),
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
            GenericAtomTerm::Literal(span, l) => GenericExpr::Lit(span.clone(), l.as_ref().clone()),
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
        ctx: crate::Context,
    ) -> Result<Vec<Box<dyn Constraint<AtomTerm, ArcSort>>>, TypeError> {
        let mut constraints = vec![];
        for atom in self.atoms.iter() {
            constraints.extend(atom.get_constraints(type_info, ctx)?.into_iter());
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
                    filter.head.prim_with_id.primitive.name(),
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
                    free_vars.remove(v);
                }
                GenericCoreAction::LetAtomTerm(_span, v, at) => {
                    add_from_atom(&mut free_vars, at);
                    free_vars.remove(v);
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

/// Shared state that threads through lowering from surface actions to core actions.
///
pub(crate) struct CoreActionContext<'a, Head, Leaf, FG> {
    /// Type environment describing functions, constructors, and primitives.
    pub typeinfo: &'a TypeInfo,
    /// Set of variables that are currently in scope during lowering.
    pub binding: &'a mut IndexSet<Leaf>,
    /// Generator used to create fresh symbols for intermediate values.
    pub fresh_gen: &'a mut FG,
    /// Whether we may rewrite `union` on constructors into `set`.
    pub union_to_set_optimization: bool,
    _marker: PhantomData<fn() -> Head>,
}

impl<'a, Head, Leaf, FG> CoreActionContext<'a, Head, Leaf, FG> {
    pub fn new(
        typeinfo: &'a TypeInfo,
        binding: &'a mut IndexSet<Leaf>,
        fresh_gen: &'a mut FG,
        union_to_set_optimization: bool,
    ) -> Self {
        Self {
            typeinfo,
            binding,
            fresh_gen,
            union_to_set_optimization,
            _marker: PhantomData,
        }
    }
}

pub(crate) trait GenericActionsExt<Head, Leaf> {
    #[allow(clippy::type_complexity)]
    fn to_core_actions<FG>(
        &self,
        ctx: &mut CoreActionContext<'_, Head, Leaf, FG>,
    ) -> Result<(GenericCoreActions<Head, Leaf>, MappedActions<Head, Leaf>), TypeError>
    where
        Head: Clone + Display + IsFunc,
        Leaf: Clone + PartialEq + Eq + Display + Hash,
        FG: FreshGen<Head, Leaf>;
}

impl<Head, Leaf> GenericActionsExt<Head, Leaf> for GenericActions<Head, Leaf>
where
    Head: Clone + Display + IsFunc,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    #[allow(clippy::type_complexity)]
    fn to_core_actions<FG>(
        &self,
        ctx: &mut CoreActionContext<'_, Head, Leaf, FG>,
    ) -> Result<(GenericCoreActions<Head, Leaf>, MappedActions<Head, Leaf>), TypeError>
    where
        Head: Clone + Display + IsFunc,
        Leaf: Clone + PartialEq + Eq + Display + Hash,
        FG: FreshGen<Head, Leaf>,
    {
        let mut norm_actions = vec![];
        let mut mapped_actions: MappedActions<Head, Leaf> = GenericActions(vec![]);
        let typeinfo = ctx.typeinfo;
        let union_to_set_optimization = ctx.union_to_set_optimization;

        // During the lowering, there are two important guaratees:
        //   Every used variable should be bound.
        //   Every introduced variable should be unbound before.
        for action in self.0.iter() {
            match action {
                GenericAction::Let(span, var, expr) => {
                    if ctx.binding.contains(var) {
                        return Err(TypeError::AlreadyDefined(var.to_string(), span.clone()));
                    }
                    let mapped_expr = expr.to_core_actions(ctx, &mut norm_actions)?;
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
                    ctx.binding.insert(var.clone());
                }
                GenericAction::Set(span, head, args, expr) => {
                    let mut mapped_args = vec![];
                    for arg in args {
                        let mapped_arg = arg.to_core_actions(ctx, &mut norm_actions)?;
                        mapped_args.push(mapped_arg);
                    }
                    let mapped_expr = expr.to_core_actions(ctx, &mut norm_actions)?;
                    norm_actions.push(GenericCoreAction::Set(
                        span.clone(),
                        head.clone(),
                        mapped_args
                            .iter()
                            .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                            .collect(),
                        mapped_expr.get_corresponding_var_or_lit(typeinfo),
                    ));
                    let v = ctx.fresh_gen.fresh(head);
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
                        let mapped_arg = arg.to_core_actions(ctx, &mut norm_actions)?;
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
                    let v = ctx.fresh_gen.fresh(head);
                    mapped_actions.0.push(GenericAction::Change(
                        span.clone(),
                        *change,
                        CorrespondingVar::new(head.clone(), v),
                        mapped_args,
                    ));
                }
                GenericAction::Union(span, e1, e2) => {
                    // Optimization: if one side is a constructor call and the other side is a variable,
                    // we can lower this to a Set action instead of a Union action.
                    // This produces invalid egglog, since top-level egglog expects only Union on constructors.
                    // We disable this with union_to_set_optimization flag for term/proof mode.
                    // TODO move this optimization to later stage so we can keep it enabled in term/proof mode.
                    match (e1, e2) {
                        (var @ GenericExpr::Var(..), GenericExpr::Call(_, f, args))
                        | (GenericExpr::Call(_, f, args), var @ GenericExpr::Var(..))
                            if f.is_constructor(typeinfo) && union_to_set_optimization =>
                        {
                            let head = f;
                            let expr = var;
                            let mut mapped_args = vec![];
                            for arg in args {
                                let mapped_arg = arg.to_core_actions(ctx, &mut norm_actions)?;
                                mapped_args.push(mapped_arg);
                            }
                            let mapped_expr = expr.to_core_actions(ctx, &mut norm_actions)?;
                            norm_actions.push(GenericCoreAction::Set(
                                span.clone(),
                                head.clone(),
                                mapped_args
                                    .iter()
                                    .map(|e| e.get_corresponding_var_or_lit(typeinfo))
                                    .collect(),
                                mapped_expr.get_corresponding_var_or_lit(typeinfo),
                            ));
                            let v = ctx.fresh_gen.fresh(head);
                            mapped_actions.0.push(GenericAction::Set(
                                span.clone(),
                                CorrespondingVar::new(head.clone(), v),
                                mapped_args,
                                mapped_expr,
                            ));
                        }
                        _ => {
                            let mapped_e1 = e1.to_core_actions(ctx, &mut norm_actions)?;
                            let mapped_e2 = e2.to_core_actions(ctx, &mut norm_actions)?;
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
                    let mapped_expr = expr.to_core_actions(ctx, &mut norm_actions)?;
                    mapped_actions
                        .0
                        .push(GenericAction::Expr(span.clone(), mapped_expr));
                }
            }
        }
        Ok((GenericCoreActions::new(norm_actions), mapped_actions))
    }
}

pub(crate) trait GenericExprExt<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> (
        Vec<GenericAtom<HeadOrEq<Head>, Leaf>>,
        MappedExpr<Head, Leaf>,
    );

    fn to_core_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        ctx: &mut CoreActionContext<'_, Head, Leaf, FG>,
        out_actions: &mut Vec<GenericCoreAction<Head, Leaf>>,
    ) -> Result<MappedExpr<Head, Leaf>, TypeError>;
}

impl<Head, Leaf> GenericExprExt<Head, Leaf> for GenericExpr<Head, Leaf>
where
    Head: Clone + Display,
    Leaf: Clone + PartialEq + Eq + Display + Hash,
{
    fn to_query(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
    ) -> (
        Vec<GenericAtom<HeadOrEq<Head>, Leaf>>,
        MappedExpr<Head, Leaf>,
    )
    where
        Head: Clone + Display,
        Leaf: Clone + PartialEq + Eq + Display + Hash,
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

    fn to_core_actions<FG: FreshGen<Head, Leaf>>(
        &self,
        ctx: &mut CoreActionContext<'_, Head, Leaf, FG>,
        out_actions: &mut Vec<GenericCoreAction<Head, Leaf>>,
    ) -> Result<MappedExpr<Head, Leaf>, TypeError> {
        let typeinfo = ctx.typeinfo;
        match self {
            GenericExpr::Lit(span, lit) => Ok(GenericExpr::Lit(span.clone(), lit.clone())),
            GenericExpr::Var(span, v) => {
                let sym = v.to_string();
                if ctx.binding.contains(v) || typeinfo.is_global(&sym) {
                    Ok(GenericExpr::Var(span.clone(), v.clone()))
                } else {
                    Err(TypeError::Unbound(sym, span.clone()))
                }
            }
            GenericExpr::Call(span, f, args) => {
                let mut norm_args = vec![];
                let mut mapped_args = vec![];
                for arg in args {
                    let mapped_arg = arg.to_core_actions(ctx, out_actions)?;
                    norm_args.push(mapped_arg.get_corresponding_var_or_lit(typeinfo));
                    mapped_args.push(mapped_arg);
                }
                let var = ctx.fresh_gen.fresh(f);
                ctx.binding.insert(var.clone());
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
        // Only variables enter this union-find. Concrete terms anchor a class;
        // two concrete representatives retain their original equality check.
        let mut variables = IndexMap::default();
        for atom in &result_rule.body.atoms {
            if atom.head.is_eq() {
                assert_eq!(atom.args.len(), 2);
                for term in &atom.args {
                    if let GenericAtomTerm::Var(_, name) = term {
                        variables
                            .entry(name.clone())
                            .or_insert_with(|| term.clone());
                    }
                }
            }
        }
        let mut parents: Vec<_> = (0..variables.len()).collect();
        let mut sizes = vec![1usize; variables.len()];
        let mut representatives: Vec<_> = variables.values().cloned().collect();
        fn find(parents: &mut [usize], mut index: usize) -> usize {
            while parents[index] != index {
                parents[index] = parents[parents[index]];
                index = parents[index];
            }
            index
        }
        for atom in &result_rule.body.atoms {
            if !atom.head.is_eq() {
                continue;
            }
            let mut roots = [None, None];
            for (position, term) in atom.args.iter().enumerate() {
                if let GenericAtomTerm::Var(_, name) = term {
                    roots[position] =
                        Some(find(&mut parents, variables.get_index_of(name).unwrap()));
                }
            }
            let lhs = roots[0]
                .map(|root| representatives[root].clone())
                .unwrap_or_else(|| atom.args[0].clone());
            let rhs = roots[1]
                .map(|root| representatives[root].clone())
                .unwrap_or_else(|| atom.args[1].clone());
            if lhs == rhs {
                continue;
            }
            match (&lhs, &rhs) {
                (GenericAtomTerm::Var(..), GenericAtomTerm::Var(..)) => {
                    let (left, right) = (roots[0].unwrap(), roots[1].unwrap());
                    let (parent, child) = if sizes[left] >= sizes[right] {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    parents[child] = parent;
                    sizes[parent] += sizes[child];
                    // Match ordered substitution's choice of the right variable,
                    // independently of the balanced tree's structural root.
                    representatives[parent] = rhs;
                }
                (GenericAtomTerm::Var(..), _) => representatives[roots[0].unwrap()] = rhs,
                (_, GenericAtomTerm::Var(..)) => representatives[roots[1].unwrap()] = lhs,
                _ => (),
            }
        }
        let mut subst = HashMap::default();
        for (index, (name, _)) in variables.iter().enumerate() {
            let representative = &representatives[find(&mut parents, index)];
            if !matches!(representative, GenericAtomTerm::Var(_, other) if other == name) {
                subst.insert(name.clone(), representative.clone());
            }
        }
        if !subst.is_empty() {
            result_rule.subst(&subst);
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
                                        GenericAtomTerm::Literal(atom.span.clone(), Literal::Unit.into()),
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

fn equiv_groups_to_eq_constraints<Head, Leaf>(
    groups: &HashMap<(Head, Vec<GenericAtomTerm<Leaf>>), Vec<GenericAtomTerm<Leaf>>>,
    span: &Span,
) -> Vec<GenericAtom<HeadOrEq<Head>, Leaf>>
where
    Leaf: Eq + Clone + Hash + Debug,
    Head: Clone,
{
    let mut eq_constraints = vec![];
    for group in groups.values() {
        let first = &group[0];
        for other in &group[1..] {
            if first == other {
                continue;
            }
            eq_constraints.push(GenericAtom {
                span: span.clone(),
                head: HeadOrEq::Eq,
                args: vec![first.clone(), other.clone()],
            });
        }
    }
    eq_constraints
}

impl<Head, Leaf> GenericCoreRule<Head, Head, Leaf>
where
    Leaf: Eq + Clone + Hash + Debug,
    Head: Clone + Eq + Hash,
{
    /// Functions in egglog follow functional dependency, and this pass removes
    /// duplicate variables based on functional dependencies.
    /// For example, if we have two atoms `R(x, y, z1)` and `R(x, y, z2)`,
    /// then we can remove one of them and add an equality constraint `z1 = z2`.
    /// This is done until fixpoint, so it is kind of like rebuilding.
    pub(crate) fn remove_dup_vars(
        mut self,
        value_eq: impl Fn(&GenericAtomTerm<Leaf>, &GenericAtomTerm<Leaf>) -> Head,
    ) -> Self {
        // Maps function calls to sets of equivalent variables to be deduplicated
        let mut groups: HashMap<(Head, Vec<GenericAtomTerm<Leaf>>), Vec<GenericAtomTerm<Leaf>>> =
            HashMap::default();

        // Remove entries wit identical (head, inputs) pair and mark respective outputs to be merged.
        self.body.atoms.retain(|atom| {
            let (out, inp) = atom.args.split_last().unwrap();
            let key = (atom.head.clone(), inp.to_owned());
            let group = groups.entry(key).or_default();
            group.push(out.clone());
            group.len() == 1
        });

        let new_atoms = equiv_groups_to_eq_constraints(&groups, &self.span);

        if new_atoms.is_empty() {
            self
        } else {
            let atoms: Vec<GenericAtom<HeadOrEq<Head>, Leaf>> = new_atoms
                .into_iter()
                .chain(self.body.atoms.into_iter().map(|atom| GenericAtom {
                    span: atom.span,
                    head: HeadOrEq::Head(atom.head),
                    args: atom.args,
                }))
                .collect();

            GenericCoreRule {
                span: self.span,
                body: Query { atoms },
                head: self.head,
            }
            .canonicalize(&value_eq)
            .remove_dup_vars(value_eq)
        }
    }
}

pub(crate) trait GenericRuleExt<Head, Leaf> {
    fn to_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
        union_to_set_optimization: bool,
    ) -> Result<GenericCoreRule<HeadOrEq<Head>, Head, Leaf>, TypeError>
    where
        Head: Clone + Display + IsFunc,
        Leaf: Clone + PartialEq + Eq + Display + Hash + Debug;
}

impl<Head, Leaf> GenericRuleExt<Head, Leaf> for GenericRule<Head, Leaf>
where
    Head: Clone + Display + IsFunc,
    Leaf: Clone + PartialEq + Eq + Display + Hash + Debug,
{
    fn to_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut impl FreshGen<Head, Leaf>,
        union_to_set_optimization: bool,
    ) -> Result<GenericCoreRule<HeadOrEq<Head>, Head, Leaf>, TypeError>
    where
        Head: Clone + Display + IsFunc,
        Leaf: Clone + PartialEq + Eq + Display + Hash + Debug,
    {
        let (body, _correspondence) = Facts(self.body.clone()).to_query(typeinfo, fresh_gen);
        let mut binding = body.get_vars();
        let mut ctx =
            CoreActionContext::new(typeinfo, &mut binding, fresh_gen, union_to_set_optimization);
        let (head, _correspondence) = self.head.to_core_actions(&mut ctx)?;
        Ok(GenericCoreRule {
            span: self.span.clone(),
            body,
            head,
        })
    }
}

pub(crate) trait ResolvedRuleExt {
    fn to_canonicalized_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut SymbolGen,
        union_to_set_optimization: bool,
    ) -> Result<ResolvedCoreRule, TypeError>;
}

impl ResolvedRuleExt for ResolvedRule {
    fn to_canonicalized_core_rule(
        &self,
        typeinfo: &TypeInfo,
        fresh_gen: &mut SymbolGen,
        union_to_set_optimization: bool,
    ) -> Result<ResolvedCoreRule, TypeError> {
        let value_eq = &typeinfo.get_prims("value-eq").unwrap()[0];
        let value_eq = |at1: &ResolvedAtomTerm, at2: &ResolvedAtomTerm| {
            ResolvedCall::Primitive(SpecializedPrimitive {
                prim_with_id: value_eq.clone(),
                input: vec![at1.output(), at2.output()],
                output: UnitSort.to_arcsort(),
            })
        };

        let rule = self.to_core_rule(typeinfo, fresh_gen, union_to_set_optimization)?;

        // The groundedness check happens before canonicalization, because canonicalization
        // may turn ungrounded variables in a query to unbounded variables in actions (e.g.,
        // `(rule ((= x y)) ((R x y)))`) but unboundedness is only checked during type checking.
        grounded_check(&rule)?;

        let rule = rule.canonicalize(&value_eq);

        let rule = rule.remove_dup_vars(value_eq);

        Ok(rule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestCoreRule = GenericCoreRule<String, String, String>;

    fn make_var(name: &str) -> GenericAtomTerm<String> {
        GenericAtomTerm::Var(span!(), name.to_string())
    }

    fn make_atom(head: &str, args: Vec<&str>) -> GenericAtom<String, String> {
        GenericAtom {
            span: span!(),
            head: head.to_string(),
            args: args.into_iter().map(make_var).collect(),
        }
    }

    fn value_eq_string(_at1: &GenericAtomTerm<String>, _at2: &GenericAtomTerm<String>) -> String {
        "value-eq".to_string()
    }

    #[test]
    fn canonicalize_shares_literal_payloads_without_changing_content_or_spans() {
        use std::collections::hash_map::DefaultHasher;

        let span = span!();
        let payload = Arc::new(Literal::String("shared payload".repeat(4096)));
        let literal = AtomTerm::Literal(span.clone(), payload.clone());
        let result = GenericCoreRule {
            span: span!(),
            body: Query {
                atoms: vec![
                    GenericAtom {
                        span: span!(),
                        head: HeadOrEq::Eq,
                        args: vec![make_var("x"), literal.clone()],
                    },
                    GenericAtom {
                        span: span!(),
                        head: HeadOrEq::Head("observe".into()),
                        args: vec![make_var("x"); 1024],
                    },
                ],
            },
            head: GenericCoreActions(vec![]),
        }
        .canonicalize(value_eq_string);
        assert_eq!(result.body.atoms.len(), 1);
        for term in &result.body.atoms[0].args {
            let AtomTerm::Literal(actual_span, actual) = term else {
                panic!("substituted literal")
            };
            assert_eq!(actual_span, &span);
            assert!(Arc::ptr_eq(actual, &payload));
        }
        let [GenericCoreAction::LetAtomTerm(_, name, AtomTerm::Literal(_, actual))] =
            result.head.0.as_slice()
        else {
            panic!("one head alias")
        };
        assert_eq!(name, "x");
        assert!(Arc::ptr_eq(actual, &payload));
        let equal = AtomTerm::Literal(span!(), Arc::new(payload.as_ref().clone()));
        assert_eq!(literal, equal);
        let hash = |term: &AtomTerm| {
            let mut hasher = DefaultHasher::new();
            term.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&literal), hash(&equal));
        for bits in [(-0.0_f64).to_bits(), 0x7ff8_0000_0000_0123] {
            let value = Arc::new(Literal::Float(f64::from_bits(bits).into()));
            let term = AtomTerm::Literal(span.clone(), value.clone());
            let AtomTerm::Literal(actual_span, actual) = term.clone() else {
                unreachable!()
            };
            assert_eq!(actual_span, span);
            assert!(Arc::ptr_eq(&actual, &value));
            let Expr::Lit(actual_span, Literal::Float(actual)) = term.to_expr::<String>() else {
                unreachable!()
            };
            assert_eq!(actual_span, span);
            assert_eq!(actual.0.to_bits(), bits);
        }
    }

    #[test]
    fn canonicalize_batches_variable_cycles_and_head_aliases() {
        let mut atoms: Vec<_> = [["x", "y"], ["y", "z"], ["z", "x"], ["w", "x"]]
            .into_iter()
            .map(|[lhs, rhs]| GenericAtom {
                span: span!(),
                head: HeadOrEq::Eq,
                args: vec![make_var(lhs), make_var(rhs)],
            })
            .collect();
        atoms.push(GenericAtom {
            span: span!(),
            head: HeadOrEq::Head("observe".to_owned()),
            args: ["x", "y", "z", "w"].into_iter().map(make_var).collect(),
        });
        let result = GenericCoreRule {
            span: span!(),
            body: Query { atoms },
            head: GenericCoreActions(vec![GenericCoreAction::Union(
                span!(),
                make_var("x"),
                make_var("w"),
            )]),
        }
        .canonicalize(value_eq_string);
        assert_eq!(result.body.atoms.len(), 1);
        assert_eq!(result.body.atoms[0].args, vec![make_var("z"); 4]);
        assert_eq!(result.head.0.len(), 4);
        for name in ["x", "y", "w"] {
            assert!(result.head.0.iter().any(
                |action| matches!(action, GenericCoreAction::LetAtomTerm(_, bound, value)
                    if bound == name && *value == make_var("z"))
            ));
        }
    }

    #[test]
    fn canonicalize_keeps_distinct_concrete_constraints_and_spans() {
        let spans = [
            span!(),
            span!(),
            span!(),
            span!(),
            span!(),
            span!(),
            span!(),
        ];
        let one = GenericAtomTerm::Literal(span!(), Literal::Int(1).into());
        let two = GenericAtomTerm::Literal(span!(), Literal::Int(2).into());
        let global = GenericAtomTerm::Global(span!(), "global".to_owned());
        let other = GenericAtomTerm::Global(span!(), "other".to_owned());
        let equations = [
            [make_var("x"), one.clone()],
            [make_var("y"), two.clone()],
            [make_var("x"), make_var("y")],
            [make_var("z"), global.clone()],
            [make_var("z"), other.clone()],
            [one.clone(), two.clone()],
            [global.clone(), one.clone()],
        ];
        let mut atoms: Vec<_> = equations
            .into_iter()
            .zip(&spans)
            .map(|(args, span)| GenericAtom {
                span: span.clone(),
                head: HeadOrEq::Eq,
                args: args.to_vec(),
            })
            .collect();
        atoms.push(GenericAtom {
            span: span!(),
            head: HeadOrEq::Head("observe".to_owned()),
            args: vec![make_var("x"), make_var("y"), make_var("z")],
        });
        let result = GenericCoreRule {
            span: span!(),
            body: Query { atoms },
            head: GenericCoreActions(vec![GenericCoreAction::Union(
                span!(),
                make_var("x"),
                make_var("z"),
            )]),
        }
        .canonicalize(value_eq_string);
        let expected = [
            (2, one.clone(), two.clone()),
            (4, global.clone(), other),
            (5, one.clone(), two.clone()),
            (6, global.clone(), one.clone()),
        ];
        assert_eq!(result.body.atoms.len(), expected.len() + 1);
        for (atom, (index, lhs, rhs)) in result.body.atoms.iter().zip(expected) {
            assert_eq!(atom.head, "value-eq");
            assert_eq!(atom.span, spans[index]);
            assert_eq!(
                atom.args,
                vec![
                    lhs,
                    rhs,
                    GenericAtomTerm::Literal(spans[index].clone(), Literal::Unit.into())
                ]
            );
        }
        assert_eq!(
            result.body.atoms.last().unwrap().args,
            vec![one.clone(), two.clone(), global.clone()]
        );
        for (name, expected) in [("x", one), ("y", two), ("z", global)] {
            assert!(result.head.0.iter().any(
                |action| matches!(action, GenericCoreAction::LetAtomTerm(_, bound, value)
                    if bound == name && *value == expected)
            ));
        }
    }

    #[test]
    fn canonicalize_matches_sequential_substitution_semantics() {
        fn head_unions(head: &GenericCoreActions<String, String>) -> Vec<[AtomTerm; 2]> {
            let mut bindings = HashMap::default();
            let mut result = Vec::new();
            for action in &head.0 {
                match action {
                    GenericCoreAction::LetAtomTerm(_, name, term) => {
                        let value = match term {
                            GenericAtomTerm::Var(_, name) => bindings.get(name).unwrap_or(term),
                            _ => term,
                        }
                        .clone();
                        bindings.insert(name.clone(), value);
                    }
                    GenericCoreAction::Union(_, lhs, rhs) => {
                        result.push([lhs, rhs].map(|term| {
                            match term {
                                GenericAtomTerm::Var(_, name) => bindings.get(name).unwrap_or(term),
                                _ => term,
                            }
                            .clone()
                        }));
                    }
                    _ => unreachable!(),
                }
            }
            result
        }
        let one = GenericAtomTerm::Literal(span!(), Literal::Int(1).into());
        let two = GenericAtomTerm::Literal(span!(), Literal::Int(2).into());
        let global = GenericAtomTerm::Global(span!(), "global".to_owned());
        let other = GenericAtomTerm::Global(span!(), "other".to_owned());
        for equations in [
            vec![
                [make_var("x"), make_var("y")],
                [make_var("y"), make_var("z")],
                [make_var("z"), make_var("x")],
            ],
            vec![
                [make_var("x"), one.clone()],
                [make_var("y"), two.clone()],
                [make_var("x"), make_var("y")],
            ],
            vec![
                [make_var("x"), make_var("y")],
                [make_var("y"), one.clone()],
                [make_var("x"), two.clone()],
            ],
            vec![
                [make_var("x"), global.clone()],
                [make_var("y"), other.clone()],
                [make_var("x"), make_var("y")],
            ],
            vec![
                [make_var("x"), make_var("y")],
                [make_var("y"), global.clone()],
                [make_var("z"), make_var("x")],
                [make_var("z"), other.clone()],
            ],
        ] {
            let mut atoms: Vec<_> = equations
                .into_iter()
                .map(|args| GenericAtom {
                    span: span!(),
                    head: HeadOrEq::Eq,
                    args: args.to_vec(),
                })
                .collect();
            atoms.push(GenericAtom {
                span: span!(),
                head: HeadOrEq::Head("observe".to_owned()),
                args: vec![make_var("x"), make_var("y"), make_var("z")],
            });
            let rule = GenericCoreRule {
                span: span!(),
                body: Query { atoms },
                head: GenericCoreActions(vec![GenericCoreAction::Union(
                    span!(),
                    make_var("x"),
                    make_var("y"),
                )]),
            };
            let mut sequential = rule.clone();
            // Reference the former one-equality-at-a-time elimination strategy.
            loop {
                let next = sequential.body.atoms.iter().find_map(|atom| {
                    if atom.head.is_eq() && atom.args[0] != atom.args[1] {
                        match &atom.args[..] {
                            [GenericAtomTerm::Var(_, name), value]
                            | [value, GenericAtomTerm::Var(_, name)] => {
                                Some((name.clone(), value.clone()))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                });
                let Some((name, value)) = next else { break };
                sequential.subst(&HashMap::from_iter([(name, value)]));
            }
            let expected = sequential.canonicalize(value_eq_string);
            let actual = rule.canonicalize(value_eq_string);
            assert_eq!(actual.body.atoms, expected.body.atoms);
            assert_eq!(head_unions(&actual.head), head_unions(&expected.head));
        }
    }

    #[test]
    fn test_remove_dup_vars_basic() {
        let rule = TestCoreRule {
            span: span!(),
            body: Query {
                atoms: vec![
                    make_atom("R", vec!["x", "y", "z1"]),
                    make_atom("R", vec!["x", "y", "z2"]),
                    make_atom("R", vec!["x", "z3"]),
                    make_atom("R", vec!["a", "b", "z4"]),
                    make_atom("R", vec!["c", "d", "z5"]),
                ],
            },
            head: GenericCoreActions::default(),
        };

        let result = rule.remove_dup_vars(value_eq_string);

        assert_eq!(result.body.atoms.len(), 4);
        assert_eq!(result.body.atoms[0].head, "R");
        assert_eq!(result.body.atoms[0].args[0], make_var("x"));
        assert_eq!(result.body.atoms[0].args[1], make_var("y"));
        assert_eq!(result.body.atoms[1].args.len(), 2);
    }

    #[test]
    fn test_remove_dup_vars_fixpoint() {
        // Test: R(x, y, z1), R(x, y, z2), R(x, y, z3) should all get unified
        // This tests that the fixpoint iteration works correctly
        let rule = TestCoreRule {
            span: span!(),
            body: Query {
                atoms: vec![
                    make_atom("R", vec!["x", "y", "z1"]),
                    make_atom("R", vec!["x", "y", "z2"]),
                    make_atom("R", vec!["x", "y", "z3"]),
                    make_atom("S", vec!["z1", "z2", "z3"]),
                    make_atom("S", vec!["z2", "z1", "z4"]),
                ],
            },
            head: GenericCoreActions::default(),
        };

        let result = rule.remove_dup_vars(value_eq_string);

        assert_eq!(result.body.atoms.len(), 2);
        assert_eq!(result.body.atoms[0].head, "R");
        assert_eq!(result.body.atoms[0].args[0], make_var("x"));
        assert_eq!(result.body.atoms[0].args[1], make_var("y"));
        assert_eq!(result.body.atoms[1].args[0], result.body.atoms[1].args[1]);

        let rule = TestCoreRule {
            span: span!(),
            body: Query {
                atoms: vec![
                    make_atom("R", vec!["x", "y", "z1"]),
                    make_atom("R", vec!["x", "y", "z2"]),
                    make_atom("R", vec!["x", "y", "z3"]),
                    make_atom("R", vec!["z1", "z2", "z1"]),
                    make_atom("R", vec!["z2", "z1", "x"]),
                    make_atom("R", vec!["z2", "z2", "y"]),
                ],
            },
            head: GenericCoreActions::default(),
        };

        let result = rule.remove_dup_vars(value_eq_string);

        assert_eq!(result.body.atoms.len(), 1);
        assert_eq!(result.body.atoms[0].head, "R");
        assert_eq!(result.body.atoms[0].args[0], result.body.atoms[0].args[1]);
        assert_eq!(result.body.atoms[0].args[0], result.body.atoms[0].args[2]);
    }

    #[test]
    fn test_remove_dup_vars_with_actions_using_removed_var() {
        let rule = TestCoreRule {
            span: span!(),
            body: Query {
                atoms: vec![
                    make_atom("R", vec!["x", "y", "z1"]),
                    make_atom("R", vec!["x", "y", "z2"]),
                ],
            },
            head: GenericCoreActions(vec![GenericCoreAction::Union(
                span!(),
                make_var("z2"),
                make_var("z1"),
            )]),
        };

        let result = rule.remove_dup_vars(value_eq_string);

        assert_eq!(result.body.atoms.len(), 1);
        assert!(matches!(
            &result.head.0.as_slice(),
            &[
                GenericCoreAction::LetAtomTerm(_, _, _),
                GenericCoreAction::Union(_, _, _)
            ]
        ));
    }
}
