use derive_more::{Deref, DerefMut, IntoIterator};
use smallvec::SmallVec;
use std::{borrow::Borrow, fmt, hash::Hash, marker::PhantomData, panic::Location, sync::{atomic::AtomicU32, Arc}};
use symbol_table::GlobalSymbol;

use crate::{
    ast::{Command, GenericAction, GenericExpr, RustSpan, Schema, Span, Subdatatypes, Variant},
    span,
};
pub type EgglogAction = GenericAction<String,String>;

#[derive(Debug)]
pub enum TxCommand {
    StringCommand { command: String },
    NativeCommand { command: Command },
}

pub trait Tx: 'static {
    /// receive is guaranteed to not be called in proc macro
    fn send(&self, sended: TxCommand);
    fn on_new(&self, node: &(impl EgglogNode + 'static));
    fn on_set(&self, node: &mut (impl EgglogNode + 'static));
    fn on_func_set<'a, F: EgglogFunc>(
        &self,
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
        output: <F::Output as EgglogFuncOutput>::Ref<'a>,
    );
}
pub trait Rx: 'static {
    fn on_func_get<'a, 'b, F: EgglogFunc>(
        &self,
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
    ) -> <F::Output as EgglogFuncOutput>::Ref<'b>;
    fn on_funcs_get<'a, 'b, F: EgglogFunc>(
        &self,
        max_size: Option<usize>,
    ) -> Vec<(
        <F::Input as EgglogFuncInputs>::Ref<'b>,
        <F::Output as EgglogFuncOutput>::Ref<'b>,
    )>;
    fn on_pull(&self, node: &(impl EgglogNode + 'static));
}

pub trait SingletonGetter: 'static {
    type RetTy;
    fn sgl() -> &'static Self::RetTy;
}

pub trait TxSgl: 'static + Sized + SingletonGetter {
    // delegate all functions from Tx
    fn receive(received: TxCommand);
    fn on_new(node: &(impl EgglogNode + 'static));
    fn on_set(node: &mut (impl EgglogNode + 'static));
    fn on_func_set<'a, F: EgglogFunc>(
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
        output: <F::Output as EgglogFuncOutput>::Ref<'a>,
    );
}
pub trait RxSgl: 'static + Sized + SingletonGetter {
    // delegate all functions from Rx
    fn on_func_get<'a, 'b, F: EgglogFunc>(
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
    ) -> <F::Output as EgglogFuncOutput>::Ref<'b>;
    fn on_funcs_get<'a, 'b, F: EgglogFunc>(
        max_size: Option<usize>,
    ) -> Vec<(
        <F::Input as EgglogFuncInputs>::Ref<'b>,
        <F::Output as EgglogFuncOutput>::Ref<'b>,
    )>;
    fn on_pull(node: &(impl EgglogNode + 'static));
}

impl<T: Tx + 'static, S: SingletonGetter<RetTy = T> + 'static> TxSgl for S {
    fn receive(received: TxCommand) {
        Self::sgl().send(received);
    }
    fn on_new(node: &(impl EgglogNode + 'static)) {
        Self::sgl().on_new(node);
    }
    fn on_set(node: &mut (impl EgglogNode + 'static)) {
        Self::sgl().on_set(node);
    }

    fn on_func_set<'a, F: EgglogFunc>(
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
        output: <F::Output as EgglogFuncOutput>::Ref<'a>,
    ) {
        Self::sgl().on_func_set::<F>(input, output);
    }
}
impl<R: Rx + 'static, S: SingletonGetter<RetTy = R> + 'static> RxSgl for S {
    fn on_func_get<'a, 'b, F: EgglogFunc>(
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
    ) -> <F::Output as EgglogFuncOutput>::Ref<'b> {
        Self::sgl().on_func_get::<F>(input)
    }

    fn on_funcs_get<'a, 'b, F: EgglogFunc>(
        max_size: Option<usize>,
    ) -> Vec<(
        <F::Input as EgglogFuncInputs>::Ref<'b>,
        <F::Output as EgglogFuncOutput>::Ref<'b>,
    )> {
        Self::sgl().on_funcs_get::<F>(max_size)
    }
    fn on_pull(node: &(impl EgglogNode + 'static)) {
        Self::sgl().on_pull(node)
    }
}

/// version control triat
/// which should be implemented by Tx
pub trait VersionCtl {
    fn locate_latest(&self, node: Sym) -> Sym;
    fn locate_next(&self, node: Sym) -> Sym;
    fn locate_prev(&self, node: Sym) -> Sym;
    fn set_latest(&self, node: &mut Sym);
    fn set_next(&self, node: &mut Sym);
    fn set_prev(&self, node: &mut Sym);
}

/// version control triat
/// which should be implemented by Tx
pub trait VersionCtlSgl {
    fn locate_latest(node: Sym) -> Sym;
    fn locate_next(node: Sym) -> Sym;
    fn locate_prev(node: Sym) -> Sym;
    fn set_latest(node: &mut Sym);
    fn set_next(node: &mut Sym);
    fn set_prev(node: &mut Sym);
}

impl<Ret: Tx + VersionCtl + 'static, S: SingletonGetter<RetTy = Ret>> VersionCtlSgl for S {
    fn locate_latest(node: Sym) -> Sym {
        Self::sgl().locate_latest(node)
    }
    fn locate_next(node: Sym) -> Sym {
        Self::sgl().locate_next(node)
    }
    fn locate_prev(node: Sym) -> Sym {
        Self::sgl().locate_prev(node)
    }
    fn set_latest(node: &mut Sym) {
        Self::sgl().set_latest(node)
    }
    fn set_next(node: &mut Sym) {
        Self::sgl().set_next(node)
    }
    fn set_prev(node: &mut Sym) {
        Self::sgl().set_prev(node)
    }
}

pub trait EgglogTy {
    const TY_NAME: &'static str;
    const TY_NAME_LOWER: &'static str;
}
pub trait UpdateCounter<T: EgglogTy> {
    fn inc_counter(&mut self, counter: &mut TyCounter<T>) -> Sym<T>;
}
pub struct TySortString(pub &'static str);
pub struct FuncSortString(pub &'static str);
pub struct TyConstructor {
    pub cons_name: &'static str,
    pub input: &'static [&'static str],
    pub output: &'static str,
    pub cost: Option<usize>,
    pub unextractable: bool,
}

impl<T> Sym<T> {
    pub fn erase(&self) -> Sym<()> {
        // safety note: type erasure
        unsafe { *&*(self as *const Sym<T> as *const Sym) }
    }
    pub fn erase_ref(&self) -> &Sym<()> {
        // safety note: type erasure
        unsafe { &*(self as *const Sym<T> as *const Sym) }
    }
    pub fn erase_mut(&mut self) -> &mut Sym<()> {
        // safety note: type erasure
        unsafe { &mut *(self as *mut Sym<T> as *mut Sym) }
    }
}

/// trait of basic functions to interact with egglog
pub trait ToEgglog {
    fn to_egglog_string(&self) -> String;
    fn to_egglog(&self) -> EgglogAction;
}

/// version control triat
/// which should be implemented by Node
pub trait LocateVersion {
    fn locate_latest(&mut self);
    fn locate_next(&mut self);
    fn locate_prev(&mut self);
}
/// trait of node behavior
pub trait EgglogNode: ToEgglog {
    fn succs_mut(&mut self) -> Vec<&mut Sym>;
    fn succs(&self) -> Vec<Sym>;
    /// set new sym and return the new sym
    fn next_sym(&mut self) -> Sym;
    // return current sym
    fn cur_sym(&self) -> Sym;
    fn cur_sym_mut(&mut self) -> &mut Sym;

    fn clone_dyn(&self) -> Box<dyn EgglogNode>;
}

// collect all sorts into inventory, so that we could send the definitions of types.
inventory::collect!(Decl);

pub trait EgglogEnumVariantTy: Clone + 'static {
    const TY_NAME: &'static str;
}
/// instance of specified EgglogTy & its VariantTy
#[derive(Debug, Clone)]
pub struct Node<T, R, I, S>
where
    T: EgglogTy,
    R: SingletonGetter,
    I: NodeInner<T>,
    S: EgglogEnumVariantTy,
{
    pub ty: I,
    pub sym: Sym<T>,
    pub _p: PhantomData<R>,
    pub _s: PhantomData<S>,
}

/// allow type erasure on S
impl<T, R, I, S> AsRef<Node<T, R, I, ()>> for Node<T, R, I, S>
where
    T: EgglogTy,
    R: SingletonGetter,
    I: NodeInner<T>,
    S: EgglogEnumVariantTy,
{
    fn as_ref(&self) -> &Node<T, R, I, ()> {
        // Safety notes:
        // 1. Node's memory layout is unaffected by PhantomData
        // 2. We're only changing the S type parameter from a concrete type to unit type (),
        //    which doesn't affect the actual data
        unsafe { &*(self as *const Node<T, R, I, S> as *const Node<T, R, I, ()>) }
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Sym<T = ()> {
    pub inner: GlobalSymbol,
    pub p: PhantomData<T>,
}

impl<T> Sym<T> {
    pub fn new(global_sym: GlobalSymbol) -> Self {
        Self {
            inner: global_sym,
            p: PhantomData,
        }
    }
    pub fn as_str(&self) -> &'static str {
        self.inner.as_str()
    }
    pub fn to_string(&self) -> String {
        self.inner.as_str().to_string()
    }
}
impl<T: std::clone::Clone> Copy for Sym<T> {}
impl<T> Clone for Sym<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            p: PhantomData,
        }
    }
}

pub trait NodeInner<T> {}
impl<T> std::fmt::Display for Sym<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.inner.as_str())
    }
}
impl<T> From<Sym<T>> for &str {
    fn from(value: Sym<T>) -> Self {
        value.inner.as_str()
    }
}
impl<T: EgglogTy> From<Syms<T>> for Syms {
    fn from(value: Syms<T>) -> Self {
        value.into_iter().map(|s| s.erase()).collect()
    }
}
/// count the number of nodes of specific EgglogTy for specific binding Tx
pub struct TyCounter<T: EgglogTy> {
    counter: AtomicU32,
    t: PhantomData<T>,
}
impl<T: EgglogTy> TyCounter<T> {
    pub const fn new() -> Self {
        TyCounter {
            counter: AtomicU32::new(0),
            t: PhantomData,
        }
    }
    // get next symbol of specified type T
    pub fn next_sym(&self) -> Sym<T> {
        Sym {
            inner: format!("{}{}", T::TY_NAME_LOWER, self.inc()).into(),
            p: PhantomData::<T>,
        }
    }
    pub fn get_counter(&self) -> u32 {
        self.counter.load(std::sync::atomic::Ordering::Acquire)
    }
    /// counter increment atomically
    pub fn inc(&self) -> u32 {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
    }
}

impl EgglogEnumVariantTy for () {
    const TY_NAME: &'static str = "Unknown";
}

#[derive(DerefMut, Deref)]
pub struct WorkAreaNode {
    pub next: Option<Sym>,
    pub prev: Option<Sym>,
    pub preds: Syms,
    #[deref]
    #[deref_mut]
    pub egglog: Box<dyn EgglogNode>,
}

impl Clone for WorkAreaNode {
    fn clone(&self) -> Self {
        Self {
            next: self.next.clone(),
            preds: self.preds.clone(),
            egglog: self.egglog.clone_dyn(),
            prev: None,
        }
    }
}
impl fmt::Debug for WorkAreaNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkAreaNode")
            .field("preds", &self.preds)
            .field("prev", &self.prev)
            .field("sym", &self.egglog.cur_sym())
            .field("succs", &self.egglog.succs())
            .finish()
    }
}
impl WorkAreaNode {
    pub fn new(node: Box<dyn EgglogNode>) -> Self {
        Self {
            preds: Syms::default(),
            egglog: node,
            next: None,
            prev: None,
        }
    }
    pub fn succs_mut(&mut self) -> impl Iterator<Item = &mut Sym> {
        self.egglog.succs_mut().into_iter()
    }
    pub fn preds_mut(&mut self) -> impl Iterator<Item = &mut Sym> {
        self.preds.iter_mut()
    }
    pub fn preds(&self) -> impl Iterator<Item = &Sym> {
        self.preds.iter()
    }
}

impl Borrow<GlobalSymbol> for Sym {
    fn borrow(&self) -> &GlobalSymbol {
        &self.inner
    }
}

#[derive(Clone, Deref, DerefMut, IntoIterator, Debug, Default)]
pub struct Syms<T = ()> {
    #[into_iterator(owned, ref, ref_mut)]
    inner: SmallVec<[Sym<T>; 4]>,
}

impl From<SmallVec<[Sym; 4]>> for Syms {
    fn from(value: SmallVec<[Sym; 4]>) -> Self {
        Syms { inner: value }
    }
}

impl<S> FromIterator<Sym<S>> for Syms<S> {
    fn from_iter<T: IntoIterator<Item = Sym<S>>>(iter: T) -> Self {
        Syms {
            inner: iter.into_iter().collect(),
        }
    }
}
impl Syms {
    pub fn new() -> Self {
        Syms {
            inner: SmallVec::new(),
        }
    }
}

/// global commit
/// This trait should be implemented for Tx singleton
/// usage:
/// ```rust
/// let last_version_node = node.clone();
/// Tx::commit(&self, node);
/// ```
pub trait TxCommit {
    fn on_commit<T: EgglogNode>(&self, node: &T);
    fn on_stage<T: EgglogNode + ?Sized>(&self, node: &T);
}

pub trait TxCommitSgl {
    fn on_commit<T: EgglogNode>(node: &T);
    fn on_stage<T: EgglogNode>(node: &T);
}

impl<Ret, S> TxCommitSgl for S
where
    Ret: Tx + VersionCtl + TxCommit,
    S: SingletonGetter<RetTy = Ret>,
{
    fn on_commit<T: EgglogNode>(node: &T) {
        S::sgl().on_commit(node);
    }

    fn on_stage<T: EgglogNode>(node: &T) {
        S::sgl().on_stage(node);
    }
}

/// single node commit
/// This trait should be implemented for Node
/// usage:
/// ```rust
/// let last_version_node = node.clone();
/// node.set_a()
///     .set_b()
///     .commit();
/// ```
pub trait Commit {
    fn commit(&self);
    fn stage(&self);
}

/// In Egglog there are 2 ways to interact with egraph
/// 1. String of egglog code
/// 2. Vector of Egglog Command Struct
/// Use this Interpreter trait to concile them
///
/// Also there are
pub trait Interpreter {
    type Interpreted;
    fn interpret(interpreted: Self::Interpreted);
}

// pub trait EgglogNodeMarker{ }

impl<T: EgglogNode> From<T> for WorkAreaNode {
    fn from(value: T) -> Self {
        WorkAreaNode::new(value.clone_dyn())
    }
}

/// Trait for input types that can be used in egglog functions
pub trait EgglogFuncInput {
    type Ref<'a>: EgglogFuncInputRef;
    fn as_node(&self) -> &dyn EgglogNode;
}
/// Trait for input tuple that can be used in egglog functions
pub trait EgglogFuncInputs {
    type Ref<'a>: EgglogFuncInputsRef;
    fn as_nodes(&self) -> Box<[&dyn EgglogNode]>;
}
/// Trait for input types ref that directly used as function argument
pub trait EgglogFuncInputRef {
    type DeRef: EgglogFuncInput + EgglogNode;
    fn as_node(&self) -> &dyn EgglogNode;
}
pub trait EgglogFuncInputsRef {
    type DeRef: EgglogFuncInputs;
    fn as_nodes(&self) -> Box<[&dyn EgglogNode]>;
}

/// Trait for output types that can be used in egglog functions
pub trait EgglogFuncOutput: 'static {
    type Ref<'a>: EgglogFuncOutputRef;
    fn as_node(&self) -> &dyn EgglogNode;
}
impl<T> EgglogFuncOutput for T
where
    T: EgglogNode + 'static,
{
    type Ref<'a> = &'a dyn AsRef<T>;
    fn as_node(&self) -> &dyn EgglogNode {
        self
    }
}
impl<T: EgglogFuncOutput + EgglogNode + 'static> EgglogFuncOutputRef for &dyn AsRef<T> {
    type DeRef<'a> = T;
    fn as_node(&self) -> &dyn EgglogNode {
        self.as_ref()
    }
}
pub trait EgglogFuncOutputRef {
    type DeRef<'a>: EgglogFuncOutput;
    fn as_node(&self) -> &dyn EgglogNode;
}
pub trait EgglogFunc {
    type Input: EgglogFuncInputs;
    type Output: EgglogFuncOutput;
    const FUNC_NAME: &'static str;
}
impl<T> EgglogFuncInput for T
where
    T: EgglogNode + 'static,
{
    type Ref<'a> = &'a dyn AsRef<T>;
    fn as_node(&self) -> &dyn EgglogNode {
        self
    }
}
impl<T> EgglogFuncInputRef for &dyn AsRef<T>
where
    T: EgglogNode + 'static,
{
    type DeRef = T;
    fn as_node(&self) -> &dyn EgglogNode {
        self.as_ref()
    }
}

pub fn collect_type_defs() -> Vec<Command> {
    let mut commands = vec![];
    // split decls to avoid undefined sort
    let mut types = Vec::<(Span, String, Subdatatypes)>::new();
    for decl in inventory::iter::<Decl> {
        match decl {
            Decl::EgglogBaseTy { name, cons } => {
                types.push((
                    span!(),
                    name.to_string(),
                    Subdatatypes::Variants(
                        cons.iter()
                            .map(|x| Variant {
                                span: span!(),
                                name: x.cons_name.to_string(),
                                types: x.input.iter().map(|y| y.to_string()).collect(),
                                cost: x.cost,
                            })
                            .collect(),
                    ),
                ));
            }
            Decl::EgglogVecTy { name, ele_ty } => {
                let ele_ty = ele_ty.to_owned();
                let ele = crate::var!(ele_ty);
                types.push((
                    span!(),
                    name.to_string(),
                    Subdatatypes::NewSort("Vec".to_string(), vec![ele]),
                ));
            }
            _ => {
                // do nothing
            }
        }
    }
    commands.push(Command::Datatypes {
        span: span!(),
        datatypes: types,
    });
    for decl in inventory::iter::<Decl> {
        match decl {
            Decl::EgglogFuncTy {
                name,
                input,
                output,
            } => {
                commands.push(Command::Function {
                    span: span!(),
                    name: name.to_string(),
                    schema: Schema {
                        input: input.iter().map(<&str>::to_string).collect(),
                        output: output.to_string(),
                    },
                    merge: Some(GenericExpr::Var(span!(), "new".to_owned())),
                });
            }
            _ => {}
        }
    }
    commands
}

macro_rules! impl_input_for_tuples {
    () => {
        #[allow(unused)]
        impl EgglogFuncInputs for () {
            type Ref<'a> = ();
            fn as_nodes(&self) -> Box<[&dyn EgglogNode]> {
                Box::new([])
            }
        }
    };
    ($($T:ident),*) => {
        #[allow(unused)]
        impl<$($T: EgglogNode + EgglogFuncInput),*> EgglogFuncInputs for ($($T,)*) {
            type Ref<'a> = ($($T::Ref<'a>,)*);

            fn as_nodes(&self) -> Box<[&dyn EgglogNode]> {
                #[allow(non_snake_case)]
                let ($($T,)*) = self;
                Box::new([$($T),*])
            }
        }
    };
}

// Continue for as many tuple sizes as needed
macro_rules! impl_input_ref_for_tuples {
    () => {
        #[allow(unused)]
        impl EgglogFuncInputsRef for () {
            type DeRef = ();
            fn as_nodes(&self) -> Box<[&dyn EgglogNode]> {
                Box::new([])
            }
        }
    };

    ($($T:ident),*) => {
        #[allow(unused)]
        impl<$($T: EgglogFuncInputRef),*> EgglogFuncInputsRef for ($($T,)*) {
            type DeRef = ($($T::DeRef,)*);
            #[allow(non_snake_case)]
            fn as_nodes(&self) -> Box<[&dyn EgglogNode]> {
                let ($($T,)*) = self;
                Box::new([$($T.as_node()),*])
            }
        }
    };
}

macro_rules! impl_for_tuples {
    ($($T:ident),*) => {
        impl_input_for_tuples!($($T),*);
        impl_input_ref_for_tuples!($($T),*);
    };
}

// Generate implementations
impl_for_tuples!();
impl_for_tuples!(T0);
impl_for_tuples!(T0, T1);
impl_for_tuples!(T0, T1, T2);
impl_for_tuples!(T0, T1, T2, T3);
impl_for_tuples!(T0, T1, T2, T3, T4);
impl_for_tuples!(T0, T1, T2, T3, T4, T5);
impl_for_tuples!(T0, T1, T2, T3, T4, T5, T6);
impl_for_tuples!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_for_tuples!(T0, T1, T2, T3, T4, T5, T6, T7, T8);

pub trait EgglogVecTy: EgglogTy {
    type EleTy: EgglogTy;
}
pub trait EgglogBaseTy: EgglogTy {
    const CONSTRUCTORS: TyConstructors;
}

#[derive(Deref)]
pub struct TyConstructors(pub &'static [TyConstructor]);

pub enum Decl {
    EgglogBaseTy {
        name: &'static str,
        cons: &'static TyConstructors,
    },
    EgglogVecTy {
        name: &'static str,
        ele_ty: &'static str,
    },
    EgglogFuncTy {
        name: &'static str,
        input: &'static [&'static str],
        output: &'static str,
    },
}

impl EgglogTy for i64 {
    const TY_NAME: &'static str = "i64";
    const TY_NAME_LOWER: &'static str = "i64";
}
impl EgglogTy for String {
    const TY_NAME: &'static str = "string";
    const TY_NAME_LOWER: &'static str = "string";
}

pub trait ToVar {
    fn to_var(&self) -> GenericExpr<&'static str, &'static str>;
}
impl ToVar for i64 {
    fn to_var(&self) -> GenericExpr<&'static str, &'static str> {
        GenericExpr::Lit(span!(), crate::ast::Literal::Int(*self))
    }
}
impl ToVar for f64 {
    fn to_var(&self) -> GenericExpr<&'static str, &'static str> {
        GenericExpr::Lit(
            span!(),
            crate::ast::Literal::Float(ordered_float::OrderedFloat::<f64>(*self)),
        )
    }
}
impl ToVar for String {
    fn to_var(&self) -> GenericExpr<&'static str, &'static str> {
        GenericExpr::Lit(span!(), crate::ast::Literal::String(self.to_owned()))
    }
}
impl ToVar for bool {
    fn to_var(&self) -> GenericExpr<&'static str, &'static str> {
        GenericExpr::Lit(span!(), crate::ast::Literal::Bool(*self))
    }
}
impl<T> ToVar for Sym<T> {
    fn to_var(&self) -> GenericExpr<&'static str, &'static str> {
        GenericExpr::Var(span!(), self.inner.into())
    }
}

pub trait ToOwnedStr {
    fn to_owned_str(&self) -> GenericExpr<String, String>;
}

impl ToOwnedStr for GenericExpr<&'static str, &'static str> {
    fn to_owned_str(&self) -> GenericExpr<String, String> {
        match self {
            GenericExpr::Lit(span, literal) => GenericExpr::Lit(span.clone(), literal.clone()),
            GenericExpr::Var(span, v) => GenericExpr::Var(span.clone(), v.to_string()),
            GenericExpr::Call(span, h, generic_exprs) => GenericExpr::Call(
                span.clone(),
                h.to_string(),
                generic_exprs.iter().map(|x| x.to_owned_str()).collect(),
            ),
        }
    }
}

impl From<&'static Location<'static>> for Span{
    fn from(value: &'static Location) -> Self {
        Span::Rust(Arc::new(RustSpan{ file: value.file(), line: value.line(), column: value.column() }))
    }
}