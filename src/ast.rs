use crate::*;

#[derive(Debug)]
pub enum Command {
    Datatype {
        name: Symbol,
        variants: Vec<Variant>,
    },
    Function(Symbol, Schema),
    Rule(Option<Symbol>, Rule),
    Action(Action),
    Run(usize),
    Extract(Expr),
    CheckEq(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: Symbol,
    pub types: Vec<Type>,
}

#[derive(Clone, Debug)]
pub enum Action {
    Define(Symbol, Expr),
    Union(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Sort(Symbol),
    Int,
}

impl Type {
    pub fn is_sort(&self) -> bool {
        matches!(self, Self::Sort(..))
    }
}

#[derive(Clone, Debug)]
pub struct Schema {
    pub input: Vec<Type>,
    pub output: Type,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum Expr<T = Value> {
    Leaf(T),
    Var(Symbol),
    Node(Symbol, Vec<Self>),
}

impl<T> Expr<T> {
    pub fn new(op: impl Into<Symbol>, children: impl IntoIterator<Item = Self>) -> Self {
        Self::Node(op.into(), children.into_iter().collect())
    }

    pub fn leaf(op: impl Into<T>) -> Self {
        Self::Leaf(op.into())
    }

    pub fn get_var(&self) -> Option<Symbol> {
        match self {
            Expr::Var(v) => Some(*v),
            _ => None,
        }
    }

    fn children(&self) -> &[Self] {
        match self {
            Expr::Var(_) | Expr::Leaf(_) => &[],
            Expr::Node(_, children) => children,
        }
    }

    pub fn walk(&self, pre: &mut impl FnMut(&Self), post: &mut impl FnMut(&Self)) {
        pre(self);
        self.children()
            .iter()
            .for_each(|child| child.walk(pre, post));
        post(self);
    }

    pub fn fold<Out>(&self, f: &mut impl FnMut(&Self, Vec<Out>) -> Out) -> Out {
        let ts = self.children().iter().map(|child| child.fold(f)).collect();
        f(self, ts)
    }
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub query: Query,
    pub actions: Vec<Action>,
}

impl Rule {
    pub fn rewrite(lhs: Pattern, rhs: Pattern) -> Self {
        let root = Symbol::from("__root");
        let query = Query::from_patterns(vec![(root, lhs)]);
        let actions = vec![Action::Union(vec![Expr::Var(root), rhs])];
        Self { query, actions }
    }
}
