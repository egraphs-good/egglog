use crate::*;

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("Arity mismatch, expected {expected} args: {expr}")]
    Arity { expr: Expr, expected: usize },
    #[error(
        "Type mismatch: expr = {expr}, expected = {expected}, actual = {actual}, reason: {reason}"
    )]
    Mismatch {
        expr: Expr,
        expected: Type,
        actual: Type,
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
    NoMatchingPrimitive { op: Symbol, inputs: Vec<Type> },
}

pub struct Context<'a> {
    pub egraph: &'a EGraph,
    pub types: HashMap<Symbol, Type>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atom {
    Func(Symbol, Vec<AtomTerm>),
    Prim(Primitive, Vec<AtomTerm>),
}

impl Atom {
    pub fn vars(&self) -> impl Iterator<Item = Symbol> + '_ {
        match self {
            Atom::Func(_, terms) | Atom::Prim(_, terms) => terms.iter().filter_map(|t| match t {
                AtomTerm::Var(v) => Some(*v),
                AtomTerm::Value(_) => None,
            }),
        }
    }
}
pub type Bindings = HashMap<Symbol, AtomTerm>;

impl<'a> Context<'a> {
    pub fn new(egraph: &'a EGraph) -> Self {
        Self {
            egraph,
            types: HashMap::default(),
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
    ) -> Result<(Vec<Atom>, Bindings), Vec<TypeError>> {
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
                    let old = leaves.insert(id, AtomTerm::Value(lit.to_value()));
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

        let mut atoms = vec![];
        // Now we can fill in the nodes with the canonical leaves
        for (node, id) in &self.nodes {
            match node {
                ENode::Func(f, ids) => atoms.push(Atom::Func(
                    *f,
                    ids.iter().chain([id]).map(&get_leaf).collect(),
                )),
                ENode::Prim(p, ids) => atoms.push(Atom::Prim(
                    p.clone(),
                    ids.iter().chain([id]).map(&get_leaf).collect(),
                )),
                _ => {}
            }
        }

        if self.errors.is_empty() {
            let mut bindings = Bindings::default();
            for (node, id) in &self.nodes {
                if let ENode::Var(var) = node {
                    bindings.insert(*var, leaves[id].clone());
                }
            }
            Ok((atoms, bindings))
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

    fn typecheck_fact(&mut self, fact: &'a Fact) {
        match fact {
            Fact::Eq(exprs) => {
                let mut later = vec![];
                let mut ty: Option<Type> = None;
                let mut ids = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    match (expr, &ty) {
                        (_, Some(expected)) => {
                            ids.push(self.check_query_expr(expr, expected.clone()))
                        }
                        // This is a variable the we couldn't infer the type of,
                        // so we'll try again later when we can check its type
                        (Expr::Var(v), None) if !self.types.contains_key(v) => later.push(expr),
                        (_, None) => match self.infer_query_expr(expr) {
                            (_, Type::Error) => (),
                            (id, t) => {
                                ty = Some(t);
                                ids.push(id);
                            }
                        },
                    }
                }

                if let Some(ty) = ty {
                    assert_ne!(ty, Type::Error);
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
                self.check_query_expr(e, Type::Unit);
            }
        }
    }

    fn check_query_expr(&mut self, expr: &'a Expr, expected: Type) -> Id {
        assert_ne!(expected, Type::Error);
        if let Expr::Var(sym) = expr {
            match self.types.entry(*sym) {
                Entry::Occupied(ty) => {
                    if ty.get() != &expected {
                        self.errors.push(TypeError::Mismatch {
                            expr: expr.clone(),
                            expected,
                            actual: ty.get().clone(),
                            reason: "mismatch".into(),
                        })
                    }
                }
                // we can actually bind the variable here
                Entry::Vacant(entry) => {
                    entry.insert(expected);
                }
            }
            self.add_node(ENode::Var(*sym))
        } else {
            let (id, actual) = self.infer_query_expr(expr);
            if actual != expected {
                self.errors.push(TypeError::Mismatch {
                    expr: expr.clone(),
                    expected,
                    actual,
                    reason: "mismatch".into(),
                })
            }
            id
        }
    }

    fn infer_query_expr(&mut self, expr: &'a Expr) -> (Id, Type) {
        match expr {
            // TODO handle global variables
            Expr::Var(sym) => {
                let ty = if let Some(ty) = self.types.get(sym) {
                    ty.clone()
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    Type::Error
                };
                (self.add_node(ENode::Var(*sym)), ty)
            }
            Expr::Lit(lit) => {
                let t = match lit {
                    Literal::Int(_) => Type::NumType(NumType::I64),
                    Literal::Bool(_) => Type::Bool,
                    Literal::Rational(_) => Type::NumType(NumType::Rational),
                    Literal::String(_) => Type::String,
                    Literal::Unit => Type::Unit,
                };
                (self.add_node(ENode::Literal(lit.clone())), t)
            }
            Expr::Call(sym, args) => {
                if let Some(f) = self.egraph.functions.get(sym) {
                    if f.decl.schema.input.len() != args.len() {
                        self.errors.push(TypeError::Arity {
                            expr: expr.clone(),
                            expected: f.decl.schema.input.len(),
                        });
                    }

                    let ids: Vec<Id> = args
                        .iter()
                        .zip(&f.decl.schema.input)
                        .map(|(arg, ty)| self.check_query_expr(arg, ty.clone()))
                        .collect();
                    let t = f.decl.schema.output.clone();
                    (self.add_node(ENode::Func(*sym, ids)), t)
                } else if let Some(prims) = self.egraph.primitives.get(sym) {
                    let (ids, arg_tys): (Vec<Id>, Vec<Type>) =
                        args.iter().map(|arg| self.infer_query_expr(arg)).unzip();
                    for prim in prims {
                        if let Some(output_type) = prim.accept(&arg_tys) {
                            let id = self.add_node(ENode::Prim(prim.clone(), ids));
                            return (id, output_type);
                        }
                    }
                    // No need to push this error if the argument types are wrong
                    if !arg_tys.contains(&Type::Error) {
                        self.errors.push(TypeError::NoMatchingPrimitive {
                            op: *sym,
                            inputs: arg_tys,
                        });
                    }
                    (self.unionfind.make_set(), Type::Error)
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    (self.unionfind.make_set(), Type::Error)
                }
            }
        }
    }
}
