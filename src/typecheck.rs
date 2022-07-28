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
        expected: OutputType,
        actual: OutputType,
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
    #[error("Failed to infer a type for expression: {0}")]
    InferenceFailure(Expr),
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum ENode {
    Literal(Literal),
    Var(Symbol),
    Node(Symbol, Vec<Id>),
}

struct QueryBuilder<'a> {
    // bindings: IndexMap<Symbol, AtomTerm>,
    // atoms: Vec<Atom>,
    unionfind: UnionFind<Info<'a>>,
    nodes: HashMap<ENode, Id>,
    // foo: UnionFind<Info<'a>>,
    errors: Vec<TypeError>,
    egraph: &'a EGraph,
}

#[derive(Clone)]
struct Info<'a> {
    ty: Option<OutputType>,
    expr: &'a Expr,
}

impl<'a> UnifyValue for Info<'a> {
    type Error = (Self, Self);

    fn merge(a: &Self, b: &Self) -> Result<Self, Self::Error> {
        let ty = match (&a.ty, &b.ty) {
            (None, None) => None,
            (None, Some(b)) => Some(b.clone()),
            (Some(a), None) => Some(a.clone()),
            (Some(at), Some(bt)) => {
                if at == bt {
                    Some(at.clone())
                } else {
                    return Err((a.clone(), b.clone()));
                }
            }
        };
        Ok(Info { ty, expr: a.expr })
    }
}

impl<'a> QueryBuilder<'a> {
    fn rebuild(&mut self) {
        let mut keep_going = true;
        while keep_going {
            keep_going = false;
            let nodes = std::mem::take(&mut self.nodes);
            for (mut node, id) in nodes {
                // canonicalize
                let id = self.unionfind.find_mut(id);
                if let ENode::Node(_, children) = &mut node {
                    for child in children {
                        *child = self.unionfind.find_mut(*child);
                    }
                }

                // reinsert and handle hit
                if let Some(old) = self.nodes.insert(node, id) {
                    keep_going = true;
                    self.unify(old, id);
                }
            }
        }
    }

    fn unify(&mut self, id1: Id, id2: Id) -> Id {
        if let Err((a, b)) = self.unionfind.try_union(id1, id2) {
            // TODO unification error?
            self.errors.push(TypeError::Mismatch {
                expr: a.expr.clone(),
                expected: a.ty.unwrap_or(OutputType::Unit),
                actual: b.ty.unwrap_or(OutputType::Unit),
                reason: "unification".into(),
            })
        }
        id1
    }

    fn unify_info(&mut self, id: Id, info: Info<'a>) {
        if let Err((old_info, new_info)) = self.unionfind.try_insert(id, info) {
            let expect = "unification can't fail on None types";
            self.errors.push(TypeError::Mismatch {
                expr: new_info.expr.clone(),
                expected: new_info.ty.expect(expect),
                actual: old_info.ty.expect(expect),
                reason: "mismatch".into(),
            })
        }
    }

    fn add_fact(&mut self, fact: &'a Fact) -> Id {
        match fact {
            Fact::Eq(exprs) => {
                assert!(exprs.len() > 1);
                let mut iter = exprs.iter();
                let mut id = self.add_expr(iter.next().unwrap());
                for e in iter {
                    let id2 = self.add_expr(e);
                    id = self.unify(id, id2);
                }
                id
            }
            Fact::Fact(e) => self.add_expr_at(e, OutputType::Unit),
        }
    }

    fn add_node(&mut self, node: ENode, info: Info<'a>) -> Id {
        match self.nodes.entry(node) {
            Entry::Occupied(e) => {
                let id = *e.get();
                self.unify_info(id, info);
                id
            }
            Entry::Vacant(e) => {
                let id = self.unionfind.make_set_with(info);
                *e.insert(id)
            }
        }
    }

    fn add_expr_at(&mut self, expr: &'a Expr, ty: OutputType) -> Id {
        let id = self.add_expr(expr);
        let info = Info { ty: Some(ty), expr };
        self.unify_info(id, info);
        id
    }

    fn add_expr(&mut self, expr: &'a Expr) -> Id {
        match expr {
            Expr::Lit(lit) => {
                let ty = Some(match lit {
                    Literal::Int(_) => OutputType::Type(InputType::NumType(NumType::I64)),
                    Literal::String(_) => OutputType::Type(InputType::String),
                    Literal::Rational(_) => OutputType::Type(InputType::NumType(NumType::Rational)),
                });
                self.add_node(ENode::Literal(lit.clone()), Info { ty, expr })
            }
            Expr::Var(var) => {
                // TODO handle constants?
                // FIXME no! constants are distinct from nullary partial functions
                self.add_node(ENode::Var(*var), Info { ty: None, expr })
            }
            Expr::Call(sym, args) => {
                let mut ids = vec![];
                let ty = if let Some(f) = self.egraph.functions.get(sym) {
                    if args.len() == f.schema.input.len() {
                        for (arg, ty) in args.iter().zip(&f.schema.input) {
                            ids.push(self.add_expr_at(arg, OutputType::Type(ty.clone())));
                        }
                    } else {
                        // arity mismatch, don't worry about constraining the inputs
                        self.errors.push(TypeError::Arity {
                            expr: expr.clone(),
                            expected: f.schema.input.len(),
                        });
                    }

                    // if we haven't pushed type constrainted arguments,
                    // add some fake stuff for now
                    if ids.len() != args.len() {
                        assert!(ids.is_empty());
                        for arg in args {
                            ids.push(self.add_expr(arg));
                        }
                    }
                    assert_eq!(ids.len(), args.len());
    
                    Some(f.schema.output.clone())
                } else if let Some(preds) = self.egraph.predicates.get(sym) {
                    // We assume the type of arguments is known at the time of inference.
                    // This type inference algorithm should be refactored in the future.
                    let mut inptypes = vec![];
                    let mut failed = false;
                    for arg in args {
                        let id = self.add_expr(arg);
                        ids.push(id);
                        if let Some(ty) = self.unionfind.get_value(id).ty.clone() {
                            inptypes.push(ty);
                        } else {
                            self.errors.push(TypeError::InferenceFailure(arg.clone()));
                            failed = true;
                        }
                    }
                    if failed {
                        None
                    } else {
                        let mut res = None;
                        for pred in preds {
                            if pred.args.len() != inptypes.len() {
                                continue;
                            }
                            if pred.args.iter().zip(inptypes.iter()).all(|(t1, t2)| match t2 {
                                OutputType::Type(t2) => t1 == t2,
                                _ => false
                            }) {
                                res = Some(OutputType::Unit);
                                break;
                            }
                        }
                        res
                    }
                } else {
                    self.errors.push(TypeError::Unbound(*sym));
                    None
                };
                let enode = ENode::Node(*sym, ids);
                log::info!("add enode {:?}", enode);
                self.add_node(enode, Info { ty, expr })
            }
        }
    }
}

impl EGraph {
    pub(crate) fn compile_query(&self, facts: Vec<Fact>) -> Result<Query, Error> {
        let mut builder = QueryBuilder {
            unionfind: Default::default(),
            nodes: Default::default(),
            errors: Default::default(),
            egraph: self,
        };

        for fact in &facts {
            builder.add_fact(fact);
        }

        builder.rebuild();
        let mut query = Query::default();

        #[derive(Default, Clone)]
        struct Class {
            vars: Vec<Symbol>,
            lits: Vec<Literal>,
            nodes: Vec<(Symbol, Vec<Id>)>,
            atomterm: Option<AtomTerm>,
        }
        let mut classes = HashMap::<Id, Class>::default(); // vec![Class::default(); builder.unionfind.len()];
        for (node, id) in builder.nodes {
            // let class = &mut classes[usize::fom(id)];
            let class = classes.entry(id).or_default();
            match node {
                ENode::Literal(l) => class.lits.push(l),
                ENode::Var(v) => class.vars.push(v),
                ENode::Node(s, ids) => class.nodes.push((s, ids)),
            }
        }

        let mut next_index_var = 0;
        for class in classes.values_mut() {
            assert!(class.lits.len() + class.vars.len() + class.nodes.len() > 0);
            let atomterm = if let Some(lit) = class.lits.first() {
                if class.lits.len() > 1 {
                    builder
                        .errors
                        .push(TypeError::TooManyLiterals(class.lits.clone()));
                }
                AtomTerm::Value(lit.to_value())
            } else {
                let i = next_index_var;
                next_index_var += 1;
                AtomTerm::Var(i)
            };
            assert!(class.atomterm.is_none());
            class.atomterm = Some(atomterm);
        }

        // let atomterms: Vec<AtomTerm> = classes.values().map(|class| {}).collect();

        // assert_eq!(classes.len(), atomterms.len());
        // for (i, (class, atomterm)) in classes.into_iter().zip(&atomterms).enumerate() {
        for (&id, class) in &classes {
            let info = builder.unionfind.get_value(id);
            let atomterm = class.atomterm.clone().unwrap();
            for &var in &class.vars {
                // TODO do something with the type
                let _ty = if let Some(ty) = info.ty.clone() {
                    if ty == OutputType::Unit {
                        builder.errors.push(TypeError::UnitVar(var));
                    }
                    ty
                } else {
                    builder.errors.push(TypeError::InferenceFailure(Expr::Var(var)));
                    OutputType::Unit
                };
                query.bindings.insert(var, atomterm.clone());
            }

            for (sym, children) in &class.nodes {
                let mut terms: Vec<AtomTerm> = children
                    .iter()
                    .map(|c| classes[c].atomterm.clone().unwrap())
                    .collect();
                // predicates "relations" do not have the output column.
                if !self.predicates.contains_key(sym) {
                    terms.push(atomterm.clone());
                }
                query.atoms.push(Atom(*sym, terms))
            }
        }

        if builder.errors.is_empty() {
            log::debug!("Compiled {facts:?} to {query:?}");
            Ok(query)
        } else {
            Err(Error::TypeErrors(builder.errors))
        }
    }
}
