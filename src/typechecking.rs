use crate::*;

#[derive(Default)]
pub struct TypeInfo {
    pub func_types: HashMap<Symbol, Schema>,
    pub global_types: HashMap<Symbol, Symbol>,
    pub local_types: HashMap<Command, HashMap<Symbol, Symbol>>,
}
/*
impl TypeInfo {
  pub(crate) fn typecheck_program(egraph: &EGraph, program: &Vec<NormCommand>) -> Result<Self, TypeError> {
    let mut type_info = TypeInfo::default();
    for command in program {
      type_info.typecheck_command(egraph, command)?;
    }
    Ok(type_info)
  }

  pub(crate) fn typecheck_command(&mut self, egraph: &EGraph, command: &NormCommand) ->  Result<(), TypeError> {
    match command {
      NormCommand::Function(fdecl) => {
        if self.func_types.insert(fdecl.name, fdecl.schema.clone()).is_some() {
          return Err(TypeError::FunctionAlreadyBound(fdecl.name));
        }
      },
      NormCommand::Sort(..) => (),

    }
    Ok(())
  }

  fn lookup(&self, ctx: &Command, sym: Symbol) -> Result<Symbol, TypeError> {

  }

  fn typecheck_expr(&self, egraph: &EGraph, ctx: &Command, expr: &Expr) -> Result<Symbol, TypeError> {
    match expr {
        Expr::Var(sym) if !egraph.functions.contains_key(sym) => {
            match self.types.entry(*sym) {
                IEntry::Occupied(ty) => {
                    // TODO name comparison??
                    if ty.get().name() != expected.name() {
                        self.errors.push(TypeError::Mismatch {
                            expr: expr.clone(),
                            expected,
                            actual: ty.get().clone(),
                            reason: "mismatch".into(),
                        })
                    }
                }
                // we can actually bind the variable here
                IEntry::Vacant(entry) => {
                    entry.insert(expected);
                }
            }
            self.add_node(ENode::Var(*sym))
        }
        _ => {
            let (id, actual) = self.infer_query_expr(expr);
            if let Some(actual) = actual {
                if actual.name() != expected.name() {
                    self.errors.push(TypeError::Mismatch {
                        expr: expr.clone(),
                        expected,
                        actual,
                        reason: "mismatch".into(),
                    })
                }
            }
            id
        }
      }
  }
}
*/
#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("Arity mismatch, expected {expected} args: {expr}")]
    Arity { expr: Expr, expected: usize },
    #[error(
        "Type mismatch: expr = {expr}, expected = {}, actual = {}, reason: {reason}", 
        .expected.name(), .actual.name(),
    )]
    Mismatch {
        expr: Expr,
        expected: ArcSort,
        actual: ArcSort,
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
    NoMatchingPrimitive { op: Symbol, inputs: Vec<Symbol> },
    #[error("Variable {0} was already defined")]
    AlreadyDefined(Symbol),
}
