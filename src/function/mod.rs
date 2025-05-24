use crate::*;

#[derive(Clone)]
pub struct Function {
    pub(crate) decl: ResolvedFunctionDecl,
    pub schema: ResolvedSchema,
    pub(crate) can_subsume: bool,
    pub new_backend_id: egglog_bridge::FunctionId,
}

#[derive(Clone, Debug)]
pub struct ResolvedSchema {
    pub input: Vec<ArcSort>,
    pub output: ArcSort,
}

impl ResolvedSchema {
    pub fn get_by_pos(&self, index: usize) -> Option<&ArcSort> {
        if self.input.len() == index {
            Some(&self.output)
        } else {
            self.input.get(index)
        }
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("decl", &self.decl)
            .field("schema", &self.schema)
            .finish()
    }
}

impl Function {
    pub(crate) fn new(egraph: &mut EGraph, decl: &ResolvedFunctionDecl) -> Result<Self, Error> {
        let mut input = Vec::with_capacity(decl.schema.input.len());
        for s in &decl.schema.input {
            input.push(match egraph.type_info.get_sort_by_name(s) {
                Some(sort) => sort.clone(),
                None => {
                    return Err(Error::TypeError(TypeError::UndefinedSort(
                        *s,
                        decl.span.clone(),
                    )))
                }
            })
        }

        let output = match egraph.type_info.get_sort_by_name(&decl.schema.output) {
            Some(sort) => sort.clone(),
            None => {
                return Err(Error::TypeError(TypeError::UndefinedSort(
                    decl.schema.output,
                    decl.span.clone(),
                )))
            }
        };

        let can_subsume = match decl.subtype {
            FunctionSubtype::Constructor => true,
            FunctionSubtype::Relation => true,
            FunctionSubtype::Custom => false,
        };

        let new_backend_id = {
            use egglog_bridge::{DefaultVal, MergeFn};
            let schema = input
                .iter()
                .chain([&output])
                .map(|sort| sort.column_ty(&egraph.backend))
                .collect();
            let default = match decl.subtype {
                FunctionSubtype::Constructor => DefaultVal::FreshId,
                FunctionSubtype::Custom => DefaultVal::Fail,
                FunctionSubtype::Relation => DefaultVal::Const(egraph.backend.primitives().get(())),
            };
            let merge = match decl.subtype {
                FunctionSubtype::Constructor => MergeFn::UnionId,
                FunctionSubtype::Relation => MergeFn::AssertEq,
                FunctionSubtype::Custom => match &decl.merge {
                    None => MergeFn::AssertEq,
                    Some(expr) => translate_expr_to_mergefn(expr, egraph)?,
                },
            };
            let name = decl.name.to_string();
            egraph.backend.add_table(egglog_bridge::FunctionConfig {
                schema,
                default,
                merge,
                name,
                can_subsume,
            })
        };

        Ok(Function {
            decl: decl.clone(),
            schema: ResolvedSchema { input, output },
            can_subsume,
            new_backend_id,
        })
    }
}

fn translate_expr_to_mergefn(
    expr: &GenericExpr<ResolvedCall, ResolvedVar>,
    egraph: &mut EGraph,
) -> Result<egglog_bridge::MergeFn, Error> {
    match expr {
        GenericExpr::Lit(_, literal) => {
            let val = literal_to_value(&egraph.backend, literal);
            Ok(egglog_bridge::MergeFn::Const(val))
        }
        GenericExpr::Var(span, resolved_var) => match resolved_var.name.as_str() {
            "old" => Ok(egglog_bridge::MergeFn::Old),
            "new" => Ok(egglog_bridge::MergeFn::New),
            // NB: type-checking should already catch unbound variables here.
            _ => Err(TypeError::Unbound(resolved_var.name, span.clone()).into()),
        },
        GenericExpr::Call(_, ResolvedCall::Func(f), args) => {
            let translated_args = args
                .iter()
                .map(|arg| translate_expr_to_mergefn(arg, egraph))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(egglog_bridge::MergeFn::Function(
                egraph.functions[&f.name].new_backend_id,
                translated_args,
            ))
        }
        GenericExpr::Call(_, ResolvedCall::Primitive(p), args) => {
            let translated_args = args
                .iter()
                .map(|arg| translate_expr_to_mergefn(arg, egraph))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(egglog_bridge::MergeFn::Primitive(
                p.primitive.1,
                translated_args,
            ))
        }
    }
}
