use egglog::{
    CommandMacro, Error, TypeInfo,
    ast::ResolvedFact,
    ast::{Actions, Command, Expr, ParseError, Rule, Schema},
    util::{FreshGen, IndexSet, SymbolGen},
};
use egglog_ast::generic_ast::Literal;

/// Implementation of the unstable-fresh! macro for egglog-experimental
pub struct FreshMacro;

impl FreshMacro {
    pub fn new() -> Self {
        Self
    }
}

type Symbol = String;

/// Options parsed from unstable-fresh! macro arguments
#[derive(Clone, Debug)]
struct FreshOptions {
    sort: Symbol,
    cost: Option<u64>,
    unextractable: bool,
}

impl CommandMacro for FreshMacro {
    fn transform(
        &self,
        command: Command,
        symbol_gen: &mut SymbolGen,
        type_info: &TypeInfo,
    ) -> Result<Vec<Command>, Error> {
        match command {
            Command::Rule { rule } => {
                // Collect fresh options - if none found, return the rule unchanged
                let fresh_options = collect_fresh_options(rule.head.clone())?;
                if fresh_options.is_empty() {
                    Ok(vec![Command::Rule { rule }])
                } else {
                    // unstable-fresh! requires TypeInfo for correct type inference
                    desugar_fresh_rule(rule, fresh_options, symbol_gen, type_info)
                }
            }
            _ => Ok(vec![command]),
        }
    }
}

// Main desugaring function using proper typechecking
fn desugar_fresh_rule(
    rule: Rule,
    fresh_options: Vec<FreshOptions>,
    symbol_gen: &mut SymbolGen,
    type_info: &TypeInfo,
) -> Result<Vec<Command>, Error> {
    let rule_span = rule.span.clone();

    // Typecheck the query to get resolved facts with type information
    let resolved_facts = type_info.typecheck_facts(symbol_gen, &rule.body)?;

    // Collect all unique variables from the resolved facts with their types
    let query_vars = collect_query_vars(&resolved_facts);

    // Generate unique constructor name
    let constructor_name = symbol_gen.fresh("GeneratedFreshTable").to_string();

    // Build schema for the constructor - use actual types from resolved facts
    let mut schema = Vec::new();

    // Add a column for each query variable with its actual type
    for (_, sort) in &query_vars {
        schema.push(sort.to_string());
    }

    // Add i64 for unique index
    schema.push("i64".to_string());

    // Use options from the first fresh! call for the constructor
    // (all fresh! calls in the same rule share the same constructor)
    let first_opts = &fresh_options[0];
    let output_sort = first_opts.sort.clone();

    // Create constructor function declaration
    let constructor_command = Command::Constructor {
        span: rule_span.clone(),
        name: constructor_name.clone(),
        schema: Schema {
            input: schema,
            output: output_sort,
        },
        cost: first_opts.cost,
        unextractable: first_opts.unextractable,
        hidden: false,
        let_binding: false,
        term_constructor: None,
    };

    // Get just the variable names for rewriting
    let query_var_names: Vec<String> = query_vars.iter().map(|(name, _)| name.clone()).collect();

    // Rewrite rule actions to use constructor using visit_exprs
    let mut fresh_index = 0i64;
    let new_actions = rule.head.visit_exprs(&mut |expr| {
        if let Expr::Call(span, head, _args) = &expr
            && head.as_str() == "unstable-fresh!"
        {
            let mut new_args: Vec<Expr> = query_var_names
                .iter()
                .map(|name| Expr::Var(span.clone(), name.clone()))
                .collect();
            new_args.push(Expr::Lit(span.clone(), Literal::Int(fresh_index)));
            fresh_index += 1;
            return Expr::Call(span.clone(), constructor_name.clone(), new_args);
        }
        expr
    });

    let new_rule = Rule {
        head: new_actions,
        ..rule
    };

    // Return both the constructor and the rule
    Ok(vec![constructor_command, Command::Rule { rule: new_rule }])
}

// Collect all unique variables from resolved facts with their types
fn collect_query_vars(resolved_facts: &[ResolvedFact]) -> IndexSet<(Symbol, Symbol)> {
    let mut vars = IndexSet::default();

    for fact in resolved_facts {
        fact.visit_vars(&mut |_span, resolved_var| {
            let name = resolved_var.name.to_string();
            let sort = resolved_var.sort.name().to_string();
            vars.insert((name, sort));
        });
    }

    vars
}

fn collect_fresh_options(actions: Actions) -> Result<Vec<FreshOptions>, Error> {
    let mut options_list = Vec::new();
    let mut error: Option<Error> = None;

    // Use visit_exprs to traverse all expressions in the actions
    let _ = actions.visit_exprs(&mut |expr| {
        if error.is_some() {
            return expr; // Skip processing if we already have an error
        }

        if let Expr::Call(span, head, args) = &expr
            && head.as_str() == "unstable-fresh!"
        {
            match parse_fresh_args(span, args) {
                Ok(opts) => options_list.push(opts),
                Err(e) => error = Some(e),
            }
        }
        expr
    });

    match error {
        Some(e) => Err(e),
        None => Ok(options_list),
    }
}

/// Parse the arguments to unstable-fresh!
/// Syntax: (unstable-fresh! SortName [:cost N] [:unextractable])
fn parse_fresh_args(span: &egglog::ast::Span, args: &[Expr]) -> Result<FreshOptions, Error> {
    if args.is_empty() {
        return Err(Error::ParseError(ParseError(
            span.clone(),
            "unstable-fresh! requires at least 1 argument (the sort name)".to_string(),
        )));
    }

    // First argument must be the sort name
    let sort = match &args[0] {
        Expr::Var(_span, sort_name) => sort_name.clone(),
        _ => {
            return Err(Error::ParseError(ParseError(
                span.clone(),
                "unstable-fresh! first argument must be a sort name".to_string(),
            )));
        }
    };

    // Default values - cost defaults to 1, unextractable defaults to false
    let mut cost = Some(1);
    let mut unextractable = false;

    // Parse keyword arguments
    let mut i = 1;
    while i < args.len() {
        match &args[i] {
            Expr::Var(kw_span, keyword) => {
                match keyword.as_str() {
                    ":cost" => {
                        i += 1;
                        if i >= args.len() {
                            return Err(Error::ParseError(ParseError(
                                kw_span.clone(),
                                ":cost requires a value".to_string(),
                            )));
                        }
                        match &args[i] {
                            Expr::Lit(_, Literal::Int(n)) => {
                                cost = Some(*n as u64);
                            }
                            _ => {
                                return Err(Error::ParseError(ParseError(
                                    kw_span.clone(),
                                    ":cost value must be an integer".to_string(),
                                )));
                            }
                        }
                    }
                    ":unextractable" => {
                        // :unextractable is a flag - its presence means true
                        unextractable = true;
                    }
                    _ => {
                        return Err(Error::ParseError(ParseError(
                            kw_span.clone(),
                            format!("unknown option: {}", keyword),
                        )));
                    }
                }
            }
            _ => {
                return Err(Error::ParseError(ParseError(
                    span.clone(),
                    "expected keyword argument (:cost or :unextractable)".to_string(),
                )));
            }
        }
        i += 1;
    }

    Ok(FreshOptions {
        sort,
        cost,
        unextractable,
    })
}
