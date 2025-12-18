//! Parse a string into egglog.

use crate::util::INTERNAL_SYMBOL_PREFIX;
use crate::*;
use egglog_ast::generic_ast::*;
use egglog_ast::span::{EgglogSpan, Span, SrcFile};
use ordered_float::OrderedFloat;

#[macro_export]
macro_rules! span {
    () => {
        Span::Rust(std::sync::Arc::new(RustSpan {
            file: file!(),
            line: line!(),
            column: column!(),
        }))
    };
}

// We do an unidiomatic thing here by using a struct instead of an enum.
// This is okay because we don't expect client code to respond
// differently to different parse errors. The benefit of this is that
// error messages are defined in the same place that they are created,
// making it easier to improve errors over time.
#[derive(Debug, Error)]
pub struct ParseError(pub Span, pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\nparse error: {}", self.0, self.1)
    }
}

macro_rules! error {
    ($span:expr, $($fmt:tt)*) => {
        Err(ParseError($span, format!($($fmt)*)))
    };
}

pub enum Sexp {
    // Will never contain `Literal::Unit`, as this
    // will be parsed as an empty `Sexp::List`.
    Literal(Literal, Span),
    Atom(String, Span),
    List(Vec<Sexp>, Span),
}

impl Sexp {
    pub fn span(&self) -> Span {
        match self {
            Sexp::Literal(_, span) => span.clone(),
            Sexp::Atom(_, span) => span.clone(),
            Sexp::List(_, span) => span.clone(),
        }
    }

    pub fn expect_uint<UInt: TryFrom<u64>>(&self, e: &'static str) -> Result<UInt, ParseError> {
        if let Sexp::Literal(Literal::Int(x), _) = self {
            if *x >= 0 {
                if let Ok(v) = (*x as u64).try_into() {
                    return Ok(v);
                }
            }
        }
        error!(
            self.span(),
            "expected {e} to be a nonnegative integer literal"
        )
    }

    pub fn expect_string(&self, e: &'static str) -> Result<String, ParseError> {
        if let Sexp::Literal(Literal::String(x), _) = self {
            return Ok(x.to_string());
        }
        error!(self.span(), "expected {e} to be a string literal")
    }

    pub fn expect_atom(&self, e: &'static str) -> Result<String, ParseError> {
        if let Sexp::Atom(symbol, _) = self {
            return Ok(symbol.clone());
        }
        error!(self.span(), "expected {e}")
    }

    pub fn expect_list(&self, e: &'static str) -> Result<&[Sexp], ParseError> {
        if let Sexp::List(sexps, _) = self {
            return Ok(sexps);
        }
        error!(self.span(), "expected {e}")
    }

    pub fn expect_call(&self, e: &'static str) -> Result<(String, &[Sexp], Span), ParseError> {
        if let Sexp::List(sexps, span) = self {
            if let [Sexp::Atom(func, _), args @ ..] = sexps.as_slice() {
                return Ok((func.clone(), args, span.clone()));
            }
        }
        error!(self.span(), "expected {e}")
    }
}

// helper for mapping a function that returns `Result`
fn map_fallible<T>(
    slice: &[Sexp],
    parser: &mut Parser,
    func: impl Fn(&mut Parser, &Sexp) -> Result<T, ParseError>,
) -> Result<Vec<T>, ParseError> {
    slice
        .iter()
        .map(|sexp| func(parser, sexp))
        .collect::<Result<_, _>>()
}

pub trait Macro<T>: Send + Sync {
    fn name(&self) -> &str;
    fn parse(&self, args: &[Sexp], span: Span, parser: &mut Parser) -> Result<T, ParseError>;
}

pub struct SimpleMacro<T, F: Fn(&[Sexp], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync>(
    String,
    F,
);

impl<T, F> SimpleMacro<T, F>
where
    F: Fn(&[Sexp], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
{
    pub fn new(head: &str, f: F) -> Self {
        Self(head.to_owned(), f)
    }
}

impl<T, F> Macro<T> for SimpleMacro<T, F>
where
    F: Fn(&[Sexp], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.0
    }

    fn parse(&self, args: &[Sexp], span: Span, parser: &mut Parser) -> Result<T, ParseError> {
        self.1(args, span, parser)
    }
}

#[derive(Clone)]
pub struct Parser {
    commands: HashMap<String, Arc<dyn Macro<Vec<Command>>>>,
    actions: HashMap<String, Arc<dyn Macro<Vec<Action>>>>,
    exprs: HashMap<String, Arc<dyn Macro<Expr>>>,
    user_defined: HashSet<String>,
    pub symbol_gen: SymbolGen,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            commands: Default::default(),
            actions: Default::default(),
            exprs: Default::default(),
            user_defined: Default::default(),
            symbol_gen: SymbolGen::new(INTERNAL_SYMBOL_PREFIX.to_string()),
        }
    }
}

impl Parser {
    fn ensure_symbol_not_reserved(&self, symbol: &str, span: &Span) -> Result<(), ParseError> {
        if self.symbol_gen.is_reserved(symbol) {
            return error!(
                span.clone(),
                "symbols starting with '{}' are reserved for egglog internals",
                self.symbol_gen.reserved_prefix()
            );
        }
        Ok(())
    }

    pub fn get_program_from_string(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<Command>, ParseError> {
        let sexps = all_sexps(SexpParser::new(filename, input))?;
        let nested: Vec<Vec<_>> = map_fallible(&sexps, self, Self::parse_command)?;
        Ok(nested.into_iter().flatten().collect())
    }

    // currently only used for testing, but no reason it couldn't be used elsewhere later
    pub fn get_expr_from_string(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Expr, ParseError> {
        let sexp = sexp(&mut SexpParser::new(filename, input))?;
        self.parse_expr(&sexp)
    }

    // Parse a fact from a string.
    pub fn get_fact_from_string(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Fact, ParseError> {
        let sexp = sexp(&mut SexpParser::new(filename, input))?;
        self.parse_fact(&sexp)
    }

    pub fn add_command_macro(&mut self, ma: Arc<dyn Macro<Vec<Command>>>) {
        self.commands.insert(ma.name().to_owned(), ma);
    }

    pub fn add_action_macro(&mut self, ma: Arc<dyn Macro<Vec<Action>>>) {
        self.actions.insert(ma.name().to_owned(), ma);
    }

    pub fn add_expr_macro(&mut self, ma: Arc<dyn Macro<Expr>>) {
        self.exprs.insert(ma.name().to_owned(), ma);
    }

    pub(crate) fn add_user_defined(&mut self, name: String) -> Result<(), Error> {
        if self.actions.contains_key(&name)
            || self.exprs.contains_key(&name)
            || self.commands.contains_key(&name)
        {
            use egglog_ast::span::{RustSpan, Span};
            return Err(Error::CommandAlreadyExists(name, span!()));
        }
        self.user_defined.insert(name);
        Ok(())
    }

    pub fn parse_command(&mut self, sexp: &Sexp) -> Result<Vec<Command>, ParseError> {
        let (head, tail, span) = sexp.expect_call("command")?;

        if let Some(macr0) = self.commands.get(&head).cloned() {
            return macr0.parse(tail, span, self);
        }

        // This prevents user-defined commands from being parsed as built-in commands.
        if self.user_defined.contains(&head) {
            let args = map_fallible(tail, self, Self::parse_expr)?;
            return Ok(vec![Command::UserDefined(span, head, args)]);
        }

        Ok(match head.as_str() {
            "sort" => match tail {
                [name] => vec![Command::Sort(span, name.expect_atom("sort name")?, None)],
                [name, call] => {
                    let (func, args, _) = call.expect_call("container sort declaration")?;
                    vec![Command::Sort(
                        span,
                        name.expect_atom("sort name")?,
                        Some((func, map_fallible(args, self, Self::parse_expr)?)),
                    )]
                }
                _ => {
                    return error!(
                        span,
                        "usages:\n(sort <name>)\n(sort <name> (<container sort> <argument sort>*))"
                    );
                }
            },
            "datatype" => match tail {
                [name, variants @ ..] => vec![Command::Datatype {
                    span,
                    name: name.expect_atom("sort name")?,
                    variants: map_fallible(variants, self, Self::variant)?,
                }],
                _ => return error!(span, "usage: (datatype <name> <variant>*)"),
            },
            "datatype*" => vec![Command::Datatypes {
                span,
                datatypes: map_fallible(tail, self, Self::rec_datatype)?,
            }],
            "function" => match tail {
                [name, inputs, output, rest @ ..] => vec![Command::Function {
                    name: name.expect_atom("function name")?,
                    schema: self.parse_schema(inputs, output)?,
                    merge: match self.parse_options(rest)?.as_slice() {
                        [(":no-merge", [])] => None,
                        [(":merge", [e])] => Some(self.parse_expr(e)?),
                        [] => {
                            return error!(
                                span,
                                "functions are required to specify merge behaviour"
                            );
                        }
                        _ => return error!(span, "could not parse function options"),
                    },
                    span,
                }],
                _ => {
                    let a = "(function <name> (<input sort>*) <output sort> :merge <expr>)";
                    let b = "(function <name> (<input sort>*) <output sort> :no-merge)";
                    return error!(span, "usages:\n{a}\n{b}");
                }
            },
            "constructor" => match tail {
                [name, inputs, output, rest @ ..] => {
                    let mut cost = None;
                    let mut unextractable = false;
                    match self.parse_options(rest)?.as_slice() {
                        [] => {}
                        [(":unextractable", [])] => unextractable = true,
                        [(":cost", [c])] => cost = Some(c.expect_uint("cost")?),
                        _ => return error!(span, "could not parse constructor options"),
                    }

                    vec![Command::Constructor {
                        span,
                        name: name.expect_atom("constructor name")?,
                        schema: self.parse_schema(inputs, output)?,
                        cost,
                        unextractable,
                    }]
                }
                _ => {
                    let a = "(constructor <name> (<input sort>*) <output sort>)";
                    let b = "(constructor <name> (<input sort>*) <output sort> :cost <cost>)";
                    let c = "(constructor <name> (<input sort>*) <output sort> :unextractable)";
                    return error!(span, "usages:\n{a}\n{b}\n{c}");
                }
            },
            "relation" => match tail {
                [name, inputs] => vec![Command::Relation {
                    span,
                    name: name.expect_atom("relation name")?,
                    inputs: map_fallible(inputs.expect_list("input sorts")?, self, |_, sexp| {
                        sexp.expect_atom("input sort")
                    })?,
                }],
                _ => return error!(span, "usage: (relation <name> (<input sort>*))"),
            },
            "ruleset" => match tail {
                [name] => vec![Command::AddRuleset(span, name.expect_atom("ruleset name")?)],
                _ => return error!(span, "usage: (ruleset <name>)"),
            },
            "unstable-combined-ruleset" => match tail {
                [name, subrulesets @ ..] => vec![Command::UnstableCombinedRuleset(
                    span,
                    name.expect_atom("combined ruleset name")?,
                    map_fallible(subrulesets, self, |_, sexp| {
                        sexp.expect_atom("subruleset name")
                    })?,
                )],
                _ => {
                    return error!(
                        span,
                        "usage: (unstable-combined-ruleset <name> <child ruleset>*)"
                    );
                }
            },
            "rule" => match tail {
                [lhs, rhs, rest @ ..] => {
                    let body =
                        map_fallible(lhs.expect_list("rule query")?, self, Self::parse_fact)?;
                    let head: Vec<Vec<_>> =
                        map_fallible(rhs.expect_list("rule actions")?, self, Self::parse_action)?;
                    let head = GenericActions(head.into_iter().flatten().collect());

                    let mut ruleset = String::new();
                    let mut name = String::new();
                    for option in self.parse_options(rest)? {
                        match option {
                            (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                            (":name", [s]) => name = s.expect_string("rule name")?,
                            _ => return error!(span, "could not parse rule option"),
                        }
                    }

                    vec![Command::Rule {
                        rule: Rule {
                            span,
                            head,
                            body,
                            name,
                            ruleset,
                        },
                    }]
                }
                _ => return error!(span, "usage: (rule (<fact>*) (<action>*) <option>*)"),
            },
            "rewrite" => match tail {
                [lhs, rhs, rest @ ..] => {
                    let lhs = self.parse_expr(lhs)?;
                    let rhs = self.parse_expr(rhs)?;

                    let mut ruleset = String::new();
                    let mut conditions = Vec::new();
                    let mut subsume = false;
                    for option in self.parse_options(rest)? {
                        match option {
                            (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                            (":subsume", []) => subsume = true,
                            (":when", [w]) => {
                                conditions = map_fallible(
                                    w.expect_list("rewrite conditions")?,
                                    self,
                                    Self::parse_fact,
                                )?
                            }
                            _ => return error!(span, "could not parse rewrite options"),
                        }
                    }

                    vec![Command::Rewrite(
                        ruleset,
                        Rewrite {
                            span,
                            lhs,
                            rhs,
                            conditions,
                        },
                        subsume,
                    )]
                }
                _ => return error!(span, "usage: (rewrite <expr> <expr> <option>*)"),
            },
            "birewrite" => match tail {
                [lhs, rhs, rest @ ..] => {
                    let lhs = self.parse_expr(lhs)?;
                    let rhs = self.parse_expr(rhs)?;

                    let mut ruleset = String::new();
                    let mut conditions = Vec::new();
                    for option in self.parse_options(rest)? {
                        match option {
                            (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                            (":when", [w]) => {
                                conditions = map_fallible(
                                    w.expect_list("rewrite conditions")?,
                                    self,
                                    Self::parse_fact,
                                )?
                            }
                            _ => return error!(span, "could not parse birewrite options"),
                        }
                    }

                    vec![Command::BiRewrite(
                        ruleset,
                        Rewrite {
                            span,
                            lhs,
                            rhs,
                            conditions,
                        },
                    )]
                }
                _ => return error!(span, "usage: (birewrite <expr> <expr> <option>*)"),
            },
            "run" => {
                if tail.is_empty() {
                    return error!(span, "usage: (run <ruleset>? <uint> <:until (<fact>*)>?)");
                }

                let has_ruleset = tail.len() >= 2 && tail[1].expect_uint::<u32>("").is_ok();

                let (ruleset, limit, rest) = if has_ruleset {
                    (
                        tail[0].expect_atom("ruleset name")?,
                        tail[1].expect_uint("number of iterations")?,
                        &tail[2..],
                    )
                } else {
                    (
                        String::new(),
                        tail[0].expect_uint("number of iterations")?,
                        &tail[1..],
                    )
                };

                let until = match self.parse_options(rest)?.as_slice() {
                    [] => None,
                    [(":until", facts)] => Some(map_fallible(facts, self, Self::parse_fact)?),
                    _ => return error!(span, "could not parse run options"),
                };

                vec![Command::RunSchedule(Schedule::Repeat(
                    span.clone(),
                    limit,
                    Box::new(Schedule::Run(span, RunConfig { ruleset, until })),
                ))]
            }
            "run-schedule" => vec![Command::RunSchedule(Schedule::Sequence(
                span,
                map_fallible(tail, self, Self::schedule)?,
            ))],
            "extract" => match tail {
                [e] => vec![Command::Extract(
                    span.clone(),
                    self.parse_expr(e)?,
                    Expr::Lit(span, Literal::Int(0)),
                )],
                [e, v] => vec![Command::Extract(
                    span,
                    self.parse_expr(e)?,
                    self.parse_expr(v)?,
                )],
                _ => return error!(span, "usage: (extract <expr> <number of variants>?)"),
            },
            "check" => vec![Command::Check(
                span,
                map_fallible(tail, self, Self::parse_fact)?,
            )],
            "push" => match tail {
                [] => vec![Command::Push(1)],
                [n] => vec![Command::Push(n.expect_uint("number of times to push")?)],
                _ => return error!(span, "usage: (push <uint>?)"),
            },
            "pop" => match tail {
                [] => vec![Command::Pop(span, 1)],
                [n] => vec![Command::Pop(span, n.expect_uint("number of times to pop")?)],
                _ => return error!(span, "usage: (pop <uint>?)"),
            },
            "print-stats" => match tail {
                [] => vec![Command::PrintOverallStatistics(span, None)],
                [Sexp::Atom(o, _), file] if o == ":file" => vec![Command::PrintOverallStatistics(
                    span,
                    Some(file.expect_string("file name")?),
                )],
                _ => {
                    return error!(
                        span,
                        "usages: (print-stats)\n(print-stats :file \"<filename>\")"
                    );
                }
            },
            "print-function" => match tail {
                [name] => vec![Command::PrintFunction(
                    span,
                    name.expect_atom("table name")?,
                    None,
                    None,
                    PrintFunctionMode::Default,
                )],
                [name, rest @ ..] => {
                    let rows: Option<usize> = rest[0].expect_uint("number of rows").ok();
                    let rest = if rows.is_some() { &rest[1..] } else { rest };

                    let mut file = None;
                    let mut mode = PrintFunctionMode::Default;
                    for opt in self.parse_options(rest)? {
                        match opt {
                            (":file", [file_name]) => {
                                file = Some(file_name.expect_string("file name")?);
                            }
                            (":mode", [Sexp::Atom(mode_str, _)]) => {
                                mode = match mode_str.as_str() {
                                    "default" => PrintFunctionMode::Default,
                                    "csv" => PrintFunctionMode::CSV,
                                    _ => {
                                        return error!(
                                            span,
                                            "Unknown print-function mode. Supported modes are `default` and `csv`."
                                        );
                                    }
                                };
                            }
                            _ => {
                                return error!(
                                    span,
                                    "Unknown option to print-function. Supported options are `:mode csv|default` and `:file \"<filename>\"`."
                                );
                            }
                        }
                    }
                    vec![Command::PrintFunction(
                        span,
                        name.expect_atom("table name")?,
                        rows,
                        file,
                        mode,
                    )]
                }
                _ => {
                    return error!(
                        span,
                        "usage: (print-function <table name> <number of rows>? <option>*)"
                    );
                }
            },
            "print-size" => match tail {
                [] => vec![Command::PrintSize(span, None)],
                [name] => vec![Command::PrintSize(
                    span,
                    Some(name.expect_atom("table name")?),
                )],
                _ => return error!(span, "usage: (print-size <table name>?)"),
            },
            "input" => match tail {
                [name, file] => vec![Command::Input {
                    span,
                    name: name.expect_atom("table name")?,
                    file: file.expect_string("file name")?,
                }],
                _ => return error!(span, "usage: (input <table name> \"<file name>\")"),
            },
            "output" => match tail {
                [file, exprs @ ..] => vec![Command::Output {
                    span,
                    file: file.expect_string("file name")?,
                    exprs: map_fallible(exprs, self, Self::parse_expr)?,
                }],
                _ => return error!(span, "usage: (output <file name> <expr>+)"),
            },
            "include" => match tail {
                [file] => vec![Command::Include(span, file.expect_string("file name")?)],
                _ => return error!(span, "usage: (include <file name>)"),
            },
            "fail" => match tail {
                [subcommand] => {
                    let mut cs = self.parse_command(subcommand)?;
                    if cs.len() != 1 {
                        todo!("extend Fail to work with multiple parsed commands")
                    }
                    vec![Command::Fail(span, Box::new(cs.remove(0)))]
                }
                _ => return error!(span, "usage: (fail <command>)"),
            },
            _ => self
                .parse_action(sexp)?
                .into_iter()
                .map(Command::Action)
                .collect(),
        })
    }

    pub fn schedule(&mut self, sexp: &Sexp) -> Result<Schedule, ParseError> {
        if let Sexp::Atom(ruleset, span) = sexp {
            return Ok(Schedule::Run(
                span.clone(),
                RunConfig {
                    ruleset: ruleset.clone(),
                    until: None,
                },
            ));
        }

        let (head, tail, span) = sexp.expect_call("schedule")?;

        Ok(match head.as_str() {
            "saturate" => Schedule::Saturate(
                span.clone(),
                Box::new(Schedule::Sequence(
                    span,
                    map_fallible(tail, self, Self::schedule)?,
                )),
            ),
            "seq" => Schedule::Sequence(span, map_fallible(tail, self, Self::schedule)?),
            "repeat" => match tail {
                [limit, tail @ ..] => Schedule::Repeat(
                    span.clone(),
                    limit.expect_uint("number of iterations")?,
                    Box::new(Schedule::Sequence(
                        span,
                        map_fallible(tail, self, Self::schedule)?,
                    )),
                ),
                _ => return error!(span, "usage: (repeat <number of iterations> <schedule>*)"),
            },
            "run" => {
                let has_ruleset = match tail.first() {
                    None => false,
                    Some(Sexp::Atom(o, _)) if *o == ":until" => false,
                    _ => true,
                };

                let (ruleset, rest) = if has_ruleset {
                    (tail[0].expect_atom("ruleset name")?, &tail[1..])
                } else {
                    (String::new(), tail)
                };

                let until = match self.parse_options(rest)?.as_slice() {
                    [] => None,
                    [(":until", facts)] => Some(map_fallible(facts, self, Self::parse_fact)?),
                    _ => return error!(span, "could not parse run options"),
                };

                Schedule::Run(span, RunConfig { ruleset, until })
            }
            _ => return error!(span, "expected either saturate, seq, repeat, or run"),
        })
    }

    pub fn parse_action(&mut self, sexp: &Sexp) -> Result<Vec<Action>, ParseError> {
        let (head, tail, span) = sexp.expect_call("action")?;

        if let Some(func) = self.actions.get(&head).cloned() {
            return func.parse(tail, span, self);
        }

        Ok(match head.as_str() {
            "let" => match tail {
                [name, value] => {
                    let binding_span = name.span();
                    let binding = name.expect_atom("binding name")?;
                    self.ensure_symbol_not_reserved(&binding, &binding_span)?;
                    vec![Action::Let(span, binding, self.parse_expr(value)?)]
                }
                _ => return error!(span, "usage: (let <name> <expr>)"),
            },
            "set" => match tail {
                [call, value] => {
                    let (func, args, _) = call.expect_call("table lookup")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    let value = self.parse_expr(value)?;
                    vec![Action::Set(span, func, args, value)]
                }
                _ => return error!(span, "usage: (set (<table name> <expr>*) <expr>)"),
            },
            "delete" => match tail {
                [call] => {
                    let (func, args, _) = call.expect_call("table lookup")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    vec![Action::Change(span, Change::Delete, func, args)]
                }
                _ => return error!(span, "usage: (delete (<table name> <expr>*))"),
            },
            "subsume" => match tail {
                [call] => {
                    let (func, args, _) = call.expect_call("table lookup")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    vec![Action::Change(span, Change::Subsume, func, args)]
                }
                _ => return error!(span, "usage: (subsume (<table name> <expr>*))"),
            },
            "union" => match tail {
                [e1, e2] => vec![Action::Union(
                    span,
                    self.parse_expr(e1)?,
                    self.parse_expr(e2)?,
                )],
                _ => return error!(span, "usage: (union <expr> <expr>)"),
            },
            "panic" => match tail {
                [message] => vec![Action::Panic(span, message.expect_string("error message")?)],
                _ => return error!(span, "usage: (panic <string>)"),
            },
            _ => vec![Action::Expr(span, self.parse_expr(sexp)?)],
        })
    }

    pub fn parse_fact(&mut self, sexp: &Sexp) -> Result<Fact, ParseError> {
        let (head, tail, span) = sexp.expect_call("fact")?;

        Ok(match head.as_str() {
            "=" => match tail {
                [e1, e2] => Fact::Eq(span, self.parse_expr(e1)?, self.parse_expr(e2)?),
                _ => return error!(span, "usage: (= <expr> <expr>)"),
            },
            _ => Fact::Fact(self.parse_expr(sexp)?),
        })
    }

    pub fn parse_expr(&mut self, sexp: &Sexp) -> Result<Expr, ParseError> {
        Ok(match sexp {
            Sexp::Literal(literal, span) => Expr::Lit(span.clone(), literal.clone()),
            Sexp::Atom(symbol, span) => Expr::Var(
                span.clone(),
                if *symbol == "_" {
                    self.symbol_gen.fresh(symbol)
                } else {
                    self.ensure_symbol_not_reserved(symbol, span)?;
                    symbol.clone()
                },
            ),
            Sexp::List(list, span) => match list.as_slice() {
                [] => Expr::Lit(span.clone(), Literal::Unit),
                _ => {
                    let (head, tail, span) = sexp.expect_call("call expression")?;

                    if let Some(func) = self.exprs.get(&head).cloned() {
                        return func.parse(tail, span, self);
                    }

                    Expr::Call(
                        span.clone(),
                        head,
                        map_fallible(tail, self, Self::parse_expr)?,
                    )
                }
            },
        })
    }

    pub fn rec_datatype(
        &mut self,
        sexp: &Sexp,
    ) -> Result<(Span, String, Subdatatypes), ParseError> {
        let (head, tail, span) = sexp.expect_call("datatype")?;

        Ok(match head.as_str() {
            "sort" => match tail {
                [name, call] => {
                    let name = name.expect_atom("sort name")?;
                    let (func, args, _) = call.expect_call("container sort declaration")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    (span, name, Subdatatypes::NewSort(func, args))
                }
                _ => {
                    return error!(
                        span,
                        "usage: (sort <name> (<container sort> <argument sort>*))"
                    );
                }
            },
            _ => {
                let variants = map_fallible(tail, self, Self::variant)?;
                (span, head, Subdatatypes::Variants(variants))
            }
        })
    }

    pub fn variant(&mut self, sexp: &Sexp) -> Result<Variant, ParseError> {
        let (name, tail, span) = sexp.expect_call("datatype variant")?;

        let (types, cost, unextractable) = match tail {
            [types @ .., Sexp::Atom(o, _)] if *o == ":unextractable" => (types, None, true),
            [types @ .., Sexp::Atom(o, _), c] if *o == ":cost" => {
                (types, Some(c.expect_uint("cost")?), false)
            }
            types => (types, None, false),
        };

        Ok(Variant {
            span,
            name,
            types: map_fallible(types, self, |_, sexp| {
                sexp.expect_atom("variant argument type")
            })?,
            cost,
            unextractable,
        })
    }

    // helper for parsing a list of options
    pub fn parse_options<'a>(
        &self,
        sexps: &'a [Sexp],
    ) -> Result<Vec<(&'a str, &'a [Sexp])>, ParseError> {
        fn option_name(sexp: &Sexp) -> Option<&str> {
            if let Sexp::Atom(s, _) = sexp {
                if let Some(':') = s.chars().next() {
                    return Some(s);
                }
            }
            None
        }

        let mut out = Vec::new();
        let mut i = 0;
        while i < sexps.len() {
            let Some(key) = option_name(&sexps[i]) else {
                return error!(sexps[i].span(), "option key must start with ':'");
            };
            i += 1;

            let start = i;
            while i < sexps.len() && option_name(&sexps[i]).is_none() {
                i += 1;
            }
            out.push((key, &sexps[start..i]));
        }
        Ok(out)
    }

    pub fn parse_schema(&self, input: &Sexp, output: &Sexp) -> Result<Schema, ParseError> {
        Ok(Schema {
            input: input
                .expect_list("input sorts")?
                .iter()
                .map(|sexp| sexp.expect_atom("input sort"))
                .collect::<Result<_, _>>()?,
            output: output.expect_atom("output sort")?,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SexpParser {
    source: Arc<SrcFile>,
    index: usize,
}

impl SexpParser {
    pub(crate) fn new(name: Option<String>, contents: &str) -> SexpParser {
        SexpParser {
            source: Arc::new(SrcFile {
                name,
                contents: contents.to_string(),
            }),
            index: 0,
        }
    }

    fn current_char(&self) -> Option<char> {
        self.source.contents[self.index..].chars().next()
    }

    fn advance_char(&mut self) {
        assert!(self.index < self.source.contents.len());
        loop {
            self.index += 1;
            if self.source.contents.is_char_boundary(self.index) {
                break;
            }
        }
    }

    fn advance_past_whitespace(&mut self) {
        let mut in_comment = false;
        loop {
            match self.current_char() {
                None => break,
                Some(';') => in_comment = true,
                Some('\n') => in_comment = false,
                Some(c) if c.is_whitespace() => {}
                Some(_) if in_comment => {}
                Some(_) => break,
            }
            self.advance_char();
        }
    }

    fn is_at_end(&self) -> bool {
        self.index == self.source.contents.len()
    }

    fn next(&mut self) -> Result<(Token, EgglogSpan), ParseError> {
        self.advance_past_whitespace();
        let mut span = EgglogSpan {
            file: self.source.clone(),
            i: self.index,
            j: self.index,
        };

        let Some(c) = self.current_char() else {
            return error!(s(span), "unexpected end of file");
        };
        self.advance_char();

        let token = match c {
            '(' => Token::Open,
            ')' => Token::Close,
            '"' => {
                let mut in_escape = false;
                let mut string = String::new();

                loop {
                    span.j = self.index;
                    match self.current_char() {
                        None => return error!(s(span), "string is missing end quote"),
                        Some('"') if !in_escape => break,
                        Some('\\') if !in_escape => in_escape = true,
                        Some(c) => {
                            string.push(match (in_escape, c) {
                                (false, c) => c,
                                (true, 'n') => '\n',
                                (true, 't') => '\t',
                                (true, '\\') => '\\',
                                (true, '\"') => '\"',
                                (true, c) => {
                                    return error!(s(span), "unrecognized escape character {c}");
                                }
                            });
                            in_escape = false;
                        }
                    }
                    self.advance_char();
                }
                self.advance_char();

                Token::String(string)
            }
            _ => {
                loop {
                    match self.current_char() {
                        Some(c) if c.is_whitespace() => break,
                        Some(';' | '(' | ')') => break,
                        None => break,
                        Some(_) => self.advance_char(),
                    }
                }
                Token::Other
            }
        };

        span.j = self.index;
        self.advance_past_whitespace();

        Ok((token, span))
    }
}

fn s(span: EgglogSpan) -> Span {
    Span::Egglog(Arc::new(span))
}

enum Token {
    Open,
    Close,
    String(String),
    Other,
}

fn sexp(ctx: &mut SexpParser) -> Result<Sexp, ParseError> {
    let mut stack: Vec<(EgglogSpan, Vec<Sexp>)> = vec![];

    loop {
        let (token, span) = ctx.next()?;

        let sexp = match token {
            Token::Open => {
                stack.push((span, vec![]));
                continue;
            }
            Token::Close => {
                if stack.is_empty() {
                    return error!(s(span), "unexpected `)`");
                }
                let (mut list_span, list) = stack.pop().unwrap();
                list_span.j = span.j;
                Sexp::List(list, s(list_span))
            }
            Token::String(sym) => Sexp::Literal(Literal::String(sym), s(span)),
            Token::Other => {
                let span = s(span);
                let s = span.string();

                if s == "true" {
                    Sexp::Literal(Literal::Bool(true), span)
                } else if s == "false" {
                    Sexp::Literal(Literal::Bool(false), span)
                } else if let Ok(int) = s.parse::<i64>() {
                    Sexp::Literal(Literal::Int(int), span)
                } else if s == "NaN" {
                    Sexp::Literal(Literal::Float(OrderedFloat(f64::NAN)), span)
                } else if s == "inf" {
                    Sexp::Literal(Literal::Float(OrderedFloat(f64::INFINITY)), span)
                } else if s == "-inf" {
                    Sexp::Literal(Literal::Float(OrderedFloat(f64::NEG_INFINITY)), span)
                } else if let Ok(float) = s.parse::<f64>() {
                    if float.is_finite() {
                        Sexp::Literal(Literal::Float(OrderedFloat(float)), span)
                    } else {
                        Sexp::Atom(s.into(), span)
                    }
                } else {
                    Sexp::Atom(s.into(), span)
                }
            }
        };

        if stack.is_empty() {
            return Ok(sexp);
        } else {
            stack.last_mut().unwrap().1.push(sexp);
        }
    }
}

pub(crate) fn all_sexps(mut ctx: SexpParser) -> Result<Vec<Sexp>, ParseError> {
    let mut sexps = Vec::new();
    ctx.advance_past_whitespace();
    while !ctx.is_at_end() {
        sexps.push(sexp(&mut ctx)?);
        ctx.advance_past_whitespace();
    }
    Ok(sexps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = Parser::default().get_expr_from_string(None, s).unwrap();
        assert_eq!(format!("{}", e), s);
    }

    #[test]
    #[rustfmt::skip]
    fn rust_span_display() {
        let actual = format!("{}", span!()).replace('\\', "/");
        assert!(actual.starts_with("At "));
        assert!(actual.contains(":"));
        assert!(actual.ends_with("src/ast/parse.rs"));
    }

    #[test]
    fn test_parser_macros() {
        let mut parser = Parser::default();
        let y = "xxxx";
        parser.add_expr_macro(Arc::new(SimpleMacro::new("qqqq", |tail, span, macros| {
            Ok(Expr::Call(
                span,
                y.into(),
                map_fallible(tail, macros, Parser::parse_expr)?,
            ))
        })));
        let s = r#"(f (qqqq a 3) 4.0 (H "hello"))"#;
        let t = r#"(f (xxxx a 3) 4.0 (H "hello"))"#;
        let e = parser.get_expr_from_string(None, s).unwrap();
        assert_eq!(format!("{}", e), t);
    }
}
