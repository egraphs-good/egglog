//! Parse a string into egglog.

use crate::*;
use ordered_float::OrderedFloat;

/// A [`Span`] contains the file name and a pair of offsets representing the start and the end.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Span {
    /// Panics if a span is needed. Prefer `Span::Rust` (see `span!`)
    /// unless this behaviour is explicitly desired.
    Panic,
    /// A span from a `.egg` file.
    /// Constructed by `parse_program` and `parse_expr`.
    Egglog(Arc<EgglogSpan>),
    /// A span from a `.rs` file. Constructed by the `span!` macro.
    Rust(Arc<RustSpan>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EgglogSpan {
    pub file: Arc<SrcFile>,
    pub i: usize,
    pub j: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RustSpan {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

#[macro_export]
macro_rules! span {
    () => {
        $crate::ast::Span::Rust(std::sync::Arc::new($crate::ast::RustSpan {
            file: file!(),
            line: line!(),
            column: column!(),
        }))
    };
}

impl Span {
    pub fn string(&self) -> &str {
        match self {
            Span::Panic => panic!("Span::Panic in Span::string"),
            Span::Rust(_) => panic!("Span::Rust cannot track end position"),
            Span::Egglog(span) => &span.file.contents[span.i..span.j],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SrcFile {
    pub name: Option<String>,
    pub contents: String,
}

struct Location {
    line: usize,
    col: usize,
}

impl SrcFile {
    fn get_location(&self, offset: usize) -> Location {
        let mut line = 1;
        let mut col = 1;
        for (i, c) in self.contents.char_indices() {
            if i == offset {
                break;
            }
            if c == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        Location { line, col }
    }
}

// we do this slightly un-idiomatic thing because `unwrap` and `expect`
// would print the entire source program without this
impl Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Span::Panic => panic!("Span::Panic in impl Display"),
            Span::Rust(span) => write!(f, "At {}:{} of {}", span.line, span.column, span.file),
            Span::Egglog(span) => {
                let start = span.file.get_location(span.i);
                let end = span
                    .file
                    .get_location((span.j.saturating_sub(1)).max(span.i));
                let quote = self.string();
                match (&span.file.name, start.line == end.line) {
                    (Some(filename), true) => write!(
                        f,
                        "In {}:{}-{} of {filename}: {quote}",
                        start.line, start.col, end.col
                    ),
                    (Some(filename), false) => write!(
                        f,
                        "In {}:{}-{}:{} of {filename}: {quote}",
                        start.line, start.col, end.line, end.col
                    ),
                    (None, false) => write!(
                        f,
                        "In {}:{}-{}:{}: {quote}",
                        start.line, start.col, end.line, end.col
                    ),
                    (None, true) => {
                        write!(f, "In {}:{}-{}: {quote}", start.line, start.col, end.col)
                    }
                }
            }
        }
    }
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
    Atom(Symbol, Span),
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

    pub fn expect_uint(&self, e: &'static str) -> Result<usize, ParseError> {
        if let Sexp::Literal(Literal::Int(x), _) = self {
            if *x >= 0 {
                return Ok(*x as usize);
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

    pub fn expect_atom(&self, e: &'static str) -> Result<Symbol, ParseError> {
        if let Sexp::Atom(symbol, _) = self {
            return Ok(*symbol);
        }
        error!(self.span(), "expected {e}")
    }

    pub fn expect_list(&self, e: &'static str) -> Result<&[Sexp], ParseError> {
        if let Sexp::List(sexps, _) = self {
            return Ok(sexps);
        }
        error!(self.span(), "expected {e}")
    }

    pub fn expect_call(&self, e: &'static str) -> Result<(Symbol, &[Sexp], Span), ParseError> {
        if let Sexp::List(sexps, span) = self {
            if let [Sexp::Atom(func, _), args @ ..] = sexps.as_slice() {
                return Ok((*func, args, span.clone()));
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
    fn name(&self) -> Symbol;
    fn parse(&self, args: &[Sexp], span: Span, parser: &mut Parser) -> Result<T, ParseError>;
}

pub struct SimpleMacro<T, F: Fn(&[Sexp], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync>(
    Symbol,
    F,
);

impl<T, F> SimpleMacro<T, F>
where
    F: Fn(&[Sexp], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
{
    pub fn new(head: &str, f: F) -> Self {
        Self(head.into(), f)
    }
}

impl<T, F> Macro<T> for SimpleMacro<T, F>
where
    F: Fn(&[Sexp], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
{
    fn name(&self) -> Symbol {
        self.0
    }

    fn parse(&self, args: &[Sexp], span: Span, parser: &mut Parser) -> Result<T, ParseError> {
        self.1(args, span, parser)
    }
}

#[derive(Clone)]
pub struct Parser {
    commands: HashMap<Symbol, Arc<dyn Macro<Vec<Command>>>>,
    actions: HashMap<Symbol, Arc<dyn Macro<Vec<Action>>>>,
    exprs: HashMap<Symbol, Arc<dyn Macro<Expr>>>,
    pub symbol_gen: SymbolGen,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            commands: Default::default(),
            actions: Default::default(),
            exprs: Default::default(),
            symbol_gen: SymbolGen::new("$".to_string()),
        }
    }
}

impl Parser {
    pub fn get_program_from_string(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Vec<Command>, ParseError> {
        let sexps = all_sexps(Context::new(filename, input))?;
        let nested: Vec<Vec<_>> = map_fallible(&sexps, self, Self::parse_command)?;
        Ok(nested.into_iter().flatten().collect())
    }

    // currently only used for testing, but no reason it couldn't be used elsewhere later
    pub fn get_expr_from_string(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Expr, ParseError> {
        let sexp = sexp(&mut Context::new(filename, input))?;
        self.parse_expr(&sexp)
    }

    pub fn add_command_macro(&mut self, ma: Arc<dyn Macro<Vec<Command>>>) {
        self.commands.insert(ma.name(), ma);
    }

    pub fn add_action_macro(&mut self, ma: Arc<dyn Macro<Vec<Action>>>) {
        self.actions.insert(ma.name(), ma);
    }

    pub fn add_expr_macro(&mut self, ma: Arc<dyn Macro<Expr>>) {
        self.exprs.insert(ma.name(), ma);
    }

    pub fn parse_command(&mut self, sexp: &Sexp) -> Result<Vec<Command>, ParseError> {
        let (head, tail, span) = sexp.expect_call("command")?;

        if let Some(macr0) = self.commands.get(&head).cloned() {
            return macr0.parse(tail, span, self);
        }

        Ok(match head.into() {
            "set-option" => match tail {
                [name, value] => vec![Command::SetOption {
                    name: name.expect_atom("option name")?,
                    value: self.parse_expr(value)?,
                }],
                _ => return error!(span, "usage: (set-option <name> <value>)"),
            },
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
                    )
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
                            )
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
                [name] => vec![Command::AddRuleset(name.expect_atom("ruleset name")?)],
                _ => return error!(span, "usage: (ruleset <name>)"),
            },
            "unstable-combined-ruleset" => match tail {
                [name, subrulesets @ ..] => vec![Command::UnstableCombinedRuleset(
                    name.expect_atom("combined ruleset name")?,
                    map_fallible(subrulesets, self, |_, sexp| {
                        sexp.expect_atom("subruleset name")
                    })?,
                )],
                _ => {
                    return error!(
                        span,
                        "usage: (unstable-combined-ruleset <name> <child ruleset>*)"
                    )
                }
            },
            "rule" => match tail {
                [lhs, rhs, rest @ ..] => {
                    let body =
                        map_fallible(lhs.expect_list("rule query")?, self, Self::parse_fact)?;
                    let head: Vec<Vec<_>> =
                        map_fallible(rhs.expect_list("rule actions")?, self, Self::parse_action)?;
                    let head = GenericActions(head.into_iter().flatten().collect());

                    let mut ruleset = "".into();
                    let mut name = "".into();
                    for option in self.parse_options(rest)? {
                        match option {
                            (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                            (":name", [s]) => name = s.expect_string("rule name")?.into(),
                            _ => return error!(span, "could not parse rule option"),
                        }
                    }

                    vec![Command::Rule {
                        ruleset,
                        name,
                        rule: Rule { span, head, body },
                    }]
                }
                _ => return error!(span, "usage: (rule (<fact>*) (<action>*) <option>*)"),
            },
            "rewrite" => match tail {
                [lhs, rhs, rest @ ..] => {
                    let lhs = self.parse_expr(lhs)?;
                    let rhs = self.parse_expr(rhs)?;

                    let mut ruleset = "".into();
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

                    let mut ruleset = "".into();
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

                let has_ruleset = tail.len() >= 2 && tail[1].expect_uint("").is_ok();

                let (ruleset, limit, rest) = if has_ruleset {
                    (
                        tail[0].expect_atom("ruleset name")?,
                        tail[1].expect_uint("number of iterations")?,
                        &tail[2..],
                    )
                } else {
                    (
                        "".into(),
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
                [] => vec![Command::PrintOverallStatistics],
                _ => return error!(span, "usage: (print-stats)"),
            },
            "print-function" => match tail {
                [name, rows] => vec![Command::PrintFunction(
                    span,
                    name.expect_atom("table name")?,
                    rows.expect_uint("number of rows to print")?,
                )],
                _ => {
                    return error!(
                        span,
                        "usage: (print-function <table name> <number of rows>)"
                    )
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
                    ruleset: *ruleset,
                    until: None,
                },
            ));
        }

        let (head, tail, span) = sexp.expect_call("schedule")?;

        Ok(match head.into() {
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
                    Some(Sexp::Atom(o, _)) if *o == ":until".into() => false,
                    _ => true,
                };

                let (ruleset, rest) = if has_ruleset {
                    (tail[0].expect_atom("ruleset name")?, &tail[1..])
                } else {
                    ("".into(), tail)
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

        Ok(match head.into() {
            "let" => match tail {
                [name, value] => vec![Action::Let(
                    span,
                    name.expect_atom("binding name")?,
                    self.parse_expr(value)?,
                )],
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

        Ok(match head.into() {
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
                if *symbol == "_".into() {
                    self.symbol_gen.fresh(symbol)
                } else {
                    *symbol
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

    fn rec_datatype(&mut self, sexp: &Sexp) -> Result<(Span, Symbol, Subdatatypes), ParseError> {
        let (head, tail, span) = sexp.expect_call("datatype")?;

        Ok(match head.into() {
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
                    )
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

        let (types, cost) = match tail {
            [types @ .., Sexp::Atom(o, _), c] if *o == ":cost".into() => {
                (types, Some(c.expect_uint("cost")?))
            }
            types => (types, None),
        };

        Ok(Variant {
            span,
            name,
            types: map_fallible(types, self, |_, sexp| {
                sexp.expect_atom("variant argument type")
            })?,
            cost,
        })
    }

    // helper for parsing a list of options
    pub fn parse_options<'a>(
        &self,
        sexps: &'a [Sexp],
    ) -> Result<Vec<(&'a str, &'a [Sexp])>, ParseError> {
        fn option_name(sexp: &Sexp) -> Option<&str> {
            if let Ok(symbol) = sexp.expect_atom("") {
                let s: &str = symbol.into();
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
pub(crate) struct Context {
    source: Arc<SrcFile>,
    index: usize,
}

impl Context {
    pub(crate) fn new(name: Option<String>, contents: &str) -> Context {
        Context {
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
                                    return error!(s(span), "unrecognized escape character {c}")
                                }
                            });
                            in_escape = false;
                        }
                    }
                    self.advance_char();
                }
                self.advance_char();

                Token::String(string.into())
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
    String(Symbol),
    Other,
}

fn sexp(ctx: &mut Context) -> Result<Sexp, ParseError> {
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

pub(crate) fn all_sexps(mut ctx: Context) -> Result<Vec<Sexp>, ParseError> {
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
        assert_eq!(format!("{}", span!()), format!("At {}:34 of src/ast/parse.rs", line!()));
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
