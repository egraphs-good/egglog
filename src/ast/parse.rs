//! Parse a string into egglog.

use crate::*;
use ordered_float::OrderedFloat;

pub type Macro<T> = fn(&[Sexp], Span, &Parser) -> Result<T, ParseError>;

#[derive(Clone)]
pub struct Parser {
    pub commands: HashMap<Symbol, Macro<Vec<Command>>>,
    pub actions: HashMap<Symbol, Macro<Vec<Action>>>,
    pub exprs: HashMap<Symbol, Macro<Expr>>,
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

pub fn parse_program(
    filename: Option<String>,
    input: &str,
    parser: &Parser,
) -> Result<Vec<Command>, ParseError> {
    let sexps = all_sexps(Context::new(filename, input))?;
    let nested = map_fallible(&sexps, parser, commands)?;
    Ok(nested.into_iter().flatten().collect())
}

// currently only used for testing, but no reason it couldn't be used elsewhere later
pub fn parse_expr(
    filename: Option<String>,
    input: &str,
    parser: &Parser,
) -> Result<Expr, ParseError> {
    let sexp = sexp(&mut Context::new(filename, input))?;
    expr(&sexp, parser)
}

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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EgglogSpan {
    file: Arc<SrcFile>,
    i: usize,
    j: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
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
            Span::Panic => panic!("Span::Panic in Span::String"),
            Span::Rust(_) => todo!(),
            Span::Egglog(span) => &span.file.contents[span.i..span.j],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SrcFile {
    name: Option<String>,
    contents: String,
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
pub struct ParseError(Span, String);

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
    fn span(&self) -> Span {
        match self {
            Sexp::Literal(_, span) => span.clone(),
            Sexp::Atom(_, span) => span.clone(),
            Sexp::List(_, span) => span.clone(),
        }
    }

    fn expect_uint(&self, e: &'static str) -> Result<usize, ParseError> {
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

    fn expect_string(&self, e: &'static str) -> Result<String, ParseError> {
        if let Sexp::Literal(Literal::String(x), _) = self {
            return Ok(x.to_string());
        }
        error!(self.span(), "expected {e} to be a string literal")
    }

    fn expect_atom(&self, e: &'static str) -> Result<Symbol, ParseError> {
        if let Sexp::Atom(symbol, _) = self {
            return Ok(*symbol);
        }
        error!(self.span(), "expected {e}")
    }

    fn expect_list(&self, e: &'static str) -> Result<&[Sexp], ParseError> {
        if let Sexp::List(sexps, _) = self {
            return Ok(sexps);
        }
        error!(self.span(), "expected {e}")
    }

    fn expect_call(&self, e: &'static str) -> Result<(Symbol, &[Sexp], Span), ParseError> {
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
    parser: &Parser,
    func: impl Fn(&Sexp, &Parser) -> Result<T, ParseError>,
) -> Result<Vec<T>, ParseError> {
    slice
        .iter()
        .map(|sexp| func(sexp, parser))
        .collect::<Result<_, _>>()
}

// helper for parsing a list of options
fn options(sexps: &[Sexp]) -> Result<Vec<(&str, &[Sexp])>, ParseError> {
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

fn commands(sexp: &Sexp, parser: &Parser) -> Result<Vec<Command>, ParseError> {
    let (head, tail, span) = sexp.expect_call("command")?;

    if let Some(func) = parser.commands.get(&head) {
        return func(tail, span, parser);
    }

    Ok(match head.into() {
        "set-option" => match tail {
            [name, value] => vec![Command::SetOption {
                name: name.expect_atom("option name")?,
                value: expr(value, parser)?,
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
                    Some((func, map_fallible(args, parser, expr)?)),
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
                variants: map_fallible(variants, parser, variant)?,
            }],
            _ => return error!(span, "usage: (datatype <name> <variant>*)"),
        },
        "datatype*" => vec![Command::Datatypes {
            span,
            datatypes: map_fallible(tail, parser, rec_datatype)?,
        }],
        "function" => match tail {
            [name, inputs, output, rest @ ..] => vec![Command::Function {
                name: name.expect_atom("function name")?,
                schema: schema(inputs, output)?,
                merge: match options(rest)?.as_slice() {
                    [(":no-merge", [])] => None,
                    [(":merge", [e])] => Some(expr(e, parser)?),
                    [] => return error!(span, "functions are required to specify merge behaviour"),
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
                match options(rest)?.as_slice() {
                    [] => {}
                    [(":unextractable", [])] => unextractable = true,
                    [(":cost", [c])] => cost = Some(c.expect_uint("cost")?),
                    _ => return error!(span, "could not parse constructor options"),
                }

                vec![Command::Constructor {
                    span,
                    name: name.expect_atom("constructor name")?,
                    schema: schema(inputs, output)?,
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
                inputs: map_fallible(inputs.expect_list("input sorts")?, parser, |sexp, _| {
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
                map_fallible(subrulesets, parser, |sexp, _| {
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
                let body = map_fallible(lhs.expect_list("rule query")?, parser, fact)?;
                let head = map_fallible(rhs.expect_list("rule actions")?, parser, actions)?;
                let head = GenericActions(head.into_iter().flatten().collect());

                let mut ruleset = "".into();
                let mut name = "".into();
                for option in options(rest)? {
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
                let lhs = expr(lhs, parser)?;
                let rhs = expr(rhs, parser)?;

                let mut ruleset = "".into();
                let mut conditions = Vec::new();
                let mut subsume = false;
                for option in options(rest)? {
                    match option {
                        (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                        (":subsume", []) => subsume = true,
                        (":when", [w]) => {
                            conditions =
                                map_fallible(w.expect_list("rewrite conditions")?, parser, fact)?
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
                let lhs = expr(lhs, parser)?;
                let rhs = expr(rhs, parser)?;

                let mut ruleset = "".into();
                let mut conditions = Vec::new();
                for option in options(rest)? {
                    match option {
                        (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                        (":when", [w]) => {
                            conditions =
                                map_fallible(w.expect_list("rewrite conditions")?, parser, fact)?
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

            let until = match options(rest)?.as_slice() {
                [] => None,
                [(":until", facts)] => Some(map_fallible(facts, parser, fact)?),
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
            map_fallible(tail, parser, schedule)?,
        ))],
        "simplify" => match tail {
            [s, e] => vec![Command::Simplify {
                span,
                schedule: schedule(s, parser)?,
                expr: expr(e, parser)?,
            }],
            _ => return error!(span, "usage: (simplify <schedule> <expr>)"),
        },
        "query-extract" => match tail {
            [rest @ .., e] => {
                let variants = match options(rest)?.as_slice() {
                    [] => 0,
                    [(":variants", [v])] => v.expect_uint("number of variants")?,
                    _ => return error!(span, "could not parse query-extract options"),
                };
                vec![Command::QueryExtract {
                    span,
                    expr: expr(e, parser)?,
                    variants,
                }]
            }
            _ => return error!(span, "usage: (query-extract <:variants <uint>>? <expr>)"),
        },
        "check" => vec![Command::Check(span, map_fallible(tail, parser, fact)?)],
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
                exprs: map_fallible(exprs, parser, expr)?,
            }],
            _ => return error!(span, "usage: (output <file name> <expr>+)"),
        },
        "include" => match tail {
            [file] => vec![Command::Include(span, file.expect_string("file name")?)],
            _ => return error!(span, "usage: (include <file name>)"),
        },
        "fail" => match tail {
            [subcommand] => {
                let mut cs = commands(subcommand, parser)?;
                if cs.len() != 1 {
                    todo!("extend Fail to work with multiple parsed commands")
                }
                vec![Command::Fail(span, Box::new(cs.remove(0)))]
            }
            _ => return error!(span, "usage: (fail <command>)"),
        },
        _ => actions(sexp, parser)
            .into_iter()
            .flatten()
            .map(Command::Action)
            .collect(),
    })
}

fn schema(input: &Sexp, output: &Sexp) -> Result<Schema, ParseError> {
    Ok(Schema {
        input: input
            .expect_list("input sorts")?
            .iter()
            .map(|sexp| sexp.expect_atom("input sort"))
            .collect::<Result<_, _>>()?,
        output: output.expect_atom("output sort")?,
    })
}

fn rec_datatype(sexp: &Sexp, parser: &Parser) -> Result<(Span, Symbol, Subdatatypes), ParseError> {
    let (head, tail, span) = sexp.expect_call("datatype")?;

    Ok(match head.into() {
        "sort" => match tail {
            [name, call] => {
                let name = name.expect_atom("sort name")?;
                let (func, args, _) = call.expect_call("container sort declaration")?;
                let args = map_fallible(args, parser, expr)?;
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
            let variants = map_fallible(tail, parser, variant)?;
            (span, head, Subdatatypes::Variants(variants))
        }
    })
}

fn variant(sexp: &Sexp, parser: &Parser) -> Result<Variant, ParseError> {
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
        types: map_fallible(types, parser, |sexp, _| {
            sexp.expect_atom("variant argument type")
        })?,
        cost,
    })
}

fn schedule(sexp: &Sexp, parser: &Parser) -> Result<Schedule, ParseError> {
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
                map_fallible(tail, parser, schedule)?,
            )),
        ),
        "seq" => Schedule::Sequence(span, map_fallible(tail, parser, schedule)?),
        "repeat" => match tail {
            [limit, tail @ ..] => Schedule::Repeat(
                span.clone(),
                limit.expect_uint("number of iterations")?,
                Box::new(Schedule::Sequence(
                    span,
                    map_fallible(tail, parser, schedule)?,
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

            let until = match options(rest)?.as_slice() {
                [] => None,
                [(":until", facts)] => Some(map_fallible(facts, parser, fact)?),
                _ => return error!(span, "could not parse run options"),
            };

            Schedule::Run(span, RunConfig { ruleset, until })
        }
        _ => return error!(span, "expected either saturate, seq, repeat, or run"),
    })
}

fn actions(sexp: &Sexp, parser: &Parser) -> Result<Vec<Action>, ParseError> {
    let (head, tail, span) = sexp.expect_call("action")?;

    if let Some(func) = parser.actions.get(&head) {
        return func(tail, span, parser);
    }

    Ok(match head.into() {
        "let" => match tail {
            [name, value] => vec![Action::Let(
                span,
                name.expect_atom("binding name")?,
                expr(value, parser)?,
            )],
            _ => return error!(span, "usage: (let <name> <expr>)"),
        },
        "set" => match tail {
            [call, value] => {
                let (func, args, _) = call.expect_call("table lookup")?;
                let args = map_fallible(args, parser, expr)?;
                let value = expr(value, parser)?;
                vec![Action::Set(span, func, args, value)]
            }
            _ => return error!(span, "usage: (set (<table name> <expr>*) <expr>)"),
        },
        "delete" => match tail {
            [call] => {
                let (func, args, _) = call.expect_call("table lookup")?;
                let args = map_fallible(args, parser, expr)?;
                vec![Action::Change(span, Change::Delete, func, args)]
            }
            _ => return error!(span, "usage: (delete (<table name> <expr>*))"),
        },
        "subsume" => match tail {
            [call] => {
                let (func, args, _) = call.expect_call("table lookup")?;
                let args = map_fallible(args, parser, expr)?;
                vec![Action::Change(span, Change::Subsume, func, args)]
            }
            _ => return error!(span, "usage: (subsume (<table name> <expr>*))"),
        },
        "union" => match tail {
            [e1, e2] => vec![Action::Union(span, expr(e1, parser)?, expr(e2, parser)?)],
            _ => return error!(span, "usage: (union <expr> <expr>)"),
        },
        "panic" => match tail {
            [message] => vec![Action::Panic(span, message.expect_string("error message")?)],
            _ => return error!(span, "usage: (panic <string>)"),
        },
        "extract" => match tail {
            [e] => vec![Action::Extract(
                span.clone(),
                expr(e, parser)?,
                Expr::Lit(span, Literal::Int(0)),
            )],
            [e, v] => vec![Action::Extract(span, expr(e, parser)?, expr(v, parser)?)],
            _ => return error!(span, "usage: (extract <expr> <number of variants>?)"),
        },
        _ => vec![Action::Expr(span, expr(sexp, parser)?)],
    })
}

fn fact(sexp: &Sexp, parser: &Parser) -> Result<Fact, ParseError> {
    let (head, tail, span) = sexp.expect_call("fact")?;

    Ok(match head.into() {
        "=" => match tail {
            [e1, e2] => Fact::Eq(span, expr(e1, parser)?, expr(e2, parser)?),
            _ => return error!(span, "usage: (= <expr> <expr>)"),
        },
        _ => Fact::Fact(expr(sexp, parser)?),
    })
}

fn expr(sexp: &Sexp, parser: &Parser) -> Result<Expr, ParseError> {
    Ok(match sexp {
        Sexp::Literal(literal, span) => Expr::Lit(span.clone(), literal.clone()),
        Sexp::Atom(symbol, span) => Expr::Var(span.clone(), *symbol),
        Sexp::List(list, span) => match list.as_slice() {
            [] => Expr::Lit(span.clone(), Literal::Unit),
            _ => {
                let (head, tail, span) = sexp.expect_call("call expression")?;

                if let Some(func) = parser.exprs.get(&head) {
                    return func(tail, span, parser);
                }

                Expr::Call(span.clone(), head, map_fallible(tail, parser, expr)?)
            }
        },
    })
}

#[derive(Clone, Debug)]
struct Context {
    source: Arc<SrcFile>,
    index: usize,
}

impl Context {
    fn new(name: Option<String>, contents: &str) -> Context {
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

fn all_sexps(mut ctx: Context) -> Result<Vec<Sexp>, ParseError> {
    let mut sexps = Vec::new();
    while !ctx.is_at_end() {
        sexps.push(sexp(&mut ctx)?);
    }
    Ok(sexps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = crate::ast::parse_expr(None, s, &Default::default()).unwrap();
        assert_eq!(format!("{}", e), s);
    }

    #[test]
    fn rust_span_display() {
        assert_eq!(format!("{}", span!()), "At 1037:34 of src/ast/parse.rs");
    }

    #[test]
    fn test_parser_macros() {
        let mut parser = Parser::default();
        parser.exprs.insert("qqqq".into(), |tail, span, macros| {
            Ok(Expr::Call(
                span,
                "xxxx".into(),
                map_fallible(tail, macros, expr)?,
            ))
        });
        let s = r#"(f (qqqq a 3) 4.0 (H "hello"))"#;
        let t = r#"(f (xxxx a 3) 4.0 (H "hello"))"#;
        let e = crate::ast::parse_expr(None, s, &parser).unwrap();
        assert_eq!(format!("{}", e), t);
    }
}
