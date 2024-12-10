//! Parse a string into egglog.

use crate::*;
use ordered_float::OrderedFloat;

pub fn parse_program(filename: Option<String>, input: &str) -> Result<Vec<Command>, ParseError> {
    let (sexps, _span, rest) = all_sexps(&Context::new(filename, input))?;
    assert!(rest.is_at_end(), "did not parse entire program");
    sexps.iter().map(command).collect()
}

// currently only used for testing, but no reason it couldn't be used elsewhere later
pub fn parse_expr(filename: Option<String>, input: &str) -> Result<Expr, ParseError> {
    let (sexp, _span, rest) = sexp(&Context::new(filename, input))?;
    assert!(rest.is_at_end(), "did not parse entire expression");
    expr(&sexp)
}

/// A [`Span`] contains the file name and a pair of offsets representing the start and the end.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Span(Arc<SrcFile>, usize, usize);

lazy_static::lazy_static! {
    pub static ref DUMMY_SPAN: Span = Span(Arc::new(SrcFile {name: None, contents: String::new()}), 0, 0);
}

impl Span {
    pub fn string(&self) -> &str {
        &self.0.contents[self.1..self.2]
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SrcFile {
    name: Option<String>,
    contents: String,
}

struct Location {
    line: usize,
    col: usize,
}

impl SrcFile {
    pub fn get_location(&self, offset: usize) -> Location {
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
        let start = self.0.get_location(self.1);
        let end = self.0.get_location((self.2.saturating_sub(1)).max(self.1));
        let quote = self.string();
        match (&self.0.name, start.line == end.line) {
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
            (None, true) => write!(f, "In {}:{}-{}: {quote}", start.line, start.col, end.col),
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

enum Sexp {
    Literal(Literal, Span),
    Atom(Symbol, Span),
    // `List`s are always nonempty, since an
    // empty list is actually a `Literal::Unit`
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
        } else if let Sexp::Literal(Literal::Unit, _) = self {
            return Ok(&[]);
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
    func: impl Fn(&Sexp) -> Result<T, ParseError>,
) -> Result<Vec<T>, ParseError> {
    slice.iter().map(func).collect::<Result<_, _>>()
}

fn command(sexp: &Sexp) -> Result<Command, ParseError> {
    let (head, tail, span) = sexp.expect_call("command")?;

    Ok(match head.into() {
        "set-option" => match tail {
            [name, value] => Command::SetOption {
                name: name.expect_atom("option name")?,
                value: expr(value)?,
            },
            _ => return error!(span, "usage: (set-option <name> <value>)"),
        },
        "sort" => match tail {
            [name] => Command::Sort(span, name.expect_atom("sort name")?, None),
            [name, call] => {
                let (func, args, _) = call.expect_call("container sort declaration")?;
                Command::Sort(
                    span,
                    name.expect_atom("sort name")?,
                    Some((func, map_fallible(args, expr)?)),
                )
            }
            _ => {
                return error!(
                    span,
                    "usages:\n(sort <name>)\n(sort <name> (<container sort> <argument sort>*))"
                )
            }
        },
        "datatype" => match tail {
            [name, variants @ ..] => Command::Datatype {
                span,
                name: name.expect_atom("sort name")?,
                variants: map_fallible(variants, variant)?,
            },
            _ => return error!(span, "usage: (datatype <name> <variant>*)"),
        },
        "datatype*" => Command::Datatypes {
            span,
            datatypes: map_fallible(tail, rec_datatype)?,
        },
        "function" => match tail {
            [name, inputs, output, merge @ ..] => Command::Function {
                name: name.expect_atom("function name")?,
                schema: schema(inputs, output)?,
                merge: match merge {
                    [Sexp::Atom(o, _)] if *o == ":no-merge".into() => None,
                    [Sexp::Atom(o, _), e] if *o == ":merge".into() => Some(expr(e)?),
                    [] => return error!(span, "functions are required to specify merge behaviour"),
                    _ => return error!(span, "could not parse function options"),
                },
                span,
            },
            _ => {
                let a = "(function <name> (<input sort>*) <output sort> :merge <expr>)";
                let b = "(function <name> (<input sort>*) <output sort> :no-merge)";
                return error!(span, "usages:\n{a}\n{b}");
            }
        },
        "constructor" => match tail {
            [name, inputs, output, options @ ..] => {
                let mut cost = None;
                let mut unextractable = false;

                match options {
                    [] => {}
                    [Sexp::Atom(o, _)] if *o == ":unextractable".into() => unextractable = true,
                    [Sexp::Atom(o, _), c] if *o == ":cost".into() => {
                        cost = Some(c.expect_uint("cost")?)
                    }
                    _ => return error!(span, "could not parse constructor options"),
                }

                Command::Constructor {
                    span,
                    name: name.expect_atom("constructor name")?,
                    schema: schema(inputs, output)?,
                    cost,
                    unextractable,
                }
            }
            _ => {
                let a = "(constructor <name> (<input sort>*) <output sort>)";
                let b = "(constructor <name> (<input sort>*) <output sort> :cost <cost>)";
                let c = "(constructor <name> (<input sort>*) <output sort> :unextractable)";
                return error!(span, "usages:\n{a}\n{b}\n{c}");
            }
        },
        "relation" => match tail {
            [name, inputs] => Command::Relation {
                span,
                name: name.expect_atom("relation name")?,
                inputs: map_fallible(inputs.expect_list("input sorts")?, |sexp| {
                    sexp.expect_atom("input sort")
                })?,
            },
            _ => return error!(span, "usage: (relation <name> (<input sort>*))"),
        },
        "ruleset" => match tail {
            [name] => Command::AddRuleset(name.expect_atom("ruleset name")?),
            _ => return error!(span, "usage: (ruleset <name>)"),
        },
        "unstable-combined-ruleset" => match tail {
            [name, subrulesets @ ..] => Command::UnstableCombinedRuleset(
                name.expect_atom("combined ruleset name")?,
                map_fallible(subrulesets, |sexp| sexp.expect_atom("subruleset name"))?,
            ),
            _ => {
                return error!(
                    span,
                    "usage: (unstable-combined-ruleset <name> <child ruleset>*)"
                )
            }
        },
        "rule" => match tail {
            [lhs, rhs, options @ ..] => {
                let body = map_fallible(lhs.expect_list("rule query")?, fact)?;
                let head = map_fallible(rhs.expect_list("rule actions")?, action)?;
                let head = GenericActions(head);

                let mut ruleset = "".into();
                let mut name = "".into();
                let mut i = 0;
                while i < options.len() {
                    match options[i].expect_atom("rule option")?.into() {
                        ":ruleset" => {
                            i += 1;
                            ruleset = options[i].expect_atom("ruleset name")?;
                        }
                        ":name" => {
                            i += 1;
                            name = options[i].expect_string("rule name")?.into();
                        }
                        _ => return error!(options[i].span(), "could not parse rule option"),
                    }
                    i += 1;
                }

                Command::Rule {
                    ruleset,
                    name,
                    rule: Rule { span, head, body },
                }
            }
            _ => return error!(span, "usage: (rule (<fact>*) (<action>*) <option>*)"),
        },
        "rewrite" => match tail {
            [lhs, rhs, options @ ..] => {
                let lhs = expr(lhs)?;
                let rhs = expr(rhs)?;

                let mut ruleset = "".into();
                let mut conditions = Vec::new();
                let mut subsume = false;
                let mut i = 0;
                while i < options.len() {
                    match options[i].expect_atom("rewrite option")?.into() {
                        ":ruleset" => {
                            i += 1;
                            ruleset = options[i].expect_atom("ruleset name")?;
                        }
                        ":subsume" => subsume = true,
                        ":when" => {
                            i += 1;
                            conditions =
                                map_fallible(options[i].expect_list("rewrite conditions")?, fact)?;
                        }
                        _ => return error!(options[i].span(), "could not parse rewrite option"),
                    }
                    i += 1;
                }

                Command::Rewrite(
                    ruleset,
                    Rewrite {
                        span,
                        lhs,
                        rhs,
                        conditions,
                    },
                    subsume,
                )
            }
            _ => return error!(span, "usage: (rewrite <expr> <expr> <option>*)"),
        },
        "birewrite" => match tail {
            [lhs, rhs, options @ ..] => {
                let lhs = expr(lhs)?;
                let rhs = expr(rhs)?;

                let mut ruleset = "".into();
                let mut conditions = Vec::new();
                let mut i = 0;
                while i < options.len() {
                    match options[i].expect_atom("rewrite option")?.into() {
                        ":ruleset" => {
                            i += 1;
                            ruleset = options[i].expect_atom("ruleset name")?;
                        }
                        ":when" => {
                            i += 1;
                            conditions =
                                map_fallible(options[i].expect_list("rewrite conditions")?, fact)?;
                        }
                        _ => return error!(options[i].span(), "could not parse rewrite option"),
                    }
                    i += 1;
                }

                Command::BiRewrite(
                    ruleset,
                    Rewrite {
                        span,
                        lhs,
                        rhs,
                        conditions,
                    },
                )
            }
            _ => return error!(span, "usage: (birewrite <expr> <expr> <option>*)"),
        },
        "run" => {
            if tail.is_empty() {
                return error!(span, "usage: (run <ruleset>? <uint> <:until (<fact>*)>?)");
            }

            let has_ruleset = tail.len() >= 2 && tail[1].expect_uint("").is_ok();

            let (ruleset, limit, options) = if has_ruleset {
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

            let until = match options {
                [] => None,
                [Sexp::Atom(o, _), facts @ ..] if *o == ":until".into() => {
                    Some(map_fallible(facts, fact)?)
                }
                _ => return error!(span, "could not parse run options"),
            };

            Command::RunSchedule(Schedule::Repeat(
                span.clone(),
                limit,
                Box::new(Schedule::Run(span, RunConfig { ruleset, until })),
            ))
        }
        "run-schedule" => {
            Command::RunSchedule(Schedule::Sequence(span, map_fallible(tail, schedule)?))
        }
        "simplify" => match tail {
            [s, e] => Command::Simplify {
                span,
                schedule: schedule(s)?,
                expr: expr(e)?,
            },
            _ => return error!(span, "usage: (simplify <schedule> <expr>)"),
        },
        "query-extract" => match tail {
            [options @ .., e] => {
                let variants = match options {
                    [] => 0,
                    [Sexp::Atom(o, _), v] if *o == ":variants".into() => {
                        v.expect_uint("number of variants")?
                    }
                    _ => return error!(span, "could not parse query-extract options"),
                };
                Command::QueryExtract {
                    span,
                    expr: expr(e)?,
                    variants,
                }
            }
            _ => return error!(span, "usage: (query-extract <:variants <uint>>? <expr>)"),
        },
        "check" => Command::Check(span, map_fallible(tail, fact)?),
        "push" => match tail {
            [] => Command::Push(1),
            [n] => Command::Push(n.expect_uint("number of times to push")?),
            _ => return error!(span, "usage: (push <uint>?)"),
        },
        "pop" => match tail {
            [] => Command::Pop(span, 1),
            [n] => Command::Pop(span, n.expect_uint("number of times to pop")?),
            _ => return error!(span, "usage: (pop <uint>?)"),
        },
        "print-stats" => match tail {
            [] => Command::PrintOverallStatistics,
            _ => return error!(span, "usage: (print-stats)"),
        },
        "print-function" => match tail {
            [name, rows] => Command::PrintFunction(
                span,
                name.expect_atom("table name")?,
                rows.expect_uint("number of rows to print")?,
            ),
            _ => {
                return error!(
                    span,
                    "usage: (print-function <table name> <number of rows>)"
                )
            }
        },
        "print-size" => match tail {
            [] => Command::PrintSize(span, None),
            [name] => Command::PrintSize(span, Some(name.expect_atom("table name")?)),
            _ => return error!(span, "usage: (print-size <table name>?)"),
        },
        "input" => match tail {
            [name, file] => Command::Input {
                span,
                name: name.expect_atom("table name")?,
                file: file.expect_string("file name")?,
            },
            _ => return error!(span, "usage: (input <table name> \"<file name>\")"),
        },
        "output" => match tail {
            [file, exprs @ ..] => Command::Output {
                span,
                file: file.expect_string("file name")?,
                exprs: map_fallible(exprs, expr)?,
            },
            _ => return error!(span, "usage: (output <file name> <expr>+)"),
        },
        "include" => match tail {
            [file] => Command::Include(span, file.expect_string("file name")?),
            _ => return error!(span, "usage: (include <file name>)"),
        },
        "fail" => match tail {
            [subcommand] => Command::Fail(span, Box::new(command(subcommand)?)),
            _ => return error!(span, "usage: (fail <command>)"),
        },
        _ => Command::Action(action(sexp)?),
    })
}

fn schema(input: &Sexp, output: &Sexp) -> Result<Schema, ParseError> {
    Ok(Schema {
        input: map_fallible(input.expect_list("input sorts")?, |sexp| {
            sexp.expect_atom("input sort")
        })?,
        output: output.expect_atom("output sort")?,
    })
}

fn rec_datatype(sexp: &Sexp) -> Result<(Span, Symbol, Subdatatypes), ParseError> {
    let (head, tail, span) = sexp.expect_call("datatype")?;

    Ok(match head.into() {
        "sort" => match tail {
            [name, call] => {
                let name = name.expect_atom("sort name")?;
                let (func, args, _) = call.expect_call("container sort declaration")?;
                let args = map_fallible(args, expr)?;
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
            let variants = map_fallible(tail, variant)?;
            (span, head, Subdatatypes::Variants(variants))
        }
    })
}

fn variant(sexp: &Sexp) -> Result<Variant, ParseError> {
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
        types: map_fallible(types, |sexp| sexp.expect_atom("variant argument type"))?,
        cost,
    })
}

fn schedule(sexp: &Sexp) -> Result<Schedule, ParseError> {
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
            Box::new(Schedule::Sequence(span, map_fallible(tail, schedule)?)),
        ),
        "seq" => Schedule::Sequence(span, map_fallible(tail, schedule)?),
        "repeat" => match tail {
            [limit, tail @ ..] => Schedule::Repeat(
                span.clone(),
                limit.expect_uint("number of iterations")?,
                Box::new(Schedule::Sequence(span, map_fallible(tail, schedule)?)),
            ),
            _ => return error!(span, "usage: (repeat <number of iterations> <schedule>*)"),
        },
        "run" => {
            let has_ruleset = match tail.first() {
                None => false,
                Some(Sexp::Atom(o, _)) if *o == ":until".into() => false,
                _ => true,
            };

            let (ruleset, options) = if has_ruleset {
                (tail[0].expect_atom("ruleset name")?, &tail[1..])
            } else {
                ("".into(), tail)
            };

            let until = match options {
                [] => None,
                [Sexp::Atom(o, _), facts @ ..] if *o == ":until".into() => {
                    Some(map_fallible(facts, fact)?)
                }
                _ => return error!(span, "could not parse run options"),
            };

            Schedule::Run(span, RunConfig { ruleset, until })
        }
        _ => return error!(span, "expected either saturate, seq, repeat, or run"),
    })
}

fn action(sexp: &Sexp) -> Result<Action, ParseError> {
    let (head, tail, span) = sexp.expect_call("action")?;

    Ok(match head.into() {
        "let" => match tail {
            [name, value] => Action::Let(span, name.expect_atom("binding name")?, expr(value)?),
            _ => return error!(span, "usage: (let <name> <expr>)"),
        },
        "set" => match tail {
            [call, value] => {
                let (func, args, _) = call.expect_call("table lookup")?;
                let args = map_fallible(args, expr)?;
                let value = expr(value)?;
                Action::Set(span, func, args, value)
            }
            _ => return error!(span, "usage: (set (<table name> <expr>*) <expr>)"),
        },
        "delete" => match tail {
            [call] => {
                let (func, args, _) = call.expect_call("table lookup")?;
                let args = map_fallible(args, expr)?;
                Action::Change(span, Change::Delete, func, args)
            }
            _ => return error!(span, "usage: (delete (<table name> <expr>*))"),
        },
        "subsume" => match tail {
            [call] => {
                let (func, args, _) = call.expect_call("table lookup")?;
                let args = map_fallible(args, expr)?;
                Action::Change(span, Change::Subsume, func, args)
            }
            _ => return error!(span, "usage: (subsume (<table name> <expr>*))"),
        },
        "union" => match tail {
            [e1, e2] => Action::Union(span, expr(e1)?, expr(e2)?),
            _ => return error!(span, "usage: (union <expr> <expr>)"),
        },
        "panic" => match tail {
            [message] => Action::Panic(span, message.expect_string("error message")?),
            _ => return error!(span, "usage: (panic <string>)"),
        },
        "extract" => match tail {
            [e] => Action::Extract(span.clone(), expr(e)?, Expr::Lit(span, Literal::Int(0))),
            [e, v] => Action::Extract(span, expr(e)?, expr(v)?),
            _ => return error!(span, "usage: (extract <expr> <number of variants>?)"),
        },
        _ => Action::Expr(span, expr(sexp)?),
    })
}

fn fact(sexp: &Sexp) -> Result<Fact, ParseError> {
    let (head, tail, span) = sexp.expect_call("fact")?;

    Ok(match head.into() {
        "=" => match tail {
            [_, _, ..] => Fact::Eq(span, map_fallible(tail, expr)?),
            _ => return error!(span, "using = with less than two arguments is not allowed"),
        },
        _ => Fact::Fact(expr(sexp)?),
    })
}

fn expr(sexp: &Sexp) -> Result<Expr, ParseError> {
    Ok(match sexp {
        Sexp::Literal(literal, span) => Expr::Lit(span.clone(), literal.clone()),
        Sexp::Atom(symbol, span) => Expr::Var(span.clone(), *symbol),
        Sexp::List(..) => {
            let (func, args, span) = sexp.expect_call("call expression")?;
            Expr::Call(span.clone(), func, map_fallible(args, expr)?)
        }
    })
}

#[derive(Clone, Debug)]
struct Context {
    source: Arc<SrcFile>,
    index: usize,
}

impl Context {
    fn new(name: Option<String>, contents: &str) -> Context {
        let mut next = Context {
            source: Arc::new(SrcFile {
                name,
                contents: contents.to_string(),
            }),
            index: 0,
        };
        next.advance_past_whitespace();
        next
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

    fn span(&self) -> Span {
        Span(self.source.clone(), self.index, self.index)
    }
}

type Res<T> = Result<(T, Span, Context), ParseError>;

trait Parser<T>: Fn(&Context) -> Res<T> + Clone {}
impl<T, F: Fn(&Context) -> Res<T> + Clone> Parser<T> for F {}

fn ident(ctx: &Context) -> Res<Symbol> {
    let mut span = ctx.span();
    if ctx.index >= ctx.source.contents.len() {
        return error!(span, "unexpected end of file");
    }

    let mut next = ctx.clone();
    loop {
        match next.current_char() {
            None => break,
            Some(c) if c.is_alphanumeric() => {}
            Some(c) if "-+*/?!=<>&|^/%_.:".contains(c) => {}
            Some(_) => break,
        }
        next.advance_char();
    }
    span.2 = next.index;

    if span.1 == span.2 {
        error!(span, "expected identifier")
    } else {
        next.advance_past_whitespace();
        Ok((Symbol::from(span.string()), span, next))
    }
}

fn text(s: &'static str) -> impl Parser<()> {
    move |ctx| {
        let mut span = ctx.span();
        span.2 = (span.1 + s.len()).min(ctx.source.contents.len());

        if span.string() == s {
            let mut next = ctx.clone();
            next.index += s.len();
            next.advance_past_whitespace();
            Ok(((), span, next))
        } else {
            error!(span, "expected \"{s}\"")
        }
    }
}

fn repeat_until<T>(
    inner: impl Parser<T>,
    end: impl Fn(&Context) -> bool + Clone,
) -> impl Parser<Vec<T>> {
    move |ctx| {
        let mut vec = Vec::new();
        let mut span = ctx.span();
        let mut next = ctx.clone();
        while !end(&next) {
            let (x, s, n) = inner(&next)?;
            vec.push(x);
            next = n;
            span.2 = s.2;
        }
        Ok((vec, span, next))
    }
}

fn repeat_until_end_paren<T>(inner: impl Parser<T>) -> impl Parser<Vec<T>> {
    repeat_until(inner, |ctx| text(")")(ctx).is_ok())
}

fn choice<T>(a: impl Parser<T>, b: impl Parser<T>) -> impl Parser<T> {
    move |ctx| a(ctx).or_else(|_| b(ctx))
}

macro_rules! choices {
    ( $x:expr ) => { $x };
    ( $x:expr , $( $xs:expr ),+ $(,)? ) => {
        choice( $x, choices!( $( $xs ),+ ) )
    };
}

fn map<T, U>(parser: impl Parser<T>, f: impl Fn(T, Span) -> U + Clone) -> impl Parser<U> {
    move |ctx| {
        let (x, span, next) = parser(ctx)?;
        Ok((f(x, span.clone()), span, next))
    }
}

fn sequence<T, U>(a: impl Parser<T>, b: impl Parser<U>) -> impl Parser<(T, U)> {
    move |ctx| {
        let (x, lo, next) = a(ctx)?;
        let (y, hi, next) = b(&next)?;
        Ok(((x, y), Span(lo.0, lo.1, hi.2), next))
    }
}

fn parens<T>(f: impl Parser<T>) -> impl Parser<T> {
    map(
        choice(
            sequence(text("["), sequence(f.clone(), text("]"))),
            sequence(text("("), sequence(f, text(")"))),
        ),
        |((), (x, ())), _| x,
    )
}

fn all_sexps(ctx: &Context) -> Res<Vec<Sexp>> {
    repeat_until(sexp, |ctx| ctx.index == ctx.source.contents.len())(ctx)
}

fn sexp(ctx: &Context) -> Res<Sexp> {
    choices!(
        map(literal, Sexp::Literal),
        map(ident, Sexp::Atom),
        map(parens(repeat_until_end_paren(sexp)), Sexp::List),
    )(ctx)
}

fn literal(ctx: &Context) -> Res<Literal> {
    choices!(
        map(sequence(text("("), text(")")), |((), ()), _| Literal::Unit),
        map(num, |x, _| Literal::Int(x)),
        map(r#f64, |x, _| Literal::F64(x)),
        map(r#bool, |x, _| Literal::Bool(x)),
        map(string, |x, _| Literal::String(x.into())),
    )(ctx)
}

fn r#bool(ctx: &Context) -> Res<bool> {
    let (_, span, next) = ident(ctx)?;
    match span.string() {
        "true" => Ok((true, span, next)),
        "false" => Ok((false, span, next)),
        _ => error!(span, "expected boolean literal"),
    }
}

fn num(ctx: &Context) -> Res<i64> {
    let (_, span, next) = ident(ctx)?;
    match span.string().parse::<i64>() {
        Ok(x) => Ok((x, span, next)),
        Err(_) => error!(span, "expected integer literal"),
    }
}

fn r#f64(ctx: &Context) -> Res<OrderedFloat<f64>> {
    use std::num::FpCategory::*;
    let (_, span, next) = ident(ctx)?;
    match span.string() {
        "NaN" => Ok((OrderedFloat(f64::NAN), span, next)),
        "inf" => Ok((OrderedFloat(f64::INFINITY), span, next)),
        "-inf" => Ok((OrderedFloat(f64::NEG_INFINITY), span, next)),
        _ => match span.string().parse::<f64>() {
            Err(_) => error!(span, "expected floating point literal"),
            // Rust will parse "infinity" as a float, which we don't want
            // we're only using `parse` to avoid implementing it ourselves anyway
            Ok(x) => match x.classify() {
                Nan | Infinite => error!(span, "expected floating point literal"),
                Zero | Subnormal | Normal => Ok((OrderedFloat(x), span, next)),
            },
        },
    }
}

fn string(ctx: &Context) -> Res<String> {
    let mut span = Span(ctx.source.clone(), ctx.index, ctx.index);
    if ctx.current_char() != Some('"') {
        return error!(span, "expected string literal");
    }

    let mut next = ctx.clone();
    let mut in_escape = false;

    next.advance_char();
    loop {
        match next.current_char() {
            None => {
                span.2 = next.index;
                return error!(span, "string is missing end quote");
            }
            Some('"') if !in_escape => break,
            Some('\\') if !in_escape => in_escape = true,
            Some(_) => in_escape = false,
        }

        next.advance_char();
    }
    next.advance_char();

    span.2 = next.index;

    next.advance_past_whitespace();

    let s = span.string();
    let s = &s[1..s.len() - 1];
    Ok((s.to_string(), span, next))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = crate::ast::parse_expr(None, s).unwrap();
        assert_eq!(format!("{}", e), s);
    }

    #[test]
    fn dummy_span_display() {
        assert_eq!(format!("{}", *super::DUMMY_SPAN), "In 1:1-1: ");
    }
}
