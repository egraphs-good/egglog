//! Parse a string into egglog.

use crate::*;
use ordered_float::OrderedFloat;

pub fn parse_program(filename: Option<String>, input: &str) -> Result<Vec<Command>, ParseError> {
    let (out, _span, rest) = program(&Context::new(filename, input))?;
    assert!(rest.is_at_end(), "did not parse entire program");
    Ok(out)
}

// currently only used for testing, but no reason it couldn't be used elsewhere later
pub fn parse_expr(filename: Option<String>, input: &str) -> Result<Expr, ParseError> {
    let (out, _span, rest) = expr(&Context::new(filename, input))?;
    assert!(rest.is_at_end(), "did not parse entire expression");
    Ok(out)
}

/// A [`Span`] contains the file name and a pair of offsets representing the start and the end.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        for (i, c) in self.contents.chars().enumerate() {
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

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start = self.0.get_location(self.1);
        let end = self.0.get_location(self.2 - 1);
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
        return Err(ParseError::Ident(span));
    }

    let mut next = ctx.clone();
    loop {
        match next.current_char() {
            None => break,
            Some(c) if c.is_alphanumeric() => {}
            Some(c) if "-+*/?!=<>&|^/%_".contains(c) => {}
            Some(_) => break,
        }
        next.advance_char();
    }
    span.2 = next.index;

    next.advance_past_whitespace();

    if span.1 == span.2 {
        Err(ParseError::Ident(span))
    } else {
        Ok((Symbol::from(span.string()), span, next))
    }
}

fn text(s: &str) -> impl Parser<()> + '_ {
    move |ctx| {
        let mut span = ctx.span();
        span.2 = (span.1 + s.len()).min(ctx.source.contents.len());

        if span.string() == s {
            let mut next = ctx.clone();
            next.index += s.len();
            next.advance_past_whitespace();
            Ok(((), span, next))
        } else {
            Err(ParseError::Text(span, s.to_string()))
        }
    }
}

fn repeat<T>(parser: impl Parser<T>) -> impl Parser<Vec<T>> {
    move |ctx| {
        let mut vec = Vec::new();
        let mut span = ctx.span();
        let mut next = ctx.clone();
        while let Ok((x, s, n)) = parser(&next) {
            vec.push(x);
            next = n;
            span.2 = s.2;
        }
        Ok((vec, span, next))
    }
}

fn repeat_all<T>(parser: impl Parser<T>) -> impl Parser<Vec<T>> {
    move |ctx| {
        let mut vec = Vec::new();
        let mut span = ctx.span();
        let mut next = ctx.clone();
        while next.index < next.source.contents.len() {
            let (x, s, n) = parser(&next)?;
            vec.push(x);
            next = n;
            span.2 = s.2;
        }
        Ok((vec, span, next))
    }
}

fn choice<T>(a: impl Parser<T>, b: impl Parser<T>) -> impl Parser<T> {
    move |ctx| match (a(ctx), b(ctx)) {
        (Ok(x), _) | (_, Ok(x)) => Ok(x),
        (_, Err(x)) => Err(x),
    }
}

macro_rules! choices {
    ( $x:expr , ) => { $x };
    ( $x:expr $( , $xs:expr )+ , ) => {
        choice( $x, choices!( $( $xs , )+ ) )
    };
}

fn map<T, U>(parser: impl Parser<T>, f: impl Fn(T, Span) -> U + Clone) -> impl Parser<U> {
    move |ctx| {
        let (x, span, next) = parser(ctx)?;
        Ok((f(x, span.clone()), span, next))
    }
}

fn wrap_err<T>(
    parser: impl Parser<T>,
    f: impl Fn(Box<ParseError>) -> ParseError + Clone,
) -> impl Parser<T> {
    move |ctx| parser(ctx).map_err(|e| f(Box::new(e)))
}

fn sequence<T, U>(a: impl Parser<T>, b: impl Parser<U>) -> impl Parser<(T, U)> {
    move |ctx| {
        let (x, lo, next) = a(ctx)?;
        let (y, hi, next) = b(&next)?;
        Ok(((x, y), Span(lo.0, lo.1, hi.2), next))
    }
}

fn sequence3<T, U, V>(
    a: impl Parser<T>,
    b: impl Parser<U>,
    c: impl Parser<V>,
) -> impl Parser<(T, U, V)> {
    map(sequence(a, sequence(b, c)), |(a, (b, c)), _| (a, b, c))
}

macro_rules! sequences {
    ( $x:expr , ) => { $x };
    ( $x:expr $( , $xs:expr )+ , ) => {
        sequence( $x, sequences!( $( $xs , )+ ) )
    };
}

fn option<T>(parser: impl Parser<T>) -> impl Parser<Option<T>> {
    move |ctx| match parser(ctx) {
        Ok((x, span, next)) => Ok((Some(x), span, next)),
        Err(_) => Ok((None, ctx.span(), ctx.clone())),
    }
}

fn parens<T>(f: impl Parser<T>) -> impl Parser<T> {
    map(choice(
        sequence3(text("["), f.clone(), text("]")),
        sequence3(text("("), f.clone(), text(")")),
    ), |((), x, ()), _| x)
}

fn program(ctx: &Context) -> Res<Vec<Command>> {
    repeat_all(command)(ctx)
}

fn rec_datatype(ctx: &Context) -> Res<(Span, Symbol, Subdatatypes)> {
    choices!(
        map(
            parens(sequence(ident, repeat(variant))),
            |(name, variants), span| (span, name, Subdatatypes::Variants(variants))
        ),
        map(
            parens(sequence3(
                text("sort"),
                ident,
                parens(sequence(ident, repeat(expr))),
            )),
            |(_, name, (head, exprs)), span| (span, name, Subdatatypes::NewSort(head, exprs))
        ),
    )(ctx)
}

fn list<T>(f: impl Parser<T>) -> impl Parser<Vec<T>> {
    parens(repeat(f))
}

fn snd<T, U>(x: Option<(T, U)>, _span: Span) -> Option<U> {
    x.map(|(_, x)| x)
}

fn command(ctx: &Context) -> Res<Command> {
    wrap_err(
        choices!(
            map(
                parens(sequence3(text("set-option"), ident, expr)),
                |(_, name, value), _| Command::SetOption { name, value }
            ),
            map(
                parens(sequence3(text("datatype"), ident, repeat(variant))),
                |(_, name, variants), span| Command::Datatype {
                    span,
                    name,
                    variants
                }
            ),
            map(
                parens(sequence3(
                    text("sort"),
                    ident,
                    parens(sequence(ident, repeat(expr)))
                )),
                |(_, name, (head, tail)), span| Command::Sort(span, name, Some((head, tail)))
            ),
            map(
                parens(sequence(text("sort"), ident)),
                |(_, name), span| Command::Sort(span, name, None),
            ),
            map(
                parens(sequence(text("datatype*"), repeat(rec_datatype))),
                |(_, datatypes), span| Command::Datatypes { span, datatypes }
            ),
            map(
                parens(sequences!(
                    text("function"),
                    ident,
                    schema,
                    cost,
                    map(option(text(":unextractable")), |x, _| x.is_some()),
                    map(option(sequence(text(":on_merge"), list(action))), snd),
                    map(option(sequence(text(":merge"), expr)), snd),
                    map(option(sequence(text(":default"), expr)), snd),
                )),
                |
                    (
                        _,
                        (name, (schema, (cost, (unextractable, (merge_action, (merge, default)))))),
                    ), span| {
                    Command::Function(FunctionDecl {
                        span,
                        name,
                        schema,
                        merge,
                        merge_action: Actions::new(merge_action.unwrap_or_default()),
                        default,
                        cost,
                        unextractable,
                        ignore_viz: false,
                    })
                }
            ),
            map(
                parens(sequence3(text("relation"), ident, list(r#type))),
                |(_, constructor, inputs), span| Command::Relation {
                    span,
                    constructor,
                    inputs
                },
            ),
            map(parens(sequence(text("ruleset"), ident)), |(_, name), _| {
                Command::AddRuleset(name)
            },),
            map(
                parens(sequence3(
                    text("unstable-combined-ruleset"),
                    ident,
                    repeat(ident)
                )),
                |(_, name, subrulesets), _| Command::UnstableCombinedRuleset(name, subrulesets),
            ),
            map(
                parens(sequences!(
                    text("rule"),
                    list(fact),
                    map(list(action), |x, _| Actions::new(x)),
                    map(option(sequence(text(":ruleset"), ident)), snd),
                    map(option(sequence(text(":name"), string)), snd),
                )),
                |(_, (body, (head, (ruleset, name)))), span| Command::Rule {
                    ruleset: ruleset.unwrap_or("".into()),
                    name: name.unwrap_or("".to_string()).into(),
                    rule: Rule { span, head, body },
                }
            ),
            map(
                parens(sequences!(
                    text("rewrite"),
                    expr,
                    expr,
                    map(option(text(":subsume")), |x, _| x.is_some()),
                    map(option(sequence(text(":when"), list(fact))), snd),
                    map(option(sequence(text(":ruleset"), ident)), snd),
                )),
                |(_, (lhs, (rhs, (subsume, (conditions, ruleset))))), span| Command::Rewrite(
                    ruleset.unwrap_or("".into()),
                    Rewrite {
                        span,
                        lhs,
                        rhs,
                        conditions: conditions.unwrap_or_default()
                    },
                    subsume
                )
            ),
            map(
                parens(sequences!(
                    text("birewrite"),
                    expr,
                    expr,
                    map(option(sequence(text(":when"), list(fact))), snd),
                    map(option(sequence(text(":ruleset"), ident)), snd),
                )),
                |(_, (lhs, (rhs, (conditions, ruleset)))), span| Command::BiRewrite(
                    ruleset.unwrap_or("".into()),
                    Rewrite {
                        span,
                        lhs,
                        rhs,
                        conditions: conditions.unwrap_or_default()
                    }
                )
            ),
            map(
                parens(sequence3(text("let"), ident, expr)),
                |(_, name, expr), span| Command::Action(Action::Let(span, name, expr)),
            ),
            map(non_let_action, |action, _| Command::Action(action)),
            map(
                parens(sequence3(
                    text("run"),
                    unum,
                    map(option(sequence(text(":until"), repeat(fact))), snd),
                )),
                |(_, limit, until), span| Command::RunSchedule(Schedule::Repeat(
                    span.clone(),
                    limit,
                    Box::new(Schedule::Run(
                        span,
                        RunConfig {
                            ruleset: "".into(),
                            until
                        }
                    ))
                )),
            ),
            map(
                parens(sequences!(
                    text("run"),
                    ident,
                    unum,
                    map(option(sequence(text(":until"), repeat(fact))), snd),
                )),
                |(_, (ruleset, (limit, until))), span| Command::RunSchedule(Schedule::Repeat(
                    span.clone(),
                    limit,
                    Box::new(Schedule::Run(span, RunConfig { ruleset, until }))
                )),
            ),
            map(
                parens(sequence3(text("simplify"), schedule, expr)),
                |((), schedule, expr), span| Command::Simplify {
                    span,
                    expr,
                    schedule
                },
            ),
            map(
                parens(sequence3(
                    text("query-extract"),
                    map(option(sequence(text(":variants"), unum)), snd),
                    expr
                )),
                |(_, variants, expr), span| Command::QueryExtract {
                    span,
                    expr,
                    variants: variants.unwrap_or(0)
                }
            ),
            map(
                parens(sequence(text("check"), repeat(fact))),
                |(_, facts), span| Command::Check(span, facts),
            ),
            map(
                parens(sequence(text("run-schedule"), repeat(schedule))),
                |(_, scheds), span| Command::RunSchedule(Schedule::Sequence(span, scheds)),
            ),
            map(parens(text("print-stats")), |_, _| {
                Command::PrintOverallStatistics
            }),
            map(parens(sequence(text("push"), option(unum))), |(_, n), _| {
                Command::Push(n.unwrap_or(1))
            },),
            map(
                parens(sequence(text("pop"), option(unum))),
                |(_, n), span| Command::Pop(span, n.unwrap_or(1)),
            ),
            map(
                parens(sequence3(text("print-function"), ident, unum)),
                |(_, sym, n), span| Command::PrintFunction(span, sym, n),
            ),
            map(
                parens(sequence(text("print-size"), option(ident))),
                |(_, sym), span| Command::PrintSize(span, sym),
            ),
            map(
                parens(sequence3(text("input"), ident, string)),
                |(_, name, file), span| Command::Input { span, name, file }
            ),
            map(
                parens(sequences!(text("output"), string, expr, repeat(expr),)),
                |(_, (file, (e, mut exprs))), span| {
                    exprs.insert(0, e);
                    Command::Output { span, file, exprs }
                }
            ),
            map(
                parens(sequence(text("fail"), command)),
                |(_, c), span| Command::Fail(span, Box::new(c)),
            ),
            map(
                parens(sequence(text("include"), string)),
                |(_, file), span| Command::Include(span, file),
            ),
        ),
        ParseError::Command,
    )(ctx)
}

fn schedule(ctx: &Context) -> Res<Schedule> {
    wrap_err(
        choices!(
            map(
                parens(sequence(text("saturate"), repeat(schedule))),
                |(_, scheds), span| Schedule::Saturate(
                    span.clone(),
                    Box::new(Schedule::Sequence(span, scheds),)
                ),
            ),
            map(
                parens(sequence(text("seq"), repeat(schedule))),
                |(_, scheds), span| Schedule::Sequence(span, scheds),
            ),
            map(
                parens(sequence3(text("repeat"), unum, repeat(schedule))),
                |(_, limit, scheds), span| Schedule::Repeat(
                    span.clone(),
                    limit,
                    Box::new(Schedule::Sequence(span, scheds),)
                )
            ),
            map(
                parens(sequence(
                    text("run"),
                    map(option(sequence(text(":until"), repeat(fact))), snd)
                )),
                |(_, until), span| Schedule::Run(
                    span,
                    RunConfig {
                        ruleset: "".into(),
                        until
                    }
                ),
            ),
            map(
                parens(sequence3(
                    text("run"),
                    ident,
                    map(option(sequence(text(":until"), repeat(fact))), snd)
                )),
                |(_, ruleset, until), span| Schedule::Run(span, RunConfig { ruleset, until })
            ),
            map(ident, |ruleset, span| Schedule::Run(
                span.clone(),
                RunConfig {
                    ruleset,
                    until: None
                }
            )),
        ),
        ParseError::Schedule,
    )(ctx)
}

fn cost(ctx: &Context) -> Res<Option<usize>> {
    map(option(sequence(text(":cost"), unum)), snd)(ctx)
}

fn action(ctx: &Context) -> Res<Action> {
    choice(
        map(
            parens(sequence3(text("let"), ident, expr)),
            |(_, name, expr), span| Action::Let(span, name, expr),
        ),
        non_let_action,
    )(ctx)
}

fn non_let_action(ctx: &Context) -> Res<Action> {
    choices!(
        map(
            parens(sequence3(
                text("set"),
                parens(sequence(ident, repeat(expr))),
                expr
            )),
            |(_, (f, args), v), span| Action::Set(span, f, args, v),
        ),
        map(
            parens(sequence(
                text("delete"),
                parens(sequence(ident, repeat(expr)))
            )),
            |(_, (f, args)), span| Action::Change(span, Change::Delete, f, args),
        ),
        map(
            parens(sequence(
                text("subsume"),
                parens(sequence(ident, repeat(expr)))
            )),
            |(_, (f, args)), span| Action::Change(span, Change::Subsume, f, args),
        ),
        map(
            parens(sequence3(text("union"), expr, expr)),
            |(_, e1, e2), span| Action::Union(span, e1, e2),
        ),
        map(
            parens(sequence(text("panic"), string)),
            |(_, msg), span| Action::Panic(span, msg),
        ),
        map(
            parens(sequence(text("extract"), expr)),
            |(_, expr), span| Action::Extract(
                span.clone(),
                expr,
                Expr::Lit(span, Literal::Int(0))
            ),
        ),
        map(
            parens(sequence3(text("extract"), expr, expr)),
            |(_, expr, variants), span| Action::Extract(span, expr, variants),
        ),
        map(parens(call_expr), |e, span| Action::Expr(span, e),),
    )(ctx)
}

fn fact(ctx: &Context) -> Res<Fact> {
    wrap_err(
        choice(
            map(
                parens(sequences!(text("="), expr, expr, repeat(expr),)),
                |(_, (e1, (e2, mut es))), span| {
                    es.insert(0, e1);
                    es.insert(1, e2);
                    Fact::Eq(span, es)
                },
            ),
            map(call_expr, |x, _| Fact::Fact(x)),
        ),
        ParseError::Fact,
    )(ctx)
}

fn schema(ctx: &Context) -> Res<Schema> {
    map(sequence(list(r#type), r#type), |(input, output), _| Schema {
        input,
        output,
    })(ctx)
}

fn expr(ctx: &Context) -> Res<Expr> {
    wrap_err(
        choices!(
            map(literal, |lit, span| Expr::Lit(span, lit)),
            map(ident, |id, span| Expr::Var(span, id)),
            call_expr,
        ),
        ParseError::Expr,
    )(ctx)
}

fn literal(ctx: &Context) -> Res<Literal> {
    wrap_err(
        choices!(
            map(sequence(text("("), text(")")), |_, _| Literal::Unit,),
            map(num, |x, _| Literal::Int(x)),
            map(r#f64, |x, _| Literal::F64(x)),
            map(r#bool, |x, _| Literal::Bool(x)),
            map(sym_string, |x, _| Literal::String(x)),
        ),
        ParseError::Literal,
    )(ctx)
}

fn r#bool(ctx: &Context) -> Res<bool> {
    let (_, span, next) = ident(ctx)?;
    match span.string() {
        "true" => Ok((true, span, next)),
        "false" => Ok((false, span, next)),
        _ => Err(ParseError::Bool(span)),
    }
}

fn call_expr(ctx: &Context) -> Res<Expr> {
    map(
        parens(sequence(ident, repeat(expr))),
        |(head, tail), span| Expr::Call(span, head, tail),
    )(ctx)
}

fn variant(ctx: &Context) -> Res<Variant> {
    map(
        parens(sequence3(ident, repeat(r#type), cost)),
        |(name, types, cost), span| Variant {
            span,
            name,
            types,
            cost,
        },
    )(ctx)
}

fn r#type(ctx: &Context) -> Res<Symbol> {
    ident(ctx)
}

fn num(ctx: &Context) -> Res<i64> {
    let (_, span, next) = ident(ctx)?;
    match span.string().parse::<i64>() {
        Ok(x) => Ok((x, span, next)),
        Err(_) => Err(ParseError::Int(span)),
    }
}

fn unum(ctx: &Context) -> Res<usize> {
    let (_, span, next) = ident(ctx)?;
    match span.string().parse::<usize>() {
        Ok(x) => Ok((x, span, next)),
        Err(_) => Err(ParseError::Int(span)),
    }
}

fn r#f64(ctx: &Context) -> Res<OrderedFloat<f64>> {
    let (_, span, next) = ident(ctx)?;
    match span.string() {
        "NaN" => Ok((OrderedFloat(f64::NAN), span, next)),
        "inf" => Ok((OrderedFloat(f64::INFINITY), span, next)),
        "-inf" => Ok((OrderedFloat(f64::NEG_INFINITY), span, next)),
        _ => match span.string().parse::<f64>() {
            Ok(x) => Ok((OrderedFloat(x), span, next)),
            Err(_) => Err(ParseError::Float(span)),
        },
    }
}

fn sym_string(ctx: &Context) -> Res<Symbol> {
    map(string, |x, _| Symbol::from(x))(ctx)
}

fn string(ctx: &Context) -> Res<String> {
    let mut span = Span(ctx.source.clone(), ctx.index, ctx.index);
    if ctx.current_char() != Some('"') {
        return Err(ParseError::String(span));
    }

    let mut next = ctx.clone();
    let mut in_escape = false;

    next.advance_char();
    loop {
        match next.current_char() {
            None => {
                span.2 = next.index;
                return Err(ParseError::MissingEndQuote(span));
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

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("{0}\nexpected \"{1}\", found \"{}\"", .0.string())]
    Text(Span, String),
    #[error("{0}\nexpected string")]
    String(Span),
    #[error("{0}\nmissing end quote for string")]
    MissingEndQuote(Span),
    #[error("{0}\nunexpected end of file")]
    Ident(Span),
    #[error("{0}\nexpected integer literal")]
    Int(Span),
    #[error("{0}\nexpected unsigned integer literal")]
    Uint(Span),
    #[error("{0}\nexpected floating point literal")]
    Float(Span),
    #[error("{0}\nexpected boolean literal")]
    Bool(Span),
    #[error("{0}\nwhen parsing command")]
    Command(Box<ParseError>),
    #[error("{0}\nwhen parsing schedule")]
    Schedule(Box<ParseError>),
    #[error("{0}\nwhen parsing action")]
    Action(Box<ParseError>),
    #[error("{0}\nwhen parsing fact")]
    Fact(Box<ParseError>),
    #[error("{0}\nwhen parsing expr")]
    Expr(Box<ParseError>),
    #[error("{0}\nwhen parsing literal")]
    Literal(Box<ParseError>),
}

#[cfg(test)]
mod tests {
    use crate::*;

    fn parse_expr(s: &str) -> Expr {
        crate::ast::parse_expr(None, s).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn test_parser() {
        let _ = parse_expr("(f a b)");
    }

    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = parse_expr(s);
        assert_eq!(format!("{}", e), s);
    }
}
