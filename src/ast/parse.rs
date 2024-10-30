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
        let end = self.0.get_location((self.2 - 1).max(self.1));
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

trait Parser<T>: Fn(&Context) -> Res<T> + Clone {
    fn map<U>(self, f: impl Fn(T, Span) -> U + Clone) -> impl Parser<U> {
        move |ctx| {
            let (x, span, next) = self(ctx)?;
            Ok((f(x, span.clone()), span, next))
        }
    }
}
impl<T, F: Fn(&Context) -> Res<T> + Clone> Parser<T> for F {}

fn ident(ctx: &Context) -> Res<Symbol> {
    let mut span = ctx.span();
    if ctx.index >= ctx.source.contents.len() {
        return Err(ParseError::EndOfFile(span));
    }

    let mut next = ctx.clone();
    loop {
        match next.current_char() {
            None => break,
            Some(c) if c.is_alphanumeric() => {}
            Some(c) if "-+*/?!=<>&|^/%_.".contains(c) => {}
            Some(_) => break,
        }
        next.advance_char();
    }
    span.2 = next.index;

    if span.1 == span.2 {
        Err(ParseError::Ident(span))
    } else {
        next.advance_past_whitespace();
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

fn sequence<T, U>(a: impl Parser<T>, b: impl Parser<U>) -> impl Parser<(T, U)> {
    move |ctx| {
        let (x, lo, next) = a(ctx)?;
        let (y, hi, next) = b(&next)?;
        Ok(((x, y), Span(lo.0, lo.1, hi.2), next))
    }
}

macro_rules! sequences {
    ( $x:expr ) => { $x };
    ( $x:expr , $( $xs:expr ),+ $(,)? ) => {
        sequence( $x, sequences!( $( $xs ),+ ) )
    };
}

fn sequence3<T, U, V>(
    a: impl Parser<T>,
    b: impl Parser<U>,
    c: impl Parser<V>,
) -> impl Parser<(T, U, V)> {
    sequences!(a, b, c).map(|(a, (b, c)), _| (a, b, c))
}

fn sequence4<T, U, V, W>(
    a: impl Parser<T>,
    b: impl Parser<U>,
    c: impl Parser<V>,
    d: impl Parser<W>,
) -> impl Parser<(T, U, V, W)> {
    sequences!(a, b, c, d).map(|(a, (b, (c, d))), _| (a, b, c, d))
}

fn option<T>(parser: impl Parser<T>) -> impl Parser<Option<T>> {
    move |ctx| match parser(ctx) {
        Ok((x, span, next)) => Ok((Some(x), span, next)),
        Err(_) => Ok((None, ctx.span(), ctx.clone())),
    }
}

fn parens<T>(f: impl Parser<T>) -> impl Parser<T> {
    choice(
        sequence3(text("["), f.clone(), text("]")),
        sequence3(text("("), f.clone(), text(")")),
    )
    .map(|((), x, ()), _| x)
}

fn program(ctx: &Context) -> Res<Vec<Command>> {
    repeat_until(command, |ctx| ctx.index == ctx.source.contents.len())(ctx)
}

fn rec_datatype(ctx: &Context) -> Res<(Span, Symbol, Subdatatypes)> {
    choice(
        parens(sequence3(
            text("sort"),
            ident,
            parens(sequence(ident, repeat_until_end_paren(expr))),
        ))
        .map(|((), name, (head, exprs)), span| (span, name, Subdatatypes::NewSort(head, exprs))),
        parens(sequence(ident, repeat_until_end_paren(variant)))
            .map(|(name, variants), span| (span, name, Subdatatypes::Variants(variants))),
    )(ctx)
}

fn list<T>(f: impl Parser<T>) -> impl Parser<Vec<T>> {
    parens(repeat_until_end_paren(f))
}

fn snd<T, U>(x: Option<(T, U)>, _span: Span) -> Option<U> {
    x.map(|(_, x)| x)
}

fn ident_after_paren(ctx: &Context) -> &str {
    match sequence(choice(text("("), text("[")), ident)(ctx) {
        Ok((((), symbol), _, _)) => symbol.into(),
        Err(_) => "",
    }
}

fn command(ctx: &Context) -> Res<Command> {
    match ident_after_paren(ctx) {
        "set-option" => {
            parens(sequence3(text("set-option"), ident, expr))
                .map(|((), name, value), _| Command::SetOption { name, value })(ctx)
        }
        "datatype" => parens(sequence3(
            text("datatype"),
            ident,
            repeat_until_end_paren(variant),
        ))
        .map(|((), name, variants), span| Command::Datatype {
            span,
            name,
            variants,
        })(ctx),
        "sort" => choice(
            parens(sequence3(
                text("sort"),
                ident,
                parens(sequence(ident, repeat_until_end_paren(expr))),
            ))
            .map(|((), name, (head, tail)), span| Command::Sort(span, name, Some((head, tail)))),
            parens(sequence(text("sort"), ident))
                .map(|((), name), span| Command::Sort(span, name, None)),
        )(ctx),
        "datatype*" => {
            parens(sequence(
                text("datatype*"),
                repeat_until_end_paren(rec_datatype),
            ))
            .map(|((), datatypes), span| Command::Datatypes { span, datatypes })(ctx)
        }
        "function" => parens(sequences!(
            text("function"),
            ident,
            schema,
            cost,
            option(text(":unextractable")).map(|x, _| x.is_some()),
            option(sequence(text(":on_merge"), list(action))).map(snd),
            option(sequence(text(":merge"), expr)).map(snd),
            option(sequence(text(":default"), expr)).map(snd),
        ))
        .map(
            |((), (name, (schema, (cost, (unextractable, (merge_action, (merge, default))))))),
             span| {
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
            },
        )(ctx),
        "relation" => parens(sequence3(text("relation"), ident, list(r#type))).map(
            |((), constructor, inputs), span| Command::Relation {
                span,
                constructor,
                inputs,
            },
        )(ctx),
        "ruleset" => parens(sequence(text("ruleset"), ident))
            .map(|((), name), _| Command::AddRuleset(name))(ctx),
        "unstable-combined-ruleset" => parens(sequence3(
            text("unstable-combined-ruleset"),
            ident,
            repeat_until_end_paren(ident),
        ))
        .map(|((), name, subrulesets), _| Command::UnstableCombinedRuleset(name, subrulesets))(
            ctx
        ),
        "rule" => parens(sequences!(
            text("rule"),
            list(fact),
            list(action).map(|x, _| Actions::new(x)),
            option(sequence(text(":ruleset"), ident)).map(snd),
            option(sequence(text(":name"), string)).map(snd),
        ))
        .map(
            |((), (body, (head, (ruleset, name)))), span| Command::Rule {
                ruleset: ruleset.unwrap_or("".into()),
                name: name.unwrap_or("".to_string()).into(),
                rule: Rule { span, head, body },
            },
        )(ctx),
        "rewrite" => parens(sequences!(
            text("rewrite"),
            expr,
            expr,
            option(text(":subsume")).map(|x, _| x.is_some()),
            option(sequence(text(":when"), list(fact))).map(snd),
            option(sequence(text(":ruleset"), ident)).map(snd),
        ))
        .map(
            |((), (lhs, (rhs, (subsume, (conditions, ruleset))))), span| {
                Command::Rewrite(
                    ruleset.unwrap_or("".into()),
                    Rewrite {
                        span,
                        lhs,
                        rhs,
                        conditions: conditions.unwrap_or_default(),
                    },
                    subsume,
                )
            },
        )(ctx),
        "birewrite" => parens(sequences!(
            text("birewrite"),
            expr,
            expr,
            option(sequence(text(":when"), list(fact))).map(snd),
            option(sequence(text(":ruleset"), ident)).map(snd),
        ))
        .map(|((), (lhs, (rhs, (conditions, ruleset)))), span| {
            Command::BiRewrite(
                ruleset.unwrap_or("".into()),
                Rewrite {
                    span,
                    lhs,
                    rhs,
                    conditions: conditions.unwrap_or_default(),
                },
            )
        })(ctx),
        "let" => parens(sequence3(text("let"), ident, expr))
            .map(|((), name, expr), span| Command::Action(Action::Let(span, name, expr)))(
            ctx
        ),
        "run" => choice(
            parens(sequence3(
                text("run"),
                unum,
                option(sequence(text(":until"), repeat_until_end_paren(fact))).map(snd),
            ))
            .map(|((), limit, until), span| {
                Command::RunSchedule(Schedule::Repeat(
                    span.clone(),
                    limit,
                    Box::new(Schedule::Run(
                        span,
                        RunConfig {
                            ruleset: "".into(),
                            until,
                        },
                    )),
                ))
            }),
            parens(sequence4(
                text("run"),
                ident,
                unum,
                option(sequence(text(":until"), repeat_until_end_paren(fact))).map(snd),
            ))
            .map(|((), ruleset, limit, until), span| {
                Command::RunSchedule(Schedule::Repeat(
                    span.clone(),
                    limit,
                    Box::new(Schedule::Run(span, RunConfig { ruleset, until })),
                ))
            }),
        )(ctx),
        "simplify" => {
            parens(sequence3(text("simplify"), schedule, expr)).map(|((), schedule, expr), span| {
                Command::Simplify {
                    span,
                    expr,
                    schedule,
                }
            })(ctx)
        }
        "query-extract" => parens(sequence3(
            text("query-extract"),
            option(sequence(text(":variants"), unum)).map(snd),
            expr,
        ))
        .map(|((), variants, expr), span| Command::QueryExtract {
            span,
            expr,
            variants: variants.unwrap_or(0),
        })(ctx),
        "check" => parens(sequence(text("check"), repeat_until_end_paren(fact)))
            .map(|((), facts), span| Command::Check(span, facts))(ctx),
        "run-schedule" => parens(sequence(
            text("run-schedule"),
            repeat_until_end_paren(schedule),
        ))
        .map(|((), scheds), span| Command::RunSchedule(Schedule::Sequence(span, scheds)))(
            ctx
        ),
        "print-stats" => {
            parens(text("print-stats")).map(|(), _| Command::PrintOverallStatistics)(ctx)
        }
        "push" => parens(sequence(text("push"), option(unum)))
            .map(|((), n), _| Command::Push(n.unwrap_or(1)))(ctx),
        "pop" => parens(sequence(text("pop"), option(unum)))
            .map(|((), n), span| Command::Pop(span, n.unwrap_or(1)))(ctx),
        "print-function" => {
            parens(sequence3(text("print-function"), ident, unum))
                .map(|((), sym, n), span| Command::PrintFunction(span, sym, n))(ctx)
        }
        "print-size" => parens(sequence(text("print-size"), option(ident)))
            .map(|((), sym), span| Command::PrintSize(span, sym))(ctx),
        "input" => {
            parens(sequence3(text("input"), ident, string))
                .map(|((), name, file), span| Command::Input { span, name, file })(ctx)
        }
        "output" => parens(sequence4(
            text("output"),
            string,
            expr,
            repeat_until_end_paren(expr),
        ))
        .map(|((), file, e, mut exprs), span| {
            exprs.insert(0, e);
            Command::Output { span, file, exprs }
        })(ctx),
        "fail" => parens(sequence(text("fail"), command))
            .map(|((), c), span| Command::Fail(span, Box::new(c)))(ctx),
        "include" => parens(sequence(text("include"), string))
            .map(|((), file), span| Command::Include(span, file))(ctx),
        _ => non_let_action.map(|action, _| Command::Action(action))(ctx),
    }
}

fn schedule(ctx: &Context) -> Res<Schedule> {
    match ident_after_paren(ctx) {
        "saturate" => parens(sequence(text("saturate"), repeat_until_end_paren(schedule))).map(
            |((), scheds), span| {
                Schedule::Saturate(span.clone(), Box::new(Schedule::Sequence(span, scheds)))
            },
        )(ctx),
        "seq" => parens(sequence(text("seq"), repeat_until_end_paren(schedule)))
            .map(|((), scheds), span| Schedule::Sequence(span, scheds))(ctx),
        "repeat" => parens(sequence3(
            text("repeat"),
            unum,
            repeat_until_end_paren(schedule),
        ))
        .map(|((), limit, scheds), span| {
            Schedule::Repeat(
                span.clone(),
                limit,
                Box::new(Schedule::Sequence(span, scheds)),
            )
        })(ctx),
        "run" => choice(
            parens(sequence(
                text("run"),
                option(sequence(text(":until"), repeat_until_end_paren(fact))).map(snd),
            ))
            .map(|((), until), span| {
                Schedule::Run(
                    span,
                    RunConfig {
                        ruleset: "".into(),
                        until,
                    },
                )
            }),
            parens(sequence3(
                text("run"),
                ident,
                option(sequence(text(":until"), repeat_until_end_paren(fact))).map(snd),
            ))
            .map(|((), ruleset, until), span| Schedule::Run(span, RunConfig { ruleset, until })),
        )(ctx),
        _ => ident.map(|ruleset, span| {
            Schedule::Run(
                span.clone(),
                RunConfig {
                    ruleset,
                    until: None,
                },
            )
        })(ctx),
    }
}

fn cost(ctx: &Context) -> Res<Option<usize>> {
    option(sequence(text(":cost"), unum)).map(snd)(ctx)
}

fn action(ctx: &Context) -> Res<Action> {
    choice(
        parens(sequence3(text("let"), ident, expr))
            .map(|((), name, expr), span| Action::Let(span, name, expr)),
        non_let_action,
    )(ctx)
}

fn non_let_action(ctx: &Context) -> Res<Action> {
    match ident_after_paren(ctx) {
        "set" => parens(sequence3(
            text("set"),
            parens(sequence(ident, repeat_until_end_paren(expr))),
            expr,
        ))
        .map(|((), (f, args), v), span| Action::Set(span, f, args, v))(ctx),
        "delete" => parens(sequence(
            text("delete"),
            parens(sequence(ident, repeat_until_end_paren(expr))),
        ))
        .map(|((), (f, args)), span| Action::Change(span, Change::Delete, f, args))(
            ctx
        ),
        "subsume" => parens(sequence(
            text("subsume"),
            parens(sequence(ident, repeat_until_end_paren(expr))),
        ))
        .map(|((), (f, args)), span| Action::Change(span, Change::Subsume, f, args))(
            ctx
        ),
        "union" => parens(sequence3(text("union"), expr, expr))
            .map(|((), e1, e2), span| Action::Union(span, e1, e2))(ctx),
        "panic" => parens(sequence(text("panic"), string))
            .map(|(_, msg), span| Action::Panic(span, msg))(ctx),
        "extract" => choice(
            parens(sequence(text("extract"), expr)).map(|((), expr), span| {
                Action::Extract(span.clone(), expr, Expr::Lit(span, Literal::Int(0)))
            }),
            parens(sequence3(text("extract"), expr, expr))
                .map(|((), expr, variants), span| Action::Extract(span, expr, variants)),
        )(ctx),
        _ => call_expr.map(|e, span| Action::Expr(span, e))(ctx),
    }
}

fn fact(ctx: &Context) -> Res<Fact> {
    let (call_expr, span, next) = call_expr(ctx)?;
    match call_expr {
        Expr::Call(_, head, ref tail) => {
            let fact = match head.into() {
                "=" if tail.len() < 2 => return Err(ParseError::EqFactLt2(span)),
                "=" => Fact::Eq(span.clone(), tail.clone()),
                _ => Fact::Fact(call_expr),
            };
            Ok((fact, span, next))
        }
        _ => unreachable!(),
    }
}

fn schema(ctx: &Context) -> Res<Schema> {
    sequence(list(r#type), r#type).map(|(input, output), _| Schema { input, output })(ctx)
}

fn expr(ctx: &Context) -> Res<Expr> {
    choices!(
        call_expr,
        literal.map(|literal, span| Expr::Lit(span, literal)),
        ident.map(|ident, span| Expr::Var(span, ident)),
    )(ctx)
}

fn literal(ctx: &Context) -> Res<Literal> {
    choices!(
        sequence(text("("), text(")")).map(|((), ()), _| Literal::Unit),
        num.map(|x, _| Literal::Int(x)),
        r#f64.map(|x, _| Literal::F64(x)),
        r#bool.map(|x, _| Literal::Bool(x)),
        string.map(|x, _| Literal::String(x.into())),
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
    parens(sequence(ident, repeat_until_end_paren(expr)))
        .map(|(head, tail), span| Expr::Call(span, head, tail))(ctx)
}

fn variant(ctx: &Context) -> Res<Variant> {
    parens(sequence3(
        ident,
        repeat_until(r#type, |ctx| r#type(ctx).is_err()),
        cost,
    ))
    .map(|(name, types, cost), span| Variant {
        span,
        name,
        types,
        cost,
    })(ctx)
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
    use std::num::FpCategory::*;
    let (_, span, next) = ident(ctx)?;
    match span.string() {
        "NaN" => Ok((OrderedFloat(f64::NAN), span, next)),
        "inf" => Ok((OrderedFloat(f64::INFINITY), span, next)),
        "-inf" => Ok((OrderedFloat(f64::NEG_INFINITY), span, next)),
        _ => match span.string().parse::<f64>() {
            Err(_) => Err(ParseError::Float(span)),
            // Rust will parse "infinity" as a float, which we don't want
            // we're only using `parse` to avoid implementing it ourselves anyway
            Ok(x) => match x.classify() {
                Nan | Infinite => Err(ParseError::Float(span)),
                Zero | Subnormal | Normal => Ok((OrderedFloat(x), span, next)),
            },
        },
    }
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
    #[error("{0}\nexpected \"{1}\"")]
    Text(Span, String),
    #[error("{0}\nexpected string")]
    String(Span),
    #[error("{0}\nmissing end quote for string")]
    MissingEndQuote(Span),
    #[error("{0}\nunexpected end of file")]
    EndOfFile(Span),
    #[error("{0}\nexpected identifier")]
    Ident(Span),
    #[error("{0}\nexpected integer literal")]
    Int(Span),
    #[error("{0}\nexpected unsigned integer literal")]
    Uint(Span),
    #[error("{0}\nexpected floating point literal")]
    Float(Span),
    #[error("{0}\nexpected boolean literal")]
    Bool(Span),
    #[error("{0}\nusing = with less than two arguments is not allowed")]
    EqFactLt2(Span),
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = crate::ast::parse_expr(None, s).unwrap();
        assert_eq!(format!("{}", e), s);
    }
}
