//! Parse a string into egglog.

use crate::*;

pub fn parse_program(filename: Option<String>, input: &str) -> Result<Vec<Command>, ParseError> {
    let (out, rest) = program(&Context::new(filename, input))?;
    assert!(rest.is_at_end(), "did not parse entire program");
    Ok(out)
}

// currently only used for testing, but no reason it couldn't be used elsewhere later
pub fn parse_expr(filename: Option<String>, input: &str) -> Result<Expr, ParseError> {
    let (out, rest) = expr(&Context::new(filename, input))?;
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
    pub fn new(name: Option<String>, contents: &str) -> Context {
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

    pub fn advance_text(&self, s: &str, skip_whitespace: bool) -> Option<(Span, Context)> {
        if self.source.contents[self.index..].starts_with(s) {
            let mut next = self.clone();
            next.index += s.len();
            if skip_whitespace {
                next.advance_past_whitespace();
            }
            Some((Span(self.source.clone(), self.index, next.index), next))
        } else {
            None
        }
    }

    fn advance_past_whitespace(&mut self) {
        let mut iter = self.source.contents[self.index..].char_indices();
        loop {
            let end = match iter.next() {
                Some((_, c)) if c.is_whitespace() => None,
                Some((_, ';')) => {
                    loop {
                        if matches!(iter.next(), Some((_, '\n' | '\r'))) {
                            break;
                        }
                    }
                    None
                }
                Some((i, _)) => Some(i),
                None => Some(self.source.contents.len()),
            };
            if let Some(i) = end {
                self.index = i;
                return;
            }
        }
    }

    pub fn is_at_end(&self) -> bool {
        self.index == self.source.contents.len()
    }
}

type Res<T> = Result<(T, Context), ParseError>;

trait Parser<T>: Fn(&Context) -> Res<T> + Clone {}
impl<T, F: Fn(&Context) -> Res<T> + Clone> Parser<T> for F {}

fn text_exact(s: &str) -> impl Parser<Span> + '_ {
    text_internal(s, false)
}

fn text(s: &str) -> impl Parser<Span> + '_ {
    text_internal(s, true)
}

fn text_internal(s: &str, skip_whitespace: bool) -> impl Parser<Span> + '_ {
    move |ctx| {
        if let Some((span, next)) = ctx.advance_text(s, skip_whitespace) {
            Ok((span, next))
        } else {
            let span = Span(ctx.source.clone(), ctx.index, ctx.index);
            Err(ParseError::ExpectedText(span, s.to_string()))
        }
    }
}

fn repeat<T>(parser: impl Parser<T>) -> impl Parser<Vec<T>> {
    move |ctx| {
        let mut vec = Vec::new();
        let mut next = ctx.clone();
        while let Ok((x, rest)) = parser(&next) {
            vec.push(x);
            next = rest;
        }
        Ok((vec, next))
    }
}

fn repeat1<T>(parser: impl Parser<T>) -> impl Parser<Vec<T>> {
    move |ctx| {
        let (x, next) = (parser.clone())(ctx)?;
        let (mut xs, next) = repeat(parser.clone())(&next)?;
        xs.insert(0, x);
        Ok((xs, next))
    }
}

fn repeat_all<T>(parser: impl Parser<T>) -> impl Parser<Vec<T>> {
    move |ctx| {
        let mut vec = Vec::new();
        let mut next = ctx.clone();
        while next.index < next.source.contents.len() {
            let (x, rest) = parser(&next)?;
            vec.push(x);
            next = rest;
        }
        Ok((vec, next))
    }
}

fn choice<T>(a: impl Parser<T>, b: impl Parser<T>) -> impl Parser<T> {
    move |ctx| a(ctx).or_else(|_| b(ctx))
}

macro_rules! choices {
    ( $x:expr , ) => { $x };
    ( $x:expr $( , $xs:expr )+ , ) => {
        choice( $x, choices!( $( $xs , )+ ) )
    };
}

fn map<T, U>(parser: impl Parser<T>, f: impl Fn(T) -> U + Clone) -> impl Parser<U> {
    move |ctx| {
        let (x, next) = parser(ctx)?;
        Ok((f(x), next))
    }
}

fn sequence<T, U>(a: impl Parser<T>, b: impl Parser<U>) -> impl Parser<(T, U)> {
    move |ctx| {
        let (x, next) = a(ctx)?;
        let (y, next) = b(&next)?;
        Ok(((x, y), next))
    }
}

fn sequence3<T, U, V>(
    a: impl Parser<T>,
    b: impl Parser<U>,
    c: impl Parser<V>,
) -> impl Parser<(T, U, V)> {
    move |ctx| {
        let (x, next) = a(ctx)?;
        let (y, next) = b(&next)?;
        let (z, next) = c(&next)?;
        Ok(((x, y, z), next))
    }
}

macro_rules! sequences {
    ( $x:expr , ) => { $x };
    ( $x:expr $( , $xs:expr )+ , ) => {
        sequence( $x, sequences!( $( $xs , )+ ) )
    };
}

fn option<T>(parser: impl Parser<T>) -> impl Parser<Option<T>> {
    move |ctx| match parser(ctx) {
        Ok((x, next)) => Ok((Some(x), next)),
        Err(_) => Ok((None, ctx.clone())),
    }
}

fn parens_span<T>(f: impl Parser<T>) -> impl Parser<(Span, T)> {
    move |ctx| {
        let ((lo, x, hi), next) = choice(
            sequence3(text("("), f.clone(), text(")")),
            sequence3(text("["), f.clone(), text("]")),
        )(ctx)?;
        Ok(((Span(lo.0, lo.1, hi.2), x), next))
    }
}

fn parens<T>(f: impl Parser<T>) -> impl Parser<T> {
    move |ctx| {
        let ((_span, x), next) = parens_span(f.clone())(ctx)?;
        Ok((x, next))
    }
}

fn program(ctx: &Context) -> Res<Vec<Command>> {
    repeat_all(command)(ctx)
}

fn rec_datatype(ctx: &Context) -> Res<(Span, Symbol, Subdatatypes)> {
    choices!(
        map(
            parens_span(sequence(ident, repeat(variant))),
            |(span, (name, variants))| (span, name, Subdatatypes::Variants(variants))
        ),
        map(
            parens_span(sequence3(
                text("sort"),
                ident,
                parens(sequence(ident, repeat(expr))),
            )),
            |(span, (_, name, (head, exprs)))| (span, name, Subdatatypes::NewSort(head, exprs))
        ),
    )(ctx)
}

fn list<T>(f: impl Parser<T>) -> impl Parser<Vec<T>> {
    parens(repeat(f))
}

fn snd<T, U>(x: Option<(T, U)>) -> Option<U> {
    x.map(|(_, x)| x)
}

fn command(ctx: &Context) -> Res<Command> {
    choices!(
        map(
            parens(sequence3(text("set-option"), ident, expr)),
            |(_, name, value)| Command::SetOption { name, value }
        ),
        map(
            parens_span(sequence3(text("datatype"), ident, repeat(variant))),
            |(span, (_, name, variants))| Command::Datatype {
                span,
                name,
                variants
            }
        ),
        map(
            parens_span(sequence3(
                text("sort"),
                ident,
                parens(sequence(ident, repeat(expr)))
            )),
            |(span, (_, name, (head, tail)))| Command::Sort(span, name, Some((head, tail)))
        ),
        map(
            parens_span(sequence(text("sort"), ident)),
            |(span, (_, name))| Command::Sort(span, name, None),
        ),
        map(
            parens_span(sequence(text("datatype*"), repeat(rec_datatype))),
            |(span, (_, datatypes))| Command::Datatypes { span, datatypes }
        ),
        map(
            parens_span(sequences!(
                text("function"),
                ident,
                schema,
                cost,
                map(option(text(":unextractable")), |x| x.is_some()),
                map(option(sequence(text(":on_merge"), list(action))), snd),
                map(option(sequence(text(":merge"), expr)), snd),
                map(option(sequence(text(":default"), expr)), snd),
            )),
            |(
                span,
                (_, (name, (schema, (cost, (unextractable, (merge_action, (merge, default))))))),
            )| {
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
            parens_span(sequence3(text("relation"), ident, list(r#type))),
            |(span, (_, constructor, inputs))| Command::Relation {
                span,
                constructor,
                inputs
            },
        ),
        map(parens(sequence(text("ruleset"), ident)), |(_, name)| {
            Command::AddRuleset(name)
        },),
        map(
            parens(sequence3(
                text("unstable-combined-ruleset"),
                ident,
                repeat(ident)
            )),
            |(_, name, subrulesets)| Command::UnstableCombinedRuleset(name, subrulesets),
        ),
        map(
            parens_span(sequences!(
                text("rule"),
                list(fact),
                map(list(action), Actions::new),
                map(option(sequence(text(":ruleset"), ident)), snd),
                map(option(sequence(text(":name"), string)), snd),
            )),
            |(span, (_, (body, (head, (ruleset, name)))))| Command::Rule {
                ruleset: ruleset.unwrap_or("".into()),
                name: name.unwrap_or("".to_string()).into(),
                rule: Rule { span, head, body },
            }
        ),
        map(
            parens_span(sequences!(
                text("rewrite"),
                expr,
                expr,
                map(option(text(":subsume")), |x| x.is_some()),
                map(option(sequence(text(":when"), list(fact))), snd),
                map(option(sequence(text(":ruleset"), ident)), snd),
            )),
            |(span, (_, (lhs, (rhs, (subsume, (conditions, ruleset))))))| Command::Rewrite(
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
            parens_span(sequences!(
                text("birewrite"),
                expr,
                expr,
                map(option(sequence(text(":when"), list(fact))), snd),
                map(option(sequence(text(":ruleset"), ident)), snd),
            )),
            |(span, (_, (lhs, (rhs, (conditions, ruleset)))))| Command::BiRewrite(
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
            parens_span(sequence3(text("let"), ident, expr)),
            |(span, (_, name, expr))| Command::Action(Action::Let(span, name, expr)),
        ),
        map(non_let_action, Command::Action),
        map(
            parens_span(sequence3(
                text("run"),
                unum,
                map(option(sequence(text(":until"), repeat(fact))), snd),
            )),
            |(span, (_, limit, until))| Command::RunSchedule(Schedule::Repeat(
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
            parens_span(sequences!(
                text("run"),
                ident,
                unum,
                map(option(sequence(text(":until"), repeat(fact))), snd),
            )),
            |(span, (_, (ruleset, (limit, until))))| Command::RunSchedule(Schedule::Repeat(
                span.clone(),
                limit,
                Box::new(Schedule::Run(span, RunConfig { ruleset, until }))
            )),
        ),
        map(
            parens_span(sequence3(text("simplify"), schedule, expr)),
            |(span, (_, schedule, expr))| Command::Simplify {
                span,
                expr,
                schedule
            },
        ),
        map(
            parens_span(sequence3(
                text("query-extract"),
                map(option(sequence(text(":variants"), unum)), snd),
                expr
            )),
            |(span, (_, variants, expr))| Command::QueryExtract {
                span,
                expr,
                variants: variants.unwrap_or(0)
            }
        ),
        map(
            parens_span(sequence(text("check"), repeat(fact))),
            |(span, (_, facts))| Command::Check(span, facts),
        ),
        map(
            parens_span(sequence(text("run-schedule"), repeat(schedule))),
            |(span, (_, scheds))| Command::RunSchedule(Schedule::Sequence(span, scheds)),
        ),
        map(parens(text("print-stats")), |_| {
            Command::PrintOverallStatistics
        }),
        map(parens(sequence(text("push"), option(unum))), |(_, n)| {
            Command::Push(n.unwrap_or(1))
        },),
        map(
            parens_span(sequence(text("pop"), option(unum))),
            |(span, (_, n))| Command::Pop(span, n.unwrap_or(1)),
        ),
        map(
            parens_span(sequence3(text("print-function"), ident, unum)),
            |(span, (_, sym, n))| Command::PrintFunction(span, sym, n),
        ),
        map(
            parens_span(sequence(text("print-size"), option(ident))),
            |(span, (_, sym))| Command::PrintSize(span, sym),
        ),
        map(
            parens_span(sequence3(text("input"), ident, string)),
            |(span, (_, name, file))| Command::Input { span, name, file }
        ),
        map(
            parens_span(sequence3(text("output"), string, repeat1(expr))),
            |(span, (_, file, exprs))| Command::Output { span, file, exprs }
        ),
        map(
            parens_span(sequence(text("fail"), command)),
            |(span, (_, c))| Command::Fail(span, Box::new(c)),
        ),
        map(
            parens_span(sequence(text("include"), string)),
            |(span, (_, file))| Command::Include(span, file),
        ),
    )(ctx)
}

fn schedule(ctx: &Context) -> Res<Schedule> {
    choices!(
        map(
            parens_span(sequence(text("saturate"), repeat(schedule))),
            |(span, (_, scheds))| Schedule::Saturate(
                span.clone(),
                Box::new(Schedule::Sequence(span, scheds),)
            ),
        ),
        map(
            parens_span(sequence(text("seq"), repeat(schedule))),
            |(span, (_, scheds))| Schedule::Sequence(span, scheds),
        ),
        map(
            parens_span(sequence3(text("repeat"), unum, repeat(schedule))),
            |(span, (_, limit, scheds))| Schedule::Repeat(
                span.clone(),
                limit,
                Box::new(Schedule::Sequence(span, scheds),)
            )
        ),
        map(
            parens_span(sequence(
                text("run"),
                map(option(sequence(text(":until"), repeat(fact))), snd)
            )),
            |(span, (_, until))| Schedule::Run(
                span,
                RunConfig {
                    ruleset: "".into(),
                    until
                }
            ),
        ),
        map(
            parens_span(sequence3(
                text("run"),
                ident,
                map(option(sequence(text(":until"), repeat(fact))), snd)
            )),
            |(span, (_, ruleset, until))| Schedule::Run(span, RunConfig { ruleset, until })
        ),
        map(parens_span(ident), |(span, ruleset)| Schedule::Run(
            span,
            RunConfig {
                ruleset,
                until: None
            }
        )),
    )(ctx)
}

fn cost(ctx: &Context) -> Res<Option<usize>> {
    map(option(sequence(text(":cost"), unum)), snd)(ctx)
}

fn non_let_action(ctx: &Context) -> Res<Action> {
    choices!(
        map(
            parens_span(sequence3(
                text("set"),
                parens(sequence(ident, repeat(expr))),
                expr
            )),
            |(span, (_, (f, args), v))| Action::Set(span, f, args, v),
        ),
        map(
            parens_span(sequence(
                text("delete"),
                parens(sequence(ident, repeat(expr)))
            )),
            |(span, (_, (f, args)))| Action::Change(span, Change::Delete, f, args),
        ),
        map(
            parens_span(sequence(
                text("subsume"),
                parens(sequence(ident, repeat(expr)))
            )),
            |(span, (_, (f, args)))| Action::Change(span, Change::Subsume, f, args),
        ),
        map(
            parens_span(sequence3(text("union"), expr, expr)),
            |(span, (_, e1, e2))| Action::Union(span, e1, e2),
        ),
        map(
            parens_span(sequence(text("panic"), string)),
            |(span, (_, msg))| Action::Panic(span, msg),
        ),
        map(
            parens_span(sequence(text("extract"), expr)),
            |(span, (_, expr))| Action::Extract(
                span.clone(),
                expr,
                Expr::Lit(span, Literal::Int(0))
            ),
        ),
        map(
            parens_span(sequence3(text("extract"), expr, expr)),
            |(span, (_, expr, variants))| Action::Extract(span, expr, variants),
        ),
        map(parens_span(call_expr), |(span, e)| Action::Expr(span, e),),
    )(ctx)
}

fn action(ctx: &Context) -> Res<Action> {
    choice(
        map(
            parens_span(sequence3(text("let"), ident, expr)),
            |(span, (_, name, expr))| Action::Let(span, name, expr),
        ),
        non_let_action,
    )(ctx)
}

fn fact(ctx: &Context) -> Res<Fact> {
    choice(
        map(
            parens_span(sequence3(text("="), repeat1(expr), expr)),
            |(span, (_, mut es, e))| {
                es.push(e);
                Fact::Eq(span, es)
            },
        ),
        map(call_expr, Fact::Fact),
    )(ctx)
}

fn schema(ctx: &Context) -> Res<Schema> {
    map(sequence(list(r#type), r#type), |(input, output)| Schema {
        input,
        output,
    })(ctx)
}

fn expr(ctx: &Context) -> Res<Expr> {
    choices!(
        map(parens_span(literal), |(span, lit)| Expr::Lit(span, lit),),
        map(parens_span(ident), |(span, id)| Expr::Var(span, id),),
        call_expr,
    )(ctx)
}

fn literal(ctx: &Context) -> Res<Literal> {
    choices!(
        map(sequence(text("("), text(")")), |_| Literal::Unit,),
        map(num, Literal::Int),
        map(r#f64, Literal::F64),
        map(r#bool, Literal::Bool),
        map(sym_string, Literal::String),
    )(ctx)
}

fn r#bool(ctx: &Context) -> Res<bool> {
    choice(map(text("true"), |_| true), map(text("false"), |_| false))(ctx)
}

fn call_expr(ctx: &Context) -> Res<Expr> {
    map(
        parens_span(sequence(ident, repeat(expr))),
        |(span, (head, tail))| Expr::Call(span, head, tail),
    )(ctx)
}

fn variant(ctx: &Context) -> Res<Variant> {
    map(
        parens_span(sequence3(ident, repeat(r#type), cost)),
        |(span, (name, types, cost))| Variant {
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

fn digit(ctx: &Context) -> Res<Span> {
    choices!(
        text_exact("0"),
        text_exact("1"),
        text_exact("2"),
        text_exact("3"),
        text_exact("4"),
        text_exact("5"),
        text_exact("6"),
        text_exact("7"),
        text_exact("8"),
        text_exact("9"),
    )(ctx)
}

fn num(ctx: &Context) -> Res<i64> {
    let start = ctx.index;
    let (_, next) = sequence3(option(text_exact("-")), repeat1(digit), text(""))(ctx)?;
    let end = next.index;
    let i = ctx.source.contents[start..end]
        .parse()
        .map_err(|_| ParseError::ExpectedInt(Span(ctx.source.clone(), start, end)))?;
    Ok((i, next))
}

fn unum(ctx: &Context) -> Res<usize> {
    let start = ctx.index;
    let (_, next) = sequence(repeat1(digit), text(""))(ctx)?;
    let end = next.index;
    let i = ctx.source.contents[start..end]
        .parse()
        .map_err(|_| ParseError::ExpectedUint(Span(ctx.source.clone(), start, end)))?;
    Ok((i, next))
}

fn r#f64(ctx: &Context) -> Res<ordered_float::OrderedFloat<f64>> {
    choices!(
        map(text("NaN"), |_| ordered_float::OrderedFloat::<f64>(
            f64::NAN
        )),
        map(text("inf"), |_| ordered_float::OrderedFloat::<f64>(
            f64::INFINITY
        )),
        map(text("-inf"), |_| ordered_float::OrderedFloat::<f64>(
            f64::NEG_INFINITY
        )),
        |ctx| {
            let start = ctx.index;
            let (_, next) = sequences!(
                option(text_exact("-")),
                repeat1(digit),
                text_exact("."),
                repeat1(digit),
                option(sequences!(
                    text_exact("e"),
                    option(text_exact("+")),
                    option(text_exact("-")),
                    repeat1(digit),
                )),
                text(""),
            )(ctx)?;
            let end = next.index;
            let f = ctx.source.contents[start..end]
                .parse()
                .map_err(|_| ParseError::ExpectedFloat(Span(ctx.source.clone(), start, end)))?;
            Ok((ordered_float::OrderedFloat::<f64>(f), next))
        },
    )(ctx)
}

fn ident(ctx: &Context) -> Res<Symbol> {
    let mut span = Span(ctx.source.clone(), ctx.index, ctx.index);
    if ctx.index >= ctx.source.contents.len() {
        return Err(ParseError::ExpectedIdent(span));
    }

    let mut next = ctx.clone();
    loop {
        loop {
            next.index += 1;
            if next.source.contents.is_char_boundary(next.index) {
                break;
            }
        }

        let Some(c) = next.source.contents[next.index..].chars().next() else {
            break;
        };
        if !c.is_alphanumeric() && !"-+*/?!=<>&|^/%_".contains(c) {
            break;
        }
    }

    span.2 = next.index;
    next.advance_past_whitespace();

    Ok((Symbol::from(span.string()), next))
}

fn sym_string(ctx: &Context) -> Res<Symbol> {
    map(string, Symbol::from)(ctx)
}

fn string(ctx: &Context) -> Res<String> {
    let mut span = Span(ctx.source.clone(), ctx.index, ctx.index);
    if !ctx.source.contents[ctx.index..].starts_with('"') {
        return Err(ParseError::ExpectedString(span));
    }

    let mut next = ctx.clone();
    let mut in_escape = false;
    loop {
        loop {
            next.index += 1;
            if next.source.contents.is_char_boundary(next.index) {
                break;
            }
        }

        let Some(c) = next.source.contents[next.index..].chars().next() else {
            span.2 = span.1 + 1;
            return Err(ParseError::MissingEndQuote(span));
        };

        if c == '"' && !in_escape {
            break;
        }

        in_escape = c == '\\' && !in_escape;
    }

    span.2 = next.index;
    next.advance_past_whitespace();

    let string = span.string();
    let string = &string[1..string.len() - 1];
    Ok((string.to_string(), next))
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("{0}\nexpected {1}, found {}", .0.string())]
    ExpectedText(Span, String),
    #[error("{0}\nexpected string")]
    ExpectedString(Span),
    #[error("{0}\nmissing end quote for string")]
    MissingEndQuote(Span),
    #[error("{0}\nexpected identifier")]
    ExpectedIdent(Span),
    #[error("{0}\nexpected integer")]
    ExpectedInt(Span),
    #[error("{0}\nexpected unsigned integer")]
    ExpectedUint(Span),
    #[error("{0}\nexpected floating point constant")]
    ExpectedFloat(Span),
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
