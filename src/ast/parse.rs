//! Parse a string into egglog.

use crate::util::INTERNAL_SYMBOL_PREFIX;
use crate::*;
use egglog_ast::generic_ast::*;
use egglog_ast::span::{EgglogSpan, Span, SrcFile};
use ordered_float::OrderedFloat;
use std::borrow::Cow;

#[macro_export]
macro_rules! span {
    () => {{
        use $crate::ast::{RustSpan, Span};
        Span::Rust(std::sync::Arc::new(RustSpan {
            file: file!(),
            line: line!(),
            column: column!(),
        }))
    }};
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

/// A parsed S-expression. Atoms borrow from the input; macros may also construct
/// atoms with owned strings. Spans own their source and can outlive the input.
pub enum Sexp<'a> {
    // Will never contain `Literal::Unit`, as this
    // will be parsed as an empty `Sexp::List`.
    Literal(Literal, Span),
    Atom(Cow<'a, str>, Span),
    List(Vec<Sexp<'a>>, Span),
}

impl<'a> Sexp<'a> {
    pub fn span(&self) -> Span {
        match self {
            Sexp::Literal(_, span) => span.clone(),
            Sexp::Atom(_, span) => span.clone(),
            Sexp::List(_, span) => span.clone(),
        }
    }

    pub fn expect_uint<UInt: TryFrom<u64>>(&self, e: &'static str) -> Result<UInt, ParseError> {
        if let Sexp::Literal(Literal::Int(x), _) = self
            && *x >= 0
            && let Ok(v) = (*x as u64).try_into()
        {
            return Ok(v);
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
            return Ok(symbol.as_ref().to_owned());
        }
        error!(self.span(), "expected {e}")
    }

    pub fn expect_list(&self, e: &'static str) -> Result<&[Sexp<'a>], ParseError> {
        if let Sexp::List(sexps, _) = self {
            return Ok(sexps);
        }
        error!(self.span(), "expected {e}")
    }

    /// Borrow the head and arguments of a call, retaining an owned source span.
    pub fn expect_call(&self, e: &'static str) -> Result<(&str, &[Sexp<'a>], Span), ParseError> {
        if let Sexp::List(sexps, span) = self
            && let [Sexp::Atom(func, _), args @ ..] = sexps.as_slice()
        {
            return Ok((func.as_ref(), args, span.clone()));
        }
        error!(self.span(), "expected {e}")
    }
}

// helper for mapping a function that returns `Result`
fn map_fallible<T>(
    slice: &[Sexp<'_>],
    parser: &mut Parser,
    func: impl Fn(&mut Parser, &Sexp<'_>) -> Result<T, ParseError>,
) -> Result<Vec<T>, ParseError> {
    slice
        .iter()
        .map(|sexp| func(parser, sexp))
        .collect::<Result<_, _>>()
}

/// Parse the `:internal-container-rebuild` annotation value (see
/// [`ContainerRebuildSpec`]). The dual of its `Display`.
fn parse_container_rebuild_spec(sexp: &Sexp<'_>) -> Result<ContainerRebuildSpec, ParseError> {
    let (head, items, span) = sexp.expect_call("container-rebuild spec")?;
    if head != "container-rebuild-spec" {
        return error!(span, "expected (container-rebuild-spec ...)");
    }
    let (prim, proof_prim) = match items {
        [prim] => (prim, None),
        [prim, proof_prim] => (
            prim,
            Some(proof_prim.expect_atom("container rebuild proof primitive name")?),
        ),
        _ => {
            return error!(
                span,
                "container-rebuild-spec needs a primitive name and an optional proof primitive name"
            );
        }
    };
    Ok(ContainerRebuildSpec {
        internal_rebuild_prim: prim.expect_atom("container rebuild primitive name")?,
        internal_rebuild_proof_prim: proof_prim,
    })
}

pub trait Macro<T>: Send + Sync {
    fn name(&self) -> &str;
    fn parse(&self, args: &[Sexp<'_>], span: Span, parser: &mut Parser) -> Result<T, ParseError>;
}

pub struct SimpleMacro<
    T,
    F: Fn(&[Sexp<'_>], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
>(String, F);

impl<T, F> SimpleMacro<T, F>
where
    F: Fn(&[Sexp<'_>], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
{
    pub fn new(head: &str, f: F) -> Self {
        Self(head.to_owned(), f)
    }
}

impl<T, F> Macro<T> for SimpleMacro<T, F>
where
    F: Fn(&[Sexp<'_>], Span, &mut Parser) -> Result<T, ParseError> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.0
    }

    fn parse(&self, args: &[Sexp<'_>], span: Span, parser: &mut Parser) -> Result<T, ParseError> {
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
    pub ensure_no_reserved_symbols: bool,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            commands: Default::default(),
            actions: Default::default(),
            exprs: Default::default(),
            user_defined: Default::default(),
            symbol_gen: SymbolGen::new(INTERNAL_SYMBOL_PREFIX.to_string()),
            ensure_no_reserved_symbols: true,
        }
    }
}

impl Parser {
    fn ensure_symbol_not_reserved(&self, symbol: &str, span: &Span) -> Result<(), ParseError> {
        if self.symbol_gen.is_reserved(symbol) && self.ensure_no_reserved_symbols {
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

    pub fn get_schedule_from_string(
        &mut self,
        filename: Option<String>,
        input: &str,
    ) -> Result<Schedule, ParseError> {
        let sexp = sexp(&mut SexpParser::new(filename, input))?;
        self.parse_schedule(&sexp)
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
            return Err(Error::CommandAlreadyExists(name, span!()));
        }
        self.user_defined.insert(name);
        Ok(())
    }

    pub fn parse_command(&mut self, sexp: &Sexp<'_>) -> Result<Vec<Command>, ParseError> {
        let (head, tail, span) = sexp.expect_call("command")?;

        if let Some(macr0) = self.commands.get(head).cloned() {
            return macr0.parse(tail, span, self);
        }

        // This prevents user-defined commands from being parsed as built-in commands.
        if self.user_defined.contains(head) {
            let args = map_fallible(tail, self, Self::parse_expr)?;
            return Ok(vec![Command::UserDefined(span, head.to_owned(), args)]);
        }

        Ok(match head {
            "sort" => {
                // Parse sort - :internal-uf/:internal-proof-func and container sorts are mutually exclusive
                // (sort <name>)
                // (sort <name> :internal-uf <uf-function>)
                // (sort <name> :internal-proof-func <internal-proof-func-name>)
                // (sort <name> (<container sort> <argument sort>*))
                match tail {
                    [name] => vec![Command::Sort {
                        span,
                        name: name.expect_atom("sort name")?,
                        presort_and_args: None,
                        uf: None,
                        proof_func: None,
                        container_rebuild: None,
                        proof_constructors: None,
                        unionable: true,
                    }],
                    [name, call @ Sexp::List(..), rest @ ..] => {
                        let (func, args, _) = call.expect_call("container sort declaration")?;
                        // Container sorts may carry :internal-proof-func (their
                        // `<CSort>Proof` table) and an :internal-container-rebuild
                        // spec (both emitted by the term/proof encoder).
                        let mut proof_func = None;
                        let mut container_rebuild = None;
                        for (key, val) in self.parse_options(rest)? {
                            match (key, val) {
                                (":internal-proof-func", [pf]) => {
                                    proof_func =
                                        Some(pf.expect_atom("internal-proof-func function name")?);
                                }
                                (":internal-container-rebuild", [spec]) => {
                                    container_rebuild = Some(parse_container_rebuild_spec(spec)?);
                                }
                                _ => {
                                    return error!(
                                        span,
                                        "usage:\n(sort <name> (<container sort> <argument sort>*) [:internal-proof-func <name>] [:internal-container-rebuild <spec>])"
                                    );
                                }
                            }
                        }
                        vec![Command::Sort {
                            span,
                            name: name.expect_atom("sort name")?,
                            presort_and_args: Some((
                                func.to_owned(),
                                map_fallible(args, self, Self::parse_expr)?,
                            )),
                            uf: None,
                            proof_func,
                            container_rebuild,
                            proof_constructors: None,
                            unionable: true,
                        }]
                    }
                    [name, rest @ ..] => {
                        // Parse :internal-uf / :internal-proof-func and the
                        // :internal-proof-names global proof-constructor record.
                        let mut uf = None;
                        let mut proof_func = None;
                        let mut proof_constructors = None;
                        for (key, val) in self.parse_options(rest)? {
                            match (key, val) {
                                (":internal-uf", [uf_ctor]) => {
                                    uf = Some((uf_ctor.expect_atom("uf constructor name")?, None));
                                }
                                (":internal-uf", [uf_ctor, uf_index]) => {
                                    uf = Some((
                                        uf_ctor.expect_atom("uf constructor name")?,
                                        Some(uf_index.expect_atom("uf index function name")?),
                                    ));
                                }
                                (":internal-proof-func", [pf]) => {
                                    proof_func =
                                        Some(pf.expect_atom("internal-proof-func function name")?);
                                }
                                (":internal-proof-names", [congr, trans, sym, normalize]) => {
                                    proof_constructors = Some(ProofConstructorNames {
                                        congr: congr.expect_atom("congr constructor")?,
                                        trans: trans.expect_atom("trans constructor")?,
                                        sym: sym.expect_atom("sym constructor")?,
                                        normalize: normalize
                                            .expect_atom("container-normalize constructor")?,
                                    });
                                }
                                _ => {
                                    return error!(
                                        span,
                                        "usages:\n(sort <name>)\n(sort <name> :internal-uf <uf-constructor> [<uf-index>])\n(sort <name> :internal-proof-func <internal-proof-func-name>)\n(sort <name> :internal-proof-names <congr> <trans> <sym> <normalize>)\n(sort <name> (<container sort> <argument sort>*))"
                                    );
                                }
                            }
                        }
                        vec![Command::Sort {
                            span,
                            name: name.expect_atom("sort name")?,
                            presort_and_args: None,
                            uf,
                            proof_func,
                            container_rebuild: None,
                            proof_constructors,
                            unionable: true,
                        }]
                    }
                    _ => {
                        return error!(
                            span,
                            "usages:\n(sort <name>)\n(sort <name> (<container sort> <argument sort>*))"
                        );
                    }
                }
            }
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
                [name, inputs, output, rest @ ..] => {
                    let mut merge = None;
                    let mut hidden = false;
                    let mut let_binding = false;
                    let mut term_constructor = None;
                    let mut unextractable = false;
                    for (key, val) in self.parse_options(rest)? {
                        match (key, val) {
                            (":no-merge", []) => {
                                if merge.is_some() {
                                    return error!(
                                        span,
                                        "conflicting merge options: :no-merge and :merge cannot both be specified"
                                    );
                                }
                                merge = Some(None);
                            }
                            (":merge", [e]) => {
                                if merge.is_some() {
                                    return error!(
                                        span,
                                        "conflicting merge options: :merge and :no-merge cannot both be specified"
                                    );
                                }
                                merge = Some(Some(self.parse_expr(e)?));
                            }
                            (":internal-hidden", []) => hidden = true,
                            (":internal-let", []) => let_binding = true,
                            (":unextractable", []) => unextractable = true,
                            (":internal-term-constructor", [tc]) => {
                                term_constructor = Some(tc.expect_atom("term constructor name")?)
                            }
                            _ => return error!(span, "could not parse function options"),
                        }
                    }
                    let merge = match merge {
                        Some(m) => m,
                        None => {
                            return error!(
                                span,
                                "functions are required to specify merge behaviour"
                            );
                        }
                    };
                    vec![Command::Function {
                        name: name.expect_atom("function name")?,
                        schema: self.parse_schema(inputs, output)?,
                        merge,
                        hidden,
                        let_binding,
                        term_constructor,
                        unextractable,
                        span,
                    }]
                }
                _ => {
                    let a = "(function <name> (<input sort>*) <output sort> :merge <expr>)";
                    let b = "(function <name> (<input sort>*) <output sort> :no-merge)";
                    return error!(span, "usages:\n{a}\n{b}");
                }
            },
            "constructor" => {
                // Parse constructor with optional annotations
                // (constructor <name> (<input sort>*) <output sort>)
                // (constructor <name> (<input sort>*) <output sort> :cost <cost>)
                // (constructor <name> (<input sort>*) <output sort> :unextractable)
                // (constructor <name> (<input sort>*) <output sort> :internal-term-constructor <constructor name>)
                match tail {
                    [name, inputs, output, rest @ ..] => {
                        let mut cost = None;
                        let mut unextractable = false;
                        let mut hidden = false;
                        let mut let_binding = false;
                        for (key, val) in self.parse_options(rest)? {
                            match (key, val) {
                                (":unextractable", []) => unextractable = true,
                                (":internal-hidden", []) => hidden = true,
                                (":internal-let", []) => let_binding = true,
                                (":cost", [c]) => cost = Some(c.expect_uint("cost")?),
                                _ => return error!(span, "could not parse constructor options"),
                            }
                        }

                        vec![Command::Constructor {
                            span,
                            name: name.expect_atom("constructor name")?,
                            schema: self.parse_schema(inputs, output)?,
                            cost,
                            unextractable,
                            hidden,
                            let_binding,
                            term_constructor: None,
                        }]
                    }
                    _ => {
                        let a = "(constructor <name> (<input sort>*) <output sort>)";
                        let b = "(constructor <name> (<input sort>*) <output sort> :cost <cost>)";
                        let c = "(constructor <name> (<input sort>*) <output sort> :unextractable)";
                        return error!(span, "usages:\n{a}\n{b}\n{c}");
                    }
                }
            }
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
                    // `:naive` and `:unsafe-seminaive` are mutually
                    // exclusive; `eval_mode` is set at most once.
                    let mut eval_mode: Option<RuleEvalMode> = None;
                    let mut no_decomp = false;
                    let mut include_subsumed = false;
                    for option in self.parse_options(rest)? {
                        match option {
                            (":ruleset", [r]) => ruleset = r.expect_atom("ruleset name")?,
                            (":name", [s]) => name = s.expect_string("rule name")?,
                            (":naive", []) | (":unsafe-seminaive", []) => {
                                let mode = if option.0 == ":naive" {
                                    RuleEvalMode::Naive
                                } else {
                                    RuleEvalMode::UnsafeSeminaive
                                };
                                if eval_mode.is_some() {
                                    return error!(
                                        span,
                                        ":naive and :unsafe-seminaive are mutually exclusive"
                                    );
                                }
                                eval_mode = Some(mode);
                            }
                            (":no-decomp", []) => no_decomp = true,
                            (":internal-include-subsumed", []) => include_subsumed = true,
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
                            eval_mode: eval_mode.unwrap_or_default(),
                            no_decomp,
                            include_subsumed,
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
                    let mut name = String::new();
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
                            (":name", [s]) => name = s.expect_string("rule name")?,
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
                            name,
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
                    let mut name = String::new();
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
                            (":name", [s]) => name = s.expect_string("rule name")?,
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
                            name,
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
                map_fallible(tail, self, Self::parse_schedule)?,
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
            "prove" => vec![Command::Prove(
                span,
                map_fallible(tail, self, Self::parse_fact)?,
            )],
            "prove-exists" => match tail {
                [constructor] => vec![Command::ProveExists(
                    span,
                    constructor.expect_atom("constructor name")?,
                )],
                _ => return error!(span, "usage: (prove-exists <constructor>)"),
            },
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
                                mode = match mode_str.as_ref() {
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
                        return error!(
                            span,
                            "(fail ...) does not support commands that expand into multiple commands"
                        );
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

    pub fn parse_schedule(&mut self, sexp: &Sexp<'_>) -> Result<Schedule, ParseError> {
        if let Sexp::Atom(ruleset, span) = sexp {
            return Ok(Schedule::Run(
                span.clone(),
                RunConfig {
                    ruleset: ruleset.as_ref().to_owned(),
                    until: None,
                },
            ));
        }

        let (head, tail, span) = sexp.expect_call("schedule")?;

        Ok(match head {
            "saturate" => Schedule::Saturate(
                span.clone(),
                Box::new(Schedule::Sequence(
                    span,
                    map_fallible(tail, self, Self::parse_schedule)?,
                )),
            ),
            "seq" => Schedule::Sequence(span, map_fallible(tail, self, Self::parse_schedule)?),
            "repeat" => match tail {
                [limit, tail @ ..] => Schedule::Repeat(
                    span.clone(),
                    limit.expect_uint("number of iterations")?,
                    Box::new(Schedule::Sequence(
                        span,
                        map_fallible(tail, self, Self::parse_schedule)?,
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

    pub fn parse_action(&mut self, sexp: &Sexp<'_>) -> Result<Vec<Action>, ParseError> {
        let (head, tail, span) = sexp.expect_call("action")?;

        if let Some(func) = self.actions.get(head).cloned() {
            return func.parse(tail, span, self);
        }

        Ok(match head {
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
                    vec![Action::Set(span, func.to_owned(), args, value)]
                }
                _ => return error!(span, "usage: (set (<table name> <expr>*) <expr>)"),
            },
            "delete" => match tail {
                [call] => {
                    let (func, args, _) = call.expect_call("table lookup")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    vec![Action::Change(span, Change::Delete, func.to_owned(), args)]
                }
                _ => return error!(span, "usage: (delete (<table name> <expr>*))"),
            },
            "subsume" => match tail {
                [call] => {
                    let (func, args, _) = call.expect_call("table lookup")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    vec![Action::Change(span, Change::Subsume, func.to_owned(), args)]
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

    pub fn parse_fact(&mut self, sexp: &Sexp<'_>) -> Result<Fact, ParseError> {
        let (head, tail, span) = sexp.expect_call("fact")?;

        Ok(match head {
            "=" => match tail {
                [e1, e2] => Fact::Eq(span, self.parse_expr(e1)?, self.parse_expr(e2)?),
                _ => return error!(span, "usage: (= <expr> <expr>)"),
            },
            _ => Fact::Fact(self.parse_expr(sexp)?),
        })
    }

    pub fn parse_expr(&mut self, sexp: &Sexp<'_>) -> Result<Expr, ParseError> {
        Ok(match sexp {
            Sexp::Literal(literal, span) => Expr::Lit(span.clone(), literal.clone()),
            Sexp::Atom(symbol, span) => Expr::Var(
                span.clone(),
                if *symbol == "_" {
                    self.symbol_gen.fresh(symbol.as_ref())
                } else {
                    self.ensure_symbol_not_reserved(symbol, span)?;
                    symbol.as_ref().to_owned()
                },
            ),
            Sexp::List(list, span) => match list.as_slice() {
                [] => Expr::Lit(span.clone(), Literal::Unit),
                _ => {
                    let (head, tail, span) = sexp.expect_call("call expression")?;

                    if let Some(func) = self.exprs.get(head).cloned() {
                        return func.parse(tail, span, self);
                    }

                    Expr::Call(
                        span.clone(),
                        head.to_owned(),
                        map_fallible(tail, self, Self::parse_expr)?,
                    )
                }
            },
        })
    }

    pub fn rec_datatype(
        &mut self,
        sexp: &Sexp<'_>,
    ) -> Result<(Span, String, Subdatatypes), ParseError> {
        let (head, tail, span) = sexp.expect_call("datatype")?;

        Ok(match head {
            "sort" => match tail {
                [name, call] => {
                    let name = name.expect_atom("sort name")?;
                    let (func, args, _) = call.expect_call("container sort declaration")?;
                    let args = map_fallible(args, self, Self::parse_expr)?;
                    (span, name, Subdatatypes::NewSort(func.to_owned(), args))
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
                (span, head.to_owned(), Subdatatypes::Variants(variants))
            }
        })
    }

    pub fn variant(&mut self, sexp: &Sexp<'_>) -> Result<Variant, ParseError> {
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
            name: name.to_owned(),
            types: map_fallible(types, self, |_, sexp| {
                sexp.expect_atom("variant argument type")
            })?,
            cost,
            unextractable,
        })
    }

    // helper for parsing a list of options
    pub fn parse_options<'s, 'a>(
        &self,
        sexps: &'s [Sexp<'a>],
    ) -> Result<Vec<(&'s str, &'s [Sexp<'a>])>, ParseError> {
        fn option_name<'s>(sexp: &'s Sexp<'_>) -> Option<&'s str> {
            if let Sexp::Atom(s, _) = sexp
                && let Some(':') = s.chars().next()
            {
                return Some(s);
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

    pub fn parse_schema(&self, input: &Sexp<'_>, output: &Sexp<'_>) -> Result<Schema, ParseError> {
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
pub(crate) struct SexpParser<'a> {
    input: &'a str,
    source: Arc<SrcFile>,
    index: usize,
}

impl<'a> SexpParser<'a> {
    pub(crate) fn new(name: Option<String>, input: &'a str) -> Self {
        SexpParser {
            input,
            source: Arc::new(SrcFile {
                name,
                contents: input.to_string(),
            }),
            index: 0,
        }
    }

    fn current_char(&self) -> Option<char> {
        self.input[self.index..].chars().next()
    }

    fn advance_char(&mut self) {
        assert!(self.index < self.input.len());
        loop {
            self.index += 1;
            if self.input.is_char_boundary(self.index) {
                break;
            }
        }
    }

    fn advance_past_whitespace(&mut self) {
        let contents = self.input;
        let bytes = contents.as_bytes();
        while let Some(&byte) = bytes.get(self.index) {
            match byte {
                // Include vertical tab, which char::is_whitespace recognizes.
                b'\t'..=b'\r' | b' ' => self.index += 1,
                b';' => {
                    self.index += bytes[self.index..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .unwrap_or(bytes.len() - self.index);
                }
                b if b.is_ascii() => break,
                _ => {
                    let c = contents[self.index..].chars().next().unwrap();
                    if !c.is_whitespace() {
                        break;
                    }
                    self.index += c.len_utf8();
                }
            }
        }
    }

    fn is_at_end(&self) -> bool {
        self.index == self.input.len()
    }

    fn next(&mut self) -> Result<(Token<'a>, EgglogSpan), ParseError> {
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
                let contents = self.input;
                while let Some(&byte) = contents.as_bytes().get(self.index) {
                    match byte {
                        b'\t'..=b'\r' | b' ' | b';' | b'(' | b')' => break,
                        b if b.is_ascii() => self.index += 1,
                        _ => {
                            let c = contents[self.index..].chars().next().unwrap();
                            if c.is_whitespace() {
                                break;
                            }
                            self.index += c.len_utf8();
                        }
                    }
                }
                Token::Other(&self.input[span.i..self.index])
            }
        };

        span.j = self.index;

        Ok((token, span))
    }
}

fn s(span: EgglogSpan) -> Span {
    Span::Egglog(Arc::new(span))
}

enum Token<'a> {
    Open,
    Close,
    String(String),
    Other(&'a str),
}

fn sexp<'a>(ctx: &mut SexpParser<'a>) -> Result<Sexp<'a>, ParseError> {
    let mut stack: Vec<(EgglogSpan, Vec<Sexp<'a>>)> = vec![];

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
            Token::Other(s) => {
                let span = self::s(span);
                let numeric = matches!(s.as_bytes()[0], b'0'..=b'9' | b'+' | b'-' | b'.');

                if s == "true" {
                    Sexp::Literal(Literal::Bool(true), span)
                } else if s == "false" {
                    Sexp::Literal(Literal::Bool(false), span)
                } else if numeric && let Ok(int) = s.parse::<i64>() {
                    Sexp::Literal(Literal::Int(int), span)
                } else if s == "NaN" {
                    Sexp::Literal(Literal::Float(OrderedFloat(f64::NAN)), span)
                } else if s == "inf" {
                    Sexp::Literal(Literal::Float(OrderedFloat(f64::INFINITY)), span)
                } else if s == "-inf" {
                    Sexp::Literal(Literal::Float(OrderedFloat(f64::NEG_INFINITY)), span)
                } else if numeric && let Ok(float) = s.parse::<f64>() {
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

pub(crate) fn all_sexps(mut ctx: SexpParser<'_>) -> Result<Vec<Sexp<'_>>, ParseError> {
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
    fn numeric_tokens_preserve_literal_and_atom_classification() {
        use Literal::{Bool, Float, Int};

        for (source, expected) in [
            ("true", Some(Bool(true))),
            ("false", Some(Bool(false))),
            ("0", Some(Int(0))),
            ("+12", Some(Int(12))),
            ("-0", Some(Int(0))),
            ("9223372036854775807", Some(Int(i64::MAX))),
            ("-9223372036854775808", Some(Int(i64::MIN))),
            (
                "9223372036854775808",
                Some(Float(OrderedFloat(9_223_372_036_854_775_808.0))),
            ),
            (
                "-9223372036854775809",
                Some(Float(OrderedFloat(-9_223_372_036_854_775_808.0))),
            ),
            (".5", Some(Float(OrderedFloat(0.5)))),
            ("-.5", Some(Float(OrderedFloat(-0.5)))),
            ("1.", Some(Float(OrderedFloat(1.0)))),
            ("+1.25e-2", Some(Float(OrderedFloat(0.0125)))),
            ("-2E3", Some(Float(OrderedFloat(-2000.0)))),
            ("1e-400", Some(Float(OrderedFloat(0.0)))),
            ("NaN", Some(Float(OrderedFloat(f64::NAN)))),
            ("inf", Some(Float(OrderedFloat(f64::INFINITY)))),
            ("-inf", Some(Float(OrderedFloat(f64::NEG_INFINITY)))),
            ("Inf", None),
            ("Infinity", None),
            ("+inf", None),
            ("-NaN", None),
            ("nan", None),
            ("-Infinity", None),
            ("TRUE", None),
            ("N1", None),
            ("λ", None),
            ("１２", None),
            ("−1", None),
            ("1e400", None),
            ("-1e400", None),
            ("+", None),
            ("-", None),
            (".", None),
            ("1e", None),
            ("0x10", None),
            ("2foo", None),
            ("+word", None),
            ("..5", None),
        ] {
            let parsed = sexp(&mut SexpParser::new(None, source)).unwrap();
            assert_eq!(parsed.span().string(), source);
            match (parsed, expected) {
                (Sexp::Literal(actual, _), Some(expected)) => {
                    assert_eq!(actual, expected, "{source}");
                }
                (Sexp::Atom(atom, _), None) => assert_eq!(atom, source),
                _ => panic!("unexpected literal/atom classification for {source}"),
            }
        }
    }

    #[test]
    fn lexer_preserves_unicode_whitespace_and_byte_spans() {
        for whitespace in [
            '\t', '\n', '\u{000b}', '\u{000c}', '\r', ' ', '\u{0085}', '\u{00a0}', '\u{1680}',
            '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
            '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
            '\u{205f}', '\u{3000}',
        ] {
            let list_text = format!("(α{whitespace}β)");
            let source = format!("{whitespace}{list_text}{whitespace}γ{whitespace}");
            let forms = all_sexps(SexpParser::new(None, &source)).unwrap();
            let [list, last] = forms.as_slice() else {
                panic!("expected two forms with {whitespace:?}");
            };
            assert_eq!(list.span().string(), list_text);
            let items = list.expect_list("list").unwrap();
            assert_eq!(items.len(), 2);
            for (item, text) in items.iter().chain([last]).zip(["α", "β", "γ"]) {
                assert_eq!(item.expect_atom("atom").unwrap(), text);
                let Span::Egglog(span) = item.span() else {
                    panic!("expected source span");
                };
                let start = source.find(text).unwrap();
                assert_eq!((span.i, span.j), (start, start + text.len()));
                assert_eq!(&span.file.contents[span.i..span.j], text);
            }
        }
    }

    #[test]
    fn lexer_preserves_atom_comment_and_string_boundaries() {
        let source = concat!(
            "alpha\"β(γ); comment ; 🦀\r\u{2028}ignored\n",
            "δ\u{200b}\u{feff}😀 ",
            r#""λ\n\t\"\\;()　""#,
            ";尾",
        );
        let forms = all_sexps(SexpParser::new(None, source)).unwrap();
        let [first, list, last, string] = forms.as_slice() else {
            panic!("expected four forms");
        };
        assert_eq!(first.expect_atom("atom").unwrap(), "alpha\"β");
        assert_eq!(first.span().string(), "alpha\"β");
        let [item] = list.expect_list("list").unwrap() else {
            panic!("expected one list item");
        };
        assert_eq!(item.expect_atom("atom").unwrap(), "γ");
        assert_eq!(list.span().string(), "(γ)");
        assert_eq!(last.expect_atom("atom").unwrap(), "δ\u{200b}\u{feff}😀");
        assert_eq!(last.span().string(), "δ\u{200b}\u{feff}😀");
        assert_eq!(string.expect_string("string").unwrap(), "λ\n\t\"\\;()　");
        assert_eq!(string.span().string(), r#""λ\n\t\"\\;()　""#);
    }

    #[test]
    fn sexp_atoms_borrow_input_and_preserve_utf8_and_escapes() {
        let source = r#"(呼 λ 1e400 "λ\n\t\"\\")"#.to_owned();
        let form = sexp(&mut SexpParser::new(Some("utf8.egg".into()), &source)).unwrap();
        let items = form.expect_list("call").unwrap();
        for (item, text) in items.iter().zip(["呼", "λ", "1e400"]) {
            let Sexp::Atom(Cow::Borrowed(atom), Span::Egglog(span)) = item else {
                panic!("expected borrowed atom");
            };
            assert_eq!(*atom, text);
            assert_eq!(atom.as_ptr(), source[span.i..span.j].as_ptr());
            assert_eq!(&span.file.contents[span.i..span.j], text);
        }
        let expr = Parser::default().parse_expr(&form).unwrap();
        drop(form);
        drop(source);
        let Expr::Call(span, head, args) = expr else {
            panic!("expected call");
        };
        assert_eq!(head, "呼");
        assert_eq!(span.string(), r#"(呼 λ 1e400 "λ\n\t\"\\")"#);
        let [
            Expr::Var(_, first),
            Expr::Var(_, second),
            Expr::Lit(_, Literal::String(value)),
        ] = args.as_slice()
        else {
            panic!("expected atom arguments and a string literal");
        };
        assert_eq!(first, "λ");
        assert_eq!(second, "1e400");
        assert_eq!(value, "λ\n\t\"\\");
    }

    #[test]
    fn macros_can_construct_owned_atoms_and_options() {
        let mut parser = Parser::default();
        parser.add_expr_macro(Arc::new(SimpleMacro::new(
            "generated",
            |args, span, parser| {
                let options = parser.parse_options(args)?;
                let [(":name", [name]), (":flag", [])] = options.as_slice() else {
                    return error!(span, "expected :name <name> :flag");
                };
                let name = format!("generated-{}", name.expect_atom("name")?);
                let generated: Sexp<'static> =
                    Sexp::List(vec![Sexp::Atom(name.into(), span.clone())], span);
                parser.parse_expr(&generated)
            },
        )));
        let expr = {
            let source = "(generated :name λ :flag)".to_owned();
            parser.get_expr_from_string(None, &source).unwrap()
        };
        assert_eq!(expr.to_string(), "(generated-λ)");
        assert_eq!(expr.span().string(), "(generated :name λ :flag)");
    }

    #[test]
    fn test_parser_display_roundtrip() {
        let s = r#"(f (g a 3) 4.0 (H "hello"))"#;
        let e = Parser::default().get_expr_from_string(None, s).unwrap();
        assert_eq!(format!("{e}"), s);
    }

    #[test]
    fn test_file_command_display_roundtrip() {
        let file = "quoted\"-backslash\\-combining\u{300}";
        let commands = [
            Command::PrintOverallStatistics(Span::Panic, Some(file.into())),
            Command::PrintFunction(
                Span::Panic,
                "f".into(),
                None,
                Some(file.into()),
                PrintFunctionMode::Default,
            ),
            Command::Input {
                span: Span::Panic,
                name: "f".into(),
                file: file.into(),
            },
            Command::Output {
                span: Span::Panic,
                file: file.into(),
                exprs: vec![],
            },
            Command::Include(Span::Panic, file.into()),
        ];

        for command in commands {
            let displayed = command.to_string();
            let reparsed = Parser::default()
                .get_program_from_string(None, &displayed)
                .unwrap();
            assert_eq!(reparsed.len(), 1);
            assert_eq!(reparsed[0].to_string(), displayed);
        }
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
        assert_eq!(format!("{e}"), t);
    }
}
