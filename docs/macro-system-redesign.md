# Egglog Macro System Redesign

*Design proposal. No code changes yet; see the open questions at the end.*

Replace egglog's three unrelated extension hooks (parse-time `Macro<T>`, post-parse `CommandMacro`, run-time `UserDefinedCommand`) with one expander modeled on Rhombus and Racket: syntax objects that can carry parsed fragments, one transformer protocol over every syntactic space, an expander handle with live type information and lifting, top-level expansion interleaved with processing, and typed macros that wait for the enclosing rule to be typechecked. Every macro in egglog-experimental gets shorter, and the failure modes from the effect-safe extraction port (egglog-experimental PR #77) stop being possible.

## Summary

Six design points, each tied to a problem the current system cannot avoid:

1. **Syntax objects that carry parsed fragments.** `Syntax` replaces `Sexp`; it is `Clone` and has an opaque `Parsed` node. Macros hand a finished `Rule` to another stage instead of printing it and re-parsing.
2. **Spaces with one transformer protocol.** Command, action, fact, expression, and schedule positions are all extensible through one registry keyed by (space, name). The builtin forms are registered through the same protocol, so a macro can delegate to a builtin or stop expansion at one.
3. **An expander handle.** Macros get recursive and partial expansion, live type information, lifting of declarations, deterministic hygienic names, and keyword-option parsing from one object, instead of `&mut Parser` or a frozen `&TypeInfo`.
4. **Interleaved top-level expansion.** Each form is expanded, desugared, typechecked, and run before the next form is expanded, so a declaration one macro emits is visible to the macros that follow.
5. **Typed macros.** A call whose expansion needs sorts is left in place, constrained by a signature, and expanded after the enclosing rule has been typechecked, with the resolved sorts in hand.
6. **Option extensions on builtin forms.** Adding `:regions` to `constructor` is a registration, not a reimplementation of `constructor`.

The macros that exist today (`for`, `with-ruleset`, `set-cost`, `with-dynamic-cost`, `unstable-fresh!`, `set-effectful`, `:regions`) all get shorter under this design; the sketches are in the section on existing macros below. Macros stay Rust code, as decided in issue #356; the proposal changes what that Rust code can see and do.

## Where we are

Three extension hooks exist on `main`, and nothing connects them.

| Mechanism | Where | What it sees | What it returns |
| --- | --- | --- | --- |
| `Macro<T>` | `src/ast/parse.rs:140` | `&[Sexp]`, span, `&mut Parser` | `Vec<Command>`, `Vec<Action>`, or `Expr` |
| `CommandMacro` | `src/command_macro.rs` | `Command`, `&mut SymbolGen`, a `&TypeInfo` snapshot | `Vec<Command>` |
| `UserDefinedCommand` | `src/lib.rs:341` | `&mut EGraph`, `&[Expr]` | outputs, at run time |
| built-in sugar | `src/ast/desugar.rs` | `Command` | `Vec<NCommand>`, not extensible |

Properties that drive the design:

- The parser keeps three `HashMap<String, Arc<dyn Macro<_>>>` tables, for commands, actions, and expressions. Facts and schedules have no hook.
- A macro registered under a builtin head replaces the builtin. The builtin parsers live in a `match head.as_str()` inside `parse_command`, unreachable from a macro except by constructing `Parser::default()`, which has no macros, a different `SymbolGen`, and no knowledge of user-defined commands.
- `Sexp` is not `Clone`.
- `process_program_internal` (`src/lib.rs:2148`) applies every `CommandMacro` to every top-level command, in registration order, against one `TypeInfo` snapshot. Only after the whole output list exists are the outputs desugared, typechecked, and run one at a time. `include` is special-cased in that loop.
- Keyword options are split by `parse_options` into `(key, values)` pairs and each form matches keys by hand, erroring on anything unknown. No form's option set can be extended.
- `TypeInfo::typecheck_facts` is public; `typecheck_rule` is private (PR #1035 proposes making it public). `EGraph::typecheck_expr_with_bindings_and_output` exists but is unreachable from a macro.
- Fresh names come from `SymbolGen`: a reserved `@` prefix, a hint, and a counter. The parser rejects user symbols with that prefix unless `ensure_no_reserved_symbols` is turned off.
- In proof mode macros must see the original `TypeInfo`, not the term-encoded one; `process_program_internal` picks it by hand.

## What went wrong in practice

The effect-safe extraction port (egglog-experimental PR #77, nine review rounds) is the best stress test the macro system has had. Its needs, in order of pain, with the root cause in the current design:

| Need | What happened | Root cause |
| --- | --- | --- |
| Sort of an expression in a rule head, for `(set-effectful e)` | Seven rounds reimplementing action typing: synthetic facts, overload search, consumer-driven narrowing, `let` sharing. Each version had a hole: write primitives, `unstable-fn` literals, `(vec-empty)`, `:naive` contexts. Fixed only by exposing `typecheck_rule`. | Macros run before typechecking, or against a frozen snapshot with query-only typing. The rule typechecker solves body and head together, and nothing lets a macro wait for that. |
| A declaration emitted by one macro, used by another (`unstable-fresh!` emits a constructor; `set-effectful` then sees the rule) | "Unbound function @GeneratedFreshTable". Worked around by lowering to a `UserDefinedCommand` that runs later. | Expansion is a batch pre-pass; outputs are not processed before the next macro runs. |
| Hand a `Rule` to a later stage | `UserDefinedCommand` takes `Vec<Expr>`, so the rule was printed with `Display` and re-parsed, with `ensure_no_reserved_symbols = false` to accept the `@` names earlier macros had introduced. A process-local id table instead of text broke determinism. | No syntax node can carry a parsed fragment. |
| Add `:regions` to `constructor`, `datatype`, `datatype*` | Shadow all three heads, strip the option out of raw `Sexp`s by hand (with a hand-written `clone_sexp`), re-parse with `Parser::default()`. | Builtin forms are not macros; no option-extension protocol; no delegation to the previous binding. |
| Wrapper composition: `with-dynamic-cost` around a declaration carrying `:regions` | The wrapper rejected the extra `effsafe-regions` command the inner expansion returned. | Everything a macro produces comes back as one flat `Vec<Command>`; every wrapper must anticipate every other macro's side outputs. |
| Hygiene | A fixed synthetic name collided with a user variable; caught in review. | Names are plain strings. Hygiene is a convention (the `@` prefix), not a mechanism. |
| Position mismatch | An expression macro cannot appear where an action is expected, and vice versa. | Three separate tables, no shared protocol. |

The smaller macros show the same pattern at lower cost: `with-ruleset` reimplements option handling for three forms, `for` and `set-cost` call the parser by hand on sub-forms, and `unstable-fresh!` scans every rule with `visit_exprs` because `CommandMacro` has no dispatch by name.

## What to take from Rhombus

Rhombus ([Flatt et al., OOPSLA 2023](https://doi.org/10.1145/3622818)) is a macro-extensible language with conventional notation built on Racket. The notation half of the paper (shrubbery grouping, enforestation, operator precedence) does not apply: s-expressions already give egglog unambiguous grouping. The expansion half maps onto egglog's problems closely.

**Spaces (paper section 5).** A space is a kind of program context: expression, binding, annotation, or a user-defined one such as regular-expression operators. Each space has its own sublanguage of macros, and one name can mean different things in different spaces. egglog already has implicit spaces (command, action, fact, expression, schedule, sort expression); three are extensible today, through three unrelated tables. Making all of them extensible through one protocol is the first step.

**The base language is written with the same mechanism.** In Rhombus, `fun`, `def`, `class`, and `[]` are macros over primitive forms, which is what makes them extensible and lets a user macro delegate to the original binding (paper Fig. 10 replaces `⊢` in one space while forwarding to `orig.(⊢)`). egglog's builtin parsers should be ordinary transformers in the same table, so a macro can wrap one instead of replacing it.

**A macro consumes a tail and returns the remainder (paper section 4.1).** The general Rhombus transformer receives all remaining terms of its group, consumes as many as it wants, and returns the expansion plus the leftover terms. egglog does not need infix parsing, but it needs exactly this protocol for keyword options: in `(constructor Name (A B) Out :cost 3 :regions (0 1))`, an option handler is a transformer over the tail after its keyword.

**Parsed nodes mixed into syntax (paper section 3.2, Fig. 8).** Rhombus syntax objects can contain `(parsed expr)` nodes: opaque, already-expanded fragments that pattern matching treats as atoms. This is how a macro hands a finished `Rule` to another stage without serializing it.

**Partial expansion and definition contexts (Flatt et al. 2012, inherited from Racket).** A module body is expanded one form at a time; a form that expands into a sequence is spliced and its first element is expanded next, and definitions become visible to later forms as they are discovered. A macro can also ask to expand a sub-form only until a form from a stop list appears. The first fixes expansion-order problems; the second is what a wrapper like `with-ruleset` needs.

**Static information (paper section 6).** Binding positions produce names with key-value static information; expressions can report information upward, and macros look it up. Rhombus keeps this modest and avoids forcing expansion order across forms. The paper also names the alternative for cases that need more: an expander that pauses a macro until enough information is available (Barrett et al. 2020). For egglog, rule variables are the binding positions, sorts are the static information, and "pause until the enclosing rule is typechecked" is the right level of ambition.

**Hygiene via scopes (Flatt 2016).** Macro-introduced identifiers carry a scope that keeps them from capturing or being captured by user identifiers. egglog's binding structure is flat (rule variables, `let`s, global declarations), so a lightweight version is enough.

**Lifting.** Racket macros can lift a definition to the enclosing module or top level. This is how a macro inside a rule declares the table it needs without returning a `Vec<Command>` that wrappers then have to understand.

## Proposed design, part 1: syntax, spaces, expander, ordering

### 1. Syntax objects

`Sexp` becomes `Syntax`: `Clone`, with identifiers that can carry a macro scope, and with an opaque `Parsed` node.

```rust
#[derive(Clone)]
pub enum Syntax {
    Literal(Literal, Span),
    Ident(Ident, Span),          // Ident = { name: String, scope: Option<ScopeId> }
    List(Vec<Syntax>, Span),
    Parsed(Parsed, Span),        // already expanded; opaque to matching
}

pub enum Parsed {
    Command(Command), Action(Action), Fact(Fact), Expr(Expr), Schedule(Schedule),
}
```

Macros receive `Syntax` and return `Syntax`. A `Parsed` node is how a macro says "this part is finished", and how a wrapper hands an inner result through unchanged. `UserDefinedCommand` keeps its `&[Expr]` signature; `Command::UserDefined` may additionally carry `Syntax` arguments for commands that want structured input.

### 2. Spaces and one transformer protocol

```rust
pub trait Space: 'static {
    type Ast: Clone;
    const NAME: &'static str;
}
pub struct Commands;  // Ast = Vec<Command>   (a sequence, spliced at top level)
pub struct Actions;   // Ast = Vec<Action>    (a sequence, spliced into the head)
pub struct Facts;     // Ast = Vec<Fact>
pub struct Exprs;     // Ast = Expr
pub struct Schedules; // Ast = Schedule

pub enum Expansion<S: Space> {
    Syntax(Syntax),   // expand again in the same space
    Ast(S::Ast),      // a core form
}

pub trait Transformer<S: Space>: Send + Sync {
    fn expand(&self, form: &Syntax, cx: &mut Expander<'_>) -> Result<Expansion<S>, Error>;
}
```

One registry holds bindings keyed by (space, name). `bind` returns the previous binding, so a shadowing transformer can keep an `Arc` to it and delegate, as in Rhombus Fig. 10:

```rust
let builtin = registry.bind::<Commands>("constructor", Arc::new(MyConstructor { .. }));
```

The builtin command, action, fact, schedule, and expression parsers are registered through this same call when an `EGraph` is constructed. The `match` in `parse_command` becomes a table of core transformers whose `expand` returns `Expansion::Ast`. `Command::UserDefined` is just the core form that `add_command` binds a name to, which collapses today's three name tables (`parser.commands`, `parser.user_defined`, `egraph.commands`) into one.

Because a macro's output is re-expanded, outputs may use other macros without the author calling the parser by hand, and a macro that produces another macro's input composes for free. A fuel limit guards against non-terminating expansion.

### 3. The expander handle

`Expander` is what a transformer gets instead of `&mut Parser`. The `EGraph` creates one per top-level form, borrowing the live `TypeInfo` (the original one in proof mode).

```rust
impl Expander<'_> {
    // Expansion
    fn expand<S: Space>(&mut self, stx: &Syntax) -> Result<S::Ast, Error>;
    fn expand_until<S: Space>(&mut self, stx: &Syntax, stop: &[&str]) -> Result<Syntax, Error>;
    fn binding<S: Space>(&self, name: &str) -> Option<Arc<dyn Transformer<S>>>;

    // Static information (live, not a snapshot)
    fn type_info(&self) -> &TypeInfo;
    fn settings(&self) -> &ExpansionSettings;            // seminaive, proofs enabled, ...
    fn typecheck_facts(&mut self, facts: &[Fact]) -> Result<Vec<ResolvedFact>, Error>;
    fn sort_of(&mut self, expr: &Expr, env: &Bindings, ctx: Context) -> Result<ArcSort, Error>;

    // Lifting: processed before the current top-level form
    fn lift(&mut self, command: Syntax);
    fn lifted(&self, name: &str) -> bool;

    // Hygiene
    fn fresh(&mut self, hint: &str) -> Ident;             // scoped, deterministic
    fn public(&self, name: impl Into<String>) -> Ident;   // deliberately unhygienic

    // Keyword options, with extension lookup (part 2, point 6)
    fn options(&mut self, form: &str, rest: &[Syntax]) -> Result<Options, Error>;
    fn error(&self, span: Span, msg: impl Display) -> Error;
}
```

`expand_until` is Racket's `local-expand` with a stop list: it expands `stx` in space `S` until the head is one of `stop` (or a core form) and returns that syntax, unexpanded further. A wrapper uses it to see the shape it cares about without committing to the rest.

### 4. Top-level expansion is interleaved with processing

```text
queue <- program forms
while let Some(form) = queue.pop_front():
    match expander.expand_until::<Commands>(form, CORE_HEADS):
        Sequence(forms) => queue.push_front(forms)      // splice; the first is next
        Core(commands)  =>
            for lifted in expander.take_lifts(): process(lifted)
            for c in commands:
                desugar(c); typecheck(c)                // TypeInfo updated here
                if running { run(c) }
```

Each core command is desugared, typechecked, and run before the next queued form is expanded, so a declaration emitted by one form is in `TypeInfo` when the next form's macros run. `include` becomes a core form that returns the file's forms as a sequence, and its special case in `process_program_internal` disappears. `CommandMacro`'s "see every command" behavior survives as a deprecated post-expansion hook on core commands, now with a live `TypeInfo`.

## Proposed design, part 2: typed macros, options, hygiene

### 5. Typed macros and lifting

Some macros need sorts that only the whole-rule constraint solve can provide. The expander handles this with one pause point.

```rust
pub trait TypedMacro: Send + Sync {
    /// How a call typechecks before expansion, in the vocabulary primitives use.
    fn signature(&self, args: &[Syntax], cx: &Expander<'_>) -> Result<TypeConstraint, Error>;

    /// Runs after the enclosing rule (or top-level action) is typechecked.
    fn expand(
        &self,
        call: &ResolvedExpr,
        scope: &ResolvedScope,       // the ResolvedRule, or the top-level action bindings
        cx: &mut Expander<'_>,
    ) -> Result<Syntax, Error>;      // replaces the call; may lift declarations
}
```

Expanding a rule then takes four steps:

1. Syntactic expansion of body and head through the spaces. A call whose head is bound as a `TypedMacro` is left in place as a pending `Expr::Call`.
2. Typecheck the rule, with each pending call constrained by its `signature`. This yields a `ResolvedRule` with sorts for every variable and subexpression.
3. Run `TypedMacro::expand` for each pending call, in order. Lifted declarations queue ahead of the rule.
4. Typecheck the now macro-free rule and continue as normal.

This is the "pause until static information is available" strategy the Rhombus paper points to, specialized to the single pause egglog needs. Write primitives, overloaded constructors, `let` sharing, and the rule's evaluation mode are handled by the real typechecker, which is why `set-effectful` drops from about two hundred lines to about fifteen (next section). `TypeInfo::typecheck_rule` becomes an implementation detail of the expander rather than a public API (PR #1035).

### 6. Extending builtin forms

Builtin forms parse their keyword options through `cx.options(form, rest)`. Unknown keys are looked up in an option-extension table before erroring:

```rust
pub trait OptionExtension: Send + Sync {
    /// `form` is the builtin's parsed positional part (name, schema, ...).
    /// Returns extra forms to splice after the declaration.
    fn apply(&self, form: &FormParts, value: &[Syntax], cx: &mut Expander<'_>)
        -> Result<Vec<Syntax>, Error>;
}

registry.add_option("constructor", ":regions", Arc::new(Regions));
registry.add_option("variant",     ":regions", Arc::new(Regions));   // datatype / datatype* variants
```

The common case (add a keyword to an existing form) needs no shadowing. The general case (change what a form means) is shadow-and-delegate through the previous binding that `bind` returns. Neither needs `Parser::default()`.

### 7. Hygiene and deterministic names

Every transformer invocation gets a fresh `ScopeId`. `cx.fresh(hint)` returns an identifier carrying that scope; identifiers created by a quasi-quote template inside the macro carry it too; identifiers taken from the macro's input keep whatever they had. When a core form is produced, scoped identifiers are renamed to `@hint_k` with a per-`EGraph` counter, the same visible scheme `SymbolGen` uses today. Consequences:

- A macro-introduced rule variable or `let` cannot capture a user variable of the same name, and a user variable cannot capture it.
- A macro-introduced table or ruleset name cannot collide with a user's.
- Resolved programs print deterministically, which eggcc depends on.
- A name users must refer to from their own code, such as `cost_table_Num`, is created with `cx.public(..)`, the explicit opt-out.

Because egglog's binding structure is flat, this gives full hygiene without scope sets. If nested scopes ever appear, `ScopeId` can grow into a set.

## The existing macros under the new design

Every macro gets shorter. Sketches use the quasi-quote syntax of PR #947 (`#x` splices a value, `#..xs` splices a sequence) for templates; it is the natural companion to this design and would produce `Syntax` directly.

**`for`** (command space). Returns syntax, so macros inside the query or actions are expanded by the expander, and the ruleset name is hygienic:

```rust
registry.bind::<Commands>("for", transformer(|form, cx| {
    let [query, actions] = form.args()? else { return Err(cx.error(form.span(), USAGE)) };
    let rs = cx.fresh("for_ruleset");
    Ok(Expansion::Syntax(syntax! {
        (ruleset #rs)
        (rule #query #actions :ruleset #rs)
        (run-schedule (run #rs))
    }))
}));
```

**`with-ruleset`**. Partial expansion with a stop list; no need to understand rules, rewrites, or their options:

```rust
for inner in rest {
    let stx = cx.expand_until::<Commands>(inner, &["rule", "rewrite", "birewrite"])?;
    if stx.has_option(":ruleset") { return Err(cx.error(stx.span(), "already has a ruleset")) }
    out.push(stx.with_option(":ruleset", ruleset.clone()));
}
Ok(Expansion::Syntax(Syntax::sequence(out)))
```

**`set-cost`** (action space). Unchanged in substance; its `let` temporaries come from `cx.fresh` and are hygienic.

**`with-dynamic-cost`**. Expands each inner declaration until `datatype`, `datatype*`, `constructor`, or `function`, then emits cost tables. An inner `:regions` lifts its `effsafe-regions` command instead of returning it, so the wrapper never sees anything but declarations. The "every wrapper anticipates every macro" problem is gone.

**`unstable-fresh!`** (typed macro). Signature: output sort is the named sort. Expansion lifts the generated constructor and replaces the call:

```rust
fn expand(&self, call, rule: &ResolvedScope, cx) -> Result<Syntax, Error> {
    let vars = rule.query_vars();                       // (name, sort) pairs, resolved
    let table = cx.fresh("GeneratedFreshTable");
    cx.lift(syntax! { (constructor #table (#..vars.sorts() i64) #(self.sort) :cost #(self.cost)) });
    Ok(syntax! { (#table #..vars.names() #(cx.next_index())) })
}
```

No `visit_exprs` scan over every rule, no `typecheck_facts` by hand, and the lifted constructor is declared before anything that mentions it.

**`set-effectful`** (typed macro). Signature: one argument of any eq-sort, output unit. Expansion:

```rust
fn expand(&self, call, _scope, cx) -> Result<Syntax, Error> {
    let sort = call.args[0].output_sort();              // resolved by the real typechecker
    let rel = cx.public(format!("effsafe_effectful_{}", sort.name()));
    if cx.type_info().get_func_type(&rel).is_none() && !cx.lifted(&rel) {
        cx.lift(syntax! { (relation #rel (#(sort.name()))) });
    }
    Ok(syntax! { (#rel #(call.args[0].clone())) })
}
```

Every case from the review rounds (a write primitive in a `let`, `(vec-empty)` narrowed by its consumer, `:naive` and `:unsafe-seminaive` rules, top-level actions) is handled because the macro never types anything itself.

**`:regions`** (option extension on `constructor` and on variants):

```rust
fn apply(&self, form: &FormParts, value: &[Syntax], cx) -> Result<Vec<Syntax>, Error> {
    let positions: Vec<u64> = value.iter().map(|v| v.expect_uint("region position")).collect::<Result<_, _>>()?;
    Ok(vec![syntax! { (effsafe-regions #(form.name()) #..positions) }])
}
```

`print-function` and `extract` with `:extractor effsafe` stay as shadow-and-delegate bindings: they handle that option and forward everything else to the builtin.

## Migration and phasing

Four phases, each useful on its own and each validated by porting the egglog-experimental macros that need it. Adapters keep the old traits compiling until the last phase.

1. **Mechanics (not breaking).** `Syntax` with `Clone` and `Parsed`; the `Expander`; the registry by space; the interleaved top-level loop. A `Macro<T>` becomes a transformer that ignores the new capabilities; `CommandMacro` becomes the deprecated post-expansion hook; `UserDefinedCommand` is unchanged. The interleaved loop alone fixes the `unstable-fresh!` / `set-effectful` ordering bug for existing code. Port `for` and `with-ruleset`.
2. **Typing.** Typed macros and lifting. Port `unstable-fresh!` and `set-effectful`; close PR #1035 in favor of the expander API, or merge it as the internal implementation.
3. **Extensible builtins.** Register the builtin parsers as core transformers; `cx.options` with option extensions; `expand_until`. Port `with-dynamic-cost`, `:regions`, and the effsafe `print-function` / `extract` shadows. Delete the `Parser::default()` workarounds.
4. **Hygiene and templates.** Scoped identifiers with deterministic renaming; quasi-quote templates producing `Syntax` (PR #947). Consider moving `rewrite`, `birewrite`, `relation`, and `datatype` desugaring from `desugar.rs` into core transformers so wrappers see them uniformly. `resolve_program` output is observable (eggcc compares it), so this step needs a diff of the test corpus before and after.

Dispatch by head name replaces applying every `CommandMacro` to every command, which should be a measurable win on eggcc-sized programs; `scripts/bench.py` is the check. Existing tests in `tests/test_command_macros.rs` and `tests/proof_mode_regression.rs` keep passing through the adapters until phase 4 retires them.

## Open questions for the team

1. **Macro output: syntax or AST?** This proposal says syntax (re-expanded) with `Parsed` as the escape hatch, as in Racket. The cost is re-walking templates; the benefit is composition without each author calling the parser.
2. **Hygiene depth.** Per-invocation scope with deterministic renaming, or none? The flat binding structure makes the former cheap, and the review finding in PR #77 argues it is worth having.
3. **Typed macros in queries.** The design covers rule heads and top-level actions. Should pending calls be allowed in the body as well?
4. **Core sugar as macros.** Should `rewrite`, `birewrite`, `relation`, and `datatype` move into the transformer table (phase 4), given that their desugared output is observable?
5. **Macros in egglog source.** Still no (#356). The design leaves room for a pattern-template form implemented as a Rust transformer later.
6. **Naming and placement.** `Syntax` in `egglog-ast`? Keep `Sexp` as an alias for one release?
7. **Proof mode.** Typed macros must run against the pre-encoding `TypeInfo`, and lifted commands must be proof-encodable. Lifted commands are ordinary commands, so the second holds; the first needs the `Expander` to be built from the right `TypeInfo`, which is where the ad hoc selection in `process_program_internal` moves.

## References

- Flatt et al. 2023. *Rhombus: A New Spin on Macros without All the Parentheses.* OOPSLA 2023. [doi:10.1145/3622818](https://doi.org/10.1145/3622818). Spaces (section 5), the macro protocol (section 4), static information (section 6).
- Flatt, Culpepper, Findler, Darais 2012. *Macros that Work Together: Compile-Time Bindings, Partial Expansion, and Definition Contexts.* JFP 22(2). Partial expansion, stop lists, definition contexts.
- Flatt 2016. *Binding as Sets of Scopes.* POPL. Hygiene.
- Chang, Knauth, Greenman 2017. *Type Systems as Macros.* POPL. Static information as a binding protocol.
- Barrett, Christiansen, Gélineau 2020. *Predictable Macros for Hindley-Milner.* TyDe. Pausing a macro until type information is available.
- Talk: [*The Rhombus Programming Language* (ML'26)](https://www.youtube.com/watch?v=EP9TjVgtNBg).
- egglog: `src/ast/parse.rs` (`Macro<T>`, `parse_options`), `src/command_macro.rs`, `src/lib.rs` (`UserDefinedCommand`, `process_program_internal`), `src/ast/desugar.rs`; PRs [#741](https://github.com/egraphs-good/egglog/pull/741), [#947](https://github.com/egraphs-good/egglog/pull/947), [#1035](https://github.com/egraphs-good/egglog/pull/1035); issue [#356](https://github.com/egraphs-good/egglog/issues/356).
- egglog-experimental: `src/sugar/for.rs`, `src/sugar/with_ruleset.rs`, `src/set_cost.rs`, `src/fresh_macro.rs`; [PR #77](https://github.com/egraphs-good/egglog-experimental/pull/77) (`src/effsafe_extract/set_effectful.rs`, `RegionsAnnotation` in `src/effsafe_extract/mod.rs`).
- eggcc users of the macros: `dag_in_context/src/utility/effectful.egg` (`set-effectful`), `schema.egg` (`:regions`), the optimization files using `unstable-fresh!`.
