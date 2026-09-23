//! A versioned, structural Egglog program shared by language frontends.
//!
//! This is the unresolved command tree, not an e-graph snapshot or a resolved
//! executable. Execution still performs macro expansion, desugaring, and core
//! typechecking. Host extensions and files referenced by commands must be made
//! available separately. See `docs/shared-program.md` for the wire contract.

use crate::ast::*;
use egglog_ast::span::Span;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::Write;

/// The maximum accepted JSON/text input or emitted JSON document size.
pub const MAX_PROGRAM_BYTES: usize = 16 * 1024 * 1024;
/// Bounds source nesting before text parsing; native Rust trees retain their
/// frontend's limits. JSON has its own 1024-level envelope limit.
pub const MAX_PROGRAM_DEPTH: usize = 256;
pub const MAX_PROGRAM_NODES: usize = 100_000;

fn check_json_limits(input: &str) -> Result<(), ProgramError> {
    if input.len() > MAX_PROGRAM_BYTES {
        return Err(ProgramError::Limit("16 MiB JSON"));
    }
    let (mut depth, mut nodes, mut quoted, mut escaped) = (0usize, 0usize, false, false);
    for byte in input.bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    depth += 1;
                    nodes += 1;
                    if depth > 1024 {
                        return Err(ProgramError::Limit("1024 JSON nesting levels"));
                    }
                    if nodes > MAX_PROGRAM_NODES {
                        return Err(ProgramError::Limit("100000 JSON objects/arrays"));
                    }
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    Ok(())
}

fn decode_json<T: DeserializeOwned>(input: &str) -> Result<T, ProgramError> {
    check_json_limits(input)?;
    let mut deserializer = serde_json::Deserializer::from_str(input);
    deserializer.disable_recursion_limit();
    let value = T::deserialize(serde_stacker::Deserializer::new(&mut deserializer))?;
    deserializer.end()?;
    Ok(value)
}

fn encode_json(value: &impl Serialize) -> Result<String, ProgramError> {
    struct BoundedJson(Vec<u8>);
    impl Write for BoundedJson {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0.len().saturating_add(bytes.len()) > MAX_PROGRAM_BYTES {
                return Err(std::io::Error::other("program exceeds 16 MiB JSON output"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut output = BoundedJson(Vec::new());
    value.serialize(serde_stacker::Serializer::new(
        &mut serde_json::Serializer::new(&mut output),
    ))?;
    let json = String::from_utf8(output.0).expect("JSON serialization produces UTF-8");
    check_json_limits(&json)?;
    Ok(json)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum ProgramFormat {
    #[serde(rename = "egglog-program-v1")]
    V1,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub format: ProgramFormat,
    pub commands: Vec<Command>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("invalid program JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("program exceeds {0}")]
    Limit(&'static str),
    #[error("invalid UTF-8 source span range")]
    InvalidSpan,
    #[error("program cannot be represented as replayable Egglog text: {0}")]
    NotReplayable(String),
}

impl Program {
    pub fn new(commands: Vec<Command>) -> Result<Self, ProgramError> {
        let program = Self {
            format: ProgramFormat::V1,
            commands,
        };
        program.validate()?;
        Ok(program)
    }

    /// Parse built-in Egglog syntax without creating or mutating an e-graph.
    /// For a parser with registered extensions, construct from its commands.
    pub fn parse(filename: Option<String>, input: &str) -> Result<Self, ProgramError> {
        if input.len() > MAX_PROGRAM_BYTES {
            return Err(ProgramError::Limit("16 MiB input"));
        }
        // Bound the parser's recursive descent before parsing. Parentheses in
        // quoted strings and line comments do not contribute to nesting.
        let (mut depth, mut nodes, mut quoted, mut escaped, mut comment, mut atom) =
            (0usize, 0usize, false, false, false, false);
        for ch in input.chars() {
            if comment {
                comment = ch != '\n';
            } else if quoted {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    quoted = false;
                }
            } else {
                match ch {
                    ';' => {
                        comment = true;
                        atom = false;
                    }
                    // The lexer recognizes strings only at a token boundary.
                    // Quotes inside atoms (for example `$x"`) are ordinary
                    // atom characters and must not hide subsequent nesting.
                    '"' if !atom => quoted = true,
                    '(' => {
                        atom = false;
                        depth += 1;
                        nodes += 1;
                        if nodes > MAX_PROGRAM_NODES {
                            return Err(ProgramError::Limit("100000 source lists"));
                        }
                        if depth > MAX_PROGRAM_DEPTH {
                            return Err(ProgramError::Limit("256 levels of nesting"));
                        }
                    }
                    ')' => {
                        atom = false;
                        depth = depth.saturating_sub(1);
                    }
                    ch if ch.is_whitespace() => atom = false,
                    _ => atom = true,
                }
            }
        }
        Self::new(Parser::default().get_program_from_string(filename, input)?)
    }

    pub fn from_json(input: &str) -> Result<Self, ProgramError> {
        let program: Self = decode_json(input)?;
        program.validate()?;
        Ok(program)
    }

    pub fn to_json(&self) -> Result<String, ProgramError> {
        self.validate()?;
        encode_json(self)
    }
    pub fn schema() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(Self)).expect("schema is JSON")
    }

    /// Diagnostic formatting only. Some native names, literals, and internal
    /// metadata cannot be written in the surface language. Use JSON to preserve
    /// the program or `to_replayable_egglog` to check surface representability.
    pub fn to_egglog(&self) -> String {
        use std::fmt::Write;
        fn write_command(command: &Command, output: &mut String) {
            match command {
                // `run-schedule` itself introduces the outer sequence. Printing
                // an extra `(seq ...)` would change every parsed schedule tree.
                Command::RunSchedule(Schedule::Sequence(_, schedules)) => {
                    output.push_str("(run-schedule");
                    for schedule in schedules {
                        write!(output, " {schedule}").unwrap();
                    }
                    output.push(')');
                }
                Command::Fail(_, nested) => {
                    output.push_str("(fail ");
                    write_command(nested, output);
                    output.push(')');
                }
                _ => write!(output, "{command}").unwrap(),
            }
        }
        let mut output = String::new();
        for command in &self.commands {
            write_command(command, &mut output);
            output.push('\n');
        }
        output
    }

    /// Render only when the built-in parser reconstructs every field except
    /// source locations and singleton schedule sequences. This does not bundle
    /// referenced files or host hooks.
    pub fn to_replayable_egglog(&self) -> Result<String, ProgramError> {
        self.validate()?;
        let text = self.to_egglog();
        let reparsed = Self::parse(None, &text)
            .map_err(|error| ProgramError::NotReplayable(error.to_string()))?;
        let mut original: serde_json::Value = decode_json(&self.to_json()?)?;
        let mut reconstructed: serde_json::Value = decode_json(&reparsed.to_json()?)?;
        // Spans also occur in positional tuple payloads, so identify the tagged
        // span variants rather than just removing fields named `span`.
        fn normalize(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Object(fields) => {
                    let is_span = match fields.get("type").and_then(|tag| tag.as_str()) {
                        Some("Egglog" | "Rust") => true,
                        Some("Panic") => !fields.contains_key("value"),
                        _ => false,
                    };
                    if is_span {
                        *value = serde_json::Value::Null;
                    } else {
                        fields.values_mut().for_each(normalize);
                        // The surface parser wraps run-schedule/saturate bodies
                        // in Sequence. Singleton sequences execute identically,
                        // as GenericSchedule::flatten_sequences also recognizes.
                        if fields.get("type").and_then(|tag| tag.as_str()) == Some("Sequence")
                            && let Some(children) = fields
                                .get("value")
                                .and_then(|payload| payload.get(1))
                                .and_then(|children| children.as_array())
                            && children.len() == 1
                        {
                            *value = children[0].clone();
                        }
                    }
                }
                serde_json::Value::Array(values) => values.iter_mut().for_each(normalize),
                _ => {}
            }
        }
        normalize(&mut original);
        normalize(&mut reconstructed);
        if original != reconstructed {
            let mut work = vec![("program".to_owned(), &original, &reconstructed)];
            let mut difference = "program".to_owned();
            while let Some((path, lhs, rhs)) = work.pop() {
                if lhs == rhs {
                    continue;
                }
                match (lhs, rhs) {
                    (serde_json::Value::Object(lhs), serde_json::Value::Object(rhs))
                        if lhs.len() == rhs.len() && lhs.keys().eq(rhs.keys()) =>
                    {
                        for (key, value) in lhs {
                            work.push((format!("{path}.{key}"), value, &rhs[key]));
                        }
                    }
                    (serde_json::Value::Array(lhs), serde_json::Value::Array(rhs))
                        if lhs.len() == rhs.len() =>
                    {
                        for (index, (lhs, rhs)) in lhs.iter().zip(rhs).enumerate() {
                            work.push((format!("{path}[{index}]"), lhs, rhs));
                        }
                    }
                    _ => {
                        difference = path;
                        break;
                    }
                }
            }
            return Err(ProgramError::NotReplayable(format!(
                "surface formatting changes {difference}; use JSON to preserve command fields, names, and literal bits"
            )));
        }
        Ok(text)
    }

    /// Iteratively validate spans in native and imported trees. Native frontend
    /// limits are unchanged; the checked wire/text methods bound their own
    /// inputs. This does not infer types or validate extension availability.
    pub fn validate(&self) -> Result<(), ProgramError> {
        for command in &self.commands {
            validate_command(command)?;
        }
        Ok(())
    }
}

// Iterative span validation shared by programs and command records.
fn validate_command(command: &Command) -> Result<(), ProgramError> {
    enum Node<'a> {
        Command(&'a Command),
        Expr(&'a Expr),
        Action(&'a Action),
        Fact(&'a Fact),
        Schedule(&'a Schedule),
        Rule(&'a Rule),
        Rewrite(&'a Rewrite),
        Span(&'a Span),
    }
    let mut pending: Vec<_> = vec![Node::Command(command)];
    while let Some(node) = pending.pop() {
        let mut add = |node| pending.push(node);
        match node {
            Node::Span(Span::Egglog(span)) => {
                if span.i > span.j
                    || span.j > span.file.contents.len()
                    || !span.file.contents.is_char_boundary(span.i)
                    || !span.file.contents.is_char_boundary(span.j)
                {
                    return Err(ProgramError::InvalidSpan);
                }
            }
            Node::Span(_) => {}
            Node::Expr(expr) => match expr {
                Expr::Var(span, _) | Expr::Lit(span, _) => add(Node::Span(span)),
                Expr::Call(span, _, args) => {
                    add(Node::Span(span));
                    for arg in args {
                        add(Node::Expr(arg));
                    }
                }
            },
            Node::Fact(fact) => match fact {
                Fact::Eq(span, lhs, rhs) => {
                    add(Node::Span(span));
                    add(Node::Expr(lhs));
                    add(Node::Expr(rhs));
                }
                Fact::Fact(expr) => add(Node::Expr(expr)),
            },
            Node::Action(action) => match action {
                Action::Let(span, _, expr) | Action::Expr(span, expr) => {
                    add(Node::Span(span));
                    add(Node::Expr(expr));
                }
                Action::Set(span, _, args, value) => {
                    add(Node::Span(span));
                    add(Node::Expr(value));
                    for arg in args {
                        add(Node::Expr(arg));
                    }
                }
                Action::Change(span, _, _, args) => {
                    add(Node::Span(span));
                    for arg in args {
                        add(Node::Expr(arg));
                    }
                }
                Action::Union(span, lhs, rhs) => {
                    add(Node::Span(span));
                    add(Node::Expr(lhs));
                    add(Node::Expr(rhs));
                }
                Action::Panic(span, _) => add(Node::Span(span)),
            },
            Node::Rule(rule) => {
                add(Node::Span(&rule.span));
                for fact in &rule.body {
                    add(Node::Fact(fact));
                }
                for action in &rule.head.0 {
                    add(Node::Action(action));
                }
            }
            Node::Rewrite(rule) => {
                add(Node::Span(&rule.span));
                add(Node::Expr(&rule.lhs));
                add(Node::Expr(&rule.rhs));
                for fact in &rule.conditions {
                    add(Node::Fact(fact));
                }
            }
            Node::Schedule(schedule) => match schedule {
                Schedule::Saturate(span, child) | Schedule::Repeat(span, _, child) => {
                    add(Node::Span(span));
                    add(Node::Schedule(child));
                }
                Schedule::Sequence(span, children) => {
                    add(Node::Span(span));
                    for child in children {
                        add(Node::Schedule(child));
                    }
                }
                Schedule::Run(span, config) => {
                    add(Node::Span(span));
                    for fact in config.until.iter().flatten() {
                        add(Node::Fact(fact));
                    }
                }
            },
            Node::Command(command) => match command {
                Command::Sort {
                    span,
                    presort_and_args,
                    ..
                } => {
                    add(Node::Span(span));
                    if let Some((_, args)) = presort_and_args {
                        for arg in args {
                            add(Node::Expr(arg));
                        }
                    }
                }
                Command::Datatype { span, variants, .. } => {
                    add(Node::Span(span));
                    for variant in variants {
                        add(Node::Span(&variant.span));
                    }
                }
                Command::Datatypes { span, datatypes } => {
                    add(Node::Span(span));
                    for (span, _, datatype) in datatypes {
                        add(Node::Span(span));
                        match datatype {
                            Subdatatypes::Variants(variants) => {
                                for variant in variants {
                                    add(Node::Span(&variant.span));
                                }
                            }
                            Subdatatypes::NewSort(_, args) => {
                                for arg in args {
                                    add(Node::Expr(arg));
                                }
                            }
                        }
                    }
                }
                Command::Function { span, merge, .. } => {
                    add(Node::Span(span));
                    if let Some(expr) = merge {
                        add(Node::Expr(expr));
                    }
                }
                Command::Constructor { span, .. }
                | Command::Relation { span, .. }
                | Command::Input { span, .. }
                | Command::AddRuleset(span, _)
                | Command::UnstableCombinedRuleset(span, _, _)
                | Command::PrintOverallStatistics(span, _)
                | Command::PrintFunction(span, _, _, _, _)
                | Command::PrintSize(span, _)
                | Command::Pop(span, _)
                | Command::Include(span, _)
                | Command::ProveExists(span, _) => add(Node::Span(span)),
                Command::Rule { rule } => add(Node::Rule(rule)),
                Command::Rewrite(_, rule, _) | Command::BiRewrite(_, rule) => {
                    add(Node::Rewrite(rule))
                }
                Command::Action(action) => add(Node::Action(action)),
                Command::Extract(span, expr, variants) => {
                    add(Node::Span(span));
                    add(Node::Expr(expr));
                    add(Node::Expr(variants));
                }
                Command::RunSchedule(schedule) => add(Node::Schedule(schedule)),
                Command::Check(span, facts) | Command::Prove(span, facts) => {
                    add(Node::Span(span));
                    for fact in facts {
                        add(Node::Fact(fact));
                    }
                }
                Command::Output { span, exprs, .. } | Command::UserDefined(span, _, exprs) => {
                    add(Node::Span(span));
                    for expr in exprs {
                        add(Node::Expr(expr));
                    }
                }
                Command::Fail(span, command) => {
                    add(Node::Span(span));
                    add(Node::Command(command));
                }
                Command::Push(_) => {}
            },
        }
    }
    Ok(())
}

/// A command-submission record, not an e-graph snapshot or whole API trace.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CommandRecord {
    pub entries: Vec<RecordedCommand>,
}

impl CommandRecord {
    /// Encode the record with the same size, nesting, and stack-growth limits
    /// as Program JSON. Malformed native source spans are rejected.
    pub fn to_json(&self) -> Result<String, ProgramError> {
        for entry in &self.entries {
            validate_command(&entry.command)?;
        }
        encode_json(self)
    }

    /// Decode a bounded record, preserving each attempted command and outcome.
    pub fn from_json(input: &str) -> Result<Self, ProgramError> {
        let record: Self = decode_json(input)?;
        for entry in &record.entries {
            validate_command(&entry.command)?;
        }
        Ok(record)
    }

    /// Preserve the attempted prefix, including a failed command. Replaying a
    /// failure can repeat its partial effects; this is not a successful-state snapshot.
    pub fn program(&self) -> Result<Program, ProgramError> {
        Program::new(
            self.entries
                .iter()
                .map(|entry| entry.command.clone())
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecordedCommand {
    pub command: Command,
    pub outcome: CommandOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "value", deny_unknown_fields)]
pub enum CommandOutcome {
    /// The command has started. It may still be running or have unwound in host code.
    Pending,
    Success,
    /// Earlier actions within this command may already have mutated the e-graph.
    Failure {
        message: String,
    },
}
