//! Enter egglog through its native constructor API. The timer starts with a parsed
//! compiler IR, and ends after reconstructing that IR from the extracted TermDag.
use egglog::ast::{Expr, Literal, Span};
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::{
    ArcSort, CommandOutput, EGraph, Primitive, Term, TermDag, TermId, Value, Write, WritePrim,
    WriteState,
};
use std::io::Write as _;
use std::{error::Error, fs, io, sync::Arc, time::Instant};

#[derive(PartialEq, Debug)]
enum Ir {
    Reg(i64),
    Atom(&'static str),
    Unary(&'static str, Box<Ir>),
    Binary(&'static str, Box<(Ir, Ir)>),
    Mux(Box<(Ir, Ir, Ir)>),
}

impl Ir {
    fn node<T>(name: &str, args: &[T], child: impl Fn(&T) -> io::Result<Self>) -> io::Result<Self> {
        // Sharing this arity check keeps the compiler's input and output types identical.
        let name = [
            "Yes", "No", "Neg", "Conj", "Disj", "Mux", "One", "Plus", "Times", "Pow2", "Shl",
        ]
        .into_iter()
        .find(|&op| op == name)
        .ok_or_else(|| io::Error::other("unknown compiler operator"))?;
        match (name, args) {
            ("Yes" | "No" | "One", []) => Ok(Self::Atom(name)),
            ("Neg" | "Pow2", [x]) => Ok(Self::Unary(name, Box::new(child(x)?))),
            ("Conj" | "Disj" | "Plus" | "Times" | "Shl", [a, b]) => {
                Ok(Self::Binary(name, Box::new((child(a)?, child(b)?))))
            }
            ("Mux", [c, a, b]) => Ok(Self::Mux(Box::new((child(c)?, child(a)?, child(b)?)))),
            _ => Err(io::Error::other(format!("wrong arity for {name}"))),
        }
    }

    fn parse(expr: &Expr) -> io::Result<Self> {
        let Expr::Call(_, name, args) = expr else {
            return Err(io::Error::other("expected compiler IR"));
        };
        if name == "Reg" {
            let [Expr::Lit(_, Literal::Int(i))] = args.as_slice() else {
                return Err(io::Error::other("expected register index"));
            };
            return Ok(Self::Reg(*i));
        }
        Self::node(name, args, Self::parse)
    }

    fn decode(dag: &TermDag, id: TermId) -> io::Result<Self> {
        let Term::App(name, args) = dag.get(id) else {
            return Err(io::Error::other("expected extracted IR"));
        };
        if name == "Reg" {
            if let [n] = args.as_slice()
                && let Term::Lit(Literal::Int(i)) = dag.get(*n)
            {
                return Ok(Self::Reg(*i));
            }
            return Err(io::Error::other("invalid extracted register"));
        }
        Self::node(name, args, |n| Self::decode(dag, *n))
    }

    fn seed(&self, state: &mut WriteState<'_, '_>) -> Result<Value, egglog::Error> {
        match self {
            Self::Reg(i) => state.add("Reg", *i),
            Self::Atom(op) => state.add(op, &[] as &[Value]),
            Self::Unary(op, x) => {
                let a = x.seed(state)?;
                state.add(op, (a,))
            }
            Self::Binary(op, args) => {
                let a = args.0.seed(state)?;
                let b = args.1.seed(state)?;
                state.add(op, (a, b))
            }
            Self::Mux(args) => {
                let c = args.0.seed(state)?;
                let a = args.1.seed(state)?;
                let b = args.2.seed(state)?;
                state.add("Mux", (c, a, b))
            }
        }
    }
}

#[derive(Clone)]
struct Seed {
    input: Arc<Ir>,
    sort: ArcSort,
}

impl Primitive for Seed {
    fn name(&self) -> &str {
        "bench-seed"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.sort.clone()], span.clone()).into_box()
    }
}

impl WritePrim for Seed {
    fn apply<'a, 'db>(&self, mut state: WriteState<'a, 'db>, _: &[Value]) -> Option<Value> {
        Some(
            self.input
                .seed(&mut state)
                .expect("input uses only declared IR constructors"),
        )
    }
}

pub(super) fn run(
    setup: &str,
    schedule: &str,
    input: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    let mut graph = EGraph::new(1);
    graph.parse_and_run_program(Some(setup.into()), &fs::read_to_string(setup)?)?;
    let mut read = |path: &str| {
        let expr = graph
            .parser
            .get_expr_from_string(Some(path.into()), &fs::read_to_string(path)?)?;
        Ok::<_, Box<dyn Error>>(Ir::parse(&expr)?)
    };
    let input = Arc::new(read(input)?);
    let want = read(expected)?;
    let sort = graph
        .get_sort_by_name("E")
        .ok_or_else(|| io::Error::other("missing sort E"))?
        .clone();
    graph.add_write_primitive(Seed { input, sort }, None);
    let run = graph.parse_program(Some(schedule.into()), &fs::read_to_string(schedule)?)?;

    let start = Instant::now();
    let outputs = graph.run_program(run)?;
    let Some(CommandOutput::ExtractBest(dag, _, term)) = outputs.last() else {
        return Err(io::Error::other("expected one extracted IR").into());
    };
    let output = Ir::decode(dag, *term)?;
    let elapsed = start.elapsed().as_nanos();

    if output != want {
        return Err(io::Error::other("unexpected optimized compiler IR").into());
    }
    writeln!(io::stdout().lock(), "compiler_ns={elapsed}")?;
    Ok(())
}
