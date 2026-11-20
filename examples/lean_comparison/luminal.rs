//! Single-threaded native egglog baseline for the 225 projections in Luminal's Llama graph.
use egglog::ast::{Literal, Span};
use egglog::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog::{
    ArcSort, Core, EGraph, Primitive, Term, TermDag, TermId, Value, Write, WritePrim, WriteState,
};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use std::{fs, io};

const LAYOUTS: usize = 5;
const PROJECTIONS: usize = 225;

struct Product {
    layout: usize,
    result: i64,
    lhs: i64,
    rhs: i64,
}

struct Root {
    layout: usize,
    result: i64,
    source: i64,
    expected: (i64, i64),
}

struct Input {
    products: Vec<Product>,
    roots: Vec<Root>,
}

impl Input {
    fn read(path: &str) -> io::Result<Self> {
        let mut products = Vec::new();
        let mut roots = Vec::new();
        let mut ids = BTreeSet::new();
        let mut counts = [0_usize; LAYOUTS];
        for (line, row) in fs::read_to_string(path)?.lines().enumerate() {
            let invalid = || {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid projection row {}", line + 1),
                )
            };
            let fields: Vec<_> = row.split_whitespace().collect();
            let [tag, layout, result, rest @ ..] = fields.as_slice() else {
                return Err(invalid());
            };
            let layout = layout
                .parse::<usize>()
                .ok()
                .filter(|&i| i < LAYOUTS)
                .ok_or_else(invalid)?;
            let reg = |text: &str| {
                text.parse::<i64>()
                    .ok()
                    .filter(|&id| id >= 0)
                    .ok_or_else(invalid)
            };
            let result = reg(result)?;
            if !ids.insert(result) {
                return Err(invalid());
            }
            match (*tag, rest) {
                ("P", [lhs, rhs]) => products.push(Product {
                    layout,
                    result,
                    lhs: reg(lhs)?,
                    rhs: reg(rhs)?,
                }),
                ("R", [source, lhs, rhs]) => {
                    roots.push(Root {
                        layout,
                        result,
                        source: reg(source)?,
                        expected: (reg(lhs)?, reg(rhs)?),
                    });
                    counts[layout] += 1;
                }
                _ => return Err(invalid()),
            }
        }
        let aliases: BTreeSet<_> = products.iter().map(|p| (p.layout, p.result)).collect();
        let sources: BTreeSet<_> = roots.iter().map(|r| (r.layout, r.source)).collect();
        if products.len() != PROJECTIONS
            || roots.len() != PROJECTIONS
            || counts.contains(&0)
            || aliases != sources
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected 225 distinct product aliases and their 225 reduction roots across five layouts",
            ));
        }
        Ok(Self { products, roots })
    }
}

struct Layout {
    reg: String,
    times: String,
    sum: String,
    dot: String,
    tensor: String,
}

impl Layout {
    fn new(i: usize) -> Self {
        Self {
            reg: format!("Reg{i}"),
            times: format!("Times{i}"),
            sum: format!("Sum{i}"),
            dot: format!("Dot{i}"),
            tensor: format!("Tensor{i}"),
        }
    }

    fn declare(&self, schema: &mut String) {
        let Self {
            reg,
            times,
            sum,
            dot,
            tensor,
        } = self;
        writeln!(
            schema,
            "(sort {tensor})
(constructor {reg} (i64) {tensor} :cost 0)
(constructor {times} ({tensor} {tensor}) {tensor})
(constructor {sum} ({tensor}) {tensor} :cost 4)
(constructor {dot} ({tensor} {tensor}) {tensor})
(rewrite ({sum} ({times} a b)) ({dot} a b))"
        )
        .unwrap();
    }

    fn decode_register(&self, dag: &TermDag, id: TermId) -> Option<i64> {
        if let Term::App(reg, args) = dag.get(id)
            && reg == &self.reg
            && let [literal] = args.as_slice()
            && let Term::Lit(Literal::Int(i)) = dag.get(*literal)
        {
            Some(*i)
        } else {
            None
        }
    }

    fn decode_dot(&self, dag: &TermDag, id: TermId) -> Option<(i64, i64)> {
        if let Term::App(dot, args) = dag.get(id)
            && dot == &self.dot
            && let [a, b] = args.as_slice()
        {
            Some((
                self.decode_register(dag, *a)?,
                self.decode_register(dag, *b)?,
            ))
        } else {
            None
        }
    }
}

#[derive(Clone)]
struct Seed {
    input: Arc<Input>,
    layouts: Arc<[Layout; LAYOUTS]>,
    roots: Arc<OnceLock<Vec<Value>>>,
    output: ArcSort,
}

impl Primitive for Seed {
    fn name(&self) -> &str {
        "seed-luminal"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![self.output.clone()], span.clone()).into_box()
    }
}

impl Seed {
    fn seed(&self, state: &mut WriteState<'_, '_>) -> Result<Vec<Value>, egglog::Error> {
        for p in &self.input.products {
            let l = &self.layouts[p.layout];
            let lhs = state.add(&l.reg, p.lhs)?;
            let rhs = state.add(&l.reg, p.rhs)?;
            let product = state.add(&l.times, (lhs, rhs))?;
            let alias = state.add(&l.reg, p.result)?;
            state.union(alias, product)?;
        }
        self.input
            .roots
            .iter()
            .map(|root| {
                let l = &self.layouts[root.layout];
                let alias = state.add(&l.reg, root.source)?;
                state.add(&l.sum, (alias,))
            })
            .collect()
    }
}

impl WritePrim for Seed {
    fn apply<'a, 'db>(&self, mut state: WriteState<'a, 'db>, _: &[Value]) -> Option<Value> {
        let roots = self.seed(&mut state).expect("declared projection schema");
        self.roots.set(roots).expect("batch seed executes once");
        Some(state.base_to_value(0_i64))
    }
}

pub(super) fn run(path: &str) -> Result<(), Box<dyn Error>> {
    let input = Arc::new(Input::read(path)?);
    let layouts = Arc::new(std::array::from_fn(Layout::new));
    let mut schema = String::new();
    for layout in layouts.iter() {
        layout.declare(&mut schema);
    }

    let setup_start = Instant::now();
    let mut graph = EGraph::new(1);
    graph.parse_and_run_program(None, &schema)?;
    let sorts: Vec<_> = layouts
        .iter()
        .map(|l| graph.get_sort_by_name(&l.tensor).unwrap().clone())
        .collect();
    let roots = Arc::new(OnceLock::new());
    graph.add_write_primitive(
        Seed {
            input: input.clone(),
            layouts: layouts.clone(),
            roots: roots.clone(),
            output: graph.get_sort_by_name("i64").unwrap().clone(),
        },
        None,
    );
    let seed = graph.parse_program(None, "(let $seeded (seed-luminal))")?;
    let round = graph.parse_program(None, "(run 1)")?;
    let start = Instant::now();
    let setup_ns = start.duration_since(setup_start).as_nanos();

    graph.run_program(seed)?;
    let seeded = Instant::now();
    graph.run_program(round)?;
    let searched = Instant::now();
    let values = roots
        .get()
        .ok_or_else(|| io::Error::other("projection seed did not execute"))?;
    let requested = input
        .roots
        .iter()
        .zip(values)
        .map(|(root, &value)| (sorts[root.layout].clone(), value))
        .collect();
    // One shared cost solution and one reconstruction cache across all 225 roots.
    let extracted = graph.extract_best(requested)?;
    if extracted.terms.len() != PROJECTIONS {
        return Err(io::Error::other("missing extraction roots").into());
    }
    for (root, best) in input.roots.iter().zip(extracted.terms) {
        let decoded = best
            .filter(|best| best.cost == 3)
            .and_then(|best| layouts[root.layout].decode_dot(&extracted.termdag, best.term));
        if decoded != Some(root.expected) {
            return Err(io::Error::other(format!(
                "register {} expected Dot{:?}, extracted {decoded:?}",
                root.result, root.expected
            ))
            .into());
        }
    }
    let done = Instant::now();
    writeln!(
        io::stdout().lock(),
        "layouts={LAYOUTS} products={PROJECTIONS} roots={PROJECTIONS} compiler_ns={} setup_ns={} seed_ns={} search_ns={} extract_ns={}",
        done.duration_since(start).as_nanos(),
        setup_ns,
        seeded.duration_since(start).as_nanos(),
        searched.duration_since(seeded).as_nanos(),
        done.duration_since(searched).as_nanos()
    )?;
    Ok(())
}
