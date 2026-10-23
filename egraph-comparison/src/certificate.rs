use crate::ids::{RowId, TermId};
use crate::{Database, Error, FunctionKind, Row, compare, refine::Partition};
use egglog_numeric_id::NumericId;
use fixedbitset::FixedBitSet;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Left,
    Right,
}

/// A finite term DAG. Children refer to earlier entries in the same vector,
/// preventing exponential expansion (and recursion on deeply nested terms).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Term {
    Literal {
        sort: String,
        value: String,
    },
    Apply {
        function: String,
        inputs: Vec<TermId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Certificate {
    /// `term` exists in `side`, but has no interpretation in the other input.
    MissingTerm {
        side: Side,
        terms: Vec<Term>,
        term: TermId,
    },
    /// Both terms exist on both sides, but are equal only in `side`.
    UnequalTerms {
        side: Side,
        terms: Vec<Term>,
        first: TermId,
        second: TermId,
    },
    /// A constructor class's depth-bounded bisimulation observation occurs on
    /// only one side. This also covers cycles that have no finite ground terms.
    /// The verifier rebuilds the disjoint union of the two constructor graphs,
    /// initially groups classes by sort, and computes exactly `rounds` successive
    /// partitions from node labels and child blocks. It then checks that `class`
    /// is a constructor root whose block has no constructor root on the other
    /// side. The certificate stores an input-local class name, not a block ID;
    /// verification recomputes every block and trusts no serialized partition.
    Structure {
        side: Side,
        class: String,
        rounds: usize,
    },
    /// A declaration is absent or has a different signature/kind.
    Declaration { function: String },
    /// This row exists only on `side`, modulo full-database bisimulation.
    Row { side: Side, row: Row },
}

fn sides<'a>(side: Side, left: &'a Database, right: &'a Database) -> (&'a Database, &'a Database) {
    match side {
        Side::Left => (left, right),
        Side::Right => (right, left),
    }
}

/// Return a certificate for any difference, preferring finite term witnesses.
/// The fallback is a finite structural observation, never a fabricated ground
/// term for an ungrounded cycle. Equal databases return `None`.
#[doc = include_str!("../tests/support/representatives.md")]
pub fn certificate(left: &Database, right: &Database) -> Result<Option<Certificate>, Error> {
    let result = compare(left, right)?;
    if result.database_equal {
        return Ok(None);
    }
    // Prefer an explicit constructor term or equality that a user can inspect.
    // Search both directions: a term can be absent from either database.
    if !result.terms_equal {
        for side in [Side::Left, Side::Right] {
            let (source, target) = sides(side, left, right);
            if let Some(witness) = ground_certificate(source, target, side) {
                return Ok(Some(witness));
            }
        }
    }
    // A pure cycle or an empty class may have no finite representative. Use a
    // finite-depth structural observation when ground-term search cannot explain
    // the constructor mismatch; refinement reaches a fixed point in finite time.
    let mut partition = Partition::new(left, right);
    partition.finish();
    if !result.terms_equal {
        for (side, source, target) in [
            (Side::Left, &partition.left, &partition.right),
            (Side::Right, &partition.right, &partition.left),
        ] {
            let other = partition.root_blocks(target);
            for (class, id) in &source.index {
                if source.roots.binary_search(id).is_ok()
                    && !other.contains(&partition.blocks[id.index()])
                {
                    return Ok(Some(Certificate::Structure {
                        side,
                        class: (*class).to_owned(),
                        rounds: partition.rounds,
                    }));
                }
            }
        }
    }
    // With constructor behavior matched (or no structural witness found),
    // declarations and full-database rows account for the remaining differences.
    for function in left.functions.keys().chain(right.functions.keys()) {
        if left.functions.get(function) != right.functions.get(function) {
            return Ok(Some(Certificate::Declaration {
                function: function.clone(),
            }));
        }
    }
    let mut partition = Partition::database(left, right);
    partition.finish();
    for side in [Side::Left, Side::Right] {
        let (source, _) = sides(side, left, right);
        for row in &source.rows {
            let witness = Certificate::Row {
                side,
                row: row.clone(),
            };
            if verify_with_partition(&witness, left, right, &partition) {
                return Ok(Some(witness));
            }
        }
    }
    Err(Error::MissingWitness)
}

// Here "productive" means that a class contains a finite ground constructor
// term (the grammar/inductive sense, not coinductive productivity). Literals and
// nullary constructors seed the worklist; an application becomes ready once all
// arguments have representatives. A pure x=f(x) cycle never becomes ready, while
// a class containing both A() and f(x) gets the representative A(). See the
// executable example on `certificate` above. Each row becomes ready once;
// repeated arguments are counted separately so one notification satisfies each
// occurrence. The result supplies reusable source terms to `ground_certificate`.
fn representatives(db: &Database) -> (Vec<Term>, BTreeMap<String, TermId>) {
    let mut terms = Vec::new();
    let mut reps = BTreeMap::new();
    let mut ready = VecDeque::new();
    for (id, class) in &db.classes {
        if let Some(value) = &class.literal {
            reps.insert(id.clone(), TermId::from_usize(terms.len()));
            terms.push(Term::Literal {
                sort: class.sort.clone(),
                value: value.clone(),
            });
        }
    }
    let mut waiting: BTreeMap<&str, Vec<RowId>> = BTreeMap::new();
    let mut remaining = vec![0; db.rows.len()];
    for (i, row) in db.rows.iter().enumerate() {
        if db.functions[&row.function].kind != FunctionKind::Constructor {
            continue;
        }
        for input in &row.inputs {
            if !reps.contains_key(input) {
                remaining[i] += 1;
                waiting.entry(input).or_default().push(RowId::from_usize(i));
            }
        }
        if remaining[i] == 0 {
            ready.push_back(RowId::from_usize(i));
        }
    }
    while let Some(i) = ready.pop_front() {
        let row = &db.rows[i.index()];
        if reps.contains_key(&row.output) {
            continue;
        }
        let term = Term::Apply {
            function: row.function.clone(),
            inputs: row.inputs.iter().map(|id| reps[id]).collect(),
        };
        reps.insert(row.output.clone(), TermId::from_usize(terms.len()));
        terms.push(term);
        if let Some(rows) = waiting.remove(row.output.as_str()) {
            for i in rows {
                remaining[i.index()] -= 1;
                if remaining[i.index()] == 0 {
                    ready.push_back(i);
                }
            }
        }
    }
    (terms, reps)
}

// Interpret a certificate's finite term DAG in a particular input database.
// Ground witness extraction uses this to test source representatives in the
// target; verification evaluates the supplied terms independently in both inputs.
// Lookup uses only literals and constructors, never ordinary function rows or
// refinement blocks. A missing argument/application yields no interpretation.
struct Evaluator<'a> {
    db: &'a Database,
    literals: BTreeMap<(&'a str, &'a str), &'a str>,
    calls: BTreeMap<(&'a str, Vec<&'a str>), &'a str>,
}

impl<'a> Evaluator<'a> {
    fn new(db: &'a Database) -> Self {
        let literals = db
            .classes
            .iter()
            .filter_map(|(id, c)| {
                c.literal
                    .as_ref()
                    .map(|v| ((c.sort.as_str(), v.as_str()), id.as_str()))
            })
            .collect();
        let calls = db
            .rows
            .iter()
            .filter(|r| db.functions[&r.function].kind == FunctionKind::Constructor)
            .map(|r| {
                (
                    (
                        r.function.as_str(),
                        r.inputs.iter().map(String::as_str).collect(),
                    ),
                    r.output.as_str(),
                )
            })
            .collect();
        Self {
            db,
            literals,
            calls,
        }
    }

    fn term(&self, term: &Term, values: &[Option<&'a str>]) -> Option<&'a str> {
        match term {
            Term::Literal { sort, value } => {
                self.literals.get(&(sort.as_str(), value.as_str())).copied()
            }
            Term::Apply { function, inputs } => {
                let schema = self.db.functions.get(function)?;
                if schema.kind != FunctionKind::Constructor {
                    return None;
                }
                let args = inputs
                    .iter()
                    .map(|&i| values.get(i.index()).copied().flatten())
                    .collect::<Option<Vec<_>>>()?;
                self.calls.get(&(function.as_str(), args)).copied()
            }
        }
    }

    fn all(&self, terms: &[Term]) -> Vec<Option<&'a str>> {
        let mut values = Vec::with_capacity(terms.len());
        for term in terms {
            values.push(self.term(term, &values));
        }
        values
    }
}

fn ground_certificate(source: &Database, target: &Database, side: Side) -> Option<Certificate> {
    let (mut terms, reps) = representatives(source);
    let evaluator = Evaluator::new(target);
    let mut values = evaluator.all(&terms);
    for row in &source.rows {
        if source.functions[&row.function].kind != FunctionKind::Constructor {
            continue;
        }
        let Some(inputs) = row
            .inputs
            .iter()
            .map(|id| reps.get(id).copied())
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let term = Term::Apply {
            function: row.function.clone(),
            inputs,
        };
        let value = evaluator.term(&term, &values);
        let term_id = TermId::from_usize(terms.len());
        terms.push(term);
        values.push(value);
        let representative = reps[&row.output];
        if value.is_none() {
            let (terms, roots) = compact(terms, &[term_id]);
            return Some(Certificate::MissingTerm {
                side,
                terms,
                term: roots[0],
            });
        }
        if values[representative.index()].is_none() {
            let (terms, roots) = compact(terms, &[representative]);
            return Some(Certificate::MissingTerm {
                side,
                terms,
                term: roots[0],
            });
        }
        if value != values[representative.index()] {
            let (terms, roots) = compact(terms, &[representative, term_id]);
            return Some(Certificate::UnequalTerms {
                side,
                terms,
                first: roots[0],
                second: roots[1],
            });
        }
    }
    None
}

// Keep only the sub-DAG reachable from the witness roots. Source extraction
// creates representatives for many unrelated classes; including them would make
// certificates unnecessarily large. Backward marking works because every child
// precedes its parent. A forward pass removes unused entries and remaps children
// and roots to dense TermIds while preserving this topological order.
fn compact(terms: Vec<Term>, roots: &[TermId]) -> (Vec<Term>, Vec<TermId>) {
    let mut used = FixedBitSet::with_capacity(terms.len());
    for &root in roots {
        used.insert(root.index());
    }
    for i in (0..terms.len()).rev() {
        if used.contains(i)
            && let Term::Apply { inputs, .. } = &terms[i]
        {
            for &child in inputs {
                used.insert(child.index());
            }
        }
    }
    let mut mapping = vec![TermId::from_usize(0); terms.len()];
    let mut result = Vec::new();
    for (i, mut term) in terms.into_iter().enumerate() {
        if !used.contains(i) {
            continue;
        }
        if let Term::Apply { inputs, .. } = &mut term {
            for child in inputs {
                *child = mapping[child.index()];
            }
        }
        mapping[i] = TermId::from_usize(result.len());
        result.push(term);
    }
    (result, roots.iter().map(|&i| mapping[i.index()]).collect())
}

/// Verify an untrusted certificate using the inputs. Ground certificates are
/// checked by ordinary constructor lookup, independently of partition refinement.
pub fn verify(certificate: &Certificate, left: &Database, right: &Database) -> Result<bool, Error> {
    left.validate()?;
    right.validate()?;
    match certificate {
        Certificate::MissingTerm { side, terms, term } => {
            if !valid_dag(terms) {
                return Ok(false);
            }
            let (source, target) = sides(*side, left, right);
            let a = Evaluator::new(source).all(terms);
            let b = Evaluator::new(target).all(terms);
            Ok(matches!(terms.get(term.index()), Some(Term::Apply { .. }))
                && a.get(term.index()).is_some_and(Option::is_some)
                && b.get(term.index()) == Some(&None))
        }
        Certificate::UnequalTerms {
            side,
            terms,
            first,
            second,
        } => {
            if !valid_dag(terms) {
                return Ok(false);
            }
            let (source, target) = sides(*side, left, right);
            let a = Evaluator::new(source).all(terms);
            let b = Evaluator::new(target).all(terms);
            let ids = (
                a.get(first.index()),
                a.get(second.index()),
                b.get(first.index()),
                b.get(second.index()),
            );
            Ok(
                matches!(ids, (Some(Some(x)), Some(Some(y)), Some(Some(u)), Some(Some(v))) if x == y && u != v),
            )
        }
        Certificate::Declaration { function } => {
            Ok(left.functions.get(function) != right.functions.get(function))
        }
        _ => {
            let mut partition = if matches!(certificate, Certificate::Structure { .. }) {
                Partition::new(left, right)
            } else {
                Partition::database(left, right)
            };
            if let Certificate::Structure { rounds, .. } = certificate {
                if *rounds > partition.blocks.len() + 1 {
                    return Ok(false);
                }
                for _ in 0..*rounds {
                    partition.step();
                }
            } else {
                partition.finish();
            }
            Ok(verify_with_partition(certificate, left, right, &partition))
        }
    }
}

// Validate the untrusted DAG's topological-order invariant before evaluation:
// each child index must be strictly less than its parent's index. This rejects
// self-cycles, forward/out-of-bounds references, and all longer cycles without
// recursion. Certificate root bounds are checked separately by `verify`.
fn valid_dag(terms: &[Term]) -> bool {
    terms.iter().enumerate().all(|(i, term)| match term {
        Term::Literal { .. } => true,
        Term::Apply { inputs, .. } => inputs.iter().all(|&child| child.index() < i),
    })
}

fn verify_with_partition(
    witness: &Certificate,
    left: &Database,
    right: &Database,
    partition: &Partition,
) -> bool {
    let (side, row) = match witness {
        Certificate::Structure { side, class, .. } => {
            let (source, target) = match side {
                Side::Left => (&partition.left, &partition.right),
                Side::Right => (&partition.right, &partition.left),
            };
            return source.index.get(class.as_str()).is_some_and(|id| {
                source.roots.binary_search(id).is_ok()
                    && !partition
                        .root_blocks(target)
                        .contains(&partition.blocks[id.index()])
            });
        }
        Certificate::Row { side, row } => (*side, row),
        _ => return false,
    };
    let (source, target) = sides(side, left, right);
    if !source.rows.contains(row) {
        return false;
    }
    let (a, b) = match side {
        Side::Left => (&partition.left, &partition.right),
        Side::Right => (&partition.right, &partition.left),
    };
    !target.rows.iter().any(|other| {
        row.function == other.function
            && row.subsumed == other.subsumed
            && row.inputs.len() == other.inputs.len()
            && row
                .inputs
                .iter()
                .chain([&row.output])
                .zip(other.inputs.iter().chain([&other.output]))
                .all(|(x, y)| {
                    partition.blocks[a.index[x.as_str()].index()]
                        == partition.blocks[b.index[y.as_str()].index()]
                })
    })
}
