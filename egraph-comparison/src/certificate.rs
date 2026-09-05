use crate::{Database, Error, FunctionKind, Row, compare, refine::Partition};
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
        inputs: Vec<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Certificate {
    /// `term` exists in `side`, but has no interpretation in the other input.
    MissingTerm {
        side: Side,
        terms: Vec<Term>,
        term: usize,
    },
    /// Both terms exist on both sides, but are equal only in `side`.
    UnequalTerms {
        side: Side,
        terms: Vec<Term>,
        first: usize,
        second: usize,
    },
    /// A constructor class's depth-bounded bisimulation observation occurs on
    /// only one side. This also covers cycles that have no finite ground terms.
    /// Replaying `rounds` refinement steps verifies the witness without trusting
    /// input-dependent block numbers. `class` is an ID in the indicated input.
    Structure {
        side: Side,
        class: String,
        rounds: usize,
    },
    /// A declaration is absent or has a different signature/kind.
    Declaration { function: String },
    /// This row exists only on `side`, modulo constructor bisimulation.
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
pub fn certificate(left: &Database, right: &Database) -> Result<Option<Certificate>, Error> {
    let result = compare(left, right)?;
    if result.database_equal {
        return Ok(None);
    }
    if !result.terms_equal {
        for side in [Side::Left, Side::Right] {
            let (source, target) = sides(side, left, right);
            if let Some(witness) = ground_certificate(source, target, side) {
                return Ok(Some(witness));
            }
        }
    }
    let mut partition = Partition::new(left, right);
    partition.finish();
    if !result.terms_equal {
        for (side, source, target) in [
            (Side::Left, &partition.left, &partition.right),
            (Side::Right, &partition.right, &partition.left),
        ] {
            let other = partition.root_blocks(target);
            for (class, id) in &source.index {
                if source.roots.contains(id) && !other.contains(&partition.blocks[*id]) {
                    return Ok(Some(Certificate::Structure {
                        side,
                        class: class.clone(),
                        rounds: partition.rounds,
                    }));
                }
            }
        }
    }
    for function in left.functions.keys().chain(right.functions.keys()) {
        if left.functions.get(function) != right.functions.get(function) {
            return Ok(Some(Certificate::Declaration {
                function: function.clone(),
            }));
        }
    }
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
    Err(Error(
        "disequality without a witness (internal error)".into(),
    ))
}

// Extract one representative per productive class with a bottom-up worklist.
// Each row becomes ready at most once; repeated arguments are counted separately.
fn representatives(db: &Database) -> (Vec<Term>, BTreeMap<String, usize>) {
    let mut terms = Vec::new();
    let mut reps = BTreeMap::new();
    let mut ready = VecDeque::new();
    for (id, class) in &db.classes {
        if let Some(value) = &class.literal {
            reps.insert(id.clone(), terms.len());
            terms.push(Term::Literal {
                sort: class.sort.clone(),
                value: value.clone(),
            });
        }
    }
    let mut waiting: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut remaining = vec![0; db.rows.len()];
    for (i, row) in db.rows.iter().enumerate() {
        if db.functions[&row.function].kind != FunctionKind::Constructor {
            continue;
        }
        for input in &row.inputs {
            if !reps.contains_key(input) {
                remaining[i] += 1;
                waiting.entry(input).or_default().push(i);
            }
        }
        if remaining[i] == 0 {
            ready.push_back(i);
        }
    }
    while let Some(i) = ready.pop_front() {
        let row = &db.rows[i];
        if reps.contains_key(&row.output) {
            continue;
        }
        let term = Term::Apply {
            function: row.function.clone(),
            inputs: row.inputs.iter().map(|id| reps[id]).collect(),
        };
        reps.insert(row.output.clone(), terms.len());
        terms.push(term);
        if let Some(rows) = waiting.remove(row.output.as_str()) {
            for i in rows {
                remaining[i] -= 1;
                if remaining[i] == 0 {
                    ready.push_back(i);
                }
            }
        }
    }
    (terms, reps)
}

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
                    .map(|&i| values.get(i).copied().flatten())
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
        let term_id = terms.len();
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
        if values[representative].is_none() {
            let (terms, roots) = compact(terms, &[representative]);
            return Some(Certificate::MissingTerm {
                side,
                terms,
                term: roots[0],
            });
        }
        if value != values[representative] {
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

fn compact(terms: Vec<Term>, roots: &[usize]) -> (Vec<Term>, Vec<usize>) {
    let mut used = vec![false; terms.len()];
    for &root in roots {
        used[root] = true;
    }
    for i in (0..terms.len()).rev() {
        if used[i]
            && let Term::Apply { inputs, .. } = &terms[i]
        {
            for &child in inputs {
                used[child] = true;
            }
        }
    }
    let mut mapping = vec![0; terms.len()];
    let mut result = Vec::new();
    for (i, mut term) in terms.into_iter().enumerate() {
        if !used[i] {
            continue;
        }
        if let Term::Apply { inputs, .. } = &mut term {
            for child in inputs {
                *child = mapping[*child];
            }
        }
        mapping[i] = result.len();
        result.push(term);
    }
    (result, roots.iter().map(|&i| mapping[i]).collect())
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
            Ok(matches!(terms.get(*term), Some(Term::Apply { .. }))
                && a.get(*term).is_some_and(Option::is_some)
                && b.get(*term) == Some(&None))
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
            let ids = (a.get(*first), a.get(*second), b.get(*first), b.get(*second));
            Ok(
                matches!(ids, (Some(Some(x)), Some(Some(y)), Some(Some(u)), Some(Some(v))) if x == y && u != v),
            )
        }
        Certificate::Declaration { function } => {
            Ok(left.functions.get(function) != right.functions.get(function))
        }
        _ => {
            let mut partition = Partition::new(left, right);
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

fn valid_dag(terms: &[Term]) -> bool {
    terms.iter().enumerate().all(|(i, term)| match term {
        Term::Literal { .. } => true,
        Term::Apply { inputs, .. } => inputs.iter().all(|&child| child < i),
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
            return source.index.get(class).is_some_and(|id| {
                source.roots.contains(id)
                    && !partition
                        .root_blocks(target)
                        .contains(&partition.blocks[*id])
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
                .all(|(x, y)| partition.blocks[a.index[x]] == partition.blocks[b.index[y]])
    })
}
