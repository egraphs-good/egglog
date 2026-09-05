use crate::{
    Database, Error, Function, HashMap,
    refine::{Graph, Label},
};
use serde::Serialize;
use smallvec::SmallVec;
use std::collections::BTreeMap;

/// Which observations participate in the canonical representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalMode {
    Terms,
    Database,
}

type NodeSignature = SmallVec<[usize; 3]>;
type Signature = (usize, Vec<NodeSignature>);

#[derive(Serialize)]
struct CanonicalClass {
    sort: usize,
    nodes: Vec<NodeSignature>,
}

#[derive(Serialize)]
struct Encoding<'a> {
    format: &'static str,
    version: u32,
    mode: CanonicalMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    declarations: Option<&'a BTreeMap<String, Function>>,
    sorts: Vec<&'a str>,
    labels: Vec<Label<'a>>,
    roots: Vec<usize>,
    classes: Vec<CanonicalClass>,
}

/// Produce versioned canonical JSON bytes (with a trailing newline).
///
/// For valid inputs in the same mode, byte equality is equivalent to `compare`'s
/// corresponding equality result, including cycles and duplicate behaviors.
/// The output is a bisimulation quotient, not an input `Database`. Canonical
/// ordering requires sorting unique block signatures on every refinement round.
pub fn canonicalize(db: &Database, mode: CanonicalMode) -> Result<Vec<u8>, Error> {
    db.validate()?;
    let mut labels = HashMap::default();
    let graph = Graph::new(db, 0, mode == CanonicalMode::Database, &mut labels);

    // Unobserved classes must not affect numbering, even indirectly through
    // unused sorts/literals. Reachability is iterative to support deep cycles.
    let mut active = vec![false; graph.nodes.len()];
    let mut pending = graph.roots.clone();
    while let Some(id) = pending.pop() {
        if std::mem::replace(&mut active[id], true) {
            continue;
        }
        pending.extend(graph.nodes[id].iter().flat_map(|n| &n.children));
    }
    let mut used_labels = vec![false; labels.len()];
    let mut sorts = Vec::new();
    let ids: Vec<_> = active
        .iter()
        .enumerate()
        .filter_map(|(i, &used)| used.then_some(i))
        .collect();
    for &id in &ids {
        sorts.push(graph.sorts[id]);
        for node in &graph.nodes[id] {
            used_labels[node.symbol] = true;
        }
    }
    sorts.sort_unstable();
    sorts.dedup();
    let mut labels: Vec<_> = labels
        .into_iter()
        .filter(|(_, id)| used_labels[*id])
        .collect();
    labels.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut symbols = vec![0; used_labels.len()];
    for (canonical, (_, original)) in labels.iter().enumerate() {
        symbols[*original] = canonical;
    }
    let labels = labels.into_iter().map(|(label, _)| label).collect();
    let mut blocks = vec![0; graph.nodes.len()];
    for &id in &ids {
        blocks[id] = sorts.binary_search(&graph.sorts[id]).unwrap();
    }
    let mut block_count = sorts.len();

    let final_signatures = loop {
        let mut signatures = HashMap::default();
        let mut next = Vec::with_capacity(ids.len());
        let mut key: Signature = (0, Vec::new());
        for &id in &ids {
            key.0 = blocks[id]; // Split existing blocks; never merge them.
            key.1.clear();
            key.1.extend(graph.nodes[id].iter().map(|node| {
                let mut signature = SmallVec::new();
                signature.push(symbols[node.symbol]);
                signature.extend(node.children.iter().map(|&child| blocks[child]));
                signature
            }));
            key.1.sort_unstable();
            key.1.dedup();
            let block = if let Some(&block) = signatures.get(&key) {
                block
            } else {
                let block = signatures.len();
                signatures.insert((key.0, std::mem::take(&mut key.1)), block);
                block
            };
            next.push(block);
        }
        // Hashing only interns exact signatures. Sorting them gives IDs that
        // depend on observations, never hash iteration or input class order.
        let mut ordered: Vec<_> = signatures.into_iter().collect();
        ordered.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut ranks = vec![0; ordered.len()];
        for (rank, (_, discovered)) in ordered.iter().enumerate() {
            ranks[*discovered] = rank;
        }
        for (&id, discovered) in ids.iter().zip(next) {
            blocks[id] = ranks[discovered];
        }
        if ordered.len() == block_count {
            // Old block is the leading key, so a round without splits preserves
            // IDs. These node signatures already reference the final numbering.
            break ordered;
        }
        block_count = ordered.len();
    };
    let mut class_sorts = vec![0; block_count];
    for &id in &ids {
        class_sorts[blocks[id]] = sorts.binary_search(&graph.sorts[id]).unwrap();
    }
    let classes = final_signatures
        .into_iter()
        .enumerate()
        .map(|(id, ((_, nodes), _))| CanonicalClass {
            sort: class_sorts[id],
            nodes,
        })
        .collect();
    let mut roots: Vec<_> = graph.roots.iter().map(|&id| blocks[id]).collect();
    roots.sort_unstable();
    roots.dedup();
    drop(graph);
    let encoding = Encoding {
        format: "egraph-comparison-canonical",
        version: 1,
        mode,
        declarations: (mode == CanonicalMode::Database).then_some(&db.functions),
        sorts,
        labels,
        roots,
        classes,
    };
    let mut bytes = serde_json::to_vec(&encoding).map_err(|error| Error(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}
