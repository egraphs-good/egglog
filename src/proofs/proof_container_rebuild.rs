//! Container rebuild for the term/proof encoding.
//!
//! Registers a container sort's rebuild primitives from its
//! [`ContainerRebuildSpec`] ([`register_container_rebuild_from_spec`]), and
//! defines the `ContainerRebuild` / `ContainerRebuildProof` primitives that
//! canonicalize a container's elements to their union-find leaders (and, in
//! proof mode, prove the rebuild). The encoder side that *builds* the spec lives
//! in [`super::proof_encoding`].

use crate::exec_state::{Internal, RegistrySealed};
use crate::*;

/// Register a container sort's rebuild primitives from its
/// [`ContainerRebuildSpec`]. Called when a container Sort command carrying an
/// `:internal-container-rebuild` annotation is typechecked, so the primitives
/// exist before the rebuild rules — both during encoding and on re-parse.
pub(crate) fn register_container_rebuild_from_spec(
    eg: &mut EGraph,
    sort_name: &str,
    spec: &ContainerRebuildSpec,
) {
    let Some(container_sort) = eg.get_sort_by_name(sort_name).cloned() else {
        return;
    };
    // Each element eq-sort's UF index table, recovered from proof_state (filled
    // by the element sorts' `:internal-uf` on re-parse) rather than the spec.
    let mut uf_names = HashMap::default();
    collect_element_uf_names(eg, &container_sort, &mut uf_names);

    eg.add_read_primitive(
        ContainerRebuild {
            name: spec.internal_rebuild_prim.clone(),
            container_sort: container_sort.clone(),
            uf_names: uf_names.clone(),
            proof_mode: spec.internal_rebuild_proof_prim.is_some(),
        },
        None,
    );

    if let Some(proof_prim) = &spec.internal_rebuild_proof_prim {
        // Each container's `<CSort>Proof` table (this sort + nested containers),
        // recovered from proof_state (filled by `:internal-proof-func`).
        let mut cproof_names = HashMap::default();
        collect_container_proof_names(eg, &container_sort, &mut cproof_names);
        // The global proof constructors, recovered from proof_state (repopulated
        // from the `Proof` sort's `:internal-proof-names` on re-parse).
        let names = &eg.proof_state.proof_names;
        let congr_name = names.congr_constructor.clone();
        let trans_name = names.eq_trans_constructor.clone();
        let sym_name = names.eq_sym_constructor.clone();
        let container_normalize_name = names.container_normalize_constructor.clone();
        let proof_sort: ArcSort = std::sync::Arc::new(EqSort {
            name: names.proof_datatype.clone(),
        });
        eg.add_full_primitive(
            ContainerRebuildProof {
                name: proof_prim.clone(),
                container_sort,
                proof_sort,
                uf_names,
                cproof_names,
                congr_name,
                trans_name,
                sym_name,
                container_normalize_name,
            },
            None,
        );
    }
}

/// Each transitively-reachable eq-sort element's UF index table, from
/// `proof_state.uf_function` (filled by element sorts' `:internal-uf`).
fn collect_element_uf_names(eg: &EGraph, sort: &ArcSort, out: &mut HashMap<String, String>) {
    for elem in sort.inner_sorts() {
        if elem.is_eq_sort() {
            if let Some(uf) = eg.proof_state.uf_function.get(elem.name()) {
                out.insert(elem.name().to_string(), uf.clone());
            }
        } else if elem.is_eq_container_sort() {
            collect_element_uf_names(eg, &elem, out);
        }
    }
}

/// The `<CSort>Proof` table for `sort` and every nested container sort, from
/// `proof_state.proof_func_parent` (filled by `:internal-proof-func`).
fn collect_container_proof_names(eg: &EGraph, sort: &ArcSort, out: &mut HashMap<String, String>) {
    if let Some(cp) = eg.proof_state.proof_func_parent.get(sort.name()) {
        out.insert(sort.name().to_string(), cp.clone());
    }
    for elem in sort.inner_sorts() {
        if elem.is_eq_container_sort() {
            collect_container_proof_names(eg, &elem, out);
        }
    }
}

/// Re-intern container `value` of `sort` with each contained value remapped
/// through `leaders` (an old-value -> union-find-leader map); the value-level
/// half of the container rebuild performed by the rebuild rules.
fn rebuild_with_leaders(
    cvs: &ContainerValues,
    es: &mut ExecutionState,
    sort: &ArcSort,
    value: Value,
    leaders: &HashMap<Value, Value>,
) -> Value {
    let type_id = sort
        .value_type()
        .expect("container sorts have a value type");
    cvs.rebuild_val_with(type_id, value, es, &|v| {
        leaders.get(&v).copied().unwrap_or(v)
    })
}

/// Recursively canonicalize a container `value` of sort `sort` for the term
/// encoding, returning the rebuilt interned value. Each element is resolved by
/// a uniform per-child rule: an eq-sort element maps to its union-find leader
/// (via `UF_<E>f`; in proof mode the index stores `(pair leader proof)`), a
/// container element is recursively rebuilt, and anything else is unchanged.
fn rebuild_container_value_rec(
    state: &mut ReadState,
    sort: &ArcSort,
    value: Value,
    uf_names: &HashMap<String, String>,
    proof_mode: bool,
) -> Option<Value> {
    let elements = {
        let cvs = state.container_values();
        sort.inner_values(cvs, value)
    };
    let mut leaders: HashMap<Value, Value> = HashMap::default();
    for (esort, eval) in &elements {
        let new = if esort.is_eq_sort() {
            match self_lookup_leader(state, uf_names, esort, *eval, proof_mode)? {
                Some(leader) => leader,
                None => *eval,
            }
        } else if esort.is_eq_container_sort() {
            rebuild_container_value_rec(state, esort, *eval, uf_names, proof_mode)?
        } else {
            *eval
        };
        if new != *eval {
            leaders.insert(*eval, new);
        }
    }
    let cvs = state.container_values();
    let es = state.raw_exec_state();
    Some(rebuild_with_leaders(cvs, es, sort, value, &leaders))
}

/// Look up an eq-sort element's union-find leader, if a `UF_<E>f` row exists.
/// In proof mode the index stores `(pair leader proof)`, so take `pair-first`.
/// Returns `Ok(None)` when there is no UF row (element is its own leader).
fn self_lookup_leader(
    state: &mut ReadState,
    uf_names: &HashMap<String, String>,
    esort: &ArcSort,
    eval: Value,
    proof_mode: bool,
) -> Option<Option<Value>> {
    let Some(uf_name) = uf_names.get(esort.name()) else {
        return Some(None);
    };
    let Some(looked_up) = state
        .lookup(uf_name, eval)
        .expect("union-find index lookup failed")
    else {
        return Some(None);
    };
    let leader = if proof_mode {
        state
            .container_values()
            .get_val::<crate::sort::PairContainer>(looked_up)?
            .first
    } else {
        looked_up
    };
    Some(Some(leader))
}

/// A term-encoding primitive that canonicalizes a container value's elements to
/// their union-find leaders (recursing through nested containers). Registered
/// per container sort by `ensure_container_rebuild` and
/// invoked from the container-column arm of the rebuild rules. It reads the
/// `UF_<E>f` tables, so it is only valid in a `:naive` rule (read-context body).
#[derive(Clone)]
struct ContainerRebuild {
    name: String,
    container_sort: ArcSort,
    /// element-sort name -> `UF_<E>f` table name (all reachable eq-sorts)
    uf_names: HashMap<String, String>,
    /// In proof mode the UF index returns `(pair leader proof)`; take pair-first.
    proof_mode: bool,
}

impl Primitive for ContainerRebuild {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![self.container_sort.clone(), self.container_sort.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl ReadPrim for ContainerRebuild {
    fn apply<'a, 'db>(&self, mut state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        rebuild_container_value_rec(
            &mut state,
            &self.container_sort,
            args[0],
            &self.uf_names,
            self.proof_mode,
        )
    }
}

/// Proof-mode counterpart of [`ContainerRebuild`]: mints a `Congr` chain
/// proving `old_container = rebuilt_container` (recursing through nested
/// containers). Reads `UF_<E>f` (element equality proofs) and `<CSort>Proof`
/// (reflexive bases), mints `Congr`/`Trans`/`Sym` terms, and anchors a
/// reflexive proof on each rebuilt container so it can be rebuilt again later.
/// It is a [`FullPrim`], valid only in a `:naive` rule's action.
#[derive(Clone)]
struct ContainerRebuildProof {
    name: String,
    container_sort: ArcSort,
    proof_sort: ArcSort,
    /// element-sort name -> `UF_<E>f` table name (all reachable eq-sorts)
    uf_names: HashMap<String, String>,
    /// container-sort name -> `<CSort>Proof` table name (all reachable containers)
    cproof_names: HashMap<String, String>,
    /// `Congr` / `Trans` / `Sym` / `ContainerNormalize` proof constructor names
    congr_name: String,
    trans_name: String,
    sym_name: String,
    container_normalize_name: String,
}

impl Primitive for ContainerRebuildProof {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![self.container_sort.clone(), self.proof_sort.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl FullPrim for ContainerRebuildProof {
    fn apply<'a, 'db>(&self, mut state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let (_rebuilt, proof) =
            rebuild_container_proof_rec(&mut state, self, &self.container_sort, args[0])?;
        Some(proof)
    }
}

/// Recursively rebuild `value` (of container sort `sort`) and produce a proof
/// that `value = rebuilt`. Returns `(rebuilt_value, proof)`. Uses the same
/// per-child resolution as [`rebuild_container_value_rec`], additionally
/// folding a `Congr` step for every changed child and recording a reflexive
/// anchor `<CSort>Proof(rebuilt) = Trans(Sym proof, proof)` so the rebuilt
/// value can itself be rebuilt in a later iteration.
fn rebuild_container_proof_rec(
    state: &mut FullState,
    prim: &ContainerRebuildProof,
    sort: &ArcSort,
    value: Value,
) -> Option<(Value, Value)> {
    // Reflexive base proof `value = value`.
    let base = state
        .lookup(prim.cproof_names.get(sort.name())?, value)
        .expect("container proof lookup failed")?;
    let elements = {
        let cvs = state.container_values();
        sort.inner_values(cvs, value)
    };

    let mut leaders: HashMap<Value, Value> = HashMap::default();
    let mut child_proofs: Vec<(usize, Value)> = vec![];
    for (j, (esort, eval)) in elements.iter().enumerate() {
        if esort.is_eq_sort() {
            if let Some(uf_name) = prim.uf_names.get(esort.name())
                && let Some(pair_val) = state
                    .lookup(uf_name, *eval)
                    .expect("union-find index lookup failed")
            {
                let (leader, proof) = {
                    let pc = state
                        .container_values()
                        .get_val::<crate::sort::PairContainer>(pair_val)?;
                    (pc.first, pc.second)
                };
                if leader != *eval {
                    leaders.insert(*eval, leader);
                    child_proofs.push((j, proof));
                }
            }
        } else if esort.is_eq_container_sort() {
            let (rebuilt_child, child_proof) =
                rebuild_container_proof_rec(state, prim, esort, *eval)?;
            if rebuilt_child != *eval {
                leaders.insert(*eval, rebuilt_child);
                child_proofs.push((j, child_proof));
            }
        }
    }

    // Rebuild the value against the collected leaders.
    let rebuilt = {
        let cvs = state.container_values();
        let es = state.raw_exec_state();
        rebuild_with_leaders(cvs, es, sort, value, &leaders)
    };

    // Fold a `Congr` step per changed child onto the reflexive base. This
    // proves `value = raw`, where `raw` is the term with children replaced in
    // place (it may be in non-canonical order, or have duplicate/clobbering
    // entries for collapsing containers).
    let congr_action = state.registry().lookup_table(&prim.congr_name)?.clone();
    let mut current = base;
    for (j, proof) in child_proofs {
        let j_val = state.base_values().get::<i64>(j as i64);
        current =
            congr_action.lookup_or_insert(state.raw_exec_state(), &[current, j_val, proof])?;
    }

    // Bridge the (possibly non-canonical) `raw` term to the canonical `rebuilt`
    // term with the container normalization: `ContainerNormalize(current)` proves
    // `value = normalize(raw)`, which the checker recomputes to match
    // `reconstruct_termdag(rebuilt)`. We mint it unconditionally; for
    // order/arity-preserving containers (Vec/Pair) the normalization is the
    // identity, so it is a no-op the proof simplifier removes.
    let normalize_action = state
        .registry()
        .lookup_table(&prim.container_normalize_name)?
        .clone();
    current = normalize_action.lookup_or_insert(state.raw_exec_state(), &[current])?;

    // Anchor a reflexive proof on the rebuilt value for future rebuilds.
    if rebuilt != value {
        let sym_action = state.registry().lookup_table(&prim.sym_name)?.clone();
        let trans_action = state.registry().lookup_table(&prim.trans_name)?.clone();
        let cproof_action = state
            .registry()
            .lookup_table(prim.cproof_names.get(sort.name())?)?
            .clone();
        // Sym(current): rebuilt = value;  Trans(Sym(current), current): rebuilt = rebuilt.
        let sym_p = sym_action.lookup_or_insert(state.raw_exec_state(), &[current])?;
        let refl = trans_action.lookup_or_insert(state.raw_exec_state(), &[sym_p, current])?;
        cproof_action.insert(state.raw_exec_state(), [rebuilt, refl].into_iter());
    }

    Some((rebuilt, current))
}
