//! Structural relocation and compatibility for canonical declaration records.
//! No native values, executable bodies, or second signature model live here.

use crate::{HashMap, HashSet, proto as pb};

/// The namespace is part of identity; sort and callable names may coincide.
fn declaration_identity(d: &pb::Declaration) -> Result<(bool, &str), String> {
    let identity = match d.kind.as_ref().ok_or("missing declaration kind")? {
        pb::declaration::Kind::EqSort(s) => (true, s.name.as_str()),
        pb::declaration::Kind::HostSortFamily(s) => (true, s.name.as_str()),
        pb::declaration::Kind::Constructor(c) => (false, c.name.as_str()),
        pb::declaration::Kind::Function(f) => (false, f.name.as_str()),
        pb::declaration::Kind::HostPrimitive(p) => (false, p.name.as_str()),
        _ => return Err("declaration reconciliation does not support this kind yet".into()),
    };
    if identity.1.is_empty() {
        return Err("empty declaration name".into());
    }
    Ok(identity)
}

/// Visits every node edge in the admitted saved-code/value shapes. Unknown
/// forms fail, rather than retaining an unrelocated or unchecked reference.
fn node_edges(
    kind: &mut pb::node::Kind,
    mut f: impl FnMut(&mut u32) -> Result<(), String>,
) -> Result<(), String> {
    use pb::{node::Kind as N, primitive_value::Value as V};
    match kind {
        N::Call(c) | N::GetCost(c) => {
            for i in &mut c.args {
                f(i)?;
            }
        }
        N::Union(u) => {
            for i in &mut u.members {
                f(i)?;
            }
        }
        N::Var(_) => (),
        N::PrimitiveValue(p) => match p.value.as_mut().ok_or("missing primitive value")? {
            V::Vec(v) | V::Set(v) | V::Multiset(v) => {
                for i in &mut v.items {
                    f(i)?;
                }
            }
            V::Map(v) => {
                for pair in &mut v.entries {
                    f(pair.key.as_mut().ok_or("missing map key")?)?;
                    f(pair.value.as_mut().ok_or("missing map value")?)?;
                }
            }
            V::Pair(v) => {
                f(v.first.as_mut().ok_or("missing pair first")?)?;
                f(v.second.as_mut().ok_or("missing pair second")?)?;
            }
            V::Maybe(v) => {
                if let Some(i) = &mut v.value {
                    f(i)?;
                }
            }
            V::Custom(v) => {
                for i in &mut v.args {
                    f(i)?;
                }
            }
            V::PartialCall(c) => {
                for i in &mut c.args {
                    f(i)?;
                }
            }
            V::Lambda(_) => return Err("lambda declaration templates are not supported yet".into()),
            _ => (),
        },
    }
    Ok(())
}

fn declaration_refs(
    d: &mut pb::Declaration,
    mut sort: impl FnMut(&mut u32) -> Result<(), String>,
    mut node: impl FnMut(&mut u32) -> Result<(), String>,
) -> Result<(), String> {
    match d.kind.as_mut().ok_or("missing declaration kind")? {
        pb::declaration::Kind::Constructor(c) => {
            for a in &mut c.inputs {
                sort(&mut a.sort)?;
            }
            sort(&mut c.output)?;
            if let Some(i) = &mut c.cost {
                node(i)?;
            }
        }
        pb::declaration::Kind::Function(c) => {
            for a in &mut c.inputs {
                sort(&mut a.sort)?;
            }
            sort(&mut c.output)?;
            if let Some(i) = &mut c.merge {
                node(i)?;
            }
        }
        pb::declaration::Kind::HostPrimitive(p) => {
            let Some(pb::host_primitive::Typing::Signature(s)) = &mut p.typing else {
                return Err("function application bindings are not supported yet".into());
            };
            for a in s.inputs.iter_mut().chain(s.varargs.iter_mut()) {
                sort(&mut a.sort)?;
            }
            sort(s.output.as_mut().ok_or("missing signature result")?)?;
        }
        pb::declaration::Kind::EqSort(_) | pb::declaration::Kind::HostSortFamily(_) => (),
        _ => return Err("unsupported declaration kind".into()),
    }
    if let Some(b) = &mut d.bindings {
        for owner in b
            .python
            .iter_mut()
            .flat_map(|b| &mut b.views)
            .filter_map(|v| v.owner.as_mut())
            .chain(
                b.rust
                    .iter_mut()
                    .flat_map(|b| &mut b.views)
                    .filter_map(|v| v.owner.as_mut()),
            )
        {
            if let Some(pb::binding_owner::Kind::Sort(i)) = &mut owner.kind {
                sort(i)?;
            }
        }
        for p in b
            .python
            .iter_mut()
            .flat_map(|b| &mut b.views)
            .flat_map(|v| &mut v.params)
        {
            if let Some(i) = &mut p.default_expr {
                node(i)?;
            }
        }
        for a in b
            .rust
            .iter_mut()
            .flat_map(|b| &mut b.views)
            .filter_map(|v| v.trait_impl.as_mut())
            .flat_map(|t| &mut t.args)
        {
            sort(a.sort.as_mut().ok_or("missing Rust trait sort")?)?;
        }
    }
    Ok(())
}

fn relocate_span(
    span: &mut Option<pb::Span>,
    source: &pb::Program,
    files: &[u32],
) -> Result<(), String> {
    if let Some(span) = span {
        let file = source
            .files
            .get(span.file as usize)
            .ok_or("span file out of bounds")?;
        if span.start > span.end
            || file.contents.as_ref().is_some_and(|s| {
                !s.is_char_boundary(span.start as usize) || !s.is_char_boundary(span.end as usize)
            })
        {
            return Err("invalid span range".into());
        }
        span.file = files[span.file as usize];
    }
    Ok(())
}

struct NodeImport<'a> {
    source: &'a pb::Program,
    sorts: &'a [u32],
    files: &'a [u32],
    nodes: &'a mut Vec<pb::Node>,
    imported: HashMap<u32, u32>,
    active: HashSet<u32>,
}

impl NodeImport<'_> {
    fn import(&mut self, index: u32) -> Result<u32, String> {
        if let Some(i) = self.imported.get(&index) {
            return Ok(*i);
        }
        if self.active.len() >= 256 || !self.active.insert(index) {
            return Err("cyclic or too-deep declaration template".into());
        }
        let mut n = self
            .source
            .nodes
            .get(index as usize)
            .ok_or("declaration node out of bounds")?
            .clone();
        n.sort_id = *self
            .sorts
            .get(n.sort_id as usize)
            .ok_or("node sort out of bounds")?;
        relocate_span(&mut n.span, self.source, self.files)?;
        node_edges(n.kind.as_mut().ok_or("missing node kind")?, |i| {
            *i = self.import(*i)?;
            Ok(())
        })?;
        self.active.remove(&index);
        let result = self
            .nodes
            .len()
            .try_into()
            .map_err(|_| "too many declaration nodes")?;
        self.nodes.push(n);
        self.imported.insert(index, result);
        Ok(result)
    }
}

#[derive(Default)]
struct NodeComparison {
    pairs: HashSet<(u32, u32)>,
    unions: HashMap<u32, u32>,
    reverse: HashMap<u32, u32>,
}

impl NodeComparison {
    fn equal(&mut self, p: &pb::Program, a: u32, b: u32) -> Result<bool, String> {
        let left = p
            .nodes
            .get(a as usize)
            .ok_or("comparison node out of bounds")?;
        let right = p
            .nodes
            .get(b as usize)
            .ok_or("comparison node out of bounds")?;
        if left.sort_id != right.sort_id {
            return Ok(false);
        }
        if matches!(left.kind, Some(pb::node::Kind::Union(_))) {
            if self.unions.get(&a).is_some_and(|old| *old != b)
                || self.reverse.get(&b).is_some_and(|old| *old != a)
            {
                return Ok(false);
            }
            self.unions.insert(a, b);
            self.reverse.insert(b, a);
        }
        if !self.pairs.insert((a, b)) {
            return Ok(true);
        }
        let mut left = left.kind.clone().ok_or("missing comparison node")?;
        let mut right = right.kind.clone().ok_or("missing comparison node")?;
        let mut lc = vec![];
        let mut rc = vec![];
        node_edges(&mut left, |i| {
            lc.push(*i);
            *i = 0;
            Ok(())
        })?;
        node_edges(&mut right, |i| {
            rc.push(*i);
            *i = 0;
            Ok(())
        })?;
        if left != right || lc.len() != rc.len() {
            return Ok(false);
        }
        for (a, b) in lc.into_iter().zip(rc) {
            if !self.equal(p, a, b)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn compatible_semantics(
    p: &pb::Program,
    a: &pb::Declaration,
    b: &pb::Declaration,
) -> Result<bool, String> {
    let mut roots = [vec![], vec![]];
    let mut declarations = [a.clone(), b.clone()];
    for (d, roots) in declarations.iter_mut().zip(&mut roots) {
        d.bindings = None;
        d.span = None;
        d.doc.clear();
        match d.kind.as_mut().unwrap() {
            pb::declaration::Kind::EqSort(s) => s.bindings = None,
            pb::declaration::Kind::HostSortFamily(s) => s.bindings = None,
            pb::declaration::Kind::Constructor(c) => {
                for a in &mut c.inputs {
                    a.name.clear();
                }
            }
            pb::declaration::Kind::Function(c) => {
                for a in &mut c.inputs {
                    a.name.clear();
                }
            }
            pb::declaration::Kind::HostPrimitive(c) => {
                let Some(pb::host_primitive::Typing::Signature(s)) = &mut c.typing else {
                    return Err("unsupported host signature".into());
                };
                for a in s.inputs.iter_mut().chain(s.varargs.iter_mut()) {
                    a.name.clear();
                }
                for label in &mut s.type_params {
                    label.clear();
                }
            }
            _ => return Err("unsupported declaration comparison".into()),
        }
        declaration_refs(
            d,
            |_| Ok(()),
            |i| {
                roots.push(*i);
                *i = 0;
                Ok(())
            },
        )?;
    }
    if declarations[0] != declarations[1] || roots[0].len() != roots[1].len() {
        return Ok(false);
    }
    let mut comparison = NodeComparison::default();
    for (a, b) in roots[0].iter().zip(&roots[1]) {
        if !comparison.equal(p, *a, *b)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn merge_language<T: Clone + PartialEq>(
    old: &mut Option<T>,
    new: &Option<T>,
) -> Result<(), String> {
    if let Some(new) = new {
        match old {
            Some(old) if old != new => return Err("conflicting language binding".into()),
            None => *old = Some(new.clone()),
            _ => (),
        }
    }
    Ok(())
}

fn merge_bindings(
    p: &pb::Program,
    old: &mut pb::Declaration,
    new: &pb::Declaration,
) -> Result<(), String> {
    let sorts = match (&mut old.kind, &new.kind) {
        (Some(pb::declaration::Kind::EqSort(a)), Some(pb::declaration::Kind::EqSort(b))) => {
            Some((&mut a.bindings, &b.bindings))
        }
        (
            Some(pb::declaration::Kind::HostSortFamily(a)),
            Some(pb::declaration::Kind::HostSortFamily(b)),
        ) => Some((&mut a.bindings, &b.bindings)),
        _ => None,
    };
    if let Some((old, Some(new))) = sorts {
        let old = old.get_or_insert_default();
        merge_language(&mut old.python, &new.python)?;
        merge_language(&mut old.rust, &new.rust)?;
        merge_language(&mut old.egglog, &new.egglog)?;
    }
    if let Some(new) = &new.bindings {
        let old = old.bindings.get_or_insert_default();
        if let (Some(a), Some(b)) = (&old.python, &new.python) {
            let mut a = a.clone();
            let mut b = b.clone();
            let mut comparison = NodeComparison::default();
            let mut ar = vec![];
            let mut br = vec![];
            for (block, roots) in [(&mut a, &mut ar), (&mut b, &mut br)] {
                for p in block.views.iter_mut().flat_map(|v| &mut v.params) {
                    if let Some(i) = &mut p.default_expr {
                        roots.push(*i);
                        *i = 0;
                    }
                }
            }
            if a != b || ar.len() != br.len() {
                return Err("conflicting Python binding".into());
            }
            for (a, b) in ar.into_iter().zip(br) {
                if !comparison.equal(p, a, b)? {
                    return Err("conflicting Python defaults".into());
                }
            }
        } else if old.python.is_none() {
            old.python = new.python.clone();
        }
        merge_language(&mut old.rust, &new.rust)?;
        merge_language(&mut old.egglog, &new.egglog)?;
    }
    Ok(())
}

/// Transactionally imports declaration closures and reconciles each language's
/// first supplied block. The returned flags identify newly installed definitions
/// in source order; compatible resupplies never ask a native engine to redeclare.
pub fn reconcile_declarations(
    source: &pb::Program,
    destination: &mut pb::Program,
) -> Result<Vec<bool>, String> {
    if source.ir_version != 1 {
        return Err("unsupported declaration IR version".into());
    }
    let mut staged = destination.clone();
    staged.ir_version = 1;
    let mut files = vec![];
    for file in &source.files {
        let i = match staged.files.iter().position(|f| f == file) {
            Some(i) => i,
            None => {
                staged.files.push(file.clone());
                staged.files.len() - 1
            }
        };
        files.push(i as u32);
    }
    let mut input_sorts = source.sorts.clone();
    for s in &mut input_sorts {
        relocate_span(&mut s.span, source, &files)?;
    }
    let mut sorts = vec![];
    for i in 0..input_sorts.len() {
        sorts.push(super::import_sort(
            &input_sorts,
            i as u32,
            &mut staged.sorts,
        )?);
    }
    let mut importer = NodeImport {
        source,
        sorts: &sorts,
        files: &files,
        nodes: &mut staged.nodes,
        imported: HashMap::default(),
        active: HashSet::default(),
    };
    let mut incoming = source.declarations.clone();
    for d in &mut incoming {
        declaration_identity(d)?;
        relocate_span(&mut d.span, source, &files)?;
        declaration_refs(
            d,
            |i| {
                *i = *sorts
                    .get(*i as usize)
                    .ok_or("declaration sort out of bounds")?;
                Ok(())
            },
            |i| {
                *i = importer.import(*i)?;
                Ok(())
            },
        )?;
    }
    let mut added = vec![];
    for d in incoming {
        let identity = declaration_identity(&d)?;
        let existing = staged
            .declarations
            .iter()
            .position(|old| declaration_identity(old).ok() == Some(identity));
        if let Some(i) = existing {
            let mut old = staged.declarations[i].clone();
            if !compatible_semantics(&staged, &old, &d)? {
                return Err(format!("conflicting declaration {}", identity.1));
            }
            merge_bindings(&staged, &mut old, &d)?;
            staged.declarations[i] = old;
            added.push(false);
        } else {
            staged.declarations.push(d);
            added.push(true);
        }
    }
    validate_bindings(&staged)?;
    retain_imported_closures(&mut staged, destination)?;
    *destination = staged;
    Ok(added)
}

// Reconciliation can retain old roots instead of the newly imported compatible
// copies. Drop only unreachable NEW arena entries; preexisting indices may also
// be addressed by a caller's in-progress commands and must remain stable.
fn retain_imported_closures(p: &mut pb::Program, previous: &pb::Program) -> Result<(), String> {
    let mut nodes = (0..previous.nodes.len() as u32).collect::<HashSet<_>>();
    let mut sorts = (0..previous.sorts.len() as u32).collect::<HashSet<_>>();
    let mut files = (0..previous.files.len() as u32).collect::<HashSet<_>>();
    let mut pending = nodes.iter().copied().collect::<Vec<_>>();
    for d in &p.declarations {
        if let Some(s) = d.span {
            files.insert(s.file);
        }
        declaration_refs(
            &mut d.clone(),
            |i| {
                sorts.insert(*i);
                Ok(())
            },
            |i| {
                pending.push(*i);
                Ok(())
            },
        )?;
    }
    let mut walked = HashSet::default();
    while let Some(i) = pending.pop() {
        if !walked.insert(i) {
            continue;
        }
        nodes.insert(i);
        let n = &p.nodes[i as usize];
        sorts.insert(n.sort_id);
        if let Some(s) = n.span {
            files.insert(s.file);
        }
        node_edges(
            &mut n.kind.clone().ok_or("missing retained node kind")?,
            |i| {
                pending.push(*i);
                Ok(())
            },
        )?;
    }
    let mut pending = sorts.iter().copied().collect::<Vec<_>>();
    let mut walked = HashSet::default();
    while let Some(i) = pending.pop() {
        if !walked.insert(i) {
            continue;
        }
        sorts.insert(i);
        let s = &p.sorts[i as usize];
        if let Some(s) = s.span {
            files.insert(s.file);
        }
        if let Some(pb::sort::Kind::Family(f)) = &s.kind {
            pending.extend(&f.args);
        }
    }
    let mut node_map = vec![None; p.nodes.len()];
    let mut sort_map = vec![None; p.sorts.len()];
    let mut file_map = vec![None; p.files.len()];
    for (keep, map) in [
        (&nodes, &mut node_map),
        (&sorts, &mut sort_map),
        (&files, &mut file_map),
    ] {
        let mut next = 0;
        for (i, slot) in map.iter_mut().enumerate() {
            if keep.contains(&(i as u32)) {
                *slot = Some(next);
                next += 1;
            }
        }
    }
    p.nodes = std::mem::take(&mut p.nodes)
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| nodes.contains(&(i as u32)).then_some(n))
        .collect();
    p.sorts = std::mem::take(&mut p.sorts)
        .into_iter()
        .enumerate()
        .filter_map(|(i, s)| sorts.contains(&(i as u32)).then_some(s))
        .collect();
    p.files = std::mem::take(&mut p.files)
        .into_iter()
        .enumerate()
        .filter_map(|(i, f)| files.contains(&(i as u32)).then_some(f))
        .collect();
    for n in &mut p.nodes {
        n.sort_id = sort_map[n.sort_id as usize].unwrap();
        if let Some(s) = &mut n.span {
            s.file = file_map[s.file as usize].unwrap();
        }
        node_edges(n.kind.as_mut().unwrap(), |i| {
            *i = node_map[*i as usize].unwrap();
            Ok(())
        })?;
    }
    for s in &mut p.sorts {
        if let Some(span) = &mut s.span {
            span.file = file_map[span.file as usize].unwrap();
        }
        if let Some(pb::sort::Kind::Family(f)) = &mut s.kind {
            for i in &mut f.args {
                *i = sort_map[*i as usize].unwrap();
            }
        }
    }
    for d in &mut p.declarations {
        if let Some(s) = &mut d.span {
            s.file = file_map[s.file as usize].unwrap();
        }
        declaration_refs(
            d,
            |i| {
                *i = sort_map[*i as usize].unwrap();
                Ok(())
            },
            |i| {
                *i = node_map[*i as usize].unwrap();
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn pattern_scope(p: &pb::Program, index: u32, parameters: usize) -> Result<(), String> {
    let mut pending = vec![index];
    let mut seen = HashSet::default();
    while let Some(i) = pending.pop() {
        if !seen.insert(i) {
            continue;
        }
        match p
            .sorts
            .get(i as usize)
            .and_then(|s| s.kind.as_ref())
            .ok_or("binding sort out of bounds")?
        {
            pb::sort::Kind::Var(i) if *i as usize >= parameters => {
                return Err("unbound presentation sort parameter".into());
            }
            pb::sort::Kind::Family(f) => {
                if let Some(d) = p
                    .declarations
                    .iter()
                    .find(|d| declaration_identity(d).ok() == Some((true, f.name.as_str())))
                    && !matches!(&d.kind, Some(pb::declaration::Kind::HostSortFamily(family)) if family.arity as usize == f.args.len())
                {
                    return Err("presentation family arity or kind mismatch".into());
                }
                pending.extend(&f.args);
            }
            pb::sort::Kind::Eq(_) | pb::sort::Kind::Var(_) => (),
            _ => return Err("function sort bindings are not supported yet".into()),
        }
    }
    Ok(())
}

fn owner_sort(p: &pb::Program, owner: &pb::BindingOwner, parameters: usize) -> Result<u32, String> {
    let Some(pb::binding_owner::Kind::Sort(i)) = owner.kind else {
        return Err("invalid or unsupported binding owner".into());
    };
    pattern_scope(p, i, parameters)?;
    Ok(i)
}

/// Checks a default's own closure, not a union of scopes from other uses. Call
/// evaluation and context capability validation belong to its emitted use.
fn closed_default(p: &pb::Program, root: u32) -> Result<(), String> {
    let mut pending = vec![root];
    let mut seen = HashSet::default();
    while let Some(i) = pending.pop() {
        if !seen.insert(i) {
            continue;
        }
        let n = p
            .nodes
            .get(i as usize)
            .ok_or("default node out of bounds")?;
        pattern_scope(p, n.sort_id, 0)?;
        let mut kind = n.kind.clone().ok_or("missing default node kind")?;
        match &kind {
            pb::node::Kind::Var(_) => return Err("free variable in default template".into()),
            pb::node::Kind::Call(call) => {
                let declaration = p
                    .declarations
                    .iter()
                    .find(|d| declaration_identity(d).ok() == Some((false, call.func.as_str())))
                    .ok_or("unknown default callee")?;
                let (inputs, output, tail, parameters) = match declaration.kind.as_ref().unwrap() {
                    pb::declaration::Kind::Constructor(c) => (&c.inputs, c.output, None, 0),
                    pb::declaration::Kind::Function(f) => (&f.inputs, f.output, None, 0),
                    pb::declaration::Kind::HostPrimitive(h) => {
                        let Some(pb::host_primitive::Typing::Signature(s)) = &h.typing else {
                            return Err("unsupported default callee signature".into());
                        };
                        (
                            &s.inputs,
                            s.output.ok_or("missing default callee result")?,
                            s.varargs.as_ref(),
                            s.type_params.len(),
                        )
                    }
                    _ => return Err("unsupported default callee".into()),
                };
                if call.args.len() < inputs.len()
                    || (tail.is_none() && call.args.len() != inputs.len())
                {
                    return Err("default call arity mismatch".into());
                }
                let mut substitution = HashMap::default();
                for (position, arg) in call.args.iter().enumerate() {
                    let expected = inputs.get(position).or(tail).unwrap().sort;
                    let actual = p
                        .nodes
                        .get(*arg as usize)
                        .ok_or("default argument out of bounds")?
                        .sort_id;
                    pattern_scope(p, actual, 0)?;
                    match_pattern(p, expected, actual, &mut substitution)?;
                }
                match_pattern(p, output, n.sort_id, &mut substitution)?;
                if substitution.len() != parameters {
                    return Err("default call does not determine every type parameter".into());
                }
            }
            pb::node::Kind::Union(u) => {
                for member in &u.members {
                    if p.nodes
                        .get(*member as usize)
                        .ok_or("default member out of bounds")?
                        .sort_id
                        != n.sort_id
                    {
                        return Err("default union sort mismatch".into());
                    }
                }
            }
            pb::node::Kind::PrimitiveValue(v) => {
                use pb::primitive_value::Value as V;
                let expected = match v.value.as_ref().ok_or("missing default value")? {
                    V::I64(_) => "i64",
                    V::F64Bits(_) => "f64",
                    V::String(_) => "String",
                    V::Bool(_) => "bool",
                    V::Unit(_) => "Unit",
                    V::Vec(v) => {
                        let Some(pb::sort::Kind::Family(f)) = &p.sorts[n.sort_id as usize].kind
                        else {
                            return Err("default Vec sort mismatch".into());
                        };
                        if f.name != "Vec" || f.args.len() != 1 {
                            return Err("default Vec sort mismatch".into());
                        }
                        for member in &v.items {
                            if p.nodes
                                .get(*member as usize)
                                .ok_or("default Vec item out of bounds")?
                                .sort_id
                                != f.args[0]
                            {
                                return Err("default Vec item sort mismatch".into());
                            }
                        }
                        node_edges(&mut kind, |i| {
                            pending.push(*i);
                            Ok(())
                        })?;
                        continue;
                    }
                    _ => return Err("unsupported default value template".into()),
                };
                if !matches!(&p.sorts[n.sort_id as usize].kind, Some(pb::sort::Kind::Family(f)) if f.name == expected && f.args.is_empty())
                {
                    return Err("default literal sort mismatch".into());
                }
            }
            _ => return Err("unsupported default node kind".into()),
        }
        node_edges(&mut kind, |i| {
            pending.push(*i);
            Ok(())
        })?;
    }
    Ok(())
}

/// Validates presentation independently of core identity. This operates only on
/// generated records and never executes a default, native body, or validator.
fn validate_bindings(p: &pb::Program) -> Result<(), String> {
    for d in &p.declarations {
        let (inputs, output, tail, parameters) =
            match d.kind.as_ref().ok_or("missing declaration kind")? {
                pb::declaration::Kind::EqSort(s) => {
                    validate_sort_bindings(s.bindings.as_ref(), 0)?;
                    if d.bindings.is_some() {
                        return Err("callable bindings on sort".into());
                    }
                    continue;
                }
                pb::declaration::Kind::HostSortFamily(s) => {
                    validate_sort_bindings(s.bindings.as_ref(), s.arity as usize)?;
                    if d.bindings.is_some() {
                        return Err("callable bindings on sort".into());
                    }
                    continue;
                }
                pb::declaration::Kind::Constructor(c) => (&c.inputs, c.output, None, 0),
                pb::declaration::Kind::Function(f) => (&f.inputs, f.output, None, 0),
                pb::declaration::Kind::HostPrimitive(h) => {
                    let Some(pb::host_primitive::Typing::Signature(s)) = &h.typing else {
                        return Err("unsupported binding signature".into());
                    };
                    (
                        &s.inputs,
                        s.output.ok_or("missing signature result")?,
                        s.varargs.as_ref(),
                        s.type_params.len(),
                    )
                }
                _ => return Err("unsupported declaration bindings".into()),
            };
        for i in inputs
            .iter()
            .map(|a| a.sort)
            .chain([output])
            .chain(tail.map(|a| a.sort))
        {
            pattern_scope(p, i, parameters)?;
        }
        let Some(b) = &d.bindings else {
            continue;
        };
        for v in b.python.iter().flat_map(|b| &b.views) {
            let kind =
                pb::PythonCallKind::try_from(v.kind).map_err(|_| "invalid Python call kind")?;
            use pb::PythonCallKind as K;
            if kind == K::Unspecified
                || matches!(kind, K::Function | K::Constant) == v.owner.is_some()
                || matches!(kind, K::Method | K::Property) != v.receiver.is_some()
                || v.path.iter().enumerate().any(|(i, component)| {
                    component.is_empty() && (kind != K::Constant || i + 1 != v.path.len())
                })
                || match kind {
                    K::Initializer => !v.path.is_empty(),
                    K::Function | K::Constant => v.path.is_empty(),
                    _ => v.path.len() != 1,
                }
                || (matches!(kind, K::Property | K::ClassVariable | K::Constant)
                    && !v.params.is_empty())
                || (kind == K::Constant && parameters != 0)
            {
                return Err("invalid Python view form".into());
            }
            let owner = v
                .owner
                .as_ref()
                .map(|o| owner_sort(p, o, parameters))
                .transpose()?;
            if kind == K::Initializer && owner != Some(output) {
                return Err("initializer result differs from owner".into());
            }
            if let Some(r) = v.receiver
                && inputs.get(r as usize).map(|a| a.sort) != owner
            {
                return Err("receiver differs from owner".into());
            }
            let slots = v
                .params
                .iter()
                .map(|p| p.core_input)
                .chain(v.receiver.map(Some))
                .collect::<Vec<_>>();
            validate_slots(&slots, inputs.len(), tail.is_some())?;
            for (position, param) in v.params.iter().enumerate() {
                let slot = param.core_input.unwrap() as usize;
                if param.name.is_empty() {
                    return Err("empty Python parameter".into());
                }
                if slot == inputs.len()
                    && (position + 1 != v.params.len() || param.default_expr.is_some())
                {
                    return Err("invalid Python tail parameter".into());
                }
                if let Some(root) = param.default_expr {
                    closed_default(p, root)?;
                    let expected = inputs[slot].sort;
                    let actual = p.nodes[root as usize].sort_id;
                    let mut substitution = HashMap::default();
                    match_pattern(p, expected, actual, &mut substitution)?;
                }
            }
            if let Some(m) = v.mutates
                && (kind == K::Initializer
                    || inputs.get(m as usize).map(|a| a.sort) != Some(output))
            {
                return Err("invalid mutated input".into());
            }
        }
        for v in b.rust.iter().flat_map(|b| &b.views) {
            if v.path.is_empty()
                || v.path.iter().any(String::is_empty)
                || (v.owner.is_some() && v.path.len() != 1)
                || ((v.receiver.is_some() || v.trait_impl.is_some()) && v.owner.is_none())
                || (v.borrowed_self && v.trait_impl.is_none())
            {
                return Err("invalid Rust view form".into());
            }
            let owner = v
                .owner
                .as_ref()
                .map(|o| owner_sort(p, o, parameters))
                .transpose()?;
            if let Some(r) = v.receiver
                && (r.core_input.is_none()
                    || inputs.get(r.core_input.unwrap() as usize).map(|a| a.sort) != owner)
            {
                return Err("Rust receiver differs from owner".into());
            }
            let slots = v
                .params
                .iter()
                .map(|p| p.core_input)
                .chain(v.receiver.map(|r| r.core_input))
                .collect::<Vec<_>>();
            validate_slots(&slots, inputs.len(), tail.is_some())?;
            for (position, param) in v.params.iter().enumerate() {
                if param.name.is_empty()
                    || (param.core_input == Some(inputs.len() as u32)
                        && position + 1 != v.params.len())
                {
                    return Err("invalid Rust parameter".into());
                }
            }
            if let Some(t) = &v.trait_impl {
                if t.path.is_empty()
                    || t.path.iter().any(String::is_empty)
                    || t.output_associated_type
                        .as_ref()
                        .is_some_and(String::is_empty)
                {
                    return Err("invalid Rust trait path".into());
                }
                for a in &t.args {
                    pattern_scope(p, a.sort.ok_or("missing trait sort")?, parameters)?;
                }
            }
        }
        for v in b.egglog.iter().flat_map(|b| &b.views) {
            if v.symbol.is_empty()
                || (v.datatype_member
                    && !matches!(d.kind, Some(pb::declaration::Kind::Constructor(_))))
            {
                return Err("invalid Egglog view".into());
            }
        }
    }
    Ok(())
}

fn validate_sort_bindings(b: Option<&pb::SortBindings>, arity: usize) -> Result<(), String> {
    if let Some(b) = b {
        for t in b.python.iter().chain(b.rust.iter()) {
            if t.path.iter().any(String::is_empty)
                || t.type_params.iter().any(String::is_empty)
                || (!t.type_params.is_empty() && t.type_params.len() != arity)
            {
                return Err("invalid sort presentation".into());
            }
        }
    }
    Ok(())
}

fn validate_slots(slots: &[Option<u32>], fixed: usize, tail: bool) -> Result<(), String> {
    let count = fixed + usize::from(tail);
    let set = slots.iter().copied().collect::<HashSet<_>>();
    if slots.len() != count
        || set.len() != count
        || slots.iter().any(|s| s.is_none_or(|s| s as usize >= count))
    {
        return Err("presentation must cover each core input exactly once".into());
    }
    Ok(())
}

fn match_pattern(
    p: &pb::Program,
    pattern: u32,
    concrete: u32,
    bindings: &mut HashMap<u32, u32>,
) -> Result<(), String> {
    match (
        &p.sorts[pattern as usize].kind,
        &p.sorts[concrete as usize].kind,
    ) {
        (Some(pb::sort::Kind::Var(v)), _) => {
            if bindings
                .insert(*v, concrete)
                .is_some_and(|old| old != concrete)
            {
                return Err("inconsistent default sort substitution".into());
            }
        }
        (Some(pb::sort::Kind::Family(a)), Some(pb::sort::Kind::Family(b)))
            if a.name == b.name && a.args.len() == b.args.len() =>
        {
            for (a, b) in a.args.iter().zip(&b.args) {
                match_pattern(p, *a, *b, bindings)?;
            }
        }
        (a, b) if a == b => (),
        _ => return Err("default sort differs from parameter".into()),
    }
    Ok(())
}
