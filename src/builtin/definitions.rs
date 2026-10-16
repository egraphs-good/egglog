//! Structural relocation and compatibility for canonical definition records.
//! No native values, executable bodies, or second signature model live here.

use crate::{HashMap, HashSet, proto as pb};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Namespace {
    EqSort,
    HostSortFamily,
    Callable,
}

/// Existing declaration tags select three namespaces; spellings may coincide.
fn declaration_identity(d: &pb::Declaration) -> Result<(Namespace, &str), String> {
    let identity = match d.kind.as_ref().ok_or("missing declaration kind")? {
        pb::declaration::Kind::EqSort(s) => (Namespace::EqSort, s.name.as_str()),
        pb::declaration::Kind::HostSortFamily(s) => (Namespace::HostSortFamily, s.name.as_str()),
        pb::declaration::Kind::Constructor(c) => (Namespace::Callable, c.name.as_str()),
        pb::declaration::Kind::Function(f) => (Namespace::Callable, f.name.as_str()),
        pb::declaration::Kind::Primitive(p) => (Namespace::Callable, p.name.as_str()),
        pb::declaration::Kind::HostPrimitive(p) => (Namespace::Callable, p.name.as_str()),
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
        pb::declaration::Kind::Primitive(p) => {
            for a in &mut p.inputs {
                sort(&mut a.sort)?;
            }
            sort(&mut p.output)?;
            node(p.body.as_mut().ok_or("missing primitive body")?)?;
        }
        pb::declaration::Kind::HostPrimitive(p) => {
            match p.typing.as_mut().ok_or("missing host typing")? {
                pb::host_primitive::Typing::Signature(s) => {
                    for a in s.inputs.iter_mut().chain(s.varargs.iter_mut()) {
                        sort(&mut a.sort)?;
                    }
                    sort(s.output.as_mut().ok_or("missing signature result")?)?;
                }
                pb::host_primitive::Typing::Application(_) => (),
            }
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

// Both declaration defaults and rule bodies live in the same canonical arenas.
fn import_arenas(
    source: &pb::Program,
    destination: &mut pb::Program,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut files = Vec::new();
    for file in &source.files {
        let index = if let Some(index) = destination.files.iter().position(|old| old == file) {
            index
        } else {
            destination.files.push(file.clone());
            destination.files.len() - 1
        };
        files.push(u32::try_from(index).map_err(|_| "too many source files")?);
    }
    let mut input_sorts = source.sorts.clone();
    for sort in &mut input_sorts {
        relocate_span(&mut sort.span, source, &files)?;
    }
    let mut sorts = Vec::new();
    for index in 0..input_sorts.len() {
        sorts.push(super::import_sort(
            &input_sorts,
            index as u32,
            &mut destination.sorts,
        )?);
    }
    Ok((sorts, files))
}

// One reference visitor serves import, structural comparison and retention.
fn rule_refs(
    rule: &mut pb::RuleDecl,
    mut node: impl FnMut(&mut u32) -> Result<(), String>,
    mut span: impl FnMut(&mut Option<pb::Span>) -> Result<(), String>,
) -> Result<(), String> {
    span(&mut rule.span)?;
    match rule.kind.as_mut().ok_or("missing rule kind")? {
        pb::rule_decl::Kind::Rule(rule) => {
            for root in &mut rule.query {
                node(root)?;
            }
            for action in &mut rule.head {
                span(&mut action.span)?;
                let call = match action.kind.as_mut().ok_or("missing action kind")? {
                    pb::action::Kind::Term(root) => {
                        node(root)?;
                        None
                    }
                    pb::action::Kind::Set(set) => {
                        node(set.value.as_mut().ok_or("missing set value")?)?;
                        Some(set.target.as_mut().ok_or("missing set target")?)
                    }
                    pb::action::Kind::SetCost(set) => {
                        node(set.cost.as_mut().ok_or("missing set-cost value")?)?;
                        Some(set.target.as_mut().ok_or("missing set-cost target")?)
                    }
                    pb::action::Kind::Delete(call) | pb::action::Kind::Subsume(call) => Some(call),
                    pb::action::Kind::Panic(_) => None,
                };
                if let Some(call) = call {
                    for arg in &mut call.args {
                        node(arg)?;
                    }
                }
            }
        }
        pb::rule_decl::Kind::Rewrite(rule) => {
            node(&mut rule.lhs)?;
            node(&mut rule.rhs)?;
            for root in &mut rule.conditions {
                node(root)?;
            }
        }
        pb::rule_decl::Kind::Birewrite(rule) => {
            node(&mut rule.lhs)?;
            node(&mut rule.rhs)?;
            for root in &mut rule.conditions {
                node(root)?;
            }
        }
    }
    Ok(())
}

/// Request-local occurrences relocated into the candidate canonical Program.
/// The candidate includes transient occurrences; retain only published named
/// closures after native installation, not all of this request's arenas.
#[derive(Debug)]
pub struct RulesetReconciliation {
    pub rulesets: Vec<usize>,
    pub rules: Vec<usize>,
}

// Canonical references have already resolved names. Return children while
// checking all occurrence edges, including unused leaves.
fn ruleset_children(program: &pb::Program, index: usize) -> Result<Vec<usize>, String> {
    match program
        .rulesets
        .get(index)
        .ok_or("ruleset index out of bounds")?
        .kind
        .as_ref()
        .ok_or("missing ruleset kind")?
    {
        pb::ruleset::Kind::Rules(list) => {
            if list
                .rules
                .iter()
                .any(|i| *i as usize >= program.rules.len())
            {
                return Err("rule index out of bounds".into());
            }
            Ok(Vec::new())
        }
        pb::ruleset::Kind::Combined(combined) => combined
            .rulesets
            .iter()
            .map(|reference| {
                let Some(pb::ruleset_ref::Kind::Index(index)) = reference.kind else {
                    return Err("unresolved canonical ruleset reference".into());
                };
                if index as usize >= program.rulesets.len() {
                    return Err("ruleset index out of bounds".into());
                }
                Ok(index as usize)
            })
            .collect(),
    }
}

fn check_ruleset_cycles(program: &pb::Program) -> Result<(), String> {
    let mut done = HashSet::default();
    for root in 0..program.rulesets.len() {
        let mut active = HashSet::default();
        let mut pending = vec![(root, false)];
        while let Some((index, leaving)) = pending.pop() {
            if leaving {
                active.remove(&index);
                done.insert(index);
                continue;
            }
            if done.contains(&index) {
                continue;
            }
            if !active.insert(index) {
                return Err("cyclic ruleset composition".into());
            }
            pending.push((index, true));
            pending.extend(
                ruleset_children(program, index)?
                    .into_iter()
                    .rev()
                    .map(|child| (child, false)),
            );
        }
    }
    Ok(())
}

struct OccurrenceComparison<'a> {
    program: &'a pb::Program,
    representatives: &'a [usize],
    old_rulesets: usize,
    old_rules: usize,
    local: bool,
    sets: HashMap<usize, usize>,
    reverse_sets: HashMap<usize, usize>,
    rules: HashMap<usize, usize>,
    reverse_rules: HashMap<usize, usize>,
}

impl OccurrenceComparison<'_> {
    fn compare(&mut self, roots: Vec<(usize, usize)>) -> Result<(), String> {
        let mut pending = roots;
        while let Some((left, right)) = pending.pop() {
            let (left, right) = (self.representatives[left], self.representatives[right]);
            // An explicit retained name and a matched local presentation may
            // both reach this occurrence. The retained reference isn't a second
            // local occurrence subject to the local-to-retained injection.
            if left == right && left < self.old_rulesets {
                continue;
            }
            if self.sets.get(&left).is_some_and(|old| *old != right)
                || self
                    .reverse_sets
                    .get(&right)
                    .is_some_and(|old| *old != left)
                || (left < self.old_rulesets && right < self.old_rulesets && left != right)
            {
                return Err("conflicting ruleset sharing".into());
            }
            if self.sets.insert(left, right).is_some() {
                continue;
            }
            self.reverse_sets.insert(right, left);
            match (
                &self.program.rulesets[left].kind,
                &self.program.rulesets[right].kind,
            ) {
                (Some(pb::ruleset::Kind::Rules(a)), Some(pb::ruleset::Kind::Rules(b)))
                    if a.rules.len() == b.rules.len() =>
                {
                    for (&a, &b) in a.rules.iter().zip(&b.rules) {
                        let (a, b) = (a as usize, b as usize);
                        if self.rules.get(&a).is_some_and(|old| *old != b)
                            || self.reverse_rules.get(&b).is_some_and(|old| *old != a)
                            || (a < self.old_rules && b < self.old_rules && a != b)
                            || (self.local && a >= self.old_rules && b >= self.old_rules && a != b)
                        {
                            return Err("conflicting rule sharing".into());
                        }
                        if self.rules.insert(a, b).is_some() {
                            continue;
                        }
                        self.reverse_rules.insert(b, a);
                        let mut pair =
                            [self.program.rules[a].clone(), self.program.rules[b].clone()];
                        let mut roots = [Vec::new(), Vec::new()];
                        for (rule, roots) in pair.iter_mut().zip(&mut roots) {
                            rule.name = None;
                            rule.doc.clear();
                            rule_refs(
                                rule,
                                |root| {
                                    roots.push(*root);
                                    *root = 0;
                                    Ok(())
                                },
                                |span| {
                                    *span = None;
                                    Ok(())
                                },
                            )?;
                        }
                        if pair[0] != pair[1] || roots[0].len() != roots[1].len() {
                            return Err("conflicting rule body or options".into());
                        }
                        let mut nodes = NodeComparison::default();
                        for (&a, &b) in roots[0].iter().zip(&roots[1]) {
                            if !nodes.equal(self.program, a, b)? {
                                return Err("conflicting rule expressions".into());
                            }
                        }
                    }
                }
                (Some(pb::ruleset::Kind::Combined(a)), Some(pb::ruleset::Kind::Combined(b)))
                    if a.rulesets.len() == b.rulesets.len() =>
                {
                    pending.extend(
                        ruleset_children(self.program, left)?
                            .into_iter()
                            .zip(ruleset_children(self.program, right)?),
                    );
                }
                _ => return Err("conflicting ruleset body".into()),
            }
        }
        Ok(())
    }
}

/// Reconcile immutable occurrence graphs on a disposable candidate. Ordered
/// correspondence is anchored by names, never by structural interning. No
/// destination state changes on failure; native compilation is the caller's gate.
pub fn reconcile_rulesets(
    source: &pb::Program,
    destination: &mut pb::Program,
) -> Result<RulesetReconciliation, String> {
    if source.ir_version != 1 {
        return Err("unsupported ruleset IR version".into());
    }
    let mut staged = destination.clone();
    staged.ir_version = 1;
    let old_sets = staged.rulesets.len();
    let old_rules = staged.rules.len();
    let (sorts, files) = import_arenas(source, &mut staged)?;
    let mut importer = NodeImport {
        source,
        sorts: &sorts,
        files: &files,
        nodes: &mut staged.nodes,
        imported: HashMap::default(),
        active: HashSet::default(),
    };
    for rule in &source.rules {
        if !matches!(
            pb::RuleEvalMode::try_from(rule.eval_mode),
            Ok(pb::RuleEvalMode::Seminaive
                | pb::RuleEvalMode::Naive
                | pb::RuleEvalMode::UnsafeSeminaive)
        ) {
            return Err("invalid rule evaluation mode".into());
        }
        if rule.name.as_ref().is_some_and(String::is_empty) {
            return Err("empty rule label".into());
        }
        let mut rule = rule.clone();
        rule_refs(
            &mut rule,
            |root| {
                *root = importer.import(*root)?;
                Ok(())
            },
            |span| relocate_span(span, source, &files),
        )?;
        staged.rules.push(rule);
    }
    let installed_names: HashMap<_, _> = staged
        .rulesets
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.name.as_ref().map(|name| (name.clone(), i)))
        .collect();
    let mut local_names = HashMap::default();
    let mut duplicates = Vec::new();
    for (index, ruleset) in source.rulesets.iter().enumerate() {
        if let Some(name) = &ruleset.name {
            if name.is_empty() && !matches!(ruleset.kind, Some(pb::ruleset::Kind::Rules(_))) {
                return Err("default ruleset must be a rule list".into());
            }
            if let Some(first) = local_names.get(name) {
                duplicates.push((old_sets + index, *first));
            } else {
                local_names.insert(name.clone(), old_sets + index);
            }
        }
    }
    for ruleset in &source.rulesets {
        let mut ruleset = ruleset.clone();
        relocate_span(&mut ruleset.span, source, &files)?;
        match ruleset.kind.as_mut().ok_or("missing ruleset kind")? {
            pb::ruleset::Kind::Rules(list) => {
                for rule in &mut list.rules {
                    if *rule as usize >= source.rules.len() {
                        return Err("rule index out of bounds".into());
                    }
                    *rule =
                        u32::try_from(old_rules + *rule as usize).map_err(|_| "too many rules")?;
                }
            }
            pb::ruleset::Kind::Combined(combined) => {
                for reference in &mut combined.rulesets {
                    let index = match reference.kind.as_ref().ok_or("missing ruleset reference")? {
                        pb::ruleset_ref::Kind::Index(index) => {
                            if *index as usize >= source.rulesets.len() {
                                return Err("ruleset index out of bounds".into());
                            }
                            old_sets + *index as usize
                        }
                        pb::ruleset_ref::Kind::Name(name) => *local_names
                            .get(name)
                            .or_else(|| installed_names.get(name))
                            .ok_or_else(|| format!("unknown ruleset {name}"))?,
                    };
                    reference.kind = Some(pb::ruleset_ref::Kind::Index(
                        u32::try_from(index).map_err(|_| "too many rulesets")?,
                    ));
                }
            }
        }
        staged.rulesets.push(ruleset);
    }
    check_ruleset_cycles(&staged)?;
    let identity: Vec<_> = (0..staged.rulesets.len()).collect();
    let mut representatives = identity.clone();
    let mut anchors = Vec::new();
    for pair in duplicates {
        let mut comparison = OccurrenceComparison {
            program: &staged,
            representatives: &identity,
            old_rulesets: old_sets,
            old_rules,
            local: true,
            sets: HashMap::default(),
            reverse_sets: HashMap::default(),
            rules: HashMap::default(),
            reverse_rules: HashMap::default(),
        };
        comparison.compare(vec![pair])?;
        for (a, b) in comparison.sets {
            if a < old_sets || b < old_sets {
                anchors.push(if a < old_sets { (b, a) } else { (a, b) });
            } else {
                let (a, b) = (representatives[a], representatives[b]);
                let (from, to) = (a.max(b), a.min(b));
                for representative in &mut representatives {
                    if *representative == from {
                        *representative = to;
                    }
                }
            }
        }
    }
    // A locally valid positional bijection can become invalid after transitive
    // identifications: P=[A,B], P=[B,A] must not collapse distinct A/B.
    for root in old_sets..staged.rulesets.len() {
        let mut seen = HashMap::default();
        let mut pending = vec![root];
        while let Some(index) = pending.pop() {
            if let Some(previous) = seen.insert(representatives[index], index) {
                if previous != index
                    && (staged.rulesets[index].name.is_none()
                        || staged.rulesets[previous].name != staged.rulesets[index].name)
                {
                    return Err("duplicate presentations collapse distinct ruleset children".into());
                }
                continue;
            }
            pending.extend(ruleset_children(&staged, index)?);
        }
    }
    for (name, &local) in &local_names {
        if let Some(&installed) = installed_names.get(name) {
            anchors.push((local, installed));
        }
    }
    let mut comparison = OccurrenceComparison {
        program: &staged,
        representatives: &representatives,
        old_rulesets: old_sets,
        old_rules,
        local: false,
        sets: HashMap::default(),
        reverse_sets: HashMap::default(),
        rules: HashMap::default(),
        reverse_rules: HashMap::default(),
    };
    comparison.compare(anchors)?;
    let mut set_map = representatives.clone();
    for (index, representative) in set_map.iter_mut().enumerate() {
        *representative = comparison
            .sets
            .get(&representatives[index])
            .copied()
            .unwrap_or(representatives[index]);
    }
    let mut rule_map: Vec<_> = (0..staged.rules.len())
        .map(|i| comparison.rules.get(&i).copied().unwrap_or(i))
        .collect();
    let mut names = HashMap::default();
    for (index, ruleset) in staged.rulesets.iter().enumerate() {
        if let Some(name) = &ruleset.name
            && names
                .insert(set_map[index], name.clone())
                .is_some_and(|old| old != *name)
        {
            return Err("conflicting ruleset name assignment".into());
        }
    }
    let mut retained_sets = Vec::new();
    let mut compact_sets = HashMap::default();
    for &index in &set_map {
        if compact_sets.contains_key(&index) {
            continue;
        }
        compact_sets.insert(index, retained_sets.len());
        let mut ruleset = staged.rulesets[index].clone();
        ruleset.name = names.get(&index).cloned();
        retained_sets.push(ruleset);
    }
    let mut retained_rules = Vec::new();
    let mut compact_rules = HashMap::default();
    for &index in &rule_map {
        if compact_rules.contains_key(&index) {
            continue;
        }
        compact_rules.insert(index, retained_rules.len());
        retained_rules.push(staged.rules[index].clone());
    }
    for index in &mut set_map {
        *index = compact_sets[index];
    }
    for index in &mut rule_map {
        *index = compact_rules[index];
    }
    for ruleset in &mut retained_sets {
        match ruleset.kind.as_mut().unwrap() {
            pb::ruleset::Kind::Rules(list) => {
                for index in &mut list.rules {
                    *index = rule_map[*index as usize] as u32;
                }
            }
            pb::ruleset::Kind::Combined(combined) => {
                for reference in &mut combined.rulesets {
                    let Some(pb::ruleset_ref::Kind::Index(index)) = &mut reference.kind else {
                        unreachable!()
                    };
                    *index = set_map[*index as usize] as u32;
                }
            }
        }
    }
    staged.rulesets = retained_sets;
    staged.rules = retained_rules;
    check_ruleset_cycles(&staged)?;
    retain_imported_closures(&mut staged, destination)?;
    *destination = staged;
    Ok(RulesetReconciliation {
        rulesets: set_map[old_sets..].to_vec(),
        rules: rule_map[old_rules..].to_vec(),
    })
}

/// Retain exactly the closures of successfully published bindings. Returns the
/// old ruleset/rule indices in their new arena order so derived native handles
/// can be retained with the same relocation. No native state is reconstructed.
pub fn retain_rulesets(
    program: &mut pb::Program,
    published: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let mut staged = program.clone();
    let roots: HashSet<_> = published.iter().copied().collect();
    let mut sets = HashSet::default();
    let mut rules = HashSet::default();
    let mut pending = published.to_vec();
    while let Some(index) = pending.pop() {
        if !sets.insert(index) {
            continue;
        }
        pending.extend(ruleset_children(&staged, index)?);
        if let Some(pb::ruleset::Kind::Rules(list)) = &staged.rulesets[index].kind {
            rules.extend(list.rules.iter().map(|i| *i as usize));
        }
    }
    let retained_sets: Vec<_> = (0..staged.rulesets.len())
        .filter(|i| sets.contains(i))
        .collect();
    let retained_rules: Vec<_> = (0..staged.rules.len())
        .filter(|i| rules.contains(i))
        .collect();
    let set_map: HashMap<_, _> = retained_sets
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new))
        .collect();
    let rule_map: HashMap<_, _> = retained_rules
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new))
        .collect();
    staged.rulesets = retained_sets
        .iter()
        .map(|&index| {
            let mut ruleset = staged.rulesets[index].clone();
            if !roots.contains(&index) {
                ruleset.name = None;
            }
            match ruleset.kind.as_mut().unwrap() {
                pb::ruleset::Kind::Rules(list) => {
                    for index in &mut list.rules {
                        *index = rule_map[&(*index as usize)] as u32;
                    }
                }
                pb::ruleset::Kind::Combined(combined) => {
                    for reference in &mut combined.rulesets {
                        let Some(pb::ruleset_ref::Kind::Index(index)) = &mut reference.kind else {
                            unreachable!()
                        };
                        *index = set_map[&(*index as usize)] as u32;
                    }
                }
            }
            ruleset
        })
        .collect();
    staged.rules = retained_rules
        .iter()
        .map(|&i| staged.rules[i].clone())
        .collect();
    retain_imported_closures(&mut staged, &pb::Program::default())?;
    *program = staged;
    Ok((retained_sets, retained_rules))
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
        d.doc = None;
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
            pb::declaration::Kind::Primitive(c) => {
                for a in &mut c.inputs {
                    a.name.clear();
                }
            }
            pb::declaration::Kind::HostPrimitive(c) => {
                match c.typing.as_mut().ok_or("missing host typing")? {
                    pb::host_primitive::Typing::Signature(s) => {
                        for a in s.inputs.iter_mut().chain(s.varargs.iter_mut()) {
                            a.name.clear();
                        }
                        for label in &mut s.type_params {
                            label.clear();
                        }
                    }
                    pb::host_primitive::Typing::Application(_) => (),
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
    let (sorts, files) = import_arenas(source, &mut staged)?;
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
    sorts.extend(
        p.sorts
            .iter()
            .enumerate()
            .filter_map(|(index, sort)| (!sort.egglog_aliases.is_empty()).then_some(index as u32)),
    );
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
    for rule in &p.rules {
        rule_refs(
            &mut rule.clone(),
            |root| {
                pending.push(*root);
                Ok(())
            },
            |span| {
                if let Some(span) = span {
                    files.insert(span.file);
                }
                Ok(())
            },
        )?;
    }
    for ruleset in &p.rulesets {
        if let Some(span) = ruleset.span {
            files.insert(span.file);
        }
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
        } else if let Some(pb::sort::Kind::Func(f)) = &s.kind {
            pending.extend(&f.params);
            pending.push(f.result);
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
        } else if let Some(pb::sort::Kind::Func(f)) = &mut s.kind {
            for i in f.params.iter_mut().chain([&mut f.result]) {
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
    for rule in &mut p.rules {
        rule_refs(
            rule,
            |root| {
                *root = node_map[*root as usize].unwrap();
                Ok(())
            },
            |span| {
                if let Some(span) = span {
                    span.file = file_map[span.file as usize].unwrap();
                }
                Ok(())
            },
        )?;
    }
    for ruleset in &mut p.rulesets {
        if let Some(span) = &mut ruleset.span {
            span.file = file_map[span.file as usize].unwrap();
        }
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
                if let Some(d) = p.declarations.iter().find(|d| {
                    declaration_identity(d).ok()
                        == Some((Namespace::HostSortFamily, f.name.as_str()))
                }) && !matches!(&d.kind, Some(pb::declaration::Kind::HostSortFamily(family)) if family.arity as usize == f.args.len())
                {
                    return Err("presentation family arity mismatch".into());
                }
                pending.extend(&f.args);
            }
            pb::sort::Kind::Eq(_) | pb::sort::Kind::Var(_) => (),
            pb::sort::Kind::Func(f) => {
                pending.extend(&f.params);
                pending.push(f.result);
            }
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
                    .find(|d| {
                        declaration_identity(d).ok()
                            == Some((Namespace::Callable, call.func.as_str()))
                    })
                    .ok_or("unknown default callee")?;
                let (inputs, output, tail, parameters) = match declaration.kind.as_ref().unwrap() {
                    pb::declaration::Kind::Constructor(c) => (&c.inputs, c.output, &[][..], 0),
                    pb::declaration::Kind::Function(f) => (&f.inputs, f.output, &[][..], 0),
                    pb::declaration::Kind::Primitive(p) => (&p.inputs, p.output, &[][..], 0),
                    pb::declaration::Kind::HostPrimitive(h) => {
                        let Some(pb::host_primitive::Typing::Signature(s)) = &h.typing else {
                            return Err("unsupported default callee signature".into());
                        };
                        (
                            &s.inputs,
                            s.output.ok_or("missing default callee result")?,
                            s.varargs.as_slice(),
                            s.type_params.len(),
                        )
                    }
                    _ => return Err("unsupported default callee".into()),
                };
                if call.args.len() < inputs.len()
                    || if tail.is_empty() {
                        call.args.len() != inputs.len()
                    } else {
                        !(call.args.len() - inputs.len()).is_multiple_of(tail.len())
                    }
                {
                    return Err("default call arity mismatch".into());
                }
                let mut substitution = HashMap::default();
                for (position, arg) in call.args.iter().enumerate() {
                    let expected = inputs
                        .get(position)
                        .unwrap_or_else(|| &tail[(position - inputs.len()) % tail.len()])
                        .sort;
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
                if let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind
                    && let Some(pb::host_primitive::Typing::Signature(signature)) =
                        &primitive.typing
                    && signature.equality_type_params.iter().any(|i| {
                        substitution.get(i).is_none_or(|sort| {
                            !matches!(p.sorts[*sort as usize].kind, Some(pb::sort::Kind::Eq(_)))
                        })
                    })
                {
                    return Err("default type parameter requires an equality sort".into());
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
                    V::Map(v) => {
                        let Some(pb::sort::Kind::Family(f)) = &p.sorts[n.sort_id as usize].kind
                        else {
                            return Err("default Map sort mismatch".into());
                        };
                        if f.name != "Map" || f.args.len() != 2 {
                            return Err("default Map sort mismatch".into());
                        }
                        for entry in &v.entries {
                            for (member, expected) in
                                [entry.key, entry.value].into_iter().zip(&f.args)
                            {
                                if p.nodes
                                    .get(member.ok_or("missing default Map child")? as usize)
                                    .ok_or("default Map child out of bounds")?
                                    .sort_id
                                    != *expected
                                {
                                    return Err("default Map child sort mismatch".into());
                                }
                            }
                        }
                        node_edges(&mut kind, |i| {
                            pending.push(*i);
                            Ok(())
                        })?;
                        continue;
                    }
                    V::Pair(v) => {
                        let Some(pb::sort::Kind::Family(f)) = &p.sorts[n.sort_id as usize].kind
                        else {
                            return Err("default Pair sort mismatch".into());
                        };
                        if f.name != "Pair" || f.args.len() != 2 {
                            return Err("default Pair sort mismatch".into());
                        }
                        for (member, expected) in [v.first, v.second].into_iter().zip(&f.args) {
                            if p.nodes
                                .get(member.ok_or("missing default Pair child")? as usize)
                                .ok_or("default Pair child out of bounds")?
                                .sort_id
                                != *expected
                            {
                                return Err("default Pair child sort mismatch".into());
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
        if matches!(&d.kind, Some(pb::declaration::Kind::HostPrimitive(h)) if matches!(h.typing, Some(pb::host_primitive::Typing::Application(_))))
        {
            if let Some(b) = &d.bindings {
                if b.rust.as_ref().is_some_and(|b| !b.views.is_empty()) {
                    return Err("function application Rust views are not supported yet".into());
                }
                if let Some(python) = &b.python {
                    if python.views.len() > 1 {
                        return Err("function application accepts at most one Python view".into());
                    }
                    for v in &python.views {
                        if v.kind != pb::PythonCallKind::Method as i32
                            || v.path != ["__call__"]
                            || !matches!(
                                v.owner.as_ref().and_then(|o| o.kind.as_ref()),
                                Some(pb::binding_owner::Kind::Function(_))
                            )
                            || v.receiver != Some(0)
                            || !matches!(v.params.as_slice(), [p] if p.core_input == Some(1)
                                && p.name == "args" && p.default_expr.is_none())
                            || v.mutates.is_some()
                            || v.tail_repetitions.is_some()
                        {
                            return Err("invalid function application Python view".into());
                        }
                    }
                }
                for view in b.egglog.iter().flat_map(|b| &b.views) {
                    if view.symbol.is_empty() || view.datatype_member {
                        return Err("invalid Egglog view".into());
                    }
                }
            }
            continue;
        }
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
                pb::declaration::Kind::Constructor(c) => (&c.inputs, c.output, &[][..], 0),
                pb::declaration::Kind::Function(f) => (&f.inputs, f.output, &[][..], 0),
                pb::declaration::Kind::Primitive(primitive) => {
                    let root = primitive.body.ok_or("missing primitive body")?;
                    if p.nodes
                        .get(root as usize)
                        .ok_or("primitive body out of bounds")?
                        .sort_id
                        != primitive.output
                    {
                        return Err("primitive body result sort mismatch".into());
                    }
                    // Scope belongs to this declaration, even when another
                    // declaration shares the same imported body nodes.
                    let mut pending = vec![root];
                    let mut seen = HashSet::default();
                    while let Some(i) = pending.pop() {
                        if !seen.insert(i) {
                            continue;
                        }
                        let node = p
                            .nodes
                            .get(i as usize)
                            .ok_or("primitive body node out of bounds")?;
                        pattern_scope(p, node.sort_id, 0)?;
                        let mut kind = node.kind.clone().ok_or("missing primitive body node")?;
                        if let pb::node::Kind::Var(name) = &kind {
                            let index = name
                                .strip_prefix('_')
                                .and_then(|s| s.parse::<usize>().ok())
                                .filter(|index| name == &format!("_{index}"));
                            if index
                                .and_then(|index| primitive.inputs.get(index))
                                .is_none_or(|arg| arg.sort != node.sort_id)
                            {
                                return Err(format!(
                                    "unbound or inconsistently typed primitive variable {name}"
                                ));
                            }
                        }
                        node_edges(&mut kind, |child| {
                            pending.push(*child);
                            Ok(())
                        })?;
                    }
                    (&primitive.inputs, primitive.output, &[][..], 0)
                }
                pb::declaration::Kind::HostPrimitive(h) => {
                    let Some(pb::host_primitive::Typing::Signature(s)) = &h.typing else {
                        return Err("unsupported binding signature".into());
                    };
                    if s.equality_type_params.windows(2).any(|w| w[0] >= w[1])
                        || s.equality_type_params
                            .iter()
                            .any(|i| *i as usize >= s.type_params.len())
                    {
                        return Err("invalid equality type parameter indices".into());
                    }
                    (
                        &s.inputs,
                        s.output.ok_or("missing signature result")?,
                        s.varargs.as_slice(),
                        s.type_params.len(),
                    )
                }
                _ => return Err("unsupported declaration bindings".into()),
            };
        for i in inputs
            .iter()
            .map(|a| a.sort)
            .chain([output])
            .chain(tail.iter().map(|a| a.sort))
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
                || (v.tail_repetitions.is_some() && !matches!(kind, K::Function | K::Method))
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
            let slots = v
                .params
                .iter()
                .map(|p| p.core_input)
                .chain(v.receiver.map(Some))
                .collect::<Vec<_>>();
            let effective = view_inputs(inputs, tail, v.tail_repetitions, v.receiver, &slots)?;
            if v.tail_repetitions.is_none() && effective.len() > inputs.len() && kind != K::Method {
                return Err("only a method can consume a tail receiver".into());
            }
            if kind == K::Initializer && owner != Some(output) {
                return Err("initializer result differs from owner".into());
            }
            if let Some(r) = v.receiver
                && effective.get(r as usize).copied() != owner
            {
                return Err("receiver differs from owner".into());
            }
            for (position, param) in v.params.iter().enumerate() {
                let slot = param.core_input.unwrap() as usize;
                if param.name.is_empty() {
                    return Err("empty Python parameter".into());
                }
                if v.tail_repetitions.is_none()
                    && slot == effective.len()
                    && (position + 1 != v.params.len() || param.default_expr.is_some())
                {
                    return Err("invalid Python tail parameter".into());
                }
                if let Some(root) = param.default_expr {
                    if slot >= inputs.len() {
                        return Err("expanded tail inputs cannot have defaults".into());
                    }
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
            let slots = v
                .params
                .iter()
                .map(|p| p.core_input)
                .chain(v.receiver.map(|r| r.core_input))
                .collect::<Vec<_>>();
            let effective = view_inputs(
                inputs,
                tail,
                v.tail_repetitions,
                v.receiver.and_then(|r| r.core_input),
                &slots,
            )?;
            if let Some(r) = v.receiver
                && (r.core_input.is_none()
                    || effective.get(r.core_input.unwrap() as usize).copied() != owner)
            {
                return Err("Rust receiver differs from owner".into());
            }
            for (position, param) in v.params.iter().enumerate() {
                if param.name.is_empty()
                    || (v.tail_repetitions.is_none()
                        && param.core_input == Some(effective.len() as u32)
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

/// Validate a view's complete mapping before allocating expanded tail inputs.
/// Sorts are interned structurally by reconciliation; their indices compare
/// invariantly, including nested family arguments and bound variables.
fn view_inputs(
    inputs: &[pb::Arg],
    tail: &[pb::Arg],
    repetitions: Option<u32>,
    receiver: Option<u32>,
    slots: &[Option<u32>],
) -> Result<Vec<u32>, String> {
    let fixed = u32::try_from(inputs.len()).map_err(|_| "too many fixed inputs")?;
    let consumes_tail = repetitions.is_none() && receiver == Some(fixed);
    if consumes_tail && tail.len() != 1 {
        return Err("tail receiver requires a homogeneous variadic signature".into());
    }
    let count = match repetitions {
        Some(repetitions) => {
            if tail.is_empty() {
                return Err("tail repetitions require a variadic host signature".into());
            }
            let width = u32::try_from(tail.len()).map_err(|_| "too many tail inputs")?;
            fixed
                .checked_add(
                    repetitions
                        .checked_mul(width)
                        .ok_or("view input count overflow")?,
                )
                .ok_or("view input count overflow")?
        }
        None => fixed
            .checked_add(u32::from(!tail.is_empty()) + u32::from(consumes_tail))
            .ok_or("view input count overflow")?,
    };
    if slots.len() != count as usize {
        return Err("presentation must cover each core input exactly once".into());
    }
    let set = slots.iter().copied().collect::<HashSet<_>>();
    if set.len() != slots.len() || slots.iter().any(|s| s.is_none_or(|s| s >= count)) {
        return Err("presentation must cover each core input exactly once".into());
    }
    // The consuming receiver adds one fixed operand, followed by a logical tail.
    let effective = if repetitions.is_some() {
        count as usize
    } else {
        inputs.len() + usize::from(consumes_tail)
    };
    Ok(inputs
        .iter()
        .chain(tail.iter().cycle())
        .take(effective)
        .map(|a| a.sort)
        .collect())
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
        (Some(pb::sort::Kind::Func(a)), Some(pb::sort::Kind::Func(b)))
            if a.params.len() == b.params.len() =>
        {
            for (a, b) in a
                .params
                .iter()
                .chain([&a.result])
                .zip(b.params.iter().chain([&b.result]))
            {
                match_pattern(p, *a, *b, bindings)?;
            }
        }
        (a, b) if a == b => (),
        _ => return Err("default sort differs from parameter".into()),
    }
    Ok(())
}
