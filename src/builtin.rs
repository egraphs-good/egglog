//! Engine-owned protobuf builtin definitions and derived native constraints.
//!
//! This migration covers scalar signatures, Pair and selected Vec/Map operations, not a
//! complete generic catalog. Opaque registrations and family gaps remain visible in
//! the inventory. Export never calls an implementation or a proof validator.

use crate::{constraint::TypeConstraint, proto as pb, typechecking::PrimitiveWithId, *};

pub mod definitions;

/// Migrated definitions plus explicit remaining registration gaps. This is not
/// a complete engine snapshot and must not be used as a successful Freeze.
pub struct BuiltinCatalog {
    /// Canonical protobuf definitions with remapped, shared sort indices.
    pub definitions: pb::Program,
    /// One entry per native registration lacking a definition; aliases can repeat.
    pub undescribed_primitives: Vec<String>,
    /// Presort factories whose generic definitions have not migrated yet.
    pub undescribed_families: Vec<String>,
    /// Reserved family operations without definitions, including before any
    /// concrete family instance has been registered.
    pub undescribed_family_primitives: Vec<String>,
    /// Installed native non-equality sorts not represented by a family entry.
    /// Concrete container aliases remain here until structural mapping exists.
    pub undescribed_sorts: Vec<String>,
}

/// An owned definition plus derived native bindings. These bindings preserve
/// nominal source sorts; they contain no independently authored type scheme.
#[derive(Clone)]
pub struct BuiltinInstance {
    pub(crate) definition: Arc<pb::Program>,
    pub(crate) dispatch_name: String,
    binding: InstanceBinding,
}

#[derive(Clone)]
enum InstanceBinding {
    Owner(Vec<NativeSortBinding>),
    // Roots are signature occurrences, never descriptor-arena slots: equal
    // logical patterns can have different actual nominal input/result sorts.
    Exact {
        sorts: Vec<ArcSort>,
        fixed: usize,
        tail: usize,
    },
    Application(Arc<FunctionSort>),
    // The implementation retains its original, possibly growing native domain.
    Open,
}

#[derive(Clone)]
enum NativeSortBinding {
    Exact(ArcSort),
    // The pattern stays in definition.sorts. This is only its representation ABI.
    Foreign(std::any::TypeId),
}

impl BuiltinInstance {
    /// Checks the implementation's representation ABI against its descriptor.
    /// Raw Values use None; typed casts must agree in both type and storage kind.
    /// Inputs describe each fixed position followed by each tail-pattern position;
    /// variadic implementations must supply converters for the complete pattern.
    pub fn check_abi(
        &self,
        inputs: &[Option<(std::any::TypeId, bool)>],
        output: Option<(std::any::TypeId, bool)>,
        varargs: bool,
    ) -> Result<(), String> {
        let declaration = self
            .definition
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostPrimitive(p)) => Some(p),
                _ => None,
            })
            .unwrap();
        let exact = match &self.binding {
            InstanceBinding::Open => {
                return Err("open native domain has no exact implementation ABI".into());
            }
            InstanceBinding::Exact { sorts, fixed, tail } => {
                if varargs == (*tail == 0) || inputs.len() != *fixed + *tail {
                    return Err(
                        "builtin implementation arity differs from canonical signature".into(),
                    );
                }
                Some(sorts.clone())
            }
            InstanceBinding::Application(function) => Some(
                std::iter::once(function.clone() as ArcSort)
                    .chain(function.inputs().iter().cloned())
                    .chain([function.output()])
                    .collect::<Vec<_>>(),
            ),
            InstanceBinding::Owner(_) => None,
        };
        if let Some(sorts) = exact {
            if (matches!(self.binding, InstanceBinding::Application(_)) && varargs)
                || inputs.len() + 1 != sorts.len()
            {
                return Err(
                    "builtin implementation arity differs from canonical application".into(),
                );
            }
            for (sort, cast) in sorts.iter().zip(inputs.iter().copied().chain([output])) {
                if let Some((ty, container)) = cast
                    && (sort.value_type() != Some(ty) || sort.is_container_sort() != container)
                {
                    return Err(format!(
                        "builtin implementation conversion disagrees with {}",
                        sort.name()
                    ));
                }
            }
            return Ok(());
        }
        let Some(pb::host_primitive::Typing::Signature(signature)) = &declaration.typing else {
            unreachable!()
        };
        if varargs == signature.varargs.is_empty()
            || inputs.len() != signature.inputs.len() + signature.varargs.len()
        {
            return Err("builtin implementation arity differs from canonical signature".into());
        }
        let indices = signature
            .inputs
            .iter()
            .map(|a| a.sort)
            .chain(signature.varargs.iter().map(|a| a.sort))
            .chain(signature.output);
        let InstanceBinding::Owner(sorts) = &self.binding else {
            unreachable!()
        };
        for (index, cast) in indices.zip(inputs.iter().copied().chain([output])) {
            if let Some((ty, container)) = cast {
                let (expected, is_container, name) = match &sorts[index as usize] {
                    NativeSortBinding::Exact(sort) => {
                        (sort.value_type(), sort.is_container_sort(), sort.name())
                    }
                    NativeSortBinding::Foreign(carrier) => {
                        let Some(pb::sort::Kind::Family(family)) =
                            &self.definition.sorts[index as usize].kind
                        else {
                            unreachable!()
                        };
                        (Some(*carrier), true, family.name.as_str())
                    }
                };
                if expected != Some(ty) || is_container != container {
                    return Err(format!(
                        "builtin implementation conversion disagrees with {name}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Derives fixed/tail native assignments from the canonical pattern arena.
    pub fn type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        if let InstanceBinding::Application(function) = &self.binding {
            let sorts = std::iter::once(function.clone() as ArcSort)
                .chain(function.inputs().iter().cloned())
                .chain([function.output()])
                .collect();
            return constraint::SimpleTypeConstraint::new("unstable-app", sorts, span.clone())
                .into_box();
        }
        assert!(
            !matches!(self.binding, InstanceBinding::Open),
            "open builtin implementations must retain their native type constraints"
        );
        let declaration = self
            .definition
            .declarations
            .iter()
            .find(|d| matches!(d.kind, Some(pb::declaration::Kind::HostPrimitive(_))))
            .unwrap();
        let Some(pb::declaration::Kind::HostPrimitive(p)) = &declaration.kind else {
            unreachable!()
        };
        let Some(pb::host_primitive::Typing::Signature(signature)) = &p.typing else {
            unreachable!()
        };
        let (fixed, tail, output) = match &self.binding {
            InstanceBinding::Exact { sorts, fixed, tail } => (
                (0..*fixed as u32).collect(),
                (*fixed as u32..(*fixed + *tail) as u32).collect(),
                (sorts.len() - 1) as u32,
            ),
            InstanceBinding::Owner(_) => (
                signature.inputs.iter().map(|a| a.sort).collect(),
                signature.varargs.iter().map(|a| a.sort).collect(),
                signature.output.unwrap(),
            ),
            _ => unreachable!(),
        };
        Box::new(InstanceConstraint {
            instance: self.clone(),
            fixed,
            tail,
            output,
            round_tail_up: false,
            name: declaration
                .bindings
                .as_ref()
                .and_then(|b| b.egglog.as_ref())
                .and_then(|b| b.views.first())
                .map_or(&p.name, |v| &v.symbol)
                .clone(),
            span: span.clone(),
        })
    }

    /// Derive the legacy flat facade from one fixed Vec-of-Pair argument.
    /// The original definition/dispatch identity is never changed or exported
    /// as an alternating-tail callable.
    pub(crate) fn vec_pair_facade_constraints(
        &self,
        name: &str,
        span: &Span,
    ) -> Box<dyn TypeConstraint> {
        let [key, value, _, _, output] =
            self.vec_pair_facade_pattern().expect("fixed Map signature");
        Box::new(InstanceConstraint {
            instance: self.clone(),
            fixed: vec![],
            tail: vec![key, value],
            output,
            name: name.into(),
            span: span.clone(),
            round_tail_up: true,
        })
    }

    /// Validate the complete canonical recipe before deriving either native
    /// flat constraints or source-call packing. Indices stay in the descriptor.
    pub(crate) fn vec_pair_facade_pattern(&self) -> Result<[u32; 5], String> {
        let primitive = self
            .definition
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostPrimitive(p)) => Some(p),
                _ => None,
            })
            .ok_or("missing Map definition")?;
        let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            return Err("expected fixed Map signature".into());
        };
        let [input] = signature.inputs.as_slice() else {
            return Err("expected one Vec argument".into());
        };
        if signature.type_params.len() != 2 || !signature.varargs.is_empty() {
            return Err("expected two Map type parameters and no tail".into());
        }
        let Some(pb::sort::Kind::Family(vector)) = &self.definition.sorts[input.sort as usize].kind
        else {
            return Err("expected Vec".into());
        };
        if vector.name != "Vec" {
            return Err("expected Vec".into());
        }
        let [element] = vector.args.as_slice() else {
            return Err("expected one Vec type argument".into());
        };
        let Some(pb::sort::Kind::Family(pair)) = &self.definition.sorts[*element as usize].kind
        else {
            return Err("expected Pair".into());
        };
        let output = signature.output.ok_or("missing Map output")?;
        let Some(pb::sort::Kind::Family(map)) = &self.definition.sorts[output as usize].kind else {
            return Err("expected Map result".into());
        };
        if pair.name != "Pair"
            || pair.args.len() != 2
            || map.name != "Map"
            || map.args != pair.args
            || pair.args.iter().enumerate().any(|(i, arg)| {
                self.definition.sorts[*arg as usize].kind != Some(pb::sort::Kind::Var(i as u32))
            })
        {
            return Err("expected Vec<Pair<K,V>> -> Map<K,V>".into());
        }
        Ok([pair.args[0], pair.args[1], *element, input.sort, output])
    }

    /// Match nested provenance against the canonical pattern while retaining
    /// exact nominal bindings for the owner and its type parameters.
    fn matches_sort(&self, index: u32, native: &ArcSort, types: &TypeInfo) -> bool {
        let InstanceBinding::Owner(sorts) = &self.binding else {
            return false;
        };
        match &sorts[index as usize] {
            NativeSortBinding::Exact(sort) => sort.name() == native.name(),
            NativeSortBinding::Foreign(carrier) => {
                let Some(pb::sort::Kind::Family(pattern)) =
                    &self.definition.sorts[index as usize].kind
                else {
                    unreachable!()
                };
                native.is_container_sort()
                    && native.value_type() == Some(*carrier)
                    && types
                        .builtin_sorts
                        .get(native.name())
                        .is_some_and(|(family, args)| {
                            *family == pattern.name
                                && args.len() == pattern.args.len()
                                && pattern
                                    .args
                                    .iter()
                                    .zip(args)
                                    .all(|(index, arg)| self.matches_sort(*index, arg, types))
                        })
            }
        }
    }

    /// Only open implementations of this very registry-owned descriptor may
    /// share a private dispatch group. Exact instances remain unique.
    pub(crate) fn can_share_dispatch(&self, existing: &[PrimitiveWithId]) -> bool {
        matches!(self.binding, InstanceBinding::Open)
            && existing.iter().all(|registered| {
                registered
                    .primitive
                    .builtin_instance()
                    .is_some_and(|other| {
                        matches!(other.binding, InstanceBinding::Open)
                            && Arc::ptr_eq(&self.definition, &other.definition)
                    })
            })
    }
}

struct InstanceConstraint {
    instance: BuiltinInstance,
    fixed: Vec<u32>,
    tail: Vec<u32>,
    output: u32,
    // Flat compatibility constructors historically report the next full group.
    round_tail_up: bool,
    name: String,
    span: Span,
}

/// Expand only complete tail groups, retaining the legacy flat-facade rounding
/// rule. Native arguments include the result; the caller reports any mismatch.
fn expand_signature<T: Clone>(
    fixed: &[T],
    tail: &[T],
    output: &T,
    argument_count: usize,
    round_tail_up: bool,
) -> Vec<T> {
    let mut sorts = fixed.to_vec();
    if !tail.is_empty() {
        let inputs = argument_count.saturating_sub(fixed.len() + 1);
        let groups = if round_tail_up {
            inputs.div_ceil(tail.len())
        } else {
            inputs / tail.len()
        };
        sorts.extend(tail.iter().cloned().cycle().take(groups * tail.len()));
    }
    sorts.push(output.clone());
    sorts
}

impl TypeConstraint for InstanceConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn constraint::Constraint<AtomTerm, ArcSort>>> {
        let sorts = expand_signature(
            &self.fixed,
            &self.tail,
            &self.output,
            arguments.len(),
            self.round_tail_up,
        );
        if arguments.len() != sorts.len() {
            return vec![constraint::impossible(
                constraint::ImpossibleConstraint::ArityMismatch {
                    atom: core::Atom {
                        span: self.span.clone(),
                        head: self.name.clone(),
                        args: arguments.to_vec(),
                    },
                    expected: sorts.len(),
                },
            )];
        }
        arguments
            .iter()
            .zip(sorts)
            .map(|(argument, index)| match &self.instance.binding {
                InstanceBinding::Exact { sorts, .. } => {
                    constraint::assign(argument.clone(), sorts[index as usize].clone())
                }
                InstanceBinding::Owner(sorts) => match &sorts[index as usize] {
                    NativeSortBinding::Exact(sort) => {
                        constraint::assign(argument.clone(), sort.clone())
                    }
                    NativeSortBinding::Foreign(_) => constraint::xor(
                        typeinfo
                            .sorts
                            .values()
                            .filter(|sort| self.instance.matches_sort(index, sort, typeinfo))
                            .map(|sort| constraint::assign(argument.clone(), sort.clone()))
                            .collect(),
                    ),
                },
                _ => unreachable!("only owner/exact bindings derive instance constraints"),
            })
            .collect()
    }
}

/// Imports a pattern recursively, remapping child indices and rejecting cycles.
/// Open variables are retained; binder/closure validation is a separate use check.
pub fn import_sort(
    source: &[pb::Sort],
    index: u32,
    destination: &mut Vec<pb::Sort>,
) -> Result<u32, String> {
    fn visit(
        source: &[pb::Sort],
        index: u32,
        destination: &mut Vec<pb::Sort>,
        active: &mut HashSet<u32>,
        cache: &mut HashMap<u32, u32>,
    ) -> Result<u32, String> {
        if let Some(index) = cache.get(&index) {
            return Ok(*index);
        }
        if active.len() >= 256 || !active.insert(index) {
            return Err("cyclic or too-deep sort pattern".into());
        }
        let mut sort = source
            .get(index as usize)
            .ok_or("sort index out of bounds")?
            .clone();
        match sort.kind.as_mut().ok_or("missing sort kind")? {
            pb::sort::Kind::Family(family) => {
                if family.name.is_empty() {
                    return Err("empty host family name".into());
                }
                for child in &mut family.args {
                    *child = visit(source, *child, destination, active, cache)?;
                }
            }
            pb::sort::Kind::Func(function) => {
                for child in &mut function.params {
                    *child = visit(source, *child, destination, active, cache)?;
                }
                function.result = visit(source, function.result, destination, active, cache)?;
            }
            pb::sort::Kind::Eq(name) if !name.is_empty() => (),
            pb::sort::Kind::Var(_) => (),
            _ => return Err("unsupported builtin sort pattern".into()),
        }
        active.remove(&index);
        if let Some(existing) = destination
            .iter()
            .position(|existing| existing.kind == sort.kind)
        {
            for alias in sort.egglog_aliases {
                if !destination[existing].egglog_aliases.contains(&alias) {
                    destination[existing].egglog_aliases.push(alias);
                }
            }
            cache.insert(index, existing as u32);
            return Ok(existing as u32);
        }
        let imported = u32::try_from(destination.len()).map_err(|_| "too many sorts")?;
        destination.push(sort);
        cache.insert(index, imported);
        Ok(imported)
    }
    visit(
        source,
        index,
        destination,
        &mut HashSet::default(),
        &mut HashMap::default(),
    )
}

pub(crate) fn normalized_signature(
    program: &pb::Program,
    primitive: &pb::HostPrimitive,
    sorts: &mut Vec<pb::Sort>,
) -> Result<pb::host_primitive::Typing, String> {
    if primitive.name.is_empty() {
        return Err("builtin definition key must not be empty".into());
    }
    let signature = match &primitive.typing {
        Some(pb::host_primitive::Typing::Signature(signature)) => signature,
        Some(typing @ pb::host_primitive::Typing::Application(_)) => return Ok(typing.clone()),
        None => return Err("missing builtin typing form".into()),
    };
    let mut signature = signature.clone();
    if signature
        .equality_type_params
        .windows(2)
        .any(|w| w[0] >= w[1])
        || signature
            .equality_type_params
            .iter()
            .any(|i| *i as usize >= signature.type_params.len())
    {
        return Err("invalid equality type parameter indices".into());
    }
    for label in &mut signature.type_params {
        label.clear();
    }
    let output = signature
        .output
        .as_mut()
        .ok_or("missing builtin result sort")?;
    *output = import_sort(&program.sorts, *output, sorts)?;
    for arg in signature
        .inputs
        .iter_mut()
        .chain(signature.varargs.iter_mut())
    {
        arg.name.clear();
        arg.sort = import_sort(&program.sorts, arg.sort, sorts)?;
    }
    let mut pending: Vec<_> = signature
        .inputs
        .iter()
        .map(|a| a.sort)
        .chain(signature.varargs.iter().map(|a| a.sort))
        .chain(signature.output)
        .collect();
    let mut seen = HashSet::default();
    while let Some(index) = pending.pop() {
        if !seen.insert(index) {
            continue;
        }
        match sorts[index as usize].kind.as_ref().unwrap() {
            pb::sort::Kind::Var(index) if (*index as usize) >= signature.type_params.len() => {
                return Err("unbound builtin type parameter".into());
            }
            pb::sort::Kind::Family(family) => pending.extend(&family.args),
            pb::sort::Kind::Func(function) => {
                pending.extend(&function.params);
                pending.push(function.result);
            }
            _ => (),
        }
    }
    Ok(pb::host_primitive::Typing::Signature(signature))
}

/// Builds a closed signature from the macro's existing type annotations. The
/// returned protobuf is the signature; native constraints are derived from it.
/// Symbolic family signatures will use the same arenas and GenericSignature,
/// but cannot be inferred from an already instantiated opaque container sort.
pub fn closed_signature(
    key: &str,
    alias: &str,
    inputs: &[(&str, ArcSort)],
    varargs: &[(&str, ArcSort)],
    output: ArcSort,
) -> Result<pb::Program, String> {
    if key.is_empty() || alias.is_empty() || key == alias {
        return Err("a builtin needs distinct nonempty definition key and source alias".into());
    }
    let mut program = pb::Program {
        ir_version: 1,
        ..Default::default()
    };
    let mut indices = vec![];
    for sort in inputs
        .iter()
        .chain(varargs)
        .map(|(_, sort)| sort)
        .chain(std::iter::once(&output))
    {
        if sort.is_eq_sort() || sort.is_container_sort() || sort.value_type().is_none() {
            return Err(format!(
                "{} requires an explicit symbolic sort definition",
                sort.name()
            ));
        }
        let kind = pb::sort::Kind::Family(pb::HostSort {
            name: sort.name().into(),
            args: vec![],
        });
        let index = match program
            .sorts
            .iter()
            .position(|sort| sort.kind.as_ref() == Some(&kind))
        {
            Some(index) => index,
            None => {
                program.sorts.push(pb::Sort {
                    kind: Some(kind),
                    ..Default::default()
                });
                program.declarations.push(pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                        name: sort.name().into(),
                        arity: 0,
                        ..Default::default()
                    })),
                    ..Default::default()
                });
                program.sorts.len() - 1
            }
        };
        indices.push(u32::try_from(index).map_err(|_| "too many signature sorts")?);
    }
    program.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
            name: key.into(),
            typing: Some(pb::host_primitive::Typing::Signature(
                pb::GenericSignature {
                    inputs: inputs
                        .iter()
                        .zip(&indices)
                        .map(|((name, _), sort)| pb::Arg {
                            name: (*name).into(),
                            sort: *sort,
                        })
                        .collect(),
                    varargs: varargs
                        .iter()
                        .zip(&indices[inputs.len()..])
                        .map(|((name, _), sort)| pb::Arg {
                            name: (*name).into(),
                            sort: *sort,
                        })
                        .collect(),
                    output: indices.last().copied(),
                    ..Default::default()
                },
            )),
        })),
        bindings: Some(pb::CallableBindings {
            egglog: Some(pb::EgglogBindings {
                views: vec![pb::EgglogCallable {
                    symbol: alias.into(),
                    datatype_member: false,
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    });
    Ok(program)
}

/// Derive one shared sort parameter from fixed homogeneous `#` macro inputs.
/// The closed result retains the scalar representation checks and provenance.
pub fn homogeneous_signature(
    key: &str,
    alias: &str,
    inputs: &[&str],
    output: ArcSort,
) -> Result<pb::Program, String> {
    if inputs.is_empty() {
        return Err("a homogeneous signature needs at least one input".into());
    }
    let mut program = closed_signature(key, alias, &[], &[], output)?;
    let variable = program.sorts.len() as u32;
    program.sorts.push(pb::Sort {
        kind: Some(pb::sort::Kind::Var(0)),
        ..Default::default()
    });
    let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
        &mut program.declarations.last_mut().unwrap().kind
    else {
        unreachable!()
    };
    let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing else {
        unreachable!()
    };
    signature.type_params.push("T".into());
    signature.inputs = inputs
        .iter()
        .map(|name| pb::Arg {
            name: (*name).into(),
            sort: variable,
        })
        .collect();
    Ok(program)
}

/// Validates and normalizes the currently supported closed signature into native
/// sort names: fixed inputs, tail pattern, then result. This does not execute
/// or inspect a body.
pub fn signature_sorts(
    program: &pb::Program,
    primitive: &pb::HostPrimitive,
) -> Result<Vec<String>, String> {
    let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
        return Err("function-application builtin typing is not implemented yet".into());
    };
    if primitive.name.is_empty() || !signature.type_params.is_empty() {
        return Err("builtin typing currently requires a named closed signature".into());
    }
    signature.inputs.iter().chain(&signature.varargs).map(|arg| arg.sort).chain(std::iter::once(signature.output.ok_or("missing builtin result sort")?)).map(|index| {
        let sort = program.sorts.get(index as usize).ok_or("builtin signature sort index out of bounds")?;
        match &sort.kind {
            Some(pb::sort::Kind::Family(family)) if family.args.is_empty() && !family.name.is_empty() => Ok(family.name.clone()),
            _ => Err("builtin signature requires a symbolic-family compiler beyond this scalar slice".into()),
        }
    }).collect()
}

pub(crate) fn definition_key(program: &pb::Program) -> Result<&str, String> {
    if program.ir_version != 1
        || !program.commands.is_empty()
        || !program.rules.is_empty()
        || !program.rulesets.is_empty()
    {
        return Err("builtin definition must contain only declaration templates".into());
    }
    let mut validated = vec![];
    for index in 0..program.sorts.len() {
        import_sort(&program.sorts, index as u32, &mut validated)?;
    }
    let mut key = None;
    for declaration in &program.declarations {
        match &declaration.kind {
            Some(pb::declaration::Kind::HostPrimitive(primitive)) if key.is_none() => {
                normalized_signature(program, primitive, &mut vec![])?;
                key = Some(primitive.name.as_str());
            }
            Some(pb::declaration::Kind::HostSortFamily(family)) if !family.name.is_empty() => {}
            _ => {
                return Err(
                    "expected exactly one builtin definition and family descriptors".into(),
                );
            }
        }
    }
    definitions::reconcile_declarations(program, &mut pb::Program::default())?;
    key.ok_or_else(|| "missing builtin definition".into())
}

/// Imports one registration's canonical protobuf into another arena, preserving
/// identity and rejecting conflicting declarations. No implementations are run
/// or installed. Returned keys name definitions, never native overload aliases.
pub fn import_definition(
    definition: &pb::Program,
    destination: &mut pb::Program,
) -> Result<String, String> {
    let key = definition_key(definition)?.to_owned();
    definitions::reconcile_declarations(definition, destination)?;
    Ok(key)
}

/// Build closed scalar member/operator views from the canonical signature.
/// Specialized tail slots have ordinary required parameters in these views.
pub fn scalar_member_bindings(
    signature: &pb::GenericSignature,
    python: &str,
    rust: &str,
    params: &[&str],
    rust_trait: Option<&str>,
    tail_repetitions: Option<u32>,
) -> pb::CallableBindings {
    assert!(signature.type_params.is_empty());
    let count = match tail_repetitions {
        Some(repetitions) => {
            assert!(!signature.varargs.is_empty());
            signature
                .inputs
                .len()
                .checked_add(
                    signature
                        .varargs
                        .len()
                        .checked_mul(repetitions as usize)
                        .unwrap(),
                )
                .unwrap()
        }
        None => {
            assert!(signature.varargs.is_empty());
            signature.inputs.len()
        }
    };
    assert_eq!(count, params.len() + 1);
    let inputs = signature
        .inputs
        .iter()
        .chain(signature.varargs.iter().cycle())
        .take(count)
        .collect::<Vec<_>>();
    let owner = Some(pb::BindingOwner {
        kind: Some(pb::binding_owner::Kind::Sort(inputs[0].sort)),
    });
    pb::CallableBindings {
        python: Some(pb::PythonBindings {
            views: vec![pb::PythonCallable {
                kind: pb::PythonCallKind::Method.into(),
                path: vec![python.into()],
                owner,
                receiver: Some(0),
                params: params
                    .iter()
                    .enumerate()
                    .map(|(index, name)| pb::PythonParameter {
                        core_input: Some(index as u32 + 1),
                        name: (*name).into(),
                        default_expr: None,
                    })
                    .collect(),
                tail_repetitions,
                ..Default::default()
            }],
        }),
        rust: Some(pb::RustBindings {
            views: vec![pb::RustCallable {
                path: vec![rust.into()],
                owner,
                receiver: Some(pb::RustReceiver {
                    core_input: Some(0),
                    borrowed: false,
                }),
                params: params
                    .iter()
                    .enumerate()
                    .map(|(index, name)| pb::RustParameter {
                        core_input: Some(index as u32 + 1),
                        name: if rust_trait.is_some() { "rhs" } else { name }.to_string(),
                        borrowed: false,
                    })
                    .collect(),
                trait_impl: rust_trait.map(|name| pb::RustTrait {
                    path: vec!["core".into(), "ops".into(), name.into()],
                    args: inputs[1..]
                        .iter()
                        .map(|arg| pb::RustType {
                            sort: Some(arg.sort),
                            borrowed: false,
                        })
                        .collect(),
                    output_associated_type: Some("Output".into()),
                }),
                tail_repetitions,
                ..Default::default()
            }],
        }),
        ..Default::default()
    }
}

struct SignatureConstraint {
    fixed: Vec<String>,
    tail: Vec<String>,
    output: String,
    name: String,
    span: Span,
}

struct HomogeneousConstraint {
    input_count: usize,
    output: String,
    name: String,
    span: Span,
}

impl TypeConstraint for HomogeneousConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn constraint::Constraint<AtomTerm, ArcSort>>> {
        constraint::AllEqualTypeConstraint::new(&self.name, self.span.clone())
            .with_exact_length(self.input_count + 1)
            .with_output_sort(
                typeinfo
                    .get_sort_by_name(&self.output)
                    .unwrap_or_else(|| panic!("registered builtin sort {} is absent", self.output))
                    .clone(),
            )
            .get(arguments, typeinfo)
    }
}

impl TypeConstraint for SignatureConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn constraint::Constraint<AtomTerm, ArcSort>>> {
        let sorts = expand_signature(
            &self.fixed,
            &self.tail,
            &self.output,
            arguments.len(),
            false,
        )
        .iter()
        .map(|name| {
            typeinfo
                .get_sort_by_name(name)
                .unwrap_or_else(|| panic!("registered builtin sort {name} is absent"))
                .clone()
        })
        .collect();
        SimpleTypeConstraint::new(&self.name, sorts, self.span.clone()).get(arguments, typeinfo)
    }
}

/// Compiles a trusted registration's canonical signature into native inference
/// constraints. Untrusted submitted descriptors are compatibility assertions;
/// they never replace this registered signature or its implementation.
pub fn type_constraints(definition: &pb::Program, span: &Span) -> Box<dyn TypeConstraint> {
    let key = definition_key(definition).expect("invalid native builtin definition");
    let (primitive, diagnostic_name) = definition
        .declarations
        .iter()
        .find_map(|declaration| match &declaration.kind {
            Some(pb::declaration::Kind::HostPrimitive(primitive)) => Some((
                primitive,
                declaration
                    .bindings
                    .as_ref()
                    .and_then(|bindings| bindings.egglog.as_ref())
                    .and_then(|bindings| bindings.views.first())
                    .map_or(key, |view| view.symbol.as_str()),
            )),
            _ => None,
        })
        .unwrap();
    let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
        unreachable!()
    };
    if !signature.type_params.is_empty() {
        assert_eq!(
            signature.type_params.len(),
            1,
            "unsupported generic signature"
        );
        assert!(
            !signature.inputs.is_empty() && signature.varargs.is_empty(),
            "generic constraints require fixed nonempty inputs"
        );
        assert!(signature.inputs.iter().all(|arg| definition.sorts[arg.sort as usize].kind == Some(pb::sort::Kind::Var(0))), "generic constraints require homogeneous inputs");
        let Some(pb::sort::Kind::Family(output)) =
            &definition.sorts[signature.output.unwrap() as usize].kind
        else {
            panic!("generic constraints require a closed scalar result");
        };
        assert!(
            output.args.is_empty(),
            "generic constraints require a closed scalar result"
        );
        return Box::new(HomogeneousConstraint {
            input_count: signature.inputs.len(),
            output: output.name.clone(),
            name: diagnostic_name.into(),
            span: span.clone(),
        });
    }
    let mut fixed = signature_sorts(definition, primitive).unwrap();
    let output = fixed.pop().unwrap();
    let tail = fixed.split_off(signature.inputs.len());
    Box::new(SignatureConstraint {
        fixed,
        tail,
        output,
        name: diagnostic_name.into(),
        span: span.clone(),
    })
}

impl TypeInfo {
    /// Export a named function target from its complete exact native signature.
    /// Source projection supplies specialized capture sorts before erasure;
    /// result encoding additionally checks the retained runtime primitive IDs.
    /// Neither path invokes the target or recovers a signature from syntax.
    pub fn export_function_target(
        &self,
        name: &str,
        inputs: &[ArcSort],
        output: &ArcSort,
        resolved: Option<&ResolvedFunctionId>,
        destination: &mut pb::Program,
    ) -> Result<String, String> {
        if let Some(function) = self.get_func_type(name) {
            if !function
                .input
                .iter()
                .chain([&function.output])
                .map(|s| s.name())
                .eq(inputs.iter().chain([output]).map(|s| s.name()))
            {
                return Err("function target signature mismatch".into());
            }
            if matches!(resolved, Some(ResolvedFunctionId::Primitive { .. })) {
                return Err("function target runtime kind mismatch".into());
            }
            return Ok(name.into());
        }
        let signature = inputs.iter().chain([output]).cloned().collect::<Vec<_>>();
        let candidates = self
            .get_prims(name)
            .ok_or("unknown function target")?
            .iter()
            .filter(|primitive| primitive.accept(&signature, self))
            .collect::<Vec<_>>();
        let mut target = None;
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let mut available = candidates
                .iter()
                .filter(|p| p.context_ids[context].is_some());
            let Some(primitive) = available.next() else {
                if let Some(ResolvedFunctionId::Primitive { context_ids }) = resolved
                    && context_ids[context].is_some()
                {
                    return Err("function target runtime context mismatch".into());
                }
                continue;
            };
            if available.next().is_some() {
                return Err(format!("ambiguous function target {name} in {context:?}"));
            }
            if let Some(resolved) = resolved {
                let ResolvedFunctionId::Primitive { context_ids } = resolved else {
                    return Err("function target runtime kind mismatch".into());
                };
                if context_ids[context] != primitive.context_ids[context] {
                    return Err("function target runtime identity mismatch".into());
                }
            }
            let key =
                if let Some(key) = primitive.primitive.export_program_definition(destination)? {
                    key
                } else {
                    let definition = primitive.primitive.builtin_definition().ok_or_else(|| {
                        format!("function target {name} has no catalog definition yet")
                    })?;
                    let key = import_definition(definition, destination)?;
                    self.export_builtin_definition(&key, destination)?;
                    key
                };
            if target.as_ref().is_some_and(|old| old != &key) {
                return Err("function target contexts have different canonical identities".into());
            }
            target = Some(key);
        }
        target.ok_or_else(|| format!("unresolved function target {name}"))
    }

    /// Installs a family's generated declaration at its native registration site.
    /// Later fragments may assert the same arity without repeating its views.
    pub fn register_builtin_family(&mut self, family: pb::HostSortFamily) -> Result<(), String> {
        // Presorts and concrete sorts have distinct native namespaces. A
        // same-named equality sort must not mask an available family factory.
        if !self.mksorts.contains_key(&family.name) {
            if let Some(sort) = self.sorts.get(&family.name) {
                if sort.is_eq_sort() || sort.is_container_sort() || family.arity != 0 {
                    return Err(
                        "builtin family disagrees with native sort kind or nullary arity".into(),
                    );
                }
            } else {
                return Err("builtin family has no native registration".into());
            }
        }
        definitions::reconcile_declarations(
            &pb::Program {
                ir_version: 1,
                declarations: vec![pb::Declaration {
                    kind: Some(pb::declaration::Kind::HostSortFamily(family)),
                    ..Default::default()
                }],
                ..Default::default()
            },
            &mut self.builtin_families,
        )?;
        Ok(())
    }

    /// Exports the exact definition and the authoritative family presentation
    /// supplies needed by its sort arena. No source overload search is involved.
    pub fn export_builtin_definition(
        &self,
        key: &str,
        destination: &mut pb::Program,
    ) -> Result<(), String> {
        let definition = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin definition")?;
        let mut staged = destination.clone();
        let families = pb::Program {
            ir_version: 1,
            declarations: self
                .builtin_families
                .declarations
                .iter()
                .filter(|d| {
                    let Some(pb::declaration::Kind::HostSortFamily(f)) = &d.kind else {
                        return false;
                    };
                    definition.sorts.iter().any(
                        |s| matches!(&s.kind, Some(pb::sort::Kind::Family(s)) if s.name == f.name),
                    )
                })
                .cloned()
                .collect(),
            ..Default::default()
        };
        definitions::reconcile_declarations(&families, &mut staged)?;
        import_definition(definition, &mut staged)?;
        *destination = staged;
        Ok(())
    }

    /// Registers the structural provenance of one native nominal family instance.
    pub fn register_builtin_sort(
        &mut self,
        family: &'static str,
        sort: ArcSort,
        parameters: Vec<ArcSort>,
    ) -> Result<(), String> {
        let expected = self
            .builtin_definitions
            .values()
            .flat_map(|p| &p.declarations)
            .find_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostSortFamily(f)) if f.name == family => Some(f.arity),
                _ => None,
            })
            .ok_or_else(|| format!("undescribed family {family}"))?;
        if expected as usize != parameters.len() || self.builtin_sorts.contains_key(sort.name()) {
            return Err(format!(
                "invalid or duplicate family instance {}",
                sort.name()
            ));
        }
        if let Some(carrier) = self
            .mksorts
            .get(family)
            .and_then(|provider| provider.container_type)
            && (!sort.is_container_sort() || sort.value_type() != Some(carrier))
        {
            return Err(format!(
                "family instance {} disagrees with its provider representation",
                sort.name()
            ));
        }
        self.builtin_sorts
            .insert(sort.name().into(), (family, parameters));
        Ok(())
    }
}

impl EGraph {
    /// Recover a function value's canonical target from its retained runtime
    /// identity and per-value capture sorts, not from its extracted text or the
    /// function sort's aggregate history of captures.
    pub fn export_function_value_target(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<String, String> {
        let function = sort
            .clone()
            .as_arc_any()
            .downcast::<FunctionSort>()
            .map_err(|_| "expected native function sort")?;
        let value = self
            .value_to_container::<FunctionContainer>(value)
            .ok_or("missing native function value")?;
        let inputs = value
            .1
            .iter()
            .map(|(sort, _)| sort)
            .chain(function.inputs())
            .cloned()
            .collect::<Vec<_>>();
        let types = self
            .proof_state
            .original_typechecking
            .as_ref()
            .map_or(&self.type_info, |original| &original.type_info);
        types.export_function_target(
            &value.2,
            &inputs,
            &function.output(),
            Some(&value.0),
            &mut pb::Program::default(),
        )
    }

    /// Select an annotated builtin call in the captured authoring environment.
    ///
    /// `native_sort_names` lists the argument sorts followed by the result sort.
    /// Names are bound to this compiler's registered sorts before the existing
    /// canonical definition checks and exact-instance selection. This also works
    /// after nonexecuting resolution, when the physical graph has not installed
    /// the declarations or has proof-instrumented signatures. No body or validator
    /// is invoked, and no mutable type environment escapes.
    pub fn resolve_builtin_call(
        &self,
        program: &pb::Program,
        name: &str,
        arguments: &[u32],
        output: u32,
        native_sort_names: &[&str],
    ) -> Result<String, String> {
        let types = self
            .proof_state
            .original_typechecking
            .as_ref()
            .map_or(&self.type_info, |original| &original.type_info);
        let sorts = native_sort_names
            .iter()
            .map(|name| {
                types
                    .get_sort_by_name(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown authoring sort {name}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        types.resolve_builtin(program, name, arguments, output, &sorts)
    }

    /// Exports structural sorts from registered family provenance, never from a
    /// native alias spelling or Rust value TypeId. Equality sorts stay nominal.
    pub fn export_sort(
        &self,
        sort: &ArcSort,
        destination: &mut Vec<pb::Sort>,
    ) -> Result<u32, String> {
        let kind = if sort.is_eq_sort() {
            pb::sort::Kind::Eq(sort.name().into())
        } else if let Ok(function) = sort.clone().as_arc_any().downcast::<FunctionSort>() {
            pb::sort::Kind::Func(pb::FuncSort {
                params: function
                    .inputs()
                    .iter()
                    .map(|sort| self.export_sort(sort, destination))
                    .collect::<Result<_, _>>()?,
                result: self.export_sort(&function.output(), destination)?,
            })
        } else if let Some((family, parameters)) = self.type_info.builtin_sorts.get(sort.name()) {
            pb::sort::Kind::Family(pb::HostSort {
                name: (*family).into(),
                args: parameters
                    .iter()
                    .map(|parameter| self.export_sort(parameter, destination))
                    .collect::<Result<_, _>>()?,
            })
        } else if !sort.is_container_sort() && sort.value_type().is_some() {
            pb::sort::Kind::Family(pb::HostSort {
                name: sort.name().into(),
                args: vec![],
            })
        } else {
            return Err(format!(
                "sort {} lacks structural catalog provenance",
                sort.name()
            ));
        };
        if let Some(index) = destination
            .iter()
            .position(|sort| sort.kind.as_ref() == Some(&kind))
        {
            return Ok(index as u32);
        }
        let index = u32::try_from(destination.len()).map_err(|_| "too many sorts")?;
        destination.push(pb::Sort {
            kind: Some(kind),
            ..Default::default()
        });
        Ok(index)
    }
}

impl TypeInfo {
    /// Match the one canonical scheme against complete, already resolved native
    /// input/result sorts. This is pure evidence checking, never native dispatch.
    pub(crate) fn check_native_signature(
        &self,
        definition: &pb::Program,
        native: &[ArcSort],
    ) -> Result<(), String> {
        // A logical binder can occur at differently named native container or
        // function roots. Compare their registered structure without replacing
        // either actual occurrence with the first binder's nominal root.
        fn validate_native_sort(native: &ArcSort, types: &TypeInfo) -> Result<(), String> {
            let registered = types
                .get_sort_by_name(native.name())
                .ok_or("unregistered native signature sort")?;
            if registered.is_eq_sort() != native.is_eq_sort()
                || registered.is_container_sort() != native.is_container_sort()
                || registered.value_type() != native.value_type()
            {
                return Err(
                    "native signature sort disagrees with its registered representation".into(),
                );
            }
            if let Some((family, _)) = types.builtin_sorts.get(native.name())
                && let Some(carrier) = types
                    .mksorts
                    .get(*family)
                    .and_then(|provider| provider.container_type)
                && (!native.is_container_sort() || native.value_type() != Some(carrier))
            {
                return Err(
                    "native signature family disagrees with its provider representation".into(),
                );
            }
            Ok(())
        }
        // None marks an active pair; completed comparisons retain their height
        // so reusing a shared sub-DAG cannot bypass the existing depth bound.
        fn same_sort(
            left: &ArcSort,
            right: &ArcSort,
            types: &TypeInfo,
            compared: &mut HashMap<(String, String), Option<usize>>,
            depth: usize,
        ) -> Option<usize> {
            if depth >= 256 {
                return None;
            }
            validate_native_sort(left, types).ok()?;
            validate_native_sort(right, types).ok()?;
            let pair = (left.name().to_owned(), right.name().to_owned());
            if let Some(completed) = compared.get(&pair) {
                return match *completed {
                    Some(height) if depth + height < 256 => Some(height),
                    _ => None,
                };
            }
            compared.insert(pair.clone(), None);
            let height = (|| {
                if left.name() == right.name() {
                    return Some(0);
                }
                if left.is_eq_sort() || right.is_eq_sort() {
                    return None;
                }
                if let (Ok(left), Ok(right)) = (
                    left.clone().as_arc_any().downcast::<FunctionSort>(),
                    right.clone().as_arc_any().downcast::<FunctionSort>(),
                ) {
                    if left.inputs().len() != right.inputs().len() {
                        return None;
                    }
                    let mut height =
                        same_sort(&left.output(), &right.output(), types, compared, depth + 1)? + 1;
                    for (left, right) in left.inputs().iter().zip(right.inputs()) {
                        height =
                            height.max(same_sort(left, right, types, compared, depth + 1)? + 1);
                    }
                    return Some(height);
                }
                match (
                    types.builtin_sorts.get(left.name()),
                    types.builtin_sorts.get(right.name()),
                ) {
                    (Some((left_family, left_args)), Some((right_family, right_args)))
                        if left_family == right_family && left_args.len() == right_args.len() =>
                    {
                        left_args.iter().zip(right_args).try_fold(
                            0_usize,
                            |height, (left, right)| {
                                same_sort(left, right, types, compared, depth + 1)
                                    .map(|child| height.max(child + 1))
                            },
                        )
                    }
                    _ => None,
                }
            })();
            if let Some(height) = height {
                compared.insert(pair, Some(height));
            } else {
                compared.remove(&pair);
            }
            height
        }
        fn unify(
            index: u32,
            native: &ArcSort,
            patterns: &[pb::Sort],
            types: &TypeInfo,
            substitution: &mut [Option<ArcSort>],
            compared: &mut HashMap<(String, String), Option<usize>>,
            depth: usize,
        ) -> Result<(), String> {
            if depth >= 256 {
                return Err("cyclic or too-deep native signature".into());
            }
            validate_native_sort(native, types)?;
            match patterns
                .get(index as usize)
                .and_then(|sort| sort.kind.as_ref())
                .ok_or("missing native signature pattern")?
            {
                pb::sort::Kind::Var(variable) => {
                    let slot = substitution
                        .get_mut(*variable as usize)
                        .ok_or("unbound signature parameter")?;
                    if slot
                        .as_ref()
                        .is_some_and(|old| same_sort(old, native, types, compared, 0).is_none())
                    {
                        return Err("inconsistent builtin substitution".into());
                    }
                    *slot = Some(native.clone());
                }
                pb::sort::Kind::Eq(name) if native.is_eq_sort() && native.name() == name => (),
                pb::sort::Kind::Family(family)
                    if family.args.is_empty()
                        && !native.is_eq_sort()
                        && !native.is_container_sort()
                        && native.name() == family.name => {}
                pb::sort::Kind::Family(pattern) => {
                    let (family, parameters) = types
                        .builtin_sorts
                        .get(native.name())
                        .ok_or("native signature lacks family provenance")?;
                    if *family != pattern.name || parameters.len() != pattern.args.len() {
                        return Err("builtin argument/result sort mismatch".into());
                    }
                    for (pattern, actual) in pattern.args.iter().zip(parameters) {
                        unify(
                            *pattern,
                            actual,
                            patterns,
                            types,
                            substitution,
                            compared,
                            depth + 1,
                        )?;
                    }
                }
                pb::sort::Kind::Func(pattern) => {
                    let function = native
                        .clone()
                        .as_arc_any()
                        .downcast::<FunctionSort>()
                        .map_err(|_| "native signature requires a function sort")?;
                    if pattern.params.len() != function.inputs().len() {
                        return Err("builtin function arity mismatch".into());
                    }
                    for (pattern, actual) in pattern.params.iter().zip(function.inputs()) {
                        unify(
                            *pattern,
                            actual,
                            patterns,
                            types,
                            substitution,
                            compared,
                            depth + 1,
                        )?;
                    }
                    unify(
                        pattern.result,
                        &function.output(),
                        patterns,
                        types,
                        substitution,
                        compared,
                        depth + 1,
                    )?;
                }
                _ => return Err("builtin argument/result sort mismatch".into()),
            }
            Ok(())
        }
        let primitive = definition
            .declarations
            .iter()
            .find_map(|declaration| match &declaration.kind {
                Some(pb::declaration::Kind::HostPrimitive(primitive)) => Some(primitive),
                _ => None,
            })
            .ok_or("missing builtin definition")?;
        for root in native {
            validate_native_sort(root, self)?;
        }
        let mut patterns = vec![];
        let mut compared = HashMap::default();
        match normalized_signature(definition, primitive, &mut patterns)? {
            pb::host_primitive::Typing::Signature(signature) => {
                let inputs = native
                    .len()
                    .checked_sub(1)
                    .ok_or("missing builtin result")?;
                if inputs < signature.inputs.len()
                    || if signature.varargs.is_empty() {
                        inputs != signature.inputs.len()
                    } else {
                        !(inputs - signature.inputs.len()).is_multiple_of(signature.varargs.len())
                    }
                {
                    return Err("builtin arity mismatch".into());
                }
                let indices = signature
                    .inputs
                    .iter()
                    .map(|argument| argument.sort)
                    .chain(
                        signature
                            .varargs
                            .iter()
                            .cycle()
                            .take(inputs - signature.inputs.len())
                            .map(|argument| argument.sort),
                    )
                    .chain(signature.output);
                let mut substitution = vec![None; signature.type_params.len()];
                for (index, actual) in indices.zip(native) {
                    unify(
                        index,
                        actual,
                        &patterns,
                        self,
                        &mut substitution,
                        &mut compared,
                        0,
                    )?;
                }
                if substitution.iter().any(Option::is_none) {
                    return Err("undetermined builtin type parameter".into());
                }
                if signature
                    .equality_type_params
                    .iter()
                    .any(|i| !substitution[*i as usize].as_ref().unwrap().is_eq_sort())
                {
                    return Err("builtin type parameter requires an equality sort".into());
                }
            }
            pb::host_primitive::Typing::Application(_) => {
                let function = native
                    .first()
                    .ok_or("function application requires a function value")?
                    .clone()
                    .as_arc_any()
                    .downcast::<FunctionSort>()
                    .map_err(|_| "function application requires a function value")?;
                if native.len() != function.inputs().len() + 2 {
                    return Err("builtin function application arity mismatch".into());
                }
                if !function
                    .inputs()
                    .iter()
                    .cloned()
                    .chain([function.output()])
                    .zip(&native[1..])
                    .all(|(expected, actual)| {
                        same_sort(&expected, actual, self, &mut compared, 0).is_some()
                    })
                {
                    return Err("function application argument/result sort mismatch".into());
                }
            }
        }
        Ok(())
    }

    /// Bind an owned descriptor to all actual fixed/tail/result occurrences of
    /// one native implementation, preserving each nominal root independently.
    pub fn instantiate_builtin_signature(
        &self,
        key: &str,
        inputs: &[ArcSort],
        tail: &[ArcSort],
        output: &ArcSort,
    ) -> Result<BuiltinInstance, String> {
        let definition = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin definition")?
            .clone();
        if !self.builtin_owners.contains_key(key) {
            return Err("builtin definition is not registry-owned".into());
        }
        let primitive = definition
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostPrimitive(p)) => Some(p),
                _ => None,
            })
            .ok_or("missing builtin definition")?;
        let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            return Err("complete signature binding requires a signature descriptor".into());
        };
        if inputs.len() != signature.inputs.len() || tail.len() != signature.varargs.len() {
            return Err("builtin implementation fixed/tail arity mismatch".into());
        }
        let sorts = inputs
            .iter()
            .chain(tail)
            .chain([output])
            .cloned()
            .collect::<Vec<_>>();
        self.check_native_signature(&definition, &sorts)?;
        let mut dispatch_name = format!(
            "__egglog_signature_{}_{}_{}_{}",
            key.len(),
            key,
            inputs.len(),
            tail.len()
        );
        for sort in &sorts {
            dispatch_name.push_str(&format!("_{}_{}", sort.name().len(), sort.name()));
        }
        Ok(BuiltinInstance {
            definition,
            dispatch_name,
            binding: InstanceBinding::Exact {
                sorts,
                fixed: inputs.len(),
                tail: tail.len(),
            },
        })
    }

    /// Attach the existing native predicate to an owned generic descriptor.
    /// Each complete acceptance is checked against that descriptor; no backend
    /// implementation is copied or registered by constructing this instance.
    pub fn instantiate_builtin_open(&self, key: &str) -> Result<BuiltinInstance, String> {
        let definition = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin definition")?
            .clone();
        if !self.builtin_owners.contains_key(key) {
            return Err("builtin definition is not registry-owned".into());
        }
        Ok(BuiltinInstance {
            definition,
            dispatch_name: format!("__egglog_open_{}_{}", key.len(), key),
            binding: InstanceBinding::Open,
        })
    }

    /// Binds a registered family definition to this exact nominal instance.
    /// Only registry-owned records can create an instance identity.
    pub fn instantiate_builtin(
        &self,
        key: &str,
        native: &ArcSort,
    ) -> Result<BuiltinInstance, String> {
        let definition = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin definition")?
            .clone();
        if definition.declarations.iter().any(|d| matches!(&d.kind, Some(pb::declaration::Kind::HostPrimitive(p)) if matches!(p.typing, Some(pb::host_primitive::Typing::Application(_))))) {
            let function = native.clone().as_arc_any().downcast::<FunctionSort>().map_err(|_| "application instance requires a native function sort")?;
            if self.builtin_owners.get(key).copied() != Some("UnstableFn") {
                return Err("application definition belongs to another provider".into());
            }
            return Ok(BuiltinInstance {
                definition,
                dispatch_name: format!("__egglog_instance_{}_{}_{}", key.len(), key, native.name()),
                binding: InstanceBinding::Application(function),
            });
        }
        let (family, parameters) = self
            .builtin_sorts
            .get(native.name())
            .ok_or("unregistered family instance")?;
        if self.builtin_owners.get(key) != Some(family) {
            return Err("builtin definition belongs to another family".into());
        }
        for declaration in &definition.declarations {
            if let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind
                && let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing
                && signature.equality_type_params.iter().any(|i| {
                    parameters
                        .get(*i as usize)
                        .is_none_or(|sort| !sort.is_eq_sort())
                })
            {
                return Err("builtin type parameter requires an equality sort".into());
            }
        }
        let mut resolved = vec![None; definition.sorts.len()];
        fn resolve(
            index: usize,
            program: &pb::Program,
            types: &TypeInfo,
            family: &str,
            parameters: &[ArcSort],
            native: &ArcSort,
            resolved: &mut [Option<NativeSortBinding>],
        ) -> Result<NativeSortBinding, String> {
            if let Some(sort) = &resolved[index] {
                return Ok(sort.clone());
            }
            let sort = match program.sorts[index]
                .kind
                .as_ref()
                .ok_or("missing sort pattern")?
            {
                pb::sort::Kind::Var(i) => NativeSortBinding::Exact(
                    parameters
                        .get(*i as usize)
                        .ok_or("unbound type parameter")?
                        .clone(),
                ),
                pb::sort::Kind::Family(f) if f.args.is_empty() => NativeSortBinding::Exact(
                    types
                        .get_sort_by_name(&f.name)
                        .ok_or("unknown scalar signature sort")?
                        .clone(),
                ),
                pb::sort::Kind::Family(f) if f.name == family => {
                    let args = f
                        .args
                        .iter()
                        .map(|i| {
                            resolve(
                                *i as usize,
                                program,
                                types,
                                family,
                                parameters,
                                native,
                                resolved,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if args.len() != parameters.len() || !args.iter().zip(parameters).all(|(arg, parameter)| {
                        matches!(arg, NativeSortBinding::Exact(sort) if sort.name() == parameter.name())
                    })
                    {
                        return Err("signature family application differs from its instance".into());
                    }
                    NativeSortBinding::Exact(native.clone())
                }
                pb::sort::Kind::Family(f) => NativeSortBinding::Foreign(
                    types
                        .mksorts
                        .get(&f.name)
                        .and_then(|provider| provider.container_type)
                        .ok_or_else(|| {
                            format!(
                                "foreign family {} has no native container representation",
                                f.name
                            )
                        })?,
                ),
                _ => return Err("unsupported native signature pattern".into()),
            };
            resolved[index] = Some(sort.clone());
            Ok(sort)
        }
        for index in 0..resolved.len() {
            resolve(
                index,
                &definition,
                self,
                family,
                parameters,
                native,
                &mut resolved,
            )?;
        }
        Ok(BuiltinInstance {
            definition,
            dispatch_name: format!("__egglog_instance_{}_{}_{}", key.len(), key, native.name()),
            binding: InstanceBinding::Owner(resolved.into_iter().map(Option::unwrap).collect()),
        })
    }

    /// Checks a closed wire call against one definition and selects its native
    /// instance without source-alias overload search. Returned compiler keys
    /// reference the same implementation/context IDs, not another declaration.
    pub fn resolve_builtin(
        &self,
        program: &pb::Program,
        name: &str,
        arguments: &[u32],
        output: u32,
        native_sorts: &[ArcSort],
    ) -> Result<String, String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        let definition = self
            .builtin_definitions
            .get(name)
            .ok_or_else(|| format!("unknown builtin definition {name}"))?;
        let mut arena = vec![];
        let actuals: Vec<_> = arguments
            .iter()
            .copied()
            .chain([output])
            .map(|i| import_sort(&program.sorts, i, &mut arena))
            .collect::<Result<_, _>>()?;
        if arena
            .iter()
            .any(|sort| matches!(sort.kind, Some(pb::sort::Kind::Var(_))))
        {
            return Err("wire call sorts must be closed".into());
        }
        fn matches_native(
            index: u32,
            native: &ArcSort,
            arena: &[pb::Sort],
            types: &TypeInfo,
        ) -> bool {
            match arena[index as usize].kind.as_ref().unwrap() {
                pb::sort::Kind::Eq(name) => native.is_eq_sort() && native.name() == name,
                pb::sort::Kind::Family(family) if family.args.is_empty() => {
                    !native.is_eq_sort()
                        && !native.is_container_sort()
                        && native.name() == family.name
                }
                pb::sort::Kind::Family(family) => {
                    types
                        .builtin_sorts
                        .get(native.name())
                        .is_some_and(|(name, parameters)| {
                            *name == family.name
                                && family.args.len() == parameters.len()
                                && family.args.iter().zip(parameters).all(|(index, native)| {
                                    matches_native(*index, native, arena, types)
                                })
                        })
                }
                pb::sort::Kind::Func(function) => native
                    .clone()
                    .as_arc_any()
                    .downcast::<FunctionSort>()
                    .is_ok_and(|sort| {
                        function.params.len() == sort.inputs().len()
                            && function
                                .params
                                .iter()
                                .zip(sort.inputs())
                                .all(|(index, sort)| matches_native(*index, sort, arena, types))
                            && matches_native(function.result, &sort.output(), arena, types)
                    }),
                _ => false,
            }
        }
        if actuals.len() != native_sorts.len()
            || !actuals
                .iter()
                .zip(native_sorts)
                .all(|(index, native)| matches_native(*index, native, &arena, self))
        {
            return Err("native instance binding disagrees with wire sort".into());
        }
        self.check_native_signature(definition, native_sorts)?;
        let candidates: Vec<_> = self
            .builtin_primitives
            .get(name)
            .into_iter()
            .flatten()
            .filter(|p| p.accept(native_sorts, self))
            .collect();
        if candidates.len() != 1 {
            return Err(format!(
                "builtin {name} has {} matching native instances",
                candidates.len()
            ));
        }
        Ok(candidates[0]
            .primitive
            .builtin_instance()
            .map_or_else(|| name.into(), |instance| instance.dispatch_name.clone()))
    }

    /// Exports the migrated registration definitions and inventories all remaining
    /// opaque primitive registrations and generic family factories.
    pub fn builtin_catalog(&self) -> Result<BuiltinCatalog, String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        let mut catalog = BuiltinCatalog {
            definitions: pb::Program {
                ir_version: 1,
                ..Default::default()
            },
            undescribed_primitives: self
                .primitives
                .values()
                .flatten()
                .filter(|primitive| primitive.primitive.builtin_definition().is_none())
                .map(|primitive| primitive.primitive.name().to_owned())
                .collect(),
            undescribed_families: self
                .mksorts
                .keys()
                .filter(|family| {
                    !self.builtin_families.declarations.iter().any(|d| matches!(&d.kind, Some(pb::declaration::Kind::HostSortFamily(f)) if &f.name == *family))
                })
                .cloned()
                .collect(),
            undescribed_family_primitives: self.builtin_family_gaps.clone(),
            undescribed_sorts: vec![],
        };
        catalog.undescribed_primitives.sort();
        catalog.undescribed_families.sort();
        catalog.undescribed_family_primitives.sort();
        let mut families = self.builtin_families.clone();
        families.ir_version = 1;
        definitions::reconcile_declarations(&families, &mut catalog.definitions)?;
        let mut keys = self.builtin_definitions.keys().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            self.export_builtin_definition(key, &mut catalog.definitions)?;
        }
        catalog.undescribed_sorts = self.sorts.values().filter(|sort| {
            !sort.is_eq_sort() && !(*sort).clone().as_arc_any().is::<FunctionSort>() && !self.builtin_sorts.contains_key(sort.name()) && !catalog.definitions.declarations.iter().any(|declaration| matches!(&declaration.kind, Some(pb::declaration::Kind::HostSortFamily(family)) if family.name == sort.name()))
        }).map(|sort| sort.name().to_owned()).collect();
        catalog.undescribed_sorts.sort();
        Ok(catalog)
    }

    /// Checks the provider's exact key and signature, not presentation metadata.
    /// Reconcile bindings against the full canonical declaration context first:
    /// a default may refer to another builtin or a previously installed table.
    pub fn check_builtin_signature(
        &self,
        program: &pb::Program,
        primitive: &pb::HostPrimitive,
    ) -> Result<(), String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        let definition = self
            .builtin_definitions
            .get(&primitive.name)
            .ok_or_else(|| format!("unknown builtin definition {}", primitive.name))?;
        let declaration = definition
            .declarations
            .iter()
            .find(|declaration| {
                matches!(
                    &declaration.kind,
                    Some(pb::declaration::Kind::HostPrimitive(_))
                )
            })
            .unwrap();
        let Some(pb::declaration::Kind::HostPrimitive(expected)) = &declaration.kind else {
            unreachable!()
        };
        let mut sorts = vec![];
        if normalized_signature(program, primitive, &mut sorts)?
            != normalized_signature(definition, expected, &mut sorts)?
        {
            return Err(format!("conflicting builtin signature {}", primitive.name));
        }
        Ok(())
    }
}
