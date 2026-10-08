//! Engine-owned protobuf builtin definitions and derived native constraints.
//!
//! This migration covers scalar signatures and selected Vec operations, not a
//! complete generic catalog. Opaque registrations and family gaps remain visible in
//! the inventory. Export never calls an implementation or a proof validator.

use crate::{constraint::TypeConstraint, proto as pb, *};

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
    sorts: Vec<ArcSort>,
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
        for (index, cast) in indices.zip(inputs.iter().copied().chain([output])) {
            if let Some((ty, container)) = cast {
                let sort = &self.sorts[index as usize];
                if sort.value_type() != Some(ty) || sort.is_container_sort() != container {
                    return Err(format!(
                        "builtin implementation conversion disagrees with {}",
                        sort.name()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Derives fixed/tail native assignments from the canonical pattern arena.
    pub fn type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
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
        Box::new(InstanceConstraint {
            fixed: signature
                .inputs
                .iter()
                .map(|a| self.sorts[a.sort as usize].clone())
                .collect(),
            tail: signature
                .varargs
                .iter()
                .map(|a| self.sorts[a.sort as usize].clone())
                .collect(),
            output: self.sorts[signature.output.unwrap() as usize].clone(),
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
}

struct InstanceConstraint {
    fixed: Vec<ArcSort>,
    tail: Vec<ArcSort>,
    output: ArcSort,
    name: String,
    span: Span,
}

impl TypeConstraint for InstanceConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn constraint::Constraint<AtomTerm, ArcSort>>> {
        let mut sorts = self.fixed.clone();
        if !self.tail.is_empty() {
            // Native arguments include the result. Only complete input groups
            // count; SimpleTypeConstraint rejects any missing prefix or remainder.
            let groups = arguments.len().saturating_sub(self.fixed.len() + 1) / self.tail.len();
            sorts.extend(
                self.tail
                    .iter()
                    .cloned()
                    .cycle()
                    .take(groups * self.tail.len()),
            );
        }
        sorts.push(self.output.clone());
        SimpleTypeConstraint::new(&self.name, sorts, self.span.clone()).get(arguments, typeinfo)
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
            pb::sort::Kind::Eq(name) if !name.is_empty() => (),
            pb::sort::Kind::Var(_) => (),
            _ => return Err("unsupported builtin sort pattern".into()),
        }
        active.remove(&index);
        if let Some(existing) = destination
            .iter()
            .position(|existing| existing.kind == sort.kind)
        {
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

fn normalized_signature(
    program: &pb::Program,
    primitive: &pb::HostPrimitive,
    sorts: &mut Vec<pb::Sort>,
) -> Result<pb::GenericSignature, String> {
    if primitive.name.is_empty() {
        return Err("builtin definition key must not be empty".into());
    }
    let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
        return Err("unsupported builtin typing form".into());
    };
    let mut signature = signature.clone();
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
            _ => (),
        }
    }
    Ok(signature)
}

/// Builds a closed signature from the macro's existing type annotations. The
/// returned protobuf is the signature; native constraints are derived from it.
/// Symbolic family signatures will use the same arenas and GenericSignature,
/// but cannot be inferred from an already instantiated opaque container sort.
pub fn closed_signature(
    key: &str,
    alias: &str,
    inputs: &[(&str, ArcSort)],
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

/// Validates and normalizes the currently supported closed signature into native
/// sort names, including its result. This does not execute or inspect a body.
pub fn signature_sorts(
    program: &pb::Program,
    primitive: &pb::HostPrimitive,
) -> Result<Vec<String>, String> {
    let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
        return Err("function-application builtin typing is not implemented yet".into());
    };
    if primitive.name.is_empty()
        || !signature.type_params.is_empty()
        || !signature.varargs.is_empty()
    {
        return Err(
            "builtin typing currently requires a named closed fixed-arity signature".into(),
        );
    }
    signature.inputs.iter().map(|arg| arg.sort).chain(std::iter::once(signature.output.ok_or("missing builtin result sort")?)).map(|index| {
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

/// The scalar addition surface views, with owners and trait types derived from
/// the canonical signature, never from a separately maintained type scheme.
pub fn scalar_add_bindings(signature: &pb::GenericSignature) -> pb::CallableBindings {
    let owner = Some(pb::BindingOwner {
        kind: Some(pb::binding_owner::Kind::Sort(signature.inputs[0].sort)),
    });
    pb::CallableBindings {
        python: Some(pb::PythonBindings {
            views: vec![pb::PythonCallable {
                kind: pb::PythonCallKind::Method.into(),
                path: vec!["__add__".into()],
                owner,
                receiver: Some(0),
                params: vec![pb::PythonParameter {
                    core_input: Some(1),
                    name: "other".into(),
                    default_expr: None,
                }],
                ..Default::default()
            }],
        }),
        rust: Some(pb::RustBindings {
            views: vec![pb::RustCallable {
                path: vec!["add".into()],
                owner,
                receiver: Some(pb::RustReceiver {
                    core_input: Some(0),
                    borrowed: false,
                }),
                params: vec![pb::RustParameter {
                    core_input: Some(1),
                    name: "rhs".into(),
                    borrowed: false,
                }],
                trait_impl: Some(pb::RustTrait {
                    path: vec!["core".into(), "ops".into(), "Add".into()],
                    args: vec![pb::RustType {
                        sort: Some(signature.inputs[1].sort),
                        borrowed: false,
                    }],
                    output_associated_type: Some("Output".into()),
                }),
                ..Default::default()
            }],
        }),
        ..Default::default()
    }
}

struct SignatureConstraint {
    sorts: Vec<String>,
    name: String,
    span: Span,
}

impl TypeConstraint for SignatureConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn constraint::Constraint<AtomTerm, ArcSort>>> {
        let sorts = self
            .sorts
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
    Box::new(SignatureConstraint {
        sorts: signature_sorts(definition, primitive).unwrap(),
        name: diagnostic_name.into(),
        span: span.clone(),
    })
}

impl TypeInfo {
    /// Installs a family's generated declaration at its native registration site.
    /// Later fragments may assert the same arity without repeating its views.
    pub fn register_builtin_family(&mut self, family: pb::HostSortFamily) -> Result<(), String> {
        if let Some(sort) = self.sorts.get(&family.name) {
            if sort.is_eq_sort() || sort.is_container_sort() || family.arity != 0 {
                return Err(
                    "builtin family disagrees with native sort kind or nullary arity".into(),
                );
            }
        } else if !self.mksorts.contains_key(&family.name) {
            return Err("builtin family has no native registration".into());
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
        self.builtin_sorts
            .insert(sort.name().into(), (family, parameters));
        Ok(())
    }
}

impl EGraph {
    /// Exports structural sorts from registered family provenance, never from a
    /// native alias spelling or Rust value TypeId. Equality sorts stay nominal.
    pub fn export_sort(
        &self,
        sort: &ArcSort,
        destination: &mut Vec<pb::Sort>,
    ) -> Result<u32, String> {
        let kind = if sort.is_eq_sort() {
            pb::sort::Kind::Eq(sort.name().into())
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
        let (family, parameters) = self
            .builtin_sorts
            .get(native.name())
            .ok_or("unregistered family instance")?;
        if self.builtin_owners.get(key) != Some(family) {
            return Err("builtin definition belongs to another family".into());
        }
        let mut resolved = vec![None; definition.sorts.len()];
        fn resolve(
            index: usize,
            program: &pb::Program,
            types: &TypeInfo,
            family: &str,
            parameters: &[ArcSort],
            native: &ArcSort,
            resolved: &mut [Option<ArcSort>],
        ) -> Result<ArcSort, String> {
            if let Some(sort) = &resolved[index] {
                return Ok(sort.clone());
            }
            let sort = match program.sorts[index]
                .kind
                .as_ref()
                .ok_or("missing sort pattern")?
            {
                pb::sort::Kind::Var(i) => parameters
                    .get(*i as usize)
                    .ok_or("unbound type parameter")?
                    .clone(),
                pb::sort::Kind::Family(f) if f.args.is_empty() => types
                    .get_sort_by_name(&f.name)
                    .ok_or("unknown scalar signature sort")?
                    .clone(),
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
                    if args
                        .iter()
                        .map(|s| s.name())
                        .ne(parameters.iter().map(|s| s.name()))
                    {
                        return Err("signature family application differs from its instance".into());
                    }
                    native.clone()
                }
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
            sorts: resolved.into_iter().map(Option::unwrap).collect(),
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
        let primitive = definition
            .declarations
            .iter()
            .find_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostPrimitive(p)) => Some(p),
                _ => None,
            })
            .unwrap();
        let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
            return Err("unsupported builtin typing".into());
        };
        if arguments.len() < signature.inputs.len()
            || if signature.varargs.is_empty() {
                arguments.len() != signature.inputs.len()
            } else {
                !(arguments.len() - signature.inputs.len()).is_multiple_of(signature.varargs.len())
            }
        {
            return Err(format!("builtin arity mismatch for {name}"));
        }
        let mut arena = vec![];
        let patterns: Vec<_> = signature
            .inputs
            .iter()
            .map(|a| a.sort)
            .chain(
                signature
                    .varargs
                    .iter()
                    .cycle()
                    .take(arguments.len() - signature.inputs.len())
                    .map(|a| a.sort),
            )
            .chain(signature.output)
            .collect();
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
        let mut substitution = vec![None; signature.type_params.len()];
        fn check(
            pattern: u32,
            actual: u32,
            patterns: &[pb::Sort],
            actuals: &[pb::Sort],
            substitution: &mut [Option<u32>],
        ) -> Result<(), String> {
            match (
                patterns.get(pattern as usize).and_then(|s| s.kind.as_ref()),
                actuals.get(actual as usize).and_then(|s| s.kind.as_ref()),
            ) {
                (
                    Some(pb::sort::Kind::Var(variable)),
                    Some(pb::sort::Kind::Family(_) | pb::sort::Kind::Eq(_)),
                ) => {
                    let slot = substitution
                        .get_mut(*variable as usize)
                        .ok_or("unbound signature parameter")?;
                    if slot.is_some_and(|old| old != actual) {
                        return Err("inconsistent builtin substitution".into());
                    }
                    *slot = Some(actual);
                    Ok(())
                }
                (Some(pb::sort::Kind::Family(p)), Some(pb::sort::Kind::Family(a)))
                    if p.name == a.name && p.args.len() == a.args.len() =>
                {
                    for (p, a) in p.args.iter().zip(&a.args) {
                        check(*p, *a, patterns, actuals, substitution)?;
                    }
                    Ok(())
                }
                (Some(pb::sort::Kind::Eq(p)), Some(pb::sort::Kind::Eq(a))) if p == a => Ok(()),
                _ => Err("builtin argument/result sort mismatch".into()),
            }
        }
        for (pattern, actual) in patterns.into_iter().zip(actuals) {
            check(
                pattern,
                actual,
                &definition.sorts,
                &arena,
                &mut substitution,
            )?;
        }
        if substitution.iter().any(Option::is_none) {
            return Err("undetermined builtin type parameter".into());
        }
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
            !sort.is_eq_sort() && !self.builtin_sorts.contains_key(sort.name()) && !catalog.definitions.declarations.iter().any(|declaration| matches!(&declaration.kind, Some(pb::declaration::Kind::HostSortFamily(family)) if family.name == sort.name()))
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
