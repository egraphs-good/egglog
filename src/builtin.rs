//! Engine-owned protobuf builtin definitions and derived native constraints.
//!
//! This first migration covers closed scalar signatures, not a complete generic
//! catalog. Opaque registrations and uninstantiated families remain visible in
//! the inventory. Export never calls an implementation or a proof validator.

use crate::{constraint::TypeConstraint, proto as pb, *};

/// Migrated definitions plus explicit remaining registration gaps. This is not
/// a complete engine snapshot and must not be used as a successful Freeze.
pub struct BuiltinCatalog {
    /// Canonical protobuf definitions with remapped, shared sort indices.
    pub definitions: pb::Program,
    /// One entry per native registration lacking a definition; aliases can repeat.
    pub undescribed_primitives: Vec<String>,
    /// Presort factories whose generic definitions have not migrated yet.
    pub undescribed_families: Vec<String>,
    /// Installed native non-equality sorts not represented by a family entry.
    /// Concrete container aliases remain here until structural mapping exists.
    pub undescribed_sorts: Vec<String>,
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
    if primitive.name.is_empty() || !signature.type_params.is_empty() || signature.varargs.is_some()
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
        || !program.nodes.is_empty()
        || !program.commands.is_empty()
        || !program.rules.is_empty()
        || !program.rulesets.is_empty()
    {
        return Err("builtin definition must contain only signature sorts and declarations".into());
    }
    let mut key = None;
    for declaration in &program.declarations {
        match &declaration.kind {
            Some(pb::declaration::Kind::HostPrimitive(primitive)) if key.is_none() => {
                signature_sorts(program, primitive)?;
                key = Some(primitive.name.as_str());
            }
            Some(pb::declaration::Kind::HostSortFamily(family))
                if family.arity == 0 && !family.name.is_empty() => {}
            _ => {
                return Err(
                    "expected exactly one builtin definition and scalar family descriptors".into(),
                );
            }
        }
    }
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
    let mut remap = vec![];
    for sort in &definition.sorts {
        let index = destination
            .sorts
            .iter()
            .position(|existing| existing.kind == sort.kind)
            .unwrap_or_else(|| {
                destination.sorts.push(sort.clone());
                destination.sorts.len() - 1
            });
        remap.push(u32::try_from(index).map_err(|_| "too many imported sorts")?);
    }
    for declaration in &definition.declarations {
        let mut declaration = declaration.clone();
        let name = match declaration.kind.as_mut().unwrap() {
            pb::declaration::Kind::HostPrimitive(primitive) => {
                let Some(pb::host_primitive::Typing::Signature(signature)) = &mut primitive.typing
                else {
                    unreachable!()
                };
                for arg in &mut signature.inputs {
                    arg.sort = remap[arg.sort as usize];
                }
                signature.output = Some(remap[signature.output.unwrap() as usize]);
                primitive.name.clone()
            }
            pb::declaration::Kind::HostSortFamily(family) => family.name.clone(),
            _ => unreachable!(),
        };
        let sort_namespace = matches!(
            declaration.kind,
            Some(pb::declaration::Kind::HostSortFamily(_))
        );
        let existing = destination.declarations.iter().find(|existing| {
            match (&existing.kind, sort_namespace) {
                (Some(pb::declaration::Kind::HostSortFamily(family)), true) => family.name == name,
                (Some(pb::declaration::Kind::EqSort(sort)), true) => sort.name == name,
                (Some(pb::declaration::Kind::HostPrimitive(primitive)), false) => {
                    primitive.name == name
                }
                (Some(pb::declaration::Kind::Primitive(primitive)), false) => {
                    primitive.name == name
                }
                (Some(pb::declaration::Kind::Relation(relation)), false) => relation.name == name,
                (Some(pb::declaration::Kind::Function(function)), false) => function.name == name,
                (Some(pb::declaration::Kind::Constructor(constructor)), false) => {
                    constructor.name == name
                }
                _ => false,
            }
        });
        if let Some(existing) = existing {
            let same_definition = match (&existing.kind, &declaration.kind) {
                (
                    Some(pb::declaration::Kind::HostPrimitive(existing)),
                    Some(pb::declaration::Kind::HostPrimitive(incoming)),
                ) => {
                    // Arg.name is diagnostic, not part of signature identity.
                    signature_sorts(destination, existing)?
                        == signature_sorts(destination, incoming)?
                }
                (existing, incoming) => existing == incoming,
            };
            if !same_definition || existing.bindings != declaration.bindings {
                return Err(format!("conflicting builtin declaration {name}"));
            }
        } else {
            destination.declarations.push(declaration);
        }
    }
    Ok(key)
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
            undescribed_families: self.mksorts.keys().cloned().collect(),
            undescribed_sorts: vec![],
        };
        catalog.undescribed_primitives.sort();
        catalog.undescribed_families.sort();
        let mut keys = self.builtin_primitives.keys().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            import_definition(
                self.builtin_primitives[key][0]
                    .primitive
                    .builtin_definition()
                    .unwrap(),
                &mut catalog.definitions,
            )?;
        }
        catalog.undescribed_sorts = self.sorts.values().filter(|sort| {
            !sort.is_eq_sort() && !catalog.definitions.declarations.iter().any(|declaration| matches!(&declaration.kind, Some(pb::declaration::Kind::HostSortFamily(family)) if family.name == sort.name()))
        }).map(|sort| sort.name().to_owned()).collect();
        catalog.undescribed_sorts.sort();
        Ok(catalog)
    }

    /// Checks a supplied definition against the exact registered key. Neither a
    /// compatible signature nor an overloaded source alias can substitute for it.
    pub fn check_builtin(
        &self,
        program: &pb::Program,
        primitive: &pb::HostPrimitive,
        bindings: Option<&pb::CallableBindings>,
    ) -> Result<(), String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        let registered = self
            .builtin_primitives
            .get(&primitive.name)
            .ok_or_else(|| format!("unknown builtin definition {}", primitive.name))?;
        let definition = registered[0].primitive.builtin_definition().unwrap();
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
        if bindings.is_some_and(|bindings| Some(bindings) != declaration.bindings.as_ref()) {
            return Err(format!("conflicting builtin bindings {}", primitive.name));
        }
        if signature_sorts(program, primitive)? != signature_sorts(definition, expected)? {
            return Err(format!("conflicting builtin signature {}", primitive.name));
        }
        Ok(())
    }
}
