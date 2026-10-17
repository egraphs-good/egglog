//! Native builtin schemes and exact implementation provenance.
//!
//! The registered descriptions drive native typing. Serialization, authored
//! bodies and language presentation belong to consumers of this registry.

use crate::{constraint::TypeConstraint, typechecking::PrimitiveWithId, *};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SortPattern {
    Param(usize),
    Eq(String),
    Family {
        name: String,
        args: Vec<SortPattern>,
    },
    Function {
        inputs: Vec<SortPattern>,
        output: Box<SortPattern>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeParameter {
    pub name: String,
    pub equality: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Argument {
    pub name: String,
    pub sort: SortPattern,
}

/// Fixed inputs followed by zero or more complete repetitions of `tail`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Signature {
    pub params: Vec<TypeParameter>,
    pub fixed: Vec<Argument>,
    pub tail: Vec<Argument>,
    pub output: SortPattern,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuiltinTyping {
    Signature(Signature),
    FunctionApplication,
    /// Literal target name and captures producing a native function value.
    /// Target-specific acceptance remains the implementation's native predicate.
    FunctionConstructor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuiltinDescription {
    pub key: String,
    pub aliases: Vec<String>,
    pub typing: BuiltinTyping,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyDescription {
    pub name: String,
    pub arity: usize,
}

/// A partial native inventory: missing descriptions remain explicit gaps.
pub struct BuiltinCatalog {
    pub descriptions: Vec<Arc<BuiltinDescription>>,
    pub families: Vec<FamilyDescription>,
    pub undescribed_primitives: Vec<String>,
    pub undescribed_families: Vec<String>,
    pub undescribed_family_primitives: Vec<String>,
    pub undescribed_sorts: Vec<String>,
}

/// Registry-owned description bound to exact native implementation sorts.
/// Cloning retains ownership identity and each nominal occurrence.
#[derive(Clone)]
pub struct BuiltinInstance {
    pub description: Arc<BuiltinDescription>,
    pub dispatch_name: String,
    binding: InstanceBinding,
}

#[derive(Clone)]
enum InstanceBinding {
    Owner(Vec<NativeSortBinding>),
    Exact {
        sorts: Vec<ArcSort>,
        fixed: usize,
        tail: usize,
    },
    Application(Arc<FunctionSort>),
    Open,
}

#[derive(Clone)]
enum NativeSortBinding {
    Exact(ArcSort),
    Family {
        name: String,
        args: Vec<NativeSortBinding>,
        carrier: std::any::TypeId,
    },
}

/// Exact table identity or registrations selected independently per context.
/// Registrations retain their original implementation, validator and IDs.
pub enum NativeFunctionTarget {
    Function(String),
    Primitive(Vec<PrimitiveWithId>),
}

impl BuiltinDescription {
    /// Validate the binder and structural shape before native publication.
    pub fn validate(&self) -> Result<(), String> {
        if self.key.is_empty()
            || self.aliases.is_empty()
            || self
                .aliases
                .iter()
                .any(|alias| alias.is_empty() || alias == &self.key)
            || self.aliases.iter().collect::<HashSet<_>>().len() != self.aliases.len()
        {
            return Err("a builtin needs a nonempty key and distinct nonempty aliases".into());
        }
        fn visit(pattern: &SortPattern, params: usize, depth: usize) -> Result<(), String> {
            if depth >= 256 {
                return Err("too-deep builtin sort pattern".into());
            }
            match pattern {
                SortPattern::Param(index) if *index >= params => {
                    Err("unbound builtin type parameter".into())
                }
                SortPattern::Param(_) => Ok(()),
                SortPattern::Eq(name) if !name.is_empty() => Ok(()),
                SortPattern::Family { name, args } if !name.is_empty() => {
                    for arg in args {
                        visit(arg, params, depth + 1)?;
                    }
                    Ok(())
                }
                SortPattern::Function { inputs, output } => {
                    for input in inputs {
                        visit(input, params, depth + 1)?;
                    }
                    visit(output, params, depth + 1)
                }
                _ => Err("empty builtin sort name".into()),
            }
        }
        if let BuiltinTyping::Signature(signature) = &self.typing {
            for pattern in signature
                .fixed
                .iter()
                .chain(&signature.tail)
                .map(|arg| &arg.sort)
                .chain([&signature.output])
            {
                visit(pattern, signature.params.len(), 0)?;
            }
        }
        Ok(())
    }
}

impl BuiltinInstance {
    /// Check typed conversions, including storage kind. Input converters cover
    /// each fixed slot followed by each repeated-tail slot.
    pub fn check_abi(
        &self,
        inputs: &[Option<(std::any::TypeId, bool)>],
        output: Option<(std::any::TypeId, bool)>,
        varargs: bool,
    ) -> Result<(), String> {
        let bindings = match &self.binding {
            InstanceBinding::Open => {
                return Err("open native domain has no exact implementation ABI".into());
            }
            InstanceBinding::Exact { sorts, fixed, tail } => {
                if varargs == (*tail == 0) || inputs.len() != *fixed + *tail {
                    return Err("builtin implementation arity differs from native signature".into());
                }
                sorts
                    .iter()
                    .cloned()
                    .map(NativeSortBinding::Exact)
                    .collect::<Vec<_>>()
            }
            InstanceBinding::Application(function) => {
                if varargs {
                    return Err(
                        "builtin implementation arity differs from native application".into(),
                    );
                }
                std::iter::once(function.clone() as ArcSort)
                    .chain(function.inputs().iter().cloned())
                    .chain([function.output()])
                    .map(NativeSortBinding::Exact)
                    .collect()
            }
            InstanceBinding::Owner(sorts) => {
                let BuiltinTyping::Signature(signature) = &self.description.typing else {
                    unreachable!()
                };
                if varargs == signature.tail.is_empty()
                    || inputs.len() != signature.fixed.len() + signature.tail.len()
                {
                    return Err("builtin implementation arity differs from native signature".into());
                }
                sorts.clone()
            }
        };
        if inputs.len() + 1 != bindings.len() {
            return Err("builtin implementation arity differs from native signature".into());
        }
        for (binding, cast) in bindings.iter().zip(inputs.iter().copied().chain([output])) {
            if let Some((ty, container)) = cast {
                let (expected, is_container, name) = match binding {
                    NativeSortBinding::Exact(sort) => {
                        (sort.value_type(), sort.is_container_sort(), sort.name())
                    }
                    NativeSortBinding::Family { name, carrier, .. } => {
                        (Some(*carrier), true, name.as_str())
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

    /// Derive the fixed/tail assignments of this bound implementation.
    pub fn type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        if let InstanceBinding::Application(function) = &self.binding {
            let sorts = std::iter::once(function.clone() as ArcSort)
                .chain(function.inputs().iter().cloned())
                .chain([function.output()])
                .collect();
            return SimpleTypeConstraint::new("unstable-app", sorts, span.clone()).into_box();
        }
        assert!(
            !matches!(self.binding, InstanceBinding::Open),
            "open builtin implementations must retain their native type constraints"
        );
        let BuiltinTyping::Signature(signature) = &self.description.typing else {
            unreachable!()
        };
        let fixed = signature.fixed.len();
        let tail = signature.tail.len();
        Box::new(InstanceConstraint {
            instance: self.clone(),
            fixed: (0..fixed).collect(),
            tail: (fixed..fixed + tail).collect(),
            output: fixed + tail,
            name: self.description.aliases[0].clone(),
            span: span.clone(),
        })
    }

    /// Only open implementations of this registry-owned description can share
    /// one private dispatch group. Exact instances remain distinct.
    pub(crate) fn can_share_dispatch(&self, existing: &[PrimitiveWithId]) -> bool {
        matches!(self.binding, InstanceBinding::Open)
            && existing.iter().all(|registered| {
                registered
                    .primitive
                    .builtin_instance()
                    .is_some_and(|other| {
                        matches!(other.binding, InstanceBinding::Open)
                            && Arc::ptr_eq(&self.description, &other.description)
                    })
            })
    }
}

struct InstanceConstraint {
    instance: BuiltinInstance,
    fixed: Vec<usize>,
    tail: Vec<usize>,
    output: usize,
    name: String,
    span: Span,
}

fn expand_signature<T: Clone>(
    fixed: &[T],
    tail: &[T],
    output: &T,
    argument_count: usize,
) -> Vec<T> {
    let mut sorts = fixed.to_vec();
    if !tail.is_empty() {
        let inputs = argument_count.saturating_sub(fixed.len() + 1);
        sorts.extend(
            tail.iter()
                .cloned()
                .cycle()
                .take((inputs / tail.len()) * tail.len()),
        );
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
        let sorts = expand_signature(&self.fixed, &self.tail, &self.output, arguments.len());
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
        fn matches(
            binding: &NativeSortBinding,
            native: &ArcSort,
            types: &TypeInfo,
            depth: usize,
        ) -> bool {
            if depth >= 256 {
                return false;
            }
            match binding {
                NativeSortBinding::Exact(sort) => sort.name() == native.name(),
                NativeSortBinding::Family {
                    name,
                    args,
                    carrier,
                } => {
                    native.is_container_sort()
                        && native.value_type() == Some(*carrier)
                        && types.builtin_sorts.get(native.name()).is_some_and(
                            |(family, parameters)| {
                                *family == name
                                    && args.len() == parameters.len()
                                    && args.iter().zip(parameters).all(|(binding, native)| {
                                        matches(binding, native, types, depth + 1)
                                    })
                            },
                        )
                }
            }
        }
        arguments
            .iter()
            .zip(sorts)
            .map(|(argument, index)| match &self.instance.binding {
                InstanceBinding::Exact { sorts, .. } => {
                    constraint::assign(argument.clone(), sorts[index].clone())
                }
                InstanceBinding::Owner(sorts) => match &sorts[index] {
                    NativeSortBinding::Exact(sort) => {
                        constraint::assign(argument.clone(), sort.clone())
                    }
                    binding @ NativeSortBinding::Family { .. } => constraint::xor(
                        typeinfo
                            .sorts
                            .values()
                            .filter(|sort| matches(binding, sort, typeinfo, 0))
                            .map(|sort| constraint::assign(argument.clone(), sort.clone()))
                            .collect(),
                    ),
                },
                _ => unreachable!("only owner/exact bindings derive instance constraints"),
            })
            .collect()
    }
}

/// Derive a closed scalar description from the macro's native type annotations.
pub fn closed_signature(
    key: &str,
    alias: &str,
    inputs: &[(&str, ArcSort)],
    varargs: &[(&str, ArcSort)],
    output: ArcSort,
) -> Result<BuiltinDescription, String> {
    let mut arguments = vec![];
    for (name, sort) in inputs
        .iter()
        .chain(varargs)
        .map(|(name, sort)| (*name, sort))
        .chain([("", &output)])
    {
        if sort.is_eq_sort() || sort.is_container_sort() || sort.value_type().is_none() {
            return Err(format!(
                "{} requires an explicit symbolic sort description",
                sort.name()
            ));
        }
        arguments.push(Argument {
            name: name.into(),
            sort: SortPattern::Family {
                name: sort.name().into(),
                args: vec![],
            },
        });
    }
    let output = arguments.pop().unwrap().sort;
    let tail = arguments.split_off(inputs.len());
    let description = BuiltinDescription {
        key: key.into(),
        aliases: vec![alias.into()],
        typing: BuiltinTyping::Signature(Signature {
            params: vec![],
            fixed: arguments,
            tail,
            output,
        }),
    };
    description.validate()?;
    Ok(description)
}

/// Derive the shared binder of fixed homogeneous `#` macro inputs.
pub fn homogeneous_signature(
    key: &str,
    alias: &str,
    inputs: &[&str],
    output: ArcSort,
) -> Result<BuiltinDescription, String> {
    if inputs.is_empty() {
        return Err("a homogeneous signature needs at least one input".into());
    }
    let mut description = closed_signature(key, alias, &[], &[], output)?;
    let BuiltinTyping::Signature(signature) = &mut description.typing else {
        unreachable!()
    };
    signature.params.push(TypeParameter {
        name: "T".into(),
        equality: false,
    });
    signature.fixed = inputs
        .iter()
        .map(|name| Argument {
            name: (*name).into(),
            sort: SortPattern::Param(0),
        })
        .collect();
    Ok(description)
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
        let sorts = expand_signature(&self.fixed, &self.tail, &self.output, arguments.len())
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

/// Compile the existing closed-scalar and homogeneous inference paths. General
/// schemes use bound instances or their retained native inference predicates.
pub fn rooted_type_constraints(
    key: &str,
    description: &BuiltinDescription,
    span: &Span,
) -> Box<dyn TypeConstraint> {
    description
        .validate()
        .expect("invalid native builtin description");
    assert_eq!(key, description.key, "builtin root identity mismatch");
    let BuiltinTyping::Signature(signature) = &description.typing else {
        panic!("native special form requires its own constraints")
    };
    let diagnostic_name = &description.aliases[0];
    if !signature.params.is_empty() {
        assert_eq!(
            signature.params.len(),
            1,
            "unsupported generic inference signature"
        );
        assert!(
            !signature.fixed.is_empty()
                && signature.tail.is_empty()
                && signature
                    .fixed
                    .iter()
                    .all(|arg| arg.sort == SortPattern::Param(0)),
            "generic inference requires homogeneous fixed inputs"
        );
        assert!(
            !signature.params[0].equality,
            "equality bounds require a bound native instance"
        );
        let SortPattern::Family { name, args } = &signature.output else {
            panic!("generic inference requires a closed scalar result")
        };
        assert!(
            args.is_empty(),
            "generic inference requires a closed scalar result"
        );
        return Box::new(HomogeneousConstraint {
            input_count: signature.fixed.len(),
            output: name.clone(),
            name: diagnostic_name.clone(),
            span: span.clone(),
        });
    }
    let mut sorts = signature
        .fixed
        .iter()
        .chain(&signature.tail)
        .map(|arg| &arg.sort)
        .chain([&signature.output])
        .map(|pattern| {
            let SortPattern::Family { name, args } = pattern else {
                panic!("closed inference requires scalar sorts")
            };
            assert!(args.is_empty(), "closed inference requires scalar sorts");
            name.clone()
        })
        .collect::<Vec<_>>();
    let output = sorts.pop().unwrap();
    let tail = sorts.split_off(signature.fixed.len());
    Box::new(SignatureConstraint {
        fixed: sorts,
        tail,
        output,
        name: diagnostic_name.clone(),
        span: span.clone(),
    })
}

impl TypeInfo {
    pub fn register_builtin_family(&mut self, family: FamilyDescription) -> Result<(), String> {
        if family.name.is_empty() {
            return Err("empty builtin family name".into());
        }
        if !self.mksorts.contains_key(&family.name) {
            let sort = self
                .sorts
                .get(&family.name)
                .ok_or("builtin family has no native registration")?;
            if sort.is_eq_sort() || sort.is_container_sort() || family.arity != 0 {
                return Err(
                    "builtin family disagrees with native sort kind or nullary arity".into(),
                );
            }
        }
        if let Some(existing) = self.builtin_families.get(&family.name) {
            if existing.as_ref() != &family {
                return Err(format!("conflicting builtin family {}", family.name));
            }
        } else {
            self.builtin_families
                .insert(family.name.clone(), Arc::new(family));
        }
        Ok(())
    }

    /// Register structural provenance without erasing exact nominal parameters.
    pub fn register_builtin_sort(
        &mut self,
        family: &'static str,
        sort: ArcSort,
        parameters: Vec<ArcSort>,
    ) -> Result<(), String> {
        let expected = self
            .builtin_families
            .get(family)
            .ok_or_else(|| format!("undescribed family {family}"))?;
        if expected.arity != parameters.len() || self.builtin_sorts.contains_key(sort.name()) {
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

    /// Access a description only from a coherent registered catalog.
    pub fn builtin_description(&self, key: &str) -> Result<&BuiltinDescription, String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        self.builtin_definitions
            .get(key)
            .map(Arc::as_ref)
            .ok_or_else(|| format!("unknown builtin description {key}"))
    }

    /// Retain the exact native family owner across graph clones and scopes.
    pub fn registered_builtin_family(&self, name: &str) -> Result<Arc<FamilyDescription>, String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        self.builtin_families
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown builtin family {name}"))
    }

    /// Recover structural patterns from native provenance, never alias spelling.
    pub fn native_sort_pattern(&self, native: &ArcSort) -> Result<SortPattern, String> {
        fn visit(
            types: &TypeInfo,
            native: &ArcSort,
            active: &mut HashSet<String>,
        ) -> Result<SortPattern, String> {
            if active.len() >= 256 || !active.insert(native.name().into()) {
                return Err("cyclic or too-deep native sort provenance".into());
            }
            let registered = types
                .get_sort_by_name(native.name())
                .ok_or("unregistered native sort")?;
            if registered.is_eq_sort() != native.is_eq_sort()
                || registered.is_container_sort() != native.is_container_sort()
                || registered.value_type() != native.value_type()
            {
                return Err("native sort disagrees with its registered representation".into());
            }
            let pattern = if native.is_eq_sort() {
                SortPattern::Eq(native.name().into())
            } else if let Ok(function) = native.clone().as_arc_any().downcast::<FunctionSort>() {
                SortPattern::Function {
                    inputs: function
                        .inputs()
                        .iter()
                        .map(|sort| visit(types, sort, active))
                        .collect::<Result<_, _>>()?,
                    output: Box::new(visit(types, &function.output(), active)?),
                }
            } else if let Some((family, parameters)) = types.builtin_sorts.get(native.name()) {
                SortPattern::Family {
                    name: (*family).into(),
                    args: parameters
                        .iter()
                        .map(|sort| visit(types, sort, active))
                        .collect::<Result<_, _>>()?,
                }
            } else if !native.is_container_sort() && native.value_type().is_some() {
                SortPattern::Family {
                    name: native.name().into(),
                    args: vec![],
                }
            } else {
                return Err(format!(
                    "sort {} lacks structural catalog provenance",
                    native.name()
                ));
            };
            active.remove(native.name());
            Ok(pattern)
        }
        visit(self, native, &mut HashSet::default())
    }

    /// Match complete native occurrences without invoking implementation code or
    /// inference predicates. Structural equality does not replace nominal roots.
    pub fn check_native_signature(
        &self,
        description: &BuiltinDescription,
        native: &[ArcSort],
    ) -> Result<(), String> {
        description.validate()?;
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
        // None marks an active pair. Completed heights keep shared sub-DAGs
        // from bypassing the depth bound on subsequent comparisons.
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
            pattern: &SortPattern,
            native: &ArcSort,
            types: &TypeInfo,
            substitution: &mut [Option<ArcSort>],
            compared: &mut HashMap<(String, String), Option<usize>>,
            depth: usize,
        ) -> Result<(), String> {
            if depth >= 256 {
                return Err("cyclic or too-deep native signature".into());
            }
            validate_native_sort(native, types)?;
            match pattern {
                SortPattern::Param(variable) => {
                    let slot = substitution
                        .get_mut(*variable)
                        .ok_or("unbound signature parameter")?;
                    if slot
                        .as_ref()
                        .is_some_and(|old| same_sort(old, native, types, compared, 0).is_none())
                    {
                        return Err("inconsistent builtin substitution".into());
                    }
                    *slot = Some(native.clone());
                }
                SortPattern::Eq(name) if native.is_eq_sort() && native.name() == name => (),
                SortPattern::Family { name, args }
                    if args.is_empty()
                        && !native.is_eq_sort()
                        && !native.is_container_sort()
                        && native.name() == name => {}
                SortPattern::Family { name, args } => {
                    let (family, parameters) = types
                        .builtin_sorts
                        .get(native.name())
                        .ok_or("native signature lacks family provenance")?;
                    if *family != name || parameters.len() != args.len() {
                        return Err("builtin argument/result sort mismatch".into());
                    }
                    for (pattern, actual) in args.iter().zip(parameters) {
                        unify(pattern, actual, types, substitution, compared, depth + 1)?;
                    }
                }
                SortPattern::Function { inputs, output } => {
                    let function = native
                        .clone()
                        .as_arc_any()
                        .downcast::<FunctionSort>()
                        .map_err(|_| "native signature requires a function sort")?;
                    if inputs.len() != function.inputs().len() {
                        return Err("builtin function arity mismatch".into());
                    }
                    for (pattern, actual) in inputs.iter().zip(function.inputs()) {
                        unify(pattern, actual, types, substitution, compared, depth + 1)?;
                    }
                    unify(
                        output,
                        &function.output(),
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
        for root in native {
            validate_native_sort(root, self)?;
        }
        let mut compared = HashMap::default();
        match &description.typing {
            BuiltinTyping::Signature(signature) => {
                let inputs = native
                    .len()
                    .checked_sub(1)
                    .ok_or("missing builtin result")?;
                if inputs < signature.fixed.len()
                    || if signature.tail.is_empty() {
                        inputs != signature.fixed.len()
                    } else {
                        !(inputs - signature.fixed.len()).is_multiple_of(signature.tail.len())
                    }
                {
                    return Err("builtin arity mismatch".into());
                }
                let patterns = signature
                    .fixed
                    .iter()
                    .map(|arg| &arg.sort)
                    .chain(
                        signature
                            .tail
                            .iter()
                            .cycle()
                            .take(inputs - signature.fixed.len())
                            .map(|arg| &arg.sort),
                    )
                    .chain([&signature.output]);
                let mut substitution = vec![None; signature.params.len()];
                for (pattern, actual) in patterns.zip(native) {
                    unify(pattern, actual, self, &mut substitution, &mut compared, 0)?;
                }
                if substitution.iter().any(Option::is_none) {
                    return Err("undetermined builtin type parameter".into());
                }
                if signature
                    .params
                    .iter()
                    .zip(&substitution)
                    .any(|(parameter, sort)| {
                        parameter.equality && !sort.as_ref().unwrap().is_eq_sort()
                    })
                {
                    return Err("builtin type parameter requires an equality sort".into());
                }
            }
            BuiltinTyping::FunctionApplication => {
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
            BuiltinTyping::FunctionConstructor => {
                if native.len() < 2
                    || native[0].name() != "String"
                    || !native
                        .last()
                        .unwrap()
                        .clone()
                        .as_arc_any()
                        .is::<FunctionSort>()
                {
                    return Err(
                        "function construction requires a String target and native function result"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Preserve every nominal root in one implementation's complete scheme.
    pub fn instantiate_builtin_signature(
        &self,
        key: &str,
        inputs: &[ArcSort],
        tail: &[ArcSort],
        output: &ArcSort,
    ) -> Result<BuiltinInstance, String> {
        let description = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin description")?
            .clone();
        if !self.builtin_owners.contains_key(key) {
            return Err("builtin description is not registry-owned".into());
        }
        let BuiltinTyping::Signature(signature) = &description.typing else {
            return Err("complete signature binding requires a signature description".into());
        };
        if inputs.len() != signature.fixed.len() || tail.len() != signature.tail.len() {
            return Err("builtin implementation fixed/tail arity mismatch".into());
        }
        let sorts = inputs
            .iter()
            .chain(tail)
            .chain([output])
            .cloned()
            .collect::<Vec<_>>();
        self.check_native_signature(&description, &sorts)?;
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
            description,
            dispatch_name,
            binding: InstanceBinding::Exact {
                sorts,
                fixed: inputs.len(),
                tail: tail.len(),
            },
        })
    }

    /// Retain the existing open native predicate under its registered scheme.
    pub fn instantiate_builtin_open(&self, key: &str) -> Result<BuiltinInstance, String> {
        let description = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin description")?
            .clone();
        if !self.builtin_owners.contains_key(key) {
            return Err("builtin description is not registry-owned".into());
        }
        Ok(BuiltinInstance {
            description,
            dispatch_name: format!("__egglog_open_{}_{}", key.len(), key),
            binding: InstanceBinding::Open,
        })
    }

    pub fn instantiate_builtin(
        &self,
        key: &str,
        native: &ArcSort,
    ) -> Result<BuiltinInstance, String> {
        let description = self
            .builtin_definitions
            .get(key)
            .ok_or("unknown builtin description")?
            .clone();
        if matches!(description.typing, BuiltinTyping::FunctionApplication) {
            let function = native
                .clone()
                .as_arc_any()
                .downcast::<FunctionSort>()
                .map_err(|_| "application instance requires a native function sort")?;
            if self.builtin_owners.get(key).copied() != Some("UnstableFn") {
                return Err("application description belongs to another provider".into());
            }
            return Ok(BuiltinInstance {
                description,
                dispatch_name: format!("__egglog_instance_{}_{}_{}", key.len(), key, native.name()),
                binding: InstanceBinding::Application(function),
            });
        }
        let (family, parameters) = self
            .builtin_sorts
            .get(native.name())
            .ok_or("unregistered family instance")?;
        if self.builtin_owners.get(key) != Some(family) {
            return Err("builtin description belongs to another family".into());
        }
        let BuiltinTyping::Signature(signature) = &description.typing else {
            return Err("native special form requires an open instance".into());
        };
        if signature
            .params
            .iter()
            .enumerate()
            .any(|(index, parameter)| {
                parameter.equality && parameters.get(index).is_none_or(|sort| !sort.is_eq_sort())
            })
        {
            return Err("builtin type parameter requires an equality sort".into());
        }
        fn resolve(
            pattern: &SortPattern,
            types: &TypeInfo,
            family: &str,
            parameters: &[ArcSort],
            native: &ArcSort,
            depth: usize,
        ) -> Result<NativeSortBinding, String> {
            if depth >= 256 {
                return Err("too-deep native signature pattern".into());
            }
            match pattern {
                SortPattern::Param(index) => Ok(NativeSortBinding::Exact(
                    parameters
                        .get(*index)
                        .ok_or("unbound type parameter")?
                        .clone(),
                )),
                SortPattern::Eq(name) => {
                    let sort = types
                        .get_sort_by_name(name)
                        .filter(|sort| sort.is_eq_sort())
                        .ok_or("unknown equality signature sort")?;
                    Ok(NativeSortBinding::Exact(sort.clone()))
                }
                SortPattern::Family { name, args } if args.is_empty() => {
                    let sort = types
                        .get_sort_by_name(name)
                        .filter(|sort| !sort.is_eq_sort() && !sort.is_container_sort())
                        .ok_or("unknown scalar signature sort")?;
                    Ok(NativeSortBinding::Exact(sort.clone()))
                }
                SortPattern::Family { name, args } => {
                    let args = args
                        .iter()
                        .map(|pattern| {
                            resolve(pattern, types, family, parameters, native, depth + 1)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if name == family {
                        if args.len() != parameters.len() || !args.iter().zip(parameters).all(|(arg, parameter)| matches!(arg, NativeSortBinding::Exact(sort) if sort.name() == parameter.name())) {
                            return Err("signature family application differs from its instance".into());
                        }
                        Ok(NativeSortBinding::Exact(native.clone()))
                    } else {
                        let carrier = types
                            .mksorts
                            .get(name)
                            .and_then(|provider| provider.container_type)
                            .ok_or_else(|| {
                                format!(
                                    "foreign family {name} has no native container representation"
                                )
                            })?;
                        Ok(NativeSortBinding::Family {
                            name: name.clone(),
                            args,
                            carrier,
                        })
                    }
                }
                SortPattern::Function { .. } => {
                    Err("function patterns require complete native occurrence binding".into())
                }
            }
        }
        let sorts = signature
            .fixed
            .iter()
            .chain(&signature.tail)
            .map(|arg| &arg.sort)
            .chain([&signature.output])
            .map(|pattern| resolve(pattern, self, family, parameters, native, 0))
            .collect::<Result<_, _>>()?;
        Ok(BuiltinInstance {
            description,
            dispatch_name: format!("__egglog_instance_{}_{}_{}", key.len(), key, native.name()),
            binding: InstanceBinding::Owner(sorts),
        })
    }

    /// Return an existing exact registration, including its runtime contexts.
    pub fn resolve_builtin(
        &self,
        key: &str,
        native: &[ArcSort],
    ) -> Result<PrimitiveWithId, String> {
        let description = self.builtin_description(key)?;
        self.check_native_signature(description, native)?;
        let candidates = self
            .builtin_primitives
            .get(key)
            .into_iter()
            .flatten()
            .filter(|primitive| primitive.accept(native, self))
            .collect::<Vec<_>>();
        if candidates.len() != 1 {
            return Err(format!(
                "builtin {key} has {} matching native instances",
                candidates.len()
            ));
        }
        Ok(candidates[0].clone())
    }

    /// Assert compatibility without replacing native authority. Binder and
    /// argument labels describe the signature, rather than its type identity.
    pub fn check_builtin_signature(&self, supplied: &BuiltinDescription) -> Result<(), String> {
        supplied.validate()?;
        let expected = self.builtin_description(&supplied.key)?;
        let mut actual = supplied.typing.clone();
        let mut wanted = expected.typing.clone();
        for typing in [&mut actual, &mut wanted] {
            if let BuiltinTyping::Signature(signature) = typing {
                for param in &mut signature.params {
                    param.name.clear();
                }
                for arg in signature.fixed.iter_mut().chain(&mut signature.tail) {
                    arg.name.clear();
                }
            }
        }
        if actual != wanted {
            return Err(format!("conflicting builtin signature {}", supplied.key));
        }
        Ok(())
    }

    pub fn builtin_catalog(&self) -> Result<BuiltinCatalog, String> {
        if !self.builtin_errors.is_empty() {
            return Err(self.builtin_errors.join("; "));
        }
        let mut catalog = BuiltinCatalog {
            descriptions: self.builtin_definitions.values().cloned().collect(),
            families: self
                .builtin_families
                .values()
                .map(|family| family.as_ref().clone())
                .collect(),
            undescribed_primitives: self
                .primitives
                .values()
                .flatten()
                .filter(|registered| {
                    registered.primitive.builtin_description().is_none()
                        && registered.primitive.requires_catalog_definition()
                })
                .map(|registered| registered.primitive.name().to_owned())
                .collect(),
            undescribed_families: self
                .mksorts
                .iter()
                .filter(|(name, provider)| {
                    provider.requires_catalog_family && !self.builtin_families.contains_key(*name)
                })
                .map(|(name, _)| name.clone())
                .collect(),
            undescribed_family_primitives: self.builtin_family_gaps.clone(),
            undescribed_sorts: self
                .sorts
                .values()
                .filter(|sort| {
                    !sort.is_eq_sort()
                        && !(*sort).clone().as_arc_any().is::<FunctionSort>()
                        && !self.builtin_sorts.contains_key(sort.name())
                        && !self.builtin_families.contains_key(sort.name())
                })
                .map(|sort| sort.name().to_owned())
                .collect(),
        };
        catalog
            .descriptions
            .sort_by(|left, right| left.key.cmp(&right.key));
        catalog
            .families
            .sort_by(|left, right| left.name.cmp(&right.name));
        catalog.undescribed_primitives.sort();
        catalog.undescribed_families.sort();
        catalog.undescribed_family_primitives.sort();
        catalog.undescribed_sorts.sort();
        Ok(catalog)
    }

    /// Resolve the same per-context native candidates used by function values.
    /// Optional runtime identity rejects stale or differently selected targets.
    pub fn resolve_function_target(
        &self,
        name: &str,
        inputs: &[ArcSort],
        output: &ArcSort,
        resolved: Option<&ResolvedFunctionId>,
    ) -> Result<NativeFunctionTarget, String> {
        if let Some(function) = self.get_func_type(name) {
            if !function
                .input
                .iter()
                .chain([&function.output])
                .map(|sort| sort.name())
                .eq(inputs.iter().chain([output]).map(|sort| sort.name()))
            {
                return Err("function target signature mismatch".into());
            }
            if matches!(resolved, Some(ResolvedFunctionId::Primitive { .. })) {
                return Err("function target runtime kind mismatch".into());
            }
            return Ok(NativeFunctionTarget::Function(name.into()));
        }
        let signature = inputs.iter().chain([output]).cloned().collect::<Vec<_>>();
        let candidates = self
            .get_prims(name)
            .ok_or("unknown function target")?
            .iter()
            .filter(|primitive| primitive.accept(&signature, self))
            .collect::<Vec<_>>();
        let mut selected: Vec<PrimitiveWithId> = vec![];
        for context in [Context::Pure, Context::Read, Context::Write, Context::Full] {
            let mut available = candidates
                .iter()
                .filter(|primitive| primitive.context_ids[context].is_some());
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
            if !selected
                .iter()
                .any(|old| old.context_ids == primitive.context_ids)
            {
                selected.push((*primitive).clone());
            }
        }
        if selected.is_empty() {
            return Err(format!("unresolved function target {name}"));
        }
        Ok(NativeFunctionTarget::Primitive(selected))
    }
}

impl EGraph {
    /// Select source sort information captured before proof instrumentation.
    pub fn authoring_type_info(&self) -> &TypeInfo {
        self.proof_state
            .original_typechecking
            .as_ref()
            .map_or(&self.type_info, |original| &original.type_info)
    }

    /// Resolve a function value from its retained runtime IDs and per-value
    /// capture sorts, without extracting text or invoking the target.
    pub fn resolve_function_value_target(
        &self,
        sort: &ArcSort,
        value: Value,
    ) -> Result<NativeFunctionTarget, String> {
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
        self.authoring_type_info().resolve_function_target(
            &value.2,
            &inputs,
            &function.output(),
            Some(&value.0),
        )
    }
}
