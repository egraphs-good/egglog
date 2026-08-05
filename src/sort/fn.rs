//! Sort to represent functions as values.
//!
//! To declare the sort, you must specify the exact number of arguments and the sort of each, followed by the output sort:
//! `(sort IntToString (UnstableFn (i64) String))`
//!
//! To create a function value, use the `(unstable-fn "name" [<partial args>])` primitive and to apply it use the `(unstable-app function arg1 arg2 ...)` primitive.
//! The number of args must match the number of arguments in the function sort.
//!
//! Each value is interned as a sequence containing an opaque resolved-function
//! descriptor, a self-describing rebuild mask, and the partially applied
//! arguments. The mask lets congruence closure rebuild function keys without
//! resolving their descriptors or reconstructing Rust container values.
use std::any::TypeId;
use std::sync::Mutex;

use crate::exec_state::Internal;
use crate::numeric_id::NumericId;
use enum_map::EnumMap;

use super::*;

#[derive(Clone, Debug)]
pub struct FunctionContainer(
    pub ResolvedFunctionId,
    pub Vec<(ArcSort, Value)>,
    pub String,
    /// Pre-registered panic id used by `FunctionContainer::apply`
    /// on capability mismatch (see [`ResolvedFunction::panic_id`]).
    /// Excluded from equality/hash — two function values that differ
    /// only in their panic id are still the same function value.
    pub ExternalFunctionId,
);

const FUNCTION_REBUILD_MASK_BITS: usize = 31;
const FUNCTION_REBUILD_MASK: u32 = (1 << FUNCTION_REBUILD_MASK_BITS) - 1;

/// A validated view over an `UnstableFn` sequence key.
///
/// The key is `[descriptor, argc, mask..., args...]`. `argc` determines both
/// the number of 31-bit rebuild-mask words and the number of trailing args, so
/// parsing never needs to resolve the descriptor through `BaseValues`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FunctionSequence<'a> {
    descriptor: Value,
    masks: &'a [Value],
    args: &'a [Value],
}

/// A resolved `UnstableFn` plus an owned copy of its partial arguments.
///
/// Higher-order container primitives prepare this once before their element
/// loop, avoiding a descriptor lookup and sequence copy for every callback.
#[derive(Clone, Debug)]
pub struct PreparedFunction {
    resolved: ResolvedFunction,
    partial_args: Vec<Value>,
}

impl PreparedFunction {
    pub(crate) fn new(resolved: ResolvedFunction, partial_args: Vec<Value>) -> Self {
        Self {
            resolved,
            partial_args,
        }
    }

    pub(crate) fn apply<'a, 'db>(
        &self,
        state: &mut crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value>
    where
        'db: 'a,
    {
        let mut combined = Vec::with_capacity(self.partial_args.len() + args.len());
        combined.extend_from_slice(&self.partial_args);
        combined.extend_from_slice(args);
        self.resolved.apply(state, &combined)
    }
}

impl<'a> FunctionSequence<'a> {
    pub(crate) fn parse(sequence: &'a [Value]) -> Self {
        let (&descriptor, rest) = sequence
            .split_first()
            .expect("serialized FunctionContainer must include its descriptor");
        let (&argc, rest) = rest
            .split_first()
            .expect("serialized FunctionContainer must include its argument count");
        let argc = argc.index();
        assert!(
            argc <= FUNCTION_REBUILD_MASK as usize,
            "serialized FunctionContainer argument count exceeds its 31-bit encoding"
        );
        let mask_words = argc.div_ceil(FUNCTION_REBUILD_MASK_BITS);
        let expected = mask_words
            .checked_add(argc)
            .expect("serialized FunctionContainer length overflow");
        assert_eq!(
            rest.len(),
            expected,
            "serialized FunctionContainer length does not match its argument count"
        );
        let (masks, args) = rest.split_at(mask_words);
        for mask in masks {
            assert_eq!(
                mask.rep() & !FUNCTION_REBUILD_MASK,
                0,
                "serialized FunctionContainer mask uses its reserved high bit"
            );
        }
        if let Some(last) = masks.last()
            && !argc.is_multiple_of(FUNCTION_REBUILD_MASK_BITS)
        {
            let used = argc % FUNCTION_REBUILD_MASK_BITS;
            let used_mask = (1u32 << used) - 1;
            assert_eq!(
                last.rep() & !used_mask,
                0,
                "serialized FunctionContainer mask sets bits beyond its argument count"
            );
        }
        Self {
            descriptor,
            masks,
            args,
        }
    }

    pub(crate) fn descriptor(self) -> Value {
        self.descriptor
    }

    pub(crate) fn args(self) -> &'a [Value] {
        self.args
    }

    fn rebuilds_arg(self, index: usize) -> bool {
        debug_assert!(index < self.args.len());
        self.masks[index / FUNCTION_REBUILD_MASK_BITS].rep()
            & (1 << (index % FUNCTION_REBUILD_MASK_BITS))
            != 0
    }
}

fn encode_function_sequence(
    descriptor: Value,
    partial_arcsorts: &[ArcSort],
    args: &[Value],
    out: &mut Vec<Value>,
) {
    assert_eq!(
        partial_arcsorts.len(),
        args.len(),
        "FunctionContainer sort and argument counts must match"
    );
    assert!(
        args.len() <= FUNCTION_REBUILD_MASK as usize,
        "FunctionContainer argument count exceeds its 31-bit encoding"
    );
    out.push(descriptor);
    out.push(Value::from_usize(args.len()));
    let mask_words = args.len().div_ceil(FUNCTION_REBUILD_MASK_BITS);
    let mask_start = out.len();
    out.resize(mask_start + mask_words, Value::from_usize(0));
    for (index, sort) in partial_arcsorts.iter().enumerate() {
        if sort.is_eq_sort() || sort.is_eq_container_sort() {
            let word = mask_start + index / FUNCTION_REBUILD_MASK_BITS;
            let bit = 1 << (index % FUNCTION_REBUILD_MASK_BITS);
            out[word] = Value::from_usize(out[word].index() | bit);
        }
    }
    out.extend_from_slice(args);
}

// implement hash and equality based on values only not arcsorts, since
// arcsorts are not comparable and any two values that are equal must have the same sort

impl PartialEq for FunctionContainer {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            && self.1.iter().map(|(_, v)| *v).collect::<Vec<_>>()
                == other.1.iter().map(|(_, v)| *v).collect::<Vec<_>>()
            && self.2 == other.2
    }
}

impl Eq for FunctionContainer {}

impl Hash for FunctionContainer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        for (_, v) in &self.1 {
            v.hash(state);
        }
        self.2.hash(state);
    }
}

impl ContainerValue for FunctionContainer {
    fn rebuild_contents(&mut self, rebuilder: &dyn ValueRebuilder) -> bool {
        let mut changed = false;
        for (s, old) in &mut self.1 {
            if s.is_eq_sort() || s.is_eq_container_sort() {
                let new = rebuilder.rebuild_val(*old);
                changed |= *old != new;
                *old = new;
            }
        }
        changed
    }
    fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.1.iter().map(|(_, v)| v).copied()
    }
}

impl SequenceContainerValue for FunctionContainer {
    fn encode_sequence(&self, base_values: &BaseValues, out: &mut Vec<Value>) {
        let partial_arcsorts = self
            .1
            .iter()
            .map(|(sort, _)| sort.clone())
            .collect::<Vec<_>>();
        let args = self.1.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        let descriptor = base_values.get(ResolvedFunction {
            id: self.0.clone(),
            partial_arcsorts: partial_arcsorts.clone(),
            name: self.2.clone(),
            panic_id: self.3,
        });
        encode_function_sequence(descriptor, &partial_arcsorts, &args, out);
    }

    fn decode_sequence(sequence: &[Value], base_values: &BaseValues) -> Self {
        let sequence = FunctionSequence::parse(sequence);
        let ResolvedFunction {
            id,
            partial_arcsorts,
            name,
            panic_id,
        } = base_values.unwrap(sequence.descriptor());
        assert_eq!(
            partial_arcsorts.len(),
            sequence.args().len(),
            "FunctionContainer descriptor arity does not match its sequence"
        );
        for (index, sort) in partial_arcsorts.iter().enumerate() {
            assert_eq!(
                sequence.rebuilds_arg(index),
                sort.is_eq_sort() || sort.is_eq_container_sort(),
                "FunctionContainer rebuild mask does not match its descriptor"
            );
        }
        Self(
            id,
            partial_arcsorts
                .into_iter()
                .zip(sequence.args().iter().copied())
                .collect(),
            name,
            panic_id,
        )
    }

    fn sequence_values(sequence: &[Value]) -> &[Value] {
        FunctionSequence::parse(sequence);
        sequence
    }

    fn visit_sequence_values(sequence: &[Value], visitor: &mut dyn FnMut(Value)) {
        let sequence = FunctionSequence::parse(sequence);
        for (index, value) in sequence.args().iter().copied().enumerate() {
            if sequence.rebuilds_arg(index) {
                visitor(value);
            }
        }
    }

    fn rebuild_sequence(
        sequence: &[Value],
        _base_values: &BaseValues,
        rebuilder: &dyn ValueRebuilder,
        out: &mut Vec<Value>,
    ) -> bool {
        let parsed = FunctionSequence::parse(sequence);
        let prefix_len = sequence.len() - parsed.args().len();
        out.extend_from_slice(&sequence[..prefix_len]);
        let mut changed = false;
        for (index, old) in parsed.args().iter().copied().enumerate() {
            let new = if parsed.rebuilds_arg(index) {
                rebuilder.rebuild_val(old)
            } else {
                old
            };
            changed |= old != new;
            out.push(new);
        }
        if changed {
            true
        } else {
            out.clear();
            false
        }
    }
}
#[derive(Debug)]
pub struct FunctionSort {
    name: String,
    inputs: Vec<ArcSort>,
    output: ArcSort,
    // store all the arcsorts for functions that were added as partial args to this function sort
    // so that we can retrieve them during extraction
    partial_arcsorts: Arc<Mutex<Vec<ArcSort>>>,
}

impl FunctionSort {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn inputs(&self) -> &[ArcSort] {
        &self.inputs
    }

    pub fn output(&self) -> ArcSort {
        self.output.clone()
    }
}

impl Presort for FunctionSort {
    fn presort_name() -> &'static str {
        "UnstableFn"
    }

    fn reserved_primitives() -> Vec<&'static str> {
        vec!["unstable-fn", "unstable-app"]
    }

    fn make_sort(
        typeinfo: &mut TypeInfo,
        name: String,
        args: &[Expr],
        span: Span,
    ) -> Result<ArcSort, TypeError> {
        if let [inputs, Expr::Var(output_span, output)] = args {
            let output_sort = typeinfo
                .get_sort_by_name(output)
                .ok_or(TypeError::UndefinedSort(
                    output.clone(),
                    output_span.clone(),
                ))?;

            let input_sorts = match inputs {
                Expr::Call(_, first, rest_args) => {
                    let mut input_names = vec![first];
                    for arg in rest_args {
                        if let Expr::Var(_, arg) = arg {
                            input_names.push(arg);
                        } else {
                            return Err(TypeError::BadPresortArguments(
                                Self::presort_name().to_owned(),
                                arg.span(),
                            ));
                        }
                    }
                    input_names
                        .into_iter()
                        .map(|arg| {
                            typeinfo
                                .get_sort_by_name(arg)
                                .ok_or(TypeError::UndefinedSort(arg.clone(), output_span.clone()))
                                .cloned()
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                // an empty list of inputs args is parsed as a unit literal
                Expr::Lit(_, Literal::Unit) => vec![],
                _ => {
                    return Err(TypeError::BadPresortArguments(
                        Self::presort_name().to_owned(),
                        inputs.span(),
                    ));
                }
            };

            Ok(Arc::new(Self {
                name,
                inputs: input_sorts,
                output: output_sort.clone(),
                partial_arcsorts: Arc::new(Mutex::new(vec![])),
            }))
        } else {
            Err(TypeError::BadPresortArguments(
                Self::presort_name().to_owned(),
                span,
            ))
        }
    }
}

impl Sort for FunctionSort {
    fn name(&self) -> &str {
        &self.name
    }

    fn column_ty(&self, _backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Id
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend
            .base_values_mut()
            .register_type::<ResolvedFunction>();
        backend.register_sequence_container_ty::<FunctionContainer>();
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn is_container_sort(&self) -> bool {
        true
    }

    fn is_eq_container_sort(&self) -> bool {
        // The sequence's rebuildable children are its captured prefix, whose
        // sorts depend on the particular wrapped target and are not known when
        // an outer container fixes its rebuild/indexing policy. Remaining
        // inputs therefore cannot soundly answer this question: for example,
        // `(Math, i64) -> Math` captured as `UnstableFn (i64) Math` still
        // contains a rebuildable `Math`. Conservatively index every function
        // identity in outer containers; the function sequence's mask keeps its
        // own occurrence index and rebuild work precise.
        true
    }

    fn serialized_name(&self, container_values: &ContainerValues, value: Value) -> String {
        let val = container_values
            .get_val::<FunctionContainer>(value)
            .unwrap();
        val.2.clone()
    }

    fn inner_sorts(&self) -> Vec<ArcSort> {
        self.partial_arcsorts.lock().unwrap().clone()
    }

    fn inner_values(
        &self,
        container_values: &ContainerValues,
        value: Value,
    ) -> Vec<(ArcSort, Value)> {
        let val = container_values
            .get_val::<FunctionContainer>(value)
            .unwrap();
        val.1.clone()
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        eg.add_pure_primitive(
            Ctor {
                name: "unstable-fn".into(),
                function: self.clone(),
            },
            None,
        );
        eg.add_pure_primitive(
            Apply {
                name: "unstable-app".into(),
                function: self.clone(),
            },
            None,
        );

        register_vec_primitives_for_function(eg, self.clone());
        register_multiset_primitives_for_function(eg, self.clone());
    }

    fn value_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<FunctionContainer>())
    }

    fn reconstruct_termdag_container(
        &self,
        container_values: &ContainerValues,
        value: Value,
        termdag: &mut TermDag,
        mut element_terms: Vec<TermId>,
    ) -> TermId {
        let name = &container_values
            .get_val::<FunctionContainer>(value)
            .unwrap()
            .2;
        let head = termdag.lit(Literal::String(name.clone()));
        element_terms.insert(0, head);
        termdag.app("unstable-fn".to_owned(), element_terms)
    }
}

/// Takes a string and any number of partially applied args of any sort and returns a function
struct FunctionCTorTypeConstraint {
    name: String,
    function: Arc<FunctionSort>,
    span: Span,
}

impl TypeConstraint for FunctionCTorTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        // Must have at least one arg (plus the return value)
        if arguments.len() < 2 {
            return vec![constraint::impossible(
                constraint::ImpossibleConstraint::ArityMismatch {
                    atom: core::Atom {
                        span: self.span.clone(),
                        head: self.name.clone(),
                        args: arguments.to_vec(),
                    },
                    expected: 2,
                },
            )];
        }
        let output_sort_constraint: Box<dyn Constraint<_, ArcSort>> = constraint::assign(
            arguments[arguments.len() - 1].clone(),
            self.function.clone(),
        );
        // If first arg is a literal string and we know the name of the function and can use that to know what
        // types to expect
        if let AtomTerm::Literal(_, Literal::String(ref name)) = arguments[0] {
            // The arguments contains the return sort as well as the function name
            let n_partial_args = arguments.len() - 2;
            if let Some(func_type) = typeinfo.get_func_type(name) {
                // the number of partial args must match the number of inputs from the func type minus the number from
                // this function sort
                if self.function.inputs.len() + n_partial_args != func_type.input.len() {
                    return vec![constraint::impossible(
                        constraint::ImpossibleConstraint::ArityMismatch {
                            atom: core::Atom {
                                span: self.span.clone(),
                                head: self.name.clone(),
                                args: arguments.to_vec(),
                            },
                            expected: self.function.inputs.len() + func_type.input.len() + 1,
                        },
                    )];
                }
                // the output type and input types (starting after the partial args) must match between these functions
                let expected_output = self.function.output.clone();
                let expected_input = self.function.inputs.clone();
                let actual_output = func_type.output.clone();
                let actual_input: Vec<ArcSort> = func_type
                    .input
                    .iter()
                    .skip(n_partial_args)
                    .cloned()
                    .collect();
                if expected_output.name() != actual_output.name()
                    || expected_input
                        .iter()
                        .map(|s| s.name())
                        .ne(actual_input.iter().map(|s| s.name()))
                {
                    return vec![constraint::impossible(
                        constraint::ImpossibleConstraint::FunctionMismatch {
                            expected_output,
                            expected_input,
                            actual_output,
                            actual_input,
                        },
                    )];
                }
                // if they match, then just make sure the partial args match as well
                return func_type
                    .input
                    .iter()
                    .take(n_partial_args)
                    .zip(arguments.iter().skip(1))
                    .map(|(expected_sort, actual_term)| {
                        constraint::assign(actual_term.clone(), expected_sort.clone())
                    })
                    .chain(once(output_sort_constraint))
                    .collect();
            }

            if let Some(primitives) = typeinfo.get_prims(name) {
                // Primitive targets are checked by asking each overload whether
                // a full call would typecheck after stitching together:
                //
                //   explicit partial args from `(unstable-fn "name" ...)`
                //   + synthetic future args from the requested UnstableFn sort
                //   + one synthetic output term
                //
                // For example, `(unstable-fn "+" old)` as `UnstableFn (i64) i64`
                // checks each `+` overload as though it were called with
                // `(old, future_arg) -> future_output`. The i64 overload matches;
                // f64/string/etc. overloads become impossible constraints. If
                // `old` is omitted, the same sort only provides one future arg,
                // so no binary `+` overload has enough arguments to match.
                let mut primitive_constraints = Vec::with_capacity(primitives.len());
                for primitive in primitives {
                    let mut primitive_args = arguments[1..arguments.len() - 1].to_vec();
                    primitive_constraints.push(Vec::new());
                    let alternative_constraints = primitive_constraints.last_mut().unwrap();
                    for (index, sort) in self
                        .function
                        .inputs
                        .iter()
                        .chain(once(&self.function.output))
                        .enumerate()
                    {
                        let term = AtomTerm::Var(
                            self.span.clone(),
                            format!(
                                "__unstable_fn_target_{}_{}_arg_{index}",
                                name,
                                self.function.name()
                            ),
                        );
                        alternative_constraints
                            .push(constraint::assign(term.clone(), sort.clone()));
                        primitive_args.push(term);
                    }
                    alternative_constraints.extend(
                        primitive
                            .primitive
                            .get_type_constraints(&self.span)
                            .get(&primitive_args, typeinfo),
                    );
                }

                // No alternatives is defensive, one alternative is ordinary
                // non-overloaded primitive resolution, and multiple alternatives
                // are overloaded primitives such as `+`; the xor lets the type
                // solver pick exactly one viable overload.
                return match primitive_constraints.len() {
                    0 => vec![constraint::impossible(
                        constraint::ImpossibleConstraint::ArityMismatch {
                            atom: core::Atom {
                                span: self.span.clone(),
                                head: self.name.clone(),
                                args: arguments.to_vec(),
                            },
                            expected: n_partial_args + self.function.inputs.len() + 2,
                        },
                    )],
                    1 => once(output_sort_constraint)
                        .chain(primitive_constraints.pop().unwrap())
                        .collect(),
                    _ => vec![
                        output_sort_constraint,
                        constraint::xor(
                            primitive_constraints
                                .into_iter()
                                .map(constraint::and)
                                .collect(),
                        ),
                    ],
                };
            }
        }

        // Otherwise we just try assuming it's this function, we don't know if it is or not
        vec![
            constraint::assign(arguments[0].clone(), StringSort.to_arcsort()),
            output_sort_constraint,
        ]
    }
}

// (unstable-fn "name" [<arg1>, <arg2>, ...])
#[derive(Clone)]
struct Ctor {
    name: String,
    function: Arc<FunctionSort>,
}

// `Ctor` (`unstable-fn "name" [...]`) serializes the descriptor and partial
// arguments directly into the sequence table. Container interning is
// idempotent, so it's safe in every context; declaring `State = PureState`
// permits this primitive inside rule queries, actions, and global contexts.
impl Primitive for Ctor {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(FunctionCTorTypeConstraint {
            name: self.name.clone(),
            function: self.function.clone(),
            span: span.clone(),
        })
    }
}

impl PurePrim for Ctor {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let (rf, args) = args.split_first().unwrap();
        let resolved: ResolvedFunction = state.base_values().unwrap(*rf);
        self.function
            .partial_arcsorts
            .lock()
            .unwrap()
            .extend(resolved.partial_arcsorts.iter().cloned());
        let mut key = Vec::new();
        encode_function_sequence(*rf, &resolved.partial_arcsorts, args, &mut key);
        Some(state.register_container_sequence::<FunctionContainer>(&key))
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedFunction {
    pub id: ResolvedFunctionId,
    pub partial_arcsorts: Vec<ArcSort>,
    pub name: String,
    /// Pre-registered runtime-panic id used by `FunctionContainer::apply`
    /// when an `unstable-fn` value is applied in a context where its
    /// wrapped function isn't valid (e.g. constructor minting in a
    /// rule body without `:naive`). Calling this id writes a
    /// descriptive message to the egraph's panic side channel and
    /// triggers early stop, so `run_rules` returns an `Err` rather
    /// than the calling thread unwinding.
    pub panic_id: ExternalFunctionId,
}
// Implement equality and hash based on id, user-visible name, and ArcSort
// names. ArcSort trait objects are not directly comparable. The panic id is a
// build-site implementation detail and intentionally remains excluded.

impl PartialEq for ResolvedFunction {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.name == other.name
            && self
                .partial_arcsorts
                .iter()
                .map(|s| s.name())
                .collect::<Vec<_>>()
                == other
                    .partial_arcsorts
                    .iter()
                    .map(|s| s.name())
                    .collect::<Vec<_>>()
    }
}

impl Eq for ResolvedFunction {}

impl Hash for ResolvedFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.name.hash(state);
        for s in &self.partial_arcsorts {
            s.name().hash(state);
        }
    }
}

impl BaseValue for ResolvedFunction {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedFunctionId {
    /// Wraps a constructor-table lookup. Only admissible in
    /// write-capable contexts (`Write`/`Full`), where
    /// `FunctionContainer::apply` mints a fresh eclass via
    /// `lookup_or_insert`. In any read-only context (`Read`/`Pure`)
    /// it triggers the pre-registered runtime panic — a no-mint
    /// constructor would silently miss instead of producing the
    /// eclass the user asked for, so the call is rejected outright.
    Constructor(egglog_bridge::TableAction),
    /// Wraps a `(function …)` lookup — any non-constructor function,
    /// regardless of its `:merge` strategy. `FunctionContainer::apply`
    /// allows this only in DB-read-capable contexts (`Read`/`Full`);
    /// `Pure` and `Write` would be untracked seminaive reads.
    Function(egglog_bridge::TableAction),
    /// Wraps a primitive. Carries the unique exact-signature runtime
    /// id found for each context at build time. At dispatch time
    /// `FunctionContainer::apply` picks the id for the application
    /// context — so the runtime selection is independent of the
    /// build-site context, and an `unstable-fn` value may flow freely
    /// from one context to another.
    Primitive {
        context_ids: EnumMap<crate::Context, Option<ExternalFunctionId>>,
    },
}

// (unstable-app <function> [<arg1>, <arg2>, ...])
//
// Registered as a `PurePrim`; `FunctionContainer::apply` reads the
// runtime context to dispatch. Distinct `FunctionSort`s produce
// different signature keys, so `unstable-app` for `MathFn` stays a
// separate overload from `unstable-app` for `i64Fun`.

#[derive(Clone)]
struct Apply {
    name: String,
    function: Arc<FunctionSort>,
}

impl Primitive for Apply {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let mut sorts: Vec<ArcSort> = vec![self.function.clone()];
        sorts.extend(self.function.inputs.clone());
        sorts.push(self.function.output.clone());
        SimpleTypeConstraint::new(&self.name, sorts, span.clone()).into_box()
    }
}

impl PurePrim for Apply {
    fn apply<'a, 'db>(
        &self,
        mut state: crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value> {
        let (fc_val, args) = args.split_first().unwrap();
        state.apply_function_value(*fc_val, args)
    }
}

impl ResolvedFunction {
    pub(crate) fn apply<'a, 'db>(
        &self,
        state: &mut crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value>
    where
        'db: 'a,
    {
        apply_resolved_function(&self.id, self.panic_id, state, args)
    }
}

impl FunctionContainer {
    /// Apply the wrapped function. `state` is always a `PureState`
    /// (the type every primitive's `apply` receives). The surrounding
    /// context is stamped onto that state by the primitive wrapper, so
    /// callers do not pass a second copy of the same context.
    pub(crate) fn apply<'a, 'db>(
        &self,
        state: &mut crate::PureState<'a, 'db>,
        args: &[Value],
    ) -> Option<Value>
    where
        'db: 'a,
    {
        let args: Vec<_> = self.1.iter().map(|(_, x)| x).chain(args).copied().collect();
        apply_resolved_function(&self.0, self.3, state, &args)
    }
}

fn apply_resolved_function<'a, 'db>(
    id: &ResolvedFunctionId,
    panic_id: ExternalFunctionId,
    state: &mut crate::PureState<'a, 'db>,
    args: &[Value],
) -> Option<Value>
where
    'db: 'a,
{
    let ctx = state.ctx();
    let can_mint = matches!(ctx, crate::Context::Write | crate::Context::Full);
    let can_read = matches!(ctx, crate::Context::Read | crate::Context::Full);
    // On capability mismatch, trigger the egglog runtime panic
    // pre-registered at the `unstable-fn` build site (see
    // `BackendRule::prim`). The panic writes to the egraph's
    // panic side channel and triggers early stop, so `run_rules`
    // surfaces the misuse as an `Err`.
    let mismatch = |state: &mut crate::PureState<'a, 'db>| -> Option<Value> {
        state.call_external_func(panic_id, &[])
    };
    match id {
        ResolvedFunctionId::Constructor(action) => {
            if can_mint {
                action.lookup_or_insert(state.raw_exec_state(), args)
            } else {
                mismatch(state)
            }
        }
        ResolvedFunctionId::Function(action) => {
            if can_read {
                action.lookup(state.raw_exec_state(), args)
            } else {
                mismatch(state)
            }
        }
        ResolvedFunctionId::Primitive { context_ids } => {
            // Pick the runtime id whose context matches the
            // application ctx.
            match context_ids[ctx] {
                Some(id) => state.call_external_func(id, args),
                None => mismatch(state),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingRemap {
        calls: AtomicUsize,
        add: usize,
    }

    impl ValueRebuilder for CountingRemap {
        fn rebuild_val(&self, value: Value) -> Value {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Value::from_usize(value.index() + self.add)
        }
    }

    fn value(index: usize) -> Value {
        Value::from_usize(index)
    }

    #[test]
    fn sequence_masks_rebuildable_args_across_word_boundaries() {
        let plain = I64Sort.to_arcsort();
        let eq: ArcSort = Arc::new(EqSort {
            name: "MaskEq".to_owned(),
        });
        let mut sorts = vec![plain; 33];
        for index in [0, 30, 31, 32] {
            sorts[index] = eq.clone();
        }
        let args = (100..133).map(value).collect::<Vec<_>>();
        let descriptor = value(77);
        let mut encoded = Vec::new();
        encode_function_sequence(descriptor, &sorts, &args, &mut encoded);

        assert_eq!(encoded[0], descriptor);
        assert_eq!(encoded[1], value(33));
        assert_eq!(encoded[2], value((1 << 0) | (1 << 30)));
        assert_eq!(encoded[3], value((1 << 0) | (1 << 1)));
        assert_eq!(&encoded[4..], args);

        let mut visited = Vec::new();
        FunctionContainer::visit_sequence_values(&encoded, &mut |value| visited.push(value));
        assert_eq!(visited, [args[0], args[30], args[31], args[32]]);

        let remap = CountingRemap {
            calls: AtomicUsize::new(0),
            add: 1_000,
        };
        let mut rebuilt = Vec::new();
        assert!(FunctionContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &remap,
            &mut rebuilt,
        ));
        assert_eq!(remap.calls.load(Ordering::Relaxed), 4);
        assert_eq!(&rebuilt[..4], &encoded[..4]);
        for index in 0..args.len() {
            let expected = if [0, 30, 31, 32].contains(&index) {
                value(args[index].index() + 1_000)
            } else {
                args[index]
            };
            assert_eq!(rebuilt[4 + index], expected);
        }
    }

    #[test]
    fn sequence_zero_args_is_descriptor_and_count_only() {
        let descriptor = value(77);
        let mut encoded = Vec::new();
        encode_function_sequence(descriptor, &[], &[], &mut encoded);
        assert_eq!(encoded, [descriptor, value(0)]);

        let parsed = FunctionSequence::parse(&encoded);
        assert_eq!(parsed.descriptor(), descriptor);
        assert!(parsed.args().is_empty());

        let mut visited = Vec::new();
        FunctionContainer::visit_sequence_values(&encoded, &mut |value| visited.push(value));
        assert!(visited.is_empty());

        let remap = CountingRemap {
            calls: AtomicUsize::new(0),
            add: 1,
        };
        let mut rebuilt = Vec::new();
        assert!(!FunctionContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &remap,
            &mut rebuilt,
        ));
        assert_eq!(remap.calls.load(Ordering::Relaxed), 0);
        assert!(rebuilt.is_empty());
    }

    #[test]
    fn sequence_rebuild_does_not_resolve_descriptor() {
        // This descriptor is deliberately absent from the empty BaseValues.
        // Raw sequence rebuild must use only the packed mask.
        let encoded = [value(900), value(2), value(1), value(10), value(20)];
        let remap = CountingRemap {
            calls: AtomicUsize::new(0),
            add: 5,
        };
        let mut rebuilt = Vec::new();
        assert!(FunctionContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &remap,
            &mut rebuilt,
        ));
        assert_eq!(remap.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            rebuilt,
            [value(900), value(2), value(1), value(15), value(20)]
        );

        let unchanged = CountingRemap {
            calls: AtomicUsize::new(0),
            add: 0,
        };
        rebuilt.clear();
        assert!(!FunctionContainer::rebuild_sequence(
            &encoded,
            &BaseValues::default(),
            &unchanged,
            &mut rebuilt,
        ));
        assert_eq!(unchanged.calls.load(Ordering::Relaxed), 1);
        assert!(rebuilt.is_empty());
    }

    #[test]
    fn sequence_codec_round_trips_descriptor_and_mixed_args() {
        let mut base_values = BaseValues::default();
        base_values.register_type::<ResolvedFunction>();
        let eq: ArcSort = Arc::new(EqSort {
            name: "RoundTripEq".to_owned(),
        });
        let function = FunctionContainer(
            ResolvedFunctionId::Primitive {
                context_ids: EnumMap::default(),
            },
            vec![(I64Sort.to_arcsort(), value(10)), (eq, value(20))],
            "round-trip".to_owned(),
            ExternalFunctionId::new(7),
        );
        let mut encoded = Vec::new();
        function.encode_sequence(&base_values, &mut encoded);
        assert_eq!(encoded[1], value(2));
        assert_eq!(encoded[2], value(2));
        assert_eq!(
            FunctionContainer::decode_sequence(&encoded, &base_values),
            function
        );

        let mut visited = Vec::new();
        FunctionContainer::visit_sequence_values(&encoded, &mut |value| visited.push(value));
        assert_eq!(visited, [value(20)]);
    }

    #[test]
    #[should_panic(expected = "reserved high bit")]
    fn sequence_parser_rejects_mask_high_bit() {
        FunctionSequence::parse(&[value(1), value(1), Value::new(1 << 31), value(2)]);
    }

    #[test]
    #[should_panic(expected = "bits beyond its argument count")]
    fn sequence_parser_rejects_noncanonical_unused_bits() {
        FunctionSequence::parse(&[value(1), value(1), value(2), value(2)]);
    }

    #[test]
    #[should_panic(expected = "length does not match its argument count")]
    fn sequence_parser_rejects_trailing_values() {
        FunctionSequence::parse(&[value(1), value(0), value(2)]);
    }
}
