use crate::{ast::ResolvedVar, core::ResolvedCall};

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub use egglog_ast::generic_ast_helpers::INTERNAL_SYMBOL_PREFIX;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// Symbols have the form `<reserved prefix><hint>_<count>`, with a separate
/// counter for each complete hint shared across all input types. The first
/// symbol omits `_0` when the hint is nonempty and does not end in `_[0-9]+`,
/// except that an empty prefix never produces the wildcard `_`.
///
/// The final underscore uniquely separates a suffixed name's hint and counter.
/// Bare names cannot have that suffix, so they cannot collide with suffixed
/// names, and checked per-hint counters never reuse a count.
/// A reserved prefix prevents collisions with user symbols.
///
/// Generating a symbol after its hint's counter is exhausted panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    hint_to_count: HashMap<String, usize>,
    reserved_string: String,
    leave_off_zero: bool,
}

impl SymbolGen {
    /// Create a new symbol generator with the given reserved prefix.
    pub fn new(reserved_string: String) -> Self {
        Self {
            hint_to_count: HashMap::default(),
            reserved_string,
            leave_off_zero: true,
        }
    }

    /// By default, the first symbol generated with a given hint omits `_0`
    /// when doing so is unambiguous. Set this to `true` to always include it.
    /// Changing this option does not reset any hint's counter.
    pub fn include_zero(&mut self, include: bool) {
        self.leave_off_zero = !include;
    }

    /// Check if this symbol generator has been used to generate any symbols.
    pub fn has_been_used(&self) -> bool {
        !self.hint_to_count.is_empty()
    }

    /// Get the reserved prefix used by this symbol generator.
    pub fn reserved_prefix(&self) -> &str {
        &self.reserved_string
    }

    /// Check if the given symbol is reserved (i.e., starts with the reserved prefix).
    pub fn is_reserved(&self, symbol: &str) -> bool {
        !self.reserved_string.is_empty() && symbol.starts_with(&self.reserved_string)
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head: ?Sized, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<str, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &str) -> String {
        let entry = self.hint_to_count.entry_ref(name_hint).or_insert(0);
        let count = *entry;
        *entry = count
            .checked_add(1)
            .expect("fresh symbol counter exhausted");
        let omit_zero = self.leave_off_zero
            && count == 0
            && !name_hint.is_empty()
            && !(self.reserved_string.is_empty() && name_hint == "_")
            && !name_hint.rsplit_once('_').is_some_and(|(_, suffix)| {
                !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
            });
        let mut buffer = itoa::Buffer::new();
        let digits = if omit_zero { "" } else { buffer.format(count) };
        let mut name = String::with_capacity(
            self.reserved_string.len() + name_hint.len() + digits.len() + usize::from(!omit_zero),
        );
        name.push_str(&self.reserved_string);
        name.push_str(name_hint);
        if !omit_zero {
            name.push('_');
            name.push_str(digits);
        }
        name
    }
}

impl FreshGen<String, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &String) -> String {
        self.fresh(name_hint.as_str())
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for SymbolGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        ResolvedVar {
            name: self.fresh(name_hint.name()),
            sort: name_hint.output().clone(),
            // fresh variables are never global references, since globals
            // are desugared away by `remove_globals`
            is_global_ref: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context, EGraph,
        ast::{FunctionSubtype, Span},
        prelude::BaseSort,
        sort::I64Sort,
        typechecking::FuncType,
    };
    use std::{panic::AssertUnwindSafe, sync::Arc};

    #[test]
    fn symbol_gen_numbers_each_hint_and_guards_bare_names() {
        let mut generator = SymbolGen::new("@".into());
        for (hint, expected) in [
            ("Proof", "@Proof"),
            ("v", "@v"),
            ("Proof", "@Proof_1"),
            ("v", "@v_1"),
            ("x1", "@x1"),
            ("x_1", "@x_1_0"),
            ("", "@_0"),
            ("λ_12", "@λ_12_0"),
            ("λ_١٢", "@λ_١٢"),
        ] {
            assert_eq!(generator.fresh(hint), expected);
        }
        let mut generator = SymbolGen::new(String::new());
        assert_eq!(generator.fresh("_"), "__0");
        assert_eq!(generator.fresh("__"), "__");
    }

    #[test]
    fn symbol_gen_is_unique_across_hints_and_input_types() {
        let egraph = EGraph::default();
        let primitive = ResolvedCall::from_resolution(
            "+",
            &[
                I64Sort.to_arcsort(),
                I64Sort.to_arcsort(),
                I64Sort.to_arcsort(),
            ],
            &egraph.type_info,
            Context::Pure,
            &Span::Panic,
        )
        .unwrap();
        for prefix in ["", "@", "λ_"] {
            let mut generator = SymbolGen::new(prefix.into());
            assert!(!generator.has_been_used());
            assert_eq!(generator.reserved_prefix(), prefix);
            assert_eq!(
                generator.is_reserved(&format!("{prefix}anything")),
                !prefix.is_empty()
            );
            assert!(!generator.is_reserved("anything"));

            let mut names = HashSet::default();
            for hint in [
                "",
                "_",
                "__",
                "x",
                "x1",
                "x11",
                "x_0",
                "x_1",
                "x_8",
                "_0",
                "__0",
                "_0_",
                "λ",
                "λ_12",
                "λ_١٢",
                "🦀",
                "a_longer_hint_with_underscores_123",
            ] {
                let owned_hint = hint.to_owned();
                let call = ResolvedCall::Func(Arc::new(FuncType {
                    name: owned_hint.clone(),
                    subtype: FunctionSubtype::Custom,
                    input: vec![],
                    output: I64Sort.to_arcsort(),
                }));
                for i in 0..3 {
                    let from_str = generator.fresh(hint);
                    let from_string = generator.fresh(&owned_hint);
                    let from_call = generator.fresh(&call);
                    assert_eq!(from_string, format!("{prefix}{hint}_{}", 3 * i + 1));
                    assert_eq!(from_call.name, format!("{prefix}{hint}_{}", 3 * i + 2));
                    assert_eq!(from_call.sort.name(), I64Sort.name());
                    assert!(!from_call.is_global_ref);
                    for name in [from_str, from_string, from_call.name] {
                        assert!(!name.is_empty());
                        assert_ne!(name, "_");
                        assert!(names.insert(name));
                    }
                }
            }
            assert_eq!(generator.fresh("+"), format!("{prefix}+"));
            let from_primitive = generator.fresh(&primitive);
            assert_eq!(from_primitive.name, format!("{prefix}+_1"));
            assert_eq!(from_primitive.sort.name(), I64Sort.name());
            assert!(!from_primitive.is_global_ref);
            assert_eq!(generator.fresh(&"+".to_owned()), format!("{prefix}+_2"));
            assert!(generator.has_been_used());
        }
    }

    #[test]
    fn symbol_gen_include_zero_and_clone_preserve_counters() {
        let mut generator = SymbolGen::new("@".into());
        generator.include_zero(true);
        assert!(!generator.has_been_used());
        assert_eq!(generator.fresh("x"), "@x_0");
        let mut cloned = generator.clone();
        assert_eq!(generator, cloned);

        generator.include_zero(false);
        assert_eq!(generator.fresh("y"), "@y");
        assert_eq!(cloned.fresh("y"), "@y_0");
        cloned.include_zero(false);
        for generator in [&mut generator, &mut cloned] {
            assert_eq!(generator.fresh("x"), "@x_1");
            assert_eq!(generator.fresh("x_0"), "@x_0_0");
            assert_eq!(generator.fresh("x_1"), "@x_1_0");
            generator.include_zero(true);
            assert_eq!(generator.fresh("y"), "@y_1");
            generator.include_zero(false);
            assert_eq!(generator.fresh("z"), "@z");
            assert_eq!(generator.fresh(""), "@_0");
        }
        assert_eq!(generator, cloned);
    }

    #[test]
    fn symbol_gen_exhaustion_does_not_wrap_or_reuse_names() {
        let mut generator = SymbolGen::new("@".into());
        generator.hint_to_count.insert("x".into(), usize::MAX - 1);
        assert_eq!(generator.fresh("x"), format!("@x_{}", usize::MAX - 1));
        for _ in 0..2 {
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| generator.fresh("x"))).is_err());
            assert_eq!(generator.hint_to_count["x"], usize::MAX);
        }
        let mut cloned = generator.clone();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| cloned.fresh("x"))).is_err());
        assert_eq!(generator.fresh("y"), "@y");
        assert_eq!(generator.fresh("y"), "@y_1");
        assert_eq!(cloned.fresh("y"), "@y");
    }
}
