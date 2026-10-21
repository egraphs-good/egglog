use crate::{ast::ResolvedVar, core::ResolvedCall};

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub use egglog_ast::generic_ast_helpers::INTERNAL_SYMBOL_PREFIX;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// Names consist of the reserved prefix, hint, and an underscore-separated counter.
/// By default, zero is omitted unless the hint ends in `_` and ASCII digits.
/// Generating a symbol after its counter is exhausted panics.
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

    /// Include zero even when the hint does not already end in `_digits`.
    /// By default, an ordinary hint starts with "var" instead of "var_0".
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

        // A bare hint ending in `_digits` could equal another hint's numbered name.
        let bare = self.leave_off_zero && count == 0 && {
            let has_counter_suffix = name_hint.rsplit_once('_').is_some_and(|(_, suffix)| {
                !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
            });
            !has_counter_suffix
        };
        let mut buffer = itoa::Buffer::new();
        let suffix = if bare { "" } else { buffer.format(count) };
        let mut name = String::with_capacity(
            self.reserved_string.len()
                + name_hint.len()
                + usize::from(!suffix.is_empty())
                + suffix.len(),
        );
        name.push_str(&self.reserved_string);
        name.push_str(name_hint);
        if !suffix.is_empty() {
            name.push('_');
            name.push_str(suffix);
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
mod symbol_gen_tests {
    use super::*;

    #[test]
    fn counter_shaped_hints_keep_their_suffix() {
        assert_eq!(SymbolGen::new(String::new()).fresh("true_1"), "true_1_0");
        assert_eq!(SymbolGen::new(String::new()).fresh("false_1"), "false_1_0");
        assert_eq!(SymbolGen::new(String::new()).fresh("NaN_1"), "NaN_1_0");
        assert_eq!(SymbolGen::new(String::new()).fresh("_123"), "_123_0");
        assert_eq!(SymbolGen::new(String::new()).fresh("__1"), "__1_0");
    }

    #[test]
    fn numbered_hints_remain_distinct() {
        let mut generator = SymbolGen::new("@".to_owned());
        let names: Vec<_> = ["x", "x_1", "x_1_1", "x", "x_0"]
            .into_iter()
            .map(|hint| generator.fresh(hint))
            .collect();
        assert_eq!(names, ["@x", "@x_1_0", "@x_1_1_0", "@x_1", "@x_0_0"]);
    }

    #[test]
    fn original_collision_is_impossible() {
        let mut generator = SymbolGen::new("@".to_owned());
        let mut seen = HashSet::default();
        for _ in 0..25 {
            for hint in [
                "n414", "n4141", "n", "x", "x1", "x11", "x_1", "", "0", "_", "@x1", "λ1", "λ١",
            ] {
                assert!(seen.insert(generator.fresh(hint)));
            }
        }
    }

    #[test]
    #[should_panic(expected = "fresh symbol counter exhausted")]
    fn exhausted_counter_panics() {
        let mut generator = SymbolGen::new("@".to_owned());
        generator.hint_to_count.insert("x".to_owned(), usize::MAX);
        generator.fresh("x");
    }

    #[test]
    fn exhaustive_short_hints_are_unique_with_zero_toggles() {
        let mut hints = vec![String::new()];
        let mut last = vec![String::new()];
        for _ in 0..4 {
            let next: Vec<_> = last
                .iter()
                .flat_map(|p| ['a', '_', '0', '1', '@', 'λ', '١'].map(|c| format!("{p}{c}")))
                .collect();
            hints.extend(next.iter().cloned());
            last = next;
        }
        for prefix in ["", "@", "_", "12"] {
            let mut generator = SymbolGen::new(prefix.to_owned());
            let mut seen = HashSet::default();
            for round in 0..4 {
                generator.include_zero(round % 2 != 0);
                for hint in &hints {
                    assert!(seen.insert(generator.fresh(hint)), "{prefix:?} {hint:?}");
                }
            }
        }
    }
}
