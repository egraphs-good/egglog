use crate::{ast::ResolvedVar, core::ResolvedCall};
use std::borrow::Cow;

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub use egglog_ast::generic_ast_helpers::INTERNAL_SYMBOL_PREFIX;

/// Generates `<reserved prefix><hint>`, adding `@<count>` after the first use.
/// Hints replace `@` with `_`, and empty or `_` hints become `v`.
/// Hints that normalize alike share a counter; panics when it is exhausted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    hint_to_count: HashMap<String, usize>,
    reserved_string: String,
}

impl SymbolGen {
    /// Create a new symbol generator with the given reserved prefix.
    pub fn new(reserved_string: String) -> Self {
        Self {
            hint_to_count: HashMap::default(),
            reserved_string,
        }
    }

    /// Check if this symbol generator has generated or reserved any symbols.
    pub fn has_been_used(&self) -> bool {
        !self.hint_to_count.is_empty()
    }

    /// Get the reserved prefix used by this symbol generator.
    pub fn reserved_prefix(&self) -> &str {
        &self.reserved_string
    }

    /// Check if the given symbol starts with the nonempty reserved prefix.
    /// This does not indicate whether the symbol has been reserved.
    pub fn is_reserved(&self, symbol: &str) -> bool {
        !self.reserved_string.is_empty() && symbol.starts_with(&self.reserved_string)
    }

    /// Prevent future fresh symbols from reusing this spelling or an earlier
    /// counter for the same normalized hint. Ignore spellings we cannot generate.
    /// Reserving an exhausted counter makes future generation for that hint panic.
    pub fn reserve(&mut self, symbol: &str) {
        let Some(name) = symbol.strip_prefix(&self.reserved_string) else {
            return;
        };
        let (hint, next) = match name.split_once('@') {
            Some((hint, suffix)) => {
                // Counters are positive canonical decimal integers: zero is omitted.
                if suffix.is_empty()
                    || suffix.starts_with('0')
                    || !suffix.bytes().all(|byte| byte.is_ascii_digit())
                {
                    return;
                }
                let Ok(count) = suffix.parse::<usize>() else {
                    return;
                };
                (hint, count.saturating_add(1))
            }
            None => (name, 1),
        };
        if hint.is_empty() || hint == "_" {
            return;
        }
        let entry = self.hint_to_count.entry_ref(hint).or_insert(0);
        *entry = (*entry).max(next);
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head: ?Sized, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<str, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &str) -> String {
        let mut name_hint = if name_hint.contains('@') {
            Cow::Owned(name_hint.replace('@', "_"))
        } else {
            Cow::Borrowed(name_hint)
        };
        if name_hint.is_empty() || name_hint == "_" {
            name_hint = Cow::Borrowed("v");
        }
        let entry = self
            .hint_to_count
            .entry_ref(name_hint.as_ref())
            .or_insert(0);
        let count = *entry;
        *entry = count
            .checked_add(1)
            .expect("fresh symbol counter exhausted");
        // Normalized hints contain no `@`, so suffixed and unsuffixed names cannot
        // collide. Hints that normalize alike share the same checked counter.
        let mut buffer = itoa::Buffer::new();
        let digits = buffer.format(count);
        let mut name =
            String::with_capacity(self.reserved_string.len() + name_hint.len() + 1 + digits.len());
        name.push_str(&self.reserved_string);
        name.push_str(&name_hint);
        if count != 0 {
            name.push('@');
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
    fn symbol_gen_numbers_normalized_hints_and_omits_zero() {
        let mut generator = SymbolGen::new("@".into());
        for (hint, expected) in [
            ("Proof", "@Proof"),
            ("v", "@v"),
            ("Proof", "@Proof@1"),
            ("v", "@v@1"),
            ("x1", "@x1"),
            ("x_1", "@x_1"),
            ("x", "@x"),
            ("x", "@x@1"),
            ("x@1", "@x_1@1"),
            ("x_1", "@x_1@2"),
            ("x@y", "@x_y"),
            ("x_y", "@x_y@1"),
            ("Ast@Proof", "@Ast_Proof"),
            ("Ast_Proof", "@Ast_Proof@1"),
            ("Ast@Proof@1", "@Ast_Proof_1"),
            ("Ast_Proof_1", "@Ast_Proof_1@1"),
            ("rewrite_var__", "@rewrite_var__"),
            ("rewrite_var__", "@rewrite_var__@1"),
            ("", "@v@2"),
            ("_", "@v@3"),
            ("@", "@v@4"),
            ("v", "@v@5"),
            ("λ_12", "@λ_12"),
            ("λ_١٢", "@λ_١٢"),
        ] {
            assert_eq!(generator.fresh(hint), expected);
        }
        let mut generator = SymbolGen::new(String::new());
        for (hint, expected) in [
            ("", "v"),
            ("_", "v@1"),
            ("@", "v@2"),
            ("v", "v@3"),
            ("__", "__"),
        ] {
            assert_eq!(generator.fresh(hint), expected);
        }
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
            assert!(!generator.is_reserved(&format!("before{prefix}after")));
            assert!(!generator.is_reserved(&format!("anything{prefix}")));
            assert!(!generator.is_reserved("anything"));

            let mut names = HashSet::default();
            for hint in [
                "",
                "_",
                "@",
                "v",
                "__",
                "x",
                "x1",
                "x11",
                "x1_",
                "x1__",
                "x1_0",
                "x_",
                "x__",
                "x_0",
                "x_1",
                "x@1",
                "x@y",
                "x_y",
                "x_8",
                "_0",
                "__0",
                "_0_",
                "λ",
                "λ_12",
                "λ_١٢",
                "🦀",
                "Ast@Proof",
                "Ast_Proof",
                "Ast@Proof@1",
                "Ast_Proof_1",
                "a_longer_hint_with_underscores_123",
            ] {
                let owned_hint = hint.to_owned();
                let call = ResolvedCall::Func(Arc::new(FuncType {
                    name: owned_hint.clone(),
                    subtype: FunctionSubtype::Custom,
                    input: vec![],
                    output: I64Sort.to_arcsort(),
                }));
                for _ in 0..3 {
                    let from_str = generator.fresh(hint);
                    let from_string = generator.fresh(&owned_hint);
                    let from_call = generator.fresh(&call);
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
            assert_eq!(from_primitive.name, format!("{prefix}+@1"));
            assert_eq!(from_primitive.sort.name(), I64Sort.name());
            assert!(!from_primitive.is_global_ref);
            assert_eq!(generator.fresh(&"+".to_owned()), format!("{prefix}+@2"));
            assert!(generator.has_been_used());
        }
    }

    #[test]
    fn symbol_gen_reservation_uses_normalized_hints_and_exact_prefixes() {
        for prefix in ["", "@", "λ_", "internal@"] {
            let mut generator = SymbolGen::new(prefix.into());
            for symbol in ["x_y", "x_y@4", "v", "λ@2"] {
                generator.reserve(&format!("{prefix}{symbol}"));
            }
            assert!(generator.has_been_used());
            assert_eq!(generator.fresh("x@y"), format!("{prefix}x_y@5"));
            assert_eq!(generator.fresh("x_y"), format!("{prefix}x_y@6"));
            for (hint, count) in [("", 1), ("_", 2), ("@", 3), ("v", 4)] {
                assert_eq!(generator.fresh(hint), format!("{prefix}v@{count}"));
            }
            assert_eq!(generator.fresh("λ"), format!("{prefix}λ@3"));
            assert_eq!(generator.fresh("unused"), format!("{prefix}unused"));
        }

        let mut generator = SymbolGen::new("λ_".into());
        for symbol in ["x", "@x", "beforeλ_x", "λx", "λ_"] {
            generator.reserve(symbol);
        }
        assert!(!generator.has_been_used());
        assert_eq!(generator.fresh("x"), "λ_x");
    }

    #[test]
    fn symbol_gen_reservation_is_monotone_and_idempotent() {
        let mut generator = SymbolGen::new("@".into());
        generator.reserve("@x");
        generator.reserve("@x");
        assert_eq!(generator.fresh("x"), "@x@1");
        assert_eq!(generator.fresh("x"), "@x@2");
        for symbol in ["@x@7", "@x@7", "@x@2", "@x", "@x@6"] {
            generator.reserve(symbol);
        }
        assert_eq!(generator.fresh("x"), "@x@8");
        generator.reserve("@x@8");
        generator.reserve("@x");
        assert_eq!(generator.fresh("x"), "@x@9");

        let mut cloned = generator.clone();
        cloned.reserve("@x@20");
        assert_eq!(generator.fresh("x"), "@x@10");
        assert_eq!(cloned.fresh("x"), "@x@21");
    }

    #[test]
    fn symbol_gen_reservation_ignores_unrepresentable_spellings() {
        for prefix in ["", "@", "internal@"] {
            let mut generator = SymbolGen::new(prefix.into());
            for symbol in [
                "", "_", "@1", "_@1", "x@", "x@0", "x@01", "x@+1", "x@-1", "x@١", "x@1@2", "x@y",
                "x@y@1",
            ] {
                generator.reserve(&format!("{prefix}{symbol}"));
            }
            generator.reserve(&format!("{prefix}x@{}0", usize::MAX));
            assert!(!generator.has_been_used());
            assert_eq!(generator.fresh("x"), format!("{prefix}x"));
            assert_eq!(generator.fresh("x@y"), format!("{prefix}x_y"));
            assert_eq!(generator.fresh("_"), format!("{prefix}v"));
        }
    }

    #[test]
    fn symbol_gen_reservation_cannot_wrap_exhausted_counters() {
        for reserved in [usize::MAX - 1, usize::MAX] {
            let mut generator = SymbolGen::new("@".into());
            generator.reserve(&format!("@x_y@{reserved}"));
            generator.reserve("@x_y");
            generator.reserve("@x_y@1");
            for hint in ["x@y", "x_y"] {
                assert!(
                    std::panic::catch_unwind(AssertUnwindSafe(|| generator.fresh(hint))).is_err()
                );
                assert_eq!(generator.hint_to_count["x_y"], usize::MAX);
            }
            assert_eq!(generator.fresh("y"), "@y");
        }
    }

    #[test]
    fn symbol_gen_clone_preserves_independent_counters() {
        let mut generator = SymbolGen::new("@".into());
        assert!(!generator.has_been_used());
        assert_eq!(generator.fresh("x"), "@x");
        let mut cloned = generator.clone();
        assert_eq!(generator, cloned);

        for generator in [&mut generator, &mut cloned] {
            assert_eq!(generator.fresh("x"), "@x@1");
            assert_eq!(generator.fresh("y"), "@y");
            assert_eq!(generator.fresh("x"), "@x@2");
            assert_eq!(generator.fresh("y"), "@y@1");
        }
        assert_eq!(generator, cloned);
    }

    #[test]
    fn symbol_gen_exhaustion_does_not_wrap_or_reuse_names() {
        let mut generator = SymbolGen::new("@".into());
        generator.hint_to_count.insert("x_y".into(), usize::MAX - 1);
        assert_eq!(generator.fresh("x@y"), format!("@x_y@{}", usize::MAX - 1));
        for hint in ["x@y", "x_y"] {
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| generator.fresh(hint))).is_err());
            assert_eq!(generator.hint_to_count["x_y"], usize::MAX);
        }
        let mut cloned = generator.clone();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| cloned.fresh("x@y"))).is_err());
        assert_eq!(generator.fresh("y"), "@y");
        assert_eq!(generator.fresh("y"), "@y@1");
        assert_eq!(cloned.fresh("y"), "@y");
    }
}
