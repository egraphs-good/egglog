//! Type inference for terms in proofs.
//!
//! This module provides type inference for terms in the proof checker,
//! allowing us to validate that proofs are well-typed and identify
//! the correct primitive overloads.

use crate::ast::{Command, GenericCommand};
use crate::{ArcSort, TypeInfo};
use egglog_ast::generic_ast::Literal;
use egglog_bridge::termdag::{Term, TermDag, TermId};
use indexmap::IndexMap;

/// A type inference context that tracks the types of terms.
pub struct TypeInferenceContext<'a> {
    termdag: &'a TermDag,
    type_info: &'a TypeInfo,
    program: &'a [Command],
    /// Inferred types for each term
    types: IndexMap<TermId, ArcSort>,
}

impl<'a> TypeInferenceContext<'a> {
    /// Create a new type inference context.
    pub fn new(termdag: &'a TermDag, type_info: &'a TypeInfo, program: &'a [Command]) -> Self {
        Self {
            termdag,
            type_info,
            program,
            types: IndexMap::new(),
        }
    }

    /// Infer the type of a term.
    pub fn infer_type(&mut self, term_id: TermId) -> Option<ArcSort> {
        // Check cache
        if let Some(ty) = self.types.get(&term_id) {
            return Some(ty.clone());
        }

        // Infer based on term structure
        let ty = match self.termdag.get(term_id) {
            Term::Lit(lit) => self.infer_literal_type(lit),
            Term::Var(_name) => {
                // Variables should be bound in the proof context
                // For now, we can't infer their types without more context
                None
            }
            Term::App(func, args) => self.infer_app_type(func, args),
            Term::UnknownLit => None,
        }?;

        // Cache the result
        self.types.insert(term_id, ty.clone());
        Some(ty)
    }

    /// Infer the type of a literal.
    fn infer_literal_type(&self, lit: &Literal) -> Option<ArcSort> {
        // Map literal types to their corresponding sorts
        match lit {
            Literal::Int(_) => self.type_info.get_sort_by_name("i64").cloned(),
            Literal::Float(_) => self.type_info.get_sort_by_name("f64").cloned(),
            Literal::String(_) => self.type_info.get_sort_by_name("String").cloned(),
            Literal::Bool(_) => self.type_info.get_sort_by_name("bool").cloned(),
            Literal::Unit => self.type_info.get_sort_by_name("Unit").cloned(),
        }
    }

    /// Infer the type of a function application.
    fn infer_app_type(&mut self, func: &str, args: &[TermId]) -> Option<ArcSort> {
        // Terms in proofs only contain constructors and functions, not primitives.
        // Primitives are evaluated during execution and their results are stored as literals.

        // Look up the function/constructor in TypeInfo
        if let Some(func_type) = self.type_info.get_func_type(func) {
            // Verify that the argument count matches
            if args.len() == func_type.input.len() {
                // Return the output type of this function/constructor
                return Some(func_type.output.clone());
            }
        }

        // Try to find the function in the program
        for cmd in self.program {
            if let GenericCommand::Function { name, .. } = cmd {
                if name == func {
                    // Found the function - get its output type from TypeInfo
                    return self
                        .type_info
                        .get_func_type(func)
                        .map(|ft| ft.output.clone());
                }
            }
        }

        None
    }

    /// Find the specific primitive that matches the given application.
    #[allow(dead_code)] // May be used in the future
    pub fn find_matching_primitive(&mut self, func: &str, args: &[TermId]) -> Option<usize> {
        if let Some(prims) = self.type_info.get_prims(func) {
            // Infer argument types
            let arg_types: Vec<Option<ArcSort>> =
                args.iter().map(|&arg| self.infer_type(arg)).collect();

            // If all argument types are known, find the matching primitive
            if arg_types.iter().all(|t| t.is_some()) {
                let arg_types: Vec<ArcSort> = arg_types.into_iter().map(|t| t.unwrap()).collect();

                // Find the primitive with matching input types
                for (idx, prim) in prims.iter().enumerate() {
                    // Check if this primitive accepts these input types
                    if prim.accept(&arg_types, self.type_info) {
                        return Some(idx);
                    }
                }
            }
        }
        None
    }
}
