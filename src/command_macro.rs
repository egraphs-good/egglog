//! Command-level macro system for egglog
//!
//! This module provides an API for external libraries to implement
//! command-level transformations, similar to procedural macros.

use crate::Error;
use crate::ast::*;
use crate::typechecking::TypeInfo;
use crate::util::SymbolGen;

/// A command macro that can transform commands during desugaring
pub trait CommandMacro: Send + Sync {
    /// Transform the command, potentially using type information.
    /// Returns the transformed commands. If the macro doesn't apply,
    /// it should return vec![command] unchanged.
    fn transform(
        &self,
        command: Command,
        symbol_gen: &mut SymbolGen,
        type_info: Option<&TypeInfo>,
    ) -> Result<Vec<Command>, Error>;
}

/// A registry of command macros
#[derive(Default, Clone)]
pub struct CommandMacroRegistry {
    macros: Vec<Arc<dyn CommandMacro>>,
}

use std::sync::Arc;

impl CommandMacroRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new command macro
    pub fn register(&mut self, macro_impl: Arc<dyn CommandMacro>) {
        self.macros.push(macro_impl);
    }

   /// Apply all matching macros to a command
   pub fn apply(
       &self,
       command: Command,
       symbol_gen: &mut SymbolGen,
       type_info: Option<&TypeInfo>,
   ) -> Result<Vec<Command>, Error> {
        // Apply the first macro (if any)
       for macro_impl in &self.macros {
            return macro_impl.transform(command, symbol_gen, type_info);
       }
       // No macro matched, return the command unchanged
       Ok(vec![command])
    }
}
