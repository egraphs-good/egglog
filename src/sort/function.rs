//! Sort to represent functions as values.
//!
//! To declare the sort, you must specify the exact number of arguments and the sort of each, followed by the output sort:
//! `(sort IntToString (Function i64 String))`
//!
//! To create a function, used the `(function "name")` primitive and to apply it use the `(apply function arg1 arg2 ...)` primitive.
//! If the number supplied args is not enough, the function will be partially applied.
//!
//! The value is stored similar to the `vec` sort, as an index into a set, where each item in
//! the set is a (Symbol, Vec<Value>) pair. The Symbol is the function name, and the Vec<Value> is
//! the list of partially applied arguments.
use std::sync::Mutex;

use crate::{ast::Literal, constraint::AllEqualTypeConstraint};

use super::*;

type ValueFunction = (Symbol, Vec<Value>);

#[derive(Debug)]
pub struct FunctionSort {
    name: Symbol,

    functions: Mutex<IndexSet<ValueFunction>>,
}
