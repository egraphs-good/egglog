use egglog::EGraph;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run() -> usize {
    EGraph::default()
        .parse_and_run_program(None, "(datatype Math (Num i64) (Add Math Math))")
        .unwrap()
        .len()
}
