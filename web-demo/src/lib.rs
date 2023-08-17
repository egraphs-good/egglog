#![allow(clippy::unused_unit)] // weird clippy bug with wasm-bindgen
use wasm_bindgen::prelude::*;

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(getter_with_clone)]
pub struct Result {
    pub text: String,
    pub dot: String,
}

#[wasm_bindgen]
pub fn run_program(input: &str) -> Result {
    let mut egraph = egglog::EGraph::default();
    match egraph.parse_and_run_program(input) {
        Ok(outputs) => {
            log::info!("egg ok, {} outputs", outputs.len());
            let serialized = egraph.serialize_for_graphviz();
            Result {
                text: outputs.join("<br>"),
                dot: serialized.to_dot(),
            }
        }
        Err(e) => {
            log::info!("egg failed");
            Result {
                text: e.to_string(),
                dot: "".to_string(),
            }
        }
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    wasm_logger::init(Default::default());
    console_error_panic_hook::set_once();
    log::info!("wasm initialized");
}
