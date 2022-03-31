#![allow(clippy::unused_unit)] // weird clippy bug with wasm-bindgen
use wasm_bindgen::prelude::*;

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn run_program(input: &str) -> String {
    let mut egraph = egg_smol::EGraph::default();
    egraph.run_program(input);
    log::info!("egg ran program successfully");
    format!("hello {input}")
}

#[wasm_bindgen(start)]
pub fn start() {
    wasm_logger::init(Default::default());
    console_error_panic_hook::set_once();
    log::info!("wasm initialized");
}
