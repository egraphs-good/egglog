#![allow(clippy::unused_unit)] // weird clippy bug with wasm-bindgen
use log::{Level, Log, Metadata, Record};
use wasm_bindgen::prelude::*;
use web_sys::console;

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
            let serialized = egraph.serialize_for_graphviz(false);
            Result {
                text: outputs.join("<br>"),
                dot: serialized.to_dot(),
            }
        }
        Err(e) => Result {
            text: e.to_string(),
            dot: "".to_string(),
        },
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    init();
    console_error_panic_hook::set_once();
}

/// The log styles
struct Style {
    lvl_trace: String,
    lvl_debug: String,
    lvl_info: String,
    lvl_warn: String,
    lvl_error: String,
}

impl Style {
    fn new() -> Self {
        let base = String::from("color: white; padding: 0 3px; background:");
        Style {
            lvl_trace: format!("{} gray;", base),
            lvl_debug: format!("{} blue;", base),
            lvl_info: format!("{} green;", base),
            lvl_warn: format!("{} orange;", base),
            lvl_error: format!("{} darkred;", base),
        }
    }

    fn get_lvl_style(&self, lvl: Level) -> &str {
        match lvl {
            Level::Trace => &self.lvl_trace,
            Level::Debug => &self.lvl_debug,
            Level::Info => &self.lvl_info,
            Level::Warn => &self.lvl_warn,
            Level::Error => &self.lvl_error,
        }
    }
}

// This is inspired by wasm_logger
struct WebDemoLogger {
    style: Style,
}

impl Log for WebDemoLogger {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            let style = &self.style;
            let s = format!(
                "<span style=\"{}\">{}</span>\n{}\n",
                style.get_lvl_style(record.level()),
                record.level(),
                record.args(),
            );
            log(record.level().as_str(), &s);
        }
    }

    fn flush(&self) {}
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen]
    fn log(level: &str, s: &str);
}

pub fn init() {
    let max_level = Level::Debug;
    let wl = WebDemoLogger {
        style: Style::new(),
    };

    match log::set_boxed_logger(Box::new(wl)) {
        Ok(_) => log::set_max_level(max_level.to_level_filter()),
        Err(e) => console::error_1(&JsValue::from(e.to_string())),
    }
}
