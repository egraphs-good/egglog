use crate::{CommandMacroRegistry, EGraph, RunReport, TypeInfo, term_encoding::EncodingState};

/// Generate a json report for the size of a serialized structu
/// By default, only uses serialize
/// Allow specalization to look into subfields

#[allow(dead_code)]
#[derive (Debug, Clone)]
pub struct SizeReport {
    name: String,
    size: usize,
    fields: Vec<(String, Box<SizeReport>)>,
}

fn up_to_two_decimals(a : usize, b : usize) -> String {
    let a100 = a * 100 / b;
    let high = a100 / 100;
    let low = a100 % 100;
    let low_str = 
        if low < 10 {
            "0".to_string() + &low.to_string()
        } else {
            low.to_string()
        };
    return high.to_string() + "." + &low_str;
}

fn pretty_print_nbytes(size: usize) -> String {
    if size < 200 {
        size.to_string() + "B"
    } else if size < 200 * 1024 {
        up_to_two_decimals(size, 1024) + "KB"
    } else if size < 200 * 1024 * 1024 {
        up_to_two_decimals(size, 1024 * 1024) + "MB"
    } else {
        up_to_two_decimals(size, 1024 * 1024 * 1024) + "GB"
    }
}

impl SizeReport {

    pub fn pretty_print(&self, level: usize) {
        if level == 0 {
            println!("{} : {}", self.name, pretty_print_nbytes(self.size));
        }
        let mut sorted_fields = self.fields.clone();
        sorted_fields.sort_by(|(_, a), (_, b)| b.size.cmp(&a.size));
        for (name, sr) in sorted_fields {
            let percentage = (sr.size as f64 / self.size as f64) * 100.0;
            println!(". {:level$}{} : {} ({:.2}%)", "", name, pretty_print_nbytes(sr.size), percentage);
            sr.pretty_print(level + 2);
        }
    }
}

pub trait GenerateSizeReport: serde::Serialize {
    fn get_sizerp(&self) -> SizeReport {
        let mut buf = flexbuffers::FlexbufferSerializer::new();
        serde::Serialize::serialize(self, &mut buf).expect("Failed to serialize in Flexbuffer");
        SizeReport {
            name: std::any::type_name::<Self>().to_string(),
            size: buf.view().len(),
            fields: Vec::new(),
        }
    }
}

impl GenerateSizeReport for egglog_bridge::EGraph {}

impl <T: serde::Serialize> GenerateSizeReport for Option<T> {} 

impl <K: serde::Serialize, V: serde::Serialize> GenerateSizeReport for egglog::util::IndexMap<K, V> {} 

impl GenerateSizeReport for TypeInfo {}

impl GenerateSizeReport for RunReport {}

impl <K: serde::Serialize, V: serde::Serialize> GenerateSizeReport for egglog_numeric_id::DenseIdMap<K, V> {}

impl GenerateSizeReport for CommandMacroRegistry {}

impl GenerateSizeReport for EncodingState {}


impl GenerateSizeReport for EGraph {
    fn get_sizerp(&self) -> SizeReport {
        let mut buf = flexbuffers::FlexbufferSerializer::new();
        serde::Serialize::serialize(self, &mut buf).expect("Failed to serialize in Flexbuffer");
        let mut ret = SizeReport {
            name: std::any::type_name::<Self>().to_string(),
            size: buf.view().len(),
            fields: Vec::new(),
        };
        ret.fields.push(("backend".to_string(), Box::new(self.backend.get_sizerp())));
        ret.fields.push(("pushed_egraph".to_string(), Box::new(self.pushed_egraph.get_sizerp())));
        ret.fields.push(("functions".to_string(), Box::new(self.functions.get_sizerp())));
        ret.fields.push(("rulesets".to_string(), Box::new(self.rulesets.get_sizerp())));
        ret.fields.push(("type_info".to_string(), Box::new(self.type_info.get_sizerp())));
        ret.fields.push(("overall_run_report".to_string(), Box::new(self.overall_run_report.get_sizerp())));
        ret.fields.push(("schedulers".to_string(), Box::new(self.schedulers.get_sizerp())));
        ret.fields.push(("commands".to_string(), Box::new(self.commands.get_sizerp())));
        ret.fields.push(("command_macros".to_string(), Box::new(self.command_macros.get_sizerp())));
        ret.fields.push(("proof_state".to_string(), Box::new(self.proof_state.get_sizerp())));
        ret
    }
}

/*
pub struct EGraph {
    backend: egglog_bridge::EGraph,

    pub parser: Parser,

    names: check_shadowing::Names,
    /// pushed_egraph forms a linked list of pushed egraphs.
    /// Pop reverts the egraph to the last pushed egraph.
    pushed_egraph: Option<Box<Self>>,

    functions: IndexMap<String, Function>,

    rulesets: IndexMap<String, Ruleset>,
    pub fact_directory: Option<PathBuf>,
    pub seminaive: bool,

    type_info: TypeInfo,
    /// The run report unioned over all runs so far.
    overall_run_report: RunReport,

    schedulers: DenseIdMap<SchedulerId, SchedulerRecord>,

    commands: IndexMap<String, Arc<dyn UserDefinedCommand>>,
    strict_mode: bool,
    warned_about_missing_global_prefix: bool,
    /// Registry for command-level macros
    command_macros: CommandMacroRegistry,
    proof_state: EncodingState,
}
    */