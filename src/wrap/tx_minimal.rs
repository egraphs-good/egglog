use super::*;
use egglog::{ast::Command, EGraph, SerializeConfig};
use std::{path::PathBuf, sync::Mutex};

pub struct TxMinimal {
    egraph: Mutex<EGraph>,
}

/// tx with miminal feature (only new function is supported)
impl TxMinimal {
    pub fn new_with_type_defs(commands: Vec<Command>) -> Self {
        Self {
            egraph: Mutex::new({
                let mut e = EGraph::default();
                e.run_program(commands).unwrap();
                e
            }),
        }
    }
    pub fn new() -> Self {
        Self::new_with_type_defs(collect_type_defs())
    }
    pub fn to_dot(&self, file_name: PathBuf) {
        let egraph = self.egraph.lock().unwrap();
        let serialized = egraph.serialize(SerializeConfig::default());
        let dot_path = file_name.with_extension("dot");
        serialized
            .to_dot_file(dot_path.clone())
            .unwrap_or_else(|_| panic!("Failed to write dot file to {dot_path:?}"));
    }
}

unsafe impl Send for TxMinimal {}
unsafe impl Sync for TxMinimal {}
// MARK: Tx
impl Tx for TxMinimal {
    fn send(&self, transmitted: TxCommand) {
        log::debug!("{:?}", transmitted);
        let mut egraph = self.egraph.lock().unwrap();
        match transmitted {
            TxCommand::StringCommand { command } => {
                egraph.parse_and_run_program(None, &command).unwrap();
            }
            TxCommand::NativeCommand { command } => {
                egraph.run_program(vec![command]).unwrap();
            }
        }
    }

    fn on_new(&self, node: &(impl EgglogNode + 'static)) {
        self.send(TxCommand::NativeCommand {
            command: node.to_egglog(),
        });
    }

    fn on_set(&self, _node: &mut (impl EgglogNode + 'static)) {
        panic!("set is unsupported for tx_minimal")
    }

    fn on_func_set<'a, F: EgglogFunc>(
        &self,
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
        output: <F::Output as EgglogFuncOutput>::Ref<'a>,
    ) {
        let input_nodes = input.as_nodes();
        let input_syms = input_nodes.iter().map(|x| x.cur_sym());
        let output = output.as_node().cur_sym();
        self.send(TxCommand::StringCommand {
            command: format!(
                "(set ({} {}) {} )",
                F::FUNC_NAME,
                input_syms.map(|x| x.as_str()).collect::<String>(),
                output
            ),
        });
    }

    // fn on_funcs_get<'a,'b, F: EgglogFunc>(
    //     &self,
    //     max_size:Option<usize>)->
    // Vec<(<F::Input as EgglogFuncInputs>::Ref<'b>,<F::Output as EgglogFuncOutput>::Ref<'b>)> {
    //     todo!()
    // }

    // fn on_func_get<'a,'b, F: EgglogFunc>(
    //     &self,
    //     input: <F::Input as EgglogFuncInputs>::Ref<'a>,
    // ) -> <F::Output as EgglogFuncOutput>::Ref<'b> {
    //     todo!()
    // }
}
