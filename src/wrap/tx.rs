use crate::ast::Command;

use super::*;
use dashmap::DashMap;
use egglog::{util::IndexSet, EGraph, SerializeConfig};
use std::{path::PathBuf, sync::Mutex};

pub struct TxNoVT {
    egraph: Mutex<EGraph>,
    map: DashMap<Sym, WorkAreaNode>,
    latest_map: DashMap<Sym, Sym>,
}

/// Tx without version ctl feature
impl TxNoVT {
    pub fn new_with_type_defs(type_defs: Vec<Command>) -> Self {
        Self {
            egraph: Mutex::new({
                let mut e = EGraph::default();
                log::info!("{:?}", type_defs);
                e.run_program(type_defs).unwrap();
                e
            }),
            map: DashMap::default(),
            latest_map: DashMap::default(),
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
    // collect all ancestors of cur_sym, without cur_sym
    pub fn collect_latest_ancestors(&self, cur_sym: Sym, index_set: &mut IndexSet<Sym>) {
        let node = self.map.get(&cur_sym).unwrap();
        let succss = node.preds.clone();
        drop(node);
        for pred in succss {
            if index_set.contains(&pred) || self.map.get(&pred).unwrap().next.is_some() {
                // do nothing
            } else {
                index_set.insert(pred);
                self.collect_latest_ancestors(pred, index_set)
            }
        }
    }
    /// start nodes is asserted to be zero input degree
    pub fn topo_sort(&self, starts: IndexSet<Sym>, index_set: &IndexSet<Sym>) -> Vec<Sym> {
        let map = &self.map;
        // init in degrees and out degrees
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        ins.resize(index_set.len(), 0);
        outs.resize(index_set.len(), 0);
        for (i, (in_degree, out_degree)) in ins.iter_mut().zip(outs.iter_mut()).enumerate() {
            let sym = index_set[i];
            let node = map.get(&sym).unwrap();
            *in_degree =
                TxNoVT::degree_in_subgraph(node.preds().into_iter().map(|x| *x), index_set);
            *out_degree = TxNoVT::degree_in_subgraph(node.succs().into_iter(), index_set);
        }
        let mut rst = Vec::new();
        let mut wait_for_release = Vec::new();
        // start node should not have any out edges in subgraph
        for start in starts {
            assert_eq!(0, outs[index_set.get_index_of(&start).unwrap()]);
            wait_for_release.push(start);
        }
        while !wait_for_release.is_empty() {
            let popped = wait_for_release.pop().unwrap();
            for target in &map.get(&popped).unwrap().preds {
                let idx = index_set.get_index_of(target).unwrap();
                outs[idx] -= 1;
                if outs[idx] == 0 {
                    wait_for_release.push(*target);
                }
            }
            rst.push(popped);
        }
        rst
    }
    /// calculate the edges in the subgraph
    pub fn degree_in_subgraph(nodes: impl Iterator<Item = Sym>, index_set: &IndexSet<Sym>) -> u32 {
        nodes.fold(0, |acc, item| {
            if index_set.contains(&item) {
                acc + 1
            } else {
                acc
            }
        })
    }

    pub fn map_latest(&self, sym: Sym) -> Sym {
        let mut cur = sym;
        while let Some(key) = self.latest_map.get(&cur) {
            cur = *key
        }
        cur
    }
    fn add_node(&self, node: &(impl EgglogNode + 'static)) {
        self.send(TxCommand::NativeCommand {
            command: node.to_egglog(),
        });
        let mut node = WorkAreaNode::new(node.clone_dyn());
        let sym = node.cur_sym();
        for succ_node in node.succs_mut() {
            *succ_node = self.map_latest(*succ_node);
            self.map
                .get_mut(succ_node)
                .unwrap_or_else(|| panic!("node {} not found", succ_node.as_str()))
                .preds
                .push(sym);
        }
        // println!("{:?}",self.map);
        self.map.insert(node.cur_sym(), node);
    }

    /// update all predecessor recursively in guest and send updated term by egglog repr to host
    /// when you update the node
    /// for non version control mode, update_symnode will update &mut old sym to latest
    fn update_symnode(&self, node: &mut (impl EgglogNode + 'static)) {
        let latest_sym = self.map_latest(node.cur_sym());
        *node.cur_sym_mut() = node.next_sym();
        let mut updated_symnode = WorkAreaNode::new(node.clone_dyn());
        let mut index_set = IndexSet::default();

        // collect all syms that will change
        self.collect_latest_ancestors(latest_sym, &mut index_set);
        let mut latest_node = self.map.get_mut(&latest_sym).unwrap();
        // chain old version and new version
        latest_node.next = Some(updated_symnode.egglog.cur_sym());
        updated_symnode.preds = latest_node.preds.clone();
        drop(latest_node);
        let mut next_syms = vec![];
        // insert copied ancestors
        for &old_sym in index_set.iter() {
            let (_, mut sym_node) = self.map.remove(&old_sym).unwrap();
            let new_sym = sym_node.next_sym();
            self.latest_map.insert(old_sym, new_sym);

            next_syms.push(new_sym);
            self.map.insert(new_sym, sym_node);
        }
        index_set.insert(latest_sym);
        let new_sym = updated_symnode.cur_sym();
        next_syms.push(updated_symnode.cur_sym());
        self.map.insert(updated_symnode.cur_sym(), updated_symnode);
        // update all preds
        for &new_sym in &next_syms {
            let mut sym_node = self.map.get_mut(&new_sym).unwrap();
            for sym in sym_node.preds_mut() {
                if let Some(idx) = index_set.get_index_of(&*sym) {
                    *sym = next_syms[idx];
                }
            }
            for sym in sym_node.succs_mut() {
                if let Some(idx) = index_set.get_index_of(&*sym) {
                    *sym = next_syms[idx];
                }
            }
        }
        let mut s = "".to_owned();
        let topo = self.topo_sort(
            IndexSet::from_iter(Some(new_sym).into_iter()),
            &IndexSet::from_iter(next_syms.into_iter()),
        );
        for new_sym in topo {
            s += self
                .map
                .get(&new_sym)
                .unwrap()
                .egglog
                .to_egglog_string()
                .as_str();
        }
        self.send(TxCommand::StringCommand { command: s });
    }
}

unsafe impl Send for TxNoVT {}
unsafe impl Sync for TxNoVT {}
// MARK: Tx
impl Tx for TxNoVT {
    fn send(&self, received: TxCommand) {
        match received {
            TxCommand::StringCommand { command } => {
                {
                    log::info!("{}", command);
                    let mut egraph = self.egraph.lock().unwrap();
                    egraph.parse_and_run_program(None, command.as_str()).unwrap();
                };
            }
            TxCommand::NativeCommand { command} => {
                let mut egraph = self.egraph.lock().unwrap();
                egraph.run_program(vec![command]).unwrap();
            }
        }
    }

    fn on_new(&self, symnode: &(impl EgglogNode + 'static)) {
        self.add_node(symnode);
    }

    fn on_set(&self, symnode: &mut (impl EgglogNode + 'static)) {
        self.update_symnode(symnode);
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
}
